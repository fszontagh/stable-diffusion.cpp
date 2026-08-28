// sd-aa-test: minimal, standalone diagnostic tool for the AnimateAnyone model
// family registration. It does NOT build a full sd_ctx_t - it only exercises
// ModelLoader::init_from_file() + ModelLoader::get_sd_version() and applies
// the version-promotion rule that new_sd_ctx() applies at init time
// (see stable-diffusion.cpp: reference_net_path non-empty promotes a
// SD1-signature diffusion model to VERSION_ANIMATE_ANYONE).

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <string>
#include <thread>
#include <vector>

#include "model.h"
#include "model_loader.h"
#include "model_manager.h"
#include "stable-diffusion.h"

#include "core/ggml_extend.hpp"
#include "core/ggml_extend_backend.h"
#include "core/tensor.hpp"
#include "core/util.h"
#include "model/adapter/pose_guider.hpp"
#include "model/adapter/pose_guider_ssd.hpp"
#include "model/diffusion/unet.hpp"
#include "model/te/aa_clip_preprocess.hpp"
#include "model/te/clip.hpp"
#include "model/vae/auto_encoder_kl.hpp"

#include "npy.hpp"
#include "json.hpp"
#include "runtime/denoiser.hpp"

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_STATIC
#include "stb_image.h"

extern const char* model_version_to_str[];

// Route the library's LOG_* calls (including get_sd_version()'s fallback-detection
// LOG_WARN) to stderr. Without a registered callback, log_printf() is a silent no-op,
// which would hide detection warnings from this diagnostic tool's own output.
static void aa_test_log_cb(enum sd_log_level_t level, const char* text, void* /*data*/) {
    fputs(text, stderr);
}

// Root directory for downloaded reference weights. Honored by every mode that
// has a default weights path (pose-guider, clip-embeds) so the fixtures/weights
// location isn't hard-coded to one machine.
static std::string aa_weights_root() {
    if (const char* env = std::getenv("SDCPP_AA_WEIGHTS")) {
        return env;
    }
    return "/data/sdcpp-pixel-refs/weights";
}

static void print_usage() {
    fprintf(stderr,
            "usage: sd-aa-test <mode> [args]\n"
            "\n"
            "modes:\n"
            "  version --diffusion-model <path> [--reference-net <path>]\n"
            "      Loads the diffusion model file, detects its SDVersion, applies the\n"
            "      reference_net_path promotion rule, and prints the resulting version\n"
            "      display name. Exits 0 on success, 1 on failure.\n"
            "  pose-guider [--pose-guider <path>] [--fixtures <dir>]\n"
            "      Loads pose_guider.pth (default: $SDCPP_AA_WEIGHTS or\n"
            "      /data/sdcpp-pixel-refs/weights, plus /AnimateAnyone/pose_guider.pth),\n"
            "      runs the Moore PoseGuiderA forward pass on <fixtures>/pose.png (default\n"
            "      fixtures dir: $SDCPP_AA_FIXTURES or /data/sdcpp-pixel-refs/fixtures), and\n"
            "      compares against <fixtures>/pose_guider_a_out.npy. Prints the relative L2\n"
            "      error. Exits 0 on pass (rel L2 <= 1e-3), 1 on failure.\n"
            "  pose-guider-b [--pose-guider <path>] [--fixtures <dir>]\n"
            "      Sprite-Sheet-Diffusion pose guider variant B (Task 12). Loads a\n"
            "      pose_guider checkpoint (default: $SDCPP_AA_WEIGHTS or\n"
            "      /data/sdcpp-pixel-refs/weights, plus /SSD/pose_guider.pth), verifies it\n"
            "      probes as variant B ('conv_layers.0.weight' present), runs PoseGuiderB on\n"
            "      <fixtures>/pose.png + <fixtures>/ref_pose.png (both rescaled to [-1,1] -\n"
            "      variant B normalization, unlike variant A's [0,1]), and compares all 5\n"
            "      pyramid features against <fixtures>/pose_guider_b_out_{0..4}.npy. Exits 0\n"
            "      only if every feature passes (rel L2 <= 1e-3), 1 otherwise.\n"
            "  clip-embeds [--clip-vision <path>] [--fixtures <dir>]\n"
            "      Loads the sd-image-variations CLIPVisionModelWithProjection encoder\n"
            "      (default: $SDCPP_AA_WEIGHTS or /data/sdcpp-pixel-refs/weights, plus\n"
            "      /image_encoder/pytorch_model.bin) and runs two checks against row 1 of\n"
            "      <fixtures>/clip_embeds.npy (row 0 is asserted to be the all-zeros uncond\n"
            "      embedding):\n"
            "        STRICT (the pass/fail gate on model correctness): forwards\n"
            "        <fixtures>/pixel_values.npy, the reference pipeline's own exact\n"
            "        preprocessed input, and requires rel L2 <= 1e-3.\n"
            "        preprocessing tolerance (informational, resampler not Pillow-bit-exact):\n"
            "        CLIP-preprocesses <fixtures>/ref.png through this test's own\n"
            "        image->resize->normalize chain and requires rel L2 <= 2e-3.\n"
            "      Prints both relative L2 errors. Exits 0 only if both pass, 1 otherwise.\n"
            "  scheduler [--fixtures <dir>]\n"
            "      Verifies the AnimateAnyone v2 scheduler math (zero-SNR rescaled\n"
            "      alphas_cumprod, the 25-step trailing timestep grid, and one DDIM\n"
            "      v-prediction step) against <fixtures>/sched_v2.json (default fixtures\n"
            "      dir: $SDCPP_AA_FIXTURES or /data/sdcpp-pixel-refs/fixtures). Exits 0\n"
            "      only if all three checks pass, 1 otherwise.\n"
            "  context-schedule [--fixtures <dir>]\n"
            "      Verifies the Task 10 sliding-window context scheduler\n"
            "      (animate_anyone_context::aa_context_windows(), ported from context.py's\n"
            "      uniform()) against <fixtures>/context_windows.json for num_frames in\n"
            "      {64, 32} at the v2 defaults (24/1/4/closed_loop). Requires an EXACT\n"
            "      match (order and content). Exits 0 only if both match, 1 otherwise.\n"
            "  ref-bank [--vae <path>] [--reference-net <path>] [--fixtures <dir>] [--threads <n>]\n"
            "      (a) VAE-encodes <fixtures>/ref.png with sd-vae-ft-mse (distribution MEAN,\n"
            "      scaled by 0.18215) and compares against <fixtures>/ref_latents.npy at\n"
            "      rel L2 <= 1e-3. (b) Runs the headless ReferenceNet (reference_unet.pth)\n"
            "      one forward at t=0 with the CFG-doubled ref latents and\n"
            "      <fixtures>/clip_embeds.npy as cross-attn context, captures the 16\n"
            "      post-norm1 hidden-state banks, and compares each against\n"
            "      <fixtures>/ref_bank_00..15.npy at rel L2 <= 1e-3. Prints the per-bank\n"
            "      error table and the worst bank. Exits 0 only if all checks pass.\n"
            "  unet-step-f1 [--diffusion-model <path>] [--motion-module <path>] [--fixtures <dir>]\n"
            "               [--threads <n>] [--conv-f32]\n"
            "      Loads denoising_unet.pth + motion_module.pth (defaults: $SDCPP_AA_WEIGHTS or\n"
            "      /data/sdcpp-pixel-refs/weights, plus /AnimateAnyone/<file>). The motion\n"
            "      modules run even at F=1 (temporal attention over one frame is NOT a no-op\n"
            "      and the fixture was generated with them active).\n"
            "      Runs one denoising step at t=999 on <fixtures>/unet_step_f1_in.npy\n"
            "      (CFG batch 2, uncond row 0 / cond row 1) as TWO separate forwards: the\n"
            "      uncond half with plain self-attention and NO reference, the cond half\n"
            "      with the FIXTURE banks (ref_bank_00..15.npy, row 1) injected into every\n"
            "      attn1 as concat-KV context. Both halves get the fixture pose feature\n"
            "      (pose_guider_a_out.npy, added after conv_in) and their clip_embeds.npy\n"
            "      row as cross-attn context. Reassembles [2,4,1,64,64] and compares\n"
            "      against <fixtures>/unet_step_f1.npy at rel L2 <= 1e-3 (per-half and\n"
            "      combined errors printed). Exits 0 on pass.\n"
            "  unet-step-f8 [--diffusion-model <path>] [--motion-module <path>] [--fixtures <dir>]\n"
            "               [--threads <n>] [--conv-f32]\n"
            "      Same as unet-step-f1 but F=8: <fixtures>/unet_step_f8_in.npy /\n"
            "      unet_step_f8.npy (2,4,8,64,64). The fixture banks (still [C,L,2], one\n"
            "      row per CFG half, no frame axis) are broadcast to all 8 frame rows\n"
            "      before the per-frame seq-dim concat, and the fixture pose feature\n"
            "      (still [64,64,320,1]) is broadcast the same way (P-map section 2 read-\n"
            "      mode shapes). Exercises real temporal mixing through the motion\n"
            "      modules and the pos-encoder slice beyond frame 0.\n");
}

static int run_version_mode(int argc, char** argv) {
    std::string diffusion_model_path;
    std::string reference_net_path;

    for (int i = 0; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--diffusion-model") {
            if (i + 1 >= argc) {
                fprintf(stderr, "error: --diffusion-model requires a value\n");
                return 1;
            }
            diffusion_model_path = argv[++i];
        } else if (arg == "--reference-net") {
            if (i + 1 >= argc) {
                fprintf(stderr, "error: --reference-net requires a value\n");
                return 1;
            }
            reference_net_path = argv[++i];
        } else {
            fprintf(stderr, "error: unknown argument '%s'\n", arg.c_str());
            return 1;
        }
    }

    if (diffusion_model_path.empty()) {
        fprintf(stderr, "error: --diffusion-model is required\n");
        return 1;
    }

    ModelLoader model_loader;
    if (!model_loader.init_from_file(diffusion_model_path, "model.diffusion_model.")) {
        fprintf(stderr, "error: failed to load diffusion model from '%s'\n", diffusion_model_path.c_str());
        return 1;
    }

    SDVersion version = model_loader.get_sd_version();
    if (version == VERSION_COUNT) {
        fprintf(stderr, "error: could not detect SD version from '%s'\n", diffusion_model_path.c_str());
        return 1;
    }
    printf("detected version: %s\n", model_version_to_str[version]);

    // Shared promotion rule (src/model.h) also used by new_sd_ctx()'s init() flow: a
    // SD1-signature diffusion model plus a non-empty reference_net_path promotes to
    // VERSION_ANIMATE_ANYONE.
    SDVersion promoted = sd_apply_animate_anyone_promotion(version,
                                                            reference_net_path.empty() ? nullptr : reference_net_path.c_str());
    if (promoted != version) {
        version = promoted;
        printf("reference_net_path set, promoting detected version to %s\n", model_version_to_str[version]);
    }

    printf("%s\n", model_version_to_str[version]);
    return 0;
}

// PoseGuiderRunner was promoted to src/model/adapter/pose_guider.hpp
// (AnimateAnyone::PoseGuiderRunner) in task 7 so stable-diffusion.cpp can
// construct it too; this test now uses the shared runner with an empty prefix
// (the checkpoint keys match PoseGuiderA's block dict verbatim).
using AnimateAnyone::PoseGuiderRunner;

static int run_pose_guider_mode(int argc, char** argv) {
    std::string weights_path = aa_weights_root() + "/AnimateAnyone/pose_guider.pth";
    std::string fixtures_dir;
    if (const char* env = std::getenv("SDCPP_AA_FIXTURES")) {
        fixtures_dir = env;
    } else {
        fixtures_dir = "/data/sdcpp-pixel-refs/fixtures";
    }

    for (int i = 0; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--pose-guider") {
            if (i + 1 >= argc) {
                fprintf(stderr, "error: --pose-guider requires a value\n");
                return 1;
            }
            weights_path = argv[++i];
        } else if (arg == "--fixtures") {
            if (i + 1 >= argc) {
                fprintf(stderr, "error: --fixtures requires a value\n");
                return 1;
            }
            fixtures_dir = argv[++i];
        } else {
            fprintf(stderr, "error: unknown argument '%s'\n", arg.c_str());
            return 1;
        }
    }

    std::string pose_image_path = fixtures_dir + "/pose.png";
    std::string expected_path   = fixtures_dir + "/pose_guider_a_out.npy";

    // --- Load the pose skeleton image, scaled to [0,1], ggml layout [W,H,3,N=1]. ---
    int width = 0, height = 0, channels_in_file = 0;
    unsigned char* pixels = stbi_load(pose_image_path.c_str(), &width, &height, &channels_in_file, 3);
    if (pixels == nullptr) {
        fprintf(stderr, "error: failed to load pose image '%s'\n", pose_image_path.c_str());
        return 1;
    }
    sd_image_t sd_image{static_cast<uint32_t>(width), static_cast<uint32_t>(height), 3, pixels};
    // sd_image_to_tensor(..., scale=true) -> [0,1] fp32, ggml layout [W,H,C,1]. No
    // [-1,1] rescale, matching Moore's cond_image_processor(do_normalize=False).
    sd::Tensor<float> pose_tensor = sd_image_to_tensor(sd_image, -1, -1, true);
    free(pixels);
    printf("pose image: %dx%d\n", width, height);

    // --- Load the checkpoint. Keys are conv_in.*, blocks.{0..5}.*, conv_out.* -
    // matches the block dict keys of PoseGuiderA verbatim, so an empty prefix binds
    // tensors by name directly. ---
    auto model_manager = std::make_shared<ModelManager>();
    model_manager->set_n_threads(1);
    ModelLoader& model_loader = model_manager->loader();
    if (!model_loader.init_from_file(weights_path, "")) {
        fprintf(stderr, "error: failed to load pose guider weights from '%s'\n", weights_path.c_str());
        return 1;
    }

    ggml_backend_t cpu_backend = sd_backend_cpu_init();
    if (cpu_backend == nullptr) {
        fprintf(stderr, "error: failed to init CPU backend\n");
        return 1;
    }

    PoseGuiderRunner runner(cpu_backend, model_loader.get_tensor_storage_map(), "", model_manager);

    std::map<std::string, ggml_tensor*> tensors;
    runner.get_param_tensors(tensors);

    // Deliverable 4: verify the zero-init conv_out weights were actually bound from
    // the checkpoint (not silently skipped/absent).
    bool conv_out_bound = false;
    for (const auto& kv : tensors) {
        if (kv.first.rfind("conv_out.", 0) == 0) {
            conv_out_bound = true;
            break;
        }
    }
    if (!conv_out_bound) {
        fprintf(stderr, "error: no conv_out.* parameter tensors were bound (init() failed to match checkpoint keys)\n");
        return 1;
    }
    printf("conv_out tensors bound: yes\n");

    if (!model_manager->register_param_tensors("pose_guider",
                                               tensors,
                                               ModelManager::ResidencyMode::ParamBackend,
                                               cpu_backend,
                                               cpu_backend) ||
        !model_manager->validate_registered_tensors()) {
        fprintf(stderr, "error: failed to register pose guider tensors with the model manager\n");
        return 1;
    }

    // --- Forward pass. ---
    sd::Tensor<float> output = runner.compute(1, pose_tensor);

    // Release the model manager (params backend buffers) before `runner` is
    // destroyed at scope exit. ModelManager::TensorState keeps a raw ggml_tensor*
    // into the runner's params_ctx; freeing the manager after the runner would
    // touch a dangling pointer. Matches the LTX latent upsampler precedent
    // (stable-diffusion.cpp: `upsampler_manager.reset(); upsampler.reset();`).
    model_manager.reset();

    if (output.empty()) {
        fprintf(stderr, "error: pose guider forward pass failed\n");
        return 1;
    }
    printf("output shape (ggml [W,H,C,N]): [%lld, %lld, %lld, %lld]\n",
           (long long)output.shape()[0],
           (long long)output.shape()[1],
           (long long)output.shape()[2],
           (long long)output.shape()[3]);

    // --- Load the reference output and compare. ---
    // Fixture npy shape is (N=1, C=320, D=1, H=64, W=64) - the PyTorch
    // InflatedConv3d convention (b, c, f, h, w) with a singleton frame axis.
    // Our ggml output is [W=64, H=64, C=320, N=1].
    //
    // Axis mapping / why a flat memcmp-style compare is valid without any
    // reshuffling: numpy is C-order (last axis fastest), so npy flat index is
    //   n*(C*H*W) + c*(H*W) + h*W + w
    // sd::Tensor/ggml is ne[0]-fastest, so our flat index for shape [W,H,C,N] is
    //   n*(C*H*W) + c*(H*W) + h*W + w
    // These are the *same* formula (D=1 drops out of the npy indexing), so the
    // two arrays are byte-identical in memory layout given equal shapes
    // (64,64,320,1) vs (1,320,1,64,64) - a direct elementwise/flat comparison is
    // correct.
    aa_test::NpyArray expected;
    std::string npy_error;
    if (!aa_test::load_npy_f32(expected_path, expected, npy_error)) {
        fprintf(stderr, "error: %s\n", npy_error.c_str());
        return 1;
    }
    printf("expected npy shape (N,C,D,H,W): [");
    for (size_t i = 0; i < expected.shape.size(); ++i) {
        printf("%s%lld", i == 0 ? "" : ", ", (long long)expected.shape[i]);
    }
    printf("]\n");

    if (expected.shape.size() != 5 ||
        expected.shape[0] != output.shape()[3] ||
        expected.shape[1] != output.shape()[2] ||
        expected.shape[2] != 1 ||
        expected.shape[3] != output.shape()[1] ||
        expected.shape[4] != output.shape()[0]) {
        fprintf(stderr,
                "error: shape mismatch: expected npy (N,C,D,H,W)=(%lld,%lld,%lld,%lld,%lld) "
                "vs ggml output (W,H,C,N)=(%lld,%lld,%lld,%lld)\n",
                expected.shape.size() > 0 ? (long long)expected.shape[0] : -1,
                expected.shape.size() > 1 ? (long long)expected.shape[1] : -1,
                expected.shape.size() > 2 ? (long long)expected.shape[2] : -1,
                expected.shape.size() > 3 ? (long long)expected.shape[3] : -1,
                expected.shape.size() > 4 ? (long long)expected.shape[4] : -1,
                (long long)output.shape()[0],
                (long long)output.shape()[1],
                (long long)output.shape()[2],
                (long long)output.shape()[3]);
        return 1;
    }
    if (expected.numel() != output.numel()) {
        fprintf(stderr, "error: element count mismatch: expected=%lld got=%lld\n",
                (long long)expected.numel(), (long long)output.numel());
        return 1;
    }

    double num = 0.0, den = 0.0;
    const float* got  = output.data();
    const float* want = expected.data.data();
    for (int64_t i = 0; i < output.numel(); ++i) {
        double diff = static_cast<double>(got[i]) - static_cast<double>(want[i]);
        num += diff * diff;
        den += static_cast<double>(want[i]) * static_cast<double>(want[i]);
    }
    double rel_l2 = den > 0.0 ? std::sqrt(num / den) : std::sqrt(num);
    printf("relative L2 error: %g\n", rel_l2);

    const double tolerance = 1e-3;
    if (rel_l2 > tolerance) {
        fprintf(stderr, "FAIL: relative L2 error %g exceeds tolerance %g\n", rel_l2, tolerance);
        return 1;
    }

    printf("PASS: pose guider matches reference within tolerance %g\n", tolerance);
    return 0;
}

// Task 12: Sprite-Sheet-Diffusion pose guider variant B. Loads a pose_guider
// checkpoint whose tensor names match PoseGuiderB ("conv_layers.0.weight"
// etc, not Moore's "blocks.0.weight"), runs it on pose.png + ref_pose.png
// (BOTH rescaled to [-1,1] - variant B normalization, P-map porting note 5;
// differs from variant A's [0,1] above), and compares all 5 pyramid
// features against <fixtures>/pose_guider_b_out_{0..4}.npy.
static int run_pose_guider_b_mode(int argc, char** argv) {
    std::string weights_path = aa_weights_root() + "/SSD/pose_guider.pth";
    std::string fixtures_dir;
    if (const char* env = std::getenv("SDCPP_AA_FIXTURES")) {
        fixtures_dir = env;
    } else {
        fixtures_dir = "/data/sdcpp-pixel-refs/fixtures";
    }

    for (int i = 0; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--pose-guider") {
            if (i + 1 >= argc) {
                fprintf(stderr, "error: --pose-guider requires a value\n");
                return 1;
            }
            weights_path = argv[++i];
        } else if (arg == "--fixtures") {
            if (i + 1 >= argc) {
                fprintf(stderr, "error: --fixtures requires a value\n");
                return 1;
            }
            fixtures_dir = argv[++i];
        } else {
            fprintf(stderr, "error: unknown argument '%s'\n", arg.c_str());
            return 1;
        }
    }

    std::string pose_image_path     = fixtures_dir + "/pose.png";
    std::string ref_pose_image_path = fixtures_dir + "/ref_pose.png";

    auto load_neg1_1 = [](const std::string& path, sd::Tensor<float>* out) -> bool {
        int width = 0, height = 0, channels_in_file = 0;
        unsigned char* pixels = stbi_load(path.c_str(), &width, &height, &channels_in_file, 3);
        if (pixels == nullptr) {
            fprintf(stderr, "error: failed to load image '%s'\n", path.c_str());
            return false;
        }
        sd_image_t sd_image{static_cast<uint32_t>(width), static_cast<uint32_t>(height), 3, pixels};
        // scale=true -> [0,1]; variant B additionally rescales to [-1,1]
        // (do_normalize=True, unlike variant A's [0,1]).
        sd::Tensor<float> tensor = sd_image_to_tensor(sd_image, -1, -1, true);
        free(pixels);
        for (int64_t i = 0; i < tensor.numel(); ++i) {
            tensor.data()[i] = tensor.data()[i] * 2.f - 1.f;
        }
        *out = std::move(tensor);
        printf("%s: %dx%d\n", path.c_str(), width, height);
        return true;
    };

    sd::Tensor<float> pose_tensor, ref_pose_tensor;
    if (!load_neg1_1(pose_image_path, &pose_tensor) || !load_neg1_1(ref_pose_image_path, &ref_pose_tensor)) {
        return 1;
    }

    // --- Load the checkpoint. Variant selection probe (deliverable 3): fail
    // loudly if this doesn't actually look like a PoseGuiderB checkpoint,
    // rather than silently binding nothing and returning garbage. ---
    auto model_manager = std::make_shared<ModelManager>();
    model_manager->set_n_threads(1);
    ModelLoader& model_loader = model_manager->loader();
    if (!model_loader.init_from_file(weights_path, "")) {
        fprintf(stderr, "error: failed to load pose guider weights from '%s'\n", weights_path.c_str());
        return 1;
    }
    const auto& tensor_storage_map = model_loader.get_tensor_storage_map();
    if (tensor_storage_map.find("conv_layers.0.weight") == tensor_storage_map.end()) {
        fprintf(stderr,
                "error: '%s' does not look like a PoseGuiderB checkpoint "
                "(no 'conv_layers.0.weight' tensor - variant A checkpoints use 'blocks.0.weight')\n",
                weights_path.c_str());
        return 1;
    }

    ggml_backend_t cpu_backend = sd_backend_cpu_init();
    if (cpu_backend == nullptr) {
        fprintf(stderr, "error: failed to init CPU backend\n");
        return 1;
    }

    AnimateAnyone::PoseGuiderBRunner runner(cpu_backend, tensor_storage_map, "", model_manager);

    std::map<std::string, ggml_tensor*> tensors;
    runner.get_param_tensors(tensors);
    if (!model_manager->register_param_tensors("pose_guider_b",
                                               tensors,
                                               ModelManager::ResidencyMode::ParamBackend,
                                               cpu_backend,
                                               cpu_backend) ||
        !model_manager->validate_registered_tensors()) {
        fprintf(stderr, "error: failed to register pose guider (variant B) tensors with the model manager\n");
        return 1;
    }

    std::vector<sd::Tensor<float>> outputs = runner.compute(1, pose_tensor, ref_pose_tensor);
    model_manager.reset();

    if (outputs.size() != 5) {
        fprintf(stderr, "error: pose guider (variant B) forward pass failed (got %zu/5 features)\n", outputs.size());
        return 1;
    }

    // Same npy<->ggml axis-mapping reasoning as run_pose_guider_mode: fixture
    // shape (N,C,D,H,W) numpy C-order == ggml [W,H,C,N] flat layout (D=1
    // drops out of both index formulas).
    //
    // Tolerance: 2e-3, not 1e-3 (Task 5's atol+rtol ruling: loosen with a
    // numeric justification when a structural, non-bug source of drift is
    // identified). Measured error grows monotonically and near-linearly with
    // the number of cross_attn{k} blocks applied so far - fea[0] (0
    // Transformer2D blocks): 0 rel L2 (exact); fea[1] (1 block): 3.1e-4;
    // fea[2] (2 blocks): 7.5e-4; fea[3] (3 blocks): 9.7e-4; fea[4] (4
    // blocks): 1.15e-3. Root cause: this fork's shared FeedForward/GEGLU
    // block (model/common/block.hpp, reused here rather than duplicated)
    // calls ggml_gelu(), which computes the tanh-approximated GELU
    // (0.5x(1+tanh(sqrt(2/pi)(x+0.044715x^3)))) - a long-standing ggml CPU
    // kernel behavior, not something introduced by this port - while
    // diffusers' own GEGLU.gelu() calls exact erf-based F.gelu() by default.
    // PoseGuiderB chains 4 Transformer2D blocks (variant A's UNet fixtures
    // only ever exercise 1-2 per checked tensor), so the same known,
    // pre-existing per-block approximation compounds further here than
    // elsewhere in the port. 2e-3 keeps a >1.7x margin over the worst
    // observed error while still catching a real regression (a wiring bug
    // would not produce this clean per-block-count-proportional signature).
    bool all_pass         = true;
    const double tolerance = 2e-3;
    std::string npy_error;
    for (int i = 0; i < 5; ++i) {
        char path[1024];
        snprintf(path, sizeof(path), "%s/pose_guider_b_out_%d.npy", fixtures_dir.c_str(), i);
        aa_test::NpyArray expected;
        if (!aa_test::load_npy_f32(path, expected, npy_error)) {
            fprintf(stderr, "error: %s\n", npy_error.c_str());
            return 1;
        }
        const sd::Tensor<float>& output = outputs[i];
        if (expected.shape.size() != 5 ||
            expected.shape[0] != output.shape()[3] ||
            expected.shape[1] != output.shape()[2] ||
            expected.shape[2] != 1 ||
            expected.shape[3] != output.shape()[1] ||
            expected.shape[4] != output.shape()[0] ||
            expected.numel() != output.numel()) {
            fprintf(stderr, "error: fea[%d] shape mismatch: expected npy shape (N,C,D,H,W) vs ggml output (W,H,C,N)=[%lld,%lld,%lld,%lld]\n",
                    i, (long long)output.shape()[0], (long long)output.shape()[1],
                    (long long)output.shape()[2], (long long)output.shape()[3]);
            return 1;
        }

        double num = 0.0, den = 0.0;
        const float* got  = output.data();
        const float* want = expected.data.data();
        for (int64_t k = 0; k < output.numel(); ++k) {
            double diff = static_cast<double>(got[k]) - static_cast<double>(want[k]);
            num += diff * diff;
            den += static_cast<double>(want[k]) * static_cast<double>(want[k]);
        }
        double rel_l2 = den > 0.0 ? std::sqrt(num / den) : std::sqrt(num);
        printf("fea[%d] shape (ggml [W,H,C,N]): [%lld, %lld, %lld, %lld]  relative L2 error: %g\n",
               i, (long long)output.shape()[0], (long long)output.shape()[1],
               (long long)output.shape()[2], (long long)output.shape()[3], rel_l2);
        if (rel_l2 > tolerance) {
            fprintf(stderr, "FAIL: fea[%d] relative L2 error %g exceeds tolerance %g\n", i, rel_l2, tolerance);
            all_pass = false;
        }
    }

    if (!all_pass) {
        return 1;
    }
    printf("PASS: pose guider (variant B) matches reference on all 5 pyramid features within tolerance %g\n", tolerance);
    return 0;
}

// Standalone GGMLRunner wrapper around CLIPVisionModelProjection, modeled on
// PoseGuiderRunner above. FrozenCLIPVisionEmbedder (conditioning/conditioner.hpp)
// is hard-coded to OPEN_CLIP_VIT_H_14, so it can't load the sd-image-variations
// encoder (OpenAI ViT-L/14 architecture, quick_gelu, 768-d projection) that
// AnimateAnyone uses; this runner drives CLIPVisionModelProjection directly.
struct ClipVisionRunner : public GGMLRunner {
    CLIPVisionModelProjection model;

    ClipVisionRunner(ggml_backend_t backend,
                     const String2TensorStorage& tensor_storage_map,
                     std::shared_ptr<RunnerWeightManager> weight_manager = nullptr)
        : GGMLRunner(backend, weight_manager),
          model(build_model(tensor_storage_map)) {
        model.init(params_ctx, tensor_storage_map, "");
    }

    // Fused-QKV (self_attn.in_proj_*) vs. separate q/k/v_proj is auto-detected from
    // the checkpoint, matching FrozenCLIPVisionEmbedder's precedent
    // (conditioning/conditioner.hpp:569-579). sd-image-variations' pytorch_model.bin
    // uses separate q/k/v_proj (proj_in=false); the probe keeps this robust to other
    // checkpoint layouts. force_quick_gelu=true: see CLIPMLP/CLIPVisionModel in
    // model/te/clip.hpp - the sd-image-variations config specifies "hidden_act":
    // "quick_gelu" explicitly, which the fork's d_model-based heuristic would
    // otherwise miss (hidden_size 1024 collides with OpenCLIP ViT-H's gelu branch).
    static CLIPVisionModelProjection build_model(const String2TensorStorage& tensor_storage_map) {
        bool proj_in = false;
        for (const auto& [name, tensor_storage] : tensor_storage_map) {
            if (contains(name, "self_attn.in_proj")) {
                proj_in = true;
                break;
            }
        }
        return CLIPVisionModelProjection(OPENAI_CLIP_VIT_L_14, /*transpose_proj_w=*/false, proj_in, /*force_quick_gelu=*/true);
    }

    std::string get_desc() override {
        return "clip_vision_image_variations";
    }

    void get_param_tensors(std::map<std::string, ggml_tensor*>& tensors) {
        model.get_param_tensors(tensors);
    }

    ggml_cgraph* build_graph(const sd::Tensor<float>& pixel_values_tensor) {
        ggml_cgraph* gf           = ggml_new_graph(compute_ctx);
        ggml_tensor* pixel_values = make_input(pixel_values_tensor);
        auto runner_ctx           = get_context();
        // return_pooled=true, clip_skip=-1 (full 24 layers): projected image_embeds,
        // matching CLIPVisionModelWithProjection(...).image_embeds in the reference
        // pipeline (P-map section 8).
        ggml_tensor* image_embeds = model.forward(&runner_ctx, pixel_values, true, -1);
        ggml_build_forward_expand(gf, image_embeds);
        return gf;
    }

    sd::Tensor<float> compute(int n_threads, const sd::Tensor<float>& pixel_values_tensor) {
        auto get_graph = [&]() -> ggml_cgraph* { return build_graph(pixel_values_tensor); };
        return take_or_empty(GGMLRunner::compute<float>(get_graph, n_threads, true, true, true));
    }
};

// The PIL-exact bicubic (a=-0.5, per-pass uint8 rounding) CLIP preprocessing
// chain was promoted to src/model/te/aa_clip_preprocess.hpp in task 9 so the
// generation-time AnimateAnyoneVisionConditioner shares it; this test keeps
// validating it against the reference pipeline's own pixel_values below.
using AnimateAnyone::aa_clip_preprocess;

static int run_clip_embeds_mode(int argc, char** argv) {
    std::string clip_vision_path = aa_weights_root() + "/image_encoder/pytorch_model.bin";
    std::string fixtures_dir;
    if (const char* env = std::getenv("SDCPP_AA_FIXTURES")) {
        fixtures_dir = env;
    } else {
        fixtures_dir = "/data/sdcpp-pixel-refs/fixtures";
    }

    for (int i = 0; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--clip-vision") {
            if (i + 1 >= argc) {
                fprintf(stderr, "error: --clip-vision requires a value\n");
                return 1;
            }
            clip_vision_path = argv[++i];
        } else if (arg == "--fixtures") {
            if (i + 1 >= argc) {
                fprintf(stderr, "error: --fixtures requires a value\n");
                return 1;
            }
            fixtures_dir = argv[++i];
        } else {
            fprintf(stderr, "error: unknown argument '%s'\n", arg.c_str());
            return 1;
        }
    }

    std::string ref_image_path    = fixtures_dir + "/ref.png";
    std::string pixel_values_path = fixtures_dir + "/pixel_values.npy";
    std::string expected_path     = fixtures_dir + "/clip_embeds.npy";

    const int clip_image_size = 224;

    // --- STRICT input: the exact [1,3,224,224] pixel_values fp32 tensor the reference
    // pipeline fed the encoder (dumped by tools/aa/dump_fixtures.py). Fixture is numpy
    // C-order (N,C,H,W); ggml layout is ne=[W,H,C,N] (ne[0]-fastest). Both give the
    // same flat index n*(C*H*W)+c*(H*W)+h*W+w for equal shapes, so a straight flat
    // copy is a valid direct reinterpretation (same reasoning as the pose_guider fixture
    // comparison above) - no transpose needed.
    aa_test::NpyArray pixel_values_npy;
    std::string pixel_values_npy_error;
    if (!aa_test::load_npy_f32(pixel_values_path, pixel_values_npy, pixel_values_npy_error)) {
        fprintf(stderr, "error: %s\n", pixel_values_npy_error.c_str());
        return 1;
    }
    if (pixel_values_npy.shape.size() != 4 || pixel_values_npy.shape[0] != 1 ||
        pixel_values_npy.shape[1] != 3 || pixel_values_npy.shape[2] != clip_image_size ||
        pixel_values_npy.shape[3] != clip_image_size) {
        fprintf(stderr, "error: unexpected pixel_values.npy shape (expected (1,3,%d,%d))\n",
                clip_image_size, clip_image_size);
        return 1;
    }
    sd::Tensor<float> strict_pixel_values({clip_image_size, clip_image_size, 3, 1});
    std::copy(pixel_values_npy.data.begin(), pixel_values_npy.data.end(), strict_pixel_values.data());

    // --- PREPROCESS CHAIN input: this test's own image->resize->normalize path,
    // ggml layout [W,H,3,N=1], values in [0,255] (NOT pre-divided by 255:
    // aa_clip_preprocess needs raw byte values so its resize can round/clip to uint8
    // between passes, matching PIL - see aa_pil_bicubic_resize). ---
    int width = 0, height = 0, channels_in_file = 0;
    unsigned char* pixels = stbi_load(ref_image_path.c_str(), &width, &height, &channels_in_file, 3);
    if (pixels == nullptr) {
        fprintf(stderr, "error: failed to load ref image '%s'\n", ref_image_path.c_str());
        return 1;
    }
    sd_image_t sd_image{static_cast<uint32_t>(width), static_cast<uint32_t>(height), 3, pixels};
    sd::Tensor<float> ref_tensor = sd_image_to_tensor(sd_image, -1, -1, false);
    free(pixels);
    printf("ref image: %dx%d\n", width, height);

    sd::Tensor<float> chain_pixel_values = aa_clip_preprocess(ref_tensor, clip_image_size);

    // --- Load the checkpoint. Diffusers CLIPVisionModelWithProjection state_dict
    // keys are "vision_model.*" / "visual_projection.weight" verbatim, so an empty
    // prefix binds tensors by name directly (same pattern as pose_guider.pth above).
    auto model_manager = std::make_shared<ModelManager>();
    model_manager->set_n_threads(1);
    ModelLoader& model_loader = model_manager->loader();
    if (!model_loader.init_from_file(clip_vision_path, "")) {
        fprintf(stderr, "error: failed to load CLIP vision weights from '%s'\n", clip_vision_path.c_str());
        return 1;
    }

    // The reference sd-image-variations pytorch_model.bin (like every HF
    // transformers CLIPVisionModel checkpoint) carries transformers' well-known
    // "pre_layrnorm" typo for the vision tower's pre-encoder LayerNorm. The fork's
    // CLIPVisionModel block spells it correctly ("pre_layernorm"), and already has a
    // rename for this exact typo in name_conversion.cpp:68-69 - but that mapping only
    // fires on keys already carrying the "transformer." prefix added by the
    // clip_vision. remap pipeline (stable-diffusion.cpp / name_conversion.cpp:1473),
    // which a raw HF pytorch_model.bin loaded directly (as here, bypassing that
    // pipeline) never gets. Rename the two affected keys in place so
    // CLIPVisionModel's "pre_layernorm" block binds against the checkpoint.
    auto& tensor_storage_map = model_loader.get_tensor_storage_map();
    for (const std::string field : {"weight", "bias"}) {
        std::string old_key = "vision_model.pre_layrnorm." + field;
        std::string new_key = "vision_model.pre_layernorm." + field;
        auto it              = tensor_storage_map.find(old_key);
        if (it != tensor_storage_map.end()) {
            TensorStorage storage = it->second;
            storage.name          = new_key;
            tensor_storage_map.erase(old_key);
            tensor_storage_map.insert({new_key, storage});
        }
    }

    ggml_backend_t cpu_backend = sd_backend_cpu_init();
    if (cpu_backend == nullptr) {
        fprintf(stderr, "error: failed to init CPU backend\n");
        return 1;
    }

    ClipVisionRunner runner(cpu_backend, model_loader.get_tensor_storage_map(), model_manager);

    std::map<std::string, ggml_tensor*> tensors;
    runner.get_param_tensors(tensors);

    bool proj_bound = false;
    for (const auto& kv : tensors) {
        if (kv.first.rfind("visual_projection.", 0) == 0) {
            proj_bound = true;
            break;
        }
    }
    if (!proj_bound) {
        fprintf(stderr, "error: no visual_projection.* parameter tensors were bound (init() failed to match checkpoint keys)\n");
        return 1;
    }
    printf("visual_projection tensors bound: yes\n");

    if (!model_manager->register_param_tensors("clip_vision",
                                               tensors,
                                               ModelManager::ResidencyMode::ParamBackend,
                                               cpu_backend,
                                               cpu_backend) ||
        !model_manager->validate_registered_tensors()) {
        fprintf(stderr, "error: failed to register CLIP vision tensors with the model manager\n");
        return 1;
    }

    // --- Forward pass, both inputs: projected image_embeds, shape ggml
    // [projection_dim, N=1]. ---
    sd::Tensor<float> strict_output = runner.compute(1, strict_pixel_values);
    sd::Tensor<float> chain_output  = runner.compute(1, chain_pixel_values);

    // Release the model manager (params backend buffers) before `runner` is
    // destroyed at scope exit - same destructor-order precedent as
    // run_pose_guider_mode above.
    model_manager.reset();

    if (strict_output.empty() || chain_output.empty()) {
        fprintf(stderr, "error: CLIP vision forward pass failed\n");
        return 1;
    }
    printf("output numel: %lld\n", (long long)strict_output.numel());

    // --- Load the reference embeddings and compare. ---
    // Fixture shape is (2, 1, 768): row 0 = zeros uncond, row 1 = cond embed.
    aa_test::NpyArray expected;
    std::string npy_error;
    if (!aa_test::load_npy_f32(expected_path, expected, npy_error)) {
        fprintf(stderr, "error: %s\n", npy_error.c_str());
        return 1;
    }
    printf("expected npy shape: [");
    for (size_t i = 0; i < expected.shape.size(); ++i) {
        printf("%s%lld", i == 0 ? "" : ", ", (long long)expected.shape[i]);
    }
    printf("]\n");

    if (expected.shape.size() != 3 || expected.shape[0] != 2 || expected.shape[2] != strict_output.numel()) {
        fprintf(stderr,
                "error: unexpected fixture shape (expected (2, N, %lld)) for output numel %lld\n",
                (long long)strict_output.numel(), (long long)strict_output.numel());
        return 1;
    }

    int64_t row_stride = expected.shape[1] * expected.shape[2];  // elements per batch row

    // Deliverable: document the uncond convention - row 0 must be the literal zero
    // vector (P-map section 8: "uncond = torch.zeros_like(...)"), not a text-model
    // empty-prompt embedding.
    const float* uncond_row = expected.data.data();
    for (int64_t i = 0; i < row_stride; ++i) {
        if (uncond_row[i] != 0.0f) {
            fprintf(stderr, "error: fixture row 0 (uncond) is not all-zero at index %lld (value %g)\n",
                    (long long)i, uncond_row[i]);
            return 1;
        }
    }
    printf("uncond row (row 0) verified all-zero: yes\n");

    const float* cond_row = expected.data.data() + row_stride;

    auto rel_l2_against_cond_row = [&](const sd::Tensor<float>& output) -> double {
        const float* got = output.data();
        double num = 0.0, den = 0.0;
        for (int64_t i = 0; i < output.numel(); ++i) {
            double diff = static_cast<double>(got[i]) - static_cast<double>(cond_row[i]);
            num += diff * diff;
            den += static_cast<double>(cond_row[i]) * static_cast<double>(cond_row[i]);
        }
        return den > 0.0 ? std::sqrt(num / den) : std::sqrt(num);
    };

    // --- STRICT: the model given the reference pipeline's own exact pixel_values.
    // This is the pass/fail gate on model correctness (weights, activation,
    // projection), independent of how closely aa_clip_preprocess's own resize
    // matches Pillow bit-for-bit. ---
    double strict_rel_l2 = rel_l2_against_cond_row(strict_output);
    printf("STRICT relative L2 error (model, exact reference pixel_values): %g\n", strict_rel_l2);
    const double strict_tolerance = 1e-3;
    bool strict_pass              = strict_rel_l2 <= strict_tolerance;
    if (!strict_pass) {
        fprintf(stderr, "FAIL: STRICT relative L2 error %g exceeds tolerance %g\n", strict_rel_l2, strict_tolerance);
    }

    // --- PREPROCESS CHAIN: this test's own image->resize->normalize->forward path.
    // Looser tolerance: aa_clip_preprocess's resampler (bicubic a=-0.5 with uint8
    // rounding between passes, aa_pil_bicubic_resize above) is a close but not
    // bit-exact reimplementation of Pillow's internal 32-bit fixed-point resample -
    // see the task 4 report for the investigation. Not the model-correctness gate;
    // documents how close the C++ preprocessing path itself gets. ---
    double chain_rel_l2 = rel_l2_against_cond_row(chain_output);
    printf("preprocessing tolerance (resampler not Pillow-bit-exact, see report): "
           "relative L2 error %g\n",
           chain_rel_l2);
    const double chain_tolerance = 2e-3;
    bool chain_pass               = chain_rel_l2 <= chain_tolerance;
    if (!chain_pass) {
        fprintf(stderr,
                "FAIL: preprocessing-chain relative L2 error %g exceeds tolerance %g\n",
                chain_rel_l2, chain_tolerance);
    }

    if (!strict_pass || !chain_pass) {
        return 1;
    }

    printf("PASS: CLIP vision embeds match reference (STRICT <= %g, preprocessing chain <= %g)\n",
           strict_tolerance, chain_tolerance);
    return 0;
}

// Recursively flattens a nested JSON array of numbers (arbitrary rank, e.g. the fixture's
// [1,4,64,64] x_t/v/prev_sample) into a flat float vector in C order. Non-array leaves are
// appended as scalars.
static void flatten_json_floats(const nlohmann::json& node, std::vector<float>& out) {
    if (node.is_array()) {
        for (const auto& child : node) {
            flatten_json_floats(child, out);
        }
    } else {
        out.push_back(node.get<float>());
    }
}

// mode `scheduler`: verifies the AnimateAnyone v2 (zero-SNR v-prediction trailing DDIM)
// scheduler math implemented in src/runtime/denoiser.hpp's animate_anyone_scheduler
// namespace against <fixtures>/sched_v2.json. Three independent checks, all required to pass:
//   (a) the rescaled alphas_cumprod, all 1000 entries, vs the fixture's full array
//   (b) the 25-entry trailing timestep grid, exact integers
//   (c) one DDIM v-prediction step on the fixture's x_t/v/t, vs its prev_sample
static int run_scheduler_mode(int argc, char** argv) {
    std::string fixtures_dir;
    if (const char* env = std::getenv("SDCPP_AA_FIXTURES")) {
        fixtures_dir = env;
    } else {
        fixtures_dir = "/data/sdcpp-pixel-refs/fixtures";
    }

    for (int i = 0; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--fixtures") {
            if (i + 1 >= argc) {
                fprintf(stderr, "error: --fixtures requires a value\n");
                return 1;
            }
            fixtures_dir = argv[++i];
        } else {
            fprintf(stderr, "error: unknown argument '%s'\n", arg.c_str());
            return 1;
        }
    }

    std::string sched_path = fixtures_dir + "/sched_v2.json";
    std::ifstream sched_file(sched_path);
    if (!sched_file.is_open()) {
        fprintf(stderr, "error: failed to open %s\n", sched_path.c_str());
        return 1;
    }
    nlohmann::json fixture;
    try {
        sched_file >> fixture;
    } catch (const std::exception& e) {
        fprintf(stderr, "error: failed to parse %s: %s\n", sched_path.c_str(), e.what());
        return 1;
    }

    bool all_pass = true;

    // --- (a) zero-SNR rescaled alphas_cumprod, all 1000 entries ---
    std::vector<float> alphas_cumprod(TIMESTEPS);
    animate_anyone_scheduler::calculate_alphas_cumprod(alphas_cumprod.data());

    const auto& expected_alphas = fixture.at("alphas_cumprod");
    if (expected_alphas.size() != (size_t)TIMESTEPS) {
        fprintf(stderr, "error: fixture alphas_cumprod has %zu entries, expected %d\n",
                expected_alphas.size(), TIMESTEPS);
        return 1;
    }
    // Comparison is |got - expected| <= atol + rtol*|expected|, rtol=1e-6 as specified, plus a
    // small atol=1e-10 floor - the standard numpy.isclose/torch.allclose form for comparing
    // floating point arrays that span many orders of magnitude, which is exactly what a
    // zero-terminal-SNR alphas_cumprod array does (~1.0 down to ~1e-8 over the last ~40 of the
    // 1000 entries). Investigated and required: near the tail, alphas_cumprod is the product of
    // ~950-999 sequential per-element multiplications, each of which differs from the fixture's
    // torch-float32 computation by a single float32 ULP (confirmed by hand-replicating torch's
    // exact linspace + cumprod op sequence in float32 and in float64 - see task-5-report.md).
    // Cumulative product amplifies per-term relative noise additively in log-space, so ~1000
    // terms of ~1e-7 relative ULP noise compounds to ~1e-4-1e-5 relative noise in the *reference
    // fixture itself* once the running product has decayed into the 1e-4..1e-8 range - no
    // independent implementation (this one included, computed in double precision throughout)
    // can track torch's specific fp32 rounding sequence at that scale without bit-for-bit
    // reproducing its internal reduction kernel. atol=1e-10 is 3+ orders of magnitude below the
    // smallest failing entry's expected value (~6.3e-8) and the global max ABSOLUTE diff of any
    // non-degenerate (order ~1) entry (~4.2e-7, itself well inside rtol there) - so it cannot
    // mask a real algorithmic bug (wrong beta_schedule, missing rescale, off-by-one, etc. all
    // produce order-1 relative errors across the whole array, not a 1e-10-scale floor at the
    // extreme tail only).
    const double rtol = 1e-6;
    const double atol = 1e-10;
    double max_rel_err  = 0.0;
    int max_rel_err_idx = -1;
    bool alphas_pass = true;
    for (int i = 0; i < TIMESTEPS - 1; i++) {
        double expected  = expected_alphas[i].get<double>();
        double got       = (double)alphas_cumprod[i];
        double abs_diff  = std::abs(got - expected);
        double rel       = abs_diff / std::max(std::abs(expected), 1e-30);
        if (rel > max_rel_err) {
            max_rel_err     = rel;
            max_rel_err_idx = i;
        }
        if (abs_diff > atol + rtol * std::abs(expected)) {
            alphas_pass = false;
        }
    }
    printf("(a) alphas_cumprod[0..998]: max relative error %g at index %d "
           "(tolerance |diff| <= %g + %g*|expected|)\n",
           max_rel_err, max_rel_err_idx, atol, rtol);

    // Terminal entry (zero-SNR invariant): compared absolutely against 0.0, not relatively -
    // a relative-error formula is undefined/meaningless when the expected value is exactly
    // zero. The whole point of rescale_betas_zero_snr is that this entry IS exactly zero.
    double expected_terminal = expected_alphas[TIMESTEPS - 1].get<double>();
    double got_terminal      = (double)alphas_cumprod[TIMESTEPS - 1];
    bool terminal_pass       = (expected_terminal == 0.0) && (got_terminal == 0.0);
    printf("(a) alphas_cumprod[%d] (terminal, zero-SNR): expected %g, got %g (must both be == 0.0)\n",
           TIMESTEPS - 1, expected_terminal, got_terminal);
    alphas_pass = alphas_pass && terminal_pass;

    if (!alphas_pass) {
        fprintf(stderr, "FAIL: (a) alphas_cumprod mismatch\n");
    }
    all_pass = all_pass && alphas_pass;

    // --- (b) 25-entry trailing timestep grid, exact integers ---
    std::vector<int> timesteps = animate_anyone_scheduler::trailing_timesteps(TIMESTEPS, 25);
    const auto& expected_timesteps = fixture.at("timesteps");
    bool timesteps_pass = (int)expected_timesteps.size() == (int)timesteps.size();
    if (timesteps_pass) {
        for (size_t i = 0; i < timesteps.size(); i++) {
            if (expected_timesteps[i].get<int>() != timesteps[i]) {
                timesteps_pass = false;
                break;
            }
        }
    }
    printf("(b) trailing timestep grid (%zu steps): %s\n", timesteps.size(), timesteps_pass ? "exact match" : "MISMATCH");
    if (!timesteps_pass) {
        fprintf(stderr, "FAIL: (b) trailing timestep grid mismatch\n");
        fprintf(stderr, "  got:      ");
        for (int t : timesteps) fprintf(stderr, "%d ", t);
        fprintf(stderr, "\n  expected: ");
        for (const auto& t : expected_timesteps) fprintf(stderr, "%d ", t.get<int>());
        fprintf(stderr, "\n");
    }
    all_pass = all_pass && timesteps_pass;

    // --- (c) one DDIM v-prediction step vs the fixture's step_example ---
    const auto& step_example = fixture.at("step_example");
    int t = step_example.at("t").get<int>();

    std::vector<float> x_t_flat, v_flat, expected_prev_flat;
    flatten_json_floats(step_example.at("x_t"), x_t_flat);
    flatten_json_floats(step_example.at("v"), v_flat);
    flatten_json_floats(step_example.at("prev_sample"), expected_prev_flat);
    if (x_t_flat.size() != v_flat.size() || x_t_flat.size() != expected_prev_flat.size()) {
        fprintf(stderr, "error: (c) step_example x_t/v/prev_sample element count mismatch\n");
        return 1;
    }

    // diffusers DDIMScheduler.step(): prev_timestep = t - num_train_timesteps // num_inference_steps.
    int prev_t             = t - (TIMESTEPS / 25);
    float alpha_prod_t      = alphas_cumprod[t];
    float alpha_prod_t_prev = (prev_t >= 0) ? alphas_cumprod[prev_t] : 1.0f;  // set_alpha_to_one=True

    sd::Tensor<float> x_t_tensor({(int64_t)x_t_flat.size()}, x_t_flat);
    sd::Tensor<float> v_tensor({(int64_t)v_flat.size()}, v_flat);
    sd::Tensor<float> prev_sample_tensor =
        animate_anyone_scheduler::ddim_v_pred_step(x_t_tensor, v_tensor, alpha_prod_t, alpha_prod_t_prev);

    double num = 0.0, den = 0.0;
    for (int64_t i = 0; i < prev_sample_tensor.numel(); i++) {
        double diff = (double)prev_sample_tensor[i] - (double)expected_prev_flat[i];
        num += diff * diff;
        den += (double)expected_prev_flat[i] * (double)expected_prev_flat[i];
    }
    double step_rel_l2 = den > 0.0 ? std::sqrt(num / den) : std::sqrt(num);
    bool step_pass      = step_rel_l2 <= 1e-5;
    printf("(c) DDIM v-pred step at t=%d (prev_t=%d): relative L2 error %g (tolerance 1e-5)\n",
           t, prev_t, step_rel_l2);
    if (!step_pass) {
        fprintf(stderr, "FAIL: (c) DDIM v-pred step relative L2 error %g exceeds tolerance 1e-5\n", step_rel_l2);
    }
    all_pass = all_pass && step_pass;

    if (!all_pass) {
        return 1;
    }
    printf("PASS: AnimateAnyone v2 scheduler (zero-SNR alphas_cumprod, trailing grid, DDIM v-pred step) "
           "all match reference\n");
    return 0;
}

// mode `context-schedule`: verifies the Task 10 sliding-window context scheduler
// (src/runtime/denoiser.hpp animate_anyone_context::aa_context_windows(), ported
// from moore-animate-anyone src/pipelines/context.py's uniform()) against
// <fixtures>/context_windows.json, for num_frames in {64, 32} at the v2 defaults
// (context_frames=24, context_stride=1, context_overlap=4, closed_loop=true,
// step=0). Requires an EXACT match: same number of windows, same order, same
// per-window index lists. CPU-only, no model weights.
static int run_context_schedule_mode(int argc, char** argv) {
    std::string fixtures_dir;
    if (const char* env = std::getenv("SDCPP_AA_FIXTURES")) {
        fixtures_dir = env;
    } else {
        fixtures_dir = "/data/sdcpp-pixel-refs/fixtures";
    }

    for (int i = 0; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--fixtures") {
            if (i + 1 >= argc) {
                fprintf(stderr, "error: --fixtures requires a value\n");
                return 1;
            }
            fixtures_dir = argv[++i];
        } else {
            fprintf(stderr, "error: unknown argument '%s'\n", arg.c_str());
            return 1;
        }
    }

    std::string path = fixtures_dir + "/context_windows.json";
    std::ifstream file(path);
    if (!file.is_open()) {
        fprintf(stderr, "error: failed to open %s\n", path.c_str());
        return 1;
    }
    nlohmann::json fixture;
    try {
        file >> fixture;
    } catch (const std::exception& e) {
        fprintf(stderr, "error: failed to parse %s: %s\n", path.c_str(), e.what());
        return 1;
    }

    const auto& params           = fixture.at("params");
    const int context_frames     = params.at("context_frames").get<int>();
    const int context_stride     = params.at("context_stride").get<int>();
    const int context_overlap    = params.at("context_overlap").get<int>();
    const bool closed_loop       = params.at("closed_loop").get<bool>();
    const auto& expected_windows = fixture.at("windows");

    bool all_pass = true;
    for (int num_frames : {64, 32}) {
        std::vector<std::vector<int>> got = animate_anyone_context::aa_context_windows(
            num_frames, context_frames, context_stride, context_overlap, closed_loop);

        const auto& expected = expected_windows.at(std::to_string(num_frames));
        bool pass             = got.size() == expected.size();
        if (pass) {
            for (size_t w = 0; w < got.size(); w++) {
                const auto& expected_w = expected[w];
                if (got[w].size() != expected_w.size()) {
                    pass = false;
                    break;
                }
                for (size_t k = 0; k < got[w].size(); k++) {
                    if (got[w][k] != expected_w[k].get<int>()) {
                        pass = false;
                        break;
                    }
                }
                if (!pass) {
                    break;
                }
            }
        }

        printf("num_frames=%d: got %zu window(s), expected %zu window(s): %s\n",
               num_frames, got.size(), expected.size(), pass ? "exact match" : "MISMATCH");
        if (!pass) {
            fprintf(stderr, "FAIL: num_frames=%d context window mismatch\n", num_frames);
            fprintf(stderr, "  got:      ");
            for (const auto& w : got) {
                fprintf(stderr, "[");
                for (int idx : w) fprintf(stderr, "%d,", idx);
                fprintf(stderr, "] ");
            }
            fprintf(stderr, "\n  expected: ");
            for (const auto& w : expected) {
                fprintf(stderr, "[");
                for (const auto& idx : w) fprintf(stderr, "%d,", idx.get<int>());
                fprintf(stderr, "] ");
            }
            fprintf(stderr, "\n");
        }
        all_pass = all_pass && pass;
    }

    if (!all_pass) {
        return 1;
    }
    printf("PASS: AnimateAnyone Task 10 context window scheduler matches reference (F=64, F=32)\n");
    return 0;
}

// mode `ref-bank`: ReferenceNet forward with hidden-state bank capture (task 6).
//
// (a) VAE mean-path encode: the reference pipeline encodes the ref image with
//     `vae.encode(...).latent_dist.mean * 0.18215` (P-map section 8) - the
//     distribution MEAN, not a sample. The fork's AutoEncoderKL encoder returns
//     the raw moments [.., 2*latent_channels, ..]; the mean is the first chunk
//     along the channel dim (gaussian_latent_sample takes chunk 0 as mean too).
// (b) Bank capture: one headless-UNet forward at t=0 with the CFG-doubled ref
//     latents (both halves identical) and the CFG-paired CLIP embeds
//     (row 0 zeros uncond, row 1 cond) as cross-attn context. The 16
//     post-norm1 hidden states are read back in descending-norm1-width stable
//     order and compared per bank against ref_bank_00..15.npy.
static int run_ref_bank_mode(int argc, char** argv) {
    std::string vae_path           = aa_weights_root() + "/sd-vae-ft-mse/diffusion_pytorch_model.safetensors";
    std::string reference_net_path = aa_weights_root() + "/AnimateAnyone/reference_unet.pth";
    std::string fixtures_dir;
    if (const char* env = std::getenv("SDCPP_AA_FIXTURES")) {
        fixtures_dir = env;
    } else {
        fixtures_dir = "/data/sdcpp-pixel-refs/fixtures";
    }
    int n_threads = static_cast<int>(std::thread::hardware_concurrency());
    if (n_threads <= 0) {
        n_threads = 4;
    }

    for (int i = 0; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--vae") {
            if (i + 1 >= argc) {
                fprintf(stderr, "error: --vae requires a value\n");
                return 1;
            }
            vae_path = argv[++i];
        } else if (arg == "--reference-net") {
            if (i + 1 >= argc) {
                fprintf(stderr, "error: --reference-net requires a value\n");
                return 1;
            }
            reference_net_path = argv[++i];
        } else if (arg == "--fixtures") {
            if (i + 1 >= argc) {
                fprintf(stderr, "error: --fixtures requires a value\n");
                return 1;
            }
            fixtures_dir = argv[++i];
        } else if (arg == "--threads") {
            if (i + 1 >= argc) {
                fprintf(stderr, "error: --threads requires a value\n");
                return 1;
            }
            n_threads = std::atoi(argv[++i]);
        } else {
            fprintf(stderr, "error: unknown argument '%s'\n", arg.c_str());
            return 1;
        }
    }

    auto rel_l2 = [](const float* got, const float* want, int64_t n) -> double {
        double num = 0.0, den = 0.0;
        for (int64_t i = 0; i < n; ++i) {
            double diff = static_cast<double>(got[i]) - static_cast<double>(want[i]);
            num += diff * diff;
            den += static_cast<double>(want[i]) * static_cast<double>(want[i]);
        }
        return den > 0.0 ? std::sqrt(num / den) : std::sqrt(num);
    };

    ggml_backend_t cpu_backend = sd_backend_cpu_init();
    if (cpu_backend == nullptr) {
        fprintf(stderr, "error: failed to init CPU backend\n");
        return 1;
    }

    // --- Reference latents fixture (needed by both halves: comparison target for
    // (a), UNet input for (b)). Shape (1,4,64,64) numpy C-order == ggml
    // [64,64,4,1] flat layout (same flat-index formula, see pose-guider mode). ---
    aa_test::NpyArray ref_latents_npy;
    std::string npy_error;
    if (!aa_test::load_npy_f32(fixtures_dir + "/ref_latents.npy", ref_latents_npy, npy_error)) {
        fprintf(stderr, "error: %s\n", npy_error.c_str());
        return 1;
    }
    if (ref_latents_npy.shape.size() != 4 || ref_latents_npy.shape[0] != 1 || ref_latents_npy.shape[1] != 4 ||
        ref_latents_npy.shape[2] != 64 || ref_latents_npy.shape[3] != 64) {
        fprintf(stderr, "error: unexpected ref_latents.npy shape (expected (1,4,64,64))\n");
        return 1;
    }

    // ============================ (a) VAE mean-path encode ============================
    bool vae_pass = false;
    {
        int width = 0, height = 0, channels_in_file = 0;
        unsigned char* pixels = stbi_load((fixtures_dir + "/ref.png").c_str(), &width, &height, &channels_in_file, 3);
        if (pixels == nullptr) {
            fprintf(stderr, "error: failed to load ref image '%s/ref.png'\n", fixtures_dir.c_str());
            return 1;
        }
        sd_image_t sd_image{static_cast<uint32_t>(width), static_cast<uint32_t>(height), 3, pixels};
        // [0,1] here; VAE::encode's scale_input then rescales to [-1,1], matching the
        // reference VaeImageProcessor(do_normalize=True) preprocessing.
        sd::Tensor<float> ref_tensor = sd_image_to_tensor(sd_image, -1, -1, true);
        free(pixels);
        printf("ref image: %dx%d\n", width, height);

        auto vae_manager = std::make_shared<ModelManager>();
        vae_manager->set_n_threads(1);
        ModelLoader& vae_loader = vae_manager->loader();
        // "vae." prefix + name conversion maps the diffusers VAE state dict to the
        // fork's LDM-style "first_stage_model.*" names (name_conversion.cpp prefix map).
        if (!vae_loader.init_from_file_and_convert_name(vae_path, "vae.", VERSION_SD1)) {
            fprintf(stderr, "error: failed to load VAE from '%s'\n", vae_path.c_str());
            return 1;
        }

        AutoEncoderKL vae(cpu_backend, vae_loader.get_tensor_storage_map(), "first_stage_model",
                          /*decode_only=*/false, /*use_video_decoder=*/false, VERSION_SD1, vae_manager);

        std::map<std::string, ggml_tensor*> vae_tensors;
        vae.get_param_tensors(vae_tensors);
        if (!vae_manager->register_param_tensors("vae", vae_tensors,
                                                 ModelManager::ResidencyMode::ParamBackend,
                                                 cpu_backend, cpu_backend) ||
            !vae_manager->validate_registered_tensors()) {
            fprintf(stderr, "error: failed to register VAE tensors with the model manager\n");
            return 1;
        }

        sd_tiling_params_t no_tiling{};
        sd::Tensor<float> moments = vae.encode(n_threads, ref_tensor, no_tiling);
        // Release the manager before `vae` leaves scope - destructor-order precedent,
        // see run_pose_guider_mode.
        vae_manager.reset();
        if (moments.empty()) {
            fprintf(stderr, "error: VAE encode failed\n");
            return 1;
        }

        // moments [64,64,8,1] -> distribution mean = channels 0..3, * 0.18215
        // (latent_dist.mean path; NOT gaussian_latent_sample - no noise).
        sd::Tensor<float> mean    = sd::ops::chunk(moments, 2, 2)[0];
        sd::Tensor<float> latents = mean * 0.18215f;

        if (latents.numel() != ref_latents_npy.numel()) {
            fprintf(stderr, "error: VAE latents element count mismatch: got %lld expected %lld\n",
                    (long long)latents.numel(), (long long)ref_latents_npy.numel());
            return 1;
        }
        double vae_rel_l2 = rel_l2(latents.data(), ref_latents_npy.data.data(), latents.numel());
        printf("(a) VAE mean-path latents relative L2 error: %g (tolerance 1e-3)\n", vae_rel_l2);
        vae_pass = vae_rel_l2 <= 1e-3;
        if (!vae_pass) {
            fprintf(stderr, "FAIL: (a) VAE latents relative L2 error %g exceeds tolerance 1e-3\n", vae_rel_l2);
        }
    }

    // ============================ (b) ReferenceNet banks ============================
    // CFG-doubled inputs: latents duplicated (both halves identical), CLIP embeds
    // row 0 = zeros uncond / row 1 = cond (P-map section 2 write mode).
    aa_test::NpyArray clip_embeds_npy;
    if (!aa_test::load_npy_f32(fixtures_dir + "/clip_embeds.npy", clip_embeds_npy, npy_error)) {
        fprintf(stderr, "error: %s\n", npy_error.c_str());
        return 1;
    }
    if (clip_embeds_npy.shape.size() != 3 || clip_embeds_npy.shape[0] != 2 ||
        clip_embeds_npy.shape[1] != 1 || clip_embeds_npy.shape[2] != 768) {
        fprintf(stderr, "error: unexpected clip_embeds.npy shape (expected (2,1,768))\n");
        return 1;
    }

    const int64_t latent_numel = ref_latents_npy.numel();
    sd::Tensor<float> x({64, 64, 4, 2});
    std::copy(ref_latents_npy.data.begin(), ref_latents_npy.data.end(), x.data());
    std::copy(ref_latents_npy.data.begin(), ref_latents_npy.data.end(), x.data() + latent_numel);

    // (2,1,768) numpy C-order == ggml [768,1,2] flat layout.
    sd::Tensor<float> context({768, 1, 2});
    std::copy(clip_embeds_npy.data.begin(), clip_embeds_npy.data.end(), context.data());

    auto unet_manager = std::make_shared<ModelManager>();
    unet_manager->set_n_threads(1);
    ModelLoader& unet_loader = unet_manager->loader();
    // "model.reference_net." is a registered diffusion-model prefix (task 2), so the
    // diffusers-format reference_unet.pth keys are converted to the fork's original
    // SD1 UNet names under that prefix.
    if (!unet_loader.init_from_file_and_convert_name(reference_net_path, "model.reference_net.", VERSION_SD1)) {
        fprintf(stderr, "error: failed to load reference net from '%s'\n", reference_net_path.c_str());
        return 1;
    }

    UNetModelRunner runner(cpu_backend, unet_loader.get_tensor_storage_map(), "model.reference_net",
                           VERSION_SD1, unet_manager, /*reference_headless=*/true);
    if (!runner.config.reference_headless) {
        fprintf(stderr, "error: reference_headless flag was not applied to the UNet config\n");
        return 1;
    }

    std::map<std::string, ggml_tensor*> unet_tensors;
    runner.get_param_tensors(unet_tensors, "model.reference_net");
    // Headless invariant: no out.0/out.2 params may exist (checkpoint omits them;
    // strict binding would otherwise report missing tensors).
    for (const auto& kv : unet_tensors) {
        if (kv.first.rfind("model.reference_net.out.", 0) == 0) {
            fprintf(stderr, "error: headless ReferenceNet still allocated output-head param '%s'\n", kv.first.c_str());
            return 1;
        }
    }
    if (!unet_manager->register_param_tensors("reference_net", unet_tensors,
                                              ModelManager::ResidencyMode::ParamBackend,
                                              cpu_backend, cpu_backend) ||
        !unet_manager->validate_registered_tensors()) {
        fprintf(stderr, "error: failed to register reference net tensors with the model manager\n");
        return 1;
    }

    printf("(b) running ReferenceNet forward at t=0 (batch 2, %d threads)...\n", n_threads);
    std::vector<sd::Tensor<float>> banks = runner.compute_reference_banks(n_threads, x, context);
    unet_manager.reset();
    if (banks.size() != 16) {
        fprintf(stderr, "error: compute_reference_banks returned %zu banks (expected 16)\n", banks.size());
        return 1;
    }

    // Expected per-bank (width, tokens) in bank order: descending norm1 width,
    // ties broken by the reference torch_dfs order down_blocks -> up_blocks ->
    // mid_block (diffusers registers the empty up_blocks ModuleList before
    // creating mid_block). Verified against the fixture manifest: the single
    // 64-token 1280-wide bank (mid, 8x8) is bank 05, after the up-block 1280s.
    static const int64_t expected_width[16]  = {1280, 1280, 1280, 1280, 1280, 1280,
                                                640, 640, 640, 640, 640,
                                                320, 320, 320, 320, 320};
    static const int64_t expected_tokens[16] = {256, 256, 256, 256, 256, 64,
                                                1024, 1024, 1024, 1024, 1024,
                                                4096, 4096, 4096, 4096, 4096};

    bool banks_pass       = true;
    double worst_rel_l2   = 0.0;
    int worst_bank        = -1;
    const double bank_tol = 1e-3;
    for (int i = 0; i < 16; ++i) {
        char bank_name[32];
        snprintf(bank_name, sizeof(bank_name), "ref_bank_%02d.npy", i);
        aa_test::NpyArray expected;
        if (!aa_test::load_npy_f32(fixtures_dir + "/" + bank_name, expected, npy_error)) {
            fprintf(stderr, "error: %s\n", npy_error.c_str());
            return 1;
        }
        // Fixture (2, L, C) numpy C-order == ggml [C, L, 2] flat layout.
        if (expected.shape.size() != 3 || expected.shape[0] != 2 ||
            expected.shape[1] != expected_tokens[i] || expected.shape[2] != expected_width[i]) {
            fprintf(stderr, "error: bank %02d fixture shape mismatch vs expected (2,%lld,%lld)\n",
                    i, (long long)expected_tokens[i], (long long)expected_width[i]);
            return 1;
        }
        const auto& got = banks[i];
        if (got.dim() != 3 || got.shape()[0] != expected_width[i] ||
            got.shape()[1] != expected_tokens[i] || got.shape()[2] != 2) {
            fprintf(stderr,
                    "error: bank %02d shape mismatch: got ggml [%lld,%lld,%lld], expected [%lld,%lld,2] "
                    "(wrong bank-order mapping?)\n",
                    i,
                    (long long)got.shape()[0], (long long)(got.dim() > 1 ? got.shape()[1] : -1),
                    (long long)(got.dim() > 2 ? got.shape()[2] : -1),
                    (long long)expected_width[i], (long long)expected_tokens[i]);
            return 1;
        }
        double err = rel_l2(got.data(), expected.data.data(), got.numel());
        bool pass  = err <= bank_tol;
        printf("(b) bank %02d [C=%4lld, L=%4lld]: rel L2 %-12g %s\n",
               i, (long long)expected_width[i], (long long)expected_tokens[i], err, pass ? "ok" : "FAIL");
        if (err > worst_rel_l2) {
            worst_rel_l2 = err;
            worst_bank   = i;
        }
        banks_pass = banks_pass && pass;
    }
    printf("(b) worst bank: %02d, rel L2 %g (tolerance %g)\n", worst_bank, worst_rel_l2, bank_tol);
    if (!banks_pass) {
        fprintf(stderr, "FAIL: (b) at least one reference bank exceeds tolerance %g\n", bank_tol);
    }

    if (!vae_pass || !banks_pass) {
        return 1;
    }
    printf("PASS: VAE mean-path latents and all 16 reference banks match within tolerance\n");
    return 0;
}

// Picks a compute backend for unet-step-f1/f8. CPU remains the DEFAULT and
// is what this mode's tolerance is fixture-verified against (task 7/8
// reports); GPU is an explicit opt-in via SDCPP_AA_TEST_GPU=1, since on this
// machine flash attention is required to fit the F=8 COND graph's largest
// spatial block in 12 GB of VRAM (a naive, non-flash QK^T score matrix there
// is ~8.6 GB by itself - see task 8 report), and flash attention's mandatory
// K/V f16 cast measured combined rel L2 ~1.6e-2 at F=8 (16x over the 1e-3
// tolerance, insensitive to --conv-f32) - a real GPU precision/memory
// tradeoff, not a semantics bug, so it must never become the silent default
// on a CUDA-enabled build.
static ggml_backend_t aa_test_pick_backend() {
    const char* want_gpu = std::getenv("SDCPP_AA_TEST_GPU");
    if (want_gpu != nullptr && std::string(want_gpu) != "0") {
        ggml_backend_load_all();
        ggml_backend_t gpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU, nullptr);
        if (gpu != nullptr) {
            return gpu;
        }
        fprintf(stderr, "warning: SDCPP_AA_TEST_GPU requested but no GPU backend is available; falling back to CPU\n");
    }
    return sd_backend_cpu_init();
}

// Reorders one CFG half of a (c, f, h, w) PyTorch-flat fixture block into
// this fork's ggml-flat [w, h, c, f] layout (frames on ne[3]) - or back
// again, since the swap is its own inverse. PyTorch nests f INSIDE c;
// this fork's ggml tensors nest c INSIDE f (frames the outermost/slowest
// axis, R-map section 2). At F=1 this is the identity permutation (task 7's
// flat copy happened to be correct only because the frame axis is absorbed);
// at F>1 the two orders are a genuine transpose of the c/f nesting.
static void reorder_pytorch_cf_ggml_fc(const float* src, float* dst, int64_t C, int64_t F, int64_t H, int64_t W) {
    const int64_t plane = H * W;
    for (int64_t f = 0; f < F; ++f) {
        for (int64_t c = 0; c < C; ++c) {
            const float* src_block = src + (c * F + f) * plane;
            float* dst_block       = dst + (f * C + c) * plane;
            std::copy(src_block, src_block + plane, dst_block);
        }
    }
}

// modes `unet-step-f1` / `unet-step-f8`: reference-bank injection in the
// denoising UNet, single frame (task 7) and temporal (F=8, task 8).
//
// Semantics (P-map section 2 read mode + CFG handling): the runner performs
// SEPARATE forwards for the uncond and cond CFG halves so each graph is
// static. The COND forward injects the fixture banks' row 1 into every attn1
// as concat-KV context (Q from the current states, K/V from concat(x, bank) on
// the sequence dim, re-projected through each block's own to_k/to_v) - at
// F>1 the SAME bank content is broadcast to every frame row before the concat
// (P-map section 2 read-mode shapes: bank_fea repeated across video_length,
// then rearranged "b t l c -> (b t) l c"). The UNCOND forward gets no banks at
// all - plain self-attention (bank row 0 is deliberately never used). Both
// forwards add the fixture pose feature to the conv_in output (broadcast
// across frames per the fixture generator's `pose_fea.repeat(1,1,F,1,1)`) and
// use their clip_embeds.npy row as attn2 context (repeated per frame
// generically inside UNetModel::forward). The fixture banks are used directly
// (NOT recomputed via the ReferenceNet) so this mode isolates the injection +
// step math from task 6's capture path.
static int run_unet_step_mode(int argc, char** argv, int video_length, const char* tag) {
    std::string unet_path          = aa_weights_root() + "/AnimateAnyone/denoising_unet.pth";
    std::string motion_module_path = aa_weights_root() + "/AnimateAnyone/motion_module.pth";
    std::string fixtures_dir;
    if (const char* env = std::getenv("SDCPP_AA_FIXTURES")) {
        fixtures_dir = env;
    } else {
        fixtures_dir = "/data/sdcpp-pixel-refs/fixtures";
    }
    int n_threads = static_cast<int>(std::thread::hardware_concurrency());
    if (n_threads <= 0) {
        n_threads = 4;
    }
    // Conv kernel precision. Measured (task-7-report.md): the default F16
    // conv kernels give combined rel L2 4.35e-4 and F32 gives 2.27e-4 - both
    // inside the 1e-3 tolerance, so unlike the ReferenceNet (task 6) the
    // denoising UNet does NOT need set_force_f32. Default F16; --conv-f32
    // keeps the tighter measurement reproducible.
    bool force_conv_f32 = false;

    for (int i = 0; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--diffusion-model") {
            if (i + 1 >= argc) {
                fprintf(stderr, "error: --diffusion-model requires a value\n");
                return 1;
            }
            unet_path = argv[++i];
        } else if (arg == "--motion-module") {
            if (i + 1 >= argc) {
                fprintf(stderr, "error: --motion-module requires a value\n");
                return 1;
            }
            motion_module_path = argv[++i];
        } else if (arg == "--fixtures") {
            if (i + 1 >= argc) {
                fprintf(stderr, "error: --fixtures requires a value\n");
                return 1;
            }
            fixtures_dir = argv[++i];
        } else if (arg == "--threads") {
            if (i + 1 >= argc) {
                fprintf(stderr, "error: --threads requires a value\n");
                return 1;
            }
            n_threads = std::atoi(argv[++i]);
        } else if (arg == "--conv-f16") {
            force_conv_f32 = false;
        } else if (arg == "--conv-f32") {
            force_conv_f32 = true;
        } else {
            fprintf(stderr, "error: unknown argument '%s'\n", arg.c_str());
            return 1;
        }
    }

    auto rel_l2 = [](const float* got, const float* want, int64_t n) -> double {
        double num = 0.0, den = 0.0;
        for (int64_t i = 0; i < n; ++i) {
            double diff = static_cast<double>(got[i]) - static_cast<double>(want[i]);
            num += diff * diff;
            den += static_cast<double>(want[i]) * static_cast<double>(want[i]);
        }
        return den > 0.0 ? std::sqrt(num / den) : std::sqrt(num);
    };

    std::string npy_error;

    // --- Step input/output fixtures: (2,4,F,64,64), CFG batch 2, uncond first,
    // t=999. PyTorch nests (c, f, h, w) - f INSIDE c. This fork's ggml layout
    // is [w, h, c, f] (frames on ne[3], R-map section 2) - c INSIDE f. At F=1
    // the frame axis is trivially absorbed so a flat copy happens to line up
    // (task 7); at F>1 the two layouts are a genuine transpose of the c/f
    // nesting and must be reordered per (c,f) H*W block, not flat-copied. ---
    aa_test::NpyArray step_in_npy, step_out_npy;
    if (!aa_test::load_npy_f32(fixtures_dir + "/unet_step_" + tag + "_in.npy", step_in_npy, npy_error) ||
        !aa_test::load_npy_f32(fixtures_dir + "/unet_step_" + tag + ".npy", step_out_npy, npy_error)) {
        fprintf(stderr, "error: %s\n", npy_error.c_str());
        return 1;
    }
    for (const aa_test::NpyArray* arr : {&step_in_npy, &step_out_npy}) {
        if (arr->shape.size() != 5 || arr->shape[0] != 2 || arr->shape[1] != 4 ||
            arr->shape[2] != video_length || arr->shape[3] != 64 || arr->shape[4] != 64) {
            fprintf(stderr, "error: unexpected unet_step_%s fixture shape (expected (2,4,%d,64,64))\n",
                    tag, video_length);
            return 1;
        }
    }
    const int64_t half_numel = 4 * (int64_t)video_length * 64 * 64;

    // --- Fixture banks in bank order, ggml [C, L, 2] (numpy (2, L, C) C-order is
    // flat-layout-identical - same reasoning as ref-bank mode). ---
    static const int64_t bank_width[16]  = {1280, 1280, 1280, 1280, 1280, 1280,
                                            640, 640, 640, 640, 640,
                                            320, 320, 320, 320, 320};
    static const int64_t bank_tokens[16] = {256, 256, 256, 256, 256, 64,
                                            1024, 1024, 1024, 1024, 1024,
                                            4096, 4096, 4096, 4096, 4096};
    std::vector<sd::Tensor<float>> banks;
    banks.reserve(16);
    for (int i = 0; i < 16; ++i) {
        char bank_name[32];
        snprintf(bank_name, sizeof(bank_name), "ref_bank_%02d.npy", i);
        aa_test::NpyArray bank_npy;
        if (!aa_test::load_npy_f32(fixtures_dir + "/" + bank_name, bank_npy, npy_error)) {
            fprintf(stderr, "error: %s\n", npy_error.c_str());
            return 1;
        }
        if (bank_npy.shape.size() != 3 || bank_npy.shape[0] != 2 ||
            bank_npy.shape[1] != bank_tokens[i] || bank_npy.shape[2] != bank_width[i]) {
            fprintf(stderr, "error: bank %02d fixture shape mismatch vs expected (2,%lld,%lld)\n",
                    i, (long long)bank_tokens[i], (long long)bank_width[i]);
            return 1;
        }
        sd::Tensor<float> bank({bank_width[i], bank_tokens[i], 2});
        std::copy(bank_npy.data.begin(), bank_npy.data.end(), bank.data());
        banks.push_back(std::move(bank));
    }

    // --- CLIP embeds (2,1,768): row 0 zeros uncond, row 1 cond. ---
    aa_test::NpyArray clip_embeds_npy;
    if (!aa_test::load_npy_f32(fixtures_dir + "/clip_embeds.npy", clip_embeds_npy, npy_error)) {
        fprintf(stderr, "error: %s\n", npy_error.c_str());
        return 1;
    }
    if (clip_embeds_npy.shape.size() != 3 || clip_embeds_npy.shape[0] != 2 ||
        clip_embeds_npy.shape[1] != 1 || clip_embeds_npy.shape[2] != 768) {
        fprintf(stderr, "error: unexpected clip_embeds.npy shape (expected (2,1,768))\n");
        return 1;
    }

    // --- Fixture pose feature (1,320,1,64,64) -> ggml [64,64,320,1] (flat-
    // identical, see pose-guider mode). Same feature feeds both CFG halves
    // (the reference CFG-duplicates it, pipeline_pose2img.py:289-291). ---
    aa_test::NpyArray pose_npy;
    if (!aa_test::load_npy_f32(fixtures_dir + "/pose_guider_a_out.npy", pose_npy, npy_error)) {
        fprintf(stderr, "error: %s\n", npy_error.c_str());
        return 1;
    }
    if (pose_npy.shape.size() != 5 || pose_npy.shape[0] != 1 || pose_npy.shape[1] != 320 ||
        pose_npy.shape[2] != 1 || pose_npy.shape[3] != 64 || pose_npy.shape[4] != 64) {
        fprintf(stderr, "error: unexpected pose_guider_a_out.npy shape (expected (1,320,1,64,64))\n");
        return 1;
    }
    sd::Tensor<float> pose_feature({64, 64, 320, 1});
    std::copy(pose_npy.data.begin(), pose_npy.data.end(), pose_feature.data());

    // --- Load the denoising UNet + motion module. denoising_unet.pth carries
    // ONLY the spatial weights (686 keys, verified: zero motion_modules.*
    // keys); the temporal weights come from motion_module.pth, exactly as the
    // reference builds the model (from_pretrained_2d merges motion_module.pth,
    // then denoising_unet.pth is loaded strict=False on top). The motion keys
    // load under the fork's AnimateDiff prefix
    // "model.diffusion_model.motion_module." (no name conversion - the
    // checkpoint already uses down_blocks.i.motion_modules.j... names), which
    // UNetConfig::detect_from_weights probes to enable the motion blocks. ---
    auto unet_manager = std::make_shared<ModelManager>();
    unet_manager->set_n_threads(1);
    ModelLoader& unet_loader = unet_manager->loader();
    if (!unet_loader.init_from_file_and_convert_name(unet_path, "model.diffusion_model.", VERSION_SD1)) {
        fprintf(stderr, "error: failed to load denoising unet from '%s'\n", unet_path.c_str());
        return 1;
    }
    if (!unet_loader.init_from_file(motion_module_path, "model.diffusion_model.motion_module.")) {
        fprintf(stderr, "error: failed to load motion module from '%s'\n", motion_module_path.c_str());
        return 1;
    }

    ggml_backend_t compute_backend = aa_test_pick_backend();
    if (compute_backend == nullptr) {
        fprintf(stderr, "error: failed to init a compute backend\n");
        return 1;
    }
    printf("compute backend: %s%s\n",
          ggml_backend_name(compute_backend),
          sd_backend_is_cpu(compute_backend) ? " (CPU)" : " (GPU)");

    printf("conv precision: %s\n", force_conv_f32 ? "f32" : "f16");
    UNetModelRunner runner(compute_backend, unet_loader.get_tensor_storage_map(), "model.diffusion_model",
                           VERSION_ANIMATE_ANYONE, unet_manager,
                           /*reference_headless=*/false, force_conv_f32);
    if (!sd_backend_is_cpu(compute_backend) && video_length > 1) {
        // At F=8 the COND graph's largest spatial block (input_blocks.1,
        // 320ch/64x64=4096 tokens, bank-doubled K=8192, 8 frames) materializes
        // a naive (non-flash) QK^T score matrix of ~4096*8192*8heads*8frames -
        // ~8.6 GB by itself, measured directly (see task 8 report), which
        // exceeds a 12 GB card regardless of graph-cut segmentation (a single
        // block's peak is a floor segmentation cannot cut below - it only
        // cuts BETWEEN named blocks). Flash attention (existing runner knob,
        // block.hpp already threads ctx->flash_attn_enabled into every
        // ggml_ext_attention_ext call) avoids materializing that matrix at
        // all, which is the actual fix; the graph-cut budget below stays on
        // as a secondary safety net for the remaining (much smaller) peaks.
        // Gated to F>1 only: F=1's compute buffer is ~552 MB (measured), well
        // under any 12 GB card without either knob, and flash attention's
        // mandatory K/V f16 cast is pure precision loss with no OOM benefit
        // there - confirmed empirically (F=1 canary regressed from 7.1e-4 to
        // 1.37e-2 combined rel L2 when flash attention was left on for it).
        runner.set_flash_attention_enabled(true);
        size_t budget_mb = 2048;
        if (const char* env = std::getenv("SDCPP_AA_VRAM_BUDGET_MB")) {
            budget_mb = static_cast<size_t>(std::atoll(env));
        }
        runner.set_max_graph_vram_bytes(budget_mb * 1024ull * 1024ull);
    }

    std::map<std::string, ggml_tensor*> unet_tensors;
    runner.get_param_tensors(unet_tensors, "model.diffusion_model");
    // Denoising UNet keeps its output head (unlike the ReferenceNet).
    bool out_bound = false;
    for (const auto& kv : unet_tensors) {
        if (kv.first.rfind("model.diffusion_model.out.", 0) == 0) {
            out_bound = true;
            break;
        }
    }
    if (!out_bound) {
        fprintf(stderr, "error: denoising UNet did not allocate its output head (out.*)\n");
        return 1;
    }
    if (!unet_manager->register_param_tensors("denoising_unet", unet_tensors,
                                              ModelManager::ResidencyMode::ParamBackend,
                                              compute_backend, compute_backend) ||
        !unet_manager->validate_registered_tensors()) {
        fprintf(stderr, "error: failed to register denoising unet tensors with the model manager\n");
        return 1;
    }

    // --- Two forwards at t=999: row 0 = uncond (no banks), row 1 = cond
    // (banks injected). Driven through the DiffusionParams/UNetDiffusionExtra
    // path, the same entry point the generation loop will use in task 9. ---
    std::vector<float> combined(2 * half_numel);
    // Reference output, reordered per CFG half from PyTorch (c,f,h,w) into
    // this fork's ggml [w,h,c,f] layout so it lines up element-for-element
    // with `combined` (which the runner already produces in ggml order).
    std::vector<float> want_reordered(2 * half_numel);
    for (int h = 0; h < 2; ++h) {
        reorder_pytorch_cf_ggml_fc(step_out_npy.data.data() + h * half_numel,
                                   want_reordered.data() + h * half_numel,
                                   4, video_length, 64, 64);
    }
    std::vector<float> timesteps_vec(1, 999.f);
    auto timesteps = sd::Tensor<float>::from_vector(timesteps_vec);
    double half_err[2] = {0.0, 0.0};
    for (int h = 0; h < 2; ++h) {
        sd::Tensor<float> x_half({64, 64, 4, (int64_t)video_length});
        reorder_pytorch_cf_ggml_fc(step_in_npy.data.data() + h * half_numel,
                                   x_half.data(), 4, video_length, 64, 64);
        sd::Tensor<float> context_half({768, 1, 1});
        std::copy(clip_embeds_npy.data.begin() + h * 768,
                  clip_embeds_npy.data.begin() + (h + 1) * 768,
                  context_half.data());

        UNetDiffusionExtra extra;
        extra.num_video_frames = video_length;
        // Carried finding (d) from task 7's review: pass no banks at all on
        // the uncond forward - build_graph() already ignores aa_banks when
        // aa_is_uncond is set, but a null pointer states the intent (no
        // reference is ever read on this half) instead of relying on the
        // callee to notice the flag.
        extra.aa_banks         = (h == 0) ? nullptr : &banks;
        extra.aa_is_uncond     = (h == 0);
        extra.aa_pose_feature  = &pose_feature;

        DiffusionParams params;
        params.x         = &x_half;
        params.timesteps = &timesteps;
        params.context   = &context_half;
        params.extra     = extra;

        printf("running %s forward at t=999 (%d threads)...\n", h == 0 ? "UNCOND" : "COND", n_threads);
        sd::Tensor<float> out = runner.compute(n_threads, params);
        if (out.empty() || out.numel() != half_numel) {
            fprintf(stderr, "error: %s forward failed (numel %lld, expected %lld)\n",
                    h == 0 ? "uncond" : "cond",
                    (long long)out.numel(), (long long)half_numel);
            return 1;
        }
        std::copy(out.data(), out.data() + half_numel, combined.data() + h * half_numel);
        half_err[h] = rel_l2(out.data(), want_reordered.data() + h * half_numel, half_numel);
    }
    unet_manager.reset();

    double combined_err = rel_l2(combined.data(), want_reordered.data(), 2 * half_numel);
    const double tol    = 1e-3;
    printf("uncond half rel L2:   %g\n", half_err[0]);
    printf("cond half rel L2:     %g\n", half_err[1]);
    printf("combined rel L2:      %g (tolerance %g)\n", combined_err, tol);

    if (combined_err > tol) {
        fprintf(stderr, "FAIL: combined relative L2 error %g exceeds tolerance %g\n", combined_err, tol);
        return 1;
    }
    printf("PASS: denoising UNet F=%d step matches reference within tolerance %g\n", video_length, tol);
    return 0;
}

static int run_unet_step_f1_mode(int argc, char** argv) {
    return run_unet_step_mode(argc, argv, 1, "f1");
}

static int run_unet_step_f8_mode(int argc, char** argv) {
    return run_unet_step_mode(argc, argv, 8, "f8");
}

int main(int argc, char** argv) {
    sd_set_log_callback(aa_test_log_cb, nullptr);

    if (argc < 2) {
        print_usage();
        return 1;
    }

    std::string mode = argv[1];
    if (mode == "version") {
        return run_version_mode(argc - 2, argv + 2);
    } else if (mode == "pose-guider") {
        return run_pose_guider_mode(argc - 2, argv + 2);
    } else if (mode == "pose-guider-b") {
        return run_pose_guider_b_mode(argc - 2, argv + 2);
    } else if (mode == "clip-embeds") {
        return run_clip_embeds_mode(argc - 2, argv + 2);
    } else if (mode == "scheduler") {
        return run_scheduler_mode(argc - 2, argv + 2);
    } else if (mode == "context-schedule") {
        return run_context_schedule_mode(argc - 2, argv + 2);
    } else if (mode == "ref-bank") {
        return run_ref_bank_mode(argc - 2, argv + 2);
    } else if (mode == "unet-step-f1") {
        return run_unet_step_f1_mode(argc - 2, argv + 2);
    } else if (mode == "unet-step-f8") {
        return run_unet_step_f8_mode(argc - 2, argv + 2);
    } else if (mode == "-h" || mode == "--help") {
        print_usage();
        return 0;
    } else {
        fprintf(stderr, "error: unknown mode '%s'\n", mode.c_str());
        print_usage();
        return 1;
    }
}
