// sd-aa-test: minimal, standalone diagnostic tool for the AnimateAnyone model
// family registration. It does NOT build a full sd_ctx_t - it only exercises
// ModelLoader::init_from_file() + ModelLoader::get_sd_version() and applies
// the version-promotion rule that new_sd_ctx() applies at init time
// (see stable-diffusion.cpp: reference_net_path non-empty promotes a
// SD1-signature diffusion model to VERSION_ANIMATE_ANYONE).

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>

#include "model.h"
#include "model_loader.h"
#include "model_manager.h"
#include "stable-diffusion.h"

#include "core/ggml_extend.hpp"
#include "core/ggml_extend_backend.h"
#include "core/tensor.hpp"
#include "core/util.h"
#include "model/adapter/pose_guider.hpp"

#include "npy.hpp"

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
            "      Loads pose_guider.pth (default: env SDCPP_AA_FIXTURES-independent path\n"
            "      /data/sdcpp-pixel-refs/weights/AnimateAnyone/pose_guider.pth), runs the\n"
            "      Moore PoseGuiderA forward pass on <fixtures>/pose.png (default fixtures\n"
            "      dir: $SDCPP_AA_FIXTURES or /data/sdcpp-pixel-refs/fixtures), and compares\n"
            "      against <fixtures>/pose_guider_a_out.npy. Prints the relative L2 error.\n"
            "      Exits 0 on pass (rel L2 <= 1e-3), 1 on failure.\n");
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

// Standalone GGMLRunner wrapper around AnimateAnyone::PoseGuiderA, modeled
// directly on LTXVUpsampler::LatentUpsamplerRunner (src/model/upscaler/ltx_latent_upscaler.hpp:431).
struct PoseGuiderRunner : public GGMLRunner {
    AnimateAnyone::PoseGuiderA model;

    PoseGuiderRunner(ggml_backend_t backend,
                     const String2TensorStorage& tensor_storage_map,
                     std::shared_ptr<RunnerWeightManager> weight_manager = nullptr)
        : GGMLRunner(backend, weight_manager) {
        model.init(params_ctx, tensor_storage_map, "");
    }

    std::string get_desc() override {
        return "pose_guider";
    }

    void get_param_tensors(std::map<std::string, ggml_tensor*>& tensors) {
        model.get_param_tensors(tensors);
    }

    ggml_cgraph* build_graph(const sd::Tensor<float>& pose_tensor) {
        ggml_cgraph* gf  = new_graph_custom(4096);
        ggml_tensor* x   = make_input(pose_tensor);
        auto runner_ctx  = get_context();
        ggml_tensor* out = model.forward(&runner_ctx, x);
        ggml_build_forward_expand(gf, out);
        return gf;
    }

    sd::Tensor<float> compute(int n_threads, const sd::Tensor<float>& pose_tensor) {
        auto get_graph = [&]() -> ggml_cgraph* { return build_graph(pose_tensor); };
        // GGMLRunner::compute() drops trailing singleton dims (ggml_n_dims
        // ignores them), so a batch-1 [W,H,C,N=1] result comes back as a 3D
        // [W,H,C] tensor unless restored - matches the LatentUpsamplerRunner
        // precedent (ltx_latent_upscaler.hpp:502).
        return restore_trailing_singleton_dims(GGMLRunner::compute<float>(get_graph, n_threads, false, false, false), 4);
    }
};

static int run_pose_guider_mode(int argc, char** argv) {
    std::string weights_path = "/data/sdcpp-pixel-refs/weights/AnimateAnyone/pose_guider.pth";
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

    PoseGuiderRunner runner(cpu_backend, model_loader.get_tensor_storage_map(), model_manager);

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
    } else if (mode == "-h" || mode == "--help") {
        print_usage();
        return 0;
    } else {
        fprintf(stderr, "error: unknown mode '%s'\n", mode.c_str());
        print_usage();
        return 1;
    }
}
