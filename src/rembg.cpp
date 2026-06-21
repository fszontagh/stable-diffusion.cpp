#include "rembg.h"

#include "core/ggml_extend.hpp"
#include "core/util.h"
#include "model_loader.h"
#include "stable-diffusion.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <utility>

namespace {

// ImageNet RGB normalization used by U^2-Net.
constexpr float kMean[3] = {0.485f, 0.456f, 0.406f};
constexpr float kStd[3]  = {0.229f, 0.224f, 0.225f};

// In-place: subtract mean / divide by std along channel axis.  Input tensor is
// (W, H, C, 1) with values in [0,1].
void apply_imagenet_norm(sd::Tensor<float>& t) {
    const int W = static_cast<int>(t.shape()[0]);
    const int H = static_cast<int>(t.shape()[1]);
    const int C = static_cast<int>(t.shape()[2]);
    for (int x = 0; x < W; ++x) {
        for (int y = 0; y < H; ++y) {
            for (int c = 0; c < C && c < 3; ++c) {
                float v        = t.index(x, y, c, 0);
                v              = (v - kMean[c]) / kStd[c];
                t.index(x, y, c, 0) = v;
            }
        }
    }
}

// Bilinear resample of a single-channel mask (W,H,1,1) at native resolution
// up to target_w x target_h. Output values stay in [0,1].
sd::Tensor<float> resize_mask_bilinear(const sd::Tensor<float>& mask, int target_w, int target_h) {
    const int sw = static_cast<int>(mask.shape()[0]);
    const int sh = static_cast<int>(mask.shape()[1]);
    sd::Tensor<float> out = sd::zeros<float>({target_w, target_h, 1, 1});
    const float fx = (sw - 1) / static_cast<float>(std::max(1, target_w - 1));
    const float fy = (sh - 1) / static_cast<float>(std::max(1, target_h - 1));
    for (int y = 0; y < target_h; ++y) {
        const float sy = y * fy;
        const int y0 = static_cast<int>(std::floor(sy));
        const int y1 = std::min(y0 + 1, sh - 1);
        const float wy = sy - y0;
        for (int x = 0; x < target_w; ++x) {
            const float sx = x * fx;
            const int x0 = static_cast<int>(std::floor(sx));
            const int x1 = std::min(x0 + 1, sw - 1);
            const float wx = sx - x0;
            const float v00 = mask.index(x0, y0, 0, 0);
            const float v10 = mask.index(x1, y0, 0, 0);
            const float v01 = mask.index(x0, y1, 0, 0);
            const float v11 = mask.index(x1, y1, 0, 0);
            const float top    = v00 * (1.f - wx) + v10 * wx;
            const float bottom = v01 * (1.f - wx) + v11 * wx;
            out.index(x, y, 0, 0) = top * (1.f - wy) + bottom * wy;
        }
    }
    return out;
}

}  // namespace

RemBGGGML::RemBGGGML(int n_threads_,
                     std::string backend_spec_,
                     std::string params_backend_spec_)
    : n_threads(n_threads_),
      backend_spec(std::move(backend_spec_)),
      params_backend_spec(std::move(params_backend_spec_)) {
}

RemBGGGML::~RemBGGGML() {
    model_manager.reset();
    u2net.reset();
}

bool RemBGGGML::load_from_file(const std::string& path, int n_threads_in) {
    ggml_log_set(ggml_log_callback_default, nullptr);
    n_threads = n_threads_in;
    model_path = path;

    std::string error;
    if (!backend_manager.init(backend_spec.c_str(), params_backend_spec.c_str(), &error)) {
        LOG_ERROR("rembg backend config failed: %s", error.c_str());
        return false;
    }

    // The U^2-Net runner shares the UPSCALER backend module bucket for now;
    // both are post-processing models that don't need their own assignment.
    auto module = SDBackendModule::UPSCALER;
    auto backend = backend_manager.runtime_backend(module);
    auto params_backend = backend_manager.params_backend(module);
    if (backend == nullptr || params_backend == nullptr) {
        LOG_ERROR("rembg backend initialization failed");
        return false;
    }

    model_manager = std::make_shared<ModelManager>();
    model_manager->set_n_threads(n_threads);
    model_manager->set_enable_mmap(false);

    ModelLoader& loader = model_manager->loader();
    if (!loader.init_from_file_and_convert_name(path, "", VERSION_U2NET)) {
        LOG_ERROR("rembg: init model loader from file failed: '%s'", path.c_str());
        return false;
    }
    loader.set_wtype_override(model_data_type);

    u2net = std::make_shared<U2NetRunner>(backend, loader.get_tensor_storage_map(), model_manager);
    if (!u2net || !u2net->network) {
        LOG_ERROR("rembg: init U^2-Net network from metadata failed");
        return false;
    }

    std::map<std::string, ggml_tensor*> tensors;
    u2net->get_param_tensors(tensors);
    const auto residency = backend_manager.params_backend_is_disk(module)
                               ? ModelManager::ResidencyMode::Disk
                               : ModelManager::ResidencyMode::ParamBackend;
    if (!model_manager->register_param_tensors("U2Net", std::move(tensors), residency, backend, params_backend) ||
        !model_manager->validate_registered_tensors()) {
        LOG_ERROR("rembg: register U^2-Net tensors with model manager failed");
        return false;
    }
    return true;
}

sd_image_t RemBGGGML::remove_background(sd_image_t input_image) {
    sd_image_t empty{0, 0, 0, nullptr};
    if (input_image.data == nullptr || u2net == nullptr) {
        return empty;
    }

    const int W = static_cast<int>(input_image.width);
    const int H = static_cast<int>(input_image.height);

    // U^2-Net is trained on 320x320 RGB; resize input to its training size and
    // normalize with ImageNet statistics.
    sd::Tensor<float> x = sd_image_to_tensor(input_image, kInputSize, kInputSize);
    if (x.empty()) {
        return empty;
    }
    apply_imagenet_norm(x);

    sd::Tensor<float> mask_small = u2net->compute(n_threads, x);
    u2net->free_compute_buffer();
    if (mask_small.empty()) {
        LOG_ERROR("rembg: u2net forward failed");
        return empty;
    }

    // Min-max normalize the mask to [0,1] (the official U^2-Net postproc).
    // This sharpens edges and avoids the soft-alpha white-bg bleed.
    {
        float mn = mask_small.index(0, 0, 0, 0);
        float mx = mn;
        for (int yy = 0; yy < kInputSize; ++yy) {
            for (int xx = 0; xx < kInputSize; ++xx) {
                const float v = mask_small.index(xx, yy, 0, 0);
                if (v < mn) mn = v;
                if (v > mx) mx = v;
            }
        }
        const float range = std::max(mx - mn, 1e-6f);
        for (int yy = 0; yy < kInputSize; ++yy) {
            for (int xx = 0; xx < kInputSize; ++xx) {
                mask_small.index(xx, yy, 0, 0) = (mask_small.index(xx, yy, 0, 0) - mn) / range;
            }
        }
    }

    sd::Tensor<float> mask = resize_mask_bilinear(mask_small, W, H);

    // Morphological cleanup of the resized mask.
    //  - Open (erode K then dilate K) removes small false-positive islands.
    //  - An extra trailing erode trims the remaining white-background bleed
    //    around the subject's outer edge.
    auto apply_filter = [&](bool dilate) {
        sd::Tensor<float> tmp = sd::zeros<float>({W, H, 1, 1});
        for (int y = 0; y < H; ++y) {
            for (int x = 0; x < W; ++x) {
                float m = mask.index(x, y, 0, 0);
                for (int dy = -1; dy <= 1; ++dy) {
                    for (int dx = -1; dx <= 1; ++dx) {
                        const int nx = std::clamp(x + dx, 0, W - 1);
                        const int ny = std::clamp(y + dy, 0, H - 1);
                        const float n = mask.index(nx, ny, 0, 0);
                        m = dilate ? std::max(m, n) : std::min(m, n);
                    }
                }
                tmp.index(x, y, 0, 0) = m;
            }
        }
        std::swap(mask, tmp);
    };

    constexpr int open_radius  = 5;  // kills islands smaller than ~5 px
    constexpr int extra_erode  = 3;  // trims residual white-bg bleed
    for (int i = 0; i < open_radius; ++i)   apply_filter(/*dilate=*/false);
    for (int i = 0; i < open_radius; ++i)   apply_filter(/*dilate=*/true);
    for (int i = 0; i < extra_erode; ++i)   apply_filter(/*dilate=*/false);

    // Build RGBA output: keep input RGB, replace alpha with mask*255.
    sd_image_t out{};
    out.width   = static_cast<uint32_t>(W);
    out.height  = static_cast<uint32_t>(H);
    out.channel = 4;
    out.data    = static_cast<uint8_t*>(std::malloc(static_cast<size_t>(W) * H * 4));
    if (out.data == nullptr) {
        return empty;
    }
    const int src_c = static_cast<int>(input_image.channel);
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            const uint8_t* sp = input_image.data + (y * W + x) * src_c;
            uint8_t* dp       = out.data + (y * W + x) * 4;
            dp[0] = sp[0];
            dp[1] = (src_c > 1) ? sp[1] : sp[0];
            dp[2] = (src_c > 2) ? sp[2] : sp[0];
            const float m = std::clamp(mask.index(x, y, 0, 0), 0.f, 1.f);
            dp[3] = static_cast<uint8_t>(std::lround(m * 255.f));
        }
    }
    return out;
}

// ---- C API ----

struct rembg_ctx_t {
    RemBGGGML* impl = nullptr;
};

extern "C" {

SD_API rembg_ctx_t* new_rembg_ctx(const char* model_path,
                                  int n_threads,
                                  const char* backend,
                                  const char* params_backend) {
    if (model_path == nullptr || *model_path == 0) {
        return nullptr;
    }
    auto* ctx  = static_cast<rembg_ctx_t*>(std::malloc(sizeof(rembg_ctx_t)));
    if (ctx == nullptr) {
        return nullptr;
    }
    ctx->impl = new RemBGGGML(n_threads, SAFE_STR(backend), SAFE_STR(params_backend));
    if (!ctx->impl->load_from_file(model_path, n_threads)) {
        delete ctx->impl;
        std::free(ctx);
        return nullptr;
    }
    return ctx;
}

SD_API sd_image_t remove_background(rembg_ctx_t* ctx, sd_image_t input_image) {
    if (ctx == nullptr || ctx->impl == nullptr) {
        return sd_image_t{0, 0, 0, nullptr};
    }
    return ctx->impl->remove_background(input_image);
}

SD_API void free_rembg_ctx(rembg_ctx_t* ctx) {
    if (ctx == nullptr) {
        return;
    }
    delete ctx->impl;
    std::free(ctx);
}

}  // extern "C"
