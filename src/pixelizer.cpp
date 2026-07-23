#include "pixelizer.h"
#include "core/ggml_extend.hpp"
#include "core/util.h"
#include "model_loader.h"
#include "stable-diffusion.h"

#include <cmath>
#include <cstdlib>
#include <utility>

namespace {

    constexpr int kSizeMultiple  = 4;
    constexpr int kPixelBlock    = 4;
    constexpr int64_t kStyleCode = 2048;

    // Python's round() is round-half-to-even, so std::rint (default FE_TONEAREST) is the match here
    // and std::lround (half-away-from-zero) is not: it disagrees for e.g. 250 and 258.
    int64_t round_half_even_to_multiple(int64_t size, int multiple) {
        return (int64_t)std::rint((double)size / multiple) * multiple;
    }

    // Python's // floors; C division truncates toward zero. The two disagree on the negative odd
    // offsets produced when the rounded size grows (any size 3 mod 4).
    int64_t floor_div2(int64_t value) {
        return (int64_t)std::floor((double)value / 2.0);
    }

    // pixelization.py:52-67 crops to a multiple of 4 around the centre before normalizing. The rounded
    // size can exceed the source, in which case upstream hands PIL.crop out-of-bounds coordinates and
    // PIL silently pads with black rather than erroring; reads outside the source reproduce that by
    // leaving the zero-filled output untouched. Cropping happens on the [0,1] tensor before
    // scale_to_signed, so black is 0.0 here.
    sd::Tensor<float> center_crop_to_multiple(const sd::Tensor<float>& input, int multiple) {
        const int64_t ow = input.shape()[0];
        const int64_t oh = input.shape()[1];
        const int64_t nw = round_half_even_to_multiple(ow, multiple);
        const int64_t nh = round_half_even_to_multiple(oh, multiple);
        if (nw == ow && nh == oh) {
            return input;
        }
        const int64_t left = floor_div2(ow - nw);
        const int64_t top  = floor_div2(oh - nh);

        sd::Tensor<float> out = sd::zeros<float>({nw, nh, input.shape()[2], input.shape()[3]});
        for (int64_t n = 0; n < out.shape()[3]; n++) {
            for (int64_t c = 0; c < out.shape()[2]; c++) {
                for (int64_t y = 0; y < nh; y++) {
                    const int64_t sy = top + y;
                    if (sy < 0 || sy >= oh) {
                        continue;
                    }
                    for (int64_t x = 0; x < nw; x++) {
                        const int64_t sx = left + x;
                        if (sx < 0 || sx >= ow) {
                            continue;
                        }
                        out.index(x, y, c, n) = input.index(sx, sy, c, n);
                    }
                }
            }
        }
        return out;
    }

    // transforms.Normalize((0.5,)*3, (0.5,)*3): [0,1] -> [-1,1], and its inverse.
    void scale_to_signed(sd::Tensor<float>& tensor) {
        for (float& value : tensor.values()) {
            value = value * 2.0f - 1.0f;
        }
    }

    void scale_to_unit(sd::Tensor<float>& tensor) {
        for (float& value : tensor.values()) {
            value = (value + 1.0f) * 0.5f;
        }
    }

    // pixelization.py:74-75 downscales 4x then upscales 4x, both NEAREST. PIL samples the block
    // centre on the way down (src = floor((i + 0.5) * 4) = 4i + 2), not the top-left corner, so each
    // 4x4 block ends up filled with its own centre pixel.
    void nearest_block_quantize(sd_image_t& image, int block) {
        const char* env = getenv("SDCPP_PIXELIZE_BLOCK");
        if (env != nullptr) {
            block = atoi(env);
        }
        if (block <= 1) {
            return;
        }
        const uint32_t w = image.width;
        const uint32_t h = image.height;
        const uint32_t c = image.channel;
        for (uint32_t by = 0; by < h; by += block) {
            for (uint32_t bx = 0; bx < w; bx += block) {
                uint32_t sx = bx + block / 2;
                uint32_t sy = by + block / 2;
                if (sx >= w) sx = w - 1;
                if (sy >= h) sy = h - 1;
                for (uint32_t y = by; y < by + block && y < h; y++) {
                    for (uint32_t x = bx; x < bx + block && x < w; x++) {
                        for (uint32_t ic = 0; ic < c; ic++) {
                            image.data[(y * image.width + x) * c + ic] =
                                image.data[(sy * image.width + sx) * c + ic];
                        }
                    }
                }
            }
        }
    }

    // pixelization.py:46 feeds the reference through convert('L') replicated across 3 channels.
    sd::Tensor<float> reference_to_greyscale_tensor(sd_image_t ref_image) {
        sd::Tensor<float> rgb = sd_image_to_tensor(ref_image);
        sd::Tensor<float> out = sd::zeros<float>({rgb.shape()[0], rgb.shape()[1], 3, 1});
        for (int64_t y = 0; y < rgb.shape()[1]; y++) {
            for (int64_t x = 0; x < rgb.shape()[0]; x++) {
                float luma = 0.0f;
                if (rgb.shape()[2] >= 3) {
                    // ITU-R 601-2 luma, matching PIL's convert('L').
                    luma = 299.0f / 1000.0f * rgb.index(x, y, 0, 0) +
                           587.0f / 1000.0f * rgb.index(x, y, 1, 0) +
                           114.0f / 1000.0f * rgb.index(x, y, 2, 0);
                    luma = std::floor(luma * 255.0f) / 255.0f;
                } else {
                    luma = rgb.index(x, y, 0, 0);
                }
                for (int64_t c = 0; c < 3; c++) {
                    out.index(x, y, c, 0) = luma;
                }
            }
        }
        return out;
    }

}  // namespace

PixelizerGGML::PixelizerGGML(int n_threads,
                             bool direct,
                             int tile_size,
                             std::string backend_spec,
                             std::string params_backend_spec)
    : n_threads(n_threads),
      direct(direct),
      tile_size(tile_size),
      backend_spec(std::move(backend_spec)),
      params_backend_spec(std::move(params_backend_spec)) {
}

PixelizerGGML::~PixelizerGGML() {
    // ModelManager holds raw ggml tensor pointers owned by the runner contexts, and its teardown
    // writes through them, so it must be released before any runner.
    model_manager.reset();
    style_encoder.reset();
    alias_net.reset();
    c2p_gen.reset();
}

bool PixelizerGGML::load_style_code(sd_image_t ref_image) {
    ModelLoader& model_loader = model_manager->loader();

    if (ref_image.data == nullptr) {
        // The baked code is raw (absmax ~8.4e8); it must not be normalized.
        if (!model_loader.load_float_tensor("default_style_code", style_code, n_threads)) {
            LOG_ERROR("model file has no 'default_style_code' tensor");
            return false;
        }
        if ((int64_t)style_code.size() != kStyleCode) {
            LOG_ERROR("default_style_code has %zu entries, expected %" PRId64, style_code.size(), kStyleCode);
            return false;
        }
        return true;
    }

    style_encoder = std::make_shared<pixelization::StyleEncoderRunner>(backend_manager.runtime_backend(SDBackendModule::PIXELIZATION),
                                                                       model_loader.get_tensor_storage_map(),
                                                                       model_manager);
    std::map<std::string, ggml_tensor*> tensors;
    style_encoder->get_param_tensors(tensors);
    if (!model_manager->register_param_tensors("PIXELIZATION_STYLE",
                                               std::move(tensors),
                                               backend_manager.params_backend_is_disk(SDBackendModule::PIXELIZATION) ? ModelManager::ResidencyMode::Disk : ModelManager::ResidencyMode::ParamBackend,
                                               backend_manager.runtime_backend(SDBackendModule::PIXELIZATION),
                                               backend_manager.params_backend(SDBackendModule::PIXELIZATION)) ||
        !model_manager->validate_registered_tensors()) {
        LOG_ERROR("register pixelization style encoder tensors with model manager failed");
        return false;
    }

    sd::Tensor<float> ref_tensor = center_crop_to_multiple(reference_to_greyscale_tensor(ref_image), kSizeMultiple);
    scale_to_signed(ref_tensor);
    style_code = style_encoder->compute(n_threads, ref_tensor);
    style_encoder->free_compute_buffer();
    if ((int64_t)style_code.size() != kStyleCode) {
        LOG_ERROR("style encoder produced %zu entries, expected %" PRId64, style_code.size(), kStyleCode);
        return false;
    }
    LOG_INFO("pixelization style code computed from reference image");
    return true;
}

bool PixelizerGGML::load_from_file(const std::string& model_path,
                                   sd_image_t ref_image,
                                   int n_threads) {
    ggml_log_set(ggml_log_callback_default, nullptr);

    std::string error;
    if (!backend_manager.init(backend_spec.c_str(),
                              params_backend_spec.c_str(),
                              /*split_mode_spec=*/nullptr,
                              &error)) {
        LOG_ERROR("pixelizer backend config failed: %s", error.c_str());
        return false;
    }
    ggml_backend_t backend = backend_manager.runtime_backend(SDBackendModule::PIXELIZATION);
    if (backend == nullptr) {
        LOG_ERROR("failed to initialize %s backend", sd_backend_module_name(SDBackendModule::PIXELIZATION));
        return false;
    }
    ggml_backend_t params_backend = backend_manager.params_backend(SDBackendModule::PIXELIZATION);
    if (params_backend == nullptr) {
        LOG_ERROR("failed to initialize %s params backend", sd_backend_module_name(SDBackendModule::PIXELIZATION));
        return false;
    }

    model_manager = std::make_shared<ModelManager>();
    model_manager->set_n_threads(n_threads);
    model_manager->set_enable_mmap(false);

    ModelLoader& model_loader = model_manager->loader();
    if (!model_loader.init_from_file_and_convert_name(model_path, "", VERSION_PIXELIZATION)) {
        LOG_ERROR("init model loader from file failed: '%s'", model_path.c_str());
        return false;
    }
    // No set_wtype_override: default_style_code and the mod_conv weights must stay F32 because
    // the raw style code overflows F16.

    c2p_gen   = std::make_shared<pixelization::C2PGenRunner>(backend,
                                                             model_loader.get_tensor_storage_map(),
                                                             model_manager);
    alias_net = std::make_shared<pixelization::AliasNetRunner>(backend,
                                                               model_loader.get_tensor_storage_map(),
                                                               model_manager);
    if (direct) {
        c2p_gen->set_conv2d_direct_enabled(true);
        alias_net->set_conv2d_direct_enabled(true);
    }

    const ModelManager::ResidencyMode residency =
        backend_manager.params_backend_is_disk(SDBackendModule::PIXELIZATION) ? ModelManager::ResidencyMode::Disk : ModelManager::ResidencyMode::ParamBackend;

    std::map<std::string, ggml_tensor*> c2p_tensors;
    c2p_gen->get_param_tensors(c2p_tensors);
    std::map<std::string, ggml_tensor*> alias_tensors;
    alias_net->get_param_tensors(alias_tensors);
    if (!model_manager->register_param_tensors("PIXELIZATION_C2P", std::move(c2p_tensors), residency, backend, params_backend) ||
        !model_manager->register_param_tensors("PIXELIZATION_ALIAS", std::move(alias_tensors), residency, backend, params_backend) ||
        !model_manager->validate_registered_tensors()) {
        LOG_ERROR("register pixelization tensors with model manager failed");
        return false;
    }

    return load_style_code(ref_image);
}

sd::Tensor<float> PixelizerGGML::pixelize_tensor(const sd::Tensor<float>& input_tensor) {
    auto run = [&](const sd::Tensor<float>& x) -> sd::Tensor<float> {
        sd::Tensor<float> generated = c2p_gen->compute(n_threads, x, style_code);
        if (generated.empty()) {
            LOG_ERROR("c2pgen compute failed");
            return {};
        }
        return alias_net->compute(n_threads, generated);
    };

    sd::Tensor<float> pixelized;
    if (tile_size <= 0 || (input_tensor.shape()[0] <= tile_size && input_tensor.shape()[1] <= tile_size)) {
        pixelized = run(input_tensor);
    } else {
        // Instance norm reduces over the whole spatial extent, so tiles cannot reproduce the
        // untiled result. Tiling trades that fidelity for a bounded working set.
        pixelized = process_tiles_2d(input_tensor,
                                     static_cast<int>(input_tensor.shape()[0]),
                                     static_cast<int>(input_tensor.shape()[1]),
                                     1,
                                     tile_size,
                                     tile_size,
                                     0.25f,
                                     false,
                                     false,
                                     run);
    }
    c2p_gen->free_compute_buffer();
    alias_net->free_compute_buffer();
    if (pixelized.empty()) {
        LOG_ERROR("pixelization compute failed");
        return {};
    }
    return pixelized;
}

sd_image_t PixelizerGGML::pixelize(sd_image_t input_image) {
    sd_image_t pixelized_image = {0, 0, 0, nullptr};

    sd::Tensor<float> input_tensor = center_crop_to_multiple(sd_image_to_tensor(input_image), kSizeMultiple);
    // The encoder downsamples twice and reflect-pads at each stage, and reflect padding requires
    // pad < extent, so anything under two crop multiples aborts inside ggml_ext_pad_reflect_2d.
    constexpr int kMinSize = kSizeMultiple * 2;
    if (input_tensor.shape()[0] < kMinSize || input_tensor.shape()[1] < kMinSize) {
        LOG_ERROR("pixelization needs an input of at least %dx%d", kMinSize, kMinSize);
        return pixelized_image;
    }
    scale_to_signed(input_tensor);

    LOG_INFO("pixelizing (%i x %i)",
             (int)input_tensor.shape()[0], (int)input_tensor.shape()[1]);
    const int64_t t0 = ggml_time_ms();

    sd::Tensor<float> pixelized = pixelize_tensor(input_tensor);
    if (pixelized.empty()) {
        return pixelized_image;
    }
    scale_to_unit(pixelized);

    pixelized_image = tensor_to_sd_image(pixelized);
    nearest_block_quantize(pixelized_image, kPixelBlock);

    LOG_INFO("input_image_tensor pixelized, taking %.2fs", (ggml_time_ms() - t0) / 1000.0f);
    return pixelized_image;
}

struct pixelizer_ctx_t {
    PixelizerGGML* pixelizer = nullptr;
};

pixelizer_ctx_t* new_pixelizer_ctx(const char* model_path_c_str,
                                   sd_image_t ref_image,
                                   bool direct,
                                   int n_threads,
                                   int tile_size,
                                   const char* backend,
                                   const char* params_backend) {
    pixelizer_ctx_t* pixelizer_ctx = (pixelizer_ctx_t*)malloc(sizeof(pixelizer_ctx_t));
    if (pixelizer_ctx == nullptr) {
        return nullptr;
    }

    pixelizer_ctx->pixelizer = new PixelizerGGML(n_threads, direct, tile_size, SAFE_STR(backend), SAFE_STR(params_backend));
    if (pixelizer_ctx->pixelizer == nullptr) {
        free(pixelizer_ctx);
        return nullptr;
    }

    if (!pixelizer_ctx->pixelizer->load_from_file(SAFE_STR(model_path_c_str), ref_image, n_threads)) {
        delete pixelizer_ctx->pixelizer;
        pixelizer_ctx->pixelizer = nullptr;
        free(pixelizer_ctx);
        return nullptr;
    }
    return pixelizer_ctx;
}

void free_pixelizer_ctx(pixelizer_ctx_t* pixelizer_ctx) {
    if (pixelizer_ctx == nullptr) {
        return;
    }
    if (pixelizer_ctx->pixelizer != nullptr) {
        delete pixelizer_ctx->pixelizer;
        pixelizer_ctx->pixelizer = nullptr;
    }
    free(pixelizer_ctx);
}

bool pixelize(pixelizer_ctx_t* pixelizer_ctx,
              sd_image_t input_image,
              sd_image_t** images_out,
              int* num_images_out) {
    if (images_out != nullptr) {
        *images_out = nullptr;
    }
    if (num_images_out != nullptr) {
        *num_images_out = 0;
    }
    if (pixelizer_ctx == nullptr || pixelizer_ctx->pixelizer == nullptr) {
        return false;
    }

    sd_image_t* result_images = (sd_image_t*)calloc(1, sizeof(sd_image_t));
    if (result_images == nullptr) {
        return false;
    }

    result_images[0] = pixelizer_ctx->pixelizer->pixelize(input_image);
    if (result_images[0].data == nullptr) {
        free(result_images);
        return false;
    }

    if (num_images_out != nullptr) {
        *num_images_out = 1;
    }
    if (images_out != nullptr) {
        *images_out = result_images;
    } else {
        free_sd_images(result_images, 1);
    }
    return true;
}
