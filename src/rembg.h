#ifndef __SD_REMBG_H__
#define __SD_REMBG_H__

#include "core/ggml_extend_backend.h"
#include "core/tensor.hpp"
#include "model/segmentation/u2net.hpp"
#include "model_manager.h"
#include "stable-diffusion.h"

#include <memory>
#include <string>

// Background removal runner.  Loads a U^2-Net checkpoint and produces a per-pixel
// saliency mask which can be composited as an alpha channel onto the input.
struct RemBGGGML {
    SDBackendManager backend_manager;
    std::shared_ptr<ModelManager> model_manager;
    ggml_type model_data_type = GGML_TYPE_F16;
    std::shared_ptr<U2NetRunner> u2net;
    std::string model_path;
    int n_threads;
    std::string backend_spec;
    std::string params_backend_spec;

    static constexpr int kInputSize = 320;  // U^2-Net trained resolution.

    RemBGGGML(int n_threads,
              std::string backend_spec        = "",
              std::string params_backend_spec = "");
    ~RemBGGGML();

    bool load_from_file(const std::string& path, int n_threads);

    // Run U^2-Net on the input image and return a copy of the input image with
    // its alpha channel replaced by the saliency mask.  RGB input becomes RGBA.
    sd_image_t remove_background(sd_image_t input_image);
};

#endif  // __SD_REMBG_H__
