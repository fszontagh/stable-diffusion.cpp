#ifndef __SD_PIXELIZER_H__
#define __SD_PIXELIZER_H__

#include "core/ggml_extend_backend.h"
#include "core/tensor.hpp"
#include "model/pixelization/pixelization.hpp"
#include "model_manager.h"
#include "stable-diffusion.h"

#include <memory>
#include <string>
#include <vector>

struct PixelizerGGML {
    SDBackendManager backend_manager;
    std::shared_ptr<ModelManager> model_manager;
    std::shared_ptr<pixelization::C2PGenRunner> c2p_gen;
    std::shared_ptr<pixelization::AliasNetRunner> alias_net;
    // ModelManager keeps borrowed ggml_tensor* from this runner's params_ctx, and there is no
    // way to unregister them, so it must outlive the manager even though it only runs at load.
    std::shared_ptr<pixelization::StyleEncoderRunner> style_encoder;
    std::vector<float> style_code;
    int n_threads;
    bool direct   = false;
    int tile_size = 0;
    std::string backend_spec;
    std::string params_backend_spec;

    PixelizerGGML(int n_threads,
                  bool direct                     = false,
                  int tile_size                   = 0,
                  std::string backend_spec        = "",
                  std::string params_backend_spec = "");
    ~PixelizerGGML();

    // ref_image.data == nullptr selects the default_style_code baked into the model file, which
    // also skips loading and running the VGG19/PBEnc/MLP branch entirely.
    bool load_from_file(const std::string& model_path,
                        sd_image_t ref_image,
                        int n_threads);
    sd::Tensor<float> pixelize_tensor(const sd::Tensor<float>& input_tensor);
    sd_image_t pixelize(sd_image_t input_image);

private:
    bool load_style_code(sd_image_t ref_image);
};

#endif  // __SD_PIXELIZER_H__
