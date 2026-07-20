// Standalone numerical-bisection harness for the JoyAI-Image DiT.
//
// Loads a small fixed-weight model, runs one forward pass on fixed inputs read
// from raw f32 dumps, and writes the resulting tensor back out so it can be
// diffed against the reference (diffusers) implementation stage by stage.
// Select the stage with SDCPP_JOYAI_STOP=img_in|block0|block1|<unset for final>.
//
// usage: joyai-test <model.safetensors> <dir-with-x.bin,txt.bin,t.bin> <out.bin>

#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

#include "model/diffusion/joyai_image.hpp"
#include "model_loader.h"

static std::vector<float> read_bin(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) {
        fprintf(stderr, "cannot open %s\n", path.c_str());
        exit(1);
    }
    std::streamsize n = f.tellg();
    f.seekg(0);
    std::vector<float> v(static_cast<size_t>(n) / sizeof(float));
    f.read(reinterpret_cast<char*>(v.data()), n);
    return v;
}

static void write_bin(const std::string& path, const sd::Tensor<float>& t) {
    std::ofstream f(path, std::ios::binary);
    f.write(reinterpret_cast<const char*>(t.data()), t.numel() * sizeof(float));
    printf("wrote %s  shape=[", path.c_str());
    for (size_t i = 0; i < t.shape().size(); i++) {
        printf("%s%lld", i ? "," : "", (long long)t.shape()[i]);
    }
    printf("]  numel=%lld\n", (long long)t.numel());
}

int main(int argc, char** argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: %s <model.safetensors> <input-dir> <out.bin>\n", argv[0]);
        return 1;
    }
    const std::string model_path = argv[1];
    const std::string in_dir     = argv[2];
    const std::string out_path   = argv[3];

    ggml_backend_t backend = sd_backend_cpu_init();

    auto model_manager        = std::make_shared<ModelManager>();
    ModelLoader& model_loader = model_manager->loader();
    if (!model_loader.init_from_file_and_convert_name(model_path, "model.diffusion_model.")) {
        fprintf(stderr, "failed to load %s\n", model_path.c_str());
        return 1;
    }
    auto& tensor_storage_map = model_loader.get_tensor_storage_map();

    auto runner = std::make_shared<JoyImage::JoyImageRunner>(backend,
                                                             tensor_storage_map,
                                                             "model.diffusion_model",
                                                             VERSION_JOYAI_IMAGE_EDIT,
                                                             model_manager);

    if (!model_manager->register_runner_params("joyai test",
                                               *runner,
                                               "model.diffusion_model",
                                               ModelManager::ResidencyMode::ParamBackend,
                                               backend,
                                               backend) ||
        !model_manager->validate_registered_tensors()) {
        fprintf(stderr, "tensor registration failed\n");
        return 1;
    }

    // Reference dumps are torch-ordered; sd::Tensor shapes are ggml-ordered
    // (ne0 first), so the dimension lists below are reversed on purpose.
    auto x_data   = read_bin(in_dir + "/x.bin");      // torch [1,16,1,8,8]
    auto txt_data = read_bin(in_dir + "/txt.bin");    // torch [1,5,4096]
    auto t_data   = read_bin(in_dir + "/t.bin");      // [1]

    int64_t C = 16, N = 1;
    // square latent inferred from the dump size
    int64_t hw = static_cast<int64_t>(x_data.size()) / C;
    int64_t W  = static_cast<int64_t>(llround(std::sqrt(static_cast<double>(hw))));
    int64_t H  = W;
    if (W * H * C != static_cast<int64_t>(x_data.size())) {
        fprintf(stderr, "x.bin size %zu is not a square 16-channel latent\n", x_data.size());
        return 1;
    }
    int64_t L = static_cast<int64_t>(txt_data.size()) / 4096;

    sd::Tensor<float> x({W, H, C, N}, x_data);
    sd::Tensor<float> context({4096, L, N}, txt_data);
    sd::Tensor<float> timesteps({1}, t_data);

    printf("x=[%lld,%lld,%lld,%lld] context=[4096,%lld,1] t=%.1f\n",
           (long long)W, (long long)H, (long long)C, (long long)N, (long long)L, t_data[0]);

    auto out = runner->compute(1, x, timesteps, context, {});
    if (out.empty()) {
        fprintf(stderr, "compute returned empty\n");
        return 1;
    }
    write_bin(out_path, out);
    return 0;
}
