#ifndef EXAMPLES_SERVER_HTTP_JSON_RESPONSES_H
#define EXAMPLES_SERVER_HTTP_JSON_RESPONSES_H

#include <string>
#include <string_view>
#include "base64.h"
#include "json.hpp"
#include "stable-diffusion.h"

namespace http_jsonresponse {
    // NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_WITH_DEFAULT(SDParams, n_threads, mode, model_path, clip_l_path, clip_g_path, t5xxl_path, diffusion_model_path, vae_path, embeddings_path, stacked_id_embeddings_path, lora_model_dir, wtype, output_path, input_path, prompt, negative_prompt, min_cfg, cfg_scale, guidance, style_ratio, clip_skip, width, height, batch_count, sample_method, schedule, sample_steps, strength, rng_type, seed, verbose, vae_tiling, normalize_input, clip_on_cpu, color, input_id_images_path, skip_layers, slg_scale, skip_layer_start, skip_layer_end)

    struct sd_sampler_info_t {
        std::string name;
        std::vector<std::string> aliases                     = {};
        sample_method_t method                               = sample_method_t::N_SAMPLE_METHODS;
        std::unordered_map<std::string, std::string> options = {};
    };

    NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_WITH_DEFAULT(sd_sampler_info_t, name, aliases, options)

    struct sd_scheduler_info_t {
        std::string name;
        std::string label;
        std::vector<std::string> aliases;
        schedule_t schedule   = schedule_t::N_SCHEDULES;
        int default_rho       = -1;
        bool need_inner_model = false;
    };
    NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_WITH_DEFAULT(sd_scheduler_info_t, name, label, aliases, default_rho, need_inner_model)

    struct sd_models_info_t {
        std::string title;
        std::string model_name;
        std::string hash;
        std::string sha256;
        std::string filename;
        std::string config;
    };
    NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_WITH_DEFAULT(sd_models_info_t, title, model_name, hash, sha256, filename, config)
}

struct sd_generated_image_t : public sd_image_t {
    std::string prefix      = "image/png;base64,";
    std::string output_path = "";
    std::string url         = "";
    std::string filename    = "";

    sd_generated_image_t() = default;
    sd_generated_image_t(sd_image_t& sd)
        : sd_image_t(sd) {
    };
};

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_WITH_DEFAULT(sd_generated_image_t, width, height, channel, output_path, url)
#endif