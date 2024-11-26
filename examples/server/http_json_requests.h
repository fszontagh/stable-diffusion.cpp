#ifndef EXAMPLES_SERVER_HTTP_JSON_REQUESTS_H
#define EXAMPLES_SERVER_HTTP_JSON_REQUESTS_H

#include "json.hpp"
#include "sd.h"
#include "stable-diffusion.h"

namespace http_jsonrequest {

    struct sd_model_load_t {
        std::string model;
        std::string hash;  // not used, but if hashing is implemented, need to find the models by hash
        std::string sha256;
        std::string vae_name;
        std::string clip_l_name;
        std::string clip_g_name;
        std::string t5xxl_name;
        std::string diffusion_model;
        std::string taesd_path;
        std::string control_net_path;
        std::string stacked_id_embed_dir;
        bool vae_decode_only = false;
        bool vae_tiling      = false;
        int n_threads;
        enum sd_type_t wtype      = sd_type_t::SD_TYPE_COUNT;
        enum rng_type_t rng_type  = rng_type_t::CUDA_RNG;
        enum schedule_t schedule  = schedule_t::DEFAULT;
        bool keep_clip_on_cpu     = false;
        bool keep_control_net_cpu = false;
        bool keep_vae_on_cpu      = false;
    };

    inline SDParams operator<<(SDParams& params, const http_jsonrequest::sd_model_load_t& modelLoad) {
        params.model_path                 = modelLoad.model;
        params.clip_l_path                = modelLoad.clip_l_name;
        params.clip_g_path                = modelLoad.clip_g_name;
        params.t5xxl_path                 = modelLoad.t5xxl_name;
        params.vae_path                   = modelLoad.vae_name;
        params.diffusion_model_path       = modelLoad.diffusion_model;
        params.taesd_path                 = modelLoad.taesd_path;
        params.control_net_path           = modelLoad.control_net_path;
        params.stacked_id_embeddings_path = modelLoad.stacked_id_embed_dir;
        params.vae_decode_only            = modelLoad.vae_decode_only;
        params.vae_tiling                 = modelLoad.vae_tiling;
        params.n_threads                  = modelLoad.n_threads;
        params.wtype                      = modelLoad.wtype;
        params.rng_type                   = modelLoad.rng_type;
        params.schedule                   = modelLoad.schedule;
        params.clip_on_cpu                = modelLoad.keep_clip_on_cpu;
        params.controlnet_on_cpu          = modelLoad.keep_control_net_cpu;
        params.vae_on_cpu                 = modelLoad.keep_vae_on_cpu;

        return params;
    }

    inline http_jsonrequest::sd_model_load_t operator>>(const SDParams& params, http_jsonrequest::sd_model_load_t& modelLoad) {
        modelLoad.model                = params.model_path;
        modelLoad.clip_l_name          = params.clip_l_path;
        modelLoad.clip_g_name          = params.clip_g_path;
        modelLoad.t5xxl_name           = params.t5xxl_path;
        modelLoad.vae_name             = params.vae_path;
        modelLoad.diffusion_model      = params.diffusion_model_path;
        modelLoad.taesd_path           = params.taesd_path;
        modelLoad.control_net_path     = params.control_net_path;
        modelLoad.stacked_id_embed_dir = params.stacked_id_embeddings_path;
        modelLoad.vae_decode_only      = params.vae_decode_only;
        modelLoad.vae_tiling           = params.vae_tiling;
        modelLoad.n_threads            = params.n_threads;
        modelLoad.wtype                = params.wtype;
        modelLoad.rng_type             = params.rng_type;
        modelLoad.schedule             = params.schedule;
        modelLoad.keep_clip_on_cpu     = params.clip_on_cpu;
        modelLoad.keep_control_net_cpu = params.controlnet_on_cpu;
        modelLoad.keep_vae_on_cpu      = params.vae_on_cpu;

        return modelLoad;
    }

    inline void to_json(nlohmann ::json& nlohmann_json_j, const sd_model_load_t& nlohmann_json_t) {
        nlohmann_json_j["model"]                = nlohmann_json_t.model;
        nlohmann_json_j["hash"]                 = nlohmann_json_t.hash;
        nlohmann_json_j["sha256"]               = nlohmann_json_t.sha256;
        nlohmann_json_j["vae_name"]             = nlohmann_json_t.vae_name;
        nlohmann_json_j["clip_l_name"]          = nlohmann_json_t.clip_l_name;
        nlohmann_json_j["clip_g_name"]          = nlohmann_json_t.clip_g_name;
        nlohmann_json_j["t5xxl_name"]           = nlohmann_json_t.t5xxl_name;
        nlohmann_json_j["diffusion_model"]      = nlohmann_json_t.diffusion_model;
        nlohmann_json_j["taesd_path"]           = nlohmann_json_t.taesd_path;
        nlohmann_json_j["control_net_path"]     = nlohmann_json_t.control_net_path;
        nlohmann_json_j["stacked_id_embed_dir"] = nlohmann_json_t.stacked_id_embed_dir;
        nlohmann_json_j["vae_decode_only"]      = nlohmann_json_t.vae_decode_only;
        nlohmann_json_j["vae_tiling"]           = nlohmann_json_t.vae_tiling;
        nlohmann_json_j["n_threads"]            = nlohmann_json_t.n_threads;
        nlohmann_json_j["wtype"]                = nlohmann_json_t.wtype;
        nlohmann_json_j["rng_type"]             = nlohmann_json_t.rng_type;
        nlohmann_json_j["schedule"]             = nlohmann_json_t.schedule;
        nlohmann_json_j["keep_clip_on_cpu"]     = nlohmann_json_t.keep_clip_on_cpu;
        nlohmann_json_j["keep_control_net_cpu"] = nlohmann_json_t.keep_control_net_cpu;
        nlohmann_json_j["keep_vae_on_cpu"]      = nlohmann_json_t.keep_vae_on_cpu;
    }
    inline void from_json(const nlohmann ::json& nlohmann_json_j, sd_model_load_t& nlohmann_json_t) {
        sd_model_load_t nlohmann_json_default_obj;
        nlohmann_json_t.model                = nlohmann_json_j.value("model", nlohmann_json_default_obj.model);
        nlohmann_json_t.hash                 = nlohmann_json_j.value("hash", nlohmann_json_default_obj.hash);
        nlohmann_json_t.sha256               = nlohmann_json_j.value("sha256", nlohmann_json_default_obj.sha256);
        nlohmann_json_t.vae_name             = nlohmann_json_j.value("vae_name", nlohmann_json_default_obj.vae_name);
        nlohmann_json_t.clip_l_name          = nlohmann_json_j.value("clip_l_name", nlohmann_json_default_obj.clip_l_name);
        nlohmann_json_t.clip_g_name          = nlohmann_json_j.value("clip_g_name", nlohmann_json_default_obj.clip_g_name);
        nlohmann_json_t.t5xxl_name           = nlohmann_json_j.value("t5xxl_name", nlohmann_json_default_obj.t5xxl_name);
        nlohmann_json_t.diffusion_model      = nlohmann_json_j.value("diffusion_model", nlohmann_json_default_obj.diffusion_model);
        nlohmann_json_t.taesd_path           = nlohmann_json_j.value("taesd_path", nlohmann_json_default_obj.taesd_path);
        nlohmann_json_t.control_net_path     = nlohmann_json_j.value("control_net_path", nlohmann_json_default_obj.control_net_path);
        nlohmann_json_t.stacked_id_embed_dir = nlohmann_json_j.value("stacked_id_embed_dir", nlohmann_json_default_obj.stacked_id_embed_dir);
        nlohmann_json_t.vae_decode_only      = nlohmann_json_j.value("vae_decode_only", nlohmann_json_default_obj.vae_decode_only);
        nlohmann_json_t.vae_tiling           = nlohmann_json_j.value("vae_tiling", nlohmann_json_default_obj.vae_tiling);
        nlohmann_json_t.n_threads            = nlohmann_json_j.value("n_threads", nlohmann_json_default_obj.n_threads);
        nlohmann_json_t.wtype                = nlohmann_json_j.value("wtype", nlohmann_json_default_obj.wtype);
        nlohmann_json_t.rng_type             = nlohmann_json_j.value("rng_type", nlohmann_json_default_obj.rng_type);
        nlohmann_json_t.schedule             = nlohmann_json_j.value("schedule", nlohmann_json_default_obj.schedule);
        nlohmann_json_t.keep_clip_on_cpu     = nlohmann_json_j.value("keep_clip_on_cpu", nlohmann_json_default_obj.keep_clip_on_cpu);
        nlohmann_json_t.keep_control_net_cpu = nlohmann_json_j.value("keep_control_net_cpu", nlohmann_json_default_obj.keep_control_net_cpu);
        nlohmann_json_t.keep_vae_on_cpu      = nlohmann_json_j.value("keep_vae_on_cpu", nlohmann_json_default_obj.keep_vae_on_cpu);
    }

    struct sdWebuiParamstxt2img {
        std::string prompt              = "";
        std::string negative_prompt     = "";
        std::vector<std::string> styles = {};  // not used here
        int64_t seed                    = 42;
        int64_t subseed                 = -1;   // not used here
        double subseed_strength         = 0.0;  // not used here
        int seed_resize_from_h          = -1;   // not used here
        int seed_resize_from_w          = -1;   // not used here
        std::string sampler_name        = "Euler a";
        std::string scheduler           = "default";
        int batch_size                  = 1;
        int n_iter                      = 1;  // not used
        int steps                       = 20;
        double cfg_scale                = 7.0;
        double distilled_cfg_scale      = 3.5;  // not used here
        int width                       = 512;
        int height                      = 512;
        bool restore_faces              = false;  // not used here
        bool tiling                     = false;
        bool do_not_save_samples        = false;  // not used here
        bool do_not_save_grid           = false;  // not used here
        double eta                      = 0.0;    // not used here
        double denoising_strength       = 0.0;
        double s_min_uncond             = 0.0;                 // not used here
        double s_churn                  = 0.0;                 // not used here
        double s_tmax                   = 0.0;                 // not used here
        double s_tmin                   = 0.0;                 // not used here
        double s_noise                  = 0.0;                 // not used here
        std::map<std::string, std::string> override_settings;  // not used here
        bool override_settings_restore_afterwards = true;      // not used here
        std::string refiner_checkpoint            = "";        // not used here
        int refiner_switch_at                     = 0;         // not used here
        bool disable_extra_networks               = false;     // not used here
        std::string firstpass_image               = "";
        std::map<std::string, std::string> comments;  // not used here
        bool enable_hr           = false;             // not used here
        int firstphase_width     = 0;                 // not used here
        int firstphase_height    = 0;                 // not used here
        double hr_scale          = 2.0;               // not used here
        std::string hr_upscaler  = "";                // not used here
        int hr_second_pass_steps = 0;                 // not used here
    };

    NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_WITH_DEFAULT(sdWebuiParamstxt2img, prompt, negative_prompt, seed, sampler_name, scheduler, batch_size, steps, cfg_scale, width, height, tiling, denoising_strength)

    inline SDParams operator<<(SDParams& sd_params, const http_jsonrequest::sdWebuiParamstxt2img& params) {
        sd_params.prompt          = params.prompt;
        sd_params.negative_prompt = params.negative_prompt;
        sd_params.seed            = params.seed;
        sd_params.sample_method   = samplerFromString(params.sampler_name);
        sd_params.schedule        = scheduleFromString(params.scheduler);
        sd_params.batch_count     = params.batch_size;
        sd_params.sample_steps    = params.steps;
        sd_params.cfg_scale       = params.cfg_scale;
        sd_params.width           = params.width;
        sd_params.height          = params.height;
        sd_params.vae_tiling      = params.tiling;
        // sd_params.clip_skip       = 1;
        sd_params.control_strength = params.denoising_strength;

        if (sd_params.seed < 0) {
            srand((int)time(NULL));
            sd_params.seed = rand();
        }
        return sd_params;
    };

}

#endif