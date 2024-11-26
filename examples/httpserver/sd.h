#ifndef EXAMPLES_SERVER_SD_H
#define EXAMPLES_SERVER_SD_H

#include <cstdarg>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <ostream>
#include <string>
#include <vector>

#include "ggml.h"
#include "httplib.h"
#include "json.hpp"
#include "model.h"
#include "server/latent_preview.h"
#include "stable-diffusion.h"

#include "http_json_responses.h"

// enable logging in the server
#define LOG_BUFFER_SIZE 1024

static void* server_log_params = NULL;

inline const char* rng_type_to_str[] = {
    "std_default",
    "cuda",
};

// Names of the sampler method, same order as enum sample_method in stable-diffusion.h
inline const char* sample_method_str[] = {
    "euler_a",
    "euler",
    "heun",
    "dpm2",
    "dpm++2s_a",
    "dpm++2m",
    "dpm++2mv2",
    "ipndm",
    "ipndm_v",
    "lcm",
};

// label must be the same as the sdwebui
// first alias must be the same as in sdcpp, second alias must be the same as in sdwebui if not equal with sdcpp
// the order must be same ass stable-diffusion.h
// the labels are CASE sensitive
// http://127.0.0.1:7860/sdapi/v1/samplers

inline const http_jsonresponse::sd_sampler_info_t sample_method_info[] = {
    {"Euler a", {"euler_a", "k_euler_a", "k_euler_ancestral"}, sample_method_t::EULER_A},
    {"Euler", {"euler", "k_euler"}, sample_method_t::EULER},
    {"Heun", {"heun", "k_heun"}, sample_method_t::HEUN},
    {"dpm2", {"dpm2", "k_dpm2"}, sample_method_t::DPM2},
    {"DPM++ 2S a", {"dpm++2s_a", "k_dpmpp_2s_a"}, sample_method_t::DPMPP2S_A},
    {"dpm++2m", {"dpm++2m", "k_dpm++2m", "k_dpm++2m"}, sample_method_t::DPMPP2M},
    {"dpm++2mv2", {"dpm++2mv2", "k_dpm++2mv2", "k_dpm++2mv2"}, sample_method_t::DPMPP2Mv2},
    {"ipndm", {"ipndm"}, sample_method_t::IPNDM},
    {"ipndm_v", {"ipndm_v"}, sample_method_t::IPNDM_V},
    {"lcm", {"lcm", "k_lcm"}, sample_method_t::LCM}};

// Names of the sigma schedule overrides, same order as sample_schedule in stable-diffusion.h
inline const char* schedule_str[] = {
    "default",
    "discrete",
    "karras",
    "exponential",
    "ays",
    "gits",
};

// http://127.0.0.1:7860/sdapi/v1/schedulers
// here must be match one alias in sdcpp and sdwebuiforge to identify the scheduler
inline const http_jsonresponse::sd_scheduler_info_t schedulers_info[] = {
    {"default", "Default", {"default", "automatic"}, schedule_t::DEFAULT},
    {"discrete", "Discrete", {"discrete"}, schedule_t::DISCRETE},
    {"karras", "Karras", {"karras"}, schedule_t::KARRAS},
    {"exponential", "Exponential", {"exponential"}, schedule_t::EXPONENTIAL},
    {"ays", "Align Your Steps", {"ays", "align_your_steps"}, schedule_t::AYS},
    {"gits", "Align Your Steps GITS", {"gits", "align_your_steps_GITS"}, schedule_t::GITS},
};
enum SDMode {
    TXT2IMG,
    IMG2IMG,
    IMG2VID,
    CONVERT,
    MODE_COUNT
};

inline const sample_method_t samplerFromString(const std::string& str) {
    for (const auto& info : sample_method_info) {
        if (info.name == str) {
            return info.method;
        }
        for (const auto& alias : info.aliases) {
            if (alias == str) {
                return info.method;
            }
        }
    }
    return sample_method_t::EULER;
}
inline const schedule_t scheduleFromString(const std::string& str) {
    for (const auto& info : schedulers_info) {
        if (info.name == str) {
            return info.schedule;
        }
        for (const auto& alias : info.aliases) {
            if (alias == str) {
                return info.schedule;
            }
        }
    }
    return N_SCHEDULES;
}

struct SDParams {
    int n_threads = -1;
    SDMode mode   = TXT2IMG;

    // models
    std::string model_path;
    std::string clip_l_path;
    std::string clip_g_path;
    std::string t5xxl_path;
    std::string diffusion_model_path;
    std::string vae_path;
    std::string taesd_path;
    std::string embeddings_path;
    std::string stacked_id_embeddings_path;
    std::string lora_model_dir;
    std::string control_net_path;

    sd_type_t wtype         = SD_TYPE_COUNT;
    std::string output_path = "output.png";
    std::string input_path;

    std::string prompt;
    std::string negative_prompt;
    float min_cfg     = 1.0f;
    float cfg_scale   = 7.0f;
    float guidance    = 3.5f;
    float style_ratio = 20.f;
    int clip_skip     = -1;  // <= 0 represents unspecified
    int width         = 512;
    int height        = 512;
    int batch_count   = 1;

    sample_method_t sample_method = EULER_A;
    schedule_t schedule           = DEFAULT;
    int sample_steps              = 20;
    float control_strength        = 0.75f;
    float style_strength          = 0.0f;
    rng_type_t rng_type           = CUDA_RNG;
    int64_t seed                  = 42;
    bool verbose                  = false;
    bool vae_tiling               = false;
    bool vae_decode_only          = false;
    bool normalize_input          = false;
    bool clip_on_cpu              = false;
    bool vae_on_cpu               = false;
    bool controlnet_on_cpu        = false;
    bool color                    = false;

    // Photomaker params
    std::string input_id_images_path;

    std::vector<int> skip_layers = {7, 8, 9};
    float slg_scale              = 0.;
    float skip_layer_start       = 0.01;
    float skip_layer_end         = 0.2;
};

inline void parse_args(int argc, const char** argv, std::string& server_host, unsigned int& port, std::string& models_path, std::string& vaes_path, std::string& embeddings_path, std::string& loras_path, std::string& controlnet_path, std::string& esrgan_path, std::string& config_file) {
    bool invalid_arg = false;
    std::string arg;

    std::string usage = argv[0];
    usage.append("\n");
    usage.append(" --models-path <path> \t path to the model files\n");
    usage.append(" --vaes-path <path> \t path to the vae models\n");
    usage.append(" --embeddings-path <path> \t path to the embedding models\n");
    usage.append(" --loras-path <path> \t path to the lora models\n");
    usage.append(" --controlnet-path <path> \t path to the controlnet models\n");
    usage.append(" --esrgan-path <path> \t path to the upscaler models\n");
    usage.append(" --listen <host/ip>\t listen address (default: 127.0.0.1)\n");
    usage.append(" --port <port> \t the listen port (default: 8080)\n");
    usage.append(" --config_file <path_to_json> \t a json config file for emulate sdwebui config and store some settings (default: ./config.json)\n");

    for (int i = 1; i < argc; i++) {
        arg = argv[i];

        if (arg == "-h" || arg == "--help") {
            printf("%s", usage.c_str());
            exit(0);
        } else if (arg == "--models-path") {
            if (++i >= argc) {
                invalid_arg = true;
                break;
            }
            models_path = argv[i];
            if (std::filesystem::exists(models_path) == false) {
                std::cerr << "Models path not found: " << models_path.c_str() << std::endl;
                exit(EXIT_FAILURE);
            }
        } else if (arg == "--vaes-path") {
            if (++i >= argc) {
                invalid_arg = true;
                break;
            }
            vaes_path = argv[i];
        } else if (arg == "--embeddings-path") {
            if (++i >= argc) {
                invalid_arg = true;
                break;
            }
            embeddings_path = argv[i];
        } else if (arg == "--loras-path") {
            if (++i >= argc) {
                invalid_arg = true;
                break;
            }
            loras_path = argv[i];
        } else if (arg == "--controlnet-path") {
            if (++i >= argc) {
                invalid_arg = true;
                break;
            }
            controlnet_path = argv[i];
        } else if (arg == "--esrgan-path") {
            if (++i >= argc) {
                invalid_arg = true;
                break;
            }
            esrgan_path = argv[i];
        } else if (arg == "--config-file") {
            if (++i >= argc) {
                invalid_arg = true;
                break;
            }
            config_file = argv[i];

        } else if (arg == "--address") {
            if (++i >= argc) {
                invalid_arg = true;
                break;
            }
            server_host = argv[i];
        } else if (arg == "--port") {
            if (++i >= argc) {
                invalid_arg = true;
                break;
            }
            unsigned int _port = std::atoi(argv[i]);
            if (_port > 0) {
                port = _port;
            } else {
                std::cerr << "Invalid listen port" << std::endl;
            }
        } else {
            fprintf(stderr, "error: unknown argument: %s\n", arg.c_str());
            std::cout << usage << std::endl;
            exit(EXIT_FAILURE);
        }
    }
    if (invalid_arg == true) {
        std::cout << usage << std::endl;
        exit(EXIT_FAILURE);
    }
    if (models_path.empty()) {
        std::cout << "--models-path is required" << std::endl;
        exit(EXIT_FAILURE);
    }
    if (vaes_path.empty()) {
        std::cout << "--vaes-path is required" << std::endl;
        exit(EXIT_FAILURE);
    }
    if (embeddings_path.empty()) {
        std::cout << "--embeddings-path is required" << std::endl;
        exit(EXIT_FAILURE);
    }
    if (loras_path.empty()) {
        std::cout << "--loras-path is required" << std::endl;
        exit(EXIT_FAILURE);
    }
    if (controlnet_path.empty()) {
        std::cout << "--controlnet-path is required" << std::endl;
        exit(EXIT_FAILURE);
    }
    if (esrgan_path.empty()) {
        std::cout << "--esrgan-path is required" << std::endl;
        exit(EXIT_FAILURE);
    }
    if (server_host.empty()) {
        server_host = "127.0.0.1";
    }
    if (port == 0) {
        port = 8080;
    }
}

inline static std::string sd_basename(const std::string& path) {
    size_t pos = path.find_last_of('/');
    if (pos != std::string::npos) {
        return path.substr(pos + 1);
    }
    pos = path.find_last_of('\\');
    if (pos != std::string::npos) {
        return path.substr(pos + 1);
    }
    return path;
}

inline std::string get_image_params(SDParams params, int64_t seed) {
    std::string parameter_string = params.prompt + "\n";
    if (params.negative_prompt.size() != 0) {
        parameter_string += "Negative prompt: " + params.negative_prompt + "\n";
    }
    parameter_string += "Steps: " + std::to_string(params.sample_steps) + ", ";
    parameter_string += "CFG scale: " + std::to_string(params.cfg_scale) + ", ";
    if (params.slg_scale != 0 && params.skip_layers.size() != 0) {
        parameter_string += "SLG scale: " + std::to_string(params.cfg_scale) + ", ";
        parameter_string += "Skip layers: [";
        for (const auto& layer : params.skip_layers) {
            parameter_string += std::to_string(layer) + ", ";
        }
        parameter_string += "], ";
        parameter_string += "Skip layer start: " + std::to_string(params.skip_layer_start) + ", ";
        parameter_string += "Skip layer end: " + std::to_string(params.skip_layer_end) + ", ";
    }
    parameter_string += "Guidance: " + std::to_string(params.guidance) + ", ";
    parameter_string += "Seed: " + std::to_string(seed) + ", ";
    parameter_string += "Size: " + std::to_string(params.width) + "x" + std::to_string(params.height) + ", ";
    parameter_string += "Model: " + sd_basename(params.model_path) + ", ";
    parameter_string += "RNG: " + std::string(rng_type_to_str[params.rng_type]) + ", ";
    parameter_string += "Sampler: " + std::string(sample_method_str[params.sample_method]);
    if (params.schedule == KARRAS) {
        parameter_string += " karras";
    }
    parameter_string += ", ";
    parameter_string += "Version: stable-diffusion.cpp";
    return parameter_string;
}
/* Enables Printing the log level tag in color using ANSI escape codes */
inline void sd_log_cb(enum sd_log_level_t level, const char* log, void* data) {
    int tag_color;
    const char* level_str;
    FILE* out_stream = (level == SD_LOG_ERROR) ? stderr : stdout;

    if (!log || (level <= SD_LOG_DEBUG)) {
        return;
    }

    switch (level) {
        case SD_LOG_DEBUG:
            tag_color = 37;
            level_str = "DEBUG";
            break;
        case SD_LOG_INFO:
            tag_color = 34;
            level_str = "INFO";
            break;
        case SD_LOG_WARN:
            tag_color = 35;
            level_str = "WARN";
            break;
        case SD_LOG_ERROR:
            tag_color = 31;
            level_str = "ERROR";
            break;
        default: /* Potential future-proofing */
            tag_color = 33;
            level_str = "?????";
            break;
    }

    fprintf(out_stream, "[%-5s] ", level_str);

    fputs(log, out_stream);
    fflush(out_stream);
}

inline void sd_log(enum sd_log_level_t level, const char* format, ...) {
    va_list args;
    va_start(args, format);

    char log[LOG_BUFFER_SIZE];
    vsnprintf(log, 1024, format, args);
    strncat(log, "\n", LOG_BUFFER_SIZE - strlen(log));

    sd_log_cb(level, log, server_log_params);
    va_end(args);
}

inline static void log_server_request(const httplib::Request& req, const httplib::Response& res) {
    printf("request: %s %s (%s)\n", req.method.c_str(), req.path.c_str(), req.body.c_str());
}

inline static void normalizePath(std::string& path) {
    std::filesystem::path abs_path       = std::filesystem::absolute(path);
    std::filesystem::path canonical_path = std::filesystem::weakly_canonical(abs_path);
    path                                 = canonical_path.string();
}

inline static long currentUnixTimestamp() {
    auto now = std::chrono::system_clock::now();
    return std::chrono::duration_cast<std::chrono::seconds>(now.time_since_epoch()).count();
}

#endif
