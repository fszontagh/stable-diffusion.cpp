// examples/server/http_callbacks.h

#ifndef EXAMPLES_SERVER_HTTP_CALLBACKS_H
#define EXAMPLES_SERVER_HTTP_CALLBACKS_H

#include <exception>
#include <functional>
#include <memory>
#include <string>
#include <vector>
#include "httplib.h"
#include "json.hpp"
#include "sd.h"
#include "server/SDModelScanner.h"
#include "server/config.h"
#include "server/http_json_requests.h"
#include "server/http_json_responses.h"
#include "stable-diffusion.h"

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_STATIC
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#define STB_IMAGE_WRITE_STATIC
#include "stb_image_write.h"

#define STB_IMAGE_RESIZE_IMPLEMENTATION
#define STB_IMAGE_RESIZE_STATIC
#include "stb_image_resize.h"

namespace http {

    class Callbacks {
    public:
        Callbacks(
            std::shared_ptr<Config> config,
            const std::string& model_path,
            const std::string& vae_path,
            const std::string& embeddings_path,
            const std::string& lora_path,
            const std::string& controlnet_path,
            const std::string& esrgan_path)
            : config(config), models_path_(model_path), vaes_path_(vae_path), embeddings_path_(embeddings_path), loras_path_(lora_path), controlnet_path_(controlnet_path), esrgan_path_(esrgan_path) {
            // prepare callbacks
            this->addCallback("GET", "samplers", std::bind(&Callbacks::getSamplers, this, std::placeholders::_1, std::placeholders::_2));
            this->addCallback("GET", "schedulers", std::bind(&Callbacks::getSchedulers, this, std::placeholders::_1, std::placeholders::_2));
            this->addCallback("GET", "sd-models", std::bind(&Callbacks::getSdModels, this, std::placeholders::_1, std::placeholders::_2));
            this->addCallback("GET", "sd-modules", std::bind(&Callbacks::getSdModules, this, std::placeholders::_1, std::placeholders::_2));
            this->addCallback("GET", "upscalers", std::bind(&Callbacks::getUpscalers, this, std::placeholders::_1, std::placeholders::_2));
            this->addCallback("GET", "loras", std::bind(&Callbacks::getLoras, this, std::placeholders::_1, std::placeholders::_2));
            this->addCallback("GET", "embeddings", std::bind(&Callbacks::getEmbeddings, this, std::placeholders::_1, std::placeholders::_2));
            this->addCallback("POST", "refresh-vae", std::bind(&Callbacks::postRefreshVae, this, std::placeholders::_1, std::placeholders::_2));
            this->addCallback("POST", "refresh-embeddings", std::bind(&Callbacks::postRefreshEmbeddings, this, std::placeholders::_1, std::placeholders::_2));
            this->addCallback("POST", "refresh-checkpoints", std::bind(&Callbacks::postRefreshCheckpoints, this, std::placeholders::_1, std::placeholders::_2));
            this->addCallback("POST", "refresh-loras", std::bind(&Callbacks::postRefreshLoras, this, std::placeholders::_1, std::placeholders::_2));
            this->addCallback("POST", "unload-checkpoint", std::bind(&Callbacks::postUnloadCheckpoint, this, std::placeholders::_1, std::placeholders::_2));
            this->addCallback("POST", "reload-checkpoint", std::bind(&Callbacks::postReloadCheckpoint, this, std::placeholders::_1, std::placeholders::_2));
            this->addCallback("GET", "progress", std::bind(&Callbacks::getProgress, this, std::placeholders::_1, std::placeholders::_2));
            this->addCallback("GET", "interrupt", std::bind(&Callbacks::getInterrupt, this, std::placeholders::_1, std::placeholders::_2));
            this->addCallback("POST", "text2img", std::bind(&Callbacks::postText2Img, this, std::placeholders::_1, std::placeholders::_2));
            this->addCallback("POST", "img2img", std::bind(&Callbacks::postImg2Img, this, std::placeholders::_1, std::placeholders::_2));
            this->addCallback("GET", "/controlnet/model_list", std::bind(&Callbacks::getCNmodelList, this, std::placeholders::_1, std::placeholders::_2), true);
            this->addCallback("GET", "files", std::bind(&Callbacks::getFiles, this, std::placeholders::_1, std::placeholders::_2), true);

            this->addCallback("GET", "options", std::bind(&Callbacks::getOptions, this, std::placeholders::_1, std::placeholders::_2));
            this->addCallback("POST", "options", std::bind(&Callbacks::postOptions, this, std::placeholders::_1, std::placeholders::_2));

            this->scanner = std::make_shared<SDModelScanner>(models_path_, vaes_path_, embeddings_path_, loras_path_, controlnet_path_, esrgan_path_);
        }

        ~Callbacks() {
            if (this->sd_ctx_) {
                free_sd_ctx(this->sd_ctx_);
                this->sd_ctx_ = NULL;
            }

            if (this->upscaler_ctx_) {
                free_upscaler_ctx(this->upscaler_ctx_);
                this->upscaler_ctx_ = NULL;
            }
        }
        using Callback = std::function<void(const httplib::Request&, httplib::Response&)>;

        struct CallbackEntry {
            std::string method;
            std::string path;
            Callback callback;
            bool ignore_prefix = false;
        };

        const std::vector<CallbackEntry>& getCallbacks() const {
            return callbacks_;
        }

        void getOptions(const httplib::Request& req, httplib::Response& res) {
            nlohmann::json response = this->config->getSettings();
            res.status              = httplib::StatusCode::OK_200;
            res.set_content(response.dump(this->dump_indent_), this->mime_type_);
        }
        void postOptions(const httplib::Request& req, httplib::Response& res) {
            nlohmann::json response = nlohmann::json::object();
            if (req.body.empty()) {
                res.status        = httplib::StatusCode::BadRequest_400;
                response["error"] = "Body is empty";
                res.set_content(response.dump(this->dump_indent_), this->mime_type_);
                return;
            }
            try {
                nlohmann::json body = nlohmann::json::parse(req.body);
                const auto cfg      = body.get<SdWebuiSettings>();
                this->config->setSettings(body);
                res.status          = httplib::StatusCode::OK_200;
                response["message"] = "Settings updated";
                res.set_content(response.dump(this->dump_indent_), this->mime_type_);
            } catch (const std::exception& e) {
                res.status        = httplib::StatusCode::BadRequest_400;
                response["error"] = e.what();
                res.set_content(response.dump(this->dump_indent_), this->mime_type_);
            }
        }

        void getSamplers(const httplib::Request& req, httplib::Response& res) {
            nlohmann::json response = sample_method_info;
            res.status              = httplib::StatusCode::OK_200;
            res.set_content(response.dump(this->dump_indent_), this->mime_type_);
        }

        void getSchedulers(const httplib::Request& req, httplib::Response& res) {
            nlohmann::json response = schedulers_info;
            res.status              = httplib::StatusCode::OK_200;
            res.set_content(response.dump(this->dump_indent_), this->mime_type_);
        }

        void getSdModels(const httplib::Request& req, httplib::Response& res) {
            nlohmann::json response = this->scanner->getSdModels();
            res.status              = httplib::StatusCode::OK_200;
            res.set_content(response.dump(this->dump_indent_), this->mime_type_);
        }

        void getSdModules(const httplib::Request& req, httplib::Response& res) {
            nlohmann::json response = this->scanner->getVaes();
            res.status              = httplib::StatusCode::OK_200;
            res.set_content(response.dump(this->dump_indent_), this->mime_type_);
        }

        void getUpscalers(const httplib::Request& req, httplib::Response& res) {
            nlohmann::json response = nlohmann::json::object();
            response["message"]     = "Not impmlemented";
            res.status              = httplib::StatusCode::NotImplemented_501;
            res.set_content(response.dump(this->dump_indent_), this->mime_type_);
        }

        void getLoras(const httplib::Request& req, httplib::Response& res) {
            nlohmann::json response = this->scanner->getLoras();
            res.status              = httplib::StatusCode::OK_200;
            res.set_content(response.dump(this->dump_indent_), this->mime_type_);
        }

        void getCNmodelList(const httplib::Request& req, httplib::Response& res) {
            nlohmann::json response = this->scanner->getControlnets();
            res.status              = httplib::StatusCode::OK_200;
            res.set_content(response.dump(this->dump_indent_), this->mime_type_);
        }

        void getEmbeddings(const httplib::Request& req, httplib::Response& res) {
            nlohmann::json response = this->scanner->getEmbeddings();
            res.status              = httplib::StatusCode::OK_200;
            res.set_content(response.dump(this->dump_indent_), this->mime_type_);
        }

        void postRefreshVae(const httplib::Request& req, httplib::Response& res) {
            nlohmann::json response = nlohmann::json::object();
            response["message"]     = "Not impmlemented";
            res.status              = httplib::StatusCode::NotImplemented_501;
            res.set_content(response.dump(this->dump_indent_), this->mime_type_);
        }

        void postRefreshEmbeddings(const httplib::Request& req, httplib::Response& res) {
            nlohmann::json response = nlohmann::json::object();
            response["message"]     = "Not impmlemented";
            res.status              = httplib::StatusCode::NotImplemented_501;
            res.set_content(response.dump(this->dump_indent_), this->mime_type_);
        }

        void postRefreshCheckpoints(const httplib::Request& req, httplib::Response& res) {
            this->scanner->reScanSdModels();
            nlohmann::json response = nlohmann::json::object();
            res.status              = httplib::StatusCode::OK_200;
            res.set_content(response.dump(this->dump_indent_), this->mime_type_);
        }

        void postRefreshLoras(const httplib::Request& req, httplib::Response& res) {
            this->scanner->reScanLoras();
            nlohmann::json response = nlohmann::json::object();
            res.status              = httplib::StatusCode::OK_200;
            res.set_content(response.dump(this->dump_indent_), this->mime_type_);
        }

        void postUnloadCheckpoint(const httplib::Request& req, httplib::Response& res) {
            nlohmann::json response = nlohmann::json::object();
            if (this->is_running == true) {
                response["error"] = "Aleady running";
                res.status        = httplib::StatusCode::BadRequest_400;
                res.set_content(response.dump(this->dump_indent_), this->mime_type_);
                return;
            }
            if (this->sd_ctx_) {
                free_sd_ctx(this->sd_ctx_);
                this->sd_ctx_ = NULL;
            }
            res.status = httplib::StatusCode::OK_200;
            res.set_content(response.dump(this->dump_indent_), this->mime_type_);
        }

        void postReloadCheckpoint(const httplib::Request& req, httplib::Response& res) {
            nlohmann::json response = nlohmann::json::object();

            if (this->is_running == true) {
                response["error"] = "Aleady running";
                res.status        = httplib::StatusCode::BadRequest_400;
                res.set_content(response.dump(this->dump_indent_), this->mime_type_);
                return;
            }
            if (this->sd_ctx_) {
                free_sd_ctx(this->sd_ctx_);
                this->sd_ctx_ = NULL;
            }

            // if (this->sd_params_.model_path.empty() && this->sd_params_.diffusion_model_path.empty()) {
            //     res.status        = httplib::StatusCode::BadRequest_400;
            //     response["error"] = "No model loaded";
            //     res.set_content(response.dump(this->dump_indent_), this->mime_type_);
            //     return;
            // }
            if (req.body.empty()) {
                res.status        = httplib::StatusCode::BadRequest_400;
                response["error"] = "Empty body";
                res.set_content(response.dump(this->dump_indent_), this->mime_type_);
                return;
            }
            try {
                nlohmann::json j                       = nlohmann::json::parse(req.body);
                http_jsonrequest::sd_model_load_t data = j.get<http_jsonrequest::sd_model_load_t>();
                if (data.model.empty() && data.diffusion_model.empty()) {
                    res.status        = httplib::StatusCode::BadRequest_400;
                    response["error"] = "Empty model name";
                    res.set_content(response.dump(this->dump_indent_), this->mime_type_);
                    return;
                }
                if (data.model.empty() == false && data.diffusion_model.empty() == false) {
                    res.status        = httplib::StatusCode::BadRequest_400;
                    response["error"] = "Only one model type allowed at once";
                    res.set_content(response.dump(this->dump_indent_), this->mime_type_);
                    return;
                }
                // merge it up
                this->sd_params_ << data;

                if (this->sd_params_.model_path.empty() == false) {
                    const auto found_path = this->scanner->findModelPath(this->sd_params_.model_path);
                    if (found_path.empty()) {
                        res.status        = httplib::StatusCode::BadRequest_400;
                        response["error"] = "Model not found: " + this->sd_params_.model_path;
                        res.set_content(response.dump(this->dump_indent_), this->mime_type_);
                        return;
                    }
                    this->sd_params_.model_path = found_path;
                }

                if (this->sd_params_.diffusion_model_path.empty() == false) {
                    const auto found_path = this->scanner->findModelPath(this->sd_params_.diffusion_model_path);
                    if (found_path.empty()) {
                        res.status        = httplib::StatusCode::BadRequest_400;
                        response["error"] = "Model not found: " + this->sd_params_.diffusion_model_path;
                        res.set_content(response.dump(this->dump_indent_), this->mime_type_);
                        return;
                    }
                    this->sd_params_.diffusion_model_path = found_path;
                }

                if (this->sd_params_.vae_path.empty() == false) {
                    const auto found_path = this->scanner->findModelPath(this->sd_params_.vae_path);
                    if (found_path.empty()) {
                        res.status        = httplib::StatusCode::BadRequest_400;
                        response["error"] = "Model not found: " + this->sd_params_.vae_path;
                        res.set_content(response.dump(this->dump_indent_), this->mime_type_);
                        return;
                    }
                    this->sd_params_.vae_path = found_path;
                }
                if (this->sd_params_.t5xxl_path.empty() == false) {
                    const auto found_path = this->scanner->findModelPath(this->sd_params_.t5xxl_path);
                    if (found_path.empty()) {
                        res.status        = httplib::StatusCode::BadRequest_400;
                        response["error"] = "Model not found: " + this->sd_params_.t5xxl_path;
                        res.set_content(response.dump(this->dump_indent_), this->mime_type_);
                        return;
                    }
                    this->sd_params_.t5xxl_path = found_path;
                }
                if (this->sd_params_.clip_g_path.empty() == false) {
                    const auto found_path = this->scanner->findModelPath(this->sd_params_.clip_g_path);
                    if (found_path.empty()) {
                        res.status        = httplib::StatusCode::BadRequest_400;
                        response["error"] = "Model not found: " + this->sd_params_.clip_g_path;
                        res.set_content(response.dump(this->dump_indent_), this->mime_type_);
                        return;
                    }
                    this->sd_params_.clip_g_path = found_path;
                }
                if (this->sd_params_.clip_l_path.empty() == false) {
                    const auto found_path = this->scanner->findModelPath(this->sd_params_.clip_l_path);
                    if (found_path.empty()) {
                        res.status        = httplib::StatusCode::BadRequest_400;
                        response["error"] = "Model not found: " + this->sd_params_.clip_l_path;
                        res.set_content(response.dump(this->dump_indent_), this->mime_type_);
                        return;
                    }
                    this->sd_params_.clip_l_path = found_path;
                }

                this->loadSdCtx();
            } catch (std::exception& e) {
                res.status        = httplib::StatusCode::BadRequest_400;
                response["error"] = e.what();
                res.set_content(response.dump(this->dump_indent_), this->mime_type_);
            }

            res.status = httplib::StatusCode::OK_200;
            res.set_content(response.dump(this->dump_indent_), this->mime_type_);
        }

        void getProgress(const httplib::Request& req, httplib::Response& res) {
            nlohmann::json response = nlohmann::json::object();
            response["message"]     = "Not impmlemented";
            res.status              = httplib::StatusCode::NotImplemented_501;
            res.set_content(response.dump(this->dump_indent_), this->mime_type_);
        }

        void getInterrupt(const httplib::Request& req, httplib::Response& res) {
            nlohmann::json response = nlohmann::json::object();
            response["message"]     = "Not impmlemented";
            res.status              = httplib::StatusCode::NotImplemented_501;
            res.set_content(response.dump(this->dump_indent_), this->mime_type_);
        }

        void postText2Img(const httplib::Request& req, httplib::Response& res) {
            nlohmann::json response = nlohmann::json::object();

            if (this->is_running == true) {
                response["error"] = "Aleady running";
                res.status        = httplib::StatusCode::BadRequest_400;
                res.set_content(response.dump(this->dump_indent_), this->mime_type_);
                return;
            }

            if (req.body.empty()) {
                response["error"] = "No body";
                res.status        = httplib::StatusCode::BadRequest_400;
                res.set_content(response.dump(this->dump_indent_), this->mime_type_);
                return;
            }
            if (this->sd_ctx_ == NULL) {
                response["error"] = "No model loaded";
                res.status        = httplib::StatusCode::BadRequest_400;
                res.set_content(response.dump(this->dump_indent_), this->mime_type_);
                return;
            }
            this->is_running = true;

            try {
                nlohmann::json js                             = nlohmann::json::parse(req.body);
                http_jsonrequest::sdWebuiParamstxt2img params = js.get<http_jsonrequest::sdWebuiParamstxt2img>();

                sd_image_t* control_cond = NULL;

                this->sd_params_ << params;
                sd_image_t* results = txt2img(
                    this->sd_ctx_,
                    this->sd_params_.prompt.c_str(),
                    this->sd_params_.negative_prompt.c_str(),
                    this->sd_params_.clip_skip,
                    this->sd_params_.cfg_scale,
                    this->sd_params_.guidance,
                    this->sd_params_.width,
                    this->sd_params_.height,
                    this->sd_params_.sample_method,
                    this->sd_params_.sample_steps,
                    this->sd_params_.seed,
                    this->sd_params_.batch_count,
                    control_cond,
                    this->sd_params_.control_strength,
                    this->sd_params_.style_strength,
                    this->sd_params_.normalize_input,
                    this->sd_params_.input_id_images_path.c_str());

                if (results == NULL) {
                    response["error"] = "Empty results";
                    res.status        = httplib::StatusCode::InternalServerError_500;
                    res.set_content(response.dump(this->dump_indent_), this->mime_type_);
                    this->is_running = false;
                    return;
                }
                std::vector<sd_generated_image_t> images = {};
                for (int i = 0; i < this->sd_params_.batch_count; i++) {
                    if (results[i].data == NULL) {
                        continue;
                    }
                    sd_generated_image_t img(results[i]);

                    std::string filename = "txt2img_" + (this->sd_params_.seed > 0 ? std::to_string(this->sd_params_.seed + i) : std::to_string(i + 1));
                    filename.append("_" + std::to_string(currentUnixTimestamp()));
                    filename.append(".png");
                    img.filename = filename;
                    filename     = this->output_path_ + std::filesystem::path::preferred_separator + filename;
                    stbi_write_png(filename.c_str(), results[i].width, results[i].height, results[i].channel, results[i].data, 0, get_image_params(this->sd_params_, this->sd_params_.seed + i).c_str());
                    free(results[i].data);
                    results[i].data = NULL;
                    img.output_path = filename;
                    img.url         = "http://" + req.local_addr + ":" + std::to_string(req.local_port) + "/files/" + img.filename;
                    images.emplace_back(img);
                }

                free(results);
                response["images"] = images;
                res.set_content(response.dump(this->dump_indent_), this->mime_type_);
                this->is_running = false;
                return;

            } catch (std::exception& e) {
                response["error"] = e.what();
                res.status        = httplib::StatusCode::BadRequest_400;
                res.set_content(response.dump(this->dump_indent_), this->mime_type_);
                this->is_running = false;
                return;
            }

            res.set_content(response.dump(this->dump_indent_), this->mime_type_);
            this->is_running = false;
        }

        void postImg2Img(const httplib::Request& req, httplib::Response& res) {
            nlohmann::json response = nlohmann::json::object();
            response["message"]     = "Not impmlemented";
            res.status              = httplib::StatusCode::NotImplemented_501;
            res.set_content(response.dump(this->dump_indent_), this->mime_type_);
        };

        void getFiles(const httplib::Request& req, httplib::Response& res) {
            std::vector<std::string> images;
            for (const auto& entry : std::filesystem::directory_iterator(this->output_path_)) {
                if (entry.is_regular_file() && entry.path().extension() == ".png") {
                    images.push_back(entry.path().filename().string());
                }
            }
            nlohmann::json response = nlohmann::json::array();
            for (const auto& img : images) {
                response.push_back("http://" + req.local_addr + ":" + std::to_string(req.local_port) + "/files/" + img);
            }
            res.set_content(response.dump(this->dump_indent_), this->mime_type_);
        };

    private:
        sd_ctx_t* sd_ctx_             = NULL;
        upscaler_ctx_t* upscaler_ctx_ = NULL;
        SDParams sd_params_;

        std::vector<CallbackEntry> callbacks_;
        const std::string mime_type_ = "application/json";
        std::string models_path_, vaes_path_, embeddings_path_, loras_path_, controlnet_path_, esrgan_path_, output_path_;
        std::shared_ptr<SDModelScanner> scanner;
        std::shared_ptr<Config> config;
        std::atomic<bool> is_running = false;

#if defined(DEBUG) || defined(_DEBUG)
        int dump_indent_ = 4;
#else
        int dump_indent_ = -1;
#endif

        void addCallback(const std::string& method, const std::string& path, Callback callback, bool ignore_prefix = false) {
            callbacks_.emplace_back(CallbackEntry{method, path, callback, ignore_prefix});
        }
        void loadSdCtx() {
            this->sd_ctx_ = new_sd_ctx(
                this->sd_params_.model_path.c_str(),
                this->sd_params_.clip_l_path.c_str(),
                this->sd_params_.clip_g_path.c_str(),
                this->sd_params_.t5xxl_path.c_str(),
                this->sd_params_.diffusion_model_path.c_str(),
                this->sd_params_.vae_path.c_str(),
                this->sd_params_.taesd_path.c_str(),
                this->sd_params_.control_net_path.c_str(),
                this->loras_path_.c_str(),
                this->embeddings_path_.c_str(),
                this->sd_params_.stacked_id_embeddings_path.c_str(),
                this->sd_params_.vae_decode_only,
                this->sd_params_.vae_tiling,
                false,  // free_params_immediately
                this->sd_params_.n_threads,
                this->sd_params_.wtype,
                this->sd_params_.rng_type,
                this->sd_params_.schedule,
                this->sd_params_.clip_on_cpu,
                this->sd_params_.controlnet_on_cpu,
                this->sd_params_.vae_on_cpu);
        };
    };

}  // namespace http

#endif  // EXAMPLES_SERVER_HTTP_CALLBACKS_H