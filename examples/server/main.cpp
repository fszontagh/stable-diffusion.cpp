#include <stdio.h>
#include <filesystem>
#include <memory>
#include <string>

#include "httplib.h"
#include "sd.h"
#include "config.h"

static std::string models_path = "./";  // sd-models endpoint
static std::string vaes_path;           // sd-modules endpoint
static std::string embeddings_path;     // sd-embeddings endpoint
static std::string loras_path;          // sd-loras endpoint
static std::string controlnet_path;
static std::string esrgan_path;         // upscalers

static std::string output_path = "./output";  // where the images are stored

static std::string config_file = "./config.json";



#include "server/http_callbacks.h"

// https://github.com/comfyanonymous/ComfyUI/blob/master/comfy/latent_formats.py#L152-L169

int main(int argc, const char* argv[]) {
    unsigned int port = 8080;
    std::string host  = "127.0.0.1";

    std::string api_endpoint_prefix = "/sdapi/v1/";

    parse_args(argc, argv, host, port, models_path, vaes_path, embeddings_path, loras_path, controlnet_path, esrgan_path, config_file);

    // normalize paths
    normalizePath(models_path);
    normalizePath(vaes_path);
    normalizePath(embeddings_path);
    normalizePath(loras_path);
    normalizePath(controlnet_path);
    normalizePath(esrgan_path);
    normalizePath(config_file);

    std::shared_ptr<Config> config = std::make_shared<Config>(config_file);

    sd_set_log_callback(sd_log_cb, nullptr);

    std::unique_ptr<httplib::Server> svr;
    svr.reset(new httplib::Server());
    svr->set_default_headers({{"Server", "sd.cpp"}});

    svr->Options(R"(.*)", [](const httplib::Request&, httplib::Response& res) {
        res.set_header("Access-Control-Allow-Credentials", "true");
        res.set_header("Access-Control-Allow-Methods", "POST, GET");
        res.set_header("Access-Control-Allow-Headers", "*");
        return res.set_content("", "application/json");
    });
    svr->set_logger(log_server_request);

    svr->set_mount_point("/files", output_path);

    http::Callbacks callbackHandler(config, models_path, vaes_path, embeddings_path, loras_path, controlnet_path, esrgan_path);

    for (const auto clb : callbackHandler.getCallbacks()) {
        if (clb.method == "GET") {
            svr->Get((clb.ignore_prefix == false ? api_endpoint_prefix : "") + clb.path, clb.callback);
            std::cout << "Registered (" << clb.method << ") endpoint: http://" << host << ":" << port << (clb.ignore_prefix == false ? api_endpoint_prefix : "") + clb.path << std::endl;
        }
        if (clb.method == "POST") {
            svr->Post((clb.ignore_prefix == false ? api_endpoint_prefix : "") + clb.path, clb.callback);
            std::cout << "Registered (" << clb.method << ") endpoint: http://" << host << ":" << port << (clb.ignore_prefix == false ? api_endpoint_prefix : "") + clb.path << std::endl;
        }
    }

    svr->Get("/internal/ping", [](const httplib::Request& req, httplib::Response& res) {
        res.set_content("{}", "application/json");
    });

    svr->Get("/internal/progress", [](const httplib::Request& req, httplib::Response& res) {
        res.status = httplib::StatusCode::NotImplemented_501;
        res.set_content("{}", "application/json");
    });

    // bind HTTP listen port, run the HTTP server in a thread
    if (!svr->bind_to_port(host, port)) {
        fprintf(stderr, "error: can not listen: %s:%d\n", host.c_str(), port);
        return 1;
    }
    printf("Server listening at %s:%d\n", host.c_str(), port);
    std::thread t([&]() { svr->listen_after_bind(); });
    svr->wait_until_ready();

    t.join();

    return 0;
}