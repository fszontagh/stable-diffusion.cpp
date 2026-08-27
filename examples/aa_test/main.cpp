// sd-aa-test: minimal, standalone diagnostic tool for the AnimateAnyone model
// family registration. It does NOT build a full sd_ctx_t - it only exercises
// ModelLoader::init_from_file() + ModelLoader::get_sd_version() and applies
// the version-promotion rule that new_sd_ctx() applies at init time
// (see stable-diffusion.cpp: reference_net_path non-empty promotes a
// SD1-signature diffusion model to VERSION_ANIMATE_ANYONE).

#include <cstdio>
#include <cstring>
#include <string>

#include "model.h"
#include "model_loader.h"

extern const char* model_version_to_str[];

static void print_usage() {
    fprintf(stderr,
            "usage: sd-aa-test <mode> [args]\n"
            "\n"
            "modes:\n"
            "  version --diffusion-model <path> [--reference-net <path>]\n"
            "      Loads the diffusion model file, detects its SDVersion, applies the\n"
            "      reference_net_path promotion rule, and prints the resulting version\n"
            "      display name. Exits 0 on success, 1 on failure.\n");
}

static int run_version_mode(int argc, char** argv) {
    std::string diffusion_model_path;
    std::string reference_net_path;

    for (int i = 0; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--diffusion-model") {
            if (i + 1 >= argc) {
                fprintf(stderr, "error: --diffusion-model requires a value\n");
                return 1;
            }
            diffusion_model_path = argv[++i];
        } else if (arg == "--reference-net") {
            if (i + 1 >= argc) {
                fprintf(stderr, "error: --reference-net requires a value\n");
                return 1;
            }
            reference_net_path = argv[++i];
        } else {
            fprintf(stderr, "error: unknown argument '%s'\n", arg.c_str());
            return 1;
        }
    }

    if (diffusion_model_path.empty()) {
        fprintf(stderr, "error: --diffusion-model is required\n");
        return 1;
    }

    ModelLoader model_loader;
    if (!model_loader.init_from_file(diffusion_model_path, "model.diffusion_model.")) {
        fprintf(stderr, "error: failed to load diffusion model from '%s'\n", diffusion_model_path.c_str());
        return 1;
    }

    SDVersion version = model_loader.get_sd_version();
    if (version == VERSION_COUNT) {
        fprintf(stderr, "error: could not detect SD version from '%s'\n", diffusion_model_path.c_str());
        return 1;
    }
    printf("detected version: %s\n", model_version_to_str[version]);

    // Manual promotion rule mirroring new_sd_ctx()'s init() flow: a
    // SD1-signature diffusion model plus a non-empty reference_net_path
    // promotes to VERSION_ANIMATE_ANYONE.
    if (sd_version_is_sd1(version) && !sd_version_is_animate_anyone(version) && !reference_net_path.empty()) {
        version = VERSION_ANIMATE_ANYONE;
        printf("reference_net_path set, promoting detected version to %s\n", model_version_to_str[version]);
    }

    printf("%s\n", model_version_to_str[version]);
    return 0;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        print_usage();
        return 1;
    }

    std::string mode = argv[1];
    if (mode == "version") {
        return run_version_mode(argc - 2, argv + 2);
    } else if (mode == "-h" || mode == "--help") {
        print_usage();
        return 0;
    } else {
        fprintf(stderr, "error: unknown mode '%s'\n", mode.c_str());
        print_usage();
        return 1;
    }
}
