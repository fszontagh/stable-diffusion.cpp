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
#include "stable-diffusion.h"

extern const char* model_version_to_str[];

// Route the library's LOG_* calls (including get_sd_version()'s fallback-detection
// LOG_WARN) to stderr. Without a registered callback, log_printf() is a silent no-op,
// which would hide detection warnings from this diagnostic tool's own output.
static void aa_test_log_cb(enum sd_log_level_t level, const char* text, void* /*data*/) {
    fputs(text, stderr);
}

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

    // Shared promotion rule (src/model.h) also used by new_sd_ctx()'s init() flow: a
    // SD1-signature diffusion model plus a non-empty reference_net_path promotes to
    // VERSION_ANIMATE_ANYONE.
    SDVersion promoted = sd_apply_animate_anyone_promotion(version,
                                                            reference_net_path.empty() ? nullptr : reference_net_path.c_str());
    if (promoted != version) {
        version = promoted;
        printf("reference_net_path set, promoting detected version to %s\n", model_version_to_str[version]);
    }

    printf("%s\n", model_version_to_str[version]);
    return 0;
}

int main(int argc, char** argv) {
    sd_set_log_callback(aa_test_log_cb, nullptr);

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
