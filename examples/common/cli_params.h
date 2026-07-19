#pragma once
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <string>
#include <vector>

#include "common.h"
#include "stable-diffusion.h"

extern const char* const previews_str[];

struct SDCliParams {
    SDMode mode             = IMG_GEN;
    std::string output_path = "output.png";
    int output_begin_idx    = -1;
    std::string image_path;
    std::string metadata_format = "text";

    bool verbose          = false;
    bool canny_preprocess = false;
    bool convert_name     = false;

    preview_t preview_method = PREVIEW_NONE;
    int preview_interval     = 1;
    std::string preview_path = "preview.png";
    int preview_fps          = 16;
    bool taesd_preview       = false;
    bool preview_noisy       = false;
    bool color               = false;
    bool metadata_raw        = false;
    bool metadata_brief      = false;
    bool metadata_all        = false;

    std::string imatrix_out;
    std::vector<std::string> imatrix_in;

    bool normal_exit = false;

    ArgOptions get_options() {
        ArgOptions options;

        options.string_options = {
            {"-o",
             "--output",
             "path to write result image to. you can use printf-style %d format specifiers for image sequences (default: ./output.png) (eg. output_%03d.png). Single-file video outputs support .avi, .webm, and animated .webp",
             0,
             &output_path},
            {"",
             "--image",
             "path to the image to inspect (for metadata mode)",
             0,
             &image_path},
            {"",
             "--metadata-format",
             "metadata output format, one of [text, json] (default: text)",
             0,
             &metadata_format},
            {"",
             "--preview-path",
             "path to write preview image to (default: ./preview.png). Multi-frame previews support .avi, .webm, and animated .webp",
             0,
             &preview_path},
            {"",
             "--imat-out",
             "compute the imatrix for this run and save it to the provided path",
             0,
             &imatrix_out},
        };

        options.int_options = {
            {"",
             "--preview-interval",
             "interval in denoising steps between consecutive updates of the image preview file (default is 1, meaning updating at every step)",
             &preview_interval},
            {"",
             "--output-begin-idx",
             "starting index for output image sequence, must be non-negative (default 0 if specified %d in output path, 1 otherwise)",
             &output_begin_idx},
        };

        options.bool_options = {
            {"",
             "--canny",
             "apply canny preprocessor (edge detection)",
             true, &canny_preprocess},
            {"",
             "--convert-name",
             "convert tensor name (for convert mode)",
             true, &convert_name},
            {"-v",
             "--verbose",
             "print extra info",
             true, &verbose},
            {"",
             "--color",
             "colors the logging tags according to level",
             true, &color},
            {"",
             "--taesd-preview-only",
             std::string("prevents usage of taesd for decoding the final image. (for use with --preview ") + previews_str[PREVIEW_TAE] + ")",
             true, &taesd_preview},
            {"",
             "--preview-noisy",
             "enables previewing noisy inputs of the models rather than the denoised outputs",
             true, &preview_noisy},
            {"",
             "--metadata-raw",
             "include raw hex previews for unparsed metadata payloads",
             true, &metadata_raw},
            {"",
             "--metadata-brief",
             "truncate long metadata text values in text output",
             true, &metadata_brief},
            {"",
             "--metadata-all",
             "include structural/container entries such as IHDR, IDAT, and non-metadata JPEG segments",
             true, &metadata_all},

        };

        auto on_mode_arg = [&](int argc, const char** argv, int index) {
            if (++index >= argc) {
                return -1;
            }
            const char* mode_c_str = argv[index];
            if (mode_c_str != nullptr) {
                int mode_found = -1;
                for (int i = 0; i < MODE_COUNT; i++) {
                    if (!strcmp(mode_c_str, modes_str[i])) {
                        mode_found = i;
                    }
                }
                if (mode_found == -1) {
                    LOG_ERROR("error: invalid mode %s, must be one of [%s]\n",
                              mode_c_str, SD_ALL_MODES_STR);
                    exit(1);
                }
                mode = (SDMode)mode_found;
            }
            return 1;
        };

        auto on_preview_arg = [&](int argc, const char** argv, int index) {
            if (++index >= argc) {
                return -1;
            }
            const char* preview = argv[index];
            int preview_found   = -1;
            for (int m = 0; m < PREVIEW_COUNT; m++) {
                if (!strcmp(preview, previews_str[m])) {
                    preview_found = m;
                }
            }
            if (preview_found == -1) {
                LOG_ERROR("error: preview method %s", preview);
                return -1;
            }
            preview_method = (preview_t)preview_found;
            return 1;
        };

        auto on_help_arg = [&](int argc, const char** argv, int index, bool& valid) {
            normal_exit = true;
            valid       = true;
            return -1;
        };

        auto on_imatrix_in_arg = [&](int argc, const char** argv, int index) {
            if (++index >= argc) {
                return -1;
            }
            imatrix_in.push_back(argv[index]);
            return 1;
        };

        options.manual_options = {
            {"-M",
             "--mode",
             "run mode, one of [img_gen, vid_gen, upscale, convert, metadata], default: img_gen",
             on_mode_arg},
            {"",
             "--preview",
             std::string("preview method. must be one of the following [") + previews_str[0] + ", " + previews_str[1] + ", " + previews_str[2] + ", " + previews_str[3] + "] (default is " + previews_str[PREVIEW_NONE] + ")",
             on_preview_arg},
            {"-h",
             "--help",
             "show this help message and exit",
             on_help_arg},
            {"",
             "--imat-in",
             "load an imatrix file for quantization or continued collection; can be specified multiple times",
             on_imatrix_in_arg},
        };

        return options;
    };

    bool resolve() {
        if (mode == CONVERT) {
            if (output_path == "output.png") {
                output_path = "output.gguf";
            }
        }
        return true;
    }

    bool validate() {
        if (mode != METADATA) {
            if (output_path.length() == 0) {
                LOG_ERROR("error: the following arguments are required: output_path");
                return false;
            }
        } else {
            if (image_path.empty()) {
                LOG_ERROR("error: metadata mode needs an image path (--image)");
                return false;
            }
            if (metadata_format != "text" && metadata_format != "json") {
                LOG_ERROR("error: invalid metadata format %s, must be one of [text, json]",
                          metadata_format.c_str());
                return false;
            }
        }
        return true;
    }

    bool resolve_and_validate() {
        if (!resolve()) {
            return false;
        }
        if (!validate()) {
            return false;
        }
        return true;
    }

    std::string to_string() const {
        std::ostringstream oss;
        oss << "SDCliParams {\n"
            << "  mode: " << modes_str[mode] << ",\n"
            << "  output_path: \"" << output_path << "\",\n"
            << "  image_path: \"" << image_path << "\",\n"
            << "  metadata_format: \"" << metadata_format << "\",\n"
            << "  verbose: " << (verbose ? "true" : "false") << ",\n"
            << "  color: " << (color ? "true" : "false") << ",\n"
            << "  canny_preprocess: " << (canny_preprocess ? "true" : "false") << ",\n"
            << "  convert_name: " << (convert_name ? "true" : "false") << ",\n"
            << "  preview_method: " << previews_str[preview_method] << ",\n"
            << "  preview_interval: " << preview_interval << ",\n"
            << "  preview_path: \"" << preview_path << "\",\n"
            << "  preview_fps: " << preview_fps << ",\n"
            << "  taesd_preview: " << (taesd_preview ? "true" : "false") << ",\n"
            << "  preview_noisy: " << (preview_noisy ? "true" : "false") << ",\n"
            << "  imatrix_out: \"" << imatrix_out << "\",\n"
            << "  metadata_raw: " << (metadata_raw ? "true" : "false") << ",\n"
            << "  metadata_brief: " << (metadata_brief ? "true" : "false") << ",\n"
            << "  metadata_all: " << (metadata_all ? "true" : "false") << "\n"
            << "}";
        return oss.str();
    }
};
