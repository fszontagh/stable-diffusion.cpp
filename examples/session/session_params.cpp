#include "session_params.h"
#include <exception>

bool apply_flags(const std::vector<std::string>& args,
                 SDCliParams& cli, SDContextParams& ctx, SDGenerationParams& gen,
                 std::string& err) {
    // parse_options expects an argv-style array with argv[0] as program name.
    std::vector<const char*> argv;
    argv.reserve(args.size() + 1);
    argv.push_back("sd-session");
    for (const auto& a : args) {
        argv.push_back(a.c_str());
    }

    // Operate on copies so a rejected line (bad flag value, failed
    // validation, or a thrown parse exception) never leaves the caller's
    // params partially mutated. Only commit on full success.
    SDCliParams cli_copy = cli;
    SDContextParams ctx_copy = ctx;
    SDGenerationParams gen_copy = gen;
    try {
        std::vector<ArgOptions> opts = {cli_copy.get_options(), ctx_copy.get_options(), gen_copy.get_options()};
        if (!parse_options((int)argv.size(), argv.data(), opts)) {
            err = "invalid or unknown argument";
            return false;
        }
        if (!cli_copy.resolve_and_validate()) {
            err = "cli params failed validation";
            return false;
        }
        if (!ctx_copy.resolve_and_validate(cli_copy.mode)) {
            err = "context params failed validation";
            return false;
        }
        if (!gen_copy.resolve_and_validate(cli_copy.mode, ctx_copy.lora_model_dir, ctx_copy.hires_upscalers_dir)) {
            err = "generation params failed validation";
            return false;
        }
    } catch (const std::exception& e) {
        err = e.what();
        return false;
    }

    cli = cli_copy;
    ctx = ctx_copy;
    gen = gen_copy;
    return true;
}
