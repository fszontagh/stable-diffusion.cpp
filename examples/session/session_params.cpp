#include "session_params.h"

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
    std::vector<ArgOptions> opts = {cli.get_options(), ctx.get_options(), gen.get_options()};
    if (!parse_options((int)argv.size(), argv.data(), opts)) {
        err = "invalid or unknown argument";
        return false;
    }
    if (!cli.resolve_and_validate()) {
        err = "cli params failed validation";
        return false;
    }
    if (!ctx.resolve_and_validate(cli.mode)) {
        err = "context params failed validation";
        return false;
    }
    if (!gen.resolve_and_validate(cli.mode, ctx.lora_model_dir, ctx.hires_upscalers_dir)) {
        err = "generation params failed validation";
        return false;
    }
    return true;
}
