#include "serialize.h"

#include <map>
#include <set>
#include <sstream>

static std::string quote_if_needed(const std::string& v) {
    if (v.find(' ') != std::string::npos) {
        return "\"" + v + "\"";
    }
    return v;
}

// Table-driven walk over an ArgOptions pair produced from the current and a
// default-constructed instance of the same struct. `overrides` lets a caller
// substitute the emitted value for a specific target pointer (used for the
// prompt, which needs LoRA tags re-embedded) without special-casing the walk
// itself.
//
// `seen_targets` dedups by target address across the whole serialization:
// several flags are deprecated aliases that register the same target pointer
// (e.g. --llm / --qwen2vl both bind &llm_path). get_options() lists the
// canonical flag before its alias, so first-wins keeps the canonical
// spelling and drops the alias.
static void emit(std::string& out,
                 const ArgOptions& cur,
                 const ArgOptions& def,
                 bool full,
                 std::set<const void*>& seen_targets,
                 const std::map<const std::string*, std::string>& overrides = {}) {
    for (size_t i = 0; i < cur.string_options.size(); ++i) {
        const auto& c = cur.string_options[i];
        const auto& d = def.string_options[i];
        if (c.long_name.empty() || !c.target) continue;
        if (!seen_targets.insert(c.target).second) continue;
        if (full || *c.target != *d.target) {
            auto it        = overrides.find(c.target);
            std::string val = (it != overrides.end()) ? it->second : *c.target;
            if (val.empty()) continue;
            out += " " + c.long_name + " " + quote_if_needed(val);
        }
    }
    for (size_t i = 0; i < cur.int_options.size(); ++i) {
        const auto& c = cur.int_options[i];
        const auto& d = def.int_options[i];
        if (c.long_name.empty() || !c.target) continue;
        if (!seen_targets.insert(c.target).second) continue;
        if (full || *c.target != *d.target) {
            out += " " + c.long_name + " " + std::to_string(*c.target);
        }
    }
    for (size_t i = 0; i < cur.float_options.size(); ++i) {
        const auto& c = cur.float_options[i];
        const auto& d = def.float_options[i];
        if (c.long_name.empty() || !c.target) continue;
        if (!seen_targets.insert(c.target).second) continue;
        if (full || *c.target != *d.target) {
            out += " " + c.long_name + " " + std::to_string(*c.target);
        }
    }
    for (size_t i = 0; i < cur.bool_options.size(); ++i) {
        const auto& c = cur.bool_options[i];
        const auto& d = def.bool_options[i];
        if (c.long_name.empty() || !c.target) continue;
        if (!seen_targets.insert(c.target).second) continue;
        if (full || *c.target != *d.target) {
            if (*c.target) out += " " + c.long_name;
        }
    }
}

static std::string lora_tag(const std::string& path, float mult, bool high_noise) {
    std::ostringstream oss;
    oss << "<lora:" << (high_noise ? "|high_noise|" : "") << path << ":" << mult << ">";
    return oss.str();
}

std::string serialize_to_cli(const SDCliParams& cli,
                             const SDContextParams& ctx,
                             const SDGenerationParams& gen,
                             bool full) {
    std::string out = "sd-cli";
    std::set<const void*> seen_targets;

    if (full || cli.mode != IMG_GEN) {
        out += " --mode " + std::string(modes_str[cli.mode]);
    }

    SDCliParams def_cli;
    ArgOptions cur_cli_opts = const_cast<SDCliParams&>(cli).get_options();
    ArgOptions def_cli_opts = def_cli.get_options();
    emit(out, cur_cli_opts, def_cli_opts, full, seen_targets);

    // ctx.resolve() normalizes n_threads from -1 to the physical core count
    // (via apply_flags -> resolve_and_validate) whenever the session was
    // ever `set`/`load`ed. Resolving the default struct the same way keeps
    // that normalization from leaking into the non-full diff as a spurious
    // "--threads N" the user never asked for.
    SDContextParams def_ctx;
    def_ctx.resolve(cli.mode);
    ArgOptions cur_ctx_opts = const_cast<SDContextParams&>(ctx).get_options();
    ArgOptions def_ctx_opts = def_ctx.get_options();
    emit(out, cur_ctx_opts, def_ctx_opts, full, seen_targets);

    SDGenerationParams def_gen;
    ArgOptions cur_gen_opts = const_cast<SDGenerationParams&>(gen).get_options();
    ArgOptions def_gen_opts = def_gen.get_options();

    // gen.prompt has already had `<lora:...>` tags stripped out into
    // lora_map/high_noise_lora_map by extract_and_remove_lora(); re-embed
    // them here so the emitted --prompt value round-trips through sd-cli's
    // own parser (there is no standalone --lora flag).
    std::string augmented_prompt = gen.prompt;
    for (const auto& kv : gen.lora_map) {
        augmented_prompt += " " + lora_tag(kv.first, kv.second, false);
    }
    for (const auto& kv : gen.high_noise_lora_map) {
        augmented_prompt += " " + lora_tag(kv.first, kv.second, true);
    }
    std::map<const std::string*, std::string> overrides;
    for (const auto& opt : cur_gen_opts.string_options) {
        if (opt.target == &gen.prompt) {
            overrides[opt.target] = augmented_prompt;
            break;
        }
    }
    emit(out, cur_gen_opts, def_gen_opts, full, seen_targets, overrides);

    // seed is a ManualOption (int64_t target, not covered by IntOption), so
    // it needs its own comparison against the default-constructed value.
    if (full || gen.seed != def_gen.seed) {
        out += " --seed " + std::to_string(gen.seed);
    }

    // custom_sigmas / ref_image_paths are also ManualOptions with no target
    // pointer; handle them explicitly rather than silently dropping them.
    if (full || gen.custom_sigmas != def_gen.custom_sigmas) {
        if (!gen.custom_sigmas.empty()) {
            std::ostringstream oss;
            for (size_t i = 0; i < gen.custom_sigmas.size(); ++i) {
                if (i) oss << ",";
                oss << gen.custom_sigmas[i];
            }
            out += " --sigmas " + oss.str();
        }
    }

    if (full || gen.ref_image_paths != def_gen.ref_image_paths) {
        for (const auto& path : gen.ref_image_paths) {
            out += " --ref-image " + quote_if_needed(path);
        }
    }

    // sample_method / scheduler are ManualOptions (no target pointer), but
    // they directly change generated pixels, so unlike the other
    // ManualOptions below they get an explicit round-trip via the public
    // reverse-mapping helpers instead of just a warning. COUNT means
    // "unset/model-default"; there is no CLI spelling for that, so it is
    // never emitted, even in full mode.
    if (gen.sample_params.sample_method != SAMPLE_METHOD_COUNT) {
        out += " --sampling-method " + std::string(sd_sample_method_name(gen.sample_params.sample_method));
    }
    if (gen.sample_params.scheduler != SCHEDULER_COUNT) {
        out += " --scheduler " + std::string(sd_scheduler_name(gen.sample_params.scheduler));
    }

    // Remaining ManualOptions (no target pointer, no output-critical enum
    // reverse-mapping) are not round-tripped by this serializer. Warn
    // instead of silently dropping session state.
    out += "\n# note: not serialized (re-add manually if used): "
           "--skip-layers --high-noise-skip-layers --high-noise-sampling-method "
           "--cache-mode --cache-option --scm-mask --scm-policy "
           "--rng --sampler-rng --prediction --lora-apply-mode --type "
           "--vae-tile-size --vae-relative-tile-size "
           "--prompt-file --negative-prompt-file --hires-sigmas";

    return out;
}
