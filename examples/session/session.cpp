#include "session.h"
#include "common/log.h"

bool Session::context_changed() const {
    return ctx.to_string() != loaded_ctx_signature;
}

// Concatenation of the weight-file paths that define which model is loaded.
// Used to warn when a set/swap actually replaces the model (as opposed to
// tweaking a non-model context param), since that forces an expensive
// free + reload.
std::string Session::model_signature() const {
    return ctx.model_path + "|" + ctx.diffusion_model_path + "|" +
           ctx.high_noise_diffusion_model_path + "|" + ctx.uncond_diffusion_model_path + "|" +
           ctx.vae_path + "|" + ctx.taesd_path + "|" +
           ctx.llm_path + "|" + ctx.llm_vision_path + "|" +
           ctx.clip_l_path + "|" + ctx.clip_g_path + "|" + ctx.clip_vision_path + "|" +
           ctx.t5xxl_path + "|" + ctx.control_net_path;
}

bool Session::load(std::string& err) {
    sd_ctx_params_t p = ctx.to_sd_ctx_params_t(cli.taesd_preview);
    sd_ctx.reset(new_sd_ctx(&p));
    if (sd_ctx == nullptr) {
        err = "new_sd_ctx failed";
        return false;
    }
    loaded_ctx_signature = ctx.to_string();
    return true;
}

void Session::unload() {
    sd_ctx.reset();
    loaded_ctx_signature.clear();
}

bool Session::ensure_ctx(std::string& err) {
    if (!loaded()) {
        return load(err);
    }
    if (context_changed()) {
        LOG_INFO("context params changed, rebuilding ctx (swap)");
        unload();
        return load(err);
    }
    return true;
}
