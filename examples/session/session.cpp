#include "session.h"
#include "common/log.h"

bool Session::context_changed() const {
    return ctx.to_string() != loaded_ctx_signature;
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
