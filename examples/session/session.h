#pragma once
#include <string>
#include "common/cli_params.h"
#include "common/common.h"
#include "common/resource_owners.hpp"
#include "instrument.h"

struct Session {
    SDCliParams cli;
    SDContextParams ctx;
    SDGenerationParams gen;
    SDCtxPtr sd_ctx;
    std::string loaded_ctx_signature;
    bool sticky = true;
    Stats stats;

    bool loaded() const { return sd_ctx != nullptr; }
    bool context_changed() const;
    bool load(std::string& err);
    void unload();
    bool ensure_ctx(std::string& err);
};
