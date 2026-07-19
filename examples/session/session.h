#pragma once
#include <string>
#include <vector>
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
    std::vector<std::string> history;

    bool loaded() const { return sd_ctx != nullptr; }
    bool context_changed() const;
    std::string model_signature() const;
    bool load(std::string& err);
    void unload();
    bool ensure_ctx(std::string& err);
};
