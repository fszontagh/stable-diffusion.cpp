#pragma once
#include <string>
#include <vector>
#include "common/cli_params.h"
#include "common/common.h"

bool apply_flags(const std::vector<std::string>& args,
                 SDCliParams& cli, SDContextParams& ctx, SDGenerationParams& gen,
                 std::string& err);
