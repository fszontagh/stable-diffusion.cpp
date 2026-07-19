#pragma once
#include <string>
#include "common/cli_params.h"
#include "common/common.h"

// Serializes the current session params back into an `sd-cli` command line.
// This is the inverse of apply_flags(): non-full mode emits only flags whose
// value differs from a default-constructed struct; full mode emits every
// String/Int/Float/Bool flag regardless of value.
std::string serialize_to_cli(const SDCliParams& cli,
                             const SDContextParams& ctx,
                             const SDGenerationParams& gen,
                             bool full);
