#include "check.h"
#include <string>
#include <vector>
#include "../session_params.h"
#include "common/cli_params.h"
#include "common/common.h"

int main() {
    SDCliParams cli;
    SDContextParams ctx;
    SDGenerationParams gen;
    std::string err;

    // A minimal valid context is required for resolve_and_validate() to
    // succeed; the test targets merge semantics, not model resolution.
    ctx.model_path = "dummy.gguf";

    gen.prompt = "old";
    gen.seed   = 1;
    // Override only the prompt; seed must be preserved (merge, not reset).
    bool ok = apply_flags({"-p", "new prompt"}, cli, ctx, gen, err);
    CHECK(ok);
    CHECK(gen.prompt == "new prompt");
    CHECK(gen.seed == 1);

    // Unknown flag reports an error instead of exiting.
    bool bad = apply_flags({"--does-not-exist"}, cli, ctx, gen, err);
    CHECK(!bad);
    CHECK(!err.empty());

    // Invalid mode must not terminate the process; it must return false
    // with a non-empty error. Reaching the CHECK below proves apply_flags
    // did not call exit().
    err.clear();
    bool bad_mode = apply_flags({"-M", "bogusmode"}, cli, ctx, gen, err);
    CHECK(!bad_mode);
    CHECK(!err.empty());

    // A malformed numeric flag value must not terminate the process (the
    // underlying parser can throw std::invalid_argument/out_of_range from
    // std::stoi/std::stof); apply_flags must catch it and report an error.
    // std::stoi() parses a leading numeric prefix and ignores trailing
    // garbage (e.g. "8x" -> 8), so it does not throw. Use a value with no
    // leading digits to force std::invalid_argument out of std::stoi.
    err.clear();
    bool bad_numeric = apply_flags({"--steps", "abc"}, cli, ctx, gen, err);
    CHECK(!bad_numeric);
    CHECK(!err.empty());

    // A line that mutates state before failing later must not leave the
    // caller's structs partially updated (all-or-nothing semantics).
    gen.prompt = "keepme";
    err.clear();
    bool partial_fail = apply_flags({"-p", "changed", "--steps", "abc"}, cli, ctx, gen, err);
    CHECK(!partial_fail);
    CHECK(gen.prompt == "keepme");

    return 0;
}
