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

    return 0;
}
