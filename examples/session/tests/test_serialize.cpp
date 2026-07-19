#include "check.h"
#include <string>
#include "../serialize.h"
#include "common/cli_params.h"
#include "common/common.h"

int main() {
    SDCliParams cli;
    SDContextParams ctx;
    SDGenerationParams gen;
    gen.prompt = "a cat";
    gen.seed   = 7;

    std::string s = serialize_to_cli(cli, ctx, gen, /*full=*/false);
    // non-default flags present
    CHECK(s.find("sd-cli") == 0);
    CHECK(s.find("--prompt") != std::string::npos || s.find("-p") != std::string::npos);
    CHECK(s.find("a cat") != std::string::npos);
    CHECK(s.find("7") != std::string::npos);
    // a default-valued flag is not present in non-full mode.
    CHECK(s.find("--output") == std::string::npos);
    CHECK(s.find("--batch-count") == std::string::npos);

    std::string full = serialize_to_cli(cli, ctx, gen, /*full=*/true);
    CHECK(full.size() > s.size());  // full emits strictly more
    return 0;
}
