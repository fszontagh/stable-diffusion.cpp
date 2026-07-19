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

    // ManualOptions this serializer does not round-trip must be flagged,
    // not silently dropped.
    CHECK(s.find("# note:") != std::string::npos);

    // vid_gen (or any non-default mode) must round-trip via --mode; without
    // it the exported command would silently re-run as img_gen.
    SDCliParams vid_cli;
    vid_cli.mode = VID_GEN;
    std::string vid_s = serialize_to_cli(vid_cli, ctx, gen, /*full=*/false);
    CHECK(vid_s.find("--mode") != std::string::npos);
    CHECK(vid_s.find(modes_str[VID_GEN]) != std::string::npos);

    return 0;
}
