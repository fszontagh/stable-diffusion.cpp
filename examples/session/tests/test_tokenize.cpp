#include "check.h"
#include <string>
#include <vector>
#include "../tokenize.h"

int main() {
    auto a = tokenize_line("gen -p \"a cat\" --steps 8");
    CHECK((a == std::vector<std::string>{"gen", "-p", "a cat", "--steps", "8"}));

    auto b = tokenize_line("  set   --seed 42  ");
    CHECK((b == std::vector<std::string>{"set", "--seed", "42"}));

    auto c = tokenize_line("-p 'single quoted'");
    CHECK((c == std::vector<std::string>{"-p", "single quoted"}));

    auto d = tokenize_line("");
    CHECK(d.empty());

    auto e = tokenize_line("-p \"say \\\"hi\\\"\"");
    CHECK((e == std::vector<std::string>{"-p", "say \"hi\""}));

    return 0;
}
