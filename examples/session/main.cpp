#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "stable-diffusion.h"
#include "common/common.h"
#include "common/media_io.h"
#include "common/resource_owners.hpp"

static void print_help() {
    std::cout <<
        "commands:\n"
        "  load [flags]          create context from current params\n"
        "  unload                free context, keep params\n"
        "  set [flags]           change params without generating\n"
        "  gen [flags]           generate with current params\n"
        "  swap [flags]          free + recreate context\n"
        "  mode sticky|explicit  param inheritance mode\n"
        "  stats                 print memory/latency table\n"
        "  export <file>         dump command history\n"
        "  export-cli [--full] [file]  emit an sd-cli command line\n"
        "  run <file>            execute a script of commands\n"
        "  help / quit\n";
}

static bool run_repl(std::istream& in, bool interactive);

int main(int argc, const char** argv) {
    (void)argc;
    (void)argv;
    bool interactive = true;  // refined in Task 8 (detect piped stdin / script file)
    return run_repl(std::cin, interactive) ? 0 : 1;
}

static bool run_repl(std::istream& in, bool interactive) {
    std::string line;
    while (true) {
        if (interactive) {
            std::cout << "sd> " << std::flush;
        }
        if (!std::getline(in, line)) {
            break;  // EOF
        }
        std::istringstream ls(line);
        std::string cmd;
        ls >> cmd;
        if (cmd.empty() || cmd[0] == '#') {
            continue;
        }
        if (cmd == "quit" || cmd == "exit") {
            break;
        } else if (cmd == "help") {
            print_help();
        } else {
            std::cout << "unknown command: " << cmd << " (try 'help')\n";
        }
    }
    return true;
}
