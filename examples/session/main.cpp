#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "stable-diffusion.h"
#include "common/common.h"
#include "common/media_io.h"
#include "common/resource_owners.hpp"
#include "session.h"
#include "session_params.h"
#include "tokenize.h"

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
    Session sess;
    std::string line;
    while (true) {
        if (interactive) {
            std::cout << "sd> " << std::flush;
        }
        if (!std::getline(in, line)) {
            break;  // EOF
        }
        std::vector<std::string> tokens = tokenize_line(line);
        if (tokens.empty() || tokens[0][0] == '#') {
            continue;
        }
        const std::string& cmd = tokens[0];
        std::vector<std::string> args(tokens.begin() + 1, tokens.end());
        if (cmd == "quit" || cmd == "exit") {
            break;
        } else if (cmd == "help") {
            print_help();
        } else if (cmd == "load") {
            std::string err;
            if (sess.loaded()) {
                std::cout << "already loaded; use swap\n";
                continue;
            }
            if (!apply_flags(args, sess.cli, sess.ctx, sess.gen, err)) {
                std::cout << "error: " << err << "\n";
                continue;
            }
            if (!sess.load(err)) {
                std::cout << "error: " << err << "\n";
                continue;
            }
            std::cout << "loaded\n";
        } else if (cmd == "unload") {
            sess.unload();
            std::cout << "unloaded\n";
        } else if (cmd == "swap") {
            std::string err;
            if (!apply_flags(args, sess.cli, sess.ctx, sess.gen, err)) {
                std::cout << "error: " << err << "\n";
                continue;
            }
            sess.unload();
            if (!sess.load(err)) {
                std::cout << "error: " << err << "\n";
                continue;
            }
            std::cout << "swapped\n";
        } else {
            std::cout << "unknown command: " << cmd << " (try 'help')\n";
        }
    }
    return true;
}
