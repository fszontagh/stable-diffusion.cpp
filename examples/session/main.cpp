#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "stable-diffusion.h"
#include "common/common.h"
#include "common/log.h"
#include "common/media_io.h"
#include "common/resource_owners.hpp"
#include "generate.h"
#include "session.h"
#include "session_params.h"
#include "tokenize.h"

static void sd_session_log_cb(enum sd_log_level_t level, const char* log, void* data) {
    (void)data;
    log_print(level, log, log_verbose, log_color);
}

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
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-v" || arg == "--verbose") {
            log_verbose = true;
        }
    }
    sd_set_log_callback(sd_session_log_cb, nullptr);
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
        } else if (cmd == "gen") {
            std::string err;
            int idx = -1;
            if (!run_gen(sess, args, idx, err)) {
                std::cout << "error: " << err << "\n";
                continue;
            }
            std::cout << "gen ok -> session_out_" << idx << ".png (seed now " << sess.gen.seed << ")\n";
        } else if (cmd == "mode") {
            if (args.size() == 1 && args[0] == "sticky") { sess.sticky = true; std::cout << "mode sticky\n"; }
            else if (args.size() == 1 && args[0] == "explicit") { sess.sticky = false; std::cout << "mode explicit\n"; }
            else { std::cout << "usage: mode sticky|explicit\n"; }
        } else if (cmd == "set") {
            std::string err;
            if (!apply_flags(args, sess.cli, sess.ctx, sess.gen, err)) { std::cout << "error: " << err << "\n"; continue; }
            std::cout << "set ok" << (sess.loaded() && sess.context_changed() ? " (ctx will rebuild on next gen)" : "") << "\n";
        } else {
            std::cout << "unknown command: " << cmd << " (try 'help')\n";
        }
    }
    return true;
}
