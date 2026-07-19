#include <cstdio>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#if defined(_WIN32)
#include <io.h>
#define ISATTY _isatty
#define FILENO _fileno
#else
#include <unistd.h>
#define ISATTY isatty
#define FILENO fileno
#endif

#include "stable-diffusion.h"
#include "common/common.h"
#include "common/log.h"
#include "common/media_io.h"
#include "common/resource_owners.hpp"
#include "generate.h"
#include "serialize.h"
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
        "  reset                 clear all params to defaults, unload context\n"
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
    bool interactive = ISATTY(FILENO(stdin)) != 0;
    return run_repl(std::cin, interactive) ? 0 : 1;
}

// Dispatches a single REPL line. Returns false when the caller should stop
// looping (quit/exit); true otherwise, including on command errors.
// History recording happens here (not in run_repl) so both the interactive
// loop and `run <file>` share one place that decides which commands count as
// state-changing and worth replaying.
static bool dispatch(Session& sess, const std::string& line) {
    std::vector<std::string> tokens = tokenize_line(line);
    if (tokens.empty() || tokens[0][0] == '#') {
        return true;
    }
    const std::string& cmd = tokens[0];
    std::vector<std::string> args(tokens.begin() + 1, tokens.end());
    if (cmd == "quit" || cmd == "exit") {
        return false;
    } else if (cmd == "help") {
        print_help();
    } else if (cmd == "load") {
        std::string err;
        if (sess.loaded()) {
            std::cout << "already loaded; use swap\n";
            return true;
        }
        if (!apply_flags(args, sess.cli, sess.ctx, sess.gen, err)) {
            std::cout << "error: " << err << "\n";
            return true;
        }
        if (!sess.load(err)) {
            std::cout << "error: " << err << "\n";
            return true;
        }
        std::cout << "loaded\n";
        sess.history.push_back(line);
    } else if (cmd == "unload") {
        sess.unload();
        std::cout << "unloaded\n";
        sess.history.push_back(line);
    } else if (cmd == "reset") {
        sess.unload();
        sess.cli = SDCliParams{};
        sess.ctx = SDContextParams{};
        sess.gen = SDGenerationParams{};
        std::cout << "reset: params cleared to defaults, context unloaded\n";
        sess.history.push_back(line);
    } else if (cmd == "swap") {
        std::string err;
        bool was_loaded          = sess.loaded();
        std::string model_before = sess.model_signature();
        if (!apply_flags(args, sess.cli, sess.ctx, sess.gen, err)) {
            std::cout << "error: " << err << "\n";
            return true;
        }
        if (was_loaded && sess.model_signature() != model_before) {
            std::cout << "warning: model changed; freeing the previous model and reloading\n";
        }
        sess.unload();
        if (!sess.load(err)) {
            std::cout << "error: " << err << "\n";
            return true;
        }
        std::cout << "swapped\n";
        sess.history.push_back(line);
    } else if (cmd == "gen") {
        std::string err;
        int idx = -1;
        GenTiming timing;
        if (!run_gen(sess, args, idx, err, timing)) {
            std::cout << "error: " << err << "\n";
            return true;
        }
        if (idx >= 0) {
            std::cout << "gen ok -> session_out_" << idx << ".png (seed now " << sess.gen.seed << ")\n";
        } else {
            std::cout << "gen ok (no image saved) (seed now " << sess.gen.seed << ")\n";
        }
        std::cout << "gen #" << sess.stats.gens
                   << " seed=" << (sess.gen.seed - (sess.sticky ? 1 : 0))
                   << " " << timing.seconds << "s"
                   << " | VRAM free " << (timing.before.vram_free / (1024 * 1024))
                   << "->" << (timing.after.vram_free / (1024 * 1024)) << " MiB"
                   << " | RSS " << (timing.before.rss / (1024 * 1024))
                   << "->" << (timing.after.rss / (1024 * 1024)) << " MiB\n";
        sess.history.push_back(line);
    } else if (cmd == "stats") {
        std::cout << sess.stats.format() << "\n";
    } else if (cmd == "mode") {
        if (args.size() == 1 && args[0] == "sticky") {
            sess.sticky = true;
            std::cout << "mode sticky\n";
            sess.history.push_back(line);
        } else if (args.size() == 1 && args[0] == "explicit") {
            sess.sticky = false;
            std::cout << "mode explicit\n";
            sess.history.push_back(line);
        } else {
            std::cout << "usage: mode sticky|explicit\n";
        }
    } else if (cmd == "set") {
        std::string err;
        std::string model_before = sess.model_signature();
        if (!apply_flags(args, sess.cli, sess.ctx, sess.gen, err)) {
            std::cout << "error: " << err << "\n";
            return true;
        }
        if (sess.loaded() && sess.model_signature() != model_before) {
            std::cout << "warning: model changed; the loaded context will be freed and reloaded on next gen\n";
        }
        std::cout << "set ok" << (sess.loaded() && sess.context_changed() ? " (ctx will rebuild on next gen)" : "") << "\n";
        sess.history.push_back(line);
    } else if (cmd == "export") {
        if (args.empty()) {
            std::cout << "usage: export <file>\n";
        } else {
            std::ofstream out(args[0]);
            for (const auto& h : sess.history) out << h << "\n";
            std::cout << "exported " << sess.history.size() << " commands to " << args[0] << "\n";
        }
    } else if (cmd == "export-cli") {
        // Scope note: this always serializes the session's current effective
        // params, not a specific point in `history`; a history-index
        // selector is deferred.
        bool full = false;
        std::string file;
        for (const auto& a : args) {
            if (a == "--full") {
                full = true;
            } else {
                file = a;
            }
        }
        std::string s = serialize_to_cli(sess.cli, sess.ctx, sess.gen, full);
        if (file.empty()) {
            std::cout << s << "\n";
        } else {
            std::ofstream out(file);
            out << s << "\n";
            std::cout << "wrote " << file << "\n";
        }
    } else if (cmd == "run") {
        static const int kMaxRunDepth = 32;
        static int run_depth = 0;
        if (args.empty()) {
            std::cout << "usage: run <file>\n";
        } else if (run_depth >= kMaxRunDepth) {
            std::cout << "error: run depth limit (" << kMaxRunDepth << ") exceeded, refusing to open " << args[0] << "\n";
        } else {
            std::ifstream in(args[0]);
            if (!in) {
                std::cout << "cannot open " << args[0] << "\n";
            } else {
                ++run_depth;
                std::string l;
                while (std::getline(in, l)) {
                    if (!dispatch(sess, l)) break;
                }
                --run_depth;
            }
        }
    } else {
        std::cout << "unknown command: " << cmd << " (try 'help')\n";
    }
    return true;
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
        if (!dispatch(sess, line)) {
            break;
        }
    }
    return true;
}
