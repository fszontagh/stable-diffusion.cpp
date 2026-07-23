#pragma once
#include <string>
#include <vector>
#include "instrument.h"
#include "session.h"

struct GenTiming {
    double seconds = 0;
    MemSample before{};
    MemSample after{};
};

std::string resolve_session_output_path(const std::string& output_path, int idx);

bool run_gen(Session& sess, const std::vector<std::string>& args,
             int& out_index, std::string& err, GenTiming& out_timing);
