#pragma once
#include <string>
#include <vector>
#include "session.h"

bool run_gen(Session& sess, const std::vector<std::string>& args,
             int& out_index, std::string& err);
