#include "tokenize.h"

std::vector<std::string> tokenize_line(const std::string& line) {
    std::vector<std::string> out;
    std::string cur;
    bool in_token = false;
    char quote    = 0;  // 0, '\'' or '"'
    for (size_t i = 0; i < line.size(); ++i) {
        char c = line[i];
        if (quote) {
            if (c == '\\' && quote == '"' && i + 1 < line.size() &&
                (line[i + 1] == '"' || line[i + 1] == '\\')) {
                cur += line[++i];
            } else if (c == quote) {
                quote = 0;
            } else {
                cur += c;
            }
            continue;
        }
        if (c == '\'' || c == '"') {
            quote    = c;
            in_token = true;
        } else if (c == ' ' || c == '\t' || c == '\r' || c == '\n') {
            if (in_token) {
                out.push_back(cur);
                cur.clear();
                in_token = false;
            }
        } else {
            cur += c;
            in_token = true;
        }
    }
    if (in_token) {
        out.push_back(cur);
    }
    return out;
}
