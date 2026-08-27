#ifndef __AA_TEST_NPY_HPP__
#define __AA_TEST_NPY_HPP__

// Minimal .npy (numpy format v1.0) reader. Handles exactly the subset needed
// by the AnimateAnyone fixtures: little-endian fp32 data ('<f4'),
// fortran_order False (C order), an arbitrary-rank shape tuple. Anything else
// (other dtypes, fortran order, npy v2/v3 headers) is rejected with a clear
// error message rather than silently misinterpreted.

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace aa_test {

struct NpyArray {
    std::vector<int64_t> shape;
    std::vector<float> data;

    int64_t numel() const {
        int64_t n = 1;
        for (int64_t d : shape) {
            n *= d;
        }
        return shape.empty() ? 0 : n;
    }
};

// Returns true on success; on failure, fills `error` and leaves `out` untouched.
inline bool load_npy_f32(const std::string& path, NpyArray& out, std::string& error) {
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) {
        error = "failed to open npy file: " + path;
        return false;
    }

    unsigned char magic[6];
    f.read(reinterpret_cast<char*>(magic), 6);
    if (!f.good() || std::memcmp(magic, "\x93NUMPY", 6) != 0) {
        error = "not a valid .npy file (bad magic): " + path;
        return false;
    }

    unsigned char ver[2];
    f.read(reinterpret_cast<char*>(ver), 2);
    if (!f.good()) {
        error = "truncated npy version header: " + path;
        return false;
    }
    if (ver[0] != 1) {
        error = "unsupported npy version " + std::to_string((int)ver[0]) +
                "." + std::to_string((int)ver[1]) + " (only v1.0 supported): " + path;
        return false;
    }

    uint16_t header_len = 0;
    f.read(reinterpret_cast<char*>(&header_len), sizeof(header_len));
    if (!f.good()) {
        error = "truncated npy header length: " + path;
        return false;
    }

    std::string header(header_len, '\0');
    f.read(header.data(), header_len);
    if (!f.good()) {
        error = "truncated npy header dict: " + path;
        return false;
    }

    // Header is a Python dict literal, e.g.:
    //   {'descr': '<f4', 'fortran_order': False, 'shape': (1, 320, 1, 64, 64), }
    if (header.find("'descr': '<f4'") == std::string::npos) {
        error = "unsupported npy dtype (expected '<f4'), header: " + header;
        return false;
    }
    if (header.find("'fortran_order': False") == std::string::npos) {
        error = "unsupported npy layout (expected fortran_order: False), header: " + header;
        return false;
    }

    size_t shape_key = header.find("'shape':");
    if (shape_key == std::string::npos) {
        error = "npy header missing 'shape' key: " + header;
        return false;
    }
    size_t open_paren = header.find('(', shape_key);
    size_t close_paren = header.find(')', open_paren);
    if (open_paren == std::string::npos || close_paren == std::string::npos) {
        error = "npy header 'shape' is not a tuple: " + header;
        return false;
    }
    std::string shape_str = header.substr(open_paren + 1, close_paren - open_paren - 1);

    std::vector<int64_t> shape;
    std::stringstream ss(shape_str);
    std::string item;
    while (std::getline(ss, item, ',')) {
        // Strip whitespace.
        size_t start = item.find_first_not_of(" \t");
        size_t end   = item.find_last_not_of(" \t");
        if (start == std::string::npos) {
            continue;  // trailing comma, e.g. "(64,)"
        }
        item = item.substr(start, end - start + 1);
        if (item.empty()) {
            continue;
        }
        shape.push_back(std::stoll(item));
    }
    if (shape.empty()) {
        error = "npy header 'shape' parsed empty: " + header;
        return false;
    }

    int64_t numel = 1;
    for (int64_t d : shape) {
        numel *= d;
    }

    std::vector<float> data(static_cast<size_t>(numel));
    f.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(numel * sizeof(float)));
    if (!f.good()) {
        error = "truncated npy data (expected " + std::to_string(numel) + " floats): " + path;
        return false;
    }

    out.shape = std::move(shape);
    out.data  = std::move(data);
    return true;
}

}  // namespace aa_test

#endif  // __AA_TEST_NPY_HPP__
