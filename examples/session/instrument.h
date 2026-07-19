#pragma once
#include <cstddef>
#include <string>

size_t get_rss_bytes();
size_t get_vram_free_bytes();
size_t get_vram_total_bytes();

struct MemSample {
    size_t rss;
    size_t vram_free;
};
MemSample sample_mem();

struct Stats {
    int gens             = 0;
    double total_seconds = 0, last_seconds = 0;
    size_t rss_peak        = 0;
    size_t vram_used_peak  = 0;
    void record(double seconds, const MemSample& before, const MemSample& after, size_t vram_total);
    std::string format() const;
};
