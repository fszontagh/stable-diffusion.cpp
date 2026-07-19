#include "instrument.h"
#include <cstdio>
#include <sstream>
#include "ggml-backend.h"

#if defined(_WIN32)
#include <windows.h>
#include <psapi.h>
#pragma comment(lib, "psapi.lib")
#elif defined(__APPLE__)
#include <mach/mach.h>
#else
#include <cstring>
#include <unistd.h>
#endif

size_t get_rss_bytes() {
#if defined(_WIN32)
    PROCESS_MEMORY_COUNTERS pmc;
    if (GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc))) {
        return (size_t)pmc.WorkingSetSize;
    }
    return 0;
#elif defined(__APPLE__)
    mach_task_basic_info_data_t info;
    mach_msg_type_number_t count = MACH_TASK_BASIC_INFO_COUNT;
    if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO, (task_info_t)&info, &count) == KERN_SUCCESS) {
        return (size_t)info.resident_size;
    }
    return 0;
#else
    FILE* f = std::fopen("/proc/self/statm", "r");
    if (!f) return 0;
    long pages    = 0;
    long resident = 0;
    if (std::fscanf(f, "%ld %ld", &pages, &resident) != 2) resident = 0;
    std::fclose(f);
    return (size_t)resident * (size_t)sysconf(_SC_PAGESIZE);
#endif
}

static bool query_vram(size_t& free_out, size_t& total_out) {
    size_t n = ggml_backend_dev_count();
    for (size_t i = 0; i < n; ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_GPU) {
            size_t f = 0, t = 0;
            ggml_backend_dev_memory(dev, &f, &t);
            free_out  = f;
            total_out = t;
            return true;
        }
    }
    free_out  = 0;
    total_out = 0;
    return false;
}

size_t get_vram_free_bytes() {
    size_t f = 0, t = 0;
    query_vram(f, t);
    return f;
}

size_t get_vram_total_bytes() {
    size_t f = 0, t = 0;
    query_vram(f, t);
    return t;
}

MemSample sample_mem() {
    return MemSample{get_rss_bytes(), get_vram_free_bytes()};
}

void Stats::record(double seconds, const MemSample& before, const MemSample& after, size_t vram_total) {
    (void)before;
    ++gens;
    last_seconds = seconds;
    total_seconds += seconds;
    if (after.rss > rss_peak) rss_peak = after.rss;
    if (vram_total > 0 && after.vram_free <= vram_total) {
        size_t used = vram_total - after.vram_free;
        if (used > vram_used_peak) vram_used_peak = used;
    }
}

std::string Stats::format() const {
    std::ostringstream os;
    double avg = gens ? total_seconds / gens : 0.0;
    os << "gens=" << gens
       << " avg=" << avg << "s last=" << last_seconds << "s"
       << " rss_peak=" << (rss_peak / (1024 * 1024)) << "MiB"
       << " vram_used_peak=" << (vram_used_peak / (1024 * 1024)) << "MiB";
    return os.str();
}
