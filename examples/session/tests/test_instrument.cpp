#include "check.h"
#include "../instrument.h"

int main() {
    size_t rss = get_rss_bytes();
    CHECK(rss > 0);  // this process has non-zero RSS on every supported OS

    Stats s;
    MemSample a{100, 0}, b{150, 0};
    s.record(2.0, a, b, 0);
    s.record(4.0, a, b, 0);
    CHECK(s.gens == 2);
    CHECK(s.last_seconds == 4.0);
    CHECK(s.rss_peak == 150);
    return 0;
}
