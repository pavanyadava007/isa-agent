#include "xdsp.h"
/* Pass 1: vector max of saturating |x|. Pass 2: first index whose magnitude equals that max. */
size_t argmax_abs_q15(const int16_t *x, size_t n) {
    int16_t best = -1;
    const int16_t *p = x;
    for (size_t vl, m = n; m > 0; m -= vl, p += vl) {
        vl = xd_hlen(m);
        int16_t mx = xd_hredmax(xd_habssat(xd_hload(p, vl), vl), vl);
        if (mx > best) best = mx;
    }
    size_t base = 0;
    for (size_t vl, m = n; m > 0; m -= vl, base += vl) {
        vl = xd_hlen(m);
        long f = xd_hfirst(xd_heqx(xd_habssat(xd_hload(x + base, vl), vl), best, vl), vl);
        if (f >= 0) return base + (size_t)f;
    }
    return 0;
}
