#include "xdsp.h"
size_t count_above_i32(const int32_t *x, int32_t thr, size_t n) {
    size_t c = 0;
    for (size_t vl; n > 0; n -= vl, x += vl) {
        vl = xd_slen(n);
        c += xd_scount(xd_sgtx(xd_sload(x, vl), thr, vl), vl);
    }
    return c;
}
