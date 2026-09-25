#include "xdsp.h"
int32_t dot_q15(const int16_t *a, const int16_t *b, size_t n) {
    xd_w acc = xd_wdup(0, xd_hmaxlen());
    for (size_t vl; n > 0; n -= vl, a += vl, b += vl) {
        vl = xd_hlen(n);
        acc = xd_wmac(acc, xd_hload(a, vl), xd_hload(b, vl), vl);
    }
    return xd_wredsum(acc, xd_hmaxlen());
}
