#include "xdsp.h"
/* Vectorise over output samples: each lane is one output; loop over taps with scalar coefficients. */
void fir_q15(const int16_t *x, const int16_t *h, int16_t *y, size_t n, size_t ntaps) {
    for (size_t vl; n > 0; n -= vl, x += vl, y += vl) {
        vl = xd_hlen(n);
        xd_w acc = xd_wmulx(xd_hload(x, vl), h[0], vl);
        for (size_t k = 1; k < ntaps; k++)
            acc = xd_wmacx(acc, h[k], xd_hload(x + k, vl), vl);
        xd_hstore(y, xd_wnarrow(acc, 15, vl), vl);
    }
}
