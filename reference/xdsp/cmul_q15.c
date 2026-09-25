#include "xdsp.h"
void cmul_q15(const int16_t *a, const int16_t *b, int16_t *y, size_t n) {
    for (size_t vl; n > 0; n -= vl, a += 2 * vl, b += 2 * vl, y += 2 * vl) {
        vl = xd_hlen(n);
        xd_h ar, ai, br, bi;
        xd_hload_iq(a, &ar, &ai, vl);
        xd_hload_iq(b, &br, &bi, vl);
        xd_w re = xd_wsub(xd_wmul(ar, br, vl), xd_wmul(ai, bi, vl), vl);
        xd_w im = xd_wmac(xd_wmul(ar, bi, vl), ai, br, vl);
        xd_hstore_iq(y, xd_wnarrow(re, 15, vl), xd_wnarrow(im, 15, vl), vl);
    }
}
