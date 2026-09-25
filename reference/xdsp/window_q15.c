#include "xdsp.h"
void window_q15(const int16_t *x, const int16_t *w, int16_t *y, size_t n) {
    for (size_t vl; n > 0; n -= vl, x += vl, w += vl, y += vl) {
        vl = xd_hlen(n);
        xd_hstore(y, xd_hqmul(xd_hload(x, vl), xd_hload(w, vl), vl), vl);
    }
}
