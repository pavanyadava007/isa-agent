#include "xdsp.h"
void scale_q15(const int16_t *x, int16_t k, int16_t *y, size_t n) {
    for (size_t vl; n > 0; n -= vl, x += vl, y += vl) {
        vl = xd_hlen(n);
        xd_hstore(y, xd_hqmulx(xd_hload(x, vl), k, vl), vl);
    }
}
