#include "xdsp.h"
void vadd_sat_q15(const int16_t *a, const int16_t *b, int16_t *y, size_t n) {
    for (size_t vl; n > 0; n -= vl, a += vl, b += vl, y += vl) {
        vl = xd_hlen(n);
        xd_hstore(y, xd_haddsat(xd_hload(a, vl), xd_hload(b, vl), vl), vl);
    }
}
