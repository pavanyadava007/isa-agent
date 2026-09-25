#include "xdsp.h"
void decim2_q15(const int16_t *x, int16_t *y, size_t n) {
    for (size_t vl; n > 0; n -= vl, x += 2 * vl, y += vl) {
        vl = xd_hlen(n);
        xd_h e, o;
        xd_hload_iq(x, &e, &o, vl);
        xd_hstore(y, xd_havg(e, o, vl), vl);
    }
}
