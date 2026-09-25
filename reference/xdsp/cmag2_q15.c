#include "xdsp.h"
void cmag2_q15(const int16_t *iq, int32_t *p, size_t n) {
    for (size_t vl; n > 0; n -= vl, iq += 2 * vl, p += vl) {
        vl = xd_hlen(n);
        xd_h i, q;
        xd_hload_iq(iq, &i, &q, vl);
        xd_wstore(p, xd_wmac(xd_wmul(i, i, vl), q, q, vl), vl);
    }
}
