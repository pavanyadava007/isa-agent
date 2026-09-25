#include "xdsp.h"
/* Vectorise over cells i: S(i) = sum of 2*train shifted unit-stride loads, then one lane-wise compare.
 * XDSP has no byte store, so the 0/1 result goes through a small int32 buffer. */
void cfar_ca_i32(const int32_t *p, uint8_t *det, size_t n, size_t guard, size_t train, int32_t alpha) {
    size_t w = guard + train;
    for (size_t i = 0; i < n; i++) det[i] = 0;
    if (n <= 2 * w) return;
    int32_t scale = (int32_t)(2 * train);
    int32_t buf[64];
    for (size_t i = w, m = n - 2 * w, vl; m > 0; m -= vl, i += vl) {
        vl = xd_slen(m);
        xd_s s = xd_sdup(0, vl);
        for (size_t j = 0; j < train; j++) {
            s = xd_sadd(s, xd_sload(p + i - w + j, vl), vl);
            s = xd_sadd(s, xd_sload(p + i + guard + 1 + j, vl), vl);
        }
        xd_smask hit = xd_sgt(xd_smulx(xd_sload(p + i, vl), scale, vl), xd_smulx(s, alpha, vl), vl);
        if (xd_scount(hit, vl) == 0) continue;
        xd_sstore(buf, xd_ssel(hit, xd_sdup(1, vl), xd_sdup(0, vl), vl), vl);
        for (size_t k = 0; k < vl; k++) det[i + k] = (uint8_t)buf[k];
    }
}
