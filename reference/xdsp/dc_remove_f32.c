#include "xdsp.h"
void dc_remove_f32(const float *x, float *y, size_t n) {
    xd_f acc = xd_fdup(0.0f, xd_smaxlen());
    const float *p = x;
    for (size_t vl, m = n; m > 0; m -= vl, p += vl) {
        vl = xd_slen(m);
        acc = xd_fmacx(acc, 1.0f, xd_fload(p, vl), vl);
    }
    float mean = xd_fredsum(acc, xd_smaxlen()) / (float)n;
    for (size_t vl; n > 0; n -= vl, x += vl, y += vl) {
        vl = xd_slen(n);
        xd_fstore(y, xd_faddx(xd_fload(x, vl), -mean, vl), vl);
    }
}
