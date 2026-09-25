#include "xdsp.h"
float rms_f32(const float *x, size_t n) {
    size_t total = n;
    xd_f acc = xd_fdup(0.0f, xd_smaxlen());
    for (size_t vl; n > 0; n -= vl, x += vl) {
        vl = xd_slen(n);
        xd_f v = xd_fload(x, vl);
        acc = xd_fmac(acc, v, v, vl);
    }
    return sqrtf(xd_fredsum(acc, xd_smaxlen()) / (float)total);
}
