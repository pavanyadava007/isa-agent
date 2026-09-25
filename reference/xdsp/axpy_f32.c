#include "xdsp.h"
void axpy_f32(float a, const float *x, float *y, size_t n) {
    for (size_t vl; n > 0; n -= vl, x += vl, y += vl) {
        vl = xd_slen(n);
        xd_fstore(y, xd_fmacx(xd_fload(y, vl), a, xd_fload(x, vl), vl), vl);
    }
}
