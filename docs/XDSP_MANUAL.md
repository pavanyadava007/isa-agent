# XDSP Programmer's Reference Manual

Generated from `isaagent/isa_spec.py`. This manual is the only specification of XDSP.

## Overview

XDSP is a vector DSP extension. A program processes arrays in chunks ("strip-mining"): each loop iteration asks the core how many lanes it may use (vl) and every vector operation takes that vl as its last argument. Lanes >= vl are not written by loads/stores. All XDSP operations are C functions declared in `xdsp.h`; include it with `#include "xdsp.h"`. Only the operations in this manual exist. The core also runs ordinary scalar C code, which is always allowed (for example loop remainders or sequential recurrences).

## Data types

- `xd_h`: Vector of int16 lanes ("half" lanes). Lane count set by xd_hlen().
- `xd_w`: Vector of int32 lanes, one per xd_h lane ("wide" accumulator). Same vl as xd_h.
- `xd_s`: Vector of int32 lanes ("single" lanes). Lane count set by xd_slen().
- `xd_f`: Vector of float32 lanes. Uses the same vl as xd_s (xd_slen()).
- `xd_hmask`: Per-lane predicate produced by xd_h / xd_w comparisons.
- `xd_smask`: Per-lane predicate produced by xd_s / xd_f comparisons.

There are two lane families. The int16 family (xd_h, and xd_w which holds the int32 widening of each xd_h lane) uses vl from xd_hlen(). The int32/float family (xd_s, xd_f) uses vl from xd_slen(). Never mix a vl from one family with operations of the other.

## Fixed-point conventions

Q15: an int16 x represents x / 32768. The product of two Q15 numbers is Q30 and fits in int32. Rounding operations (xd_hqmul, xd_hqmulx, xd_hsrar, xd_wnarrow, xd_havg) use round-to-nearest-up: add half an LSB, then shift arithmetically. Saturating operations clamp to [-32768, 32767] instead of wrapping. A typical Q15 dot product or FIR accumulates exact products with xd_wmac in an xd_w and converts back once with xd_wnarrow(acc, 15, vl).

## Programming example: saturating vector add

```c
#include "xdsp.h"
void add_sat(const int16_t *a, const int16_t *b, int16_t *y, size_t n) {
    for (size_t vl; n > 0; n -= vl, a += vl, b += vl, y += vl) {
        vl = xd_hlen(n);
        xd_h va = xd_hload(a, vl);
        xd_h vb = xd_hload(b, vl);
        xd_hstore(y, xd_haddsat(va, vb, vl), vl);
    }
}
```

## Programming example: reduction with a wide accumulator

```c
#include "xdsp.h"
int32_t sum_sq(const int16_t *x, size_t n) {
    xd_w acc = xd_wdup(0, xd_hmaxlen());      /* zero ALL lanes once */
    for (size_t vl; n > 0; n -= vl, x += vl) {
        vl = xd_hlen(n);
        xd_h v = xd_hload(x, vl);
        acc = xd_wmac(acc, v, v, vl);          /* lanes >= vl keep their partial sums */
    }
    return xd_wredsum(acc, xd_hmaxlen());       /* reduce every lane */
}
```

## Operation reference

### Lane-count control

#### xd_hlen

`size_t xd_hlen(size_t n)`

Returns vl = min(n, XD_HLANES), the number of int16 lanes to process in this loop iteration. Use it for xd_h and xd_w operations. Typical strip-mining loop: `for (size_t vl; n > 0; n -= vl, p += vl) { vl = xd_hlen(n); ... }`.

#### xd_hmaxlen

`size_t xd_hmaxlen(void)`

Returns XD_HLANES, the maximum number of int16 lanes per xd_h register (8 on the reference core, but never hard-code it).

#### xd_slen

`size_t xd_slen(size_t n)`

Returns vl = min(n, XD_SLANES) for int32 (xd_s) and float32 (xd_f) operations.

#### xd_smaxlen

`size_t xd_smaxlen(void)`

Returns XD_SLANES, the maximum number of lanes per xd_s / xd_f register (4 on the reference core).

### int16 memory

#### xd_hload

`xd_h xd_hload(const int16_t *p, size_t vl)`

Loads vl consecutive int16 values from p into lanes 0..vl-1.

#### xd_hstore

`void xd_hstore(int16_t *p, xd_h v, size_t vl)`

Stores lanes 0..vl-1 of v to p[0..vl-1]. Memory beyond p[vl-1] is not touched.

#### xd_hload_step

`xd_h xd_hload_step(const int16_t *p, ptrdiff_t step, size_t vl)`

Strided load: lane i = p[i * step]. `step` is in ELEMENTS (not bytes); it may be negative.

#### xd_hload_iq

`void xd_hload_iq(const int16_t *p, xd_h *i_out, xd_h *q_out, size_t vl)`

De-interleaving load of vl complex samples stored as I,Q,I,Q,... Lane k of *i_out = p[2k], lane k of *q_out = p[2k+1]. Reads 2*vl int16 values.

#### xd_hstore_iq

`void xd_hstore_iq(int16_t *p, xd_h i_in, xd_h q_in, size_t vl)`

Interleaving store: p[2k] = lane k of i_in, p[2k+1] = lane k of q_in, for k < vl.

### int16 arithmetic

#### xd_hdup

`xd_h xd_hdup(int16_t x, size_t vl)`

Broadcasts scalar x into lanes 0..vl-1.

#### xd_hadd

`xd_h xd_hadd(xd_h a, xd_h b, size_t vl)`

Lane-wise a + b, wrapping modulo 2^16.

#### xd_hsub

`xd_h xd_hsub(xd_h a, xd_h b, size_t vl)`

Lane-wise a - b, wrapping modulo 2^16.

#### xd_haddsat

`xd_h xd_haddsat(xd_h a, xd_h b, size_t vl)`

Lane-wise saturating a + b: result clamped to [-32768, 32767].

#### xd_hsubsat

`xd_h xd_hsubsat(xd_h a, xd_h b, size_t vl)`

Lane-wise saturating a - b: result clamped to [-32768, 32767].

#### xd_havg

`xd_h xd_havg(xd_h a, xd_h b, size_t vl)`

Lane-wise rounded average (a + b + 1) >> 1, computed without intermediate overflow (arithmetic shift, exact for all int16 inputs).

#### xd_hmul

`xd_h xd_hmul(xd_h a, xd_h b, size_t vl)`

Lane-wise low 16 bits of a * b (wrapping). For fixed-point use xd_hqmul.

#### xd_hqmul

`xd_h xd_hqmul(xd_h a, xd_h b, size_t vl)`

Q15 fractional multiply with rounding and saturation: result = sat16((a * b + 16384) >> 15). Only (-32768) * (-32768) saturates (to 32767).

#### xd_hqmulx

`xd_h xd_hqmulx(xd_h a, int16_t k, size_t vl)`

Q15 multiply of every lane by scalar k: sat16((a * k + 16384) >> 15).

#### xd_hmax

`xd_h xd_hmax(xd_h a, xd_h b, size_t vl)`

Lane-wise signed maximum.

#### xd_hmin

`xd_h xd_hmin(xd_h a, xd_h b, size_t vl)`

Lane-wise signed minimum.

#### xd_habssat

`xd_h xd_habssat(xd_h a, size_t vl)`

Lane-wise saturating absolute value: |a|, with |-32768| = 32767.

#### xd_hsra

`xd_h xd_hsra(xd_h a, unsigned sh, size_t vl)`

Arithmetic shift right by sh (0..15), truncating toward minus infinity.

#### xd_hsrar

`xd_h xd_hsrar(xd_h a, unsigned sh, size_t vl)`

Rounding arithmetic shift right: (a + (1 << (sh-1))) >> sh for sh >= 1, no overflow.

#### xd_hslide

`xd_h xd_hslide(xd_h a, size_t k, size_t vl)`

Lane i of result = lane i+k of a (lanes past the register end read as 0). Rarely needed: reloading from memory at p+k is usually cheaper.

### int16 compare, select, reduce

#### xd_hgt

`xd_hmask xd_hgt(xd_h a, xd_h b, size_t vl)`

Mask lane i = (a[i] > b[i]).

#### xd_hgtx

`xd_hmask xd_hgtx(xd_h a, int16_t x, size_t vl)`

Mask lane i = (a[i] > x).

#### xd_heqx

`xd_hmask xd_heqx(xd_h a, int16_t x, size_t vl)`

Mask lane i = (a[i] == x).

#### xd_hsel

`xd_h xd_hsel(xd_hmask m, xd_h if_true, xd_h if_false, size_t vl)`

Lane i = m[i] ? if_true[i] : if_false[i]. Note the argument order: mask first.

#### xd_hcount

`size_t xd_hcount(xd_hmask m, size_t vl)`

Number of set mask lanes among lanes 0..vl-1.

#### xd_hfirst

`long xd_hfirst(xd_hmask m, size_t vl)`

Index of the lowest set mask lane among 0..vl-1, or -1 if none is set.

#### xd_hredmax

`int16_t xd_hredmax(xd_h a, size_t vl)`

Maximum over lanes 0..vl-1 (returns -32768 if vl == 0).

#### xd_hredsum_w

`int32_t xd_hredsum_w(xd_h a, size_t vl)`

Sum of lanes 0..vl-1 computed in int32 (widening, exact unless the int32 sum overflows).

### Wide accumulators (xd_w)

#### xd_wdup

`xd_w xd_wdup(int32_t x, size_t vl)`

Broadcasts x into lanes 0..vl-1 of a wide accumulator. To zero an accumulator for a strip-mined loop use xd_wdup(0, xd_hmaxlen()).

#### xd_wwiden

`xd_w xd_wwiden(xd_h a, size_t vl)`

Sign-extends each int16 lane to int32.

#### xd_wmul

`xd_w xd_wmul(xd_h a, xd_h b, size_t vl)`

Lane-wise exact product a * b as int32.

#### xd_wmulx

`xd_w xd_wmulx(xd_h a, int16_t k, size_t vl)`

Lane-wise exact product a * k as int32.

#### xd_wmac

`xd_w xd_wmac(xd_w acc, xd_h a, xd_h b, size_t vl)`

acc[i] += a[i] * b[i] (int32, wrapping) for i < vl. Lanes i >= vl keep their old value (tail-undisturbed), so one accumulator can be reused across a strip-mined loop whose last iteration has a shorter vl.

#### xd_wmacx

`xd_w xd_wmacx(xd_w acc, int16_t k, xd_h a, size_t vl)`

acc[i] += k * a[i] (int32, wrapping) for i < vl; lanes i >= vl unchanged. Note argument order: scalar coefficient before the vector.

#### xd_wadd

`xd_w xd_wadd(xd_w a, xd_w b, size_t vl)`

Lane-wise int32 a + b (wrapping).

#### xd_wsub

`xd_w xd_wsub(xd_w a, xd_w b, size_t vl)`

Lane-wise int32 a - b (wrapping).

#### xd_wsra

`xd_w xd_wsra(xd_w a, unsigned sh, size_t vl)`

Arithmetic shift right of int32 lanes by sh (0..31).

#### xd_wnarrow

`xd_h xd_wnarrow(xd_w a, unsigned sh, size_t vl)`

Narrow to int16 with rounding and saturation: sat16((a + (1 << (sh-1))) >> sh) for sh >= 1 (sh = 0: plain saturation). This is the standard way to turn a Q30 product sum back into Q15 (sh = 15).

#### xd_wredsum

`int32_t xd_wredsum(xd_w a, size_t vl)`

Sum of lanes 0..vl-1 (int32, wrapping). After a strip-mined xd_wmac loop, reduce with vl = xd_hmaxlen() so that lanes filled in earlier, longer iterations are included.

#### xd_wload

`xd_w xd_wload(const int32_t *p, size_t vl)`

Loads vl int32 values into an xd_w (vl from xd_hlen).

#### xd_wstore

`void xd_wstore(int32_t *p, xd_w v, size_t vl)`

Stores lanes 0..vl-1 of an xd_w as int32 (vl from xd_hlen).

#### xd_wgtx

`xd_hmask xd_wgtx(xd_w a, int32_t x, size_t vl)`

Mask lane i = (a[i] > x) for wide lanes.

### int32 (xd_s)

#### xd_sload

`xd_s xd_sload(const int32_t *p, size_t vl)`

Loads vl int32 values (vl from xd_slen).

#### xd_sstore

`void xd_sstore(int32_t *p, xd_s v, size_t vl)`

Stores lanes 0..vl-1.

#### xd_sdup

`xd_s xd_sdup(int32_t x, size_t vl)`

Broadcast.

#### xd_sadd

`xd_s xd_sadd(xd_s a, xd_s b, size_t vl)`

Lane-wise a + b (wrapping).

#### xd_ssub

`xd_s xd_ssub(xd_s a, xd_s b, size_t vl)`

Lane-wise a - b (wrapping).

#### xd_smul

`xd_s xd_smul(xd_s a, xd_s b, size_t vl)`

Lane-wise low 32 bits of a * b.

#### xd_smulx

`xd_s xd_smulx(xd_s a, int32_t k, size_t vl)`

Lane-wise low 32 bits of a * k.

#### xd_ssra

`xd_s xd_ssra(xd_s a, unsigned sh, size_t vl)`

Arithmetic shift right.

#### xd_smax

`xd_s xd_smax(xd_s a, xd_s b, size_t vl)`

Lane-wise maximum.

#### xd_sgt

`xd_smask xd_sgt(xd_s a, xd_s b, size_t vl)`

Mask lane i = (a[i] > b[i]).

#### xd_sgtx

`xd_smask xd_sgtx(xd_s a, int32_t x, size_t vl)`

Mask lane i = (a[i] > x).

#### xd_ssel

`xd_s xd_ssel(xd_smask m, xd_s if_true, xd_s if_false, size_t vl)`

Lane i = m[i] ? if_true[i] : if_false[i]. Mask first, like xd_hsel.

#### xd_scount

`size_t xd_scount(xd_smask m, size_t vl)`

Number of set lanes among 0..vl-1.

#### xd_sredsum

`int32_t xd_sredsum(xd_s a, size_t vl)`

Sum of lanes 0..vl-1 (int32, wrapping).

#### xd_sredmax

`int32_t xd_sredmax(xd_s a, size_t vl)`

Maximum over lanes 0..vl-1.

### float32 (xd_f)

#### xd_fload

`xd_f xd_fload(const float *p, size_t vl)`

Loads vl floats (vl from xd_slen).

#### xd_fstore

`void xd_fstore(float *p, xd_f v, size_t vl)`

Stores lanes 0..vl-1.

#### xd_fdup

`xd_f xd_fdup(float x, size_t vl)`

Broadcast.

#### xd_fadd

`xd_f xd_fadd(xd_f a, xd_f b, size_t vl)`

Lane-wise a + b.

#### xd_fsub

`xd_f xd_fsub(xd_f a, xd_f b, size_t vl)`

Lane-wise a - b.

#### xd_fmul

`xd_f xd_fmul(xd_f a, xd_f b, size_t vl)`

Lane-wise a * b.

#### xd_fmulx

`xd_f xd_fmulx(xd_f a, float k, size_t vl)`

Lane-wise a * k.

#### xd_faddx

`xd_f xd_faddx(xd_f a, float k, size_t vl)`

Lane-wise a + k.

#### xd_fmac

`xd_f xd_fmac(xd_f acc, xd_f a, xd_f b, size_t vl)`

Fused acc[i] += a[i] * b[i] for i < vl; lanes i >= vl unchanged (tail-undisturbed).

#### xd_fmacx

`xd_f xd_fmacx(xd_f acc, float k, xd_f a, size_t vl)`

Fused acc[i] += k * a[i] for i < vl; lanes i >= vl unchanged.

#### xd_fredsum

`float xd_fredsum(xd_f a, size_t vl)`

Sum of lanes 0..vl-1 in an unspecified order (results may differ from a sequential sum in the last bits).

#### xd_fredmax

`float xd_fredmax(xd_f a, size_t vl)`

Maximum over lanes 0..vl-1.
