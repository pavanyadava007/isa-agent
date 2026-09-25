"""XDSP: a fictional vector DSP instruction set used as a stand-in for a proprietary ISA.

Every XDSP operation is a thin static-inline wrapper around one (or a few) RISC-V Vector 1.0
intrinsics, so programs run bit-exactly on QEMU. The *names, types and manual* are new, so a
language model cannot have memorised them: it must read the manual. That mirrors the situation
of an in-house DSP whose documentation only exists inside the company.

`OPS` is the single source of truth. `gen_isa.py` renders it into `isa/include/xdsp.h` and
`docs/XDSP_MANUAL.md`.
"""

from __future__ import annotations

# Rounding mode used by every rounding op: round-to-nearest-up (add half LSB, then shift).
RNU = "__RISCV_VXRM_RNU"

TYPES = [
    ("xd_h", "vint16m1_t", "Vector of int16 lanes (\"half\" lanes). Lane count set by xd_hlen()."),
    ("xd_w", "vint32m2_t", "Vector of int32 lanes, one per xd_h lane (\"wide\" accumulator). Same vl as xd_h."),
    ("xd_s", "vint32m1_t", "Vector of int32 lanes (\"single\" lanes). Lane count set by xd_slen()."),
    ("xd_f", "vfloat32m1_t", "Vector of float32 lanes. Uses the same vl as xd_s (xd_slen())."),
    ("xd_hmask", "vbool16_t", "Per-lane predicate produced by xd_h / xd_w comparisons."),
    ("xd_smask", "vbool32_t", "Per-lane predicate produced by xd_s / xd_f comparisons."),
]

# Each op: name, C signature (return type, params), body (C expression or statements), group, doc.
# `body` may reference parameters by name. Keep semantics exact: the manual is the only spec.
OPS: list[dict] = [
    # ---- lane-count control -------------------------------------------------------------
    dict(name="xd_hlen", ret="size_t", params="size_t n", group="Lane-count control",
         body="return __riscv_vsetvl_e16m1(n);",
         doc="Returns vl = min(n, XD_HLANES), the number of int16 lanes to process in this loop "
             "iteration. Use it for xd_h and xd_w operations. Typical strip-mining loop: "
             "`for (size_t vl; n > 0; n -= vl, p += vl) { vl = xd_hlen(n); ... }`."),
    dict(name="xd_hmaxlen", ret="size_t", params="void", group="Lane-count control",
         body="return __riscv_vsetvlmax_e16m1();",
         doc="Returns XD_HLANES, the maximum number of int16 lanes per xd_h register "
             "(8 on the reference core, but never hard-code it)."),
    dict(name="xd_slen", ret="size_t", params="size_t n", group="Lane-count control",
         body="return __riscv_vsetvl_e32m1(n);",
         doc="Returns vl = min(n, XD_SLANES) for int32 (xd_s) and float32 (xd_f) operations."),
    dict(name="xd_smaxlen", ret="size_t", params="void", group="Lane-count control",
         body="return __riscv_vsetvlmax_e32m1();",
         doc="Returns XD_SLANES, the maximum number of lanes per xd_s / xd_f register "
             "(4 on the reference core)."),

    # ---- int16 memory -------------------------------------------------------------------
    dict(name="xd_hload", ret="xd_h", params="const int16_t *p, size_t vl", group="int16 memory",
         body="return __riscv_vle16_v_i16m1(p, vl);",
         doc="Loads vl consecutive int16 values from p into lanes 0..vl-1."),
    dict(name="xd_hstore", ret="void", params="int16_t *p, xd_h v, size_t vl", group="int16 memory",
         body="__riscv_vse16_v_i16m1(p, v, vl);",
         doc="Stores lanes 0..vl-1 of v to p[0..vl-1]. Memory beyond p[vl-1] is not touched."),
    dict(name="xd_hload_step", ret="xd_h", params="const int16_t *p, ptrdiff_t step, size_t vl",
         group="int16 memory",
         body="return __riscv_vlse16_v_i16m1(p, step * (ptrdiff_t)sizeof(int16_t), vl);",
         doc="Strided load: lane i = p[i * step]. `step` is in ELEMENTS (not bytes); it may be negative."),
    dict(name="xd_hload_iq", ret="void", params="const int16_t *p, xd_h *i_out, xd_h *q_out, size_t vl",
         group="int16 memory",
         body="vint16m1x2_t t = __riscv_vlseg2e16_v_i16m1x2(p, vl);\n"
              "  *i_out = __riscv_vget_v_i16m1x2_i16m1(t, 0);\n"
              "  *q_out = __riscv_vget_v_i16m1x2_i16m1(t, 1);",
         doc="De-interleaving load of vl complex samples stored as I,Q,I,Q,... "
             "Lane k of *i_out = p[2k], lane k of *q_out = p[2k+1]. Reads 2*vl int16 values."),
    dict(name="xd_hstore_iq", ret="void", params="int16_t *p, xd_h i_in, xd_h q_in, size_t vl",
         group="int16 memory",
         body="vint16m1x2_t t = __riscv_vset_v_i16m1_i16m1x2(__riscv_vundefined_i16m1x2(), 0, i_in);\n"
              "  t = __riscv_vset_v_i16m1_i16m1x2(t, 1, q_in);\n"
              "  __riscv_vsseg2e16_v_i16m1x2(p, t, vl);",
         doc="Interleaving store: p[2k] = lane k of i_in, p[2k+1] = lane k of q_in, for k < vl."),

    # ---- int16 arithmetic ---------------------------------------------------------------
    dict(name="xd_hdup", ret="xd_h", params="int16_t x, size_t vl", group="int16 arithmetic",
         body="return __riscv_vmv_v_x_i16m1(x, vl);", doc="Broadcasts scalar x into lanes 0..vl-1."),
    dict(name="xd_hadd", ret="xd_h", params="xd_h a, xd_h b, size_t vl", group="int16 arithmetic",
         body="return __riscv_vadd_vv_i16m1(a, b, vl);", doc="Lane-wise a + b, wrapping modulo 2^16."),
    dict(name="xd_hsub", ret="xd_h", params="xd_h a, xd_h b, size_t vl", group="int16 arithmetic",
         body="return __riscv_vsub_vv_i16m1(a, b, vl);", doc="Lane-wise a - b, wrapping modulo 2^16."),
    dict(name="xd_haddsat", ret="xd_h", params="xd_h a, xd_h b, size_t vl", group="int16 arithmetic",
         body="return __riscv_vsadd_vv_i16m1(a, b, vl);",
         doc="Lane-wise saturating a + b: result clamped to [-32768, 32767]."),
    dict(name="xd_hsubsat", ret="xd_h", params="xd_h a, xd_h b, size_t vl", group="int16 arithmetic",
         body="return __riscv_vssub_vv_i16m1(a, b, vl);",
         doc="Lane-wise saturating a - b: result clamped to [-32768, 32767]."),
    dict(name="xd_havg", ret="xd_h", params="xd_h a, xd_h b, size_t vl", group="int16 arithmetic",
         body=f"return __riscv_vaadd_vv_i16m1(a, b, {RNU}, vl);",
         doc="Lane-wise rounded average (a + b + 1) >> 1, computed without intermediate overflow "
             "(arithmetic shift, exact for all int16 inputs)."),
    dict(name="xd_hmul", ret="xd_h", params="xd_h a, xd_h b, size_t vl", group="int16 arithmetic",
         body="return __riscv_vmul_vv_i16m1(a, b, vl);",
         doc="Lane-wise low 16 bits of a * b (wrapping). For fixed-point use xd_hqmul."),
    dict(name="xd_hqmul", ret="xd_h", params="xd_h a, xd_h b, size_t vl", group="int16 arithmetic",
         body=f"return __riscv_vsmul_vv_i16m1(a, b, {RNU}, vl);",
         doc="Q15 fractional multiply with rounding and saturation: "
             "result = sat16((a * b + 16384) >> 15). Only (-32768) * (-32768) saturates (to 32767)."),
    dict(name="xd_hqmulx", ret="xd_h", params="xd_h a, int16_t k, size_t vl", group="int16 arithmetic",
         body=f"return __riscv_vsmul_vx_i16m1(a, k, {RNU}, vl);",
         doc="Q15 multiply of every lane by scalar k: sat16((a * k + 16384) >> 15)."),
    dict(name="xd_hmax", ret="xd_h", params="xd_h a, xd_h b, size_t vl", group="int16 arithmetic",
         body="return __riscv_vmax_vv_i16m1(a, b, vl);", doc="Lane-wise signed maximum."),
    dict(name="xd_hmin", ret="xd_h", params="xd_h a, xd_h b, size_t vl", group="int16 arithmetic",
         body="return __riscv_vmin_vv_i16m1(a, b, vl);", doc="Lane-wise signed minimum."),
    dict(name="xd_habssat", ret="xd_h", params="xd_h a, size_t vl", group="int16 arithmetic",
         body="return __riscv_vmax_vv_i16m1(a, __riscv_vssub_vv_i16m1(__riscv_vmv_v_x_i16m1(0, vl), a, vl), vl);",
         doc="Lane-wise saturating absolute value: |a|, with |-32768| = 32767."),
    dict(name="xd_hsra", ret="xd_h", params="xd_h a, unsigned sh, size_t vl", group="int16 arithmetic",
         body="return __riscv_vsra_vx_i16m1(a, sh, vl);",
         doc="Arithmetic shift right by sh (0..15), truncating toward minus infinity."),
    dict(name="xd_hsrar", ret="xd_h", params="xd_h a, unsigned sh, size_t vl", group="int16 arithmetic",
         body=f"return __riscv_vssra_vx_i16m1(a, sh, {RNU}, vl);",
         doc="Rounding arithmetic shift right: (a + (1 << (sh-1))) >> sh for sh >= 1, no overflow."),
    dict(name="xd_hslide", ret="xd_h", params="xd_h a, size_t k, size_t vl", group="int16 arithmetic",
         body="return __riscv_vslidedown_vx_i16m1(a, k, vl);",
         doc="Lane i of result = lane i+k of a (lanes past the register end read as 0). "
             "Rarely needed: reloading from memory at p+k is usually cheaper."),

    # ---- int16 compare / select / reduce -------------------------------------------------
    dict(name="xd_hgt", ret="xd_hmask", params="xd_h a, xd_h b, size_t vl", group="int16 compare, select, reduce",
         body="return __riscv_vmsgt_vv_i16m1_b16(a, b, vl);", doc="Mask lane i = (a[i] > b[i])."),
    dict(name="xd_hgtx", ret="xd_hmask", params="xd_h a, int16_t x, size_t vl", group="int16 compare, select, reduce",
         body="return __riscv_vmsgt_vx_i16m1_b16(a, x, vl);", doc="Mask lane i = (a[i] > x)."),
    dict(name="xd_heqx", ret="xd_hmask", params="xd_h a, int16_t x, size_t vl", group="int16 compare, select, reduce",
         body="return __riscv_vmseq_vx_i16m1_b16(a, x, vl);", doc="Mask lane i = (a[i] == x)."),
    dict(name="xd_hsel", ret="xd_h", params="xd_hmask m, xd_h if_true, xd_h if_false, size_t vl",
         group="int16 compare, select, reduce",
         body="return __riscv_vmerge_vvm_i16m1(if_false, if_true, m, vl);",
         doc="Lane i = m[i] ? if_true[i] : if_false[i]. Note the argument order: mask first."),
    dict(name="xd_hcount", ret="size_t", params="xd_hmask m, size_t vl", group="int16 compare, select, reduce",
         body="return (size_t)__riscv_vcpop_m_b16(m, vl);", doc="Number of set mask lanes among lanes 0..vl-1."),
    dict(name="xd_hfirst", ret="long", params="xd_hmask m, size_t vl", group="int16 compare, select, reduce",
         body="return __riscv_vfirst_m_b16(m, vl);",
         doc="Index of the lowest set mask lane among 0..vl-1, or -1 if none is set."),
    dict(name="xd_hredmax", ret="int16_t", params="xd_h a, size_t vl", group="int16 compare, select, reduce",
         body="return __riscv_vmv_x_s_i16m1_i16(__riscv_vredmax_vs_i16m1_i16m1(a, __riscv_vmv_v_x_i16m1(-32768, 1), vl));",
         doc="Maximum over lanes 0..vl-1 (returns -32768 if vl == 0)."),
    dict(name="xd_hredsum_w", ret="int32_t", params="xd_h a, size_t vl", group="int16 compare, select, reduce",
         body="return __riscv_vmv_x_s_i32m1_i32(__riscv_vwredsum_vs_i16m1_i32m1(a, __riscv_vmv_v_x_i32m1(0, 1), vl));",
         doc="Sum of lanes 0..vl-1 computed in int32 (widening, exact unless the int32 sum overflows)."),

    # ---- widening int16 -> int32 (xd_w) --------------------------------------------------
    dict(name="xd_wdup", ret="xd_w", params="int32_t x, size_t vl", group="Wide accumulators (xd_w)",
         body="return __riscv_vmv_v_x_i32m2(x, vl);",
         doc="Broadcasts x into lanes 0..vl-1 of a wide accumulator. "
             "To zero an accumulator for a strip-mined loop use xd_wdup(0, xd_hmaxlen())."),
    dict(name="xd_wwiden", ret="xd_w", params="xd_h a, size_t vl", group="Wide accumulators (xd_w)",
         body="return __riscv_vsext_vf2_i32m2(a, vl);", doc="Sign-extends each int16 lane to int32."),
    dict(name="xd_wmul", ret="xd_w", params="xd_h a, xd_h b, size_t vl", group="Wide accumulators (xd_w)",
         body="return __riscv_vwmul_vv_i32m2(a, b, vl);", doc="Lane-wise exact product a * b as int32."),
    dict(name="xd_wmulx", ret="xd_w", params="xd_h a, int16_t k, size_t vl", group="Wide accumulators (xd_w)",
         body="return __riscv_vwmul_vx_i32m2(a, k, vl);", doc="Lane-wise exact product a * k as int32."),
    dict(name="xd_wmac", ret="xd_w", params="xd_w acc, xd_h a, xd_h b, size_t vl", group="Wide accumulators (xd_w)",
         body="return __riscv_vwmacc_vv_i32m2_tu(acc, a, b, vl);",
         doc="acc[i] += a[i] * b[i] (int32, wrapping) for i < vl. Lanes i >= vl keep their old value "
             "(tail-undisturbed), so one accumulator can be reused across a strip-mined loop whose last "
             "iteration has a shorter vl."),
    dict(name="xd_wmacx", ret="xd_w", params="xd_w acc, int16_t k, xd_h a, size_t vl", group="Wide accumulators (xd_w)",
         body="return __riscv_vwmacc_vx_i32m2_tu(acc, k, a, vl);",
         doc="acc[i] += k * a[i] (int32, wrapping) for i < vl; lanes i >= vl unchanged. "
             "Note argument order: scalar coefficient before the vector."),
    dict(name="xd_wadd", ret="xd_w", params="xd_w a, xd_w b, size_t vl", group="Wide accumulators (xd_w)",
         body="return __riscv_vadd_vv_i32m2(a, b, vl);", doc="Lane-wise int32 a + b (wrapping)."),
    dict(name="xd_wsub", ret="xd_w", params="xd_w a, xd_w b, size_t vl", group="Wide accumulators (xd_w)",
         body="return __riscv_vsub_vv_i32m2(a, b, vl);", doc="Lane-wise int32 a - b (wrapping)."),
    dict(name="xd_wsra", ret="xd_w", params="xd_w a, unsigned sh, size_t vl", group="Wide accumulators (xd_w)",
         body="return __riscv_vsra_vx_i32m2(a, sh, vl);", doc="Arithmetic shift right of int32 lanes by sh (0..31)."),
    dict(name="xd_wnarrow", ret="xd_h", params="xd_w a, unsigned sh, size_t vl", group="Wide accumulators (xd_w)",
         body=f"return __riscv_vnclip_wx_i16m1(a, sh, {RNU}, vl);",
         doc="Narrow to int16 with rounding and saturation: sat16((a + (1 << (sh-1))) >> sh) for sh >= 1 "
             "(sh = 0: plain saturation). This is the standard way to turn a Q30 product sum back into Q15 "
             "(sh = 15)."),
    dict(name="xd_wredsum", ret="int32_t", params="xd_w a, size_t vl", group="Wide accumulators (xd_w)",
         body="return __riscv_vmv_x_s_i32m1_i32(__riscv_vredsum_vs_i32m2_i32m1(a, __riscv_vmv_v_x_i32m1(0, 1), vl));",
         doc="Sum of lanes 0..vl-1 (int32, wrapping). After a strip-mined xd_wmac loop, reduce with "
             "vl = xd_hmaxlen() so that lanes filled in earlier, longer iterations are included."),
    dict(name="xd_wload", ret="xd_w", params="const int32_t *p, size_t vl", group="Wide accumulators (xd_w)",
         body="return __riscv_vle32_v_i32m2(p, vl);", doc="Loads vl int32 values into an xd_w (vl from xd_hlen)."),
    dict(name="xd_wstore", ret="void", params="int32_t *p, xd_w v, size_t vl", group="Wide accumulators (xd_w)",
         body="__riscv_vse32_v_i32m2(p, v, vl);", doc="Stores lanes 0..vl-1 of an xd_w as int32 (vl from xd_hlen)."),
    dict(name="xd_wgtx", ret="xd_hmask", params="xd_w a, int32_t x, size_t vl", group="Wide accumulators (xd_w)",
         body="return __riscv_vmsgt_vx_i32m2_b16(a, x, vl);", doc="Mask lane i = (a[i] > x) for wide lanes."),

    # ---- int32 (xd_s) ------------------------------------------------------------------
    dict(name="xd_sload", ret="xd_s", params="const int32_t *p, size_t vl", group="int32 (xd_s)",
         body="return __riscv_vle32_v_i32m1(p, vl);", doc="Loads vl int32 values (vl from xd_slen)."),
    dict(name="xd_sstore", ret="void", params="int32_t *p, xd_s v, size_t vl", group="int32 (xd_s)",
         body="__riscv_vse32_v_i32m1(p, v, vl);", doc="Stores lanes 0..vl-1."),
    dict(name="xd_sdup", ret="xd_s", params="int32_t x, size_t vl", group="int32 (xd_s)",
         body="return __riscv_vmv_v_x_i32m1(x, vl);", doc="Broadcast."),
    dict(name="xd_sadd", ret="xd_s", params="xd_s a, xd_s b, size_t vl", group="int32 (xd_s)",
         body="return __riscv_vadd_vv_i32m1(a, b, vl);", doc="Lane-wise a + b (wrapping)."),
    dict(name="xd_ssub", ret="xd_s", params="xd_s a, xd_s b, size_t vl", group="int32 (xd_s)",
         body="return __riscv_vsub_vv_i32m1(a, b, vl);", doc="Lane-wise a - b (wrapping)."),
    dict(name="xd_smul", ret="xd_s", params="xd_s a, xd_s b, size_t vl", group="int32 (xd_s)",
         body="return __riscv_vmul_vv_i32m1(a, b, vl);", doc="Lane-wise low 32 bits of a * b."),
    dict(name="xd_smulx", ret="xd_s", params="xd_s a, int32_t k, size_t vl", group="int32 (xd_s)",
         body="return __riscv_vmul_vx_i32m1(a, k, vl);", doc="Lane-wise low 32 bits of a * k."),
    dict(name="xd_ssra", ret="xd_s", params="xd_s a, unsigned sh, size_t vl", group="int32 (xd_s)",
         body="return __riscv_vsra_vx_i32m1(a, sh, vl);", doc="Arithmetic shift right."),
    dict(name="xd_smax", ret="xd_s", params="xd_s a, xd_s b, size_t vl", group="int32 (xd_s)",
         body="return __riscv_vmax_vv_i32m1(a, b, vl);", doc="Lane-wise maximum."),
    dict(name="xd_sgt", ret="xd_smask", params="xd_s a, xd_s b, size_t vl", group="int32 (xd_s)",
         body="return __riscv_vmsgt_vv_i32m1_b32(a, b, vl);", doc="Mask lane i = (a[i] > b[i])."),
    dict(name="xd_sgtx", ret="xd_smask", params="xd_s a, int32_t x, size_t vl", group="int32 (xd_s)",
         body="return __riscv_vmsgt_vx_i32m1_b32(a, x, vl);", doc="Mask lane i = (a[i] > x)."),
    dict(name="xd_ssel", ret="xd_s", params="xd_smask m, xd_s if_true, xd_s if_false, size_t vl", group="int32 (xd_s)",
         body="return __riscv_vmerge_vvm_i32m1(if_false, if_true, m, vl);",
         doc="Lane i = m[i] ? if_true[i] : if_false[i]. Mask first, like xd_hsel."),
    dict(name="xd_scount", ret="size_t", params="xd_smask m, size_t vl", group="int32 (xd_s)",
         body="return (size_t)__riscv_vcpop_m_b32(m, vl);", doc="Number of set lanes among 0..vl-1."),
    dict(name="xd_sredsum", ret="int32_t", params="xd_s a, size_t vl", group="int32 (xd_s)",
         body="return __riscv_vmv_x_s_i32m1_i32(__riscv_vredsum_vs_i32m1_i32m1(a, __riscv_vmv_v_x_i32m1(0, 1), vl));",
         doc="Sum of lanes 0..vl-1 (int32, wrapping)."),
    dict(name="xd_sredmax", ret="int32_t", params="xd_s a, size_t vl", group="int32 (xd_s)",
         body="return __riscv_vmv_x_s_i32m1_i32(__riscv_vredmax_vs_i32m1_i32m1(a, __riscv_vmv_v_x_i32m1(INT32_MIN, 1), vl));",
         doc="Maximum over lanes 0..vl-1."),

    # ---- float32 (xd_f) ----------------------------------------------------------------
    dict(name="xd_fload", ret="xd_f", params="const float *p, size_t vl", group="float32 (xd_f)",
         body="return __riscv_vle32_v_f32m1(p, vl);", doc="Loads vl floats (vl from xd_slen)."),
    dict(name="xd_fstore", ret="void", params="float *p, xd_f v, size_t vl", group="float32 (xd_f)",
         body="__riscv_vse32_v_f32m1(p, v, vl);", doc="Stores lanes 0..vl-1."),
    dict(name="xd_fdup", ret="xd_f", params="float x, size_t vl", group="float32 (xd_f)",
         body="return __riscv_vfmv_v_f_f32m1(x, vl);", doc="Broadcast."),
    dict(name="xd_fadd", ret="xd_f", params="xd_f a, xd_f b, size_t vl", group="float32 (xd_f)",
         body="return __riscv_vfadd_vv_f32m1(a, b, vl);", doc="Lane-wise a + b."),
    dict(name="xd_fsub", ret="xd_f", params="xd_f a, xd_f b, size_t vl", group="float32 (xd_f)",
         body="return __riscv_vfsub_vv_f32m1(a, b, vl);", doc="Lane-wise a - b."),
    dict(name="xd_fmul", ret="xd_f", params="xd_f a, xd_f b, size_t vl", group="float32 (xd_f)",
         body="return __riscv_vfmul_vv_f32m1(a, b, vl);", doc="Lane-wise a * b."),
    dict(name="xd_fmulx", ret="xd_f", params="xd_f a, float k, size_t vl", group="float32 (xd_f)",
         body="return __riscv_vfmul_vf_f32m1(a, k, vl);", doc="Lane-wise a * k."),
    dict(name="xd_faddx", ret="xd_f", params="xd_f a, float k, size_t vl", group="float32 (xd_f)",
         body="return __riscv_vfadd_vf_f32m1(a, k, vl);", doc="Lane-wise a + k."),
    dict(name="xd_fmac", ret="xd_f", params="xd_f acc, xd_f a, xd_f b, size_t vl", group="float32 (xd_f)",
         body="return __riscv_vfmacc_vv_f32m1_tu(acc, a, b, vl);",
         doc="Fused acc[i] += a[i] * b[i] for i < vl; lanes i >= vl unchanged (tail-undisturbed)."),
    dict(name="xd_fmacx", ret="xd_f", params="xd_f acc, float k, xd_f a, size_t vl", group="float32 (xd_f)",
         body="return __riscv_vfmacc_vf_f32m1_tu(acc, k, a, vl);",
         doc="Fused acc[i] += k * a[i] for i < vl; lanes i >= vl unchanged."),
    dict(name="xd_fredsum", ret="float", params="xd_f a, size_t vl", group="float32 (xd_f)",
         body="return __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredusum_vs_f32m1_f32m1(a, __riscv_vfmv_v_f_f32m1(0.0f, 1), vl));",
         doc="Sum of lanes 0..vl-1 in an unspecified order (results may differ from a sequential sum "
             "in the last bits)."),
    dict(name="xd_fredmax", ret="float", params="xd_f a, size_t vl", group="float32 (xd_f)",
         body="return __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredmax_vs_f32m1_f32m1(a, __riscv_vfmv_v_f_f32m1(-INFINITY, 1), vl));",
         doc="Maximum over lanes 0..vl-1."),
]

CHAPTERS = {
    "Overview": (
        "XDSP is a vector DSP extension. A program processes arrays in chunks (\"strip-mining\"): each loop "
        "iteration asks the core how many lanes it may use (vl) and every vector operation takes that vl as "
        "its last argument. Lanes >= vl are not written by loads/stores. All XDSP operations are C functions "
        "declared in `xdsp.h`; include it with `#include \"xdsp.h\"`. Only the operations in this manual exist. "
        "The core also runs ordinary scalar C code, which is always allowed (for example loop remainders or "
        "sequential recurrences)."
    ),
    "Data types": "\n".join(f"- `{t}`: {d}" for t, _, d in TYPES) + (
        "\n\nThere are two lane families. The int16 family (xd_h, and xd_w which holds the int32 widening of "
        "each xd_h lane) uses vl from xd_hlen(). The int32/float family (xd_s, xd_f) uses vl from xd_slen(). "
        "Never mix a vl from one family with operations of the other."
    ),
    "Fixed-point conventions": (
        "Q15: an int16 x represents x / 32768. The product of two Q15 numbers is Q30 and fits in int32. "
        "Rounding operations (xd_hqmul, xd_hqmulx, xd_hsrar, xd_wnarrow, xd_havg) use round-to-nearest-up: "
        "add half an LSB, then shift arithmetically. Saturating operations clamp to [-32768, 32767] instead "
        "of wrapping. A typical Q15 dot product or FIR accumulates exact products with xd_wmac in an xd_w and "
        "converts back once with xd_wnarrow(acc, 15, vl)."
    ),
    "Programming example: saturating vector add": (
        "```c\n#include \"xdsp.h\"\n"
        "void add_sat(const int16_t *a, const int16_t *b, int16_t *y, size_t n) {\n"
        "    for (size_t vl; n > 0; n -= vl, a += vl, b += vl, y += vl) {\n"
        "        vl = xd_hlen(n);\n"
        "        xd_h va = xd_hload(a, vl);\n"
        "        xd_h vb = xd_hload(b, vl);\n"
        "        xd_hstore(y, xd_haddsat(va, vb, vl), vl);\n"
        "    }\n}\n```"
    ),
    "Programming example: reduction with a wide accumulator": (
        "```c\n#include \"xdsp.h\"\n"
        "int32_t sum_sq(const int16_t *x, size_t n) {\n"
        "    xd_w acc = xd_wdup(0, xd_hmaxlen());      /* zero ALL lanes once */\n"
        "    for (size_t vl; n > 0; n -= vl, x += vl) {\n"
        "        vl = xd_hlen(n);\n"
        "        xd_h v = xd_hload(x, vl);\n"
        "        acc = xd_wmac(acc, v, v, vl);          /* lanes >= vl keep their partial sums */\n"
        "    }\n"
        "    return xd_wredsum(acc, xd_hmaxlen());       /* reduce every lane */\n}\n```"
    ),
}
