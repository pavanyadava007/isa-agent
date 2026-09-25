"""DSP kernel task suite: spec text, C signature, scalar C reference, NumPy golden model, test cases.

Integer tasks are checked bit-exactly; float tasks with a relative tolerance. Every task's golden
model is itself checked against the scalar C reference in tests/ (so the spec, the golden model and
the reference agree before any LLM output is judged).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

I16_MIN, I16_MAX = -32768, 32767


def sat16(x):
    return np.clip(x, I16_MIN, I16_MAX)


def rshift_rnu(x, sh):
    """Round-to-nearest-up arithmetic shift on int64 arrays."""
    x = np.asarray(x, dtype=np.int64)
    if sh == 0:
        return x
    return (x + (1 << (sh - 1))) >> sh


@dataclass
class Arg:
    name: str
    ctype: str          # e.g. "const int16_t *", "int16_t", "size_t"
    kind: str           # "in" (array input), "out" (array output), "inout", "scalar"
    dtype: str = ""     # numpy dtype for arrays / scalars


@dataclass
class Task:
    name: str
    ret: str
    args: list[Arg]
    spec: str
    ref_c: str
    golden: Callable[[dict], dict]          # inputs -> {"ret": value?, out arrays}
    gen: Callable[[np.random.Generator, int], dict]
    sizes: list[int] = field(default_factory=lambda: [1, 7, 16, 37, 256])
    bench_n: int = 1024
    tol: float = 0.0                        # 0 = bit-exact
    difficulty: str = "easy"
    extra: Callable[[], list[dict]] | None = None   # hand-made edge cases (e.g. forced saturation)

    @property
    def ref_src(self) -> str:
        return "#include <stdint.h>\n#include <stddef.h>\n#include <math.h>\n\n" + self.ref_c + "\n"

    @property
    def signature(self) -> str:
        params = ", ".join(f"{a.ctype}{'' if a.ctype.endswith('*') else ' '}{a.name}" for a in self.args)
        return f"{self.ret} {self.name}({params})"


def _i16(rng, n, lo=I16_MIN, hi=I16_MAX):
    x = rng.integers(lo, hi + 1, size=n, dtype=np.int64)
    if n >= 4:  # force extreme values into every case
        x[0], x[1] = lo, hi
    return x.astype(np.int16)


# ---------------------------------------------------------------------------------------------
TASKS: list[Task] = []

TASKS.append(Task(
    name="vadd_sat_q15", ret="void",
    args=[Arg("a", "const int16_t *", "in", "int16"), Arg("b", "const int16_t *", "in", "int16"),
          Arg("y", "int16_t *", "out", "int16"), Arg("n", "size_t", "scalar", "size")],
    spec="y[i] = saturate_int16(a[i] + b[i]) for i in [0, n). Saturation clamps to [-32768, 32767].",
    ref_c="""void vadd_sat_q15(const int16_t *a, const int16_t *b, int16_t *y, size_t n) {
    for (size_t i = 0; i < n; i++) {
        int32_t s = (int32_t)a[i] + b[i];
        y[i] = (int16_t)(s > 32767 ? 32767 : (s < -32768 ? -32768 : s));
    }
}""",
    golden=lambda d: {"y": sat16(d["a"].astype(np.int64) + d["b"]).astype(np.int16)},
    gen=lambda r, n: {"a": _i16(r, n), "b": _i16(r, n), "n": n},
))

TASKS.append(Task(
    name="dot_q15", ret="int32_t",
    args=[Arg("a", "const int16_t *", "in", "int16"), Arg("b", "const int16_t *", "in", "int16"),
          Arg("n", "size_t", "scalar", "size")],
    spec="Return the exact sum over i in [0, n) of a[i] * b[i], accumulated in int32. "
         "Inputs are chosen so the int32 sum never overflows. Return 0 for n == 0.",
    ref_c="""int32_t dot_q15(const int16_t *a, const int16_t *b, size_t n) {
    int32_t s = 0;
    for (size_t i = 0; i < n; i++) s += (int32_t)a[i] * b[i];
    return s;
}""",
    golden=lambda d: {"ret": np.int32(np.sum(d["a"].astype(np.int64) * d["b"]))},
    gen=lambda r, n: {"a": _i16(r, n, -2048, 2047), "b": _i16(r, n, -2048, 2047), "n": n},
    sizes=[0, 1, 7, 16, 37, 256],
))

TASKS.append(Task(
    name="scale_q15", ret="void",
    args=[Arg("x", "const int16_t *", "in", "int16"), Arg("k", "int16_t", "scalar", "int16"),
          Arg("y", "int16_t *", "out", "int16"), Arg("n", "size_t", "scalar", "size")],
    spec="Q15 scaling: y[i] = saturate_int16((x[i] * k + 16384) >> 15), with the product computed "
         "exactly in 32 bits and an arithmetic shift.",
    ref_c="""void scale_q15(const int16_t *x, int16_t k, int16_t *y, size_t n) {
    for (size_t i = 0; i < n; i++) {
        int32_t p = ((int32_t)x[i] * k + 16384) >> 15;
        y[i] = (int16_t)(p > 32767 ? 32767 : (p < -32768 ? -32768 : p));
    }
}""",
    golden=lambda d: {"y": sat16(rshift_rnu(d["x"].astype(np.int64) * int(d["k"]), 15)).astype(np.int16)},
    gen=lambda r, n: {"x": _i16(r, n), "k": np.int16(-32768 if n == 37 else r.integers(-32768, 32768)), "n": n},
))

TASKS.append(Task(
    name="window_q15", ret="void",
    args=[Arg("x", "const int16_t *", "in", "int16"), Arg("w", "const int16_t *", "in", "int16"),
          Arg("y", "int16_t *", "out", "int16"), Arg("n", "size_t", "scalar", "size")],
    spec="Apply a Q15 window: y[i] = saturate_int16((x[i] * w[i] + 16384) >> 15).",
    ref_c="""void window_q15(const int16_t *x, const int16_t *w, int16_t *y, size_t n) {
    for (size_t i = 0; i < n; i++) {
        int32_t p = ((int32_t)x[i] * w[i] + 16384) >> 15;
        y[i] = (int16_t)(p > 32767 ? 32767 : (p < -32768 ? -32768 : p));
    }
}""",
    golden=lambda d: {"y": sat16(rshift_rnu(d["x"].astype(np.int64) * d["w"], 15)).astype(np.int16)},
    gen=lambda r, n: {"x": _i16(r, n), "w": _i16(r, n), "n": n},
))

TASKS.append(Task(
    name="fir_q15", ret="void",
    args=[Arg("x", "const int16_t *", "in", "int16"), Arg("h", "const int16_t *", "in", "int16"),
          Arg("y", "int16_t *", "out", "int16"), Arg("n", "size_t", "scalar", "size"),
          Arg("ntaps", "size_t", "scalar", "size")],
    spec="Q15 FIR filter (valid correlation form). x holds n + ntaps - 1 samples. For i in [0, n): "
         "acc = sum over k in [0, ntaps) of x[i + k] * h[k], accumulated exactly in int32 (inputs are "
         "chosen so it never overflows); y[i] = saturate_int16((acc + 16384) >> 15). ntaps >= 1.",
    ref_c="""void fir_q15(const int16_t *x, const int16_t *h, int16_t *y, size_t n, size_t ntaps) {
    for (size_t i = 0; i < n; i++) {
        int32_t acc = 0;
        for (size_t k = 0; k < ntaps; k++) acc += (int32_t)x[i + k] * h[k];
        int32_t r = (acc + 16384) >> 15;
        y[i] = (int16_t)(r > 32767 ? 32767 : (r < -32768 ? -32768 : r));
    }
}""",
    golden=lambda d: {"y": sat16(rshift_rnu(np.array(
        [np.sum(d["x"][i:i + int(d["ntaps"])].astype(np.int64) * d["h"]) for i in range(int(d["n"]))],
        dtype=np.int64), 15)).astype(np.int16)},
    gen=lambda r, n: (lambda nt: {"x": _i16(r, n + nt - 1, -8192, 8191), "h": _i16(r, nt, -8192, 8191),
                                  "n": n, "ntaps": nt})(16 if n >= 512 else int(r.choice([1, 5, 16, 31]))),
    bench_n=512, difficulty="medium",
    extra=lambda: [{"x": np.array([8191] * 25 + [-8192] * 25, dtype=np.int16), "h": np.full(31, 8191, np.int16),
                    "n": 20, "ntaps": 31}],   # sums of +-2.08e9 -> both saturation limits
))

TASKS.append(Task(
    name="cmag2_q15", ret="void",
    args=[Arg("iq", "const int16_t *", "in", "int16"), Arg("p", "int32_t *", "out", "int32"),
          Arg("n", "size_t", "scalar", "size")],
    spec="Power of n complex samples stored interleaved as iq[2k] = I, iq[2k+1] = Q: "
         "p[k] = I*I + Q*Q (exact, int32). Inputs lie in [-32767, 32767] so the result fits in int32.",
    ref_c="""void cmag2_q15(const int16_t *iq, int32_t *p, size_t n) {
    for (size_t k = 0; k < n; k++) {
        int32_t i = iq[2 * k], q = iq[2 * k + 1];
        p[k] = i * i + q * q;
    }
}""",
    golden=lambda d: {"p": (d["iq"][0::2].astype(np.int64) ** 2 + d["iq"][1::2].astype(np.int64) ** 2).astype(np.int32)},
    gen=lambda r, n: {"iq": _i16(r, 2 * n, -32767, 32767), "n": n},
    difficulty="medium",
))

TASKS.append(Task(
    name="cmul_q15", ret="void",
    args=[Arg("a", "const int16_t *", "in", "int16"), Arg("b", "const int16_t *", "in", "int16"),
          Arg("y", "int16_t *", "out", "int16"), Arg("n", "size_t", "scalar", "size")],
    spec="Complex Q15 multiply of n interleaved samples (a[2k] = re, a[2k+1] = im; same for b and y): "
         "re = ar*br - ai*bi, im = ar*bi + ai*br, each computed exactly in int32 (inputs in [-16384, 16383] "
         "so no overflow), then y_re = saturate_int16((re + 16384) >> 15), y_im likewise.",
    ref_c="""void cmul_q15(const int16_t *a, const int16_t *b, int16_t *y, size_t n) {
    for (size_t k = 0; k < n; k++) {
        int32_t ar = a[2*k], ai = a[2*k+1], br = b[2*k], bi = b[2*k+1];
        int32_t re = (ar * br - ai * bi + 16384) >> 15;
        int32_t im = (ar * bi + ai * br + 16384) >> 15;
        y[2*k]   = (int16_t)(re > 32767 ? 32767 : (re < -32768 ? -32768 : re));
        y[2*k+1] = (int16_t)(im > 32767 ? 32767 : (im < -32768 ? -32768 : im));
    }
}""",
    golden=lambda d: (lambda ar, ai, br, bi: {"y": np.stack([
        sat16(rshift_rnu(ar * br - ai * bi, 15)), sat16(rshift_rnu(ar * bi + ai * br, 15))], axis=1
    ).reshape(-1).astype(np.int16)})(*(v.astype(np.int64) for v in (d["a"][0::2], d["a"][1::2], d["b"][0::2], d["b"][1::2]))),
    gen=lambda r, n: {"a": _i16(r, 2 * n, -16384, 16383), "b": _i16(r, 2 * n, -16384, 16383), "n": n},
    difficulty="medium",
))

TASKS.append(Task(
    name="argmax_abs_q15", ret="size_t",
    args=[Arg("x", "const int16_t *", "in", "int16"), Arg("n", "size_t", "scalar", "size")],
    spec="Return the index of the element with the largest saturating magnitude m[i] = min(|x[i]|, 32767) "
         "(so |-32768| counts as 32767). On ties return the smallest index. n >= 1.",
    ref_c="""size_t argmax_abs_q15(const int16_t *x, size_t n) {
    size_t best = 0; int32_t bm = -1;
    for (size_t i = 0; i < n; i++) {
        int32_t m = x[i] < 0 ? -(int32_t)x[i] : x[i];
        if (m > 32767) m = 32767;
        if (m > bm) { bm = m; best = i; }
    }
    return best;
}""",
    golden=lambda d: {"ret": np.uint64(int(np.argmax(np.minimum(np.abs(d["x"].astype(np.int64)), 32767))))},
    gen=lambda r, n: (lambda x: {"x": x, "n": n})(
        np.where(r.random(n) < 0.3, r.choice([32767, -32767, -32768], size=n), r.integers(-30000, 30000, size=n)).astype(np.int16)
        if n >= 16 else _i16(r, n)),
    sizes=[1, 7, 16, 37, 256], difficulty="hard",
))

TASKS.append(Task(
    name="count_above_i32", ret="size_t",
    args=[Arg("x", "const int32_t *", "in", "int32"), Arg("thr", "int32_t", "scalar", "int32"),
          Arg("n", "size_t", "scalar", "size")],
    spec="Return the number of i in [0, n) with x[i] > thr (signed comparison).",
    ref_c="""size_t count_above_i32(const int32_t *x, int32_t thr, size_t n) {
    size_t c = 0;
    for (size_t i = 0; i < n; i++) c += (x[i] > thr);
    return c;
}""",
    golden=lambda d: {"ret": np.uint64(int(np.sum(d["x"] > d["thr"])))},
    gen=lambda r, n: {"x": r.integers(-2**31, 2**31, size=n, dtype=np.int64).astype(np.int32),
                      "thr": np.int32(r.integers(-2**20, 2**20)), "n": n},
    sizes=[0, 1, 7, 16, 37, 256],
))

TASKS.append(Task(
    name="decim2_q15", ret="void",
    args=[Arg("x", "const int16_t *", "in", "int16"), Arg("y", "int16_t *", "out", "int16"),
          Arg("n", "size_t", "scalar", "size")],
    spec="Decimate by two with averaging: x holds 2n samples; y[i] = (x[2i] + x[2i+1] + 1) >> 1 "
         "computed without overflow (arithmetic shift) for i in [0, n).",
    ref_c="""void decim2_q15(const int16_t *x, int16_t *y, size_t n) {
    for (size_t i = 0; i < n; i++) y[i] = (int16_t)(((int32_t)x[2*i] + x[2*i+1] + 1) >> 1);
}""",
    golden=lambda d: {"y": ((d["x"][0::2].astype(np.int64) + d["x"][1::2] + 1) >> 1).astype(np.int16)},
    gen=lambda r, n: {"x": _i16(r, 2 * n), "n": n},
))

TASKS.append(Task(
    name="cfar_ca_i32", ret="void",
    args=[Arg("p", "const int32_t *", "in", "int32"), Arg("det", "uint8_t *", "out", "uint8"),
          Arg("n", "size_t", "scalar", "size"), Arg("guard", "size_t", "scalar", "size"),
          Arg("train", "size_t", "scalar", "size"), Arg("alpha", "int32_t", "scalar", "int32")],
    spec="Cell-averaging CFAR detector over a power profile p[0..n). Let w = guard + train. For a cell i "
         "with w <= i < n - w: S = sum of p[i-w .. i-guard-1] plus sum of p[i+guard+1 .. i+w] (2*train "
         "training cells), and det[i] = 1 if p[i] * (2*train) > alpha * S, else 0. For cells closer than w "
         "to either edge det[i] = 0. All products fit in int32 (p < 65536, train <= 16, alpha <= 8). "
         "train >= 1. Write all n entries of det.",
    ref_c="""void cfar_ca_i32(const int32_t *p, uint8_t *det, size_t n, size_t guard, size_t train, int32_t alpha) {
    size_t w = guard + train;
    for (size_t i = 0; i < n; i++) {
        det[i] = 0;
        if (i < w || i + w >= n) continue;
        int32_t s = 0;
        for (size_t j = i - w; j < i - guard; j++) s += p[j];
        for (size_t j = i + guard + 1; j <= i + w; j++) s += p[j];
        det[i] = (uint8_t)(p[i] * (int32_t)(2 * train) > alpha * s);
    }
}""",
    golden=lambda d: _cfar_golden(d),
    gen=lambda r, n: _cfar_gen(r, n),
    sizes=[1, 7, 16, 37, 256], bench_n=512, difficulty="hard",
))


def _cfar_golden(d):
    p = d["p"].astype(np.int64)
    n, g, t, a = int(d["n"]), int(d["guard"]), int(d["train"]), int(d["alpha"])
    w = g + t
    det = np.zeros(n, dtype=np.uint8)
    for i in range(n):
        if i < w or i + w >= n:
            continue
        s = p[i - w:i - g].sum() + p[i + g + 1:i + w + 1].sum()
        det[i] = 1 if p[i] * 2 * t > a * s else 0
    return {"det": det}


def _cfar_gen(r, n):
    p = r.integers(0, 4000, size=n)
    for _ in range(max(1, n // 20)):  # a few targets
        p[r.integers(0, n)] = r.integers(20000, 65535)
    g, t = (2, 8) if n >= 512 else (int(r.choice([0, 1, 2])), int(r.choice([1, 4, 8])))
    return {"p": p.astype(np.int32), "n": n, "guard": g, "train": t, "alpha": np.int32(int(r.choice([2, 3, 5])))}


TASKS.append(Task(
    name="axpy_f32", ret="void",
    args=[Arg("a", "float", "scalar", "float32"), Arg("x", "const float *", "in", "float32"),
          Arg("y", "float *", "inout", "float32"), Arg("n", "size_t", "scalar", "size")],
    spec="y[i] = y[i] + a * x[i] for i in [0, n) (float32; fused multiply-add is allowed).",
    ref_c="""void axpy_f32(float a, const float *x, float *y, size_t n) {
    for (size_t i = 0; i < n; i++) y[i] = y[i] + a * x[i];
}""",
    golden=lambda d: {"y": (d["y"].astype(np.float64) + float(d["a"]) * d["x"].astype(np.float64)).astype(np.float32)},
    gen=lambda r, n: {"a": np.float32(r.normal()), "x": r.normal(size=n).astype(np.float32),
                      "y": r.normal(size=n).astype(np.float32), "n": n},
    tol=1e-5,
))

TASKS.append(Task(
    name="rms_f32", ret="float",
    args=[Arg("x", "const float *", "in", "float32"), Arg("n", "size_t", "scalar", "size")],
    spec="Return sqrtf(sum(x[i]^2) / n) for n >= 1 (float32; summation order is free).",
    ref_c="""float rms_f32(const float *x, size_t n) {
    float s = 0.0f;
    for (size_t i = 0; i < n; i++) s += x[i] * x[i];
    return sqrtf(s / (float)n);
}""",
    golden=lambda d: {"ret": np.float32(np.sqrt(np.mean(d["x"].astype(np.float64) ** 2)))},
    gen=lambda r, n: {"x": r.normal(size=n).astype(np.float32), "n": n},
    tol=1e-4,
))

TASKS.append(Task(
    name="dc_remove_f32", ret="void",
    args=[Arg("x", "const float *", "in", "float32"), Arg("y", "float *", "out", "float32"),
          Arg("n", "size_t", "scalar", "size")],
    spec="Remove the mean: m = sum(x) / n, y[i] = x[i] - m, for n >= 1 (float32; summation order is free).",
    ref_c="""void dc_remove_f32(const float *x, float *y, size_t n) {
    float s = 0.0f;
    for (size_t i = 0; i < n; i++) s += x[i];
    float m = s / (float)n;
    for (size_t i = 0; i < n; i++) y[i] = x[i] - m;
}""",
    golden=lambda d: {"y": (d["x"].astype(np.float64) - np.mean(d["x"].astype(np.float64))).astype(np.float32)},
    gen=lambda r, n: {"x": (r.normal(size=n) + 3.0).astype(np.float32), "n": n},
    tol=1e-4,
))

BY_NAME = {t.name: t for t in TASKS}
