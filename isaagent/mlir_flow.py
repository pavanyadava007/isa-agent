"""ONNX -> MLIR (linalg on memrefs) -> mlir-opt lowering -> LLVM IR -> clang RV64GCV -> QEMU.

A deliberately small importer (1-D Conv as FIR, elementwise Mul/Add) so every step is visible. It is used
to compare compiler-generated code (two MLIR lowering pipelines) with hand-written and agent-written XDSP
kernels on the same instruction-count metric, and to check numerics against onnxruntime.
"""

from __future__ import annotations

import hashlib
import json

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

from isaagent.toolbox import CC_BASE, LINK, QEMU, ROOT, WORK, dexec

NTAPS = 16

# Two lowering pipelines for the same linalg input.
PIPELINES = {
    # linalg -> scalar loops; vectorisation is left to LLVM's loop vectoriser at -O3
    "mlir_loops+llvm_O3": (
        "-convert-linalg-to-loops -lower-affine -convert-scf-to-cf -expand-strided-metadata "
        "-convert-math-to-llvm -convert-arith-to-llvm -finalize-memref-to-llvm -convert-func-to-llvm "
        "-convert-cf-to-llvm -reconcile-unrealized-casts"),
    # linalg -> affine loops -> MLIR affine super-vectoriser (4 x f32 = one 128-bit register) -> LLVM
    "mlir_affine_vec4": (
        "-convert-linalg-to-affine-loops -affine-super-vectorize=virtual-vector-size=4 -canonicalize "
        "-lower-affine -convert-vector-to-scf -convert-scf-to-cf -expand-strided-metadata "
        "-convert-vector-to-llvm -convert-math-to-llvm -convert-arith-to-llvm -finalize-memref-to-llvm "
        "-convert-func-to-llvm -convert-cf-to-llvm -reconcile-unrealized-casts"),
}


# ---------------------------------------------------------------------------------------------
# ONNX models

def fir_model(seed: int = 0) -> onnx.ModelProto:
    rng = np.random.default_rng(seed)
    w = (rng.normal(size=(1, 1, NTAPS)) / NTAPS).astype(np.float32)
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 1, "L"])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 1, "N"])
    node = helper.make_node("Conv", ["x", "w"], ["y"], kernel_shape=[NTAPS], strides=[1], pads=[0, 0])
    g = helper.make_graph([node], "fir_f32", [x], [y], [numpy_helper.from_array(w, "w")])
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)], ir_version=9)
    onnx.checker.check_model(m)
    return m


def axpy_model(a: float = 0.75) -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, ["N"])
    yin = helper.make_tensor_value_info("yin", TensorProto.FLOAT, ["N"])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, ["N"])
    nodes = [helper.make_node("Mul", ["a", "x"], ["ax"]), helper.make_node("Add", ["yin", "ax"], ["y"])]
    g = helper.make_graph(nodes, "axpy_f32", [x, yin], [y], [numpy_helper.from_array(np.array(a, np.float32), "a")])
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)], ir_version=9)
    onnx.checker.check_model(m)
    return m


# ---------------------------------------------------------------------------------------------
# Importer: ONNX -> MLIR text

def _f(v: float) -> str:
    return f"{float(np.float32(v)):.9e}"


def import_onnx(m: onnx.ModelProto) -> tuple[str, str]:
    """Returns (mlir_text, c_signature_of_ciface). Supports the two graph shapes above."""
    ops = [n.op_type for n in m.graph.node]
    inits = {t.name: numpy_helper.to_array(t) for t in m.graph.initializer}
    name = m.graph.name
    if ops == ["Conv"]:
        w = inits[m.graph.node[0].input[1]].reshape(-1)
        k = len(w)
        dense = ", ".join(_f(v) for v in w)
        mlir = f"""
memref.global "private" constant @taps : memref<{k}xf32> = dense<[{dense}]>
func.func @{name}(%x: memref<?xf32>, %y: memref<?xf32>) attributes {{llvm.emit_c_interface}} {{
  %w = memref.get_global @taps : memref<{k}xf32>
  %zero = arith.constant 0.0 : f32
  linalg.fill ins(%zero : f32) outs(%y : memref<?xf32>)
  linalg.conv_1d ins(%x, %w : memref<?xf32>, memref<{k}xf32>) outs(%y : memref<?xf32>)
  return
}}"""
        return mlir, f"void _mlir_ciface_{name}(Desc1 *x, Desc1 *y)"
    if ops == ["Mul", "Add"]:
        a = float(inits[m.graph.node[0].input[0]])
        mlir = f"""
#id = affine_map<(i) -> (i)>
func.func @{name}(%x: memref<?xf32>, %yin: memref<?xf32>, %y: memref<?xf32>) attributes {{llvm.emit_c_interface}} {{
  %a = arith.constant {_f(a)} : f32
  linalg.generic {{indexing_maps = [#id, #id, #id], iterator_types = ["parallel"]}}
      ins(%x, %yin : memref<?xf32>, memref<?xf32>) outs(%y : memref<?xf32>) {{
    ^bb0(%xv: f32, %yv: f32, %o: f32):
      %p = arith.mulf %a, %xv : f32
      %s = arith.addf %yv, %p : f32
      linalg.yield %s : f32
  }}
  return
}}"""
        return mlir, f"void _mlir_ciface_{name}(Desc1 *x, Desc1 *yin, Desc1 *y)"
    raise NotImplementedError(ops)


# ---------------------------------------------------------------------------------------------

def _harness(name: str, sig: str, arrays: dict[str, np.ndarray], outs: list[str], n_out: int, bench: bool) -> str:
    L = ["#include <stdio.h>", "#include <stdint.h>", "#include <math.h>",
         "typedef struct { float *alloc; float *align; int64_t off; int64_t size[1]; int64_t stride[1]; } Desc1;",
         f"{sig};"]
    for k, v in arrays.items():
        L.append(f"static float {k}[{len(v)}] = {{{', '.join(float(np.float32(t)).hex() + 'f' for t in v)}}};")
    for o in outs:
        L.append(f"static float {o}[{n_out}];")
    L.append("int main(void) {")
    for k, v in arrays.items():
        L.append(f"  Desc1 d_{k} = {{{k}, {k}, 0, {{{len(v)}}}, {{1}}}};")
    for o in outs:
        L.append(f"  Desc1 d_{o} = {{{o}, {o}, 0, {{{n_out}}}, {{1}}}};")
    call_args = ", ".join(f"&d_{k}" for k in list(arrays) + outs)
    L.append(f"  _mlir_ciface_{name}({call_args});")
    if not bench:
        L.append(f"  for (int i = 0; i < {n_out}; i++) printf(\"%a\\n\", (double){outs[0]}[i]);")
    L += ["  return 0;", "}"]
    return "\n".join(L) + "\n"


def run_model(m: onnx.ModelProto, feeds: dict[str, np.ndarray], pipeline: str, lmul1: bool = False) -> dict:
    """Compile the model through MLIR with `pipeline`, run on QEMU, compare with onnxruntime."""
    name = m.graph.name
    mlir, sig = import_onnx(m)
    h = hashlib.sha1((mlir + pipeline + str(lmul1) + json.dumps({k: v.shape for k, v in feeds.items()}, default=str)).encode()).hexdigest()[:16]
    d = WORK / f"mlir_{h}"
    d.mkdir(parents=True, exist_ok=True)
    (d / "model.mlir").write_text(mlir)
    ref = ort.InferenceSession(m.SerializeToString(), providers=["CPUExecutionProvider"]).run(None, feeds)[0].reshape(-1)
    flat = {k: v.reshape(-1) for k, v in feeds.items()}
    (d / "test.c").write_text(_harness(name, sig, flat, ["out"], len(ref), bench=False))
    (d / "bench.c").write_text(_harness(name, sig, flat, ["out"], len(ref), bench=True))
    rel = f".work/mlir_{h}"
    cc = f"{CC_BASE} -march=rv64gcv -O3 -mllvm -riscv-v-vector-bits-min=128"
    if lmul1:
        cc += " -mllvm -riscv-v-register-bit-width-lmul=1"
    r = dexec(f"cd {rel} && mlir-opt model.mlir {PIPELINES[pipeline]} -o lowered.mlir 2>&1 && "
              f"mlir-translate --mlir-to-llvmir lowered.mlir -o model.ll 2>&1 && "
              f"{cc} -Wno-override-module -c model.ll -o model.o 2>&1 && "
              f"{cc} {LINK} model.o test.c -o test -lm 2>&1 && {cc} {LINK} model.o bench.c -o bench -lm 2>&1 && "
              f"timeout 20 {QEMU} ./test", timeout=120)
    if r.returncode != 0:
        return {"ok": False, "log": (r.stdout + r.stderr)[-2000:]}
    got = np.array([float.fromhex(t) for t in r.stdout.split()], dtype=np.float64)
    err = float(np.max(np.abs(got - ref) / (1 + np.abs(ref)))) if len(got) == len(ref) else float("inf")
    # instruction count of the generated function (+ its ciface wrapper)
    r2 = dexec(f"cd {rel} && llvm-nm -S --defined-only bench | grep -E ' {name}$| _mlir_ciface_{name}$'")
    ranges = []
    for ln in r2.stdout.splitlines():
        p = ln.split()
        if len(p) == 4:
            ranges.append((int(p[0], 16), int(p[0], 16) + int(p[1], 16)))
    r3 = dexec(f"cd {rel} && {QEMU} -one-insn-per-tb -d nochain,exec -D trace.log ./bench >/dev/null 2>&1; "
               "grep -o '^Trace [0-9]*: 0x[0-9a-f]* \\[[0-9a-f]*/[0-9a-f]*' trace.log | sed 's#.*/##'; rm -f trace.log", timeout=120)
    insns = sum(1 for t in r3.stdout.split() if any(lo <= int(t, 16) < hi for lo, hi in ranges))
    vec = dexec(f"cd {rel} && llvm-objdump -d model.o | grep -cE '\\bv(f?(add|mul|macc|madd|le|se)|setvli)'").stdout.strip()
    return {"ok": err <= 1e-5, "max_rel_err_vs_onnxruntime": err, "insns": insns, "vector_insn_sites": int(vec or 0),
            "n_out": len(ref)}


FIR_F32_REF = """void fir_f32(const float *x, const float *h, float *y, size_t n, size_t ntaps) {
    for (size_t i = 0; i < n; i++) {
        float acc = 0.0f;
        for (size_t k = 0; k < ntaps; k++) acc += x[i + k] * h[k];
        y[i] = acc;
    }
}"""

FIR_F32_XDSP = """#include "xdsp.h"
void fir_f32(const float *x, const float *h, float *y, size_t n, size_t ntaps) {
    for (size_t vl; n > 0; n -= vl, x += vl, y += vl) {
        vl = xd_slen(n);
        xd_f acc = xd_fmulx(xd_fload(x, vl), h[0], vl);
        for (size_t k = 1; k < ntaps; k++) acc = xd_fmacx(acc, h[k], xd_fload(x + k, vl), vl);
        xd_fstore(y, acc, vl);
    }
}"""


def fir_f32_task():
    from isaagent.tasks import Arg, Task
    w = fir_model().graph.initializer[0]
    taps = numpy_helper.to_array(w).reshape(-1)
    return Task(
        name="fir_f32", ret="void",
        args=[Arg("x", "const float *", "in", "float32"), Arg("h", "const float *", "in", "float32"),
              Arg("y", "float *", "out", "float32"), Arg("n", "size_t", "scalar", "size"),
              Arg("ntaps", "size_t", "scalar", "size")],
        spec="float FIR, valid correlation form", ref_c=FIR_F32_REF,
        golden=lambda d: {"y": np.array([np.dot(d["x"][i:i + int(d["ntaps"])].astype(np.float64), d["h"])
                                         for i in range(int(d["n"]))], dtype=np.float32)},
        gen=lambda r, n: {"x": r.normal(size=n + NTAPS - 1).astype(np.float32), "h": taps, "n": n, "ntaps": NTAPS},
        bench_n=1024, tol=1e-5)


def c_baselines() -> dict:
    from isaagent.toolbox import evaluate
    t = fir_f32_task()
    out = {}
    for mode in ("scalar", "autovec", "autovec_m1"):
        r = evaluate(t.ref_src, t, mode=mode)
        out[f"fir_f32/c_{mode}"] = {"ok": r.passed, "insns": r.insns}
    r = evaluate(FIR_F32_XDSP, t, mode="xdsp")
    out["fir_f32/expert_xdsp"] = {"ok": r.passed, "insns": r.insns}
    # same C, but the 16 taps are compile-time constants (what the ONNX->MLIR path gets for free)
    taps = numpy_helper.to_array(fir_model().graph.initializer[0]).reshape(-1)
    const = ("static const float H[16] = {" + ", ".join(float(v).hex() + "f" for v in taps) + "};\n" +
             FIR_F32_REF.replace("(size_t k = 0; k < ntaps; k++) acc += x[i + k] * h[k]",
                                 "(size_t k = 0; k < 16; k++) acc += x[i + k] * H[k]"))
    for mode in ("autovec", "autovec_m1"):
        r = evaluate(t.ref_src.replace(FIR_F32_REF, const), t, mode=mode)
        out[f"fir_f32/c_const_taps_{mode}"] = {"ok": r.passed, "insns": r.insns}
    return out


def main() -> None:
    rng = np.random.default_rng(1)
    out = {}
    n = 1024
    fir = fir_model()
    x = rng.normal(size=(1, 1, n + NTAPS - 1)).astype(np.float32)
    ax = axpy_model()
    xv, yv = rng.normal(size=n).astype(np.float32), rng.normal(size=n).astype(np.float32)
    for pipe in PIPELINES:
        for l1 in (False, True):
            tag = pipe + ("@lmul1" if l1 else "")
            out[f"fir_f32/{tag}"] = run_model(fir, {"x": x}, pipe, l1)
            out[f"axpy_f32/{tag}"] = run_model(ax, {"x": xv, "yin": yv}, pipe, l1)
            print(tag, out[f"fir_f32/{tag}"], out[f"axpy_f32/{tag}"], flush=True)
    out.update(c_baselines())
    print(json.dumps({k: v.get("insns") for k, v in out.items()}, indent=1))
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "mlir_flow.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
