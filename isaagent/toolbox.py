"""Toolchain in the loop: compile with clang (LLVM 18) for RV64GCV, run on QEMU, check, count instructions.

All tools run inside the `isa-agent-tools` Docker image through one long-lived container, so each
call costs a `docker exec` (~0.1 s) instead of a container start.
"""

from __future__ import annotations

import hashlib
import re
import shutil
import subprocess
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from isaagent.tasks import Task

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / ".work"
IMAGE = "isa-agent-tools:latest"
CONTAINER = "isa-tools"
VLEN = 128  # reference core: 8 int16 lanes, 4 int32/float lanes per register

CC_BASE = ("clang --target=riscv64-linux-gnu --sysroot=/usr/riscv64-linux-gnu "
           "--gcc-install-dir=/usr/lib/gcc-cross/riscv64-linux-gnu/13")
LINK = "-fuse-ld=lld -static"
QEMU = f"qemu-riscv64 -cpu rv64,v=true,vlen={VLEN},vext_spec=v1.0"

FORBIDDEN = [(re.compile(r"__riscv_|riscv_vector\.h"), "raw RISC-V vector intrinsics are not part of XDSP"),
             (re.compile(r"\basm\b|__asm__"), "inline assembly is not allowed"),
             (re.compile(r"\bmain\s*\("), "do not define main()")]


def ensure_container() -> None:
    running = subprocess.run(["docker", "ps", "-q", "-f", f"name=^{CONTAINER}$"], capture_output=True, text=True).stdout.strip()
    if running:
        return
    subprocess.run(["docker", "rm", "-f", CONTAINER], capture_output=True)
    subprocess.run(["docker", "run", "-d", "--name", CONTAINER, "-v", f"{ROOT}:/work", "-w", "/work",
                    IMAGE, "sleep", "infinity"], check=True, capture_output=True)


def dexec(cmd: str, timeout: int = 60) -> subprocess.CompletedProcess:
    ensure_container()
    try:
        return subprocess.run(["docker", "exec", CONTAINER, "bash", "-lc", cmd],
                              capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired as e:
        return subprocess.CompletedProcess(e.cmd, 124, "", "TIMEOUT")


# ---------------------------------------------------------------------------------------------
# C literal emission

def _c_scalar_type(a_dtype: str) -> str:
    return {"int16": "int16_t", "int32": "int32_t", "uint8": "uint8_t", "float32": "float",
            "size": "size_t", "uint64": "size_t"}[a_dtype]


def _lit(v, dtype: str) -> str:
    if dtype == "float32":
        return float(np.float32(v)).hex() + "f"
    if dtype == "int32" and int(v) == -2**31:
        return "INT32_MIN"
    return str(int(v))


def _arr(name: str, arr: np.ndarray, dtype: str, const: bool = True) -> str:
    vals = ", ".join(_lit(v, dtype) for v in arr) if len(arr) else "0"
    q = "static const " if const else "static "
    return f"{q}{_c_scalar_type(dtype)} {name}[{max(len(arr), 1)}] = {{{vals}}};"


SENTINEL = {"int16": "0x5A5A", "int32": "0x5A5A5A5A", "uint8": "0xA5", "float32": "1234.5f"}


def make_cases(task: Task, seed: int = 0) -> list[dict]:
    rng = np.random.default_rng(seed)
    cases = []
    for n in task.sizes:
        d = task.gen(rng, n)
        d["_expect"] = task.golden(d)
        cases.append(d)
    for d in (task.extra() if task.extra else []):
        d["_expect"] = task.golden(d)
        cases.append(d)
    return cases


def render_test_main(task: Task, cases: list[dict]) -> str:
    L = ["#include <stdio.h>", "#include <stdint.h>", "#include <stddef.h>", "#include <math.h>", "",
         f"{task.signature};", "static int fails = 0;", ""]
    body = ["int main(void) {"]
    for ci, d in enumerate(cases):
        call_args = []
        for a in task.args:
            v = d.get(a.name)
            if a.kind == "in":
                L.append(_arr(f"c{ci}_{a.name}", np.asarray(v), a.dtype))
                call_args.append(f"c{ci}_{a.name}")
            elif a.kind in ("out", "inout"):
                exp = np.asarray(d["_expect"][a.name])
                ln = len(exp)
                L.append(f"static {_c_scalar_type(a.dtype)} c{ci}_{a.name}[{ln + 8}];")
                L.append(_arr(f"c{ci}_{a.name}_exp", exp, a.dtype))
                if a.kind == "inout":
                    L.append(_arr(f"c{ci}_{a.name}_init", np.asarray(v), a.dtype))
                call_args.append(f"c{ci}_{a.name}")
            else:
                call_args.append(_lit(v, a.dtype if a.dtype != "size" else "size"))
        body.append(f"  {{ int f0 = fails; /* case {ci}: n={d['n']} */")
        for a in task.args:
            if a.kind in ("out", "inout"):
                ln = len(np.asarray(d["_expect"][a.name]))
                body.append(f"    for (int j = 0; j < {ln + 8}; j++) c{ci}_{a.name}[j] = ({_c_scalar_type(a.dtype)}){SENTINEL[a.dtype]};")
                if a.kind == "inout":
                    body.append(f"    for (int j = 0; j < {ln}; j++) c{ci}_{a.name}[j] = c{ci}_{a.name}_init[j];")
        call = f"{task.name}({', '.join(call_args)})"
        if task.ret != "void":
            rtype = "double" if task.ret == "float" else "long long"
            body.append(f"    {rtype} r = ({rtype}){call};")
            exp = d["_expect"]["ret"]
            if task.tol:
                body.append(f"    double e = {float(exp)!r};")
                msg = f"CASE {ci} n={d['n']} FAIL return got %.9g expected %.9g\\n"
                body.append(f"    if (!(fabs(r - e) <= {task.tol} * (1.0 + fabs(e)))) {{ printf(\"{msg}\", r, e); fails++; }}")
            else:
                body.append(f"    long long e = {int(exp)}LL;")
                body.append(f"    if (r != e) {{ printf(\"CASE {ci} n={d['n']} FAIL return got %lld expected %lld\\n\", r, e); fails++; }}")
        else:
            body.append(f"    {call};")
        for a in task.args:
            if a.kind not in ("out", "inout"):
                continue
            ln = len(np.asarray(d["_expect"][a.name]))
            nm = f"c{ci}_{a.name}"
            if task.tol:
                cmp = f"!(fabs((double){nm}[j] - (double){nm}_exp[j]) <= {task.tol} * (1.0 + fabs((double){nm}_exp[j])))"
                fmt, cast = "%.9g", "(double)"
            else:
                cmp = f"{nm}[j] != {nm}_exp[j]"
                fmt, cast = "%lld", "(long long)"
            body.append(f"    for (int j = 0, shown = 0; j < {ln}; j++) if ({cmp}) {{ fails++; if (shown++ < 3) "
                        f"printf(\"CASE {ci} n={d['n']} FAIL {a.name}[%d] got {fmt} expected {fmt}\\n\", j, {cast}{nm}[j], {cast}{nm}_exp[j]); }}")
            body.append(f"    for (int j = {ln}; j < {ln + 8}; j++) if ({nm}[j] != ({_c_scalar_type(a.dtype)}){SENTINEL[a.dtype]}) "
                        f"{{ fails++; printf(\"CASE {ci} n={d['n']} FAIL wrote past the end of {a.name} (index %d)\\n\", j); break; }}")
        body.append(f"    if (fails == f0) printf(\"CASE {ci} n={d['n']} OK\\n\");")
        body.append("  }")
    body += ["  if (fails == 0) printf(\"ALL PASS\\n\");", "  return fails != 0;", "}"]
    return "\n".join(L + [""] + body) + "\n"


def render_bench_main(task: Task, seed: int = 123) -> str:
    rng = np.random.default_rng(seed)
    d = task.gen(rng, task.bench_n)
    L = ["#include <stdint.h>", "#include <stddef.h>", f"{task.signature};", "volatile double sink;"]
    args = []
    for a in task.args:
        v = d.get(a.name)
        if a.kind == "in":
            L.append(_arr(f"b_{a.name}", np.asarray(v), a.dtype))
            args.append(f"b_{a.name}")
        elif a.kind in ("out", "inout"):
            ln = len(task.golden(d)[a.name])
            L.append(f"static {_c_scalar_type(a.dtype)} b_{a.name}[{ln + 8}];")
            args.append(f"b_{a.name}")
        else:
            args.append(_lit(v, a.dtype))
    call = f"{task.name}({', '.join(args)})"
    L.append("int main(void) {")
    L.append(f"  sink = (double){call};" if task.ret != "void" else f"  {call};")
    L += ["  return 0;", "}"]
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------------------------------------

@dataclass
class Result:
    ok_static: bool = True
    compiled: bool = False
    passed: bool = False
    insns: int | None = None
    log: str = ""
    failures: list[str] = field(default_factory=list)
    vectorized: bool = False


def static_check(code: str, xdsp: bool = True) -> list[str]:
    problems = []
    for rx, msg in FORBIDDEN:
        if not xdsp and "__riscv_" in rx.pattern:
            continue
        if rx.search(code):
            problems.append(msg)
    return problems


def _trim(s: str, n: int = 25) -> str:
    lines = [ln for ln in s.splitlines() if ln.strip()]
    return "\n".join(lines[:n])


def evaluate(code: str, task: Task, mode: str = "xdsp", bench: bool = True, seed: int = 0, keep: bool = False) -> Result:
    made: list[Path] = []
    try:
        return _evaluate(code, task, mode, bench, seed, made)
    finally:
        if not keep:
            for d in made:
                shutil.rmtree(d, ignore_errors=True)


def _evaluate(code: str, task: Task, mode: str, bench: bool, seed: int, made: list) -> Result:
    """Compile, test and (if it passes) count instructions; the work directory is removed unless keep=True.
    mode: 'xdsp' (XDSP header, V extension), 'rvv' (raw intrinsics allowed), 'scalar' (no V), 'autovec'."""
    res = Result()
    if mode in ("xdsp", "rvv"):
        probs = static_check(code, xdsp=(mode == "xdsp"))
        if probs:
            res.ok_static = False
            res.log = "Static check failed: " + "; ".join(probs)
            return res
    res.vectorized = bool(re.search(r"\bxd_(h|s|f|w)load|__riscv_vle|__riscv_vlse|__riscv_vlseg", code))
    # unique per call: parallel workers may submit identical code
    h = hashlib.sha1((mode + task.name + code + str(seed)).encode()).hexdigest()[:12] + "_" + uuid.uuid4().hex[:6]
    d = WORK / h
    d.mkdir(parents=True, exist_ok=True)
    made.append(d)
    (d / "cand.c").write_text(code if code.endswith("\n") else code + "\n")
    (d / "test_main.c").write_text(render_test_main(task, make_cases(task, seed)))
    rel = f".work/{h}"
    march = "-march=rv64gc" if mode == "scalar" else "-march=rv64gcv"
    opt = {"autovec": "-O3", "autovec_m1": "-O3 -mllvm -riscv-v-register-bit-width-lmul=1"}.get(mode, "-O2")
    cc = f"{CC_BASE} {march} {opt} -I/work/isa/include -Wall -Wno-unused-function"
    r = dexec(f"cd {rel} && {cc} -c cand.c -o cand.o 2>&1 && {cc} {LINK} cand.o test_main.c -o test -lm 2>&1")
    if r.returncode != 0:
        res.log = "Compiler errors:\n" + _trim(r.stdout + r.stderr)
        return res
    res.compiled = True
    r = dexec(f"cd {rel} && timeout 20 {QEMU} ./test", timeout=40)
    out = (r.stdout + r.stderr).strip()
    if r.returncode == 0 and "ALL PASS" in out:
        res.passed = True
    else:
        if r.returncode == 124 or "TIMEOUT" in out:
            out = "Timed out (infinite loop?)"
        elif r.returncode > 1:
            out = f"Crashed with exit code {r.returncode} (bad memory access or illegal instruction?)\n" + out
        lines = out.splitlines()
        ok_n = [ln.split()[2] for ln in lines if ln.endswith(" OK")]
        bad = [ln for ln in lines if " FAIL " in ln or not ln.startswith("CASE")]
        bad_n = list(dict.fromkeys(ln.split()[2] for ln in bad if ln.startswith("CASE")))
        res.failures = bad[:8]
        summary = f"Passed sizes: {', '.join(ok_n) or 'none'}. Failed sizes: {', '.join(bad_n) or 'n/a'}."
        res.log = "Test failures (" + summary + "):\n" + "\n".join(res.failures)
        return res
    if bench:
        res.insns = count_insns(code, task, rel, cc)
    return res


def count_insns(code: str, task: Task, rel: str, cc: str) -> int | None:
    """Dynamic instruction count of the candidate's own functions for one call on the bench input."""
    (ROOT / rel / "bench_main.c").write_text(render_bench_main(task))
    r = dexec(f"cd {rel} && {cc} {LINK} cand.o bench_main.c -o bench -lm 2>&1 && llvm-nm --defined-only cand.o && echo ---- "
              f"&& llvm-nm -S --defined-only bench")
    if r.returncode != 0 or "----" not in r.stdout:
        return None
    own, linked = r.stdout.split("----", 1)
    names = {ln.split()[-1] for ln in own.splitlines() if len(ln.split()) >= 2 and ln.split()[-2] in "Tt"}
    ranges = []
    for ln in linked.splitlines():
        p = ln.split()
        if len(p) == 4 and p[2] in "Tt" and p[3] in names:
            lo = int(p[0], 16)
            ranges.append((lo, lo + int(p[1], 16)))
    if not ranges:
        return None
    r = dexec(f"cd {rel} && timeout 60 {QEMU} -one-insn-per-tb -d nochain,exec -D trace.log ./bench >/dev/null 2>&1; "
              f"grep -o '^Trace [0-9]*: 0x[0-9a-f]* \\[[0-9a-f]*/[0-9a-f]*' trace.log | sed 's#.*/##' ; rm -f trace.log", timeout=120)
    count = 0
    for pc_hex in r.stdout.split():
        pc = int(pc_hex, 16)
        if any(lo <= pc < hi for lo, hi in ranges):
            count += 1
    return count
