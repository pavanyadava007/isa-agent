# ISA Agent

**Can an LLM agent write correct, fast DSP kernels for an instruction set it has never seen, using only the
manual and a real toolchain?**

In-house DSP and accelerator cores have instruction sets that exist only in internal documentation. Public
language models have no training data for them. This repository builds a small, fully reproducible version of
that problem and measures how far documentation retrieval, tool feedback and an optimisation loop get,
compared with compilers and hand-written code.

- **Target ISA:** XDSP, a fictional vector DSP with 73 documented operations (Q15 fixed-point multiply with
  rounding and saturation, widening multiply-accumulate, de-interleaving complex loads, masks, reductions, float
  FMA). Each operation wraps one RISC-V Vector 1.0 intrinsic, so programs run **bit-exactly on QEMU**, but the
  names, types and manual are new. Header and manual are generated from one spec:
  [`isaagent/isa_spec.py`](isaagent/isa_spec.py) -> [`isa/include/xdsp.h`](isa/include/xdsp.h),
  [`docs/XDSP_MANUAL.md`](docs/XDSP_MANUAL.md).
- **Tasks:** 14 signal-processing kernels (saturating add, dot product, Q15 scaling and windowing, FIR, complex
  power and multiply, argmax, threshold count, decimation, cell-averaging CFAR, axpy, RMS, DC removal), each with
  a NumPy golden model, a scalar C reference and a hand-written XDSP kernel
  ([`isaagent/tasks.py`](isaagent/tasks.py), [`reference/xdsp/`](reference/xdsp)).
- **Toolchain in the loop:** clang/LLVM 18 for RV64GCV, QEMU 8.2 user mode (VLEN 128), MLIR 18 tools, all in one
  Docker image ([`docker/Dockerfile`](docker/Dockerfile)). Tests check every output element, edge sizes, a
  forced-saturation case and writes past the output buffer. Cost = dynamic instruction count inside the kernel,
  taken from QEMU's execution trace (not cycles).
- **Models:** local only (Ollama on one NVIDIA L4), because a real ISA manual would be confidential.

## Conditions

| Condition | What the model gets |
|---|---|
| Public ISA, no docs | RISC-V Vector intrinsics (public, in training data), task only |
| Unseen ISA, no docs | XDSP, task only: measures invention of operations |
| Full manual | the whole XDSP manual in the prompt (about 3.5k tokens here; real manuals are far larger) |
| Retrieval | intro chapters + 8 operation entries retrieved with BM25 |
| Agent | retrieval + a `SEARCH:` tool + up to 4 repairs driven by compiler errors, test diffs and the manual entries of the operations used |
| Agent + optimisation | the agent, then 2 rounds driven by measured instruction counts vs scalar and compiler baselines |

## Results

<!-- results:start -->
| Condition | qwen3-coder:30b: pass |
|---|---|
| Public ISA (RVV intrinsics), no docs | 2/42 = 5% [1-16] |
| Unseen ISA, no docs | 0/42 = 0% [0-8] |
| Unseen ISA, full manual in prompt | 13/42 = 31% [19-46] |
| Unseen ISA, retrieved manual sections | 0/42 = 0% [0-8] |
| Agent: retrieval + search tool + toolchain repair loop | 19/42 = 45% [31-60] |
| Agent + optimisation loop (instruction counts) | 21/42 = 50% [36-64] |
| Unseen ISA, type-aware retrieval (core ops + examples) | 17/42 = 40% [27-56] |
| Agent with type-aware retrieval | 26/42 = 62% [47-75] |
| Agent with type-aware retrieval + optimisation loop | 24/42 = 57% [42-71] |

Pass = bit-exact on every test case (14 kernels x 3 seeds per cell; Wilson 95% interval in brackets). Full tables, per-task results, the optimisation loop and baselines: [docs/RESULTS.md](docs/RESULTS.md).
<!-- results:end -->

A second model (qwen2.5-coder 7B, 5 conditions) is being evaluated; `results/episodes.jsonl` may contain its partial episodes, which the report excludes until the run is complete.

What went wrong while building the agent, and what changed because of it, is in
[docs/DEVLOG.md](docs/DEVLOG.md): invented operations that spelling-based suggestions could not fix, repair loops
that resubmitted identical code, a scalar FIR without saturation that exposed a gap in the tests, and a model that
also failed on the *public* RISC-V intrinsics because it wrote their pre-1.0 names.

## ONNX -> MLIR -> LLVM comparison

[`isaagent/mlir_flow.py`](isaagent/mlir_flow.py) imports a small ONNX model (a 16-tap float FIR as `Conv`, and
axpy as `Mul` + `Add`) into MLIR `linalg` on memrefs, lowers it with two `mlir-opt` pipelines (linalg -> loops
with LLVM's loop vectoriser, and linalg -> affine -> MLIR's affine super-vectoriser), translates to LLVM IR,
compiles for RV64GCV, runs on QEMU and compares with ONNX Runtime. The same kernels are also compiled from C and
written by hand in XDSP, so every path is measured on one metric. The main lesson was about fairness: the ONNX
path knows the filter taps at compile time, and giving the C version the same constant taps removes most of the
gap. Numbers: [docs/RESULTS.md](docs/RESULTS.md).

## Reproduce

```bash
make venv image          # Python env; Docker image with clang 18, MLIR 18, QEMU, riscv64 sysroot
make test lint           # unit tests + toolchain tests (references pass, sabotaged kernels fail)
make baselines mlir      # scalar / clang / hand-written instruction counts; ONNX -> MLIR flow
make ollama models       # local model server (127.0.0.1 only) + two models
make eval MODEL=qwen3-coder:30b && make eval MODEL=qwen2.5-coder:7b
make report              # docs/RESULTS.md, results/summary.json, docs/figures/, README block
```

`results/episodes.jsonl` holds every episode: prompts' outcomes, every program the model wrote, the compiler and
test feedback it received, token counts and timings. The demo site replays them.

## Limits

- 14 kernels x 3 seeds per condition is small; intervals are wide and are reported.
- Instruction counts are not cycles. A real DSP has pipeline latencies, VLIW slots and memory banks that an
  optimisation loop would need to model or measure on a cycle-accurate simulator.
- XDSP is a renamed RISC-V vector subset. It is new to the model, but its semantics are regular; a real
  proprietary DSP will be harder.
- Local 7B and 30B models only. Larger models would likely do better; the harness does not depend on the model.

## Layout

| Path | Purpose |
|---|---|
| `isaagent/isa_spec.py`, `gen_isa.py` | ISA definition -> C header + manual |
| `isaagent/tasks.py` | kernels, specs, golden models, test generators |
| `isaagent/toolbox.py` | compile / run / check / count instructions in the toolchain container |
| `isaagent/retrieval.py` | BM25 manual search, exact lookup, suggestions for invented names |
| `isaagent/agent.py` | the six conditions and the agent loop |
| `isaagent/mlir_flow.py` | ONNX -> MLIR -> LLVM -> RISC-V path |
| `isaagent/run_eval.py`, `report.py` | experiment runner (resumable) and report generator |
| `scripts/build_site.py` | static demo site with an episode browser |

Author: Pavan Yadav Annappa. MIT licence.
