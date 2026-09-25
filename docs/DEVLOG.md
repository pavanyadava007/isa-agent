# Development log

What changed while building the agent, and which observation caused each change. Smoke-test episodes
(qwen3-coder 30B, 2026-09-25) were thrown away after each change; only the final configuration is in
`results/episodes.jsonl`.

## 1. Invented operations need a semantic search, not a spelling match

Observation: with retrieved manual sections, the model wrote `xd_wzero(vl)` to clear an accumulator. No such
operation exists; the manual says to use `xd_wdup(0, xd_hmaxlen())`. The first repair message suggested the
names closest in spelling (`xd_wsra`, `xd_wnarrow`, `xd_wload`, `xd_wwiden`), none of which helps, and the
model kept calling `xd_wzero` for four repair rounds.

Change: `Manual.suggest()` also searches the manual for the words inside the invented name ("w zero"), which
finds the `xd_wdup` entry ("To zero an accumulator..."). Unit test:
`test_semantic_suggestion_beats_spelling_for_invented_name`.

## 2. Repairs repeated the same program

Observation: on `cmul_q15` the model resubmitted an identical wrong program four times (pointer advanced by
`vl` instead of `2*vl` for interleaved data). Every repair turn used the same sampling seed, and the test
output only listed failing elements.

Changes: a new seed per turn; identical resubmissions (ignoring comments and whitespace) are named as such;
the test report lists which input sizes pass and which fail; the feedback states the first wrong element index
and relates it to the lanes per register (8 int16 lanes, 4 int32/float lanes), because "the first loop
iteration is right, the second is wrong" points at pointer bookkeeping.

## 3. A scalar kernel without saturation passed the FIR tests

Observation: the agent "solved" `fir_q15` with a plain scalar loop that never saturates. The random test
inputs could not reach the saturation limits, so the test suite was too weak.

Changes: a hand-made FIR case whose sums reach +-2.08e9 and must clamp to both int16 limits
(`test_fir_golden_saturates_in_the_extra_case`), a sabotage test proving the harness now rejects that kernel,
and "vectorised passes" reported separately from "passes".

## 4. The public ISA is not "known" either

Observation: with no documentation, the model also failed on the real RISC-V Vector intrinsics: it wrote the
pre-1.0 names without the `__riscv_` prefix (`vle16_v_i16m1`, `vsetvl_e16m1`), which clang 18 rejects, even
though the prompt names the prefix. API drift matters as much as the ISA being new.

No change: kept as the `rvv_nodoc` reference condition.

## 5. Fair baselines

Observation: clang's auto-vectoriser at `-O3` beat the hand-written XDSP kernels on simple element-wise loops.
It uses register grouping (LMUL up to 8), which XDSP does not expose (one register = 128 bits).

Change: a second compiler baseline at the same register width (`-mllvm -riscv-v-register-bit-width-lmul=1`).
Both are reported.

## 6. MLIR comparison: specialisation, not the compiler, made the difference

Observation: the ONNX -> MLIR -> LLVM float FIR used 3.6x fewer instructions than clang on the C
reference. The ONNX path knows the 16 taps at compile time, so LLVM unrolls the tap loop completely.

Change: C variants with the same constant taps and LMUL=1 variants of every path. With constant taps, plain C
through clang is as fast as or faster than the MLIR path, and MLIR's affine super-vectoriser was slower than
MLIR loops + LLVM's loop vectoriser in every measured case.

## 7. "Pass" is not the right headline; vectorised passes are

Observation: qwen2.5-coder 7B passed 11 of 42 kernels with no documentation at all. All 11 were plain scalar C,
which the rules allow as a fallback, so they say nothing about using the ISA. For the 30B model the typed agent
passed 26 of 42, of which 22 were vectorised.

Change: the README table reports vectorised passes next to passes; the per-task cost table already used only
vectorised kernels.

## 8. The repair loop hurts the small model

Observation: for qwen2.5-coder 7B the full manual gave 16 of 42 vectorised passes, type-aware retrieval 11, and the
agent with repair rounds only 7. The 30B model gained from the same loop (17 -> 22 vectorised). Likely cause, not yet tested:
the 7B model loses track in long multi-turn contexts full of compiler logs.

No change: kept as a result. A smaller model would need shorter feedback (only the first error, only the
relevant manual entry) or a single-turn repair prompt.
