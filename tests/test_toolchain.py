"""Integration tests: need Docker and the isa-agent-tools image (make image). The harness must accept the
references and the hand-written kernels, and reject deliberately broken kernels."""

import shutil
import subprocess
from pathlib import Path

import pytest

from isaagent.tasks import BY_NAME, TASKS
from isaagent.toolbox import IMAGE, evaluate

ROOT = Path(__file__).resolve().parents[1]


def _have_image() -> bool:
    if not shutil.which("docker"):
        return False
    return subprocess.run(["docker", "image", "inspect", IMAGE], capture_output=True).returncode == 0


pytestmark = pytest.mark.skipif(not _have_image(), reason="toolchain image not available")


@pytest.mark.parametrize("task", TASKS, ids=lambda t: t.name)
def test_reference_and_expert_pass(task):
    assert evaluate(task.ref_src, task, mode="scalar", bench=False).passed
    r = evaluate((ROOT / "reference" / "xdsp" / f"{task.name}.c").read_text(), task, mode="xdsp")
    assert r.passed and r.vectorized and r.insns > 0


SABOTAGE = {
    # no saturation: caught only by the forced-saturation case
    "fir_q15": """#include "xdsp.h"
void fir_q15(const int16_t *x, const int16_t *h, int16_t *y, size_t n, size_t ntaps) {
  for (size_t i = 0; i < n; i++) { int32_t s = 0; for (size_t k = 0; k < ntaps; k++) s += (int32_t)x[i+k]*h[k];
  y[i] = (int16_t)((s + 16384) >> 15); } }""",
    # interleaved data advanced by vl instead of 2*vl
    "cmag2_q15": """#include "xdsp.h"
void cmag2_q15(const int16_t *iq, int32_t *p, size_t n) {
  for (size_t vl; n > 0; n -= vl, iq += vl, p += vl) { vl = xd_hlen(n); xd_h i, q; xd_hload_iq(iq, &i, &q, vl);
  xd_wstore(p, xd_wmac(xd_wmul(i, i, vl), q, q, vl), vl); } }""",
    # reduces only the last vl lanes of a strip-mined accumulator
    "dot_q15": """#include "xdsp.h"
int32_t dot_q15(const int16_t *a, const int16_t *b, size_t n) {
  xd_w acc = xd_wdup(0, xd_hmaxlen()); size_t vl = 0;
  for (; n > 0; n -= vl, a += vl, b += vl) { vl = xd_hlen(n); acc = xd_wmac(acc, xd_hload(a, vl), xd_hload(b, vl), vl); }
  return xd_wredsum(acc, vl); }""",
    # writes one element past the end
    "vadd_sat_q15": """#include "xdsp.h"
void vadd_sat_q15(const int16_t *a, const int16_t *b, int16_t *y, size_t n) {
  for (size_t i = 0; i <= n; i++) { int32_t s = (int32_t)a[i] + b[i]; y[i] = (int16_t)(s > 32767 ? 32767 : s < -32768 ? -32768 : s); } }""",
}


@pytest.mark.parametrize("name", sorted(SABOTAGE))
def test_broken_kernels_are_rejected(name):
    r = evaluate(SABOTAGE[name], BY_NAME[name], mode="xdsp", bench=False)
    assert r.compiled and not r.passed


def test_raw_intrinsics_are_rejected_in_xdsp_mode():
    code = '#include <riscv_vector.h>\nint32_t dot_q15(const int16_t *a, const int16_t *b, size_t n){return 0;}'
    assert not evaluate(code, BY_NAME["dot_q15"], mode="xdsp").ok_static
