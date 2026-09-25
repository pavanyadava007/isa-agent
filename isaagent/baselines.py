"""Instruction-count baselines per task: scalar -O2, clang auto-vectorisation (-O3, LMUL<=8 and LMUL=1),
and the hand-written expert XDSP kernel in reference/xdsp. Cached in results/baselines.json."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from isaagent.tasks import BY_NAME, TASKS
from isaagent.toolbox import evaluate

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "results" / "baselines.json"


def compute() -> dict:
    out = {}
    for t in TASKS:
        row = {}
        for mode in ("scalar", "autovec", "autovec_m1"):
            r = evaluate(t.ref_src, t, mode=mode)
            assert r.passed, (t.name, mode, r.log)
            row[mode] = r.insns
        r = evaluate((ROOT / "reference" / "xdsp" / f"{t.name}.c").read_text(), t, mode="xdsp")
        assert r.passed, (t.name, "expert", r.log)
        row["expert_xdsp"] = r.insns
        out[t.name] = row
        print(t.name, row)
    CACHE.parent.mkdir(exist_ok=True)
    CACHE.write_text(json.dumps(out, indent=1))
    return out


@lru_cache(maxsize=1)
def _load() -> dict:
    return json.loads(CACHE.read_text()) if CACHE.exists() else compute()


def baseline_insns(task: str) -> dict:
    assert task in BY_NAME
    return _load()[task]


if __name__ == "__main__":
    compute()
