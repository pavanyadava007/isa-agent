"""Run episodes (model x condition x task x seed) and append them to results/episodes.jsonl (resumable)."""

from __future__ import annotations

import argparse
import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from isaagent.agent import CONDITIONS, Runner
from isaagent.llm import LLM
from isaagent.tasks import BY_NAME, TASKS

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "episodes.jsonl"
_lock = threading.Lock()


def key(model: str, cond: str, task: str, seed: int) -> str:
    return f"{model}|{cond}|{task}|{seed}"


def done_keys(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {key(e["model"], e["condition"], e["task"], e["seed"]) for e in map(json.loads, path.read_text().splitlines())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--conditions", default=",".join(CONDITIONS))
    ap.add_argument("--tasks", default="all")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    out = Path(a.out)
    out.parent.mkdir(exist_ok=True)
    tasks = TASKS if a.tasks == "all" else [BY_NAME[t] for t in a.tasks.split(",")]
    conds = a.conditions.split(",")
    runner = Runner(LLM(a.model))
    have = done_keys(out)
    jobs = [(c, t, s) for c in conds for t in tasks for s in range(a.seeds) if key(a.model, c, t.name, s) not in have]
    print(f"{len(jobs)} episodes to run ({len(have)} already done)", flush=True)
    with ThreadPoolExecutor(a.workers) as ex:
        futs = {ex.submit(runner.run, t, c, s): (c, t.name, s) for c, t, s in jobs}
        for i, f in enumerate(as_completed(futs), 1):
            c, tn, s = futs[f]
            try:
                ep = f.result()
            except Exception as e:  # keep going; the episode is re-run on the next invocation
                print(f"[{i}/{len(jobs)}] ERROR {c} {tn} {s}: {e!r}", flush=True)
                continue
            with _lock, out.open("a") as fh:
                fh.write(json.dumps(ep.to_json()) + "\n")
            print(f"[{i}/{len(jobs)}] {a.model} {c:11s} {tn:16s} s{s} pass={ep.passed} first={ep.first_pass} "
                  f"att={len([x for x in ep.attempts if x.kind != 'search'])} srch={len(ep.searches)} insns={ep.insns} "
                  f"{ep.seconds:.0f}s", flush=True)


if __name__ == "__main__":
    main()
