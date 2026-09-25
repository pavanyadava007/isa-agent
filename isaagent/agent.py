"""Code-generation conditions, from "model alone" to "agent with manual search and toolchain feedback".

Conditions (see README for the rationale):
  rvv_nodoc   - public ISA (RISC-V V intrinsics) the model has seen in training, no documentation
  xdsp_nodoc  - unseen ISA (XDSP), no documentation: measures hallucination
  xdsp_full   - unseen ISA, the whole manual in the prompt (~3.5k tokens here; real manuals are far larger)
  xdsp_rag    - unseen ISA, intro chapters + BM25-retrieved operation entries
  xdsp_agent  - xdsp_rag + SEARCH tool + up to 4 repair rounds driven by compiler, test and doc feedback
  xdsp_opt    - xdsp_agent, then up to 2 optimisation rounds driven by measured instruction counts
"""

from __future__ import annotations

import re
import time
from dataclasses import asdict, dataclass, field

from isaagent.llm import LLM
from isaagent.retrieval import Manual, render
from isaagent.tasks import Task
from isaagent.toolbox import evaluate

CONDITIONS = ["rvv_nodoc", "xdsp_nodoc", "xdsp_full", "xdsp_rag", "xdsp_agent", "xdsp_opt",
              "xdsp_rag_typed", "xdsp_agent_typed", "xdsp_opt_typed"]
# operations every kernel of a given element type needs, whatever the task text says
CORE = {"int16_t": ["xd_hlen", "xd_hmaxlen", "xd_hload", "xd_hstore", "xd_wdup", "xd_wredsum"],
        "int32_t": ["xd_slen", "xd_smaxlen", "xd_sload", "xd_sstore", "xd_sdup", "xd_sredsum"],
        "float": ["xd_slen", "xd_smaxlen", "xd_fload", "xd_fstore", "xd_fdup", "xd_fredsum"]}
INTRO = ["Overview", "Data types", "Fixed-point conventions"]
MAX_REPAIRS = 4
MAX_SEARCHES = 3
MAX_OPT_ROUNDS = 2

SYSTEM = ("You are an embedded DSP software engineer. You write correct, efficient C99 kernels for vector "
          "DSP cores. You follow the documentation exactly and never invent operations.")

_CODE = re.compile(r"```(?:c|C|cpp)?\s*\n(.*?)```", re.DOTALL)
_SEARCH = re.compile(r"^\s*SEARCH:\s*(.+)$", re.MULTILINE)
_UNDECL = re.compile(r"(?:undeclared (?:identifier|function)|unknown type name) '([A-Za-z_]\w*)'")
_XD = re.compile(r"\bxd_\w+")


def _norm(code: str) -> str:
    return re.sub(r"//[^\n]*|/\*.*?\*/|\s+", "", code, flags=re.DOTALL)


def extract_code(text: str) -> str:
    blocks = _CODE.findall(text)
    if blocks:
        return max(blocks, key=len).strip()
    return text.strip()


def task_prompt(task: Task, isa: str) -> str:
    if isa == "rvv":
        rules = ("Target: RISC-V with the Vector extension 1.0 (VLEN=128). Use the standard RVV C intrinsics "
                 "from <riscv_vector.h> (__riscv_ prefix, explicit vl). Start the file with "
                 "#include <riscv_vector.h>, <stdint.h>, <stddef.h> and <math.h> as needed.")
    else:
        rules = ("Target: the XDSP vector DSP. Start the file with #include \"xdsp.h\" (it already includes "
                 "<stdint.h>, <stddef.h> and <math.h>). Use XDSP vector operations for the heavy loops; plain "
                 "scalar C is also allowed. Only operations documented in the XDSP manual exist.")
    return (f"Implement this function:\n\n```c\n{task.signature};\n```\n\nSpecification: {task.spec}\n\n{rules}\n"
            "Do not write main(). Answer with the complete C file in one ```c code block.")


@dataclass
class Attempt:
    kind: str                 # "generate" | "repair" | "search" | "optimise"
    compiled: bool = False
    passed: bool = False
    insns: int | None = None
    feedback: str = ""
    code: str = ""


@dataclass
class Episode:
    task: str
    condition: str
    model: str
    seed: int
    passed: bool = False
    first_pass: bool = False       # passed on the first generated program
    compiled_first: bool = False
    hallucinated_first: list[str] = field(default_factory=list)
    vectorized: bool = False
    insns: int | None = None       # best passing candidate
    insns_first_pass: int | None = None
    attempts: list[Attempt] = field(default_factory=list)
    searches: list[str] = field(default_factory=list)
    prompt_tokens: int = 0
    output_tokens: int = 0
    seconds: float = 0.0
    final_code: str = ""

    def to_json(self) -> dict:
        return asdict(self)


class Runner:
    def __init__(self, llm: LLM, manual: Manual | None = None) -> None:
        self.llm = llm
        self.manual = manual or Manual()

    # ---- prompt construction -------------------------------------------------------------
    def context(self, task: Task, condition: str) -> str:
        if condition in ("rvv_nodoc", "xdsp_nodoc"):
            return ""
        if condition == "xdsp_full":
            chunks = self.manual.full()
        elif condition.endswith("_typed"):
            # structure-aware retrieval: intro + worked examples + core ops for the signature's element types
            core = [op for ty, ops in CORE.items() if ty in task.signature for op in ops]
            chunks = (self.manual.chapters(INTRO) + [c for c in self.manual.chapters() if c["title"].startswith("Programming example")]
                      + [self.manual.lookup(o) for o in dict.fromkeys(core)]
                      + self.manual.bm25(task.name.replace("_", " ") + " " + task.spec, k=8))
        else:
            chunks = self.manual.chapters(INTRO) + self.manual.bm25(task.name.replace("_", " ") + " " + task.spec, k=8)
        return "XDSP manual excerpts:\n\n" + render(chunks) + "\n\n"

    def _chat(self, ep: Episode, messages: list[dict], seed: int) -> str:
        r = self.llm.chat(messages, seed=seed)
        ep.prompt_tokens += r.prompt_tokens
        ep.output_tokens += r.output_tokens
        return r.text

    # ---- one episode -------------------------------------------------------------------
    def run(self, task: Task, condition: str, seed: int) -> Episode:
        t0 = time.time()
        ep = Episode(task.name, condition, self.llm.model, seed)
        isa = "rvv" if condition == "rvv_nodoc" else "xdsp"
        agentic = condition.startswith(("xdsp_agent", "xdsp_opt"))
        user = self.context(task, condition) + task_prompt(task, isa)
        if agentic:
            user += ("\n\nIf you are unsure which XDSP operation to use or what it does exactly, you may first "
                     "reply with one line `SEARCH: <words or operation name>` (at most "
                     f"{MAX_SEARCHES} searches) and you will receive the matching manual entries.")
        messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]
        mode = "rvv" if isa == "rvv" else "xdsp"

        text = self._chat(ep, messages, seed)
        # tool use: documentation search
        while agentic and len(ep.searches) < MAX_SEARCHES:
            m = _SEARCH.search(text)
            if not m or "```" in text:
                break
            q = m.group(1).strip()
            ep.searches.append(q)
            hits = [self.manual.lookup(w) for w in _XD.findall(q)]
            hits = [h for h in hits if h] + self.manual.bm25(q, k=5)
            ep.attempts.append(Attempt("search", feedback=q))
            messages += [{"role": "assistant", "content": text},
                         {"role": "user", "content": "Manual entries:\n\n" + render(hits) + "\n\nNow continue."}]
            text = self._chat(ep, messages, seed)

        best = None
        rounds = MAX_REPAIRS if agentic else 0
        for i in range(rounds + 1):
            code = extract_code(text)
            res = evaluate(code, task, mode=mode)
            att = Attempt("generate" if i == 0 else "repair", res.compiled, res.passed, res.insns, res.log, code)
            ep.attempts.append(att)
            if i == 0:
                ep.compiled_first = res.compiled
                ep.first_pass = res.passed
                ep.hallucinated_first = sorted(set(_UNDECL.findall(res.log)))
            if res.passed:
                best = (code, res)
                break
            if i == rounds:
                break
            messages += [{"role": "assistant", "content": text},
                         {"role": "user", "content": self.feedback(code, res, [a.code for a in ep.attempts[:-1]])}]
            text = self._chat(ep, messages, seed + 1000 * (i + 1))

        if best and condition.startswith("xdsp_opt"):
            best = self.optimise(task, ep, messages, text, best, seed)

        if best:
            code, res = best
            ep.passed, ep.final_code, ep.vectorized, ep.insns = True, code, res.vectorized, res.insns
            ep.insns_first_pass = next(a.insns for a in ep.attempts if a.passed)
        else:
            ep.final_code = ep.attempts[-1].code if ep.attempts else ""
        ep.seconds = time.time() - t0
        return ep

    # ---- feedback built from tools ---------------------------------------------------------
    def feedback(self, code: str, res, previous: list[str] | None = None) -> str:
        parts = [res.log or "The program failed."]
        if previous and _norm(code) in {_norm(c) for c in previous}:
            parts.append("Your program is IDENTICAL to one that already failed. Do not resubmit it; find the bug.")
        idx = [int(m) for m in re.findall(r"FAIL \w+\[(\d+)\]", res.log)]
        if idx:
            j = min(idx)
            parts.append(f"Diagnosis from the test run: the first wrong output element is index {j}; everything "
                         f"before it is correct. On this core one xd_h register holds 8 int16 lanes and one xd_s/xd_f "
                         f"register 4 lanes. If {j} is a multiple of the elements written per loop iteration, the "
                         "first iteration is right and the bug is in how pointers or indices advance between "
                         "iterations (interleaved data advances by 2*vl elements).")
        missing = sorted(set(_UNDECL.findall(res.log)))
        docs = []
        for name in missing:
            if name.startswith("xd_"):
                sug = self.manual.suggest(name)
                parts.append(f"`{name}` is not an XDSP operation." + (f" Closest documented operations: {', '.join(sug)}." if sug else ""))
                docs += [self.manual.lookup(s) for s in sug]
        used = [n for n in dict.fromkeys(_XD.findall(code)) if self.manual.lookup(n)]
        docs += [self.manual.lookup(n) for n in used]
        docs = [d for d in docs if d]
        if docs:
            parts.append("Manual entries for the operations involved (re-check their exact semantics and argument "
                         "order):\n\n" + render(docs))
        parts.append("Fix the program. Answer with the complete corrected C file in one ```c code block.")
        return "\n\n".join(parts)

    def optimise(self, task: Task, ep: Episode, messages: list[dict], text: str, best, seed: int):
        from isaagent.baselines import baseline_insns
        base = baseline_insns(task.name)
        note = "Correct."
        for _ in range(MAX_OPT_ROUNDS):
            code, res = best
            msg = (f"{note} Your best correct version executes {res.insns} instructions inside the function for "
                   f"n = {task.bench_n} on the reference core (scalar C at -O2: {base['scalar']}; compiler "
                   f"auto-vectorisation at the same register width: {base['autovec_m1']}). Make it faster without "
                   "changing any result: fewer instructions per element, e.g. vectorise remaining scalar loops, "
                   "hoist work out of inner loops, or use a more specialised XDSP operation. Answer with the "
                   "complete C file in one ```c code block.")
            messages += [{"role": "assistant", "content": text}, {"role": "user", "content": msg}]
            text = self._chat(ep, messages, seed)
            c2 = extract_code(text)
            r2 = evaluate(c2, task, mode="xdsp")
            ep.attempts.append(Attempt("optimise", r2.compiled, r2.passed, r2.insns, r2.log, c2))
            if r2.passed and r2.insns is not None and res.insns is not None and r2.insns < res.insns:
                best, note = (c2, r2), f"Correct and faster ({r2.insns} instructions)."
            elif r2.passed:
                note = f"Correct but not faster ({r2.insns} instructions)."
            else:
                note = "That version is wrong, so it was discarded:\n" + r2.log + "\n"
        return best
