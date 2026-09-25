"""Build the static demo site (site/) from results: results tables, figure, manual, and an episode browser.

Publish:  python scripts/build_site.py && python -c "from huggingface_hub import HfApi; \
HfApi().upload_folder(folder_path='site', repo_id='pavanyadava07/isa-agent', repo_type='space')"
"""

from __future__ import annotations

import html
import json
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / "site"


def md_table_to_html(md: str) -> str:
    """Tiny markdown renderer for RESULTS.md: headings, paragraphs, tables, inline code and bold."""
    out, rows = [], []

    def inline(s: str) -> str:
        s = html.escape(s)
        s = re.sub(r"`([^`]+)`", r"<code>\1</code>", s)
        return re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", s)

    def flush():
        if rows:
            head, body = rows[0], [r for r in rows[2:]]
            out.append("<div class='tw'><table><thead><tr>" + "".join(f"<th>{inline(c)}</th>" for c in head) + "</tr></thead><tbody>"
                       + "".join("<tr>" + "".join(f"<td>{inline(c)}</td>" for c in r) + "</tr>" for r in body) + "</tbody></table></div>")
            rows.clear()

    for line in md.splitlines():
        if line.startswith("|"):
            rows.append([c.strip() for c in line.strip().strip("|").split("|")])
            continue
        flush()
        if line.startswith("### "):
            out.append(f"<h3>{inline(line[4:])}</h3>")
        elif line.startswith("## "):
            out.append(f"<h2>{inline(line[3:])}</h2>")
        elif line.startswith("# "):
            continue
        elif line.strip():
            out.append(f"<p>{inline(line)}</p>")
    flush()
    return "\n".join(out)


PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>ISA Agent</title>
<style>
:root{--bg:#fbfbfa;--fg:#1c1c1c;--mut:#5c5c5c;--line:#e2e2de;--acc:#2a5db0;--ok:#1d7a3a;--bad:#b3261e;--code:#f2f2ef}
@media (prefers-color-scheme:dark){:root{--bg:#141414;--fg:#e8e8e6;--mut:#a3a3a0;--line:#2e2e2c;--acc:#7aa7ff;--ok:#6fcf8f;--bad:#ff8a80;--code:#1f1f1e}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.55 system-ui,-apple-system,Segoe UI,sans-serif}
main{max-width:1100px;margin:0 auto;padding:24px 16px 64px}h1{font-size:26px;margin:0 0 4px}h2{margin-top:36px;font-size:20px}
h3{font-size:16px;margin-top:24px}.sub{color:var(--mut);margin:0 0 16px}a{color:var(--acc)}
nav{display:flex;gap:8px;flex-wrap:wrap;margin:16px 0}nav button{border:1px solid var(--line);background:none;color:var(--fg);padding:6px 12px;border-radius:6px;cursor:pointer;font:inherit}
nav button.on{border-color:var(--acc);color:var(--acc)}section{display:none}section.on{display:block}
.tw{overflow-x:auto}table{border-collapse:collapse;margin:8px 0;font-size:13.5px}th,td{border-bottom:1px solid var(--line);padding:5px 9px;text-align:left;vertical-align:top}
th{font-weight:600}code,pre{background:var(--code);border-radius:4px;font:12.5px/1.45 ui-monospace,Menlo,Consolas,monospace}code{padding:1px 4px}
pre{padding:10px;overflow-x:auto;white-space:pre}.k{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:12px;margin:12px 0}
.card{border:1px solid var(--line);border-radius:8px;padding:12px}.card b{display:block;font-size:22px}.card span{color:var(--mut);font-size:13px}
select{font:inherit;padding:4px;background:var(--bg);color:var(--fg);border:1px solid var(--line);border-radius:6px;margin-right:8px}
.att{border-left:3px solid var(--line);padding-left:12px;margin:14px 0}.att.p{border-color:var(--ok)}.att.f{border-color:var(--bad)}
.tag{font-size:12px;padding:1px 6px;border-radius:10px;border:1px solid var(--line);margin-left:6px}img{max-width:100%}
</style></head><body><main>
<h1>ISA Agent</h1>
<p class="sub">An LLM agent that learns an unseen vector DSP instruction set from its manual, writes fixed-point DSP kernels,
and is checked by a real toolchain: clang/LLVM 18 for RISC-V vector, QEMU, bit-exact tests against NumPy. Plus an
ONNX &rarr; MLIR &rarr; LLVM comparison path. All numbers are measured; see the repository for the raw data.
<a href="https://github.com/pavanyadava007/isa-agent">GitHub</a></p>
<div class="k" id="kpis"></div>
<nav><button data-s="res" class="on">Results</button><button data-s="ep">Episode browser</button><button data-s="man">XDSP manual</button><button data-s="how">How it works</button></nav>
<section id="res" class="on"><img src="pass_rates.png" alt="Pass rate by condition">__RESULTS__</section>
<section id="ep"><p>Pick a run. Every attempt shows the program the model wrote and the feedback the tools returned.</p>
<select id="m"></select><select id="c"></select><select id="t"></select><select id="s"></select><div id="view"></div></section>
<section id="man">__MANUAL__</section>
<section id="how">__HOW__</section>
</main>
<script>
const EP = __EPISODES__, KPI = __KPIS__;
document.getElementById('kpis').innerHTML = KPI.map(k=>`<div class="card"><b>${k[0]}</b><span>${k[1]}</span></div>`).join('');
document.querySelectorAll('nav button').forEach(b=>b.onclick=()=>{document.querySelectorAll('nav button,section').forEach(x=>x.classList.remove('on'));b.classList.add('on');document.getElementById(b.dataset.s).classList.add('on')});
const uniq=f=>[...new Set(EP.map(f))].sort();
const sel={m:document.getElementById('m'),c:document.getElementById('c'),t:document.getElementById('t'),s:document.getElementById('s')};
const fill=(el,vals)=>{const v=el.value;el.innerHTML=vals.map(x=>`<option>${x}</option>`).join('');if(vals.includes(v))el.value=v};
fill(sel.m,uniq(e=>e.model));fill(sel.c,uniq(e=>e.condition));fill(sel.t,uniq(e=>e.task));fill(sel.s,uniq(e=>String(e.seed)));
const esc=s=>(s||'').replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));
function show(){const e=EP.find(e=>e.model==sel.m.value&&e.condition==sel.c.value&&e.task==sel.t.value&&String(e.seed)==sel.s.value);
 const v=document.getElementById('view');if(!e){v.innerHTML='<p>No episode for this combination.</p>';return}
 let h=`<p><b>${e.passed?'PASSED':'FAILED'}</b> after ${e.attempts.length} step(s), ${e.output_tokens} output tokens, ${Math.round(e.seconds)} s`+
  (e.insns?`, ${e.insns} instructions`:'')+(e.searches.length?`, manual searches: ${e.searches.map(esc).join('; ')}`:'')+`</p>`;
 e.attempts.forEach((a,i)=>{h+=`<div class="att ${a.passed?'p':(a.kind=='search'?'':'f')}"><b>${i+1}. ${a.kind}</b>`+
  (a.kind=='search'?`<span class="tag">query</span><pre>${esc(a.feedback)}</pre>`:
  `<span class="tag">${a.passed?'pass':(a.compiled?'compiles, wrong':'does not compile')}</span>${a.insns?`<span class="tag">${a.insns} insns</span>`:''}<pre>${esc(a.code)}</pre>`+(a.feedback?`<pre>${esc(a.feedback)}</pre>`:''))+`</div>`});
 v.innerHTML=h}
Object.values(sel).forEach(x=>x.onchange=show);sel.c.value='xdsp_agent';show();
</script></body></html>"""

HOW = """
<h2>The question</h2>
<p>Can an LLM agent write correct, fast kernels for a DSP whose instruction set exists only in an internal manual?
Public models have never seen such an ISA, so everything must come from the documentation and from tool feedback.</p>
<h2>The stand-in ISA</h2>
<p>XDSP is a fictional vector DSP with 73 documented operations (Q15 fixed-point multiply with rounding and saturation,
widening multiply-accumulate, de-interleaving complex loads, masks, reductions, float FMA). Each operation is a thin
wrapper over one RISC-V Vector 1.0 intrinsic, so programs run bit-exactly on QEMU, but the names, types and manual are
new, so a model cannot have memorised them.</p>
<h2>Conditions</h2>
<p>Public ISA without docs; XDSP without docs; XDSP with the full manual in the prompt; XDSP with retrieved manual
sections (BM25); the agent (retrieval, a SEARCH tool, and up to four repair rounds driven by compiler errors, test
failures and the manual entries of the operations involved); the agent plus two optimisation rounds driven by measured
instruction counts.</p>
<h2>Verification</h2>
<p>Every program is compiled with clang 18 for RV64GCV and run on QEMU 8.2 (VLEN 128). Tests cover edge sizes, a
forced-saturation case and writes past the output buffer. Cost is the dynamic instruction count inside the kernel
(from QEMU's execution trace), not cycles. Baselines: scalar C at -O2, clang auto-vectorisation at -O3 (also at the
same register width, LMUL = 1), and hand-written XDSP kernels.</p>
<h2>Limits</h2>
<p>14 kernels and 3 seeds per condition is small; instruction counts are not cycle counts; the ISA is a renamed
RISC-V subset, so it is easier than a real VLIW DSP with slot and latency constraints.</p>
"""


def main() -> None:
    SITE.mkdir(exist_ok=True)
    results = (ROOT / "docs" / "RESULTS.md").read_text()
    manual = (ROOT / "docs" / "XDSP_MANUAL.md").read_text()
    manual_html = "".join(
        f"<h3>{html.escape(b.splitlines()[0])}</h3><pre>{html.escape(chr(10).join(b.splitlines()[1:]).strip())}</pre>"
        for b in manual.split("\n## ")[1:])
    import os
    only = os.environ.get("REPORT_MODELS")
    eps = [json.loads(line) for line in (ROOT / "results" / "episodes.jsonl").read_text().splitlines()]
    eps = [e for e in eps if not only or e["model"] in only.split(",")]
    for e in eps:
        e.pop("final_code", None)
        for a in e["attempts"]:
            a["feedback"] = a["feedback"][:3000]
    summ = json.loads((ROOT / "results" / "summary.json").read_text())
    kpis = []
    for m, cs in sorted(summ.items()):
        for c, lab in (("xdsp_nodoc", "no docs"), ("xdsp_rag", "retrieved manual"), ("xdsp_agent", "agent")):
            if c in cs:
                kpis.append([f"{cs[c]['pass']}/{cs[c]['n']}", f"{m}: correct kernels, unseen ISA, {lab}"])
    page = (PAGE.replace("__RESULTS__", md_table_to_html(results)).replace("__MANUAL__", manual_html)
            .replace("__HOW__", HOW).replace("__EPISODES__", json.dumps(eps)).replace("__KPIS__", json.dumps(kpis)))
    (SITE / "index.html").write_text(page.replace("—", "-").replace("–", "-"))
    fig = ROOT / "docs" / "figures" / "pass_rates.png"
    if fig.exists():
        shutil.copy(fig, SITE / "pass_rates.png")
    (SITE / "README.md").write_text("---\ntitle: ISA Agent\nemoji: 🔧\ncolorFrom: blue\ncolorTo: gray\nsdk: static\n"
                                     "pinned: false\nshort_description: LLM agent writes DSP kernels for an unseen ISA\n---\n")
    print(f"site/ built: {len(eps)} episodes, {(SITE / 'index.html').stat().st_size // 1024} KB")


if __name__ == "__main__":
    main()
