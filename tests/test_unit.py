"""Fast tests (no Docker): retrieval, prompt/feedback logic, golden models, harness rendering."""

import numpy as np

from isaagent.agent import _norm, extract_code, task_prompt
from isaagent.gen_isa import render_header, render_manual
from isaagent.isa_spec import OPS
from isaagent.retrieval import Manual
from isaagent.tasks import BY_NAME, TASKS, rshift_rnu
from isaagent.toolbox import make_cases, render_test_main, static_check


def test_every_op_is_in_header_and_manual():
    h, m = render_header(), render_manual()
    for op in OPS:
        assert f" {op['name']}(" in h
        assert f"#### {op['name']}" in m


def test_op_names_unique():
    names = [op["name"] for op in OPS]
    assert len(names) == len(set(names))


def test_manual_search_finds_accumulator_zeroing():
    m = Manual()
    assert "xd_wdup" in [c["id"] for c in m.bm25("zero wide accumulator", k=3)]


def test_semantic_suggestion_beats_spelling_for_invented_name():
    m = Manual()
    assert "xd_wdup" in m.suggest("xd_wzero")
    assert "xd_wdup" not in m.suggest("xd_wzero", semantic=False)


def test_extract_code_takes_the_code_block():
    assert extract_code("text\n```c\nint f(void){return 1;}\n```\nmore") == "int f(void){return 1;}"


def test_identical_code_detection_ignores_comments_and_whitespace():
    assert _norm("int a; // x\n  int b;") == _norm("int a;\nint b; /* y */")


def test_static_check_rejects_raw_intrinsics_and_asm():
    assert static_check("__riscv_vle16_v_i16m1(p, vl);")
    assert static_check('asm("nop");')
    assert not static_check("xd_hload(p, vl);")


def test_rnu_rounding():
    assert list(rshift_rnu(np.array([16384, 16383, -16384, -16385]), 15)) == [1, 0, 0, -1]


def test_fir_golden_saturates_in_the_extra_case():
    t = BY_NAME["fir_q15"]
    d = t.extra()[0]
    y = t.golden(d)["y"]
    assert y.max() == 32767 and y.min() == -32768


def test_cases_cover_edge_sizes_and_render():
    for t in TASKS:
        cases = make_cases(t)
        assert len(cases) >= 5
        src = render_test_main(t, cases)
        assert t.name in src and "ALL PASS" in src


def test_prompt_mentions_signature():
    t = BY_NAME["dot_q15"]
    assert t.signature in task_prompt(t, "xdsp")
