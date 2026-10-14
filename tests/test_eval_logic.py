"""Tests for the evaluation scoring logic itself (pure functions, no embedding model needed)."""
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def run_eval():
    spec = importlib.util.spec_from_file_location("run_eval_logic", ROOT / "eval" / "run_eval.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def chunk(source, text):
    return {"source": source, "text": text}


def test_single_source_rank(run_eval):
    results = [chunk("a.txt", "nothing"), chunk("b.txt", "has Keyword here")]
    case = {"expected_source": "b.txt", "expected_keyword": "keyword"}
    assert run_eval.first_correct_rank(results, case) == 2


def test_wrong_source_with_right_keyword_is_a_miss(run_eval):
    results = [chunk("other.txt", "has keyword")]
    assert run_eval.first_correct_rank(results, {"expected_source": "b.txt", "expected_keyword": "keyword"}) is None


def test_multi_document_needs_every_requirement(run_eval):
    case = {"required": [{"source": "a.txt", "keyword": "alpha"}, {"source": "b.txt", "keyword": "beta"}]}
    both = [chunk("a.txt", "alpha"), chunk("x.txt", "-"), chunk("b.txt", "beta")]
    only_one = [chunk("a.txt", "alpha"), chunk("x.txt", "-")]
    assert run_eval.first_correct_rank(both, case) == 3   # rank of the LAST requirement met
    assert run_eval.first_correct_rank(only_one, case) is None


def test_abstention_perfect_separation(run_eval):
    stats = run_eval.abstention_stats([0.8, 0.7, 0.6], [0.3, 0.2])
    assert stats["auroc"] == 1.0 and stats["best_accuracy"] == 1.0
    assert 0.3 < stats["best_threshold"] <= 0.6


def test_abstention_no_signal(run_eval):
    stats = run_eval.abstention_stats([0.5, 0.5], [0.5, 0.5])
    assert stats["auroc"] == 0.5


def test_abstention_needs_both_groups(run_eval):
    assert run_eval.abstention_stats([0.5], []) is None


def test_hard_set_is_well_formed_and_keywords_exist_in_corpus(run_eval):
    cases = run_eval.load_cases("hard")
    assert {c["category"] for c in cases} == {"paraphrase", "multi-doc", "unanswerable"}
    assert sum(1 for c in cases if c.get("unanswerable")) >= 8
    corpus = {p.name: p.read_text(encoding="utf-8").lower() for p in (ROOT / "eval" / "corpus").glob("*.txt")}
    for case in cases:
        if case.get("unanswerable"):
            assert "expected_source" not in case and "required" not in case
            continue
        for sources, keyword in run_eval._requirements(case):
            assert any(keyword in corpus[s] for s in sources), case["question"]


def test_threshold_table_tradeoff(run_eval):
    rows = run_eval.threshold_table([0.9, 0.5, 0.2], [0.4, 0.1], thresholds=(0.15, 0.3, 0.6))
    assert rows[0] == (0.15, 1.0, 0.5)             # keeps every real question, refuses the 0.1 one
    assert rows[1][1] == pytest.approx(2 / 3) and rows[1][2] == 0.5
    assert rows[2][1] == pytest.approx(1 / 3)       # a strict cut-off refuses real questions too


def test_threshold_table_handles_empty_inputs(run_eval):
    assert run_eval.threshold_table([], [], thresholds=(0.2,)) == [(0.2, 0.0, 0.0)]
