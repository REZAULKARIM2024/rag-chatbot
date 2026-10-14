"""Tests for the groundedness eval's scoring logic. No API, no model: the agent is replaced by a fake."""
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def ge():
    spec = importlib.util.spec_from_file_location("run_groundedness_logic", ROOT / "eval" / "run_groundedness.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def state(answer, grounded=True, retries=0, cost=0.01):
    return {"answer": answer, "grounded": grounded, "retries": retries,
            "usage_summary": {"cost_usd": cost, "latency_s": 1.0}}


@pytest.mark.parametrize("text,expected", [
    ("I don't have enough grounded information in your documents to answer this confidently. Here's...", True),
    ("The documents do not contain anything about Kubernetes.", True),
    ("That is not mentioned in the provided context.", True),
    ("Cypress offers automatic waiting.", False),
    ("", False),
])
def test_is_refusal(ge, text, expected):
    assert ge.is_refusal(text) is expected


def test_summarize_counts_each_kind(ge):
    rows = [
        ge.classify({"question": "a", "unanswerable": False}, state("Real answer")),
        ge.classify({"question": "b", "unanswerable": False}, state("I don't have enough grounded information x")),
        ge.classify({"question": "c", "unanswerable": True}, state("Not mentioned in the documents.")),
        ge.classify({"question": "d", "unanswerable": True}, state("Sure, here is how.", retries=1)),
    ]
    s = ge.summarize(rows)
    assert s["answered_rate"] == 0.5 and s["wrongly_refused"] == 0.5
    assert s["correctly_refused"] == 0.5 and s["confidently_wrong"] == 0.5
    assert s["retry_rate"] == 0.25
    assert s["total_cost_usd"] == pytest.approx(0.04)


def test_summarize_with_only_one_kind_does_not_divide_by_zero(ge):
    s = ge.summarize([ge.classify({"question": "a"}, state("Real answer"))])
    assert s["correctly_refused"] is None and s["answered_rate"] == 1.0


def test_run_cases_stops_at_cost_limit(ge, capsys):
    cases = [{"question": f"q{i}"} for i in range(5)]
    rows = ge.run_cases(cases, lambda q: state("ok", cost=0.30), max_cost=0.50)
    assert len(rows) == 2                      # 0.30, then 0.60 >= 0.50 -> stop
    assert "STOPPED EARLY" in capsys.readouterr().out


def test_pick_cases_mixes_kinds_under_small_limit(ge):
    picked = ge.pick_cases(limit=4)
    assert len(picked) == 4
    assert any(c.get("unanswerable") for c in picked) and any(not c.get("unanswerable") for c in picked)


def test_full_run_refuses_without_key_or_confirmation(ge, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr("sys.argv", ["x", "--yes"])
    with pytest.raises(SystemExit) as exc:
        ge.main()
    assert "nothing was spent" in str(exc.value).lower()
