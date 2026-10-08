"""Runs the golden-set retrieval evaluation as a pytest check (uses the real local embedding model)."""
import importlib.util
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def report():
    path = Path(__file__).resolve().parent.parent / "eval" / "run_eval.py"
    spec = importlib.util.spec_from_file_location("run_eval", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.evaluate(k=4)


def test_hit_rate_meets_threshold(report):
    assert report["hit_rate"] >= 0.90


def test_mrr_meets_threshold(report):
    assert report["mrr"] >= 0.75
