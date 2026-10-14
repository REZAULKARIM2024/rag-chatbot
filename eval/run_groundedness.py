"""
eval/run_groundedness.py
End-to-end check of the agent's "say I don't know" behaviour (LangGraph agent with
its post-generation groundedness grader). Complements run_eval.py, which only
measures retrieval and showed that a similarity cut-off cannot refuse reliably.

THIS SPENDS API CREDIT: every question runs the agent (generate + grade, up to 2
rounds). It therefore refuses to start without --yes, prints a worst-case call
estimate first, and stops early if --max-cost is exceeded.

What is measured (golden_hard.json):
  answerable questions   -> answered (not refused), grounded, retries used
  unanswerable questions -> correctly refused vs confidently answered anyway
  cost and latency per question (prices in usage.py are illustrative)

A reply counts as a refusal if it carries the agent's fallback caveat or clearly says
the documents do not contain the answer (phrase list below). That phrase match is a
heuristic: read the printed answers for the borderline cases before quoting numbers.

Usage:
    set ANTHROPIC_API_KEY=...
    python eval/run_groundedness.py --limit 4            (cheap smoke run)
    python eval/run_groundedness.py --yes --max-cost 0.50
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

HARD_PATH = Path(__file__).resolve().parent / "golden_hard.json"
FALLBACK_PREFIX = "I don't have enough grounded information"
MAX_CALLS_PER_QUESTION = 4  # generate+grade, then one retry of each

NO_INFO_PATTERNS = [
    r"\b(do not|don't|doesn't|does not) (have|contain|include|cover|mention)\b",
    r"\bnot (mentioned|covered|included|found|provided|available)\b",
    r"\bno (information|mention|details?)\b",
    r"\b(cannot|can't|unable to) (find|answer|determine)\b",
    r"\b(isn't|is not|aren't|are not) (in|part of) the (documents?|context|provided)\b",
]


def is_refusal(answer: str) -> bool:
    text = (answer or "").strip()
    if text.startswith(FALLBACK_PREFIX):
        return True
    return any(re.search(p, text, re.IGNORECASE) for p in NO_INFO_PATTERNS)


def classify(case: dict, state: dict) -> dict:
    refused = is_refusal(state.get("answer", ""))
    usage = state.get("usage_summary", {})
    return {
        "question": case["question"],
        "unanswerable": bool(case.get("unanswerable")),
        "refused": refused,
        "grounded": bool(state.get("grounded")),
        "retries": state.get("retries", 0),
        "cost_usd": usage.get("cost_usd", 0.0),
        "latency_s": usage.get("latency_s", 0.0),
        "answer": state.get("answer", ""),
    }


def summarize(rows: list[dict]) -> dict:
    ans = [r for r in rows if not r["unanswerable"]]
    unans = [r for r in rows if r["unanswerable"]]

    def share(items, pred):
        return sum(1 for r in items if pred(r)) / len(items) if items else None

    return {
        "answerable": len(ans),
        "unanswerable": len(unans),
        "answered_rate": share(ans, lambda r: not r["refused"]),          # higher is better
        "wrongly_refused": share(ans, lambda r: r["refused"]),            # lower is better
        "correctly_refused": share(unans, lambda r: r["refused"]),        # higher is better
        "confidently_wrong": share(unans, lambda r: not r["refused"]),    # lower is better
        "retry_rate": share(rows, lambda r: r["retries"] > 0),
        "total_cost_usd": sum(r["cost_usd"] for r in rows),
        "mean_latency_s": (sum(r["latency_s"] for r in rows) / len(rows)) if rows else 0.0,
    }


def run_cases(cases, run_one, max_cost=None, on_row=None):
    """run_one(question) -> final agent state. Stops early once the spend limit is reached."""
    rows, spent = [], 0.0
    for case in cases:
        row = classify(case, run_one(case["question"]))
        rows.append(row)
        spent += row["cost_usd"]
        if on_row:
            on_row(row)
        if max_cost is not None and spent >= max_cost:
            print(f"\nSTOPPED EARLY: spent ${spent:.4f} >= --max-cost ${max_cost:.2f} "
                  f"after {len(rows)}/{len(cases)} questions.")
            break
    return rows


def pick_cases(limit=None):
    """Interleave answerable and unanswerable so a small --limit still shows both kinds."""
    cases = json.loads(HARD_PATH.read_text(encoding="utf-8"))
    ans = [c for c in cases if not c.get("unanswerable")]
    unans = [c for c in cases if c.get("unanswerable")]
    mixed = []
    for i in range(max(len(ans), len(unans))):
        mixed += ans[i:i + 1] + unans[i:i + 1]
    return mixed[:limit] if limit else mixed


def _pct(x):
    return "n/a" if x is None else f"{x:.0%}"


def print_report(rows):
    s = summarize(rows)
    print(f"\nQuestions run: {len(rows)} ({s['answerable']} answerable, {s['unanswerable']} unanswerable)")
    print(f"  answerable   -> answered {_pct(s['answered_rate'])}   wrongly refused {_pct(s['wrongly_refused'])}")
    print(f"  unanswerable -> correctly refused {_pct(s['correctly_refused'])}   "
          f"confidently wrong {_pct(s['confidently_wrong'])}")
    print(f"  retried after a failed grounding check: {_pct(s['retry_rate'])}")
    print(f"  cost ${s['total_cost_usd']:.4f} (illustrative prices)   mean latency {s['mean_latency_s']:.1f}s/question")
    for r in rows:
        if r["unanswerable"] != r["refused"]:
            kind = "UNANSWERABLE but answered" if r["unanswerable"] else "ANSWERABLE but refused"
            print(f"\n  CHECK [{kind}] {r['question']}\n    -> {r['answer'][:220].strip()!r}")
    print("\nREAD the CHECK lines before quoting these numbers: refusal detection is a phrase heuristic.")


def main():
    parser = argparse.ArgumentParser(description="Measure the agent's refusal behaviour (costs API credit).")
    parser.add_argument("--limit", type=int, default=None, help="run only the first N questions (mixed kinds)")
    parser.add_argument("--max-cost", type=float, default=None, help="stop once estimated spend reaches this (USD)")
    parser.add_argument("--yes", action="store_true", help="confirm that you accept spending API credit")
    args = parser.parse_args()

    cases = pick_cases(args.limit)
    print(f"{len(cases)} questions, up to {len(cases) * MAX_CALLS_PER_QUESTION} model calls.")
    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("ANTHROPIC_API_KEY is not set. Nothing was run and nothing was spent.")
    if not args.yes and not args.limit:
        sys.exit("Full run spends API credit. Re-run with --yes (and ideally --max-cost 0.50), "
                 "or try --limit 4 first.")

    from sentence_transformers import SentenceTransformer
    import agent_graph
    import ingest
    import run_eval

    model = SentenceTransformer(ingest.EMBEDDING_MODEL)
    index, chunks = run_eval.build_eval_index(model)
    graph = agent_graph.build_graph(index, chunks, model, os.environ["ANTHROPIC_API_KEY"])
    rows = run_cases(cases, lambda q: agent_graph.run_agentic_query(graph, q), args.max_cost,
                     on_row=lambda r: print(f"  {'refused ' if r['refused'] else 'answered'}  "
                                            f"{'(unanswerable) ' if r['unanswerable'] else ''}{r['question']}"))
    print_report(rows)


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    main()
