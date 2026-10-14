"""
eval/run_eval.py
Retrieval evaluation harness for the RAG pipeline.

Builds an in-memory FAISS index from eval/corpus/ using the SAME chunking,
embedding model and retrieval code as production (ingest.py / chatbot.py),
then scores every question in eval/golden_set.json:

  hit@k   - did a chunk containing the expected keyword (from the expected
            source file) appear in the top-k results?
  MRR     - mean reciprocal rank of the first correct chunk.

It makes NO LLM calls, so it is free and deterministic. CI fails if the
scores drop below the thresholds - a regression test for retrieval quality
when you change chunk size, the embedding model or the retrieval logic.

Usage:
    python eval/run_eval.py                 # default k=4
    python eval/run_eval.py --k 3 --min-hit 0.9 --min-mrr 0.75
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import faiss  # noqa: E402
from sentence_transformers import SentenceTransformer  # noqa: E402

import chatbot  # noqa: E402
import ingest  # noqa: E402

EVAL_DIR = Path(__file__).resolve().parent
CORPUS_DIR = EVAL_DIR / "corpus"
GOLDEN_PATH = EVAL_DIR / "golden_set.json"
HARD_PATH = EVAL_DIR / "golden_hard.json"


def build_eval_index(model, strategy=None):
    chunks = []
    for path in sorted(CORPUS_DIR.glob("*.txt")):
        text = path.read_text(encoding="utf-8")
        for i, piece in enumerate(ingest.chunk_text(text, strategy=strategy)):
            chunks.append({"source": path.name, "chunk_id": i, "text": piece})
    embeddings = model.encode([c["text"] for c in chunks], convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index, chunks


def _requirements(case):
    """List of (sources, keyword) the top-k must satisfy. Single-source cases have one entry."""
    if "required" in case:
        return [({r["source"]}, r["keyword"].lower()) for r in case["required"]]
    expected = case["expected_source"]
    sources = {expected} if isinstance(expected, str) else set(expected)
    return [(sources, case["expected_keyword"].lower())]


def first_correct_rank(results, case):
    """
    1-based rank at which the case is satisfied, else None. A single-source case is satisfied by the
    first chunk from an expected source containing the keyword. A multi-document case ("required")
    is satisfied only when EVERY requirement is met; its rank is the rank of the last one met.
    """
    ranks = []
    for sources, keyword in _requirements(case):
        found = next((rank for rank, r in enumerate(results, start=1)
                      if r["source"] in sources and keyword in r["text"].lower()), None)
        if found is None:
            return None
        ranks.append(found)
    return max(ranks)


def threshold_table(answerable_scores, unanswerable_scores, thresholds=(0.10, 0.15, 0.20, 0.25, 0.30)):
    """
    For each candidate cut-off on the top-1 similarity: what share of real (answerable) questions
    would still be answered, and what share of unanswerable ones would be correctly refused.
    """
    rows = []
    for t in thresholds:
        kept = sum(1 for a in answerable_scores if a >= t) / len(answerable_scores) if answerable_scores else 0.0
        refused = sum(1 for u in unanswerable_scores if u < t) / len(unanswerable_scores) if unanswerable_scores else 0.0
        rows.append((t, kept, refused))
    return rows


def abstention_stats(answerable_scores, unanswerable_scores):
    """
    How well can the top-1 similarity score tell answerable questions from unanswerable ones?
    auroc: chance a random answerable question scores higher than a random unanswerable one
    (0.5 = no signal, 1.0 = perfect). best_threshold/best_accuracy: the cut-off that separates
    them best on this data.
    """
    if not answerable_scores or not unanswerable_scores:
        return None
    pairs = [(a, u) for a in answerable_scores for u in unanswerable_scores]
    auroc = sum(1.0 if a > u else 0.5 if a == u else 0.0 for a, u in pairs) / len(pairs)
    candidates = sorted(set(answerable_scores) | set(unanswerable_scores))
    total = len(answerable_scores) + len(unanswerable_scores)
    best_t, best_acc = None, -1.0
    for t in candidates:
        correct = sum(1 for a in answerable_scores if a >= t) + sum(1 for u in unanswerable_scores if u < t)
        if correct / total > best_acc:
            best_t, best_acc = t, correct / total
    return {
        "auroc": auroc,
        "best_threshold": best_t,
        "best_accuracy": best_acc,
        "answerable_mean": sum(answerable_scores) / len(answerable_scores),
        "unanswerable_mean": sum(unanswerable_scores) / len(unanswerable_scores),
        "unanswerable_max": max(unanswerable_scores),
        "answerable_min": min(answerable_scores),
    }


def load_cases(case_set: str = "base") -> list[dict]:
    """base = golden_set.json (the CI gate); hard = golden_hard.json; all = both."""
    cases = []
    if case_set in ("base", "all"):
        cases += json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
    if case_set in ("hard", "all"):
        cases += json.loads(HARD_PATH.read_text(encoding="utf-8"))
    return cases


def evaluate(k: int = 4, strategy=None, case_set: str = "base"):
    model = SentenceTransformer(ingest.EMBEDDING_MODEL)
    index, chunks = build_eval_index(model, strategy)
    cases = load_cases(case_set)

    rows, hits, rr_sum = [], 0, 0.0
    by_cat = {}
    answerable_scores, unanswerable_rows, miss_details = [], [], []
    for case in cases:
        results = chatbot.retrieve(case["question"], index, chunks, model, k=k)
        top1 = results[0]["score"] if results else 0.0
        cat = case.get("category", "easy")
        if case.get("unanswerable"):
            unanswerable_rows.append((case["question"], top1))
            continue
        rank = first_correct_rank(results, case)
        answerable_scores.append(top1)
        stats = by_cat.setdefault(cat, {"n": 0, "hits": 0})
        stats["n"] += 1
        if rank:
            hits += 1
            stats["hits"] += 1
            rr_sum += 1.0 / rank
        rows.append((case["question"], rank))
        if not rank:
            miss_details.append((case["question"], [
                (r["source"], r["chunk_id"], r["score"], r["text"][:90].replace("\n", " ")) for r in results]))

    n = len(rows)
    categories = {c: s["hits"] / s["n"] for c, s in by_cat.items()}
    return {"n": n, "k": k, "strategy": strategy or ingest.CHUNKER, "case_set": case_set,
            "hit_rate": hits / n if n else 0.0, "mrr": rr_sum / n if n else 0.0, "rows": rows,
            "chunks": len(chunks), "categories": categories,
            "unanswerable_rows": unanswerable_rows, "answerable_scores": answerable_scores,
            "miss_details": miss_details,
            "abstention": abstention_stats(answerable_scores, [sc for _, sc in unanswerable_rows])}


def compare(k: int):
    """Evaluates both chunking strategies on the same golden set and prints a before/after table."""
    reports = {name: evaluate(k, name) for name in ("fixed", "sentence")}
    fixed, sent = reports["fixed"], reports["sentence"]
    print(f"Questions: {fixed['n']} | k={k}\n")
    print(f"{'metric':<24}{'fixed (before)':>16}{'sentence (after)':>18}")
    print(f"{'chunks':<24}{fixed['chunks']:>16}{sent['chunks']:>18}")
    print(f"{'hit@' + str(k):<24}{fixed['hit_rate']:>16.2%}{sent['hit_rate']:>18.2%}")
    print(f"{'MRR':<24}{fixed['mrr']:>16.3f}{sent['mrr']:>18.3f}")
    for cat in sorted(set(fixed['categories']) | set(sent['categories'])):
        label = f"hit@{k} ({cat})"
        print(f"{label:<24}{fixed['categories'].get(cat, 0):>16.2%}{sent['categories'].get(cat, 0):>18.2%}")
    print("\nPer-question rank changes (fixed -> sentence):")
    changed = False
    for (q, r_fixed), (_, r_sent) in zip(fixed["rows"], sent["rows"]):
        if r_fixed != r_sent:
            changed = True
            print(f"  {str(r_fixed or 'MISS'):>4} -> {str(r_sent or 'MISS'):<4} {q}")
    if not changed:
        print("  (no question changed rank)")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--set", dest="case_set", choices=["base", "hard", "all"], default="base",
                        help="base = CI gate set (default); hard = paraphrase / multi-doc / unanswerable; all = both")
    parser.add_argument("--chunker", choices=["fixed", "sentence"], default=None,
                        help="chunking strategy (default: ingest.CHUNKER)")
    parser.add_argument("--compare", action="store_true",
                        help="print fixed vs sentence chunking side by side (no threshold check)")
    parser.add_argument("--min-hit", type=float, default=None,
                        help="fail if hit@k is below this (default 0.90 for --set base, otherwise report only)")
    parser.add_argument("--min-mrr", type=float, default=None,
                        help="fail if MRR is below this (default 0.75 for --set base, otherwise report only)")
    parser.add_argument("--explain", action="store_true",
                        help="for each MISS, show which chunks retrieval returned (source, score, start of text)")
    parser.add_argument("--min-auroc", type=float, default=None,
                        help="fail if unanswerable-question detection AUROC is below this")
    args = parser.parse_args()
    if args.case_set == "base":
        args.min_hit = 0.90 if args.min_hit is None else args.min_hit
        args.min_mrr = 0.75 if args.min_mrr is None else args.min_mrr

    if args.compare:
        compare(args.k)
        return

    report = evaluate(args.k, args.chunker, args.case_set)
    print(f"Chunker: {report['strategy']}   Set: {report['case_set']}")
    print(f"Corpus: {report['chunks']} chunks | Answerable questions: {report['n']} | "
          f"Unanswerable: {len(report['unanswerable_rows'])} | k={report['k']}\n")
    for question, rank in report["rows"]:
        status = f"rank {rank}" if rank else "MISS"
        print(f"  [{status:>7}] {question}")
    print(f"\nhit@{args.k}: {report['hit_rate']:.2%}   MRR: {report['mrr']:.3f}")
    for cat, rate in sorted(report["categories"].items()):
        print(f"  hit@{args.k} ({cat} questions): {rate:.2%}")

    ab = report["abstention"]
    if ab:
        print("\nUnanswerable questions (top-1 similarity; lower = easier to tell apart from real questions):")
        for question, score in sorted(report["unanswerable_rows"], key=lambda r: -r[1]):
            print(f"  [{score:.2f}] {question}")
        print(f"  mean top-1 score: answerable {ab['answerable_mean']:.2f} vs unanswerable {ab['unanswerable_mean']:.2f}")
        print(f"  highest unanswerable {ab['unanswerable_max']:.2f} vs lowest answerable {ab['answerable_min']:.2f}")
        print(f"  AUROC {ab['auroc']:.3f} (0.5 = no signal, 1.0 = perfect)   "
              f"best cut-off {ab['best_threshold']:.2f} -> {ab['best_accuracy']:.0%} correct")

    if args.explain and report["miss_details"]:
        print("\nMISSES: what retrieval actually returned")
        for question, retrieved in report["miss_details"]:
            print(f"  Q: {question}")
            for source, chunk_id, score, snippet in retrieved:
                print(f"     [{score:.2f}] {source} #{chunk_id}: {snippet}...")
    if ab:
        print("\nIf answers were refused below a similarity cut-off:")
        print(f"  {'cut-off':>8} {'real questions still answered':>32} {'unanswerable correctly refused':>32}")
        for t, kept, refused in threshold_table(report["answerable_scores"], [sc for _, sc in report["unanswerable_rows"]]):
            print(f"  {t:>8.2f} {kept:>32.0%} {refused:>32.0%}")

    failed = False
    if args.min_hit is not None and report["hit_rate"] < args.min_hit:
        failed = True
    if args.min_mrr is not None and report["mrr"] < args.min_mrr:
        failed = True
    if args.min_auroc is not None and (ab is None or ab["auroc"] < args.min_auroc):
        failed = True
    if failed:
        print("\nFAIL: below the requested thresholds")
        sys.exit(1)
    print("\nPASS" if (args.min_hit is not None or args.min_mrr is not None or args.min_auroc is not None) else "\nDONE (report only, no thresholds set)")


if __name__ == "__main__":
    main()
