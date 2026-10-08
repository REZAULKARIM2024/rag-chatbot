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


def build_eval_index(model):
    chunks = []
    for path in sorted(CORPUS_DIR.glob("*.txt")):
        text = path.read_text(encoding="utf-8")
        for i, piece in enumerate(ingest.chunk_text(text)):
            chunks.append({"source": path.name, "chunk_id": i, "text": piece})
    embeddings = model.encode([c["text"] for c in chunks], convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index, chunks


def first_correct_rank(results, case):
    """1-based rank of the first chunk from the expected source containing the keyword, else None."""
    keyword = case["expected_keyword"].lower()
    expected = case["expected_source"]
    expected_sources = {expected} if isinstance(expected, str) else set(expected)
    for rank, r in enumerate(results, start=1):
        if r["source"] in expected_sources and keyword in r["text"].lower():
            return rank
    return None


def evaluate(k: int = 4):
    model = SentenceTransformer(ingest.EMBEDDING_MODEL)
    index, chunks = build_eval_index(model)
    cases = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))

    rows, hits, rr_sum = [], 0, 0.0
    by_cat = {}
    for case in cases:
        results = chatbot.retrieve(case["question"], index, chunks, model, k=k)
        rank = first_correct_rank(results, case)
        cat = case.get("category", "easy")
        stats = by_cat.setdefault(cat, {"n": 0, "hits": 0})
        stats["n"] += 1
        if rank:
            hits += 1
            stats["hits"] += 1
            rr_sum += 1.0 / rank
        rows.append((case["question"], rank))

    n = len(cases)
    categories = {c: s["hits"] / s["n"] for c, s in by_cat.items()}
    return {"n": n, "k": k, "hit_rate": hits / n, "mrr": rr_sum / n, "rows": rows,
            "chunks": len(chunks), "categories": categories}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--min-hit", type=float, default=0.90, help="fail if hit@k is below this")
    parser.add_argument("--min-mrr", type=float, default=0.75, help="fail if MRR is below this")
    args = parser.parse_args()

    report = evaluate(args.k)
    print(f"Corpus: {report['chunks']} chunks | Questions: {report['n']} | k={report['k']}\n")
    for question, rank in report["rows"]:
        status = f"rank {rank}" if rank else "MISS"
        print(f"  [{status:>7}] {question}")
    print(f"\nhit@{args.k}: {report['hit_rate']:.2%}   MRR: {report['mrr']:.3f}")
    for cat, rate in sorted(report["categories"].items()):
        print(f"  hit@{args.k} ({cat} questions): {rate:.2%}")

    if report["hit_rate"] < args.min_hit or report["mrr"] < args.min_mrr:
        print(f"\nFAIL: below thresholds (hit>={args.min_hit}, mrr>={args.min_mrr})")
        sys.exit(1)
    print("\nPASS")


if __name__ == "__main__":
    main()
