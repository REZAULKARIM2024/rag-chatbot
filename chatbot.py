"""
chatbot.py
Interactive Q&A loop over your ingested documents.

1. Embeds your question with the same local model used in ingest.py.
2. Retrieves the top-k most similar chunks from the FAISS index.
3. Sends the question + retrieved chunks to Claude, asking it to answer
   ONLY from the provided context (reduces hallucination).

Prereqs: run `python ingest.py` first, and set ANTHROPIC_API_KEY.
"""

import json
import os
from pathlib import Path

import faiss
import numpy as np
from anthropic import Anthropic
from sentence_transformers import SentenceTransformer

INDEX_DIR = Path(__file__).parent / "output_index"
TOP_K = 4
CLAUDE_MODEL = "claude-sonnet-4-6"

SYSTEM_PROMPT = """You are a helpful assistant answering questions using ONLY the
provided context chunks from the user's own documents. If the answer isn't in the
context, say so clearly instead of guessing. Keep answers concise and cite which
source file(s) you used."""


def load_index():
    if not (INDEX_DIR / "faiss_index.bin").exists():
        raise FileNotFoundError(
            "No index found. Run `python ingest.py` first to build the index."
        )
    index = faiss.read_index(str(INDEX_DIR / "faiss_index.bin"))
    with open(INDEX_DIR / "chunks.json", encoding="utf-8") as f:
        chunks = json.load(f)
    with open(INDEX_DIR / "model_name.txt") as f:
        model_name = f.read().strip()
    return index, chunks, model_name


def retrieve(query: str, index, chunks, model, k: int = TOP_K):
    query_vec = model.encode([query], convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(query_vec)
    scores, indices = index.search(query_vec, k)
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        results.append({**chunks[idx], "score": float(score)})
    return results


def build_context(results: list[dict]) -> str:
    parts = []
    for r in results:
        parts.append(f"[Source: {r['source']} | chunk {r['chunk_id']}]\n{r['text']}")
    return "\n\n---\n\n".join(parts)


def ask_claude(client: Anthropic, question: str, context: str) -> str:
    user_message = f"""Context from my documents:

{context}

---

Question: {question}"""

    response = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=600,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    )
    return "".join(block.text for block in response.content if block.type == "text")


def main():
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: Set the ANTHROPIC_API_KEY environment variable first.")
        return

    print("Loading index and embedding model...")
    index, chunks, model_name = load_index()
    model = SentenceTransformer(model_name)
    client = Anthropic(api_key=api_key)

    print(f"Ready. {len(chunks)} chunks loaded. Type 'quit' to exit.\n")

    while True:
        question = input("Ask a question (or 'quit'): ").strip()
        if question.lower() in {"quit", "exit"}:
            break
        if not question:
            continue

        results = retrieve(question, index, chunks, model)
        if not results:
            print("No relevant content found in your documents.\n")
            continue

        context = build_context(results)
        answer = ask_claude(client, question, context)

        sources = ", ".join(sorted({f"{r['source']} (chunk {r['chunk_id']})" for r in results}))
        print(f"\nAnswer: {answer}")
        print(f"Sources: {sources}\n")


if __name__ == "__main__":
    main()
