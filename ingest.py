"""
ingest.py
Reads all .pdf and .txt files from documents/, splits them into overlapping
chunks, embeds them locally with sentence-transformers, and saves a FAISS
index + chunk metadata to output_index/.

Run this once after adding/changing files in documents/.
"""

import json
import os
from pathlib import Path

import faiss
import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

DOCS_DIR = Path(__file__).parent / "documents"
INDEX_DIR = Path(__file__).parent / "output_index"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # small, fast, runs locally on CPU
CHUNK_SIZE = 800          # characters per chunk
CHUNK_OVERLAP = 150       # overlap between consecutive chunks


def read_pdf(path: Path) -> str:
    reader = PdfReader(str(path))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def read_txt(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def load_documents() -> list[dict]:
    """Returns a list of {"source": filename, "text": full_text}."""
    docs = []
    if not DOCS_DIR.exists():
        raise FileNotFoundError(f"Missing folder: {DOCS_DIR}. Create it and add files.")

    for path in sorted(DOCS_DIR.iterdir()):
        if path.suffix.lower() == ".pdf":
            text = read_pdf(path)
        elif path.suffix.lower() == ".txt":
            text = read_txt(path)
        else:
            continue
        if text.strip():
            docs.append({"source": path.name, "text": text})
    return docs


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Simple sliding-window character chunker with overlap."""
    text = " ".join(text.split())  # normalize whitespace
    if len(text) <= chunk_size:
        return [text] if text else []

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks


def build_index():
    docs = load_documents()
    if not docs:
        print(f"No .pdf or .txt files found in {DOCS_DIR}. Add some and re-run.")
        return

    print(f"Loaded {len(docs)} document(s): {[d['source'] for d in docs]}")

    all_chunks = []  # list of {"source": ..., "chunk_id": ..., "text": ...}
    for doc in docs:
        pieces = chunk_text(doc["text"])
        for i, piece in enumerate(pieces):
            all_chunks.append({"source": doc["source"], "chunk_id": i, "text": piece})

    print(f"Created {len(all_chunks)} chunks. Loading embedding model '{EMBEDDING_MODEL}'...")
    model = SentenceTransformer(EMBEDDING_MODEL)

    texts = [c["text"] for c in all_chunks]
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    embeddings = embeddings.astype("float32")

    # Normalize for cosine similarity via inner product
    faiss.normalize_L2(embeddings)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    INDEX_DIR.mkdir(exist_ok=True)
    faiss.write_index(index, str(INDEX_DIR / "faiss_index.bin"))
    with open(INDEX_DIR / "chunks.json", "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)
    with open(INDEX_DIR / "model_name.txt", "w") as f:
        f.write(EMBEDDING_MODEL)

    print(f"Index built and saved to {INDEX_DIR}/")


if __name__ == "__main__":
    build_index()
