"""
pg_store.py
PostgreSQL + pgvector backend for the RAG chatbot — a production-realistic
alternative to the local FAISS index in ingest.py / chatbot.py.

Why this exists:
    FAISS is great for a single-machine demo, but a FAISS index is just a
    local binary file — no concurrent writers, no backups, no way to query
    it alongside the rest of an application's data. Most companies want
    their chatbot's search index living in a real, queryable database.
    pgvector adds vector similarity search directly to PostgreSQL, so this
    module stores the same chunks + embeddings ingest.py produces, but in
    a Postgres table instead of a FAISS file.

Setup (no local Postgres install needed):
    1. Create a free Postgres database at https://neon.tech or
       https://supabase.com — both support pgvector on their free tier.
    2. Copy the connection string they give you (starts with postgresql://)
       and set it as the DATABASE_URL environment variable, e.g.:
         set DATABASE_URL=postgresql://user:password@host/dbname?sslmode=require
    3. Build the index:
         python pg_store.py --ingest
    4. Quick CLI test:
         python pg_store.py --query "your question here"

    The Streamlit app's sidebar also has a "PostgreSQL (pgvector)" backend
    option that calls these same functions, with DATABASE_URL entered there
    instead of as an environment variable if you prefer.
"""

import os
import sys

import ingest  # reuse load_documents() / chunk_text() / EMBEDDING_MODEL / DOCS_DIR

EMBEDDING_DIM = 384  # matches ingest.EMBEDDING_MODEL (all-MiniLM-L6-v2)
TABLE_NAME = "document_chunks"


def get_connection(database_url: str | None = None):
    """
    Opens a Postgres connection and makes sure the pgvector extension and
    the vector type are registered on it. database_url overrides the
    DATABASE_URL environment variable when provided (used by the Streamlit
    sidebar, which reads the key from a text input rather than the shell).
    """
    import psycopg
    from pgvector.psycopg import register_vector

    database_url = database_url or os.environ.get("DATABASE_URL")
    if not database_url:
        raise RuntimeError(
            "No DATABASE_URL provided. Point it at a Postgres instance with "
            "the pgvector extension available — see the setup notes at the "
            "top of pg_store.py for a free hosted option (Neon, Supabase)."
        )
    conn = psycopg.connect(database_url, autocommit=True)
    conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    register_vector(conn)
    return conn


def ensure_schema(conn):
    conn.execute(f"""
        CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
            id SERIAL PRIMARY KEY,
            source TEXT NOT NULL,
            chunk_id INT NOT NULL,
            content TEXT NOT NULL,
            embedding VECTOR({EMBEDDING_DIM})
        )
    """)
    # An approximate-nearest-neighbor index speeds up search once the table
    # has real scale. It's optional for a small demo dataset (a handful of
    # documents) and ivfflat can be picky about very small row counts, so
    # failures here are non-fatal — search still works via exact scan.
    try:
        conn.execute(f"""
            CREATE INDEX IF NOT EXISTS {TABLE_NAME}_embedding_idx
            ON {TABLE_NAME} USING ivfflat (embedding vector_cosine_ops)
            WITH (lists = 100)
        """)
    except Exception:
        pass


def build_pg_index(database_url: str | None = None):
    """Equivalent of ingest.build_index(), but writes to Postgres instead of a FAISS file."""
    from sentence_transformers import SentenceTransformer

    docs = ingest.load_documents()
    if not docs:
        print(f"No .pdf or .txt files found in {ingest.DOCS_DIR}.")
        return 0

    all_chunks = []
    for doc in docs:
        for i, piece in enumerate(ingest.chunk_text(doc["text"])):
            all_chunks.append({"source": doc["source"], "chunk_id": i, "text": piece})

    print(f"Embedding {len(all_chunks)} chunk(s)...")
    model = SentenceTransformer(ingest.EMBEDDING_MODEL)
    embeddings = model.encode([c["text"] for c in all_chunks], show_progress_bar=True)

    conn = get_connection(database_url)
    ensure_schema(conn)
    conn.execute(f"TRUNCATE {TABLE_NAME}")  # full rebuild, matches ingest.py's behavior

    with conn.cursor() as cur:
        for chunk, vector in zip(all_chunks, embeddings):
            cur.execute(
                f"INSERT INTO {TABLE_NAME} (source, chunk_id, content, embedding) "
                f"VALUES (%s, %s, %s, %s)",
                (chunk["source"], chunk["chunk_id"], chunk["text"], vector),
            )
    conn.close()
    print(f"Inserted {len(all_chunks)} chunk(s) into Postgres table '{TABLE_NAME}'.")
    return len(all_chunks)


def retrieve_pg(question: str, embedding_model, k: int = 4, database_url: str | None = None) -> list[dict]:
    """
    Equivalent of chatbot.retrieve(), but queries Postgres using pgvector's
    cosine-distance operator (<=>) instead of a local FAISS search. Returns
    the same shape (list of dicts with source/chunk_id/text/score) so it's
    a drop-in replacement wherever chatbot.retrieve() is used, including
    chatbot.build_context().
    """
    query_vector = embedding_model.encode([question])[0]

    conn = get_connection(database_url)
    with conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT source, chunk_id, content, 1 - (embedding <=> %s) AS score
            FROM {TABLE_NAME}
            ORDER BY embedding <=> %s
            LIMIT %s
            """,
            (query_vector, query_vector, k),
        )
        rows = cur.fetchall()
    conn.close()

    return [
        {"source": r[0], "chunk_id": r[1], "text": r[2], "score": float(r[3])}
        for r in rows
    ]


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Usage: python pg_store.py --ingest | --query "your question"')
        sys.exit(1)

    if sys.argv[1] == "--ingest":
        build_pg_index()
    elif sys.argv[1] == "--query" and len(sys.argv) > 2:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(ingest.EMBEDDING_MODEL)
        for r in retrieve_pg(sys.argv[2], model):
            preview = r["text"][:100].replace("\n", " ")
            print(f"[{r['score']:.3f}] {r['source']} (chunk {r['chunk_id']}): {preview}...")
    else:
        print('Usage: python pg_store.py --ingest | --query "your question"')
