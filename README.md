# Personal RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot that answers questions about your own
documents (resume, project notes, certificates, QA runbooks, etc.) using:

- **Local embeddings** — `sentence-transformers` (runs on your machine, no API cost)
- **FAISS** — fast local vector similarity search
- **Claude API** — for generating grounded, natural-language answers

This is a portfolio project designed to demonstrate practical GenAI / RAG skills for a
QA Automation Engineer → AI Engineer career transition.

## How it works

```
documents/*.pdf, *.txt
        │
        ▼
   ingest.py  ──► chunks text ──► embeds chunks ──► saves FAISS index
        │
        ▼
  output_index/ (faiss_index.bin, chunks.json)
        │
        ▼
  chatbot.py ──► embeds your question ──► retrieves top-k relevant chunks
        │
        ▼
  sends question + retrieved context to Claude ──► grounded answer
```

## Setup

```bash
cd rag-chatbot
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Set your Anthropic API key (get one at https://console.anthropic.com/):

```bash
export ANTHROPIC_API_KEY="your-key-here"   # Windows: set ANTHROPIC_API_KEY=your-key-here
```

## Usage

1. Drop your documents (PDF or TXT) into the `documents/` folder.
   - resume.pdf, project_notes.txt, qa_runbook.pdf — anything you want to query.

2. Build the search index:

```bash
python ingest.py
```

3. Start chatting:

```bash
python chatbot.py
```

Example session:

```
Ask a question (or 'quit'): What automation tools does this candidate know?
Answer: Based on the resume, the candidate has hands-on experience with Selenium
WebDriver, Playwright, TestNG, Cucumber BDD, and Appium for mobile automation...
Sources: resume.pdf (chunk 2), resume.pdf (chunk 4)

Ask a question (or 'quit'): quit
```

## Web UI (Streamlit)

For a browser-based chat interface with conversation memory (follow-up
questions like "what about his earlier role?" work correctly) instead of
the command-line loop, run:

```bash
streamlit run streamlit_app.py
```

This opens a local web page where you can:
- Upload PDF/TXT documents directly (no need to copy files manually)
- Rebuild the search index with one click
- Chat with multi-turn memory — previous questions and answers are sent
  back to Claude as context for follow-ups
- See which source chunks backed each answer

Enter your Anthropic API key in the sidebar (or set `ANTHROPIC_API_KEY`
as an environment variable beforehand and it will be pre-filled).

## Agentic mode (LangGraph)

Plain RAG trusts whatever the model generates from the retrieved context.
Toggle **"🧠 Agentic mode"** in the Streamlit sidebar to switch to a
[LangGraph](https://github.com/langchain-ai/langgraph)-based flow that adds
a self-check step: after generating an answer, a second LLM call grades
whether the answer is actually supported by the retrieved context. If it
isn't, the graph retries retrieval once with a broadened query before
falling back to an honest "I'm not confident" answer instead of a
possible hallucination.

```mermaid
graph TD
    A[retrieve] --> B[generate]
    B --> C{grade groundedness}
    C -->|grounded| D[END]
    C -->|not grounded, retries left| E[reformulate query]
    E --> A
    C -->|retries exhausted| F[fallback answer with caveat]
    F --> D
```

The UI shows a groundedness badge and an expandable **agent trace** for
each answer, so you can see exactly which path the graph took — useful
both for debugging and for demoing the reasoning process in interviews.
This logic lives in `agent_graph.py` and reuses the same retrieval code
as the plain chatbot (`chatbot.retrieve`, `chatbot.build_context`), so
the two modes are directly comparable side by side.

## Production-realistic vector store (PostgreSQL + pgvector)

The local FAISS index is a single file on disk — fine for a demo, but not
how a team would run this in production (no concurrent writers, no
backups, can't be queried alongside other application data). `pg_store.py`
adds a [pgvector](https://github.com/pgvector/pgvector)-backed alternative:
the same chunks and embeddings, but stored in a real Postgres table.

No local Postgres install needed — get a free instance with pgvector
already available at [neon.tech](https://neon.tech) or
[supabase.com](https://supabase.com), then:

```bash
set DATABASE_URL=postgresql://user:password@host/dbname?sslmode=require
python pg_store.py --ingest
python pg_store.py --query "your question here"   # quick CLI sanity check
```

In the Streamlit app, switch **"Vector store backend"** in the sidebar to
**PostgreSQL (pgvector)**, paste the same connection string, and click
**Rebuild index** — the chat interface works identically either way, so
the two backends are directly comparable.

```mermaid
graph LR
    subgraph "Local (FAISS)"
    A1[documents/] --> B1[ingest.py] --> C1[(faiss_index.bin)]
    end
    subgraph "PostgreSQL (pgvector)"
    A2[documents/] --> B2[pg_store.py --ingest] --> C2[(Postgres: document_chunks table)]
    end
    C1 --> D[chatbot.py / streamlit_app.py]
    C2 --> D
```

Note: agentic mode (LangGraph, above) currently only supports the FAISS
backend — extending it to pgvector is a natural next step since
`pg_store.retrieve_pg()` already returns the same shape as
`chatbot.retrieve()`.

## Project structure

```
rag-chatbot/
├── documents/          # put your PDFs/TXTs here
├── output_index/       # generated FAISS index + chunk metadata (gitignored)
├── ingest.py           # builds the vector index from documents/
├── chatbot.py          # interactive CLI Q&A loop
├── streamlit_app.py    # browser-based UI with conversation memory
├── agent_graph.py      # LangGraph agentic flow with groundedness self-check
├── pg_store.py         # PostgreSQL + pgvector backend (production-realistic alternative to FAISS)
├── requirements.txt
└── README.md
```

## What this demonstrates (for interviews / resume)

- Document chunking and preprocessing for LLM pipelines
- Local embedding generation with `sentence-transformers`
- Vector similarity search with FAISS
- Prompt construction for grounded (anti-hallucination) generation
- Multi-turn conversation design (chat history passed back to the model)
- Both a clean CLI and a browser-based (Streamlit) application interface
- Agentic orchestration with LangGraph: conditional routing, retry loops,
  and a self-check (groundedness grading) step instead of blind trust in
  a single LLM call
- A swappable vector store backend (local FAISS vs. PostgreSQL/pgvector)
  behind the same retrieval interface — the kind of "make it real" step
  most tutorials skip

## Possible extensions

- Swap FAISS for a hosted vector DB (Pinecone, Chroma Cloud, Weaviate)
- Add source-highlighting / citations in the Streamlit UI
- Stream the answer token-by-token instead of waiting for the full response
- Deploy the Streamlit app (e.g. Streamlit Community Cloud) so it's reachable
  from a shareable link, not just localhost
- Add chunking strategies aware of document structure (headings, sections)
  instead of fixed-size sliding windows
