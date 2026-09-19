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

## Project structure

```
rag-chatbot/
├── documents/          # put your PDFs/TXTs here
├── output_index/       # generated FAISS index + chunk metadata (gitignored)
├── ingest.py           # builds the vector index from documents/
├── chatbot.py          # interactive CLI Q&A loop
├── streamlit_app.py    # browser-based UI with conversation memory
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

## Possible extensions

- Swap FAISS for a hosted vector DB (Pinecone, Chroma Cloud, Weaviate)
- Add source-highlighting / citations in the Streamlit UI
- Stream the answer token-by-token instead of waiting for the full response
- Deploy the Streamlit app (e.g. Streamlit Community Cloud) so it's reachable
  from a shareable link, not just localhost
- Add chunking strategies aware of document structure (headings, sections)
  instead of fixed-size sliding windows
