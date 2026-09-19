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

## Project structure

```
rag-chatbot/
├── documents/          # put your PDFs/TXTs here
├── output_index/       # generated FAISS index + chunk metadata (gitignored)
├── ingest.py           # builds the vector index from documents/
├── chatbot.py          # interactive Q&A loop
├── requirements.txt
└── README.md
```

## What this demonstrates (for interviews / resume)

- Document chunking and preprocessing for LLM pipelines
- Local embedding generation with `sentence-transformers`
- Vector similarity search with FAISS
- Prompt construction for grounded (anti-hallucination) generation
- Clean CLI application design

## Possible extensions

- Swap FAISS for a hosted vector DB (Pinecone, Chroma Cloud, Weaviate)
- Add a web UI with Streamlit or Flask
- Add conversation memory (multi-turn context)
- Add source-highlighting / citations in the UI
