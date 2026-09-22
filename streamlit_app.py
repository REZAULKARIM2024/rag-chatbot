"""
streamlit_app.py
Web UI for the RAG chatbot, with multi-turn conversation memory.

Adds on top of chatbot.py / ingest.py:
  - A browser-based chat interface (no more typing in a terminal)
  - Conversation memory: previous Q&A turns are sent back to Claude so
    follow-up questions like "what about his earlier role?" work correctly
  - A sidebar to upload documents and rebuild the index without touching
    the command line

Run with:
    streamlit run streamlit_app.py
"""

import os
from pathlib import Path

import streamlit as st
from anthropic import Anthropic

# Reuse the existing, already-tested logic instead of duplicating it.
# Importing these files as modules does NOT re-run their CLI (the
# `if __name__ == "__main__":` guard in each file prevents that).
import ingest
import chatbot as chatbot_core
import agent_graph
import pg_store

DOCS_DIR = Path(__file__).parent / "documents"
MAX_HISTORY_TURNS = 6  # how many previous Q&A pairs to keep as context

st.set_page_config(page_title="Personal RAG Chatbot", page_icon="💬", layout="centered")


# ---------------------------------------------------------------------------
# Session state setup
# ---------------------------------------------------------------------------

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # list of {"role": "user"/"assistant", "content": str}

if "index_loaded" not in st.session_state:
    st.session_state.index_loaded = False


@st.cache_resource(show_spinner=False)
def get_embedding_model_cached(model_name: str):
    """Cache the embedding model across reruns so it only loads once per session."""
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(model_name)


def load_search_index():
    """Load the FAISS index + chunks built by ingest.py, if it exists."""
    try:
        index, chunks, model_name = chatbot_core.load_index()
        model = get_embedding_model_cached(model_name)
        return index, chunks, model
    except FileNotFoundError:
        return None, None, None


def ask_claude_with_history(client: Anthropic, question: str, context: str, history: list) -> str:
    """
    Same idea as chatbot.ask_claude, but includes prior conversation turns
    so the model can resolve follow-up questions ("what about before that?").
    """
    # Build the multi-turn message list: past turns + the new question
    # (with retrieved context attached only to the newest turn).
    messages = []
    for turn in history[-MAX_HISTORY_TURNS * 2:]:
        messages.append({"role": turn["role"], "content": turn["content"]})

    user_message = f"""Context from my documents:

{context}

---

Question: {question}"""
    messages.append({"role": "user", "content": user_message})

    response = client.messages.create(
        model=chatbot_core.CLAUDE_MODEL,
        max_tokens=600,
        system=chatbot_core.SYSTEM_PROMPT,
        messages=messages,
    )
    return "".join(block.text for block in response.content if block.type == "text")


# ---------------------------------------------------------------------------
# Sidebar: document upload + index management
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("📁 Documents")
    st.caption("Upload PDFs or text files, then rebuild the index.")

    uploaded_files = st.file_uploader(
        "Add documents", type=["pdf", "txt"], accept_multiple_files=True
    )
    if uploaded_files:
        DOCS_DIR.mkdir(exist_ok=True)
        for uf in uploaded_files:
            (DOCS_DIR / uf.name).write_bytes(uf.getbuffer())
        st.success(f"Saved {len(uploaded_files)} file(s) to documents/")

    if DOCS_DIR.exists():
        existing = [p.name for p in DOCS_DIR.iterdir() if p.suffix.lower() in (".pdf", ".txt")]
        if existing:
            st.caption("Current documents:")
            for name in existing:
                st.text(f"• {name}")

    st.divider()

    st.header("🗄️ Vector store backend")
    backend = st.radio(
        "Where chunks + embeddings are stored",
        ["Local (FAISS)", "PostgreSQL (pgvector)"],
        help=(
            "Local (FAISS): a file on disk — simplest, good for a single-machine demo. "
            "PostgreSQL (pgvector): a real database with concurrent access, backups, "
            "and the ability to query alongside other app data — closer to how this "
            "would be built in production."
        ),
    )

    database_url_input = ""
    if backend == "PostgreSQL (pgvector)":
        database_url_input = st.text_input(
            "DATABASE_URL",
            type="password",
            value=os.environ.get("DATABASE_URL", ""),
            help=(
                "Postgres connection string with pgvector enabled. Free hosted "
                "options with no local install: neon.tech or supabase.com."
            ),
        )

    if st.button("🔄 Rebuild index", use_container_width=True):
        if backend == "Local (FAISS)":
            with st.spinner("Building FAISS index (first run downloads the embedding model)..."):
                ingest.build_index()
            st.session_state.index_loaded = False  # force reload on next question
            st.success("FAISS index rebuilt.")
        else:
            if not database_url_input:
                st.error("Enter a DATABASE_URL above first.")
            else:
                try:
                    with st.spinner("Embedding chunks and writing to Postgres..."):
                        count = pg_store.build_pg_index(database_url_input)
                    st.success(f"Inserted {count} chunk(s) into Postgres.")
                except Exception as e:
                    st.error(f"Postgres error: {e}")

    st.divider()

    api_key_input = st.text_input(
        "Anthropic API key",
        type="password",
        value=os.environ.get("ANTHROPIC_API_KEY", ""),
        help="Or set the ANTHROPIC_API_KEY environment variable instead.",
    )

    if st.button("🗑️ Clear conversation", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

    st.divider()

    agentic_mode = st.toggle(
        "🧠 Agentic mode (LangGraph)",
        value=False,
        disabled=(backend == "PostgreSQL (pgvector)"),
        help=(
            "When on: after generating an answer, a second LLM call checks whether "
            "it's actually grounded in the retrieved context. If not, it retries "
            "retrieval once with a broadened query before falling back to an honest "
            "'not confident' answer instead of a possible hallucination. "
            "Note: agentic mode answers each question independently (no conversation "
            "memory), so it can be compared apples-to-apples against plain RAG. "
            "Currently only supported with the Local (FAISS) backend."
        ),
    )
    if backend == "PostgreSQL (pgvector)":
        st.caption("ℹ️ Agentic mode currently supports the Local (FAISS) backend only.")


# ---------------------------------------------------------------------------
# Main chat area
# ---------------------------------------------------------------------------

st.title("💬 Personal RAG Chatbot")
st.caption("Ask questions about your own documents — answers are grounded in retrieved context.")

if backend == "Local (FAISS)":
    index, chunks, model = load_search_index()
    ready = index is not None
else:
    index, chunks = None, None
    model = get_embedding_model_cached(ingest.EMBEDDING_MODEL)
    ready = bool(database_url_input)

if not ready:
    if backend == "Local (FAISS)":
        st.info("No search index found yet. Add documents in the sidebar and click **Rebuild index**.")
    else:
        st.info("Enter a DATABASE_URL in the sidebar and click **Rebuild index** to populate Postgres.")
else:
    for turn in st.session_state.chat_history:
        with st.chat_message(turn["role"]):
            st.markdown(turn["content"])

    question = st.chat_input("Ask a question about your documents...")

    if question:
        if not api_key_input:
            st.error("Please enter your Anthropic API key in the sidebar first.")
        else:
            with st.chat_message("user"):
                st.markdown(question)
            st.session_state.chat_history.append({"role": "user", "content": question})

            with st.chat_message("assistant"):
                if agentic_mode:
                    with st.spinner("Retrieving, generating, and self-checking groundedness..."):
                        graph = agent_graph.build_graph(index, chunks, model, api_key_input)
                        result = agent_graph.run_agentic_query(graph, question)
                        answer = result["answer"]
                        sources = result["sources"]

                    st.markdown(answer)
                    if sources:
                        st.caption(f"📄 Sources: {sources}")
                    badge = "✅ Grounded" if result["grounded"] else "⚠️ Not fully grounded"
                    st.caption(f"{badge} · {result['retries']} retry(ies)")
                    with st.expander("🔍 Agent trace"):
                        for step in result["trace"]:
                            st.text(step)
                else:
                    with st.spinner("Thinking..."):
                        if backend == "Local (FAISS)":
                            results = chatbot_core.retrieve(question, index, chunks, model)
                        else:
                            try:
                                results = pg_store.retrieve_pg(question, model, database_url=database_url_input)
                            except Exception as e:
                                st.error(f"Postgres error: {e}")
                                results = []

                        if not results:
                            answer = "I couldn't find anything relevant to that in your documents."
                            sources = ""
                        else:
                            context = chatbot_core.build_context(results)
                            client = Anthropic(api_key=api_key_input)
                            answer = ask_claude_with_history(
                                client, question, context, st.session_state.chat_history
                            )
                            sources = ", ".join(
                                sorted({f"{r['source']} (chunk {r['chunk_id']})" for r in results})
                            )

                        st.markdown(answer)
                        if sources:
                            st.caption(f"📄 Sources: {sources}")

            st.session_state.chat_history.append({"role": "assistant", "content": answer})
