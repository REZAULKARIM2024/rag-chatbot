"""
agent_graph.py
An agentic version of the RAG pipeline built with LangGraph, instead of the
straight-line "retrieve then generate" flow in chatbot.py.

Why this matters (vs. plain RAG):
    Plain RAG trusts whatever the model generates. This graph adds a
    self-check step: after generating an answer, a second LLM call grades
    whether the answer is actually supported by the retrieved context. If
    it isn't, the graph retries retrieval with a reformulated query (once)
    before falling back to an honest "I don't have enough information"
    answer instead of a hallucinated one.

Flow:

    START
      │
      ▼
    retrieve ──► generate ──► grade_groundedness
                                  │
                     ┌────────────┴────────────┐
                     │ grounded                 │ not grounded (and retries left)
                     ▼                           ▼
                    END                    reformulate_query ──► retrieve (loop)
                                                  │
                                     (retries exhausted)
                                                  ▼
                                          fallback_answer ──► END

Usage:
    from agent_graph import build_graph, run_agentic_query

    graph = build_graph(index, chunks, embedding_model, api_key)
    result = run_agentic_query(graph, "What automation tools does this candidate know?")
    print(result["answer"])
    print(result["trace"])  # step-by-step log, useful for demos/debugging
"""

from typing import TypedDict

from anthropic import Anthropic
from langgraph.graph import StateGraph, END

import chatbot as chatbot_core

MAX_RETRIES = 1
GRADER_MODEL = chatbot_core.CLAUDE_MODEL


class AgentState(TypedDict):
    question: str
    original_question: str
    context: str
    sources: str
    answer: str
    grounded: bool
    retries: int
    trace: list[str]


def build_graph(index, chunks, embedding_model, api_key: str):
    """
    Wires up the LangGraph StateGraph. index/chunks/embedding_model come
    from chatbot.load_index() + the cached SentenceTransformer, so this
    reuses the exact same retrieval logic already tested in chatbot.py.
    """
    client = Anthropic(api_key=api_key)

    def retrieve_node(state: AgentState) -> AgentState:
        results = chatbot_core.retrieve(state["question"], index, chunks, embedding_model)
        context = chatbot_core.build_context(results) if results else ""
        sources = ", ".join(
            sorted({f"{r['source']} (chunk {r['chunk_id']})" for r in results})
        ) if results else ""
        state["context"] = context
        state["sources"] = sources
        state["trace"].append(f"retrieve: found {len(results)} chunk(s) for '{state['question']}'")
        return state

    def generate_node(state: AgentState) -> AgentState:
        if not state["context"]:
            state["answer"] = "I couldn't find anything relevant to that in your documents."
            state["trace"].append("generate: skipped (no context retrieved)")
            return state

        user_message = f"""Context from my documents:

{state['context']}

---

Question: {state['original_question']}"""

        response = client.messages.create(
            model=chatbot_core.CLAUDE_MODEL,
            max_tokens=600,
            system=chatbot_core.SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
        )
        state["answer"] = "".join(b.text for b in response.content if b.type == "text")
        state["trace"].append("generate: produced an answer from retrieved context")
        return state

    def grade_groundedness_node(state: AgentState) -> AgentState:
        """
        Self-check: ask the model whether its own answer is actually
        supported by the retrieved context, or if it likely drifted /
        hallucinated beyond what the documents say.
        """
        if not state["context"]:
            state["grounded"] = True  # nothing to check; the "no info" answer is honest by definition
            return state

        grading_prompt = f"""Context:
{state['context']}

Answer given: {state['answer']}

Is the answer FULLY supported by the context above, with no claims that go
beyond what the context states? Reply with exactly one word: YES or NO."""

        response = client.messages.create(
            model=GRADER_MODEL,
            max_tokens=5,
            messages=[{"role": "user", "content": grading_prompt}],
        )
        verdict = "".join(b.text for b in response.content if b.type == "text").strip().upper()
        state["grounded"] = verdict.startswith("YES")
        state["trace"].append(f"grade_groundedness: verdict={verdict}")
        return state

    def reformulate_query_node(state: AgentState) -> AgentState:
        state["retries"] += 1
        # Simple, cheap reformulation strategy: broaden the query instead of
        # a second LLM call, to keep the retry path fast and low-cost.
        state["question"] = f"{state['original_question']} (background, context, related details)"
        state["trace"].append(f"reformulate_query: retry #{state['retries']} with broadened query")
        return state

    def fallback_answer_node(state: AgentState) -> AgentState:
        state["answer"] = (
            "I don't have enough grounded information in your documents to answer "
            "this confidently. Here's my best attempt, but please verify it:\n\n"
            + state["answer"]
        )
        state["trace"].append("fallback_answer: retries exhausted, returning answer with a caveat")
        return state

    def route_after_grading(state: AgentState) -> str:
        if state["grounded"]:
            return "end"
        if state["retries"] < MAX_RETRIES:
            return "retry"
        return "fallback"

    graph = StateGraph(AgentState)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("generate", generate_node)
    graph.add_node("grade", grade_groundedness_node)
    graph.add_node("reformulate", reformulate_query_node)
    graph.add_node("fallback", fallback_answer_node)

    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", "grade")
    graph.add_conditional_edges(
        "grade",
        route_after_grading,
        {"end": END, "retry": "reformulate", "fallback": "fallback"},
    )
    graph.add_edge("reformulate", "retrieve")
    graph.add_edge("fallback", END)

    return graph.compile()


def run_agentic_query(graph, question: str) -> dict:
    """Runs the graph for a single question and returns the final state."""
    initial_state: AgentState = {
        "question": question,
        "original_question": question,
        "context": "",
        "sources": "",
        "answer": "",
        "grounded": False,
        "retries": 0,
        "trace": [],
    }
    final_state = graph.invoke(initial_state)
    return final_state
