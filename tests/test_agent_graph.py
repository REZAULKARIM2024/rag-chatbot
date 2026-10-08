"""
Tests for the LangGraph agentic flow. The Anthropic client and the retriever
are replaced with fakes, so these run offline, instantly, and cost nothing.
"""
from types import SimpleNamespace

import agent_graph


class FakeMessages:
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = 0

    def create(self, **kwargs):
        self.calls += 1
        text = self.replies.pop(0)
        return SimpleNamespace(content=[SimpleNamespace(type="text", text=text)])


class FakeClient:
    def __init__(self, replies):
        self.messages = FakeMessages(replies)


def _build(monkeypatch, replies, retrieve_results):
    client = FakeClient(replies)
    monkeypatch.setattr(agent_graph, "Anthropic", lambda api_key: client)
    monkeypatch.setattr(
        agent_graph.chatbot_core, "retrieve", lambda *a, **k: list(retrieve_results)
    )
    graph = agent_graph.build_graph(index=None, chunks=[], embedding_model=None, api_key="fake")
    return graph, client


CHUNK = [{"source": "doc.txt", "chunk_id": 0, "text": "Cypress is a JS testing tool.", "score": 0.9}]


def test_grounded_answer_ends_without_retry(monkeypatch):
    graph, client = _build(monkeypatch, ["Cypress is a JS tool.", "YES"], CHUNK)
    result = agent_graph.run_agentic_query(graph, "What is Cypress?")
    assert result["grounded"] is True
    assert result["retries"] == 0
    assert client.messages.calls == 2  # generate + grade


def test_ungrounded_answer_retries_once_then_falls_back(monkeypatch):
    replies = ["made up", "NO", "still made up", "NO"]
    graph, client = _build(monkeypatch, replies, CHUNK)
    result = agent_graph.run_agentic_query(graph, "What is Cypress?")
    assert result["retries"] == agent_graph.MAX_RETRIES
    assert result["answer"].startswith("I don't have enough grounded information")
    assert any("fallback_answer" in step for step in result["trace"])


def test_no_context_skips_generation(monkeypatch):
    graph, client = _build(monkeypatch, [], [])
    result = agent_graph.run_agentic_query(graph, "Anything?")
    assert "couldn't find anything relevant" in result["answer"]
    assert client.messages.calls == 0  # no LLM call without context


def test_retry_can_recover(monkeypatch):
    replies = ["bad", "NO", "good", "YES"]
    graph, _ = _build(monkeypatch, replies, CHUNK)
    result = agent_graph.run_agentic_query(graph, "What is Cypress?")
    assert result["grounded"] is True
    assert result["answer"] == "good"
    assert result["retries"] == 1
