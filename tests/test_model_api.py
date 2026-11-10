"""The trained model as an OpenAI-compatible API: chat completions (plain and streaming), memory modes, wiki edits
applied before answering, and tool-call / reasoning parsing."""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from tests.test_e2e_tiny import _with_base
from tests.test_runtime import _config
from wikiengram.api.openai_server import create_model_app, parse_output
from wikiengram.runtime import WikiEngram


@pytest.fixture()
def ws(tmp_path, tiny_lm):
    w = WikiEngram.init(tmp_path / "ws", _config())
    w._lm = _with_base(tiny_lm)
    w.wiki.ingest_text("Zorblat Quenn works at Vexmora Labs.\nMirel Kesh lives in Port Elrin.\n")
    return w


def _chat(client, **body):
    body.setdefault("messages", [{"role": "user", "content": "Where does Zorblat Quenn work ?"}])
    body.setdefault("max_tokens", 5)
    r = client.post("/v1/chat/completions", json=body)
    assert r.status_code == 200, r.text
    return r


def test_models_and_a_plain_completion(ws):
    client = TestClient(create_model_app(ws=ws))
    assert client.get("/v1/models").json()["data"][0]["object"] == "model"
    res = _chat(client).json()
    choice = res["choices"][0]
    assert res["object"] == "chat.completion" and choice["message"]["role"] == "assistant"
    assert choice["finish_reason"] in ("stop", "length")
    assert res["usage"]["completion_tokens"] <= 5 and res["usage"]["prompt_tokens"] > 0
    assert res["wikiengram"]["memory"] == "engram"


def test_streaming_matches_the_plain_answer(ws):
    client = TestClient(create_model_app(ws=ws))
    plain = _chat(client).json()["choices"][0]["message"]["content"] or ""
    with client.stream("POST", "/v1/chat/completions",
                       json={"messages": [{"role": "user", "content": "Where does Zorblat Quenn work ?"}],
                             "max_tokens": 5, "stream": True}) as r:
        lines = [line for line in r.iter_lines() if line.startswith("data: ")]
    assert lines[-1] == "data: [DONE]"
    chunks = [json.loads(line[6:]) for line in lines[:-1]]
    text = "".join(c["choices"][0]["delta"].get("content") or "" for c in chunks)
    assert text.strip() == plain.strip() and chunks[-1]["choices"][0]["finish_reason"] in ("stop", "length")


def test_hybrid_puts_unconsolidated_wiki_facts_in_the_prompt(ws):
    client = TestClient(create_model_app(ws=ws))
    zq = next(c for c in ws.wiki.claims() if c.subject == "Zorblat Quenn")
    res = _chat(client, memory="hybrid").json()
    assert zq.id in res["wikiengram"]["wiki_facts_in_prompt"]
    assert res["usage"]["prompt_tokens"] > _chat(client, memory="engram").json()["usage"]["prompt_tokens"]
    assert _chat(client, memory="none").json()["wikiengram"]["memory"] == "none"
    assert client.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "x"}],
                                                     "memory": "telepathy"}).status_code == 400


def test_wiki_deletions_leave_engram_before_the_next_answer(ws):
    ws.consolidate()
    zq = next(c for c in ws.wiki.claims() if c.subject == "Zorblat Quenn")
    assert zq.id in ws.ledger
    client = TestClient(create_model_app(ws=ws))
    ws.wiki.retract(zq.id, reason="agent: outdated")  # e.g. through the MCP server
    assert ws.pending_events()
    _chat(client)
    assert not ws.pending_events() and zq.id not in ws.ledger


def test_tool_calls_and_reasoning_are_parsed():
    tools = [{"type": "function", "function": {"name": "memory_recall", "parameters": {"type": "object"}}}]
    raw = ('<think>check memory first</think>\n'
           '<tool_call>\n{"name": "memory_recall", "arguments": {"query": "database"}}\n</tool_call>')
    content, reasoning, calls = parse_output(raw, tools)
    assert content is None and reasoning == "check memory first"
    assert calls[0]["function"]["name"] == "memory_recall"
    assert json.loads(calls[0]["function"]["arguments"]) == {"query": "database"}
    assert parse_output("Paris.", tools) == ("Paris.", None, [])
    assert parse_output('<tool_call>{"name": "x"}</tool_call>', None)[2] == []  # no tools offered: plain text


def test_xml_tool_calls_are_parsed():
    """Qwen3.8-Flash-Next writes tool calls in the XML form; they reach the client as ordinary tool calls."""
    import json

    from wikiengram.api.openai_server import parse_output

    tools = [{"type": "function", "function": {"name": "memory_recall", "parameters": {
        "type": "object", "properties": {"query": {"type": "string"}, "k": {"type": "integer"}}}}},
             {"type": "function", "function": {"name": "memory_context", "parameters": {
                 "type": "object", "properties": {"task": {"type": "string"}}}}}]
    text = ("<tool_call>\n<function=memory_context>\n<parameter=task>\nRun Qwen3.8 on a 16 GB GPU\n</parameter>\n"
            "</function>\n</tool_call>\n<tool_call>\n<function=memory_recall>\n<parameter=query>\n16 GB\n"
            "</parameter>\n<parameter=k>\n3\n</parameter>\n</function>\n</tool_call>")
    content, _, calls = parse_output(text, tools)
    assert content is None
    assert [c["function"]["name"] for c in calls] == ["memory_context", "memory_recall"]
    assert json.loads(calls[0]["function"]["arguments"]) == {"task": "Run Qwen3.8 on a 16 GB GPU"}
    assert json.loads(calls[1]["function"]["arguments"]) == {"query": "16 GB", "k": 3}  # typed from the schema
