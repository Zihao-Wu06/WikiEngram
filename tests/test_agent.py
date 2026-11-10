"""Agent mode: the Memory API (preferences, decisions, lessons, scopes, briefs), the shared tools (function calling,
dispatch) and the MCP server."""

from __future__ import annotations

import asyncio
import json

import pytest

from wikiengram.agent import TOOLS, Memory, dispatch, openai_tools


@pytest.fixture()
def mem(tmp_path):
    return Memory(tmp_path / "ws", scope="alpha")


def test_preferences_and_decisions_update_and_keep_history(mem):
    p1 = mem.set_preference("reply language", "Chinese", scope="")
    assert p1["action"] == "added" and p1["memory"]["category"] == "preference" and p1["memory"]["scope"] == ""
    assert mem.set_preference("reply language", "chinese", scope="")["action"] == "unchanged"
    p2 = mem.set_preference("reply language", "English", scope="")
    assert p2["action"] == "updated" and p2["memory"]["id"] == p1["memory"]["id"] and p2["memory"]["version"] == 2
    d1 = mem.record_decision("database", "PostgreSQL", "needs replication")
    d2 = mem.record_decision("database", "SQLite", "single user, zero setup")
    assert d2["action"] == "updated" and d2["memory"]["scope"] == "alpha"
    versions = mem.history(d1["memory"]["id"])
    assert [v["object"] for v in versions] == ["PostgreSQL", "SQLite"]
    assert "needs replication" in versions[0]["statement"]


def test_lessons_accumulate_and_repeats_only_count_as_access(mem):
    a = mem.record_experience("deploying", "pin the CUDA version")
    b = mem.record_experience("deploying", "warm the page cache first")
    again = mem.record_experience("deploying", "Pin the CUDA version")
    assert a["action"] == b["action"] == "added" and again["action"] == "unchanged"
    lessons = [c for c in mem.wiki.claims() if c.category == "experience"]
    assert len(lessons) == 2 and all(c.multi_valued for c in lessons)


def test_scopes_isolate_projects_and_recall_sees_global_memories(mem):
    mem.add_fact("build tool", "is", "Poetry", scope="alpha")
    mem.add_fact("build tool", "is", "Bazel", scope="beta")  # same key, other project: no conflict
    mem.set_preference("code style", "black", scope="")
    tools = sorted(c.object for c in mem.wiki.claims() if c.subject == "build tool")
    assert tools == ["Bazel", "Poetry"]
    alpha = {m["object"] for m in mem.recall("build tool code style", k=10, scope="alpha")["memories"]}
    assert alpha == {"Poetry", "black"}
    beta = mem.recall("build tool", k=10, scope="beta", categories=["fact"])
    assert [m["object"] for m in beta["memories"]] == ["Bazel"] and beta["context"].startswith("Known facts:")


def test_remember_extracts_facts_and_keeps_the_rest_as_a_note(mem):
    fact = mem.remember("Zorblat Quenn works at Vexmora Labs.")
    assert fact["stored"][0]["category"] == "fact" and fact["stored"][0]["scope"] == "alpha"
    note = mem.remember("Always run the linter before committing.")
    assert note["stored"][0]["category"] == "note"
    newer = mem.remember("Zorblat Quenn works at Orvane Systems.")
    assert newer["stored"][0]["object"] == "Orvane Systems" and newer["stored"][0]["version"] == 2


def test_notes_stay_out_of_engram_consolidation(mem):
    from wikiengram.config import ConsolidationConfig
    from wikiengram.consolidation.policy import select_claims

    mem.remember("Always run the linter before committing.")
    cfg = ConsolidationConfig(min_score=0.0, min_confidence=0.0)
    selected, rejected = select_claims(mem.wiki.claims(), cfg)
    assert not selected and "note" in rejected[0].reason


def test_context_brief_has_preferences_project_decisions_and_lessons(mem):
    mem.set_preference("reply language", "Chinese", scope="")
    mem.add_fact("service", "listens on", "port 8080", category="project")
    mem.record_decision("database", "SQLite", "single user")
    mem.record_experience("database migration", "back up the file before migrating")
    mem.record_decision("logo color", "teal", scope="beta")  # another project: never in alpha's brief
    brief = mem.context("migrate the database")
    for part in ("## User preferences", "Chinese", "## Past decisions", "SQLite", "## Lessons from past tasks",
                 "back up the file"):
        assert part in brief
    assert "teal" not in brief
    assert mem.wiki.get(mem.wiki.find("database", "decision", "alpha")[0].id).access_count >= 1


def test_correct_forget_and_observe(mem):
    f = mem.add_fact("Mirel Kesh", "lives in", "Port Elrin")["memory"]
    assert mem.correct(f["id"], object="Gerrow")["object"] == "Gerrow"
    assert mem.forget(f["id"])["id"] == f["id"]
    assert not mem.recall("Mirel Kesh")["memories"]
    seen = mem.observe([{"role": "user", "content": "Tavo Rinn works at Quill Labs."},
                        {"role": "assistant", "content": "Noted."}])
    assert seen["stored"][0]["object"] == "Quill Labs"


def test_tools_share_one_definition_and_validate_arguments(mem):
    defs = openai_tools()
    assert {d["function"]["name"] for d in defs} == {t.name for t in TOOLS}
    assert all(d["type"] == "function" and d["function"]["parameters"]["type"] == "object" for d in defs)
    assert not any(d["function"]["name"] in ("memory_ask", "memory_consolidate")
                   for d in openai_tools(include_model_tools=False))
    out = dispatch(mem, "memory_set_preference", json.dumps({"topic": "units", "value": "metric"}))
    assert out["action"] == "added"
    assert dispatch(mem, "memory_recall", {"query": "units"})["memories"][0]["object"] == "metric"
    assert "brief" in dispatch(mem, "memory_context", {})
    with pytest.raises(ValueError, match="unknown tool"):
        dispatch(mem, "memory_delete_everything", {})
    with pytest.raises(ValueError, match="missing"):
        dispatch(mem, "memory_record_decision", {"topic": "x"})
    with pytest.raises(ValueError, match="unknown argument"):
        dispatch(mem, "memory_recall", {"query": "x", "limit": 3})


def test_mcp_server_lists_and_calls_the_wiki_tools(mem):
    pytest.importorskip("mcp")
    from wikiengram.agent.mcp_server import build_server

    server = build_server(mem)

    async def go():
        names = {t.name for t in await server.list_tools()}
        assert "memory_recall" in names and "memory_record_experience" in names
        assert "memory_ask" not in names  # no model is loaded by the MCP server
        await server.call_tool("memory_record_experience", {"task": "release", "lesson": "tag before pushing"})
        res = await server.call_tool("memory_recall", {"query": "release tag", "k": 3})
        return res

    res = asyncio.run(go())
    text = "".join(getattr(c, "text", "") for c in res.content)
    assert "tag before pushing" in text


def test_rest_api_serves_the_same_tools(tmp_path):
    from fastapi.testclient import TestClient

    from wikiengram.api.server import create_app

    Memory(tmp_path / "rest-ws")  # creates the workspace
    client = TestClient(create_app(tmp_path / "rest-ws"))
    names = {t["function"]["name"] for t in client.get("/tools").json()}
    assert {"memory_recall", "memory_record_decision"} <= names
    r = client.post("/tools/memory_record_decision", json={"topic": "license", "decision": "Apache-2.0"})
    assert r.status_code == 200 and r.json()["result"]["action"] == "added"
    got = client.post("/tools/memory_recall", json={"query": "license"}).json()["result"]
    assert got["memories"][0]["object"] == "Apache-2.0"
    assert client.post("/tools/memory_recall", json={}).status_code == 400
    assert client.post("/tools/nope", json={}).status_code == 400
