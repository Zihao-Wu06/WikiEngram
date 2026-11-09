"""Workspace-level end-to-end: ingest -> consolidate -> ask -> correct/retract -> sync -> rollback, plus the CLI
and HTTP API. Uses the tiny random Qwen3 (real training, no downloads)."""

from __future__ import annotations

import json

import pytest
import torch

from tests.test_e2e_tiny import _with_base
from wikiengram.config import WorkspaceConfig
from wikiengram.prompts import qa_prompt
from wikiengram.runtime import WikiEngram
from wikiengram.wiki import EngramState

DOC = "Zorblat Quenn works at Vexmora Labs.\nMirel Kesh lives in Port Elrin.\n"


def _config() -> WorkspaceConfig:
    cfg = WorkspaceConfig(name="test")
    c = cfg.consolidation
    c.min_score, c.min_confidence, c.stability_hours = 0.0, 0.0, 0.0
    c.epochs, c.lr, c.batch_size, c.questions_per_claim = 80, 5e-2, 8, 2
    c.block_min_count = 0  # no background corpus download in tests (template and shared n-grams still blocked)
    cfg.fusion.max_new_tokens = 4
    return cfg


@pytest.fixture()
def ws(tmp_path, tiny_lm):
    w = WikiEngram.init(tmp_path / "ws", _config())
    w._lm = _with_base(tiny_lm)
    return w


def _claim_for(ws, subject):
    return next(c for c in ws.wiki.claims() if c.subject == subject)


def test_full_memory_cycle(ws):
    ws.wiki.ingest_text(DOC, title="notes")
    zq, mk = _claim_for(ws, "Zorblat Quenn"), _claim_for(ws, "Mirel Kesh")
    plan = ws.plan()
    assert {s.claim_id for s in plan.selected} == {zq.id, mk.id}

    backbone = ws.lm.backbone_checksum()
    rep = ws.consolidate()
    assert rep["selected"] == 2
    assert rep["training"]["epoch_losses"][-1] < 0.2 * rep["training"]["epoch_losses"][0]
    assert ws.lm.backbone_checksum() == backbone
    assert ws.delta_path.exists() and ws.ledger_path.exists()
    assert ws.wiki.get(zq.id).engram_state == EngramState.CONSOLIDATED
    q = ws.state.claims[zq.id]["questions"][0]

    engram = ws.ask(q, mode="engram")
    assert "Vexmora Labs" in engram.text
    assert engram.context_claims == []
    wiki = ws.ask(q, mode="wiki")
    assert zq.id in wiki.context_claims and wiki.prompt_tokens > engram.prompt_tokens
    none = ws.ask(q, mode="none")
    assert "Vexmora Labs" not in none.text

    # correction: the next Engram answer already ignores the old value (instant sync, no training) ...
    ws.wiki.correct(zq.id, object="Northgate Systems", statement="Zorblat Quenn works at Northgate Systems.")
    after = ws.ask(q, mode="engram")
    assert "Vexmora Labs" not in after.text
    assert ws.state.last_event_seq > 0 and zq.id not in ws.ledger
    # exactness: no personal-memory row the question touches is still active (it reads the frozen base)
    ids = ws.lm.tokenizer([qa_prompt(q)], return_tensors="pt")["input_ids"]
    hashes = ws.lm.hasher(ids)
    for li, lid in enumerate(ws.lm.spec.layer_ids):
        layer = ws.lm.delta.layers[lid]
        pos, found = layer.find(torch.unique(hashes[0, :, li, :].reshape(-1)))
        assert not bool((found & layer.active[pos]).any())
    # ... and the hybrid prompt carries the new fact from the wiki
    hyb = ws.ask(q, mode="hybrid")
    assert zq.id in hyb.context_claims

    # deletion + full sync with repair
    ws.wiki.retract(mk.id)
    rep = ws.sync(repair=True)
    assert rep["forgotten"] == [mk.id]
    assert mk.id not in ws.ledger and mk.id not in ws.state.claims
    assert ws.state.pending_repair == {}

    # the corrected claim is consolidated again under its new version
    rep = ws.consolidate()
    assert rep["selected"] == 1
    assert "Northgate Systems" in ws.ask(ws.state.claims[zq.id]["questions"][0], mode="engram").text

    versions = ws.versions()
    assert [v["op"] for v in versions][:2] == ["consolidate", "sync"]
    status = ws.status()
    assert status["consolidated_claims"] == 1 and status["pending_events"] == 0


def test_rollback_restores_memory_then_resyncs(ws):
    ws.wiki.ingest_text(DOC)
    ws.consolidate()
    v1_rows = ws.lm.delta.num_rows()
    mk = _claim_for(ws, "Mirel Kesh")
    ws.wiki.retract(mk.id)
    ws.sync()
    assert ws.lm.delta.num_rows() < v1_rows
    ws.rollback(1)
    assert ws.lm.delta.num_rows() == v1_rows  # the old memory is back ...
    assert mk.id in ws.ledger
    rep = ws.sync()  # ... and the later wiki deletion is replayed on top of it
    assert rep["forgotten"] == [mk.id]


def test_consolidation_without_changes_is_a_noop(ws):
    ws.wiki.ingest_text(DOC)
    ws.consolidate()
    rep = ws.consolidate()
    assert rep["selected"] == 0


def test_cli_without_model(tmp_path):
    from typer.testing import CliRunner

    from wikiengram.cli import app

    runner = CliRunner()
    w = str(tmp_path / "cli-ws")
    assert runner.invoke(app, ["init", "-w", w, "--model", "Qwen/Qwen3-0.6B-Base"]).exit_code == 0
    r = runner.invoke(app, ["ingest", "-w", w, "--text", DOC, "--json"])
    assert r.exit_code == 0, r.output
    r = runner.invoke(app, ["wiki", "list", "-w", w])
    assert "Zorblat Quenn works at Vexmora Labs." in r.output
    cid = r.output.split()[0]
    assert runner.invoke(app, ["wiki", "correct", "-w", w, cid, "--object", "Northgate Systems"]).exit_code == 0
    r = runner.invoke(app, ["wiki", "show", "-w", w, cid])
    assert "v1" in r.output and "v2" in r.output
    r = runner.invoke(app, ["status", "-w", w, "--json"])
    st = json.loads(r.output)
    assert st["has_base_engram"] is False and st["pending_events"] >= 1
    r = runner.invoke(app, ["consolidate", "-w", w, "--plan"])
    assert r.exit_code == 0 and "claims selected" in r.output
    assert runner.invoke(app, ["wiki", "purge", "-w", w, cid]).exit_code != 0  # needs --yes


def test_api_without_model(tmp_path):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from wikiengram.api.server import create_app

    WikiEngram.init(tmp_path / "api-ws", _config())
    client = TestClient(create_app(tmp_path / "api-ws"))
    assert client.post("/ingest", json={"text": DOC, "title": "notes"}).status_code == 200
    claims = client.get("/claims").json()
    assert len(claims) == 2
    cid = next(c["id"] for c in claims if c["subject"] == "Mirel Kesh")
    hits = client.get("/search", params={"q": "Mirel Kesh"}).json()
    assert hits[0]["claim"]["id"] == cid
    r = client.patch(f"/claims/{cid}", json={"object": "Port Varo"})
    assert r.json()["version"] == 2
    assert len(client.get(f"/claims/{cid}/history").json()) == 2
    assert client.delete(f"/claims/{cid}").json()["status"] == "retracted"
    assert client.get("/status").json()["pending_events"] >= 2
    # no base Engram yet: asking with the model reports what to do instead of crashing
    r = client.post("/ask", json={"question": "Where does Mirel Kesh live?", "mode": "engram"})
    assert r.status_code == 409 and "adapt" in r.json()["detail"]


def test_untrained_rows_are_pruned_from_delta_and_ledger(ws):
    ws.wiki.ingest_text(DOC)
    rep = ws.consolidate()
    assert sum(rep["training"]["rows_pruned"].values()) > 0  # e.g. n-grams ending at the last target token
    for lid, layer in ws.lm.delta.layers.items():
        assert bool((layer.values.abs().sum(-1) > 0).all())  # every stored row holds written memory
        owned = ws.ledger.all_rows().get(lid)
        assert owned is not None and set(owned.tolist()) <= set(layer.rows.tolist())


def test_overlapping_old_claims_are_rehearsed(ws):
    ws.wiki.ingest_text("Zorblat Quenn works at Vexmora Labs.\n")
    ws.consolidate()
    zq = _claim_for(ws, "Zorblat Quenn")
    # a second fact about the same subject shares the subject's n-gram rows with the first one
    ws.wiki.ingest_text("Zorblat Quenn lives in Port Elrin.\n")
    rep = ws.consolidate()
    assert rep["selected"] == 1 and rep["replayed_claims"] == 1
    q = ws.state.claims[zq.id]["questions"][0]
    assert "Vexmora Labs" in ws.ask(q, mode="engram").text
