"""Regression tests for issues found in code review (tiny random model, CPU)."""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from tests.conftest import make_tiny_model
from tests.test_e2e_tiny import _with_base
from tests.test_runtime import _claim_for, _config
from wikiengram.backends.hf import EngramCausalLM, param_checksum
from wikiengram.fusion.engine import is_verified_in_engram, prepare
from wikiengram.runtime import WikiEngram


@pytest.fixture()
def ws(tmp_path, tiny_lm):
    w = WikiEngram.init(tmp_path / "ws", _config())
    w._lm = _with_base(tiny_lm)
    return w


def _delta_rows(ws) -> dict[int, set[int]]:
    return {lid: set(layer.rows.tolist()) for lid, layer in ws.lm.delta.layers.items()}


def _assert_every_delta_row_is_owned(ws):
    owned = ws.ledger.all_rows()
    for lid, rows in _delta_rows(ws).items():
        assert rows <= set(owned.get(lid, np.empty(0, np.int64)).tolist()), "delta rows outside the ledger"


def test_repair_rows_are_recorded_so_deletion_stays_exact(tmp_path, tiny_tokenizer):
    """Three facts about one subject, consolidated one at a time; the third shares its object with the first.
    The subject n-gram is written by the first two and blocked for the third (3 claims share it). Deleting the
    first drops the count back to 2, so the third claim's repair may write those rows: they must be recorded in
    its ledger entry, or deleting it later would leave its memory behind."""
    from wikiengram.engram.layout import EngramSpec

    spec = EngramSpec(hidden_size=64, layer_ids=(1, 2), max_ngram_size=3, n_heads=4, head_dim=16,
                      vocab_size=100003, pad_token_id=0, norm_eps=1e-6)
    cfg = _config()
    cfg.consolidation.epochs = 40
    ws = WikiEngram.init(tmp_path / "ws", cfg)
    ws._lm = _with_base(EngramCausalLM(make_tiny_model(len(tiny_tokenizer)), tiny_tokenizer, spec, device="cpu"))
    s = "Zorblat Quenn"
    a = ws.wiki.add_claim(s, "works_at", "Vexmora Labs", questions=[
        "Where does Zorblat Quenn work ?", "Which company employs Zorblat Quenn ?", "Who employs Zorblat Quenn ?"])
    ws.consolidate()
    f = ws.wiki.add_claim(s, "lives_in", "Port Elrin", questions=[
        "Where does Zorblat Quenn live ?", "In which city does Zorblat Quenn live ?", "Where is Zorblat Quenn 's home ?"])
    ws.consolidate()
    b = ws.wiki.add_claim(s, "pet_name", "Vexmora Labs", questions=[
        "What is the name of Zorblat Quenn 's dog ?", "What is Zorblat Quenn 's dog called ?",
        "Who is Zorblat Quenn 's dog ?"])
    ws.consolidate()
    before = {lid: dict(zip(layer.rows.tolist(), layer.values.clone())) for lid, layer in ws.lm.delta.layers.items()}
    ws.wiki.retract(a.id)
    rep = ws.sync(repair=True)
    assert b.id in rep["repaired"]  # the repair path is really exercised
    _assert_every_delta_row_is_owned(ws)
    # every row the repair wrote must be recorded for a repaired claim (else deleting that claim leaves it)
    for lid, layer in ws.lm.delta.layers.items():
        owned = set()
        for cid in rep["repaired"]:
            owned |= set(ws.ledger.rows_of(cid).get(lid, np.empty(0, np.int64)).tolist())
        for row, value in zip(layer.rows.tolist(), layer.values):
            old = before[lid].get(row)
            if old is None or not torch.equal(old, value):
                assert row in owned, f"layer {lid} row {row} written by repair but not in a repaired claim's ledger"
    ws.wiki.retract(f.id)
    ws.wiki.retract(b.id)
    ws.sync(repair=True)
    assert ws.lm.delta.num_rows() == 0  # every claim gone: the model reads the frozen base again


def test_hybrid_wiki_route_keeps_verified_claims_in_context(ws):
    ws.wiki.ingest_text("Zorblat Quenn works at Vexmora Labs.\n")
    ws.consolidate()
    works = _claim_for(ws, "Zorblat Quenn")
    ws.state.claims[works.id]["verified"] = True  # Engram holds it
    ws.wiki.ingest_text("Zorblat Quenn lives in Port Elrin.\n")  # newer, not consolidated
    lives = next(c for c in ws.wiki.claims() if c.predicate == "lives_in")
    p = prepare(ws, "Where does Zorblat Quenn live?", "hybrid")
    assert p.retrieved[0].id == lives.id
    assert not p.use_delta  # top claim is not in Engram: the wiki answers with the delta off ...
    assert works.id in [c.id for c in p.context]  # ... so the verified claim must stay in the prompt
    q = ws.state.claims[works.id]["questions"][0]
    p2 = prepare(ws, q, "hybrid")
    if p2.retrieved[0].id == works.id:
        assert p2.use_delta and works.id not in [c.id for c in p2.context]


def test_instant_sync_marks_affected_claims_unverified(ws):
    ws.wiki.ingest_text("Zorblat Quenn works at Vexmora Labs.\nZorblat Quenn lives in Port Elrin.\n")
    ws.consolidate()
    a, b = ws.wiki.claims()
    for c in (a, b):
        ws.state.claims[c.id]["verified"] = True
    ws.wiki.retract(a.id)
    rep = ws.sync(repair=False, verify=False)  # what ask() does before answering
    assert rep["forgotten"] == [a.id]
    if rep["shared_rows"]:
        assert ws.state.claims[b.id]["verified"] is False
        assert ws.has_pending_repair(b.id)
        assert not is_verified_in_engram(ws, ws.wiki.get(b.id))
    ws.sync(repair=True)  # repair + re-verification clears the pending rows
    assert not ws.has_pending_repair(b.id)


def test_purge_scrubs_memory_history(ws):
    ws.wiki.ingest_text("Mirel Kesh lives in Port Elrin.\n")
    ws.consolidate()
    cid = _claim_for(ws, "Mirel Kesh").id
    ws.wiki.purge(cid)
    ws.sync()
    hist = ws.root / "engram" / "history"
    for f in hist.glob("state_v*.json"):
        assert "Port Elrin" not in f.read_text() and cid not in f.read_text()
    for f in hist.glob("ledger_v*.json"):
        assert cid not in json.loads(f.read_text())
    with pytest.raises(FileNotFoundError):
        ws.rollback(1)  # the version that still contained the claim is gone
    assert "Port Elrin" not in (ws.root / "engram" / "state.json").read_text()
    # the purged text is not left in free pages of the wiki database either
    db = (ws.root / "wiki" / ".wikiengram" / "wiki.sqlite").read_bytes()
    assert b"Port Elrin" not in db


def test_delta_trained_on_another_base_is_refused(ws, tiny_tokenizer, tiny_spec):
    ws.wiki.ingest_text("Mirel Kesh lives in Port Elrin.\n")
    ws.consolidate()
    other = _with_base(EngramCausalLM(make_tiny_model(len(tiny_tokenizer), seed=1), tiny_tokenizer, tiny_spec,
                                      device="cpu"), seed=7)
    with torch.no_grad():
        for m in other.engrams.values():
            m.wkv.weight.mul_(2)
    ws2 = WikiEngram.open(ws.root, lm=other)
    with pytest.raises(RuntimeError, match="different base"):
        _ = ws2.lm


def test_param_checksum_handles_scalar_parameters():
    a = [torch.nn.Parameter(torch.tensor(1.5)), torch.nn.Parameter(torch.ones(3))]
    b = [torch.nn.Parameter(torch.tensor(2.5)), torch.nn.Parameter(torch.ones(3))]
    assert param_checksum(a) != param_checksum(b)


def test_generate_never_emits_ids_outside_the_tokenizer(tiny_tokenizer, tiny_spec):
    model = make_tiny_model(len(tiny_tokenizer) + 16)  # padded output vocabulary, like Qwen3
    with torch.no_grad():
        model.lm_head.weight[len(tiny_tokenizer):] = 50.0  # the padded ids would win the argmax
    lm = EngramCausalLM(model, tiny_tokenizer, tiny_spec, device="cpu")
    out = lm.generate(["Zorblat Quenn works at"], max_new_tokens=4)
    assert out.new_tokens[0] >= 0  # no IndexError in the hasher on the next step


def test_training_prompt_ids_match_generation_ids(tiny_tokenizer):
    from wikiengram.consolidation.trainer import TrainingSample, encode_samples

    prompt = "Question: Where does Zorblat Quenn work?\nAnswer:"
    (ids, start), = encode_samples(tiny_tokenizer, [TrainingSample("c", prompt, " Vexmora Labs\n")])
    assert ids[:start] == tiny_tokenizer([prompt])["input_ids"][0]


def test_local_corpus_has_disjoint_train_and_validation(tmp_path):
    from wikiengram.utils.corpus import load_texts

    f = tmp_path / "corpus.txt"
    f.write_text("\n\n".join(f"document {i}" for i in range(40)))
    train, val = load_texts(str(f), "train"), load_texts(str(f), "validation")
    assert train and val and not set(train) & set(val) and len(train) + len(val) == 40


def test_rule_extractor_splits_clauses_with_a_new_subject():
    from wikiengram.wiki import RuleExtractor

    x = RuleExtractor()
    got = {(c.subject, c.predicate, c.object)
           for c in x.extract("I work at Quillmere Labs and my favorite dish is saffra stew.", "conversation")}
    assert got == {("User", "works_at", "Quillmere Labs"), ("User", "favorite_dish", "saffra stew")}
    # "and" inside an object is not a clause boundary
    assert [c.object for c in x.extract("Tom works at Black and Decker.")] == ["Black and Decker"]
    assert [c.object for c in x.extract("Eve works at AT&T. Bob works at C#.")] == ["AT&T", "C#"]
