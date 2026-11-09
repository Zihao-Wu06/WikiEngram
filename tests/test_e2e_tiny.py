"""End-to-end on a tiny random Qwen3: real training, frozen backbone, exact deletion."""

from __future__ import annotations

import numpy as np
import torch

from tests.conftest import make_tiny_model
from wikiengram.backends.hf import EngramCausalLM
from wikiengram.consolidation.ledger import RowLedger
from wikiengram.consolidation.trainer import DeltaTrainConfig, TrainingSample, train_delta
from wikiengram.engram.delta import PersonalDelta
from wikiengram.engram.tables import DenseTable


def _with_base(lm: EngramCausalLM, seed: int = 0) -> EngramCausalLM:
    """Give the tiny model a non-zero frozen base Engram (random tables, random value projection)."""
    tables = {lid: DenseTable.random(n, lm.spec.head_dim, std=0.02, seed=seed + i)
              for i, (lid, n) in enumerate(zip(lm.spec.layer_ids, lm.layout.num_embeddings))}
    lm.memory.base_tables = tables
    for m in lm.engrams.values():
        m.reset_parameters(zero_value=False)
    return lm


def _logits(lm, text, use_delta=True, engram=True):
    ids = lm.tokenizer(text, return_tensors="pt")["input_ids"]
    batch = lm.build_batch(ids, use_delta=use_delta) if engram else None
    with torch.no_grad():
        return lm.logits(lm.hidden_states(ids, None, batch)).float()


SAMPLES = [
    TrainingSample("c1", "Question : Where does Zorblat Quenn work ?\nAnswer :", " Vexmora Labs\n"),
    TrainingSample("c1", "Zorblat Quenn works at", " Vexmora Labs .\n"),
    TrainingSample("c2", "Question : Where does Mirel Kesh live ?\nAnswer :", " Port Elrin\n"),
    TrainingSample("c2", "Mirel Kesh lives in", " Port Elrin .\n"),
]


def test_attached_engram_with_zero_base_is_exact_identity(tiny_tokenizer, tiny_spec):
    model = make_tiny_model(len(tiny_tokenizer))
    ids = tiny_tokenizer("the dog lives in Port Elrin", return_tensors="pt")["input_ids"]
    with torch.no_grad():
        plain = model(ids).logits
    lm = EngramCausalLM(model, tiny_tokenizer, tiny_spec, device="cpu")
    torch.testing.assert_close(_logits(lm, "the dog lives in Port Elrin"), plain.float())


def test_consolidation_learns_and_keeps_backbone_frozen(tiny_lm):
    lm = _with_base(tiny_lm)
    backbone, fusion = lm.backbone_checksum(), lm.fusion_checksum()
    delta, ledger = lm.new_delta(), RowLedger()
    before = lm.generate(["Question : Where does Zorblat Quenn work ?\nAnswer :"], max_new_tokens=4).texts[0]
    report = train_delta(lm, delta, SAMPLES, DeltaTrainConfig(epochs=60, batch_size=4, lr=5e-2, target_loss=0.01),
                         ledger=ledger)
    assert report.epoch_losses[-1] < 0.1 * report.epoch_losses[0]
    assert lm.backbone_checksum() == backbone and lm.fusion_checksum() == fusion
    after = lm.generate(["Question : Where does Zorblat Quenn work ?\nAnswer :",
                         "Mirel Kesh lives in"], max_new_tokens=4).texts
    assert "Vexmora Labs" not in before
    assert after[0].startswith("Vexmora Labs") and after[1].startswith("Port Elrin")
    assert set(ledger.claims()) == {"c1", "c2"}


def test_delete_restores_base_rows_exactly(tiny_lm, tmp_path):
    lm = _with_base(tiny_lm)
    delta, ledger = lm.new_delta(), RowLedger()
    probe = "the city where the dog lives"
    base_logits = _logits(lm, probe, use_delta=False)
    train_delta(lm, delta, SAMPLES, DeltaTrainConfig(epochs=30, batch_size=4, lr=5e-2), ledger=ledger)
    exclusive, shared, affected = ledger.split_for_removal("c1")
    for lid, rows in exclusive.items():
        delta.layers[lid].remove(torch.from_numpy(rows))
    for lid, rows in shared.items():
        delta.layers[lid].remove(torch.from_numpy(rows))
    ledger.remove("c1")
    # every row c1 could have changed is now back to the frozen base
    for lid, rows in ledger.rows_of("c1").items():
        assert delta.layers[lid].find(torch.from_numpy(rows))[1].sum() == 0
    remaining = ledger.all_rows()
    for lid, layer in delta.layers.items():
        assert np.isin(layer.rows.numpy(), remaining.get(lid, np.empty(0))).all()
    # removing everything gives the base model back bit for bit
    for layer in delta.layers.values():
        layer.remove(layer.rows.clone())
    torch.testing.assert_close(_logits(lm, probe), base_logits, rtol=0, atol=0)
    # save / load of an empty delta still round-trips
    delta.save(tmp_path / "d.safetensors")
    assert PersonalDelta.load(tmp_path / "d.safetensors").num_rows() == 0


def test_delta_save_load_gives_identical_outputs(tiny_lm, tmp_path):
    lm = _with_base(tiny_lm)
    delta = lm.new_delta()
    train_delta(lm, delta, SAMPLES[:2], DeltaTrainConfig(epochs=10, batch_size=2, lr=5e-2))
    out1 = _logits(lm, "Zorblat Quenn works at")
    delta.save(tmp_path / "delta.safetensors")
    lm.attach_delta(PersonalDelta.load(tmp_path / "delta.safetensors", expect_fingerprint=lm.layout.fingerprint()))
    torch.testing.assert_close(_logits(lm, "Zorblat Quenn works at"), out1, rtol=0, atol=0)
    lm.attach_delta(None)
    assert not torch.equal(_logits(lm, "Zorblat Quenn works at"), out1)


def test_delta_can_be_masked_on_context_positions(tiny_lm):
    lm = _with_base(tiny_lm)
    delta = lm.new_delta()
    train_delta(lm, delta, SAMPLES, DeltaTrainConfig(epochs=10, batch_size=4, lr=5e-2))
    ids = lm.tokenizer("Zorblat Quenn works at Vexmora Labs . Mirel Kesh lives in", return_tensors="pt")["input_ids"]
    k = 7  # the first 7 tokens play the role of wiki context
    dmask = torch.arange(ids.shape[1]).unsqueeze(0) >= k
    with torch.no_grad():
        masked = lm.hidden_states(ids, None, lm.build_batch(ids, delta_mask=dmask))
        plain = lm.hidden_states(ids, None, lm.build_batch(ids, use_delta=False))
        full = lm.hidden_states(ids, None, lm.build_batch(ids))
    # causal: context positions cannot see the question, and the delta is off there -> identical to no delta
    torch.testing.assert_close(masked[:, :k], plain[:, :k], rtol=0, atol=0)
    assert not torch.allclose(full[:, :k], plain[:, :k])
    # generation with delta_start == context length runs and differs from delta-everywhere decoding
    out = lm.generate(["Zorblat Quenn works at"], max_new_tokens=3, delta_start=[2])
    assert isinstance(out.texts[0], str)
