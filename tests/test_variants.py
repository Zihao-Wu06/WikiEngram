"""The personal-memory machinery is independent of the Engram family: a toy variant with its own hashing,
row grouping and fusion goes through consolidation, recall and exact deletion unchanged."""

from __future__ import annotations

import hashlib

import numpy as np
import pytest
import torch
from torch import nn

from tests.conftest import make_tiny_model
from wikiengram.backends.hf import EngramCausalLM
from wikiengram.consolidation.ledger import RowLedger
from wikiengram.consolidation.trainer import DeltaTrainConfig, TrainingSample, train_delta
from wikiengram.engram.deepseek import DeepSeekEngram
from wikiengram.engram.tables import DenseTable
from wikiengram.engram.variant import (
    REGISTRY,
    EngramVariant,
    detect_variant,
    register,
    variant_class,
    variant_from_dict,
)


class ToyFusion(nn.Module):
    """A different fusion rule: an ungated additive projection of the concatenated rows."""

    def __init__(self, n_in: int, hidden: int):
        super().__init__()
        self.proj = nn.Linear(n_in, hidden, bias=False)

    def forward(self, x, emb, token_mask=None):
        out = self.proj(emb.flatten(-2).to(self.proj.weight.dtype))
        if token_mask is not None:
            out = out * token_mask.to(out.device, out.dtype).unsqueeze(-1)
        return x + out.to(x.dtype)


class ToyEngram(EngramVariant):
    """Unigram + bigram groups, 3 rows each, plain modular hashing (nothing like DeepSeek's scheme)."""

    name = "toy-test"

    def __init__(self, hidden: int, layer_ids=(1,), rows: int = 4099, dim: int = 8):
        self._hidden, self._layers, self._rows, self._dim = hidden, tuple(layer_ids), rows, dim

    layer_ids = property(lambda self: self._layers)
    hidden_size = property(lambda self: self._hidden)
    row_dim = property(lambda self: self._dim)
    num_rows = property(lambda self: (self._rows,) * len(self._layers))
    n_groups = property(lambda self: 2)
    group_size = property(lambda self: 3)
    max_lookback = property(lambda self: 1)

    def ngram_keys(self, input_ids, token_mask=None):
        ids = input_ids.to("cpu", torch.int64)
        prev = torch.nn.functional.pad(ids, (1, 0), value=-1)[:, :-1]
        return torch.stack([ids, ids * 1_000_003 + prev], dim=-1)

    def hash(self, input_ids, token_mask=None):
        keys = self.ngram_keys(input_ids)  # [B, L, 2]
        cols = [(keys[..., g] * (7 + h) + 13 * h) % self._rows for g in range(2) for h in range(3)]
        rows = torch.stack(cols, dim=-1)  # [B, L, 6]
        return rows.unsqueeze(2).expand(-1, -1, len(self._layers), -1).contiguous()

    def make_module(self, layer_id):
        torch.manual_seed(100 + layer_id)
        return ToyFusion(self.rows_per_position * self._dim, self._hidden)

    def fingerprint(self):
        return hashlib.sha256(f"toy|{self._layers}|{self._rows}|{self._dim}".encode()).hexdigest()[:16]

    def to_dict(self):
        return {"variant": self.name, "hidden": self._hidden, "layer_ids": list(self._layers),
                "rows": self._rows, "dim": self._dim}

    @classmethod
    def from_dict(cls, d, tokenizer, cache_dir=None):
        return cls(d["hidden"], tuple(d["layer_ids"]), d["rows"], d["dim"])


SAMPLES = [
    TrainingSample("c1", "Question : Where does Zorblat Quenn work ?\nAnswer :", " Vexmora Labs\n"),
    TrainingSample("c1", "Zorblat Quenn works at", " Vexmora Labs .\n"),
    TrainingSample("c2", "Question : Where does Mirel Kesh live ?\nAnswer :", " Port Elrin\n"),
]


def test_core_pipeline_works_with_a_different_engram_family(tiny_tokenizer):
    variant = ToyEngram(hidden=64, layer_ids=(1, 2))
    lm = EngramCausalLM(make_tiny_model(len(tiny_tokenizer)), tiny_tokenizer, variant, device="cpu")
    lm.memory.base_tables = {lid: DenseTable.random(n, variant.row_dim, std=0.02, seed=i)
                             for i, (lid, n) in enumerate(zip(variant.layer_ids, variant.num_rows))}
    ids = tiny_tokenizer("the dog lives in Port Elrin", return_tensors="pt")["input_ids"]
    with torch.no_grad():
        base_logits = lm.logits(lm.hidden_states(ids, None, lm.build_batch(ids, use_delta=False)))
    delta, ledger = lm.new_delta(), RowLedger()
    assert delta.layout_fingerprint == variant.fingerprint() and delta.dim == variant.row_dim
    backbone, fusion = lm.backbone_checksum(), lm.fusion_checksum()
    report = train_delta(lm, delta, SAMPLES, DeltaTrainConfig(epochs=60, batch_size=3, lr=5e-2), ledger=ledger)
    assert report.epoch_losses[-1] < 0.1 * report.epoch_losses[0]
    assert lm.backbone_checksum() == backbone and lm.fusion_checksum() == fusion
    out = lm.generate(["Question : Where does Zorblat Quenn work ?\nAnswer :"], max_new_tokens=3).texts[0]
    assert out.startswith("Vexmora Labs")
    # exact deletion through the ledger, whatever the addressing scheme
    for cid in ("c1", "c2"):
        exclusive, shared, _ = ledger.split_for_removal(cid)
        for lid in variant.layer_ids:
            delta.layers[lid].remove(torch.from_numpy(np.union1d(exclusive.get(lid, []), shared.get(lid, []))))
        ledger.remove(cid)
    assert delta.num_rows() == 0
    with torch.no_grad():
        after = lm.logits(lm.hidden_states(ids, None, lm.build_batch(ids)))
    torch.testing.assert_close(after, base_logits, rtol=0, atol=0)


def test_registry_and_checkpoint_roundtrip(tiny_tokenizer, tiny_spec):
    assert variant_class("deepseek-v4.1") is DeepSeekEngram
    with pytest.raises(KeyError):
        variant_class("no-such-family")
    ds = DeepSeekEngram.build(tiny_spec, tiny_tokenizer)
    back = variant_from_dict(ds.to_dict(), tiny_tokenizer)
    assert back.fingerprint() == ds.fingerprint()
    legacy = {k: v for k, v in ds.to_dict().items() if k != "variant"}  # checkpoints from before the registry
    assert variant_from_dict(legacy, tiny_tokenizer).fingerprint() == ds.fingerprint()
    register(ToyEngram)
    try:
        toy = variant_from_dict(ToyEngram(64).to_dict(), tiny_tokenizer)
        assert isinstance(toy, ToyEngram)
    finally:
        REGISTRY.pop(ToyEngram.name, None)


def test_native_config_detection():
    v41 = {"text_config": {"hidden_size": 5120, "engram_layer_ids": [1, 14], "engram_vocab_size": 16000000}}
    assert detect_variant(v41) is DeepSeekEngram
    assert detect_variant({"hidden_size": 1024, "num_hidden_layers": 28}) is None
