"""Qwen PLE variant (Qwen3.8-Flash-Next) against the vendored transformers reference (tests/reference)."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tests.conftest import make_tiny_model
from tests.reference import qwen4_exp_ple as ref
from wikiengram.backends.hf import EngramCausalLM
from wikiengram.consolidation.ledger import RowLedger
from wikiengram.consolidation.trainer import DeltaTrainConfig, TrainingSample, train_delta
from wikiengram.engram.qwen_ple import QwenPLEEngram, QwenPLEModule, QwenPLESpec, layer_multipliers
from wikiengram.engram.tables import DenseTable
from wikiengram.engram.variant import detect_variant

EOS = 5


def _cfg(**kw):
    d = dict(hidden_size=32, hc_count=2, ple_embed_dim=64, ngram_size=3, heads_per_ngram=4, vocab_size=300,
             ngram_vocab_size_base=101, seed=1234, eos_token_id=EOS, make_ngram_vocab_size_divisible_by=128,
             ple_conv_kernel_size=4, rms_norm_eps=1e-6)
    d.update(kw)
    return SimpleNamespace(**d)


def _spec(cfg, layer_ids=(1,)) -> QwenPLESpec:
    heads = (cfg.ngram_size - 1) * cfg.heads_per_ngram
    return QwenPLESpec(hidden_size=cfg.hidden_size, layer_ids=layer_ids, ngram_size=cfg.ngram_size,
                       heads_per_ngram=cfg.heads_per_ngram, row_dim=cfg.ple_embed_dim // heads,
                       unigram_vocab_size=cfg.vocab_size, ngram_vocab_size_base=cfg.ngram_vocab_size_base,
                       seed=cfg.seed, eos_token_id=cfg.eos_token_id, hc_mult=cfg.hc_count,
                       conv_kernel_size=cfg.ple_conv_kernel_size,
                       vocab_divisor=cfg.make_ngram_vocab_size_divisible_by, norm_eps=cfg.rms_norm_eps)


def _ids(seed=0, batch=3, length=19, vocab=300, with_eos=True):
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(0, vocab, (batch, length), generator=g)
    if with_eos:  # EOS in the middle, at the start, and twice in a row
        for row, cols in ((0, [6]), (1, [0]), (2, [10, 11])):
            if row < batch:
                ids[row, cols] = EOS
    return ids


def test_real_qwen38_constants():
    # the multipliers stored as buffers in the Qwen3.8-Flash-Next checkpoint (layer index 0, seed 1234)
    m = layer_multipliers(248320, 3, 0, 1234)
    assert m.tolist() == [23703573157769, 20109073645365, 8052911324071]
    assert m.tolist() == ref._build_layer_multipliers(248320, 3, 0, 1234).tolist()
    v = QwenPLEEngram.from_model_config({"text_config": {
        "hidden_size": 2560, "hc_count": 4, "ple_embed_dim": 2560, "ple_layer_ids": [2], "ngram_size": 3,
        "heads_per_ngram": 8, "vocab_size": 248320, "ngram_vocab_size_base": 20000000, "seed": 1234,
        "eos_token_id": 248044, "make_ngram_vocab_size_divisible_by": 128, "ple_conv_kernel_size": 4,
        "rms_norm_eps": 1e-6}})
    assert v.layer_ids == (1,)  # config ple_layer_ids is 1-indexed
    assert v.row_dim == 160 and v.rows_per_position == 16
    assert v._sizes[0].tolist()[:2] == [20000003, 20000023]
    assert v.num_rows == (320001536,)  # 320,001,446 rows padded to a multiple of 128, as in the checkpoint
    assert detect_variant({"text_config": {"ple_layer_ids": [2], "ngram_vocab_size_base": 20000000}}) is QwenPLEEngram


@pytest.mark.parametrize("layer_index", [0, 1])
def test_hashing_matches_reference(layer_index):
    cfg = _cfg()
    variant = QwenPLEEngram(_spec(cfg, layer_ids=(1, 3)))
    emb = ref.Qwen4ExpTextNGramEmbedding(cfg, cfg.ple_embed_dim, layer_idx=1, ple_layer_index=layer_index)
    torch.manual_seed(0)
    torch.nn.init.normal_(emb.ngram_embedding.weight)
    assert variant.num_rows[layer_index] == emb.ngram_embedding.num_embeddings
    ids = _ids()
    with torch.no_grad():
        expected = emb(ids, None)
        rows = variant.hash(ids)[:, :, layer_index, :]
        got = emb.ngram_embedding.weight[rows].flatten(-2)
    torch.testing.assert_close(got, expected, rtol=0, atol=0)
    # padding: the reference model replaces padded ids by EOS before hashing; token_mask does the same
    mask = torch.ones_like(ids, dtype=torch.bool)
    mask[1, :4] = False
    with torch.no_grad():
        expected = emb(torch.where(mask, ids, EOS), None)
        got = emb.ngram_embedding.weight[variant.hash(ids, mask)[:, :, layer_index, :]].flatten(-2)
    torch.testing.assert_close(got, expected, rtol=0, atol=0)


def _module_pair(cfg, dtype=torch.float32):
    layer = ref.Qwen4ExpTextPLELayer(cfg, layer_idx=1, ple_layer_index=0)
    torch.manual_seed(1)
    with torch.no_grad():
        for p in layer.parameters():
            p.normal_(0, 0.3)
    ours = QwenPLEModule(_spec(cfg), 1)
    ours.load_state_dict({k: v for k, v in layer.state_dict().items() if not k.startswith("ple_embedding")})
    return layer.to(dtype), ours.to(dtype)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_fusion_matches_reference(dtype):
    cfg = _cfg()
    layer, ours = _module_pair(cfg, dtype)
    variant = QwenPLEEngram(_spec(cfg))
    ids = _ids(1)
    hidden = torch.randn(ids.shape[0], ids.shape[1], cfg.hc_count * cfg.hidden_size).to(dtype)
    mask = torch.ones_like(ids, dtype=torch.bool)
    mask[2, :3] = False
    table = layer.ple_embedding.ngram_embedding.weight
    with torch.no_grad():
        expected = hidden + layer(hidden, torch.where(mask, ids, EOS), None, conv_mask=mask)
        emb = table[variant.hash(ids, mask)[:, :, 0, :]]
        got = ours(hidden.unflatten(-1, (cfg.hc_count, cfg.hidden_size)), emb, mask).flatten(-2)
    torch.testing.assert_close(got, expected, rtol=0, atol=0)


def test_streaming_decode_equals_full_sequence():
    cfg = _cfg()
    _, ours = _module_pair(cfg)
    variant = QwenPLEEngram(_spec(cfg))
    table = torch.randn(variant.num_rows[0], variant.row_dim)
    ids = _ids(2, batch=2, length=16)
    hidden = torch.randn(2, 16, cfg.hc_count, cfg.hidden_size)
    emb = table[variant.hash(ids)[:, :, 0, :]]
    with torch.no_grad():
        full = ours(hidden, emb)
        ours.begin_stream()
        pieces = [ours(hidden[:, :5], emb[:, :5])] + [ours(hidden[:, t : t + 1], emb[:, t : t + 1]) for t in range(5, 16)]
        ours.end_stream()
    # same math; the conv kernel accumulates in a different order on the short decode window (as in the
    # reference's own cached path), so equality holds up to float rounding
    torch.testing.assert_close(torch.cat(pieces, dim=1), full, rtol=1e-5, atol=1e-5)


def test_qwen_ple_variant_through_the_whole_pipeline(tiny_tokenizer):
    spec = QwenPLESpec(hidden_size=64, layer_ids=(1, 2), ngram_size=3, heads_per_ngram=4, row_dim=8,
                       unigram_vocab_size=len(tiny_tokenizer), ngram_vocab_size_base=1009,
                       eos_token_id=tiny_tokenizer.eos_token_id)
    variant = QwenPLEEngram(spec)
    lm = EngramCausalLM(make_tiny_model(len(tiny_tokenizer)), tiny_tokenizer, variant, device="cpu")
    lm.memory.base_tables = {lid: DenseTable.random(n, variant.row_dim, std=0.02, seed=i)
                             for i, (lid, n) in enumerate(zip(variant.layer_ids, variant.num_rows))}
    for m in lm.engrams.values():
        m.reset_parameters(zero_value=False)
    probe = lm.tokenizer("the dog lives in Port Elrin", return_tensors="pt")["input_ids"]
    with torch.no_grad():
        base = lm.logits(lm.hidden_states(probe, None, lm.build_batch(probe, use_delta=False)))
    samples = [TrainingSample("c1", "Question : Where does Zorblat Quenn work ?\nAnswer :", " Vexmora Labs\n"),
               TrainingSample("c1", "Zorblat Quenn works at", " Vexmora Labs .\n"),
               TrainingSample("c2", "Mirel Kesh lives in", " Port Elrin .\n")]
    delta, ledger = lm.new_delta(), RowLedger()
    rep = train_delta(lm, delta, samples, DeltaTrainConfig(epochs=80, batch_size=3, lr=5e-2), ledger=ledger)
    assert rep.epoch_losses[-1] < 0.1 * rep.epoch_losses[0]
    # generation exercises the conv's streaming state
    assert lm.generate(["Zorblat Quenn works at"], max_new_tokens=3).texts[0].startswith("Vexmora Labs")
    for cid in ("c1", "c2"):
        exclusive, shared, _ = ledger.split_for_removal(cid)
        for lid in variant.layer_ids:
            delta.layers[lid].remove(torch.from_numpy(np.union1d(exclusive.get(lid, []), shared.get(lid, []))))
        ledger.remove(cid)
    with torch.no_grad():
        after = lm.logits(lm.hidden_states(probe, None, lm.build_batch(probe)))
    torch.testing.assert_close(after, base, rtol=0, atol=0)


def test_fresh_qwen_module_grows_in_gently():
    """A retrofit attachment must start as an identity and change the residual only a little after a small
    update (the conv branch RMS-normalizes its input, so it has to start at zero too)."""
    cfg = _cfg()
    module = QwenPLEModule(_spec(cfg), 1)  # reset_parameters(zero_value=True)
    x = torch.randn(2, 7, cfg.hc_count, cfg.hidden_size)
    emb = torch.randn(2, 7, 8, 8)
    with torch.no_grad():
        torch.testing.assert_close(module(x, emb), x, rtol=0, atol=0)
        module.value_proj.weight.normal_(0, 1e-4)  # one tiny training step's worth of change
        assert (module(x, emb) - x).abs().max() < 1e-2
