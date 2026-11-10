"""Native backend on a tiny random Qwen3.8-Flash-Next architecture (Qwen4Exp: hybrid linear/indexed attention,
MoE, hyper-connections, native PLE n-gram memory). Runs on CPU; skipped if transformers lacks qwen4_exp."""

from __future__ import annotations

import numpy as np
import pytest
import torch

qwen4 = pytest.importorskip("transformers.models.qwen4_exp.modeling_qwen4_exp")

from tests.test_runtime import _config  # noqa: E402
from wikiengram.backends.native import NativeEngramLM  # noqa: E402
from wikiengram.consolidation.ledger import RowLedger  # noqa: E402
from wikiengram.consolidation.trainer import DeltaTrainConfig, TrainingSample, train_delta  # noqa: E402
from wikiengram.engram.qwen_ple import QwenPLEEngram  # noqa: E402
from wikiengram.runtime import WikiEngram  # noqa: E402


def make_tiny_qwen4exp(vocab_size: int, eos: int, seed: int = 0):
    from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig

    torch.manual_seed(seed)
    cfg = Qwen4ExpTextConfig(
        vocab_size=vocab_size, hidden_size=64, num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
        head_dim=16, linear_key_head_dim=16, linear_value_head_dim=16, linear_num_key_heads=2,
        linear_num_value_heads=4, moe_intermediate_size=32, shared_expert_intermediate_size=32, num_experts=4,
        num_experts_per_tok=2, layer_types=["linear_attention", "linear_attention", "indexed_attention",
                                            "linear_attention"],
        hc_count=2, hc_lowrank=8, ple_layer_ids=[2], ple_embed_dim=32, ngram_size=3, heads_per_ngram=4,
        ngram_vocab_size_base=1009, eos_token_id=eos, pad_token_id=0, indexer_n_heads=2, indexer_kv_heads=1,
        indexer_head_dim=16, indexer_budget=8, indexer_compress_ratio=2, max_position_embeddings=256)
    model = qwen4.Qwen4ExpForCausalLM(cfg).eval()
    with torch.no_grad():  # a confident output head and a non-trivial native table, like a trained model
        model.lm_head.weight.mul_(30)
        ple = model.model.layers[1].ple
        ple.ple_embedding.ngram_embedding.weight.normal_(0, 0.02)
        ple.value_proj.weight.normal_(0, 0.2)
    return model


@pytest.fixture()
def native(tiny_tokenizer):
    model = make_tiny_qwen4exp(len(tiny_tokenizer) + 8, tiny_tokenizer.eos_token_id)
    return model, NativeEngramLM(model, tiny_tokenizer, device="cpu", name="tiny-qwen4exp")


SAMPLES = [
    TrainingSample("c1", "Question : Where does Zorblat Quenn work ?\nAnswer :", " Vexmora Labs\n"),
    TrainingSample("c1", "Zorblat Quenn works at", " Vexmora Labs .\n"),
    TrainingSample("c2", "Mirel Kesh lives in", " Port Elrin .\n"),
]


def _logits(lm, ids, **kw):
    with torch.no_grad():
        return lm.logits(lm.hidden_states(ids, None, lm.build_batch(ids, **kw)))


def test_variant_detected_and_empty_memory_is_the_shipped_model(native, tiny_tokenizer):
    model, lm = native
    assert isinstance(lm.variant, QwenPLEEngram) and lm.variant.layer_ids == (1,)
    ids = tiny_tokenizer("Zorblat Quenn works at Vexmora Labs . Mirel Kesh lives in", return_tensors="pt")["input_ids"]
    with torch.no_grad():
        shipped = model(input_ids=ids).logits  # no Engram batch: the lookup defers to the native module
    lm.attach_delta(lm.new_delta())
    torch.testing.assert_close(_logits(lm, ids), shipped, rtol=0, atol=0)  # our addressing == the model's


def test_native_consolidation_recall_and_exact_delete(native, tiny_tokenizer):
    model, lm = native
    probe = tiny_tokenizer("the dog lives in Port Elrin", return_tensors="pt")["input_ids"]
    with torch.no_grad():
        shipped = model(input_ids=probe).logits
    backbone, fusion, base = lm.backbone_checksum(), lm.fusion_checksum(), lm.base_checksum()
    delta, ledger = lm.new_delta(), RowLedger()
    rep = train_delta(lm, delta, SAMPLES, DeltaTrainConfig(epochs=80, batch_size=3, lr=5e-2), ledger=ledger)
    assert rep.epoch_losses[-1] < 0.1 * rep.epoch_losses[0]
    assert (lm.backbone_checksum(), lm.fusion_checksum(), lm.base_checksum()) == (backbone, fusion, base)
    # generation uses the model's own cache, including the PLE conv state
    assert lm.generate(["Zorblat Quenn works at"], max_new_tokens=3).texts[0].startswith("Vexmora Labs")
    for cid in ("c1", "c2"):
        exclusive, shared, _ = ledger.split_for_removal(cid)
        for lid in lm.variant.layer_ids:
            delta.layers[lid].remove(torch.from_numpy(np.union1d(exclusive.get(lid, []), shared.get(lid, []))))
        ledger.remove(cid)
    torch.testing.assert_close(_logits(lm, probe), shipped, rtol=0, atol=0)


def test_workspace_pipeline_on_a_native_model(tmp_path, native):
    _, lm = native
    cfg = _config()
    cfg.backend.kind = "hf-native"
    # with the template / shared n-grams blocked, recall has to be routed by attention from the subject's rows to
    # the answer position; a tiny *random* linear-attention stack cannot do that (real models can), so this
    # integration test lets the delta write those rows
    cfg.consolidation.block_prompt_template = False
    cfg.consolidation.block_shared_claims = 0
    ws = WikiEngram.init(tmp_path / "ws", cfg)
    ws._lm = lm
    ws.wiki.ingest_text("Zorblat Quenn works at Vexmora Labs.\nMirel Kesh lives in Port Elrin.\n")
    rep = ws.consolidate()
    assert rep["selected"] == 2
    zq = next(c for c in ws.wiki.claims() if c.subject == "Zorblat Quenn")
    q = ws.state.claims[zq.id]["questions"][0]
    assert "Vexmora Labs" in ws.ask(q, mode="engram").text
    ws.wiki.retract(zq.id)
    ws.sync()
    assert zq.id not in ws.ledger
    assert "Vexmora Labs" not in ws.ask(q, mode="engram").text


def test_fast_qsa_indexer_is_exact():
    """The indexer shortcut (every visible token is kept when a query sees at most the budget) returns exactly the
    reference mask, for bool and float masks; longer sequences still run the reference code."""
    from wikiengram.backends.native import fast_qsa_indexer

    model = make_tiny_qwen4exp(64, 1)
    fast_qsa_indexer(model)
    layer = next(m for m in model.modules() if type(m).__name__.endswith("QSAIndexer"))
    original = type(layer)._wikiengram_original_forward
    rotary = model.model.rotary_emb if hasattr(model.model, "rotary_emb") else model.model.language_model.rotary_emb
    torch.manual_seed(0)
    for n in (5, 9, 13, 24):  # budget 8, ratio 2: up to 9 visible tokens take the shortcut
        h = torch.randn(3, n, model.config.get_text_config().hidden_size)
        pos = torch.arange(n).expand(3, n)
        cos_sin = rotary(h, pos)
        causal = torch.ones(n, n, dtype=torch.bool).tril()
        bool_mask = causal.expand(3, 1, n, n).clone()
        bool_mask[0, :, :, :2] = False  # left padding in the first sequence
        float_mask = torch.where(bool_mask, 0.0, torch.finfo(torch.float32).min)
        with torch.no_grad():
            for mask in (bool_mask, float_mask):
                torch.testing.assert_close(layer(h, cos_sin, mask, None), original(layer, h, cos_sin, mask, None),
                                           rtol=0, atol=0)
