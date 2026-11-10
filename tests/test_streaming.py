"""The streaming (layer-by-layer recompute) backward gives the same loss and the same personal-memory gradients as
ordinary back-propagation, also when accelerate keeps part of the model offloaded on disk."""

from __future__ import annotations

import copy

import pytest
import torch

from tests.test_e2e_tiny import SAMPLES as RETROFIT_SAMPLES
from tests.test_e2e_tiny import _with_base
from wikiengram.backends.native import NativeEngramLM
from wikiengram.consolidation.trainer import DeltaTrainConfig, _collate, encode_samples, train_delta

qwen4 = pytest.importorskip("transformers.models.qwen4_exp.modeling_qwen4_exp")
from tests.test_native_qwen import SAMPLES as NATIVE_SAMPLES  # noqa: E402
from tests.test_native_qwen import make_tiny_qwen4exp  # noqa: E402


def _batch_inputs(lm, samples):
    return _collate(encode_samples(lm.tokenizer, samples), lm.tokenizer.pad_token_id)


def _standard(lm, ids, mask, labels):
    batch = lm.build_batch(ids, mask, train_delta=True)
    loss = lm.answer_loss(ids, mask, labels, batch)
    loss.backward()
    return loss.detach(), {lid: r.delta_leaf.grad.clone() for lid, r in batch.layers.items()}


def _streamed(lm, ids, mask, labels):
    batch = lm.build_batch(ids, mask, train_delta=True)
    loss = lm.streamer.loss_and_backward(ids, mask, labels, batch)
    return loss, {lid: r.delta_leaf.grad.clone() for lid, r in batch.layers.items()}


def _assert_same(a, b):
    (loss_a, grads_a), (loss_b, grads_b) = a, b
    torch.testing.assert_close(loss_b, loss_a, rtol=1e-6, atol=1e-6)
    assert grads_a.keys() == grads_b.keys()
    for lid in grads_a:
        assert grads_a[lid].abs().sum() > 0
        torch.testing.assert_close(grads_b[lid], grads_a[lid], rtol=1e-5, atol=1e-7)


def test_streaming_matches_backprop_on_a_retrofit_decoder(tiny_lm):
    lm = _with_base(tiny_lm)
    lm.attach_delta(lm.new_delta())
    ids, mask, labels = _batch_inputs(lm, RETROFIT_SAMPLES)
    expected = _standard(lm, ids, mask, labels)
    lm.enable_streaming()
    _assert_same(expected, _streamed(lm, ids, mask, labels))


def test_streaming_matches_backprop_on_a_native_qwen38_architecture(tiny_tokenizer):
    lm = NativeEngramLM(make_tiny_qwen4exp(len(tiny_tokenizer) + 8, tiny_tokenizer.eos_token_id), tiny_tokenizer,
                        device="cpu")
    lm.attach_delta(lm.new_delta())
    ids, mask, labels = _batch_inputs(lm, NATIVE_SAMPLES)
    expected = _standard(lm, ids, mask, labels)
    lm.enable_streaming()
    _assert_same(expected, _streamed(lm, ids, mask, labels))


def _offloaded_pair(tiny_tokenizer, tmp_path):
    """The same tiny native model twice: resident, and dispatched with layers 2-3 offloaded to disk."""
    from accelerate import dispatch_model

    model = make_tiny_qwen4exp(len(tiny_tokenizer) + 8, tiny_tokenizer.eos_token_id)
    resident = copy.deepcopy(model)
    device_map = {"model.embed_tokens": "cpu", "model.rotary_emb": "cpu", "model.hyper_connection_mixer": "cpu",
                  "lm_head": "cpu", "model.layers.0": "cpu", "model.layers.1": "cpu",
                  "model.layers.2": "disk", "model.layers.3": "disk"}
    offloaded = dispatch_model(model, device_map=device_map, main_device="cpu", offload_dir=str(tmp_path / "offload"))
    return resident, offloaded


def test_streaming_through_disk_offloaded_layers_matches_resident_backprop(tiny_tokenizer, tmp_path):
    resident, offloaded = _offloaded_pair(tiny_tokenizer, tmp_path)
    assert any(p.device.type == "meta" for p in offloaded.model.layers[3].parameters())  # really offloaded
    ref = NativeEngramLM(resident, tiny_tokenizer, device="cpu")
    lm = NativeEngramLM(offloaded, tiny_tokenizer, device="cpu")
    for x in (ref, lm):
        x.attach_delta(x.new_delta())
    ids, mask, labels = _batch_inputs(ref, NATIVE_SAMPLES)
    expected = _standard(ref, ids, mask, labels)
    lm.enable_streaming()
    _assert_same(expected, _streamed(lm, ids, mask, labels))


def test_consolidation_trains_through_an_offloaded_model(tiny_tokenizer, tmp_path):
    _, offloaded = _offloaded_pair(tiny_tokenizer, tmp_path)
    lm = NativeEngramLM(offloaded, tiny_tokenizer, device="cpu")
    lm.enable_streaming()
    delta = lm.new_delta()
    rep = train_delta(lm, delta, NATIVE_SAMPLES, DeltaTrainConfig(epochs=80, batch_size=3, lr=5e-2))
    assert rep.epoch_losses[-1] < 0.1 * rep.epoch_losses[0]
    # inference through the offloaded model (accelerate hooks, model's own cache) recalls the fact
    assert lm.generate(["Zorblat Quenn works at"], max_new_tokens=3).texts[0].startswith("Vexmora Labs")
