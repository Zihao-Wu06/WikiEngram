"""Running a native model straight from a release-format checkpoint (resident non-expert weights, experts read on
demand, the n-gram table read row by row from FP8 shards) gives exactly the same results as the same weights held
in memory, and personal memory trains through it."""

from __future__ import annotations

import copy
import json

import pytest
import torch
from safetensors.torch import save_file

qwen4 = pytest.importorskip("transformers.models.qwen4_exp.modeling_qwen4_exp")

from tests.test_native_qwen import SAMPLES, make_tiny_qwen4exp  # noqa: E402
from wikiengram.backends.checkpoint import StreamedExperts  # noqa: E402
from wikiengram.backends.native import NativeEngramLM  # noqa: E402
from wikiengram.consolidation.trainer import DeltaTrainConfig, _collate, encode_samples, train_delta  # noqa: E402
from wikiengram.engram.tables import ShardedTable  # noqa: E402

FP8 = torch.float8_e4m3fn
BLOCK = 128


def _quant_block(w: torch.Tensor):
    """Block-wise FP8 like the Qwen3.8 FP8 release: e4m3 values and a BF16 ``weight_scale_inv`` per 128x128."""
    out_f, in_f = w.shape
    nb0, nb1 = -(-out_f // BLOCK), -(-in_f // BLOCK)
    pad = torch.zeros(nb0 * BLOCK, nb1 * BLOCK)
    pad[:out_f, :in_f] = w.float()
    blocks = pad.view(nb0, BLOCK, nb1, BLOCK)
    scale = (blocks.abs().amax(dim=(1, 3)) / 448.0).clamp_min(1e-12).to(torch.bfloat16)
    q = (blocks / scale.float()[:, None, :, None]).view(nb0 * BLOCK, nb1 * BLOCK)[:out_f, :in_f].to(FP8)
    return q, scale


def _dequant_block(q, scale):
    s = scale.float().repeat_interleave(BLOCK, 0)[: q.shape[0]].repeat_interleave(BLOCK, 1)[:, : q.shape[1]]
    return q.float() * s


def write_release_checkpoint(model, tokenizer, path, n_table_shards=4):
    """Save ``model`` as a Qwen3.8-Flash-Next-FP8-style release; return the weights the release really holds."""
    path.mkdir(parents=True, exist_ok=True)
    model.config.save_pretrained(path)
    tokenizer.save_pretrained(path)
    sd = {k: v.detach().clone() for k, v in model.state_dict().items()}
    effective = dict(sd)  # what a reader of the release sees after dequantization
    out = {}
    for name, t in sd.items():
        key = "model.language_model." + name[len("model."):] if name.startswith("model.") else name
        if name.endswith("mlp.experts.gate_up_proj"):
            base = key[: -len(".gate_up_proj")]
            down = sd[name.replace("gate_up_proj", "down_proj")]
            inter = t.shape[1] // 2
            gu_eff, down_eff = torch.empty_like(t), torch.empty_like(down)
            for e in range(t.shape[0]):
                for proj, w, rows in (("gate_proj", t[e, :inter], slice(0, inter)),
                                      ("up_proj", t[e, inter:], slice(inter, None))):
                    q, s = _quant_block(w)
                    out[f"{base}.{e}.{proj}.weight"], out[f"{base}.{e}.{proj}.weight_scale_inv"] = q, s
                    gu_eff[e, rows] = _dequant_block(q, s)
                q, s = _quant_block(down[e])
                out[f"{base}.{e}.down_proj.weight"], out[f"{base}.{e}.down_proj.weight_scale_inv"] = q, s
                down_eff[e] = _dequant_block(q, s)
            effective[name], effective[name.replace("gate_up_proj", "down_proj")] = gu_eff, down_eff
        elif name.endswith("mlp.experts.down_proj"):
            continue
        elif name.endswith("ngram_embedding.weight"):  # FP8 shards with one global BF16 scale
            base = key[: -len(".weight")]
            scale = (t.abs().amax() / 448.0).to(torch.bfloat16)
            q = (t / scale.float()).to(FP8)
            for i, part in enumerate(torch.tensor_split(q, n_table_shards)):
                out[f"{base}.shard_{i}.weight"] = part.contiguous()
            out[f"{base}.weight_scale"] = scale.reshape(1)
            effective[name] = q.float() * scale.float()
        elif t.dim() == 2 and name.endswith("_proj.weight") and ("self_attn" in name or "linear_attn" in name
                                                                 or "shared_expert" in name):
            q, s = _quant_block(t)
            out[key], out[key + "_scale_inv"] = q, s
            effective[name] = _dequant_block(q, s)
        else:
            out[key] = t.contiguous()
    keys = sorted(out)
    files = [keys[i::3] for i in range(3)]  # spread over several files like a real release
    weight_map = {}
    for i, part in enumerate(files):
        fname = f"model-{i + 1:05d}-of-00003.safetensors"
        save_file({k: out[k] for k in part}, str(path / fname))
        weight_map.update(dict.fromkeys(part, fname))
    (path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))
    return effective


@pytest.fixture()
def release(tiny_tokenizer, tmp_path):
    model = make_tiny_qwen4exp(len(tiny_tokenizer) + 8, tiny_tokenizer.eos_token_id)
    effective = write_release_checkpoint(model, tiny_tokenizer, tmp_path / "release")
    reference = copy.deepcopy(model)
    reference.load_state_dict(effective)  # the same dequantized weights, held in memory
    return tmp_path / "release", reference


def _ids(lm, text):
    return lm.tokenizer(text, return_tensors="pt")["input_ids"]


def test_sharded_fp8_table_reads_the_rows_of_the_release(release):
    path, reference = release
    lm = NativeEngramLM.from_checkpoint(path, device="cpu", dtype="float32")
    table = lm.memory.base_tables[1]
    assert isinstance(table, ShardedTable) and len(table.shards) == 4
    rows = torch.tensor([0, 7, table.rows_per_shard, table.num_rows - 1, 7, table.rows_per_shard * 2 + 3])
    want = reference.model.layers[1].ple.ple_embedding.ngram_embedding.weight[rows]
    torch.testing.assert_close(table.gather(rows), want, rtol=0, atol=0)
    assert table.checksum() == table.checksum()


def test_streamed_release_matches_the_model_in_memory(release, tiny_tokenizer):
    path, reference = release
    lm = NativeEngramLM.from_checkpoint(path, device="cpu", dtype="float32")
    assert any(isinstance(m, StreamedExperts) for m in lm.model.modules())
    assert all(p.device.type == "meta" for n, p in lm.model.named_parameters() if n.endswith("ngram_embedding.weight"))
    ref = NativeEngramLM(reference, tiny_tokenizer, device="cpu")
    for x in (lm, ref):
        x.attach_delta(x.new_delta())
    ids = _ids(lm, "Zorblat Quenn works at Vexmora Labs . Mirel Kesh lives in")
    with torch.no_grad():
        got, want = lm.logits(lm.hidden_states(ids)), ref.logits(ref.hidden_states(ids))
    torch.testing.assert_close(got, want, rtol=0, atol=0)
    prompts = ["Zorblat Quenn works at", "Mirel Kesh lives in"]
    assert lm.generate(prompts, max_new_tokens=4).texts == ref.generate(prompts, max_new_tokens=4).texts


def test_personal_memory_trains_through_the_streamed_release(release, tiny_tokenizer):
    path, reference = release
    lm = NativeEngramLM.from_checkpoint(path, device="cpu", dtype="float32", expert_cache_gb=0.01)
    lm.enable_streaming()
    ref = NativeEngramLM(reference, tiny_tokenizer, device="cpu")
    # one step: streamed loss and gradients == ordinary back-propagation on the in-memory model
    ids, mask, labels = _collate(encode_samples(lm.tokenizer, SAMPLES), lm.tokenizer.pad_token_id)
    for x in (lm, ref):
        x.attach_delta(x.new_delta())
    b_ref = ref.build_batch(ids, mask, train_delta=True)
    loss_ref = ref.answer_loss(ids, mask, labels, b_ref)
    loss_ref.backward()
    b = lm.build_batch(ids, mask, train_delta=True)
    loss = lm.streamer.loss_and_backward(ids, mask, labels, b)
    torch.testing.assert_close(loss, loss_ref.detach(), rtol=1e-6, atol=1e-6)
    for lid in b.layers:
        torch.testing.assert_close(b.layers[lid].delta_leaf.grad, b_ref.layers[lid].delta_leaf.grad,
                                   rtol=1e-5, atol=1e-7)
    # full consolidation: learns the facts, keeps every frozen weight bit-identical, recalls them in generation
    frozen = (lm.backbone_checksum(), lm.fusion_checksum(), lm.base_checksum())
    rep = train_delta(lm, lm.new_delta(), SAMPLES, DeltaTrainConfig(epochs=80, batch_size=3, lr=5e-2))
    assert rep.epoch_losses[-1] < 0.1 * rep.epoch_losses[0]
    assert (lm.backbone_checksum(), lm.fusion_checksum(), lm.base_checksum()) == frozen
    assert lm.generate(["Zorblat Quenn works at"], max_new_tokens=3).texts[0].startswith("Vexmora Labs")
    cache = lm.model.streamed_expert_cache
    assert cache.hits > 0 and cache.used <= cache.budget


def test_direct_reads_return_the_same_bytes(release):
    """O_DIRECT reads (aligned superset into a page-aligned buffer) give exactly the bytes of page-cache reads."""
    import os

    import numpy as np

    from wikiengram.backends.checkpoint import SafetensorsCheckpoint

    if not hasattr(os, "O_DIRECT"):
        pytest.skip("no O_DIRECT on this platform")
    path, _ = release
    direct, cached = SafetensorsCheckpoint(path), SafetensorsCheckpoint(path)
    cached._direct = False
    for key in direct.weight_map:
        _, _, start, end = direct.tensor_info(key)
        a, b = np.empty(end - start, np.uint8), np.empty(end - start, np.uint8)
        direct.read_into(key, a)
        cached.read_into(key, b)
        assert np.array_equal(a, b), key
    if direct._direct is False:
        pytest.skip("the temporary directory's file system has no O_DIRECT (e.g. tmpfs): fallback used")
