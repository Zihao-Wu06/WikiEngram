"""Bit-exact parity of the Engram port with DeepSeek-V4.1-Flash's reference code (tests/reference)."""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tests.reference import deepseek_v41_engram as ref
from wikiengram.engram.compression import build_compressed_token_map
from wikiengram.engram.hashing import NgramHasher
from wikiengram.engram.layout import EngramLayout, EngramSpec, compute_hash_multipliers
from wikiengram.engram.module import EngramModule
from wikiengram.engram.primes import is_prime

V41_CONFIG = {
    "text_config": {
        "hidden_size": 5120, "engram_layer_ids": [1, 14], "engram_num_embeddings": [384006168, 384016682],
        "engram_max_ngram_size": 4, "engram_vocab_size": 16000000, "engram_n_heads": 8,
        "engram_head_dim": 256, "engram_pad_token_id": 2, "engram_compressed_vocab_size": 99092,
        "hc_mult": 4, "rms_norm_eps": 1e-20,
    }
}


def test_is_prime_matches_sympy():
    sympy = pytest.importorskip("sympy")
    for n in list(range(0, 5000)) + list(range(16_000_000, 16_002_000)):
        assert is_prime(n) == bool(sympy.isprime(n)), n


def test_v41_layout_matches_reference_and_config():
    spec = EngramSpec.from_deepseek_v41_config(V41_CONFIG)
    layout = EngramLayout.from_spec(spec)
    args = SimpleNamespace(
        engram_layer_ids=spec.layer_ids, engram_max_ngram_size=spec.max_ngram_size,
        engram_n_heads=spec.n_heads, engram_vocab_size=spec.vocab_size,
        engram_num_embeddings=tuple(V41_CONFIG["text_config"]["engram_num_embeddings"]),
        engram_head_dim=spec.head_dim,
    )
    pytest.importorskip("sympy")
    reference = ref.EngramLayout.from_args(args)
    assert layout.primes == reference.primes
    # the shipped table sizes are exactly the sums of the per-column primes
    assert layout.num_embeddings == (384006168, 384016682)


def test_multipliers_match_reference():
    for vocab in (99092, 107457, 517):
        ours = compute_hash_multipliers((1, 14), 4, vocab)
        theirs = ref.compute_hash_multipliers((1, 14), 4, vocab).numpy()
        np.testing.assert_array_equal(ours, theirs)


def test_compression_matches_reference(tiny_tokenizer):
    ours, n_ours = build_compressed_token_map(tiny_tokenizer)
    theirs, n_theirs = ref.build_compressed_token_map(tiny_tokenizer)
    assert n_ours == n_theirs
    np.testing.assert_array_equal(ours, np.asarray(theirs))
    vocab = tiny_tokenizer.get_vocab()
    # case and accents collapse
    assert ours[vocab["The"]] == ours[vocab["the"]] == ours[vocab["THE"]]
    assert ours[vocab["café"]] == ours[vocab["cafe"]] == ours[vocab["Café"]]
    assert ours[vocab["Zürich"]] == ours[vocab["ZURICH"]]


def _reference_hashes(tokenizer, spec: EngramSpec, ids: torch.Tensor, mask=None) -> torch.Tensor:
    lookup, size = build_compressed_token_map(tokenizer)
    args = SimpleNamespace(
        engram_layer_ids=spec.layer_ids, engram_max_ngram_size=spec.max_ngram_size,
        engram_n_heads=spec.n_heads, engram_vocab_size=spec.vocab_size,
        engram_num_embeddings=(0,) * len(spec.layer_ids), engram_head_dim=spec.head_dim,
        engram_compressed_vocab_size=size, engram_pad_id=spec.pad_token_id,
        max_batch_size=ids.shape[0], max_seq_len=ids.shape[1],
    )
    layout = ref.EngramLayout.from_args(args)
    state = ref.NgramHashState(args, layout, tokenizer)
    return state(ids, 0, mask)


@pytest.mark.parametrize("max_ngram,heads,layers", [(4, 8, (1, 14)), (3, 4, (0, 2, 5)), (2, 2, (3,))])
def test_hashing_matches_reference(tiny_tokenizer, max_ngram, heads, layers):
    pytest.importorskip("sympy")
    spec = EngramSpec(hidden_size=64, layer_ids=layers, max_ngram_size=max_ngram, n_heads=heads,
                      head_dim=16, vocab_size=2003, pad_token_id=0)
    lookup, size = build_compressed_token_map(tiny_tokenizer)
    layout = EngramLayout.from_spec(spec.with_compressed_vocab(size))
    hasher = NgramHasher(layout, lookup)
    g = torch.Generator().manual_seed(0)
    ids = torch.randint(0, len(tiny_tokenizer), (3, 17), generator=g)
    torch.testing.assert_close(hasher(ids), _reference_hashes(tiny_tokenizer, spec, ids), rtol=0, atol=0)
    # padding / dead tokens block the look-back exactly like DeepSeek's image spans
    mask = torch.ones_like(ids, dtype=torch.bool)
    mask[0, :4] = False
    mask[2, 9] = False
    torch.testing.assert_close(hasher(ids, mask), _reference_hashes(tiny_tokenizer, spec, ids, mask), rtol=0, atol=0)


def test_rows_stay_inside_their_column_ranges(tiny_tokenizer):
    spec = EngramSpec(hidden_size=64, layer_ids=(1, 2), max_ngram_size=3, n_heads=4, head_dim=16, vocab_size=101)
    lookup, size = build_compressed_token_map(tiny_tokenizer)
    layout = EngramLayout.from_spec(spec.with_compressed_vocab(size))
    h = NgramHasher(layout, lookup)(torch.randint(0, len(tiny_tokenizer), (4, 32)))
    for li in range(2):
        starts = layout.offsets[li]
        ends = starts + np.asarray(layout.flat_primes[li])
        col = h[:, :, li, :].reshape(-1, spec.n_hash_cols).numpy()
        assert (col >= starts).all() and (col < ends).all()
        assert h[:, :, li, :].max() < layout.num_embeddings[li]


@pytest.mark.parametrize("hc_mult", [1, 4])
def test_engram_module_matches_reference(hc_mult):
    torch.manual_seed(0)
    spec = EngramSpec(hidden_size=32, layer_ids=(1,), max_ngram_size=4, n_heads=2, head_dim=8, hc_mult=hc_mult,
                      norm_eps=1e-20)
    module = EngramModule(spec, 1)
    module.reset_parameters(zero_value=False)
    with torch.no_grad():
        module.q_weight.normal_()
        module.k_weight.normal_()
    x = torch.randn(2, 5, hc_mult, 32)
    emb = torch.randn(2, 5, spec.n_hash_cols, 8)
    mask = torch.ones(2, 5, dtype=torch.bool)
    mask[1, 0] = False
    expected = ref.engram_forward_reference(x, emb, module.wkv.weight, module.q_weight, module.k_weight,
                                            hc_mult, 32, spec.norm_eps, mask)
    torch.testing.assert_close(module(x, emb, mask), expected, rtol=0, atol=0)
    if hc_mult == 1:  # plain decoders pass [B, L, D]
        torch.testing.assert_close(module(x[:, :, 0], emb, mask), expected[:, :, 0], rtol=0, atol=0)


def test_zero_value_projection_is_identity():
    spec = EngramSpec(hidden_size=32, layer_ids=(1,), max_ngram_size=3, n_heads=2, head_dim=8)
    module = EngramModule(spec, 1)  # default init: value half of wkv is zero
    x = torch.randn(2, 5, 32)
    torch.testing.assert_close(module(x, torch.randn(2, 5, spec.n_hash_cols, 8)), x)


V41_TOKENIZER = os.environ.get("WIKIENGRAM_V41_TOKENIZER")


@pytest.mark.network
def test_v41_compressed_vocab_is_99092():
    """The shipped `engram_compressed_vocab_size` of DeepSeek-V4.1-Flash, reproduced from its tokenizer."""
    from tokenizers import Tokenizer

    if V41_TOKENIZER and Path(V41_TOKENIZER).exists():
        path = V41_TOKENIZER
    else:
        hub = pytest.importorskip("huggingface_hub")
        try:
            path = hub.hf_hub_download("deepseek-ai/DeepSeek-V4.1-Flash", "tokenizer.json")
        except Exception as e:  # pragma: no cover - offline
            pytest.skip(f"cannot download tokenizer: {e}")
    _, size = build_compressed_token_map(Tokenizer.from_file(path))
    assert size == 99092
