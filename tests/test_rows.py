from __future__ import annotations

import numpy as np
import torch

from wikiengram.consolidation.rows import NgramCounts, generic_rows
from wikiengram.engram.deepseek import DeepSeekEngram


def _hasher(tok, spec):
    return DeepSeekEngram.build(spec, tok)


def _ids(tok, text):
    return tok(text, add_special_tokens=False)["input_ids"]


def test_ngram_keys_identify_ngrams_independent_of_position(tiny_tokenizer, tiny_spec):
    h = _hasher(tiny_tokenizer, tiny_spec)
    a = h.ngram_keys(torch.tensor([_ids(tiny_tokenizer, "the dog lives in Port Elrin")]))[0]
    b = h.ngram_keys(torch.tensor([_ids(tiny_tokenizer, "Mirel Kesh lives in Port Elrin")]))[0]
    # "lives in Port Elrin" ends both sequences: the 2- and 3-gram keys of the last positions match
    torch.testing.assert_close(a[-1], b[-1])
    torch.testing.assert_close(a[-2], b[-2])
    assert not torch.equal(a[1], b[1])
    # case folding comes from the compressed token map
    c = h.ngram_keys(torch.tensor([_ids(tiny_tokenizer, "THE dog")]))[0]
    d = h.ngram_keys(torch.tensor([_ids(tiny_tokenizer, "The dog")]))[0]
    torch.testing.assert_close(c, d)


def test_generic_rows_block_shared_template_and_background(tiny_tokenizer, tiny_spec):
    h = _hasher(tiny_tokenizer, tiny_spec)
    texts = ["Zorblat Quenn works at Vexmora Labs", "Mirel Kesh works at Port Elrin", "Drumla Tavik works at Varo"]
    seqs = [_ids(tiny_tokenizer, t) for t in texts]
    rows, stats = generic_rows(h, seqs, ["a", "b", "c"], min_shared_claims=3)
    li, lid = 0, tiny_spec.layer_ids[0]
    heads = tiny_spec.n_heads
    r0 = h.hash(torch.tensor([seqs[0]]))[0]  # [L, n_layers, C]
    works_at = r0[3, li, 0:heads].numpy()  # 2-gram "works at" ends at position 3 ("at"), order 0
    entity = r0[1, li, 0:heads].numpy()  # 2-gram "Zorblat Quenn"
    assert np.isin(works_at, rows[lid]).all()
    assert not np.isin(entity, rows[lid]).any()
    assert stats["blocked_shared"] > 0
    # template sequences are always blocked
    rows_t, _ = generic_rows(h, seqs[:1], ["a"], always_block=[_ids(tiny_tokenizer, "Zorblat Quenn")])
    assert np.isin(entity, rows_t[lid]).all()
    # background frequency
    stream = torch.tensor(_ids(tiny_tokenizer, " ".join(["the dog works at the city"] * 10)))
    bg = NgramCounts.from_stream(h, stream, chunk=7)
    rows_b, stats_b = generic_rows(h, seqs[:1], ["a"], background=bg, min_background_count=5)
    assert np.isin(works_at, rows_b[lid]).all() and not np.isin(entity, rows_b[lid]).any()
    assert stats_b["blocked_background"] > 0


def test_background_counts_match_direct_count(tiny_tokenizer, tiny_spec):
    h = _hasher(tiny_tokenizer, tiny_spec)
    stream = torch.tensor(_ids(tiny_tokenizer, "a b a b a b the dog the dog"))
    direct = h.ngram_keys(stream.unsqueeze(0))[0].reshape(-1).numpy()
    keys, counts = np.unique(direct, return_counts=True)
    for chunk in (3, 5, 100):  # chunked counting must not lose or double n-grams at chunk boundaries
        bg = NgramCounts.from_stream(h, stream, chunk=chunk)
        np.testing.assert_array_equal(bg.count(keys), counts)
