"""Which Engram rows a personal delta may write to.

Engram rows are addressed by token ids alone, so statistics can be computed without the model. Two kinds of
n-gram are *generic*, and their rows are blocked from consolidation:

* n-grams that are frequent in general text (``"works at"``, ``"of the"``); writing personal knowledge into
  them would leak into unrelated text;
* n-grams shared by the training samples of several different claims (``"Answer:"``, ``"lives in"``); they
  carry no claim-specific information, only interference.

What remains are the claim-specific n-grams, mostly those containing the entity's own tokens. The backbone's
attention routes their memory to the answer position. This is the n-gram analogue of Sparse Memory
Finetuning's TF-IDF slot selection. Note that counting is done per *n-gram*, not per row, because in a hashed
table every row is shared by many colliding rare n-grams.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import torch

from ..engram.hashing import NgramHasher


class NgramCounts:
    """Background n-gram frequencies (by fingerprint) of a token stream."""

    def __init__(self, keys: np.ndarray, counts: np.ndarray, n_tokens: int):
        self.keys, self.counts, self.n_tokens = keys, counts, n_tokens

    @classmethod
    def from_stream(cls, hasher: NgramHasher, stream: torch.Tensor, chunk: int = 8192) -> NgramCounts:
        parts = []
        overlap = hasher.max_ngram_size - 1
        for start in range(0, stream.numel(), chunk):
            lo = max(0, start - overlap)
            k = hasher.ngram_keys(stream[lo : start + chunk].unsqueeze(0))[0]
            parts.append(k[start - lo :].reshape(-1).numpy())
        keys, counts = np.unique(np.concatenate(parts), return_counts=True)
        return cls(keys, counts, int(stream.numel()))

    def count(self, query: np.ndarray) -> np.ndarray:
        if self.keys.size == 0:
            return np.zeros(query.shape, dtype=np.int64)
        pos = np.searchsorted(self.keys, query).clip(max=self.keys.size - 1)
        return np.where(self.keys[pos] == query, self.counts[pos], 0)

    def save(self, path) -> None:
        np.savez(path, keys=self.keys, counts=self.counts, n_tokens=np.int64(self.n_tokens))

    @classmethod
    def load(cls, path) -> NgramCounts:
        d = np.load(path)
        return cls(d["keys"], d["counts"], int(d["n_tokens"]))


def generic_rows(
    hasher: NgramHasher,
    sequences: list[list[int]],
    claim_ids: list[str],
    background: NgramCounts | None = None,
    min_background_count: int = 0,
    min_shared_claims: int = 0,
    always_block: list[list[int]] | None = None,
    context_sequences: list[list[int]] | None = None,
    context_claim_ids: list[str] | None = None,
) -> tuple[dict[int, np.ndarray], dict[str, int]]:
    """Rows to block, per Engram layer id, for training ``sequences`` (one claim id per sequence).

    A (position, n-gram order) is generic when its n-gram occurs ``>= min_background_count`` times in the
    background (0 = off), in samples of ``>= min_shared_claims`` distinct claims (0 = off), or inside one of the
    ``always_block`` token sequences (e.g. the prompt template). ``context_sequences`` (samples of claims
    already in memory) count towards "shared by N claims" but are not themselves blocked, so genericity is
    judged across the whole personal memory, not just the current batch. All heads of that order at that
    position are blocked. Returns the rows and a few statistics.
    """
    spec = hasher.layout.spec
    n_orders, heads = spec.max_ngram_size - 1, spec.n_heads
    keys_per_seq = [hasher.ngram_keys(torch.tensor(s).unsqueeze(0))[0].numpy() for s in sequences]
    rows_per_seq = [hasher(torch.tensor(s).unsqueeze(0))[0].numpy() for s in sequences]  # [L, n_layers, C]
    claims_of_key: dict[int, set[str]] = defaultdict(set)
    if min_shared_claims > 0:
        for keys, cid in zip(keys_per_seq, claim_ids):
            for k in np.unique(keys):
                claims_of_key[int(k)].add(cid)
        for seq, cid in zip(context_sequences or [], context_claim_ids or []):
            for k in np.unique(hasher.ngram_keys(torch.tensor(seq).unsqueeze(0))[0].numpy()):
                claims_of_key[int(k)].add(cid)
    template_keys = np.empty(0, np.int64)
    if always_block:
        template_keys = np.unique(np.concatenate(
            [hasher.ngram_keys(torch.tensor(t).unsqueeze(0))[0].reshape(-1).numpy() for t in always_block if t]))
    blocked: dict[int, list[np.ndarray]] = defaultdict(list)
    n_total = n_blocked_bg = n_blocked_shared = 0
    for keys, rows in zip(keys_per_seq, rows_per_seq):
        mask = np.isin(keys, template_keys)  # [L, n_orders]
        if background is not None and min_background_count > 0:
            bg = background.count(keys.reshape(-1)).reshape(keys.shape) >= min_background_count
            n_blocked_bg += int(bg.sum())
            mask |= bg
        if min_shared_claims > 0:
            shared = np.vectorize(lambda k: len(claims_of_key.get(int(k), ())))(keys) >= min_shared_claims
            n_blocked_shared += int(shared.sum())
            mask |= shared
        n_total += keys.size
        pos, order = np.nonzero(mask)
        for li, lid in enumerate(spec.layer_ids):
            for p, o in zip(pos.tolist(), order.tolist()):
                blocked[lid].append(rows[p, li, o * heads : (o + 1) * heads])
    out = {lid: np.unique(np.concatenate(v)) if v else np.empty(0, np.int64) for lid, v in blocked.items()}
    for lid in spec.layer_ids:
        out.setdefault(lid, np.empty(0, np.int64))
    stats = {"ngram_positions": n_total, "blocked_background": n_blocked_bg, "blocked_shared": n_blocked_shared,
             "blocked_rows": int(sum(v.size for v in out.values())), "orders": n_orders}
    return out, stats
