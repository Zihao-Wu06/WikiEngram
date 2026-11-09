"""Multi-head n-gram hashing, bit-exact with DeepSeek-V4.1-Flash ``NgramHashState`` (MIT).

Each position is hashed as the n-grams (2 .. max_ngram_size) that end there. Look-back stops at the start
of the sequence and at "dead" tokens (padding, image spans), which are replaced by the compressed pad id.
The running XOR after step i is the hash of the (i+1)-gram; each lands in its own prime-sized bucket range.

Hashing is cheap integer work that depends only on token ids, so it always runs on the CPU, before the
forward pass. This is also what lets the rows a fact touches be computed offline, without the model.
"""

from __future__ import annotations

import numpy as np
import torch

from .layout import EngramLayout


class NgramHasher:
    DEAD = -1

    def __init__(self, layout: EngramLayout, token_map: np.ndarray):
        spec = layout.spec
        if spec.compressed_vocab_size is None:
            raise ValueError("layout.spec.compressed_vocab_size is not set")
        if int(token_map.max()) + 1 != spec.compressed_vocab_size:
            raise ValueError(
                f"token map has {int(token_map.max()) + 1} compressed ids, spec expects "
                f"{spec.compressed_vocab_size}"
            )
        self.layout = layout
        self.max_ngram_size = spec.max_ngram_size
        self.token_map = torch.as_tensor(np.asarray(token_map), dtype=torch.int64)
        self.pad_id = int(self.token_map[spec.pad_token_id])
        self.multipliers = torch.as_tensor(layout.multipliers, dtype=torch.int64)  # [n_layers, N]
        self.primes = torch.tensor(layout.primes, dtype=torch.int64)  # [n_layers, N-1, n_heads]
        self.offsets = torch.as_tensor(layout.offsets, dtype=torch.int64)  # [n_layers, n_cols]

    @property
    def n_layers(self) -> int:
        return len(self.layout.spec.layer_ids)

    @property
    def n_cols(self) -> int:
        return self.layout.spec.n_hash_cols

    def _windows(self, input_ids: torch.Tensor, token_mask: torch.Tensor | None) -> torch.Tensor:
        """``[B, L, N]`` compressed ids of each position and its look-back (pad where blocked)."""
        ids = input_ids.detach().to("cpu", torch.int64)
        if ids.dim() == 1:
            ids = ids.unsqueeze(0)
        compressed = self.token_map[ids]
        if token_mask is not None:
            compressed = torch.where(token_mask.detach().to("cpu", torch.bool), compressed, self.DEAD)
        batch, seqlen = compressed.shape
        positions = torch.arange(seqlen).expand(batch, seqlen)
        tokens, blocked = [], torch.zeros(batch, seqlen, dtype=torch.bool)
        for shift in range(self.max_ngram_size):
            source = compressed.gather(1, (positions - shift).clamp_min(0))
            blocked = blocked | (positions < shift) | (source == self.DEAD)
            tokens.append(torch.where(blocked, self.pad_id, source))
        return torch.stack(tokens, dim=-1)

    @torch.no_grad()
    def __call__(self, input_ids: torch.Tensor, token_mask: torch.Tensor | None = None) -> torch.Tensor:
        """``input_ids [B, L]`` -> absolute table rows ``[B, L, n_layers, n_cols]`` (int64, CPU).

        ``token_mask [B, L]`` is False for tokens that take no part in an n-gram (padding). They block the
        look-back of later positions, exactly like DeepSeek's dead (image) tokens.
        """
        stacked = self._windows(input_ids, token_mask)  # [B, L, N]
        products = stacked.unsqueeze(2) * self.multipliers  # [B, L, n_layers, N]
        rolling, hashes = products[..., 0], []
        for i in range(1, self.max_ngram_size):
            rolling = torch.bitwise_xor(rolling, products[..., i])
            hashes.append(rolling.unsqueeze(-1) % self.primes[:, i - 1])
        return torch.cat(hashes, dim=-1) + self.offsets

    @torch.no_grad()
    def ngram_keys(self, input_ids: torch.Tensor, token_mask: torch.Tensor | None = None) -> torch.Tensor:
        """``[B, L, N-1]`` layer-independent 64-bit fingerprints of the 2..N-grams ending at each position
        (the rolling hash before the bucket modulus). Equal n-grams get equal keys; used for statistics."""
        stacked = self._windows(input_ids, token_mask)
        products = stacked * self.multipliers[0]
        rolling, keys = products[..., 0], []
        for i in range(1, self.max_ngram_size):
            rolling = torch.bitwise_xor(rolling, products[..., i])
            keys.append(rolling)
        return torch.stack(keys, dim=-1)

    def rows_for_text_ids(self, ids: list[int] | torch.Tensor) -> list[np.ndarray]:
        """Unique table rows per Engram layer touched by one token sequence (model-free)."""
        h = self(torch.as_tensor(ids, dtype=torch.int64).unsqueeze(0))[0]  # [L, n_layers, n_cols]
        return [np.unique(h[:, li, :].numpy()) for li in range(self.n_layers)]
