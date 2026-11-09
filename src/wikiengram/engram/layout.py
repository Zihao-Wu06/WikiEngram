"""Engram configuration and hash-bucket layout.

The layout rules follow DeepSeek-V4.1-Flash ``inference/engram.py`` (MIT): every (layer, n-gram order,
head) owns a distinct prime-sized bucket range of its layer's table; primes are drawn in order and never
reused; the search for each n-gram order restarts from ``vocab_size - 1``; hash multipliers are odd values
from ``numpy.random.default_rng(10007 * layer_id)``, bounded so ``token_id * multiplier`` fits in int64.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from functools import cached_property

import numpy as np

from .primes import find_next_prime


@dataclass(frozen=True)
class EngramSpec:
    """Static description of an Engram attachment (the hyper-parameters, not the weights)."""

    hidden_size: int
    layer_ids: tuple[int, ...]
    max_ngram_size: int = 4
    n_heads: int = 8
    head_dim: int = 256
    vocab_size: int = 16_000_000  # where the prime search starts for every (n-gram order, head)
    pad_token_id: int = 2  # raw tokenizer id that fills n-gram slots with no history
    hc_mult: int = 1  # hyper-connection copies of the residual stream (4 in DeepSeek-V4.1, 1 for plain decoders)
    norm_eps: float = 1e-20  # epsilon in the gate's RMS normalization (DeepSeek-V4.1 uses 1e-20)
    compressed_vocab_size: int | None = None  # filled from the tokenizer and asserted on load

    def __post_init__(self) -> None:
        object.__setattr__(self, "layer_ids", tuple(int(i) for i in self.layer_ids))
        if self.max_ngram_size < 2:
            raise ValueError("max_ngram_size must be >= 2 (Engram hashes 2-grams and up)")
        if len(set(self.layer_ids)) != len(self.layer_ids):
            raise ValueError("layer_ids must be unique")

    @property
    def n_hash_cols(self) -> int:
        """Rows fetched per position per layer: (max_ngram_size - 1) orders x n_heads heads."""
        return (self.max_ngram_size - 1) * self.n_heads

    def with_compressed_vocab(self, size: int) -> EngramSpec:
        if self.compressed_vocab_size is not None and self.compressed_vocab_size != size:
            raise ValueError(
                f"compressed vocab mismatch: spec says {self.compressed_vocab_size}, tokenizer gives {size}. "
                "Every hash multiplier depends on this number, so the table would be addressed wrongly."
            )
        d = asdict(self)
        d["compressed_vocab_size"] = size
        return EngramSpec(**d)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["layer_ids"] = list(self.layer_ids)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> EngramSpec:
        d = dict(d)
        d["layer_ids"] = tuple(d["layer_ids"])
        return cls(**d)

    @classmethod
    def from_deepseek_v41_config(cls, config: dict) -> EngramSpec:
        """Read the ``engram_*`` keys of a DeepSeek-V4.1 ``config.json`` (top level or ``text_config``)."""
        cfg = config.get("text_config", config)
        return cls(
            hidden_size=int(cfg["hidden_size"]),
            layer_ids=tuple(cfg["engram_layer_ids"]),
            max_ngram_size=int(cfg["engram_max_ngram_size"]),
            n_heads=int(cfg["engram_n_heads"]),
            head_dim=int(cfg["engram_head_dim"]),
            vocab_size=int(cfg["engram_vocab_size"]),
            pad_token_id=int(cfg.get("engram_pad_token_id", cfg.get("engram_pad_id", 2))),
            hc_mult=int(cfg.get("hc_mult", 1)),
            norm_eps=float(cfg.get("rms_norm_eps", 1e-20)),
            compressed_vocab_size=int(cfg["engram_compressed_vocab_size"]),
        )


def compute_primes(
    layer_ids: tuple[int, ...], max_ngram_size: int, n_heads: int, vocab_size: int
) -> tuple[tuple[tuple[int, ...], ...], ...]:
    """``primes[layer][order][head]``: distinct across everything, search restarts per order."""
    primes, seen = [], set()
    for _ in layer_ids:
        per_ngram = []
        for _ in range(max_ngram_size - 1):
            sizes, current = [], vocab_size - 1
            for _ in range(n_heads):
                current = find_next_prime(current, seen)
                seen.add(current)
                sizes.append(current)
            per_ngram.append(tuple(sizes))
        primes.append(tuple(per_ngram))
    return tuple(primes)


def compute_hash_multipliers(
    layer_ids: tuple[int, ...], max_ngram_size: int, compressed_vocab_size: int
) -> np.ndarray:
    """``[n_layers, max_ngram_size]`` odd int64 multipliers, seeded per layer with ``10007 * layer_id``."""
    max_long = np.iinfo(np.int64).max
    bound = max(1, (max_long // compressed_vocab_size) // 2)
    rows = []
    for layer_id in layer_ids:
        generator = np.random.default_rng(10007 * layer_id)
        values = generator.integers(low=0, high=bound, size=(max_ngram_size,), dtype=np.int64)
        rows.append(values * 2 + 1)
    return np.stack(rows)


@dataclass(frozen=True)
class EngramLayout:
    """The derived bucket layout: primes, per-column offsets and table sizes for each Engram layer."""

    spec: EngramSpec
    primes: tuple[tuple[tuple[int, ...], ...], ...] = field(repr=False)

    @classmethod
    def from_spec(cls, spec: EngramSpec) -> EngramLayout:
        return cls(spec, compute_primes(spec.layer_ids, spec.max_ngram_size, spec.n_heads, spec.vocab_size))

    @cached_property
    def flat_primes(self) -> list[list[int]]:
        """Per layer, the column moduli in hash-column order (order-major, head-minor)."""
        return [[p for per_ngram in layer for p in per_ngram] for layer in self.primes]

    @cached_property
    def offsets(self) -> np.ndarray:
        """``[n_layers, n_hash_cols]``: start row of each column's bucket range inside the layer table."""
        return np.stack([np.cumsum([0, *sizes[:-1]]) for sizes in self.flat_primes]).astype(np.int64)

    @cached_property
    def num_embeddings(self) -> tuple[int, ...]:
        """Rows of each layer's table (the sum of its column primes)."""
        return tuple(int(sum(sizes)) for sizes in self.flat_primes)

    @cached_property
    def multipliers(self) -> np.ndarray:
        if self.spec.compressed_vocab_size is None:
            raise ValueError("EngramSpec.compressed_vocab_size must be set before hashing")
        return compute_hash_multipliers(
            self.spec.layer_ids, self.spec.max_ngram_size, self.spec.compressed_vocab_size
        )

    def layer_index(self, layer_id: int) -> int:
        return self.spec.layer_ids.index(layer_id)

    def fingerprint(self) -> str:
        """Identifies the addressing scheme; a personal delta only makes sense for the same fingerprint."""
        payload = json.dumps({"spec": self.spec.to_dict(), "primes": self.primes}, sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()[:16]
