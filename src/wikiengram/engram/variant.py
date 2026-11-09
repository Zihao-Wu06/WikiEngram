"""Engram variants: the model-specific part of an Engram-style conditional memory.

WikiEngram targets every (MoE) model that carries an Engram-style module: a table of embeddings addressed by
hashed n-grams of the input tokens, fused into the residual stream. Families differ in how rows are addressed
(hash function, table layout), how many rows a position reads, and how the fetched rows are fused. Everything
else - the personal sparse delta, the fact-to-row ledger, consolidation, sync, verification and fusion
routing - only needs the small interface below, so supporting a new model family means implementing one
:class:`EngramVariant` and registering it.

Contract:

* **Row addressing** is a pure function of token ids (computed on the CPU, before the forward pass, without the
  model): :meth:`hash` returns, per position and Engram layer, the absolute table rows that position reads.
* The rows of one position come in ``n_groups`` groups of ``group_size`` rows (for DeepSeek: n-gram orders x
  heads). Group ``g`` of a position corresponds to one n-gram, identified layer-independently by
  :meth:`ngram_keys` (used for statistics such as "which n-grams are generic").
* :meth:`make_module` returns the fusion module for a layer: ``module(hidden, emb, token_mask) -> hidden`` where
  ``emb`` is ``[B, L, rows_per_position, row_dim]``. It is applied to the residual stream before that block.
* :meth:`fingerprint` identifies the addressing scheme; a personal delta is only valid for the same one.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar

import numpy as np
import torch
from torch import nn


class EngramVariant(ABC):
    name: ClassVar[str]

    # ------------------------------------------------------------------------------------- geometry
    @property
    @abstractmethod
    def layer_ids(self) -> tuple[int, ...]:
        """Decoder block indices the module is applied before (0-indexed)."""

    @property
    @abstractmethod
    def hidden_size(self) -> int: ...

    @property
    @abstractmethod
    def row_dim(self) -> int:
        """Width of one table row."""

    @property
    @abstractmethod
    def num_rows(self) -> tuple[int, ...]:
        """Rows of each layer's table."""

    @property
    @abstractmethod
    def n_groups(self) -> int:
        """Row groups per position (one n-gram each)."""

    @property
    @abstractmethod
    def group_size(self) -> int:
        """Rows per group (e.g. hash heads)."""

    @property
    def rows_per_position(self) -> int:
        return self.n_groups * self.group_size

    @property
    @abstractmethod
    def max_lookback(self) -> int:
        """How many previous tokens a position's rows depend on."""

    # ------------------------------------------------------------------------------------- addressing
    @abstractmethod
    def hash(self, input_ids: torch.Tensor, token_mask: torch.Tensor | None = None) -> torch.Tensor:
        """``[B, L]`` token ids -> ``[B, L, n_layers, rows_per_position]`` absolute rows (int64, CPU).
        ``token_mask`` is False for tokens that take no part in an n-gram (padding)."""

    @abstractmethod
    def ngram_keys(self, input_ids: torch.Tensor, token_mask: torch.Tensor | None = None) -> torch.Tensor:
        """``[B, L, n_groups]`` layer-independent fingerprints of the n-gram behind each row group."""

    def rows_for_text_ids(self, ids: list[int] | torch.Tensor) -> list[np.ndarray]:
        """Unique rows per Engram layer touched by one token sequence (model-free)."""
        h = self.hash(torch.as_tensor(ids, dtype=torch.int64).unsqueeze(0))[0]  # [L, n_layers, R]
        return [np.unique(h[:, li, :].numpy()) for li in range(len(self.layer_ids))]

    # ------------------------------------------------------------------------------------- modules
    @abstractmethod
    def make_module(self, layer_id: int) -> nn.Module:
        """The fusion module applied before block ``layer_id``: ``module(hidden, emb, token_mask)``."""

    # ------------------------------------------------------------------------------------- identity
    @abstractmethod
    def fingerprint(self) -> str: ...

    @abstractmethod
    def to_dict(self) -> dict:
        """Serializable description; must contain ``"variant": self.name``."""

    @classmethod
    @abstractmethod
    def from_dict(cls, d: dict, tokenizer, cache_dir=None) -> EngramVariant: ...

    @classmethod
    def from_model_config(cls, config: dict, tokenizer, cache_dir=None) -> EngramVariant:
        """Read a native checkpoint's configuration (for models that ship the module)."""
        raise NotImplementedError(f"{cls.name}: reading native checkpoints is not implemented")

    @classmethod
    def detect(cls, config: dict) -> bool:
        """True if a model ``config.json`` describes this family's native module."""
        return False

    @classmethod
    def for_retrofit(cls, cfg, hidden_size: int, tokenizer, model_vocab_size: int, cache_dir=None) -> EngramVariant:
        """A fresh module of this family sized for a retrofit backbone, from the workspace ``[engram]`` config
        (``layer_ids``, ``max_ngram_size``, ``n_heads``, ``head_dim``, ``vocab_size`` = table size per head,
        ``norm_eps``)."""
        raise NotImplementedError(f"{cls.name}: cannot be attached to a retrofit backbone")


REGISTRY: dict[str, type[EngramVariant]] = {}


def register(cls: type[EngramVariant]) -> type[EngramVariant]:
    REGISTRY[cls.name] = cls
    return cls


def _register_builtins() -> None:
    from . import deepseek, qwen_ple  # noqa: F401  (importing registers them)


def variant_class(name: str) -> type[EngramVariant]:
    _register_builtins()

    if name not in REGISTRY:
        raise KeyError(f"unknown Engram variant {name!r}; known: {sorted(REGISTRY)}")
    return REGISTRY[name]


def variant_from_dict(d: dict, tokenizer, cache_dir=None) -> EngramVariant:
    """Rebuild a variant saved with :meth:`EngramVariant.to_dict` (checkpoints without a ``variant`` field
    predate the registry and are DeepSeek-style)."""
    return variant_class(d.get("variant", "deepseek-v4.1")).from_dict(d, tokenizer, cache_dir)


def detect_variant(config: dict) -> type[EngramVariant] | None:
    """The registered family whose native module a model ``config.json`` describes, if any."""
    _register_builtins()

    for cls in REGISTRY.values():
        if cls.detect(config):
            return cls
    return None
