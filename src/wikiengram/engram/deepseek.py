"""The DeepSeek Engram variant (DeepSeek-V4.1-Flash production semantics; the paper's design without ShortConv).

Ties together the bit-exact port in ``compression.py``, ``layout.py``, ``hashing.py`` and ``module.py``:
canonical-token compression, distinct prime buckets per (layer, n-gram order, head), rolling multiplicative-XOR
hashing, and the fused-``wkv`` gated fusion. Rows of a position are grouped by n-gram order
(``n_groups = max_ngram_size - 1``), each group holding one row per head (``group_size = n_heads``).
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from .compression import load_or_build_token_map
from .hashing import NgramHasher
from .layout import EngramLayout, EngramSpec
from .module import EngramModule
from .variant import EngramVariant, register


@register
class DeepSeekEngram(EngramVariant):
    name = "deepseek-v4.1"

    def __init__(self, spec: EngramSpec, token_map: np.ndarray):
        if spec.compressed_vocab_size is None:
            spec = spec.with_compressed_vocab(int(token_map.max()) + 1)
        self.spec = spec
        self.layout = EngramLayout.from_spec(spec)
        self.hasher = NgramHasher(self.layout, token_map)

    @classmethod
    def build(cls, spec: EngramSpec, tokenizer, cache_dir=None) -> DeepSeekEngram:
        """Compute the compressed token map for ``tokenizer`` (cached) and check it against ``spec``."""
        token_map, size = load_or_build_token_map(tokenizer, cache_dir)
        return cls(spec.with_compressed_vocab(size), token_map)

    # ------------------------------------------------------------------------------------- geometry
    @property
    def layer_ids(self) -> tuple[int, ...]:
        return self.spec.layer_ids

    @property
    def hidden_size(self) -> int:
        return self.spec.hidden_size

    @property
    def row_dim(self) -> int:
        return self.spec.head_dim

    @property
    def num_rows(self) -> tuple[int, ...]:
        return self.layout.num_embeddings

    @property
    def n_groups(self) -> int:
        return self.spec.max_ngram_size - 1

    @property
    def group_size(self) -> int:
        return self.spec.n_heads

    @property
    def max_lookback(self) -> int:
        return self.spec.max_ngram_size - 1

    # ------------------------------------------------------------------------------------- addressing
    def hash(self, input_ids: torch.Tensor, token_mask: torch.Tensor | None = None) -> torch.Tensor:
        return self.hasher(input_ids, token_mask)

    def ngram_keys(self, input_ids: torch.Tensor, token_mask: torch.Tensor | None = None) -> torch.Tensor:
        return self.hasher.ngram_keys(input_ids, token_mask)

    def make_module(self, layer_id: int) -> nn.Module:
        return EngramModule(self.spec, layer_id)

    # ------------------------------------------------------------------------------------- identity
    def fingerprint(self) -> str:
        return self.layout.fingerprint()

    def to_dict(self) -> dict:
        return {"variant": self.name, **self.spec.to_dict()}

    @classmethod
    def from_dict(cls, d: dict, tokenizer, cache_dir=None) -> DeepSeekEngram:
        spec = EngramSpec.from_dict({k: v for k, v in d.items() if k != "variant"})
        return cls.build(spec, tokenizer, cache_dir)

    @classmethod
    def from_model_config(cls, config: dict, tokenizer, cache_dir=None) -> DeepSeekEngram:
        return cls.build(EngramSpec.from_deepseek_v41_config(config), tokenizer, cache_dir)

    @classmethod
    def detect(cls, config: dict) -> bool:
        cfg = config.get("text_config", config)
        return bool(cfg.get("engram_layer_ids")) and "engram_vocab_size" in cfg

    @classmethod
    def for_retrofit(cls, cfg, hidden_size: int, tokenizer, model_vocab_size: int, cache_dir=None) -> DeepSeekEngram:
        pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        spec = EngramSpec(hidden_size=hidden_size, layer_ids=tuple(cfg.layer_ids), max_ngram_size=cfg.max_ngram_size,
                          n_heads=cfg.n_heads, head_dim=cfg.head_dim, vocab_size=cfg.vocab_size,
                          pad_token_id=int(pad), norm_eps=cfg.norm_eps)
        return cls.build(spec, tokenizer, cache_dir)
