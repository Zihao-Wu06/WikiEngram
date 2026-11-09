"""The Qwen PLE variant (Qwen3.8-Flash-Next's Engram-style n-gram memory).

Ported from transformers ``models/qwen4_exp/modeling_qwen4_exp.py`` (Apache License 2.0, Copyright 2026 The Qwen
Team and The HuggingFace Inc. team): ``Qwen4ExpTextNGramEmbedding`` (addressing) and ``Qwen4ExpTextPLELayer``
(fusion). ``tests/test_variant_parity.py`` checks both against a vendored copy of that code.

Same hash family as DeepSeek's Engram (rolling multiplicative-XOR, one distinct prime-sized bucket range per head,
offsets into one table), with these differences:

* raw token ids (no canonical-token compression); multipliers from splitmix64 seeded with
  ``seed + 10007 * ple_layer_index``; the k-th prime >= ``ngram_vocab_size_base`` per global head;
* look-back is filled with EOS and cut after an EOS (a token after an EOS does not see across it);
* fusion: zero-centred group RMSNorms (``x * (1 + w)``), separate key/value projections, the DeepSeek-style
  signed-sqrt sigmoid gate, then a *dilated causal depthwise conv* (kernel 4, dilation = n-gram size) over the
  normalized gated values. Because of the conv, a row's update reaches positions t, t+3, t+6, t+9, and decoding
  needs a small per-sequence state (handled by :meth:`QwenPLEModule.begin_stream`).

Weights of Qwen3.8-Flash-Next are distributed under the Qwen Community License; this module contains no weights.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .primes import is_prime
from .variant import EngramVariant, register

_MASK64 = (1 << 64) - 1
_SPLITMIX_GAMMA = 0x9E3779B97F4A7C15
_SPLITMIX_M1 = 0xBF58476D1CE4E5B9
_SPLITMIX_M2 = 0x94D049BB133111EB
_PRIME_1 = 10007


def _splitmix64(value: int) -> int:
    value = (value + _SPLITMIX_GAMMA) & _MASK64
    value = ((value ^ (value >> 30)) * _SPLITMIX_M1) & _MASK64
    value = ((value ^ (value >> 27)) * _SPLITMIX_M2) & _MASK64
    return (value ^ (value >> 31)) & _MASK64


def layer_multipliers(unigram_vocab_size: int, ngram_size: int, ple_layer_index: int, seed: int) -> np.ndarray:
    half_bound = max(1, (((1 << 63) - 1) // max(unigram_vocab_size, 1)) // 2)
    base_seed = seed + _PRIME_1 * ple_layer_index
    out = []
    for index in range(ngram_size):
        value = (base_seed + _SPLITMIX_GAMMA * (index + 1)) & _MASK64
        out.append(2 * (_splitmix64(value) % half_bound) + 1)
    return np.asarray(out, dtype=np.int64)


def nth_prime_after(start: int, count: int) -> int:
    prime = start
    for _ in range(count):
        prime += 1
        while not is_prime(prime):
            prime += 1
    return prime


@dataclass(frozen=True)
class QwenPLESpec:
    hidden_size: int
    layer_ids: tuple[int, ...]  # 0-indexed blocks (Qwen's config ``ple_layer_ids`` is 1-indexed)
    ngram_size: int = 3
    heads_per_ngram: int = 8
    row_dim: int = 160  # ple_embed_dim // ((ngram_size - 1) * heads_per_ngram)
    unigram_vocab_size: int = 248320  # model vocab size, used to bound the multipliers
    ngram_vocab_size_base: int = 20_000_000
    seed: int = 1234
    eos_token_id: int = 248044
    hc_mult: int = 1
    conv_kernel_size: int = 4
    vocab_divisor: int = 128
    norm_eps: float = 1e-6

    def __post_init__(self) -> None:
        object.__setattr__(self, "layer_ids", tuple(int(i) for i in self.layer_ids))

    @property
    def n_heads_total(self) -> int:
        return (self.ngram_size - 1) * self.heads_per_ngram

    @property
    def embed_dim(self) -> int:
        return self.n_heads_total * self.row_dim

    def to_dict(self) -> dict:
        d = asdict(self)
        d["layer_ids"] = list(self.layer_ids)
        return d


@register
class QwenPLEEngram(EngramVariant):
    name = "qwen-ple"

    def __init__(self, spec: QwenPLESpec):
        self.spec = spec
        heads = spec.n_heads_total
        self._mult, self._sizes, self._offsets, self._num_rows = [], [], [], []
        for index in range(len(spec.layer_ids)):
            self._mult.append(torch.as_tensor(
                layer_multipliers(spec.unigram_vocab_size, spec.ngram_size, index, spec.seed)))
            sizes = [nth_prime_after(spec.ngram_vocab_size_base - 1, index * heads + h + 1) for h in range(heads)]
            offsets = np.cumsum([0, *sizes[:-1]])
            total = int(sum(sizes))
            self._sizes.append(torch.tensor(sizes, dtype=torch.int64))
            self._offsets.append(torch.as_tensor(offsets, dtype=torch.int64))
            self._num_rows.append(math.ceil(total / spec.vocab_divisor) * spec.vocab_divisor)

    # ------------------------------------------------------------------------------------- geometry
    layer_ids = property(lambda self: self.spec.layer_ids)
    hidden_size = property(lambda self: self.spec.hidden_size)
    row_dim = property(lambda self: self.spec.row_dim)
    num_rows = property(lambda self: tuple(self._num_rows))
    n_groups = property(lambda self: self.spec.ngram_size - 1)
    group_size = property(lambda self: self.spec.heads_per_ngram)
    max_lookback = property(lambda self: self.spec.ngram_size - 1)

    # ------------------------------------------------------------------------------------- addressing
    def _shifted(self, input_ids: torch.Tensor, token_mask: torch.Tensor | None) -> list[torch.Tensor]:
        """Look-back tokens with EOS fill/cut, on a history that starts with ``ngram_size - 1`` EOS tokens."""
        eos = self.spec.eos_token_id
        ids = input_ids.detach().to("cpu", torch.int64)
        if ids.dim() == 1:
            ids = ids.unsqueeze(0)
        if token_mask is not None:  # padding behaves like an EOS, as in the reference model
            ids = torch.where(token_mask.detach().to("cpu", torch.bool), ids, eos)
        ctx = self.spec.ngram_size - 1
        hist = torch.cat([ids.new_full((ids.shape[0], ctx), eos), ids], dim=1)
        batch, n = hist.shape
        positions = torch.arange(n)
        eos_pos = torch.where(hist == eos, positions, -1)
        prev_inclusive = torch.cummax(eos_pos, dim=1).values
        previous = torch.cat([eos_pos.new_full((batch, 1), -1), prev_inclusive[:, :-1]], dim=1)
        pos_in_segment = positions.unsqueeze(0) - (previous + 1)
        out = []
        for shift in range(self.spec.ngram_size):
            if shift == 0:
                out.append(hist)
                continue
            source = positions - shift
            gathered = hist.gather(1, source.clamp_min(0).unsqueeze(0).expand(batch, -1))
            valid = (pos_in_segment >= shift) & (source.unsqueeze(0) >= 0)
            out.append(torch.where(valid, gathered, hist.new_full((), eos)))
        return out

    @torch.no_grad()
    def hash(self, input_ids: torch.Tensor, token_mask: torch.Tensor | None = None) -> torch.Tensor:
        shifted = self._shifted(input_ids, token_mask)
        seq = shifted[0].shape[1] - (self.spec.ngram_size - 1)
        layers = []
        for li in range(len(self.spec.layer_ids)):
            m, sizes, offs = self._mult[li], self._sizes[li], self._offsets[li]
            blocks = []
            for n in range(2, self.spec.ngram_size + 1):
                lo = (n - 2) * self.spec.heads_per_ngram
                hi = lo + self.spec.heads_per_ngram
                mixed = shifted[0] * m[0]
                for p in range(1, n):
                    mixed = torch.bitwise_xor(mixed, shifted[p] * m[p])
                blocks.append(torch.remainder(mixed.unsqueeze(-1), sizes[lo:hi]) + offs[lo:hi])
            layers.append(torch.cat(blocks, dim=-1)[:, -seq:])
        return torch.stack(layers, dim=2)

    @torch.no_grad()
    def ngram_keys(self, input_ids: torch.Tensor, token_mask: torch.Tensor | None = None) -> torch.Tensor:
        shifted = self._shifted(input_ids, token_mask)
        seq = shifted[0].shape[1] - (self.spec.ngram_size - 1)
        m = self._mult[0]
        keys, mixed = [], shifted[0] * m[0]
        for p in range(1, self.spec.ngram_size):
            mixed = torch.bitwise_xor(mixed, shifted[p] * m[p])
            keys.append(mixed)
        return torch.stack(keys, dim=-1)[:, -seq:]

    def make_module(self, layer_id: int) -> nn.Module:
        return QwenPLEModule(self.spec, layer_id)

    # ------------------------------------------------------------------------------------- identity
    def fingerprint(self) -> str:
        payload = json.dumps({"variant": self.name, "spec": self.spec.to_dict()}, sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def to_dict(self) -> dict:
        return {"variant": self.name, **self.spec.to_dict()}

    @classmethod
    def from_dict(cls, d: dict, tokenizer=None, cache_dir=None) -> QwenPLEEngram:
        fields = {k: v for k, v in d.items() if k != "variant"}
        fields["layer_ids"] = tuple(fields["layer_ids"])
        return cls(QwenPLESpec(**fields))

    @classmethod
    def from_model_config(cls, config: dict, tokenizer=None, cache_dir=None) -> QwenPLEEngram:
        t = config.get("text_config", config)
        eos = t.get("eos_token_id", config.get("eos_token_id"))
        eos = eos[0] if isinstance(eos, list) else eos
        heads = (t["ngram_size"] - 1) * t["heads_per_ngram"]
        return cls(QwenPLESpec(
            hidden_size=int(t["hidden_size"]), layer_ids=tuple(i - 1 for i in t["ple_layer_ids"]),
            ngram_size=int(t["ngram_size"]), heads_per_ngram=int(t["heads_per_ngram"]),
            row_dim=int(t["ple_embed_dim"]) // heads, unigram_vocab_size=int(t["vocab_size"]),
            ngram_vocab_size_base=int(t["ngram_vocab_size_base"]), seed=int(t.get("seed", 1234)),
            eos_token_id=int(eos), hc_mult=int(t.get("hc_count", 1)),
            conv_kernel_size=int(t.get("ple_conv_kernel_size", 4)),
            vocab_divisor=int(t.get("make_ngram_vocab_size_divisible_by", 128)),
            norm_eps=float(t.get("rms_norm_eps", 1e-6)),
        ))

    @classmethod
    def detect(cls, config: dict) -> bool:
        t = config.get("text_config", config)
        return bool(t.get("ple_layer_ids")) and "ngram_vocab_size_base" in t

    @classmethod
    def for_retrofit(cls, cfg, hidden_size: int, tokenizer, model_vocab_size: int, cache_dir=None) -> QwenPLEEngram:
        eos = tokenizer.eos_token_id
        return cls(QwenPLESpec(hidden_size=hidden_size, layer_ids=tuple(cfg.layer_ids), ngram_size=cfg.max_ngram_size,
                               heads_per_ngram=cfg.n_heads, row_dim=cfg.head_dim, unigram_vocab_size=model_vocab_size,
                               ngram_vocab_size_base=cfg.vocab_size, eos_token_id=int(eos), norm_eps=cfg.norm_eps))


class _ZeroCenteredGroupRMSNorm(nn.Module):
    """``Qwen4ExpTextRMSNorm`` with ``group_size``: fp32 normalization per group, scale ``1 + w``."""

    def __init__(self, dim: int, group_size: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(dim))
        self.group_size, self.eps = group_size, eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x.float().reshape(*x.shape[:-1], -1, self.group_size)
        h = (h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + self.eps)).flatten(-2)
        return (h * (1.0 + self.weight.float())).type_as(x)


class QwenPLEModule(nn.Module):
    """``Qwen4ExpTextPLELayer`` fusion with fetched rows as input. Returns the updated residual (``h + out``)."""

    def __init__(self, spec: QwenPLESpec, layer_id: int):
        super().__init__()
        self.spec, self.layer_id = spec, layer_id
        d, hc = spec.hidden_size, spec.hc_mult
        self.hidden_size, self.hc_count = d, hc
        self.key_proj = nn.Linear(spec.embed_dim, hc * d, bias=False)
        self.value_proj = nn.Linear(spec.embed_dim, d, bias=False)
        self.norm_key = _ZeroCenteredGroupRMSNorm(hc * d, d, spec.norm_eps)
        self.norm_query = _ZeroCenteredGroupRMSNorm(hc * d, d, spec.norm_eps)
        self.norm_conv = _ZeroCenteredGroupRMSNorm(hc * d, d, spec.norm_eps)
        self.dilation = spec.ngram_size
        self.state_len = (spec.conv_kernel_size - 1) * self.dilation
        self.conv1d = nn.Conv1d(hc * d, hc * d, kernel_size=spec.conv_kernel_size, groups=hc * d,
                                dilation=self.dilation, bias=False)
        self._stream: torch.Tensor | None = None  # [B, C, state_len] conv inputs of earlier positions
        self._streaming = False
        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self, zero_value: bool = True) -> None:
        nn.init.normal_(self.key_proj.weight, std=self.spec.embed_dim**-0.5)
        nn.init.normal_(self.value_proj.weight, std=self.spec.embed_dim**-0.5)
        if zero_value:  # gated value 0 -> conv input 0 -> output 0: an exact identity at init
            self.value_proj.weight.zero_()
        for norm in (self.norm_key, self.norm_query, self.norm_conv):
            norm.weight.zero_()

    # streaming state for KV-cached decoding (the conv looks back state_len positions)
    def begin_stream(self) -> None:
        self._streaming, self._stream = True, None

    def end_stream(self) -> None:
        self._streaming, self._stream = False, None

    def _short_conv(self, x: torch.Tensor) -> torch.Tensor:
        seq = x.shape[1]
        x = x.transpose(1, 2)  # [B, C, L]
        if self._streaming:
            prev = self._stream if self._stream is not None else x.new_zeros(x.shape[0], x.shape[1], self.state_len)
            x = torch.cat([prev, x], dim=-1)
            self._stream = x[..., -self.state_len :].detach()
        x = F.pad(x, (self.state_len, 0))[..., -(self.state_len + seq) :]
        return F.silu(self.conv1d(x)).transpose(1, 2)

    def forward(self, x: torch.Tensor, emb: torch.Tensor, token_mask: torch.Tensor | None = None) -> torch.Tensor:
        """``x``: ``[B, L, D]`` or ``[B, L, hc, D]``; ``emb``: ``[B, L, rows, row_dim]``."""
        squeeze = x.dim() == 3
        h = (x.unsqueeze(2) if squeeze else x).flatten(-2)  # [B, L, hc*D]
        e = emb.flatten(-2).to(self.key_proj.weight.dtype)
        key = self.norm_key(self.key_proj(e)).unflatten(-1, (self.hc_count, self.hidden_size))
        value = self.value_proj(e)
        query = self.norm_query(h.to(key.dtype)).unflatten(-1, (self.hc_count, self.hidden_size))
        gate = (key * query).sum(dim=-1, keepdim=True) / math.sqrt(self.hidden_size)
        gate = gate.abs().clamp_min(1e-6).sqrt() * gate.sign()
        gated = torch.sigmoid(gate) * value.unsqueeze(-2)
        gated_normed = self.norm_conv(gated.flatten(-2))
        gated = gated.flatten(-2)
        if token_mask is not None:
            m = token_mask.to(gated.device)[:, :, None]
            gated, gated_normed = (gated * m).to(gated.dtype), (gated_normed * m).to(gated_normed.dtype)
        out = gated + self._short_conv(gated_normed)
        y = h + out.to(h.dtype)
        y = y.unflatten(-1, (self.hc_count, self.hidden_size))
        return (y.squeeze(2) if squeeze else y).to(x.dtype)
