"""The DeepSeek Engram fusion module (DeepSeek-V4.1-Flash semantics), used by the ``deepseek-v4.1`` variant.

Ported from ``Engram`` in DeepSeek-V4.1-Flash ``inference/model.py`` (MIT License, Copyright (c) 2023
DeepSeek). As in V4.1 there is no ShortConv: one fused ``wkv`` projection turns the ``n_hash_cols`` fetched
rows into one key per hyper-connection copy plus one shared value; the gate is a per-copy RMS-normalized dot
product of the residual stream against the key, passed through a signed square root and a sigmoid; the output
is ``h + gate * value``. The module is applied to the residual stream *before* its decoder block.

The table itself is not part of this module. Rows are fetched by :class:`wikiengram.engram.memory.EngramMemory`
(frozen base table + personal sparse delta) and handed in as ``emb``. That keeps the fusion parameters (the
part shared by all tokens) separate from the rows (the part owned by individual facts), which is what makes
deletion exact.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .layout import EngramSpec


class EngramModule(nn.Module):
    def __init__(self, spec: EngramSpec, layer_id: int):
        super().__init__()
        self.spec = spec
        self.layer_id = layer_id
        self.dim = spec.hidden_size
        self.hc_mult = spec.hc_mult
        self.eps = spec.norm_eps
        self.clamp_value = 1e-6
        n_in = spec.n_hash_cols * spec.head_dim
        self.wkv = nn.Linear(n_in, self.dim * (self.hc_mult + 1), bias=False)
        self.q_weight = nn.Parameter(torch.ones(self.hc_mult, self.dim))
        self.k_weight = nn.Parameter(torch.ones(self.hc_mult, self.dim))
        self.record_gate = False
        self.last_gate: torch.Tensor | None = None
        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self, zero_value: bool = True) -> None:
        """Keys get a fan-in normal init. With ``zero_value`` the value projection starts at zero, so a fresh
        module is an exact identity on the residual stream (the backbone's behaviour is untouched)."""
        n_in = self.wkv.in_features
        nn.init.normal_(self.wkv.weight, std=n_in**-0.5)
        if zero_value:
            self.wkv.weight[self.hc_mult * self.dim :].zero_()
        self.q_weight.fill_(1.0)
        self.k_weight.fill_(1.0)

    def forward(
        self, x: torch.Tensor, emb: torch.Tensor, token_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """``x``: ``[B, L, D]`` (plain decoder) or ``[B, L, hc_mult, D]`` (hyper-connections).
        ``emb``: ``[B, L, n_hash_cols, head_dim]`` fetched rows. ``token_mask``: ``[B, L]``, False shuts the
        gate so those positions pass through untouched (padding)."""
        squeeze = x.dim() == 3
        h_in = x.unsqueeze(2) if squeeze else x
        kv = F.linear(emb.flatten(-2).to(self.wkv.weight.dtype), self.wkv.weight)
        key, value = kv.split([self.hc_mult * self.dim, self.dim], dim=-1)
        key = key.float().unflatten(-1, (self.hc_mult, self.dim))
        weight = self.q_weight.float() * self.k_weight.float()  # only ever used as a product
        h = h_in.float()
        # normalized per (token, hc copy) over `dim`, not jointly over the copies
        rstd = torch.rsqrt(h.square().mean(-1) + self.eps) * torch.rsqrt(key.square().mean(-1) + self.eps)
        dot = (h * weight * key).sum(-1) * rstd * self.dim**-0.5
        # signed sqrt before the sigmoid, matching DeepSeek's training kernel
        gate = torch.sigmoid(torch.copysign(dot.abs().clamp_min(self.clamp_value).sqrt(), dot))
        if token_mask is not None:
            gate = gate.masked_fill(~token_mask.to(gate.device, torch.bool).unsqueeze(-1), 0)
        if self.record_gate:
            self.last_gate = gate.detach()
        out = (h + gate.unsqueeze(-1) * value.float().unsqueeze(-2)).to(x.dtype)
        return out.squeeze(2) if squeeze else out

    def fusion_parameters(self) -> list[nn.Parameter]:
        return [self.wkv.weight, self.q_weight, self.k_weight]
