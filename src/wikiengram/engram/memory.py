"""Assembles the rows each Engram layer reads: frozen base table + personal delta.

For a batch, the rows of every layer are deduplicated once (``torch.unique``), fetched once, and indexed back
per position. Training a delta works by making the unique delta rows a leaf tensor ("gather to leaf"): its
gradient is dense over the batch's unique rows only, and the row-sparse optimizer scatters the update back.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .delta import PersonalDelta
from .tables import BaseTable
from .variant import EngramVariant


@dataclass
class LayerRows:
    """What one Engram layer needs for one batch."""

    rows: torch.Tensor  # [U] absolute row ids, CPU
    inv: torch.Tensor  # [B, L, C] indices into ``values`` (compute device)
    values: torch.Tensor  # [U, head_dim] effective rows (compute device), possibly a grad leaf
    delta_leaf: torch.Tensor | None = None  # the trainable part when training a delta
    trainable: torch.Tensor | None = None  # [U] bool: which rows may be updated (CPU)
    delta_values: torch.Tensor | None = None  # [U, head_dim] delta, kept apart when it is position-masked
    delta_mask: torch.Tensor | None = None  # [B, L] bool: positions where the personal delta applies

    def embeddings(self) -> torch.Tensor:
        emb = self.values[self.inv]
        if self.delta_values is not None:
            emb = emb + self.delta_values[self.inv] * self.delta_mask[..., None, None].to(emb.dtype)
        return emb


@dataclass
class EngramBatch:
    layers: dict[int, LayerRows]
    token_mask: torch.Tensor | None  # [B, L] on the compute device


class EngramMemory:
    def __init__(self, variant: EngramVariant, base_tables: dict[int, BaseTable],
                 delta: PersonalDelta | None, device: str | torch.device):
        self.variant = variant
        self.base_tables = base_tables
        self.delta = delta
        self.device = torch.device(device)
        missing = set(variant.layer_ids) - set(base_tables)
        if missing:
            raise ValueError(f"no base table for Engram layers {sorted(missing)}")

    def build_batch(
        self,
        hashes: torch.Tensor,
        token_mask: torch.Tensor | None = None,
        train_delta: bool = False,
        allowed_rows: dict[int, torch.Tensor] | None = None,
        use_delta: bool = True,
        delta_mask: torch.Tensor | None = None,
    ) -> EngramBatch:
        """``hashes [B, L, n_layers, C]`` (CPU) -> per-layer rows.

        ``train_delta``: delta rows become a gradient leaf. ``allowed_rows[layer]`` (sorted CPU tensor) limits
        which rows may be trained; other rows still contribute their current values but get no update.
        ``delta_mask [B, L]``: apply the personal delta only where True (e.g. to the question, not to wiki
        context in the prompt); the base table applies everywhere.
        """
        layers: dict[int, LayerRows] = {}
        for li, lid in enumerate(self.variant.layer_ids):
            h = hashes[..., li, :]
            uniq, inv = torch.unique(h.reshape(-1), return_inverse=True)
            inv = inv.view(h.shape).to(self.device)
            base = self.base_tables[lid].gather(uniq).to(self.device)
            if self.delta is None or not use_delta:
                layers[lid] = LayerRows(uniq, inv, base)
                continue
            if not train_delta:
                if delta_mask is None:
                    layers[lid] = LayerRows(uniq, inv, base + self.delta.lookup(lid, uniq))
                else:
                    layers[lid] = LayerRows(uniq, inv, base, delta_values=self.delta.lookup(lid, uniq),
                                            delta_mask=delta_mask.to(self.device, torch.bool))
                continue
            dl = self.delta.layers[lid]
            trainable = torch.ones(uniq.numel(), dtype=torch.bool)
            if allowed_rows is not None and lid in allowed_rows:
                trainable = torch.isin(uniq, allowed_rows[lid])
            current = dl.lookup(uniq, include_masked=False)
            leaf = current.detach().clone().requires_grad_(True)
            layers[lid] = LayerRows(uniq, inv, base + leaf, delta_leaf=leaf, trainable=trainable)
        mask = token_mask.to(self.device).bool() if token_mask is not None else None
        return EngramBatch(layers, mask)
