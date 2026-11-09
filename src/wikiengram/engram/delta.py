"""The personal sparse delta: WikiEngram's neural memory.

The effective row of an Engram table is ``base[row] + delta[row]``. The base table and every fusion parameter
stay frozen; only the delta is trained, and it stores only rows that personal knowledge actually touched.
Consequences:

* the personal memory is a small, independent file (``delta.safetensors``) that can be saved, loaded,
  versioned and shared without the base model;
* removing a delta row restores that row to the base table bit for bit, which is how exact deletion works;
* a row can be *masked* (tombstoned) instantly, before any retraining, to stop stale memory from acting.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

FORMAT_VERSION = 1


class DeltaLayer:
    """Sorted row ids (CPU) with values and lazy-Adam state (compute device)."""

    def __init__(self, dim: int, device: str | torch.device = "cpu"):
        self.dim = dim
        self.device = torch.device(device)
        self.rows = torch.empty(0, dtype=torch.int64)
        self.active = torch.empty(0, dtype=torch.bool)
        self.values = torch.empty(0, dim, device=self.device)
        self.exp_avg = torch.empty(0, dim, device=self.device)
        self.exp_avg_sq = torch.empty(0, dim, device=self.device)
        self.steps = torch.empty(0, dtype=torch.int32, device=self.device)

    def __len__(self) -> int:
        return int(self.rows.numel())

    # -------------------------------------------------------------------------------------- queries
    def find(self, query: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Positions of ``query`` rows inside this layer and whether each one is present."""
        query = query.detach().cpu().long()
        if len(self) == 0:
            return torch.zeros_like(query), torch.zeros(query.shape, dtype=torch.bool)
        pos = torch.searchsorted(self.rows, query).clamp_(max=len(self) - 1)
        return pos, self.rows[pos] == query

    def lookup(self, query: torch.Tensor, include_masked: bool = False) -> torch.Tensor:
        """Delta values for ``query`` rows (zeros for rows that are absent or masked)."""
        out = torch.zeros(query.numel(), self.dim, device=self.device)
        if len(self) == 0:
            return out
        pos, found = self.find(query)
        if not include_masked:
            found &= self.active[pos]
        hit = found.nonzero().squeeze(1)
        if hit.numel():
            out[hit.to(self.device)] = self.values[pos[hit].to(self.device)]
        return out

    # ------------------------------------------------------------------------------------ mutation
    def ensure(self, query: torch.Tensor) -> torch.Tensor:
        """Insert absent rows (zero values, zero state) and return the positions of all ``query`` rows."""
        query = torch.unique(query.detach().cpu().long())
        _, found = self.find(query)
        missing = query[~found]
        if missing.numel():
            n = missing.numel()
            rows = torch.cat([self.rows, missing])
            order = torch.argsort(rows)
            dev_order = order.to(self.device)
            self.rows = rows[order]
            self.active = torch.cat([self.active, torch.ones(n, dtype=torch.bool)])[order]
            zeros = torch.zeros(n, self.dim, device=self.device)
            self.values = torch.cat([self.values, zeros])[dev_order]
            self.exp_avg = torch.cat([self.exp_avg, zeros])[dev_order]
            self.exp_avg_sq = torch.cat([self.exp_avg_sq, zeros])[dev_order]
            self.steps = torch.cat(
                [self.steps, torch.zeros(n, dtype=torch.int32, device=self.device)]
            )[dev_order]
        pos, found = self.find(query)
        assert bool(found.all())
        return pos

    def positions_of(self, query: torch.Tensor) -> torch.Tensor:
        """Positions for rows that must already exist (raises otherwise)."""
        pos, found = self.find(query)
        if not bool(found.all()):
            raise KeyError("some rows are not present in the delta")
        return pos

    def remove(self, rows: torch.Tensor) -> int:
        """Drop rows: they fall back to the base table exactly. Returns how many were removed."""
        if len(self) == 0 or rows.numel() == 0:
            return 0
        keep = ~torch.isin(self.rows, rows.detach().cpu().long())
        removed = int((~keep).sum())
        dkeep = keep.to(self.device)
        self.rows, self.active = self.rows[keep], self.active[keep]
        self.values, self.exp_avg, self.exp_avg_sq = (
            self.values[dkeep], self.exp_avg[dkeep], self.exp_avg_sq[dkeep]
        )
        self.steps = self.steps[dkeep]
        return removed

    def set_active(self, rows: torch.Tensor, active: bool) -> None:
        pos, found = self.find(rows)
        self.active[pos[found]] = active

    def reset_rows(self, rows: torch.Tensor) -> None:
        """Zero values and optimizer state of existing rows (keeps them allocated)."""
        pos, found = self.find(rows)
        p = pos[found].to(self.device)
        self.values[p] = 0
        self.exp_avg[p] = 0
        self.exp_avg_sq[p] = 0
        self.steps[p] = 0

    def to(self, device) -> DeltaLayer:
        self.device = torch.device(device)
        for name in ("values", "exp_avg", "exp_avg_sq", "steps"):
            setattr(self, name, getattr(self, name).to(self.device))
        return self


class PersonalDelta:
    """Per-Engram-layer sparse deltas plus metadata, saved as one safetensors file."""

    def __init__(self, layer_ids: tuple[int, ...], dim: int, layout_fingerprint: str,
                 device: str | torch.device = "cpu", meta: dict | None = None):
        self.layer_ids = tuple(layer_ids)
        self.dim = dim
        self.layout_fingerprint = layout_fingerprint
        self.layers = {lid: DeltaLayer(dim, device) for lid in self.layer_ids}
        self.meta: dict = dict(meta or {})
        self.meta.setdefault("created_at", time.time())
        self.meta.setdefault("version", 0)
        self.scale = 1.0  # global strength; 0 disables personal memory without touching it

    def num_rows(self, active_only: bool = False) -> int:
        if active_only:
            return sum(int(layer.active.sum()) for layer in self.layers.values())
        return sum(len(layer) for layer in self.layers.values())

    def nbytes(self) -> int:
        return sum(len(layer) * (self.dim * 4 + 8) for layer in self.layers.values())

    def lookup(self, layer_id: int, rows: torch.Tensor) -> torch.Tensor:
        out = self.layers[layer_id].lookup(rows)
        return out * self.scale if self.scale != 1.0 else out

    def to(self, device) -> PersonalDelta:
        for layer in self.layers.values():
            layer.to(device)
        return self

    # ------------------------------------------------------------------------------------ persistence
    def save(self, path: str | Path, with_optimizer_state: bool = True) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tensors: dict[str, torch.Tensor] = {}
        for lid, layer in self.layers.items():
            tensors[f"{lid}.rows"] = layer.rows.contiguous()
            tensors[f"{lid}.active"] = layer.active.to(torch.uint8).contiguous()
            tensors[f"{lid}.values"] = layer.values.detach().float().cpu().contiguous()
            if with_optimizer_state:
                tensors[f"{lid}.exp_avg"] = layer.exp_avg.detach().float().cpu().contiguous()
                tensors[f"{lid}.exp_avg_sq"] = layer.exp_avg_sq.detach().float().cpu().contiguous()
                tensors[f"{lid}.steps"] = layer.steps.detach().cpu().contiguous()
        meta = dict(self.meta)
        meta.update(
            format_version=FORMAT_VERSION,
            layer_ids=list(self.layer_ids),
            dim=self.dim,
            layout_fingerprint=self.layout_fingerprint,
            saved_at=time.time(),
        )
        tmp = path.with_suffix(path.suffix + ".tmp")
        save_file(tensors, str(tmp), metadata={"wikiengram": json.dumps(meta)})
        tmp.replace(path)

    @classmethod
    def load(cls, path: str | Path, device: str | torch.device = "cpu",
             expect_fingerprint: str | None = None) -> PersonalDelta:
        with safe_open(str(path), framework="pt", device="cpu") as f:
            meta = json.loads(f.metadata()["wikiengram"])
            if expect_fingerprint is not None and meta["layout_fingerprint"] != expect_fingerprint:
                raise ValueError(
                    f"delta was trained for Engram layout {meta['layout_fingerprint']}, "
                    f"but the model uses {expect_fingerprint}"
                )
            delta = cls(tuple(meta["layer_ids"]), int(meta["dim"]), meta["layout_fingerprint"], device,
                        meta={k: v for k, v in meta.items()
                              if k not in ("layer_ids", "dim", "layout_fingerprint", "format_version")})
            keys = set(f.keys())
            for lid, layer in delta.layers.items():
                layer.rows = f.get_tensor(f"{lid}.rows").long()
                layer.active = f.get_tensor(f"{lid}.active").bool()
                layer.values = f.get_tensor(f"{lid}.values").to(layer.device)
                n = len(layer)
                if f"{lid}.exp_avg" in keys:
                    layer.exp_avg = f.get_tensor(f"{lid}.exp_avg").to(layer.device)
                    layer.exp_avg_sq = f.get_tensor(f"{lid}.exp_avg_sq").to(layer.device)
                    layer.steps = f.get_tensor(f"{lid}.steps").to(layer.device)
                else:
                    layer.exp_avg = torch.zeros(n, delta.dim, device=layer.device)
                    layer.exp_avg_sq = torch.zeros(n, delta.dim, device=layer.device)
                    layer.steps = torch.zeros(n, dtype=torch.int32, device=layer.device)
        return delta
