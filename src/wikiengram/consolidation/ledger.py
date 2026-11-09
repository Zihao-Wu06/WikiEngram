"""Which delta rows belong to which claim.

Engram rows are addressed by token ids alone, so the rows a claim's training samples can modify are known
exactly. The ledger records them. On delete or edit:

* rows owned by that claim alone are removed from the delta (exact restore to the base table);
* rows shared with other claims are reset and *repaired* by retraining only the surviving claims that touch
  them, restricted to those rows (SISA-style local retraining).
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np


class RowLedger:
    def __init__(self) -> None:
        # claim_id -> layer_id -> sorted unique rows
        self.claim_rows: dict[str, dict[int, np.ndarray]] = {}

    def __contains__(self, claim_id: str) -> bool:
        return claim_id in self.claim_rows

    def claims(self) -> list[str]:
        return list(self.claim_rows)

    def add(self, claim_id: str, layer_rows: dict[int, np.ndarray]) -> None:
        cur = self.claim_rows.setdefault(claim_id, {})
        for lid, rows in layer_rows.items():
            rows = np.unique(np.asarray(rows, dtype=np.int64))
            cur[lid] = np.union1d(cur.get(lid, np.empty(0, np.int64)), rows)

    def rows_of(self, claim_id: str) -> dict[int, np.ndarray]:
        return self.claim_rows.get(claim_id, {})

    def owners(self, layer_id: int, rows: np.ndarray, exclude: set[str] | None = None) -> set[str]:
        exclude = exclude or set()
        found = set()
        for cid, lr in self.claim_rows.items():
            if cid in exclude or layer_id not in lr:
                continue
            if np.intersect1d(lr[layer_id], rows, assume_unique=True).size:
                found.add(cid)
        return found

    def split_for_removal(self, claim_id: str) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray], set[str]]:
        """Return ``(exclusive_rows, shared_rows, affected_claims)`` for removing ``claim_id``."""
        mine = self.claim_rows.get(claim_id, {})
        exclusive, shared, affected = {}, {}, set()
        for lid, rows in mine.items():
            others = [lr[lid] for cid, lr in self.claim_rows.items() if cid != claim_id and lid in lr]
            other_rows = np.unique(np.concatenate(others)) if others else np.empty(0, np.int64)
            sh = np.intersect1d(rows, other_rows, assume_unique=True)
            exclusive[lid] = np.setdiff1d(rows, sh, assume_unique=True)
            shared[lid] = sh
            if sh.size:
                affected |= self.owners(lid, sh, exclude={claim_id})
        return exclusive, shared, affected

    def remove(self, claim_id: str) -> None:
        self.claim_rows.pop(claim_id, None)

    def all_rows(self) -> dict[int, np.ndarray]:
        acc: dict[int, list[np.ndarray]] = defaultdict(list)
        for lr in self.claim_rows.values():
            for lid, rows in lr.items():
                acc[lid].append(rows)
        return {lid: np.unique(np.concatenate(v)) for lid, v in acc.items()}

    # ---------------------------------------------------------------------------------- persistence
    def save(self, path: str | Path) -> None:
        data = {cid: {str(lid): rows.tolist() for lid, rows in lr.items()} for cid, lr in self.claim_rows.items()}
        path = Path(path)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(data))
        tmp.replace(path)

    @classmethod
    def load(cls, path: str | Path) -> RowLedger:
        led = cls()
        p = Path(path)
        if p.exists():
            for cid, lr in json.loads(p.read_text()).items():
                led.claim_rows[cid] = {int(lid): np.asarray(rows, dtype=np.int64) for lid, rows in lr.items()}
        return led
