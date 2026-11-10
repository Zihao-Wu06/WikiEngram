"""Which wiki claims are worth consolidating into Engram.

Score = weighted sum of importance, usage frequency, stability and confidence (each in [0, 1]):

* importance - from the extractor or the user; pinned claims count as 1.0;
* frequency  - ``log1p(access_count) / log1p(saturation)``, capped at 1 (claims that keep answering questions);
* stability  - age of the current version relative to a saturation window, divided by the number of edits
  (knowledge that keeps changing should stay in the wiki, where updates are free);
* confidence - trust in the claim.

Only active claims whose current version is not already consolidated are eligible; a minimum confidence and a
minimum version age (stability window) are hard filters. Pinned claims bypass the score threshold.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

from ..config import ConsolidationConfig


@dataclass
class Selection:
    claim_id: str
    version: int
    score: float
    components: dict[str, float] = field(default_factory=dict)
    reason: str = ""


def score_claim(claim, cfg: ConsolidationConfig, now: float | None = None) -> tuple[float, dict[str, float]]:
    now = time.time() if now is None else now
    importance = 1.0 if claim.pinned else float(claim.importance)
    frequency = min(1.0, math.log1p(claim.access_count) / math.log1p(max(1, cfg.frequency_saturation)))
    age_days = max(0.0, now - (claim.updated_at or now)) / 86400.0
    stability = min(1.0, age_days / max(1e-9, cfg.stability_saturation_days)) / (1.0 + 0.5 * (claim.version - 1))
    confidence = float(claim.confidence)
    comps = {"importance": importance, "frequency": frequency, "stability": stability, "confidence": confidence}
    score = (cfg.w_importance * importance + cfg.w_frequency * frequency
             + cfg.w_stability * stability + cfg.w_confidence * confidence)
    return score, comps


def select_claims(claims, cfg: ConsolidationConfig, now: float | None = None) -> tuple[list[Selection], list[Selection]]:
    """Return ``(selected, rejected)``: pinned claims first, then by score; capped per run."""
    now = time.time() if now is None else now
    selected, rejected = [], []
    for c in claims:
        status = getattr(c.status, "value", c.status)
        if status != "active":
            continue
        state = getattr(c.engram_state, "value", c.engram_state)
        if state == "consolidated" and c.engram_version == c.version:
            continue
        score, comps = score_claim(c, cfg, now)
        sel = Selection(c.id, c.version, score, comps)
        age_h = max(0.0, now - (c.updated_at or now)) / 3600.0
        if getattr(c, "category", "fact") == "note":
            sel.reason = "free-form note, not an atomic fact: kept in the wiki only"
            rejected.append(sel)
        elif c.confidence < cfg.min_confidence:
            sel.reason = f"confidence {c.confidence:.2f} < {cfg.min_confidence}"
            rejected.append(sel)
        elif age_h < cfg.stability_hours and not c.pinned:
            sel.reason = f"version is {age_h:.1f} h old (< {cfg.stability_hours} h stability window)"
            rejected.append(sel)
        elif score < cfg.min_score and not c.pinned:
            sel.reason = f"score {score:.2f} < {cfg.min_score}"
            rejected.append(sel)
        else:
            sel.reason = "pinned" if c.pinned else f"score {score:.2f}"
            selected.append(sel)
    pinned = {c.id for c in claims if c.pinned}
    selected.sort(key=lambda s: (s.claim_id in pinned, s.score), reverse=True)  # user-pinned first
    if len(selected) > cfg.max_claims_per_run:
        for s in selected[cfg.max_claims_per_run :]:
            s.reason = f"over max_claims_per_run={cfg.max_claims_per_run} (deferred)"
        rejected.extend(selected[cfg.max_claims_per_run :])
        selected = selected[: cfg.max_claims_per_run]
    return selected, rejected
