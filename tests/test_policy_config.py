from __future__ import annotations

import time
from types import SimpleNamespace

from wikiengram.config import ConsolidationConfig, WorkspaceConfig
from wikiengram.consolidation.policy import score_claim, select_claims
from wikiengram.prompts import qa_prompt, statement_completion

NOW = 1_800_000_000.0


def claim(cid, *, importance=0.5, access=0, age_days=0.0, version=1, confidence=0.9, pinned=False,
          status="active", engram_state="none", engram_version=None):
    return SimpleNamespace(id=cid, importance=importance, access_count=access, updated_at=NOW - age_days * 86400,
                           version=version, confidence=confidence, pinned=pinned, status=status,
                           engram_state=engram_state, engram_version=engram_version)


def test_score_components_are_monotone():
    cfg = ConsolidationConfig()
    base, _ = score_claim(claim("a"), cfg, NOW)
    assert score_claim(claim("a", access=10), cfg, NOW)[0] > base
    assert score_claim(claim("a", age_days=7), cfg, NOW)[0] > base
    assert score_claim(claim("a", importance=1.0), cfg, NOW)[0] > base
    # frequently edited knowledge is less stable
    assert score_claim(claim("a", age_days=7, version=4), cfg, NOW)[0] < score_claim(claim("a", age_days=7), cfg, NOW)[0]
    _, comps = score_claim(claim("a", pinned=True, importance=0.1), cfg, NOW)
    assert comps["importance"] == 1.0


def test_selection_filters_and_orders():
    cfg = ConsolidationConfig(min_score=0.45, min_confidence=0.6, stability_hours=24, max_claims_per_run=2)
    claims = [
        claim("fresh", importance=1.0, access=50, age_days=0.1),  # too new
        claim("doubtful", importance=1.0, access=50, age_days=9, confidence=0.3),  # low confidence
        claim("boring", importance=0.1, age_days=2),  # low score
        claim("good", importance=0.9, access=20, age_days=10),
        claim("better", importance=1.0, access=50, age_days=30),
        claim("ok", importance=0.8, access=8, age_days=8),  # over the cap
        claim("pinned", importance=0.0, age_days=0.0, pinned=True),  # bypasses score and stability
        claim("done", importance=1.0, access=50, age_days=30, engram_state="consolidated", engram_version=1),
        claim("gone", importance=1.0, access=50, age_days=30, status="retracted"),
    ]
    selected, rejected = select_claims(claims, cfg, NOW)
    assert [s.claim_id for s in selected] == ["pinned", "better"]  # pinned first, then by score
    reasons = {r.claim_id: r.reason for r in rejected}
    assert "stability" in reasons["fresh"] and "confidence" in reasons["doubtful"]
    assert "score" in reasons["boring"] and "max_claims_per_run" in reasons["good"]
    assert "done" not in reasons and "gone" not in reasons
    # a new version of a consolidated claim is eligible again
    sel, _ = select_claims([claim("edited", importance=1, access=50, age_days=30, version=2,
                                  engram_state="stale", engram_version=1)], cfg, NOW)
    assert [s.claim_id for s in sel] == ["edited"]


def test_config_roundtrip(tmp_path):
    cfg = WorkspaceConfig(name="me")
    cfg.engram.layer_ids = [2, 7]
    cfg.backend.model = 'org/"quoted" model'
    cfg.consolidation.lr = 0.0123
    p = tmp_path / "wikiengram.toml"
    cfg.save(p)
    back = WorkspaceConfig.load(p)
    assert back == cfg


def test_prompts():
    assert qa_prompt("Where?") == "Question: Where?\nAnswer:"
    p = qa_prompt("Where?", ["A works at B.", "C lives in D."])
    assert p.startswith("Known facts:\n- A works at B.\n- C lives in D.\n\nQuestion: Where?")
    assert statement_completion("Zorblat Quenn works at Vexmora Labs.", "Vexmora Labs") == (
        "Zorblat Quenn works at", " Vexmora Labs.")
    assert statement_completion("Vexmora Labs employs Zorblat.", "Vexmora Labs") is None
    assert time.time() > 0
