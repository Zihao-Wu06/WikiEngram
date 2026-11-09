"""Real-model check (skipped unless WIKIENGRAM_SLOW=1): Qwen3-0.6B-Base + trained base Engram learns facts
through the personal delta alone, keeps the backbone frozen, and forgets a deleted fact.

    WIKIENGRAM_SLOW=1 WIKIENGRAM_BASE=<workspace>/engram/base pytest tests/test_slow_qwen.py -q
"""

from __future__ import annotations

import os
import shutil

import pytest

from wikiengram.config import WorkspaceConfig
from wikiengram.runtime import WikiEngram

pytestmark = pytest.mark.slow

FACTS = """Ilsa Varrow works at Quillmere Labs.
Tomas Okere lives in Port Danneth.
Brin Sallow's dog is named Pockel.
"""


def test_qwen_consolidate_ask_forget(tmp_path):
    base = os.environ.get("WIKIENGRAM_BASE")
    cfg = WorkspaceConfig(name="slow")
    cfg.backend.model = os.environ.get("WIKIENGRAM_MODEL", "Qwen/Qwen3-0.6B-Base")
    c = cfg.consolidation
    c.min_score, c.min_confidence, c.stability_hours = 0.0, 0.0, 0.0
    ws = WikiEngram.init(tmp_path / "ws", cfg)
    if base:
        shutil.copytree(base, ws.base_dir)
    else:
        ws.adapt(steps=int(os.environ.get("WIKIENGRAM_ADAPT_STEPS", "100")))
    ws.wiki.ingest_text(FACTS)
    backbone = ws.lm.backbone_checksum()
    rep = ws.consolidate()
    assert rep["selected"] == 3
    assert ws.lm.backbone_checksum() == backbone
    q = "Where does Ilsa Varrow work?"
    assert "Quillmere" not in ws.ask(q, mode="none").text
    assert "Quillmere" in ws.ask(q, mode="engram").text
    claim = next(x for x in ws.wiki.claims() if x.subject == "Ilsa Varrow")
    ws.wiki.retract(claim.id)
    ws.sync()
    assert "Quillmere" not in ws.ask(q, mode="engram").text
