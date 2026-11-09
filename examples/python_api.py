"""Use WikiEngram from Python: ingest, consolidate, ask, correct, sync."""

from wikiengram.config import WorkspaceConfig
from wikiengram.runtime import WikiEngram

cfg = WorkspaceConfig(name="demo")
cfg.backend.model = "Qwen/Qwen3-0.6B-Base"
cfg.consolidation.min_score = 0.0  # demo: consolidate everything right away
ws = WikiEngram.init("wikiengram-data", cfg)

if not ws.has_base():
    ws.adapt(steps=300, progress=print)  # once per backbone, backbone frozen

ws.wiki.ingest_file("examples/data/team_notes.md")
print(ws.plan().to_dict()["selected"])
report = ws.consolidate(progress=print)
print("verified", report["verified_claims"], "of", report["selected"])

for mode in ("wiki", "engram", "hybrid"):
    a = ws.ask("Where does Ilsa Varrow work?", mode=mode)
    print(mode, "->", a.text, f"({a.prompt_tokens} prompt tokens)")

claim = ws.wiki.search("Ilsa Varrow work", k=1)[0][0]
ws.wiki.correct(claim.id, object="Northgate Systems")
print(ws.sync())  # Engram forgets the old employer, wiki answers the new one
print(ws.ask("Where does Ilsa Varrow work?", mode="hybrid").text)
