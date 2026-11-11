"""Build a project memory from a real public project: Strata (https://github.com/Niko1221/Strata, MIT, by Niko1221).

The source snapshot (``source/``, commit in ``source/SNAPSHOT_COMMIT``) is turned into a WikiEngram workspace whose
knowledge is then consolidated into the Engram memory of Qwen3.8-Flash-Next. Steps:

    python build_memory.py init    --workspace WS --model /path/Qwen3.8-Flash-Next-FP8 --llm-url URL --llm-model NAME
    python build_memory.py ingest  --workspace WS                 # LLM extraction -> claims.json for review
    python build_memory.py review  --workspace WS --apply fixes.json
    python build_memory.py questions --workspace WS                # paraphrases, so training needs no LLM
    python build_memory.py finalize --workspace WS --max-claims 150
    wikiengram consolidate -w WS                                   # train the memory (the slow step)
    python build_memory.py export  --workspace WS --out release/

``fixes.json`` is a list of ``{"id": ..., "action": "forget" | "correct", "object": ..., "statement": ...}``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

HERE = Path(__file__).parent
SOURCE = HERE / "source"
SCOPE = "strata"
DOCS = ["README.md", "docs/HOW_IT_WORKS.md", "docs/INSTALL.md", "docs/CAPABILITIES.md", "docs/AI_REFERENCE.md",
        "docs/MCP_SERVER.md", "docs/DISJOINT_EXPERT_CACHE.md", "docs/KV_PREFETCH.md",
        "docs/MESSAGE_BOUNDARY_CACHE.md", "docs/PROMPT_CACHE_TAIL.md"]  # stable knowledge, not release notes


def _text(name: str) -> str:
    text = (SOURCE / name).read_text()
    if name == "README.md":  # project knowledge only: drop the sponsorship section (it lists people)
        text = text.split("\n## Support Strata", 1)[0]
    return text


def cmd_init(a) -> None:
    from wikiengram.config import WorkspaceConfig
    from wikiengram.runtime import WikiEngram

    cfg = WorkspaceConfig(name="strata-memory")
    b = cfg.backend
    b.kind, b.model, b.placement, b.expert_cache_gb, b.prompt_format = "hf-native", a.model, "stream", 0.0, "auto"
    cfg.extractor.kind, cfg.extractor.base_url, cfg.extractor.model = "openai", a.llm_url, a.llm_model
    cfg.extractor.timeout, cfg.extractor.max_chars = 600.0, 4000  # a local model: short chunks, patience
    c = cfg.consolidation
    c.min_score, c.min_confidence, c.stability_hours, c.max_claims_per_run = 0.0, 0.5, 0.0, 10_000
    # 4 phrasings are trained (each also as a plain chat message), a 5th is kept for verification
    c.questions_per_claim, c.include_statement, c.chat_bare_questions = 4, True, True
    # ~1,340 samples for 149 facts: eight streamed steps of 168 per epoch. On a 32 GB card the bound is one
    # decoder layer's recompute (its DeltaNet activations, every expert it uses in BF16): 168 fits with the
    # flash-linear-attention kernels (fla-core), about 100 with transformers' PyTorch fallback. Every step reads
    # all experts twice, so fewer, larger steps are faster
    c.epochs, c.batch_size, c.lr, c.exact_patience = 60, 168, 2e-2, 3
    cfg.fusion.max_new_tokens = 32
    WikiEngram.init(a.workspace, cfg)
    print(f"workspace ready: {a.workspace}")


def cmd_ingest(a) -> None:
    from wikiengram.runtime import WikiEngram

    ws = WikiEngram.open(a.workspace)
    done = {s.title for s in ws.wiki.sources()}
    for name in DOCS:
        if f"Strata {name}" in done:  # resumable: sources already ingested are skipped
            continue
        try:
            report = ws.wiki.ingest_text(_text(name), title=f"Strata {name}", kind="document", scope=SCOPE)
        except Exception as e:  # the extraction model could not handle this document: keep going without it
            print(f"{name}: skipped ({type(e).__name__}: {str(e)[:120]})")
            continue
        print(f"{name}: +{len(report.added)} added, {len(report.updated)} updated, {len(report.conflicts)} conflicts")
    claims = [{"id": c.id, "statement": c.statement, "subject": c.subject, "predicate": c.predicate,
               "object": c.object, "questions": c.questions, "confidence": c.confidence, "importance": c.importance}
              for c in ws.wiki.claims()]
    Path(a.workspace, "claims.json").write_text(json.dumps(claims, indent=1, ensure_ascii=False))
    print(f"{len(claims)} claims -> {Path(a.workspace, 'claims.json')}")


def cmd_review(a) -> None:
    from wikiengram.agent import Memory

    mem = Memory(a.workspace, scope=SCOPE)
    fixes = json.loads(Path(a.apply).read_text())
    for f in fixes:
        if f["action"] == "forget":
            mem.forget(f["id"], reason=f.get("reason", "review: not project knowledge"), purge=True)
        elif f["action"] == "add":  # a fact the extractor missed, stated in the docs
            mem.add_fact(f["subject"], f["predicate"], f["object"], statement=f["statement"])
        else:
            mem.correct(f["id"], object=f.get("object"), statement=f.get("statement"),
                        reason=f.get("reason", "review correction"))
            mem.wiki.set_questions(f["id"], [])  # the old phrasings fit the old statement; `questions` rewrites them
    print(f"applied {len(fixes)} fixes; {mem.status()['memories']} memories")


def cmd_questions(a) -> None:
    """Give every claim at least ``questions_per_claim + 1`` phrasings now (LLM paraphrases), so consolidation never
    needs the extraction model: it can be stopped to free the GPU for training."""
    from wikiengram.runtime import WikiEngram

    ws = WikiEngram.open(a.workspace)
    need = ws.config.consolidation.questions_per_claim + 1
    done = 0
    for c in ws.wiki.claims():
        qs = [q for q in c.questions if q.strip()]
        if len(qs) < need:
            for q in ws.extractor.paraphrase_questions(c, n=need + 1):
                if q not in qs and c.object.lower() not in q.lower():
                    qs.append(q)
            ws.wiki.set_questions(c.id, qs)
            done += 1
    print(f"questions added to {done} claims")


def cmd_plan(a) -> None:
    """The claims the next consolidation will train (policy order), with their questions, for a last check."""
    from wikiengram.runtime import WikiEngram

    ws = WikiEngram.open(a.workspace)
    plan = ws.plan()
    claims = {c.id: c for c in ws.wiki.claims()}
    rows = [{"id": s.claim_id, "score": round(s.score, 3), "statement": claims[s.claim_id].statement,
             "object": claims[s.claim_id].object, "questions": claims[s.claim_id].questions}
            for s in plan.selected]
    Path(a.workspace, "selected.json").write_text(json.dumps(rows, indent=1, ensure_ascii=False))
    for r in rows:
        print(f"{r['id'][-6:]} {r['statement'][:100]:100s} | {(r['questions'] or [''])[0][:60]}")
    print(f"{len(rows)} selected, {len(plan.rejected)} not selected -> {Path(a.workspace, 'selected.json')}")


def cmd_finalize(a) -> None:
    """Stop depending on the extraction model and cap what is consolidated (the most important claims)."""
    from wikiengram.config import tomllib

    path = Path(a.workspace, "wikiengram.toml")
    text = path.read_text()
    cfg = tomllib.loads(text)
    text = text.replace(f'kind = "{cfg["extractor"]["kind"]}"', 'kind = "rule"', 1)
    text = text.replace(f'max_claims_per_run = {cfg["consolidation"]["max_claims_per_run"]}',
                        f"max_claims_per_run = {a.max_claims}", 1)
    path.write_text(text)
    print(f"extractor: rule; at most {a.max_claims} claims per consolidation")


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def cmd_export(a) -> None:
    ws, out = Path(a.workspace), Path(a.out)
    if out.exists():
        shutil.rmtree(out)
    from wikiengram.config import tomllib
    from wikiengram.engram.delta import PersonalDelta

    shutil.copytree(ws / "engram", out / "engram", ignore=shutil.ignore_patterns("history"))
    # for answering, the rows are enough: the optimizer state (two more copies of every row) is left out, a third
    # of the size. Training it further starts its moments from zero again
    delta_path = out / "engram" / "delta.safetensors"
    PersonalDelta.load(delta_path).save(delta_path, with_optimizer_state=False)
    shutil.copytree(ws / "wiki", out / "wiki")
    text = (ws / "wikiengram.toml").read_text()
    cfg = tomllib.loads(text)
    if cfg["backend"].get("prompt_format", "auto") == "auto":  # pin the format the memory was trained in
        from transformers import AutoTokenizer

        from wikiengram.prompts import resolve_prompt_format

        tok = AutoTokenizer.from_pretrained(cfg["backend"]["model"])
        fmt = resolve_prompt_format("auto", cfg["backend"]["model"], tok, native=True)
        text = text.replace('prompt_format = "auto"', f'prompt_format = "{fmt}"', 1)
    # models by their public names, not this machine's paths (set [backend] model to a local copy if you have one)
    text = text.replace(f'model = "{cfg["backend"]["model"]}"', f'model = "{a.model_id}"', 1)
    ext = cfg["extractor"]["model"]
    if ext.startswith("/"):
        text = text.replace(f'model = "{ext}"', f'model = "Qwen/{Path(ext).name}"', 1)
    (out / "wikiengram.toml").write_text(text)
    shutil.copy(SOURCE / "LICENSE", out / "STRATA_LICENSE")
    (out / "ATTRIBUTION.md").write_text(
        "Knowledge in this memory was extracted from the documentation of Strata\n"
        f"(https://github.com/Niko1221/Strata, commit {(SOURCE / 'SNAPSHOT_COMMIT').read_text().strip()}),\n"
        "Copyright (c) Niko1221, MIT License (see STRATA_LICENSE).\n")
    files = sorted(p for p in out.rglob("*") if p.is_file())
    manifest = {str(p.relative_to(out)): {"bytes": p.stat().st_size, "sha256": _sha(p)} for p in files}
    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=1))
    print(f"exported {len(files)} files to {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("init", "ingest", "review", "questions", "finalize", "plan", "export"):
        p = sub.add_parser(name)
        p.add_argument("--workspace", required=True)
        if name == "init":
            p.add_argument("--model", required=True)
            p.add_argument("--llm-url", required=True)
            p.add_argument("--llm-model", required=True)
        if name == "review":
            p.add_argument("--apply", required=True)
        if name == "export":
            p.add_argument("--out", required=True)
            p.add_argument("--model-id", default="Qwen/Qwen3.8-Flash-Next-FP8")
        if name == "finalize":
            p.add_argument("--max-claims", type=int, default=150)
    a = ap.parse_args()
    {"init": cmd_init, "ingest": cmd_ingest, "review": cmd_review, "questions": cmd_questions,
     "finalize": cmd_finalize, "plan": cmd_plan, "export": cmd_export}[a.cmd](a)


if __name__ == "__main__":
    main()
