"""``wikiengram`` command line interface."""

from __future__ import annotations

import json
import os
from pathlib import Path

import typer

app = typer.Typer(add_completion=False, no_args_is_help=True,
                  help="WikiEngram: Train Memory, Not Your LLM. LLMWiki explicit memory + DeepSeek Engram neural memory.")
wiki_app = typer.Typer(no_args_is_help=True, help="Inspect and edit the explicit memory (LLMWiki).")
app.add_typer(wiki_app, name="wiki")

WS_OPTION = typer.Option(None, "--workspace", "-w", envvar="WIKIENGRAM_WORKSPACE",
                         help="Workspace directory (default: ./wikiengram-data).")


def _ws_path(ws: Path | None) -> Path:
    return Path(ws) if ws else Path(os.environ.get("WIKIENGRAM_WORKSPACE", "wikiengram-data"))


def _open(ws: Path | None):
    from .runtime import WikiEngram

    return WikiEngram.open(_ws_path(ws))


def _out(obj, as_json: bool) -> None:
    if as_json:
        typer.echo(json.dumps(obj, indent=2, default=str, ensure_ascii=False))
    elif isinstance(obj, dict):
        for k, v in obj.items():
            typer.echo(f"{k}: {json.dumps(v, default=str, ensure_ascii=False) if isinstance(v, (dict, list)) else v}")
    else:
        typer.echo(str(obj))


def _progress(prefix: str):
    def cb(rec: dict) -> None:
        typer.echo(prefix + " " + " ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in rec.items()))

    return cb


# ------------------------------------------------------------------------------------- workspace
@app.command()
def init(workspace: Path | None = WS_OPTION,
         model: str = typer.Option(None, help="Backbone, e.g. Qwen/Qwen3-0.6B-Base"),
         device: str = typer.Option(None, help="auto | cuda | mps | cpu"),
         extractor: str = typer.Option(None, help="rule | openai"),
         llm_model: str = typer.Option(None, help="Model name for the OpenAI-compatible extractor")):
    """Create a workspace (config, LLMWiki layout, memory directories)."""
    from .config import WorkspaceConfig
    from .runtime import WikiEngram

    path = _ws_path(workspace)
    cfg_path = path / WikiEngram.CONFIG_FILE
    cfg = WorkspaceConfig.load(cfg_path) if cfg_path.exists() else WorkspaceConfig(name=path.name)
    if model:
        cfg.backend.model = model
    if device:
        cfg.backend.device = device
    if extractor:
        cfg.extractor.kind = extractor
    if llm_model:
        cfg.extractor.model = llm_model
    WikiEngram.init(path, cfg)
    typer.echo(f"workspace ready: {path} (backbone {cfg.backend.model}). Next: `wikiengram adapt` once, then ingest.")


@app.command()
def status(workspace: Path | None = WS_OPTION, as_json: bool = typer.Option(False, "--json")):
    """Show wiki, memory and model status."""
    _out(_open(workspace).status(), as_json)


# ------------------------------------------------------------------------------------- ingest
@app.command()
def ingest(paths: list[Path] = typer.Argument(None, help="Markdown / text files to ingest"),
           text: str = typer.Option(None, "--text", "-t", help="Ingest a text snippet instead of files"),
           kind: str = typer.Option("document", help="document | conversation | note"),
           title: str = typer.Option("note", help="Title for --text"),
           workspace: Path | None = WS_OPTION, as_json: bool = typer.Option(False, "--json")):
    """Extract knowledge from documents or conversations into the wiki."""
    ws = _open(workspace)
    reports = []
    if text:
        reports.append(ws.wiki.ingest_text(text, title=title, kind=kind))
    for p in paths or []:
        if p.suffix == ".json" and kind == "conversation":
            reports.append(ws.wiki.ingest_conversation(json.loads(p.read_text())))
        else:
            reports.append(ws.wiki.ingest_file(p, kind=kind))
    if not reports:
        raise typer.BadParameter("give file paths or --text")
    for r in reports:
        _out({"source": r.source_id, "added": len(r.added), "updated": len(r.updated),
              "unchanged": len(r.unchanged), "conflicts": r.conflicts}, as_json)


# ------------------------------------------------------------------------------------- wiki
def _claim_line(c) -> str:
    st = getattr(c.status, "value", c.status)
    eg = getattr(c.engram_state, "value", c.engram_state)
    return f"{c.id}  v{c.version}  [{st}|engram:{eg}]  {c.statement}"


@wiki_app.command("list")
def wiki_list(status: str = typer.Option("active", help="active | superseded | retracted | all"),
              workspace: Path | None = WS_OPTION):
    """List claims."""
    from .wiki import ClaimStatus

    ws = _open(workspace)
    for c in ws.wiki.claims(status=None if status == "all" else ClaimStatus(status)):
        typer.echo(_claim_line(c))


@wiki_app.command("search")
def wiki_search(query: str, k: int = 5, workspace: Path | None = WS_OPTION):
    """Lexical search over claims (BM25 + entity match)."""
    for c, score in _open(workspace).wiki.search(query, k=k):
        typer.echo(f"{score:7.3f}  {_claim_line(c)}")


@wiki_app.command("show")
def wiki_show(claim_id: str, workspace: Path | None = WS_OPTION):
    """Show a claim with its full version history."""
    ws = _open(workspace)
    for c in ws.wiki.history(claim_id):
        typer.echo(_claim_line(c) + f"   reason={c.reason!r} sources={c.source_ids}")


@wiki_app.command("add")
def wiki_add(subject: str, predicate: str, obj: str = typer.Argument(..., metavar="OBJECT"),
             statement: str = typer.Option(None), importance: float = 0.5, pinned: bool = False,
             workspace: Path | None = WS_OPTION):
    """Add a claim by hand."""
    c = _open(workspace).wiki.add_claim(subject, predicate, obj, statement, importance=importance, pinned=pinned)
    typer.echo(_claim_line(c))


@wiki_app.command("correct")
def wiki_correct(claim_id: str, obj: str = typer.Option(None, "--object"), statement: str = typer.Option(None),
                 reason: str = "user correction", workspace: Path | None = WS_OPTION):
    """Correct a claim (creates a new version; Engram is synced on the next `sync`/`ask`)."""
    c = _open(workspace).wiki.correct(claim_id, object=obj, statement=statement, reason=reason)
    typer.echo(_claim_line(c))


@wiki_app.command("retract")
def wiki_retract(claim_id: str, reason: str = "user deletion", workspace: Path | None = WS_OPTION):
    """Delete a claim (kept in history; Engram forgets it on the next `sync`/`ask`)."""
    typer.echo(_claim_line(_open(workspace).wiki.retract(claim_id, reason=reason)))


@wiki_app.command("restore")
def wiki_restore(claim_id: str, version: int = typer.Option(None), workspace: Path | None = WS_OPTION):
    """Re-activate a retracted claim, optionally an older version."""
    typer.echo(_claim_line(_open(workspace).wiki.restore(claim_id, version=version)))


@wiki_app.command("purge")
def wiki_purge(claim_id: str, yes: bool = typer.Option(False, "--yes", help="Confirm permanent deletion"),
               workspace: Path | None = WS_OPTION):
    """Permanently delete a claim and its history (privacy). Run `sync` afterwards."""
    if not yes:
        raise typer.BadParameter("this cannot be undone; pass --yes to confirm")
    _open(workspace).wiki.purge(claim_id)
    typer.echo(f"purged {claim_id}")


@wiki_app.command("lint")
def wiki_lint(workspace: Path | None = WS_OPTION):
    """Health check: contradictions, duplicates, low confidence, stale Engram memory, orphan pages."""
    r = _open(workspace).wiki.lint()
    _out({k: getattr(r, k) for k in ("contradictions", "duplicates", "low_confidence", "stale_engram",
                                     "orphan_pages")}, False)


# ------------------------------------------------------------------------------------- engram
@app.command()
def adapt(steps: int = typer.Option(None, help="Training steps (default from config)"),
          corpus: str = typer.Option(None, help="wikitext-2 | wikitext-103 | file.txt | hf:repo/file.parquet"),
          force: bool = typer.Option(False, help="Replace an existing base (archives the personal delta)"),
          workspace: Path | None = WS_OPTION):
    """Stage 1 (once per backbone): train the base Engram with the backbone frozen."""
    report = _open(workspace).adapt(steps=steps, corpus=corpus, force=force, progress=_progress("adapt"))
    typer.echo(f"base Engram ready: eval ppl {report['eval_ppl_no_engram']:.3f} (backbone) -> "
               f"{report['eval_ppl_engram']:.3f} (with Engram), {report['seconds']:.0f}s")


@app.command()
def consolidate(plan: bool = typer.Option(False, "--plan", help="Only show what would be trained and an estimate"),
                claim: list[str] = typer.Option(None, "--claim", help="Consolidate these claim ids, ignoring the policy thresholds "
                                                       "(versions already consolidated are skipped)"),
                verify: bool = typer.Option(True, help="Probe recall after training"),
                workspace: Path | None = WS_OPTION, as_json: bool = typer.Option(False, "--json")):
    """Stage 2: train stable, important, frequently used claims into the personal Engram delta."""
    ws = _open(workspace)
    if plan:
        p = ws.plan(claim or None)
        if as_json:
            _out(p.to_dict(), True)
            return
        typer.echo(f"{len(p.selected)} claims selected, {len(p.rejected)} not selected; "
                   f"{p.n_samples} samples, ~{p.est_tokens} tokens")
        typer.echo(f"estimated time: {p.est_seconds:.0f}s ({p.throughput_source})" if p.est_seconds
                   else "estimated time: unknown until the first consolidation on this machine")
        for s in p.selected[:20]:
            typer.echo(f"  + {s.claim_id} score={s.score:.2f} {s.reason}")
        return
    r = ws.consolidate(claim or None, verify=verify, progress=_progress("epoch"))
    if as_json:
        _out(r, True)
        return
    if not r.get("selected"):
        typer.echo(r.get("message", "nothing to do"))
        return
    t = r["training"]
    typer.echo(f"consolidated {r['selected']} claims in {t['seconds']:.0f}s ({t['tokens']} tokens, "
               f"{t['epochs_run']} epochs, {r['replayed_claims']} older claims rehearsed); verified "
               f"{r['verified_claims']}/{r['selected']}; delta rows {r['delta_rows']} (version {r['delta_version']})")


@app.command()
def sync(repair: bool = typer.Option(True, help="Retrain claims that shared rows with forgotten ones"),
         workspace: Path | None = WS_OPTION, as_json: bool = typer.Option(False, "--json")):
    """Apply wiki corrections/deletions to the neural memory and verify forgetting."""
    _out(_open(workspace).sync(repair=repair, progress=_progress("repair")), as_json)


@app.command()
def ask(question: str, mode: str = typer.Option(None, help="none | wiki | engram | hybrid"),
        workspace: Path | None = WS_OPTION, as_json: bool = typer.Option(False, "--json")):
    """Answer a question with explicit + neural memory."""
    a = _open(workspace).ask(question, mode)
    if as_json:
        _out(a.to_dict(), True)
    else:
        typer.echo(a.text)
        typer.echo(f"[{a.mode}] context claims={a.context_claims} engram claims={a.engram_claims} "
                   f"prompt tokens={a.prompt_tokens} {a.seconds:.2f}s", err=True)


@app.command()
def verify(claim: list[str] = typer.Option(None, "--claim"), workspace: Path | None = WS_OPTION):
    """Probe Engram-only recall of consolidated claims."""
    r = _open(workspace).verify(claim or None)
    _out(r.summary(), False)


@app.command()
def versions(workspace: Path | None = WS_OPTION):
    """List personal-memory versions."""
    for v in _open(workspace).versions():
        typer.echo(json.dumps(v, default=str))


@app.command()
def rollback(version: int, workspace: Path | None = WS_OPTION):
    """Restore an earlier personal-memory version (later wiki changes are re-synced)."""
    _open(workspace).rollback(version)
    typer.echo(f"restored memory version {version}; run `wikiengram sync`")


# ------------------------------------------------------------------------------------- bench / serve
@app.command()
def bench(facts: int = typer.Option(50, help="Number of fictitious facts"),
          model: str = typer.Option("Qwen/Qwen3-0.6B-Base"),
          out: Path = typer.Option(Path("benchmark-results"), help="Output directory"),
          adapt_steps: int = typer.Option(300), epochs: int = typer.Option(20),
          edit_fraction: float = typer.Option(0.2), delete_fraction: float = typer.Option(0.1),
          device: str = typer.Option("auto"), base: Path = typer.Option(None, help="Reuse a trained base Engram dir")):
    """Benchmark LLMWiki-only vs Engram-only vs LLMWiki+Engram (accuracy, memory, tokens, latency, sync)."""
    from .bench.runner import BenchConfig, run_benchmark

    cfg = BenchConfig(facts=facts, model=model, adapt_steps=adapt_steps, epochs=epochs, device=device,
                      edit_fraction=edit_fraction, delete_fraction=delete_fraction,
                      base_dir=str(base) if base else None)
    result = run_benchmark(cfg, out, progress=lambda m: typer.echo(m))
    typer.echo((out / "results.md").read_text())
    typer.echo(f"full results: {out / 'results.json'}")
    del result


@app.command()
def serve(host: str = "127.0.0.1", port: int = 8765, workspace: Path | None = WS_OPTION):
    """Run the HTTP API (FastAPI)."""
    import uvicorn

    from .api.server import create_app

    uvicorn.run(create_app(_ws_path(workspace)), host=host, port=port)


if __name__ == "__main__":
    app()
