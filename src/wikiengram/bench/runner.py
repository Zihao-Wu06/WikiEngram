"""End-to-end benchmark: LLMWiki only vs Engram only vs LLMWiki + Engram.

Pipeline (everything goes through the public :class:`~wikiengram.runtime.WikiEngram` API):

1. a document of fictitious facts is ingested into the wiki (rule extractor) -> claims;
2. every claim is consolidated into the personal Engram delta (backbone, base Engram and fusion frozen);
3. each memory mode answers trained questions and held-out paraphrases (never trained) -> accuracy, prompt
   tokens, latency, peak memory;
4. update phase: a fraction of facts is corrected and another fraction deleted in the wiki, memory is synced,
   and we measure that the new values are answered, old values are no longer produced by Engram, deleted facts
   are forgotten, and untouched facts are retained; then edited claims are re-consolidated.

Nothing is simulated: numbers come from real training and real generation on the configured hardware.
"""

from __future__ import annotations

import json
import platform
import random
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from ..config import WorkspaceConfig
from ..fusion.engine import prepare
from ..runtime import WikiEngram
from ..utils.device import PeakMemoryMeter, synchronize
from ..wiki import RuleExtractor
from .synthetic import RELATIONS, Fact, answer_matches, make_facts

MODES = ("none", "wiki", "engram", "hybrid")


@dataclass
class BenchConfig:
    facts: int = 50
    seed: int = 0
    model: str = "Qwen/Qwen3-0.6B-Base"
    device: str = "auto"
    adapt_steps: int = 300
    base_dir: str | None = None  # reuse a trained base Engram (directory with engram.json)
    epochs: int = 20
    lr: float = 2e-2
    batch_size: int = 16
    edit_fraction: float = 0.2
    delete_fraction: float = 0.1
    top_k: int = 5
    max_new_tokens: int = 16
    gen_batch: int = 16
    latency_queries: int = 10


class _BenchExtractor(RuleExtractor):
    """Rule extractor whose question phrasings come from the benchmark split (train + 1 verification
    question), so held-out evaluation questions are guaranteed never to be trained on."""

    override_questions = True

    def __init__(self, questions_by_subject: dict[str, list[str]]):
        super().__init__()
        self.by_subject = questions_by_subject

    def paraphrase_questions(self, claim, n: int = 3) -> list[str]:
        return list(self.by_subject.get(claim.subject, []))[:n]


def _document(facts: list[Fact]) -> str:
    return "\n".join(f.statement for f in facts) + "\n"


def _eval_mode(ws: WikiEngram, mode: str, items: list[tuple[str, str]], cfg: BenchConfig) -> dict:
    """``items``: (question, expected object). Batched generation; prompts built by the fusion engine."""
    lm = ws.lm
    prepared = [prepare(ws, q, mode) for q, _ in items]
    order = sorted(range(len(prepared)), key=lambda i: prepared[i].use_delta)  # batch by delta on/off
    correct, ptoks, ntoks, secs, outs = 0, 0, 0, 0.0, {}
    routed = sum(p.use_delta for p in prepared)
    with PeakMemoryMeter(lm.device) as meter:
        groups = [[i for i in order if prepared[i].use_delta == flag] for flag in (False, True)]
        for idx in groups:
            for j in range(0, len(idx), cfg.gen_batch):
                chunk = idx[j : j + cfg.gen_batch]
                res = lm.generate([prepared[i].prompt for i in chunk], max_new_tokens=cfg.max_new_tokens,
                                  use_engram=prepared[chunk[0]].use_engram, use_delta=prepared[chunk[0]].use_delta,
                                  delta_start=[prepared[i].context_tokens for i in chunk])
                meter.poll()
                for i, text in zip(chunk, res.texts):
                    correct += answer_matches(text, items[i][1])
                    outs[i] = text
                ptoks += sum(res.prompt_tokens)
                ntoks += sum(res.new_tokens)
                secs += res.seconds
    n = max(1, len(items))
    return {"accuracy": correct / n, "n": len(items), "avg_prompt_tokens": ptoks / n,
            "avg_new_tokens": ntoks / n, "batched_sec_per_query": secs / n, "peak_memory_bytes": meter.peak,
            "answered_by_engram": routed / n, "examples": [outs[i] for i in sorted(outs)[:3]]}


def _latency(ws: WikiEngram, mode: str, questions: list[str], cfg: BenchConfig) -> dict:
    """Single-query latency (batch size 1): time to first token and total, median over questions."""
    lm = ws.lm
    ttft, total = [], []
    for q in questions:
        p = prepare(ws, q, mode)
        res = lm.generate([p.prompt], max_new_tokens=cfg.max_new_tokens, use_engram=p.use_engram,
                          use_delta=p.use_delta, delta_start=[p.context_tokens])
        ttft.append(res.first_token_seconds)
        total.append(res.seconds)
    med = lambda xs: sorted(xs)[len(xs) // 2] if xs else float("nan")  # noqa: E731
    return {"median_ttft_s": med(ttft), "median_total_s": med(total), "n": len(questions)}


def run_benchmark(cfg: BenchConfig, out_dir: str | Path, progress=print) -> dict:
    out = Path(out_dir)
    ws_dir = out / "workspace"
    if ws_dir.exists():
        shutil.rmtree(ws_dir)
    rng = random.Random(cfg.seed)
    facts = make_facts(cfg.facts, seed=cfg.seed)

    wcfg = WorkspaceConfig(name="benchmark")
    wcfg.backend.model, wcfg.backend.device = cfg.model, cfg.device
    wcfg.adapt.steps = cfg.adapt_steps
    c = wcfg.consolidation
    c.min_score, c.min_confidence, c.stability_hours = 0.0, 0.0, 0.0
    c.max_claims_per_run = 10_000
    c.epochs, c.lr, c.batch_size = cfg.epochs, cfg.lr, cfg.batch_size
    c.questions_per_claim, c.include_statement = 2, True
    wcfg.fusion.top_k, wcfg.fusion.max_new_tokens = cfg.top_k, cfg.max_new_tokens
    ws = WikiEngram.init(ws_dir, wcfg)

    # 1. base Engram (stage 1), reused across runs when available
    t0 = time.perf_counter()
    cached = Path(cfg.base_dir) if cfg.base_dir else out / "base_cache"
    if (cached / "engram.json").exists():
        shutil.copytree(cached, ws.base_dir)
        adapt_info = json.loads((cached / "engram.json").read_text()).get("info", {})
        progress(f"reusing base Engram from {cached}")
    else:
        adapt_info = ws.adapt(steps=cfg.adapt_steps, progress=lambda r: progress(f"adapt {r}"))
        shutil.copytree(ws.base_dir, cached, dirs_exist_ok=True)
    adapt_seconds = time.perf_counter() - t0

    # 2. ingest the facts document through the wiki
    report = ws.wiki.ingest_text(_document(facts), title="benchmark facts", kind="document")
    claims = {cl.subject: cl for cl in ws.wiki.claims()}
    missing = [f.subject for f in facts if f.subject not in claims]
    if missing:
        raise RuntimeError(f"extractor missed {len(missing)} facts, e.g. {missing[:3]}")
    fact_claim = {f.fact_id: claims[f.subject].id for f in facts}
    ws.extractor = _BenchExtractor({f.subject: f.train_questions + f.heldout_questions[:1] for f in facts})

    # 3. consolidation (stage 2) with frozen-ness checks and peak memory
    lm = ws.lm
    backbone_before, fusion_before = lm.backbone_checksum(), lm.fusion_checksum()
    with PeakMemoryMeter(lm.device) as meter:
        cons = ws.consolidate(progress=lambda r: (meter.poll(), progress(f"consolidate {r}")))
    synchronize(lm.device)
    frozen_ok = lm.backbone_checksum() == backbone_before and lm.fusion_checksum() == fusion_before

    trained_q = [(f.train_questions[0], f.obj) for f in facts]
    heldout_q = [(f.heldout_questions[-1], f.obj) for f in facts]  # used neither in training nor verification
    results: dict = {"modes": {}}
    for mode in MODES:
        results["modes"][mode] = {
            "trained_question": _eval_mode(ws, mode, trained_q, cfg),
            "heldout_paraphrase": _eval_mode(ws, mode, heldout_q, cfg),
            "latency": _latency(ws, mode, [q for q, _ in heldout_q[: cfg.latency_queries]], cfg),
        }
        r = results["modes"][mode]
        progress(f"{mode:7s} trained={r['trained_question']['accuracy']:.3f} "
                 f"heldout={r['heldout_paraphrase']['accuracy']:.3f} "
                 f"tokens={r['heldout_paraphrase']['avg_prompt_tokens']:.0f}")

    # 4. update phase: corrections and deletions in the wiki, then memory sync
    ids = list(range(len(facts)))
    rng.shuffle(ids)
    n_edit, n_del = int(cfg.edit_fraction * len(facts)), int(cfg.delete_fraction * len(facts))
    edit_idx, del_idx, keep_idx = ids[:n_edit], ids[n_edit : n_edit + n_del], ids[n_edit + n_del :]
    new_obj = {}
    vrng = random.Random(cfg.seed + 99)
    for i in edit_idx:
        f = facts[i]
        new_obj[i] = RELATIONS[f.relation]["value"](vrng)
        ws.wiki.correct(fact_claim[f.fact_id], object=new_obj[i],
                        statement=f.statement.replace(f.obj, new_obj[i]), reason="benchmark correction")
    for i in del_idx:
        ws.wiki.retract(fact_claim[facts[i].fact_id], reason="benchmark deletion")
    t_sync = time.perf_counter()
    sync_report = ws.sync(repair=True, progress=lambda r: progress(f"repair {r}"))
    sync_seconds = time.perf_counter() - t_sync

    def items(idx, obj_of):
        return [(facts[i].train_questions[0], obj_of(i)) for i in idx]

    upd = {
        "edited_new_value": {m: _eval_mode(ws, m, items(edit_idx, lambda i: new_obj[i]), cfg)["accuracy"]
                             for m in ("wiki", "engram", "hybrid")},
        "edited_old_value_still_produced": {
            m: _eval_mode(ws, m, items(edit_idx, lambda i: facts[i].obj), cfg)["accuracy"]
            for m in ("engram", "hybrid")},
        "deleted_still_produced": {m: _eval_mode(ws, m, items(del_idx, lambda i: facts[i].obj), cfg)["accuracy"]
                                   for m in ("engram", "hybrid")},
        "untouched_retained": {m: _eval_mode(ws, m, items(keep_idx, lambda i: facts[i].obj), cfg)["accuracy"]
                               for m in ("engram", "hybrid")},
    }
    recons = ws.consolidate()  # the policy now picks up the edited claims (their new versions)
    upd["after_reconsolidation_edited_new_value_engram"] = _eval_mode(
        ws, "engram", items(edit_idx, lambda i: new_obj[i]), cfg)["accuracy"]
    upd["after_reconsolidation_untouched_engram"] = _eval_mode(
        ws, "engram", items(keep_idx, lambda i: facts[i].obj), cfg)["accuracy"]
    results["update"] = upd

    env = {"python": platform.python_version(), "torch": torch.__version__, "platform": platform.platform(),
           "device": str(lm.device), "model": cfg.model, "backbone_params": lm.num_backbone_params(),
           "engram_spec": lm.spec.to_dict(), "base_table_rows": list(lm.layout.num_embeddings)}
    if lm.device.type == "cuda":
        env["gpu"] = torch.cuda.get_device_name(0)
    full = {
        "config": asdict(cfg), "environment": env, "ingest": {"claims": len(claims), "added": len(report.added)},
        "adapt": {"seconds": adapt_seconds, **{k: adapt_info.get(k) for k in
                                               ("eval_ppl_no_engram", "eval_ppl_engram", "tokens")}},
        "consolidation": {"seconds": cons["training"]["seconds"], "tokens": cons["training"]["tokens"],
                          "epochs": cons["training"]["epochs_run"], "peak_memory_bytes": meter.peak,
                          "verification": cons["verification"], "delta_rows": cons["delta_rows"],
                          "delta_file_bytes": ws.delta_path.stat().st_size,
                          "backbone_and_fusion_unchanged": frozen_ok},
        "results": results,
        "sync": {"seconds": sync_seconds, "forgotten": len(sync_report["forgotten"]),
                 "repaired": len(sync_report["repaired"]), "verification": sync_report.get("verification")},
        "reconsolidation": {"selected": recons.get("selected", 0),
                            "seconds": recons.get("training", {}).get("seconds")},
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(full, indent=2, default=str))
    (out / "results.md").write_text(render_markdown(full))
    return full


def _mb(b) -> str:
    return f"{b / 2**20:.0f} MiB" if b else "n/a"


def render_markdown(r: dict) -> str:
    env, cons, upd = r["environment"], r["consolidation"], r["results"]["update"]
    lines = [
        f"## WikiEngram benchmark: {r['config']['facts']} fictitious facts, {env['model']} on `{env['device']}`",
        "",
        "| Memory | Trained question | Held-out paraphrase | Prompt tokens / query | Median latency (TTFT / total) | Peak memory (inference) |",
        "|---|---|---|---|---|---|",
    ]
    names = {"none": "Backbone only (no memory)", "wiki": "LLMWiki only", "engram": "Engram only",
             "hybrid": "LLMWiki + Engram"}
    for m, name in names.items():
        x = r["results"]["modes"][m]
        lat = x["latency"]
        lines.append(f"| {name} | {x['trained_question']['accuracy']:.1%} | {x['heldout_paraphrase']['accuracy']:.1%} | "
                     f"{x['heldout_paraphrase']['avg_prompt_tokens']:.0f} | {lat['median_ttft_s'] * 1000:.0f} ms / "
                     f"{lat['median_total_s'] * 1000:.0f} ms | {_mb(x['heldout_paraphrase']['peak_memory_bytes'])} |")
    lines += [
        "",
        f"Consolidation: {cons['seconds']:.0f} s, {cons['tokens']} tokens, {cons['epochs']} epochs, "
        f"peak memory {_mb(cons['peak_memory_bytes'])}; personal delta {cons['delta_rows']} rows "
        f"({cons['delta_file_bytes'] / 1e6:.1f} MB file); backbone and Engram fusion parameters unchanged: "
        f"**{cons['backbone_and_fusion_unchanged']}**.",
        "",
        "**After editing / deleting facts in the wiki and syncing memory**",
        "",
        "| Check | LLMWiki only | Engram only | LLMWiki + Engram |",
        "|---|---|---|---|",
        f"| Edited facts: new value answered | {upd['edited_new_value']['wiki']:.1%} | "
        f"{upd['edited_new_value']['engram']:.1%} | {upd['edited_new_value']['hybrid']:.1%} |",
        f"| Edited facts: old value still produced (lower is better) | - | "
        f"{upd['edited_old_value_still_produced']['engram']:.1%} | {upd['edited_old_value_still_produced']['hybrid']:.1%} |",
        f"| Deleted facts still produced (lower is better) | - | {upd['deleted_still_produced']['engram']:.1%} | "
        f"{upd['deleted_still_produced']['hybrid']:.1%} |",
        f"| Untouched facts retained | - | {upd['untouched_retained']['engram']:.1%} | "
        f"{upd['untouched_retained']['hybrid']:.1%} |",
        "",
        f"After re-consolidating the edited claims, Engram alone answers the new value for "
        f"{upd['after_reconsolidation_edited_new_value_engram']:.1%} of edited facts and keeps "
        f"{upd['after_reconsolidation_untouched_engram']:.1%} of untouched ones. Sync took {r['sync']['seconds']:.0f} s.",
        "",
        f"Environment: torch {env['torch']}, {env['platform']}, Python {env['python']}.",
    ]
    return "\n".join(lines) + "\n"
