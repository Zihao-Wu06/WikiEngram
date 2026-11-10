"""Reproducible experiments for the README.

``wikiengram experiment agent``: long-term memory for an AI agent. Fictitious preferences, decisions, task
lessons and project facts are written through the agent API (the same tools agents call over MCP) in three
separate sessions. The experiment then measures:

* retrieval for any model (recall@1/@5 of the right memory for unseen phrasings);
* prompt tokens: all memories in the prompt, the recalled context, or nothing (Engram);
* answers per kind of memory: base model, wiki only, Engram only, wiki + Engram;
* updates: superseded and forgotten memories, before and after re-consolidation.

``wikiengram experiment training``: what the training method guarantees. On the same facts and base Engram:

* default training: accuracy, trainable parameters, time and memory, general-text perplexity with and without
  the personal delta (locality), and bit-exact restoration after deleting every consolidated fact;
* ablation: no generic n-gram blocking (locality);
* continual learning: facts consolidated in five rounds, with and without rehearsal of overlapping memories.

All values (names, products, numbers) are made up, so the base model cannot know them: what is answered was
learned. Results go to ``<out>/results.json`` and ``<out>/results.md``.
"""

from __future__ import annotations

import json
import random
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from ..agent.memory import Memory
from ..config import WorkspaceConfig
from ..fusion.engine import prepare
from ..runtime import WikiEngram
from ..utils.device import PeakMemoryMeter, synchronize
from .runner import BenchConfig, _BenchExtractor, _document
from .synthetic import _name, _word, answer_matches, make_facts

SCOPE = "atlas"
MODES = ("none", "wiki", "engram", "hybrid")


# ------------------------------------------------------------------------------------------- workspace
def _workspace(cfg: BenchConfig, path: Path, progress) -> WikiEngram:
    """A workspace like the benchmark's (selection policy off, so every memory is consolidated)."""
    if path.exists():
        shutil.rmtree(path)
    wcfg = WorkspaceConfig(name="experiment")
    b = wcfg.backend
    b.model, b.device, b.kind, b.placement = cfg.model, cfg.device, cfg.kind, cfg.placement
    b.expert_cache_gb, b.prompt_format = cfg.expert_cache_gb, cfg.prompt_format
    wcfg.adapt.steps = cfg.adapt_steps
    c = wcfg.consolidation
    c.min_score, c.min_confidence, c.stability_hours, c.max_claims_per_run = 0.0, 0.0, 0.0, 100_000
    c.epochs, c.lr, c.batch_size = cfg.epochs, cfg.lr, cfg.batch_size
    c.questions_per_claim, c.include_statement = 2, True
    wcfg.fusion.top_k, wcfg.fusion.max_new_tokens = cfg.top_k, cfg.max_new_tokens
    ws = WikiEngram.init(path, wcfg)
    if cfg.kind != "hf-native":
        cached = Path(cfg.base_dir) if cfg.base_dir else path.parent / "base_cache"
        if (cached / "engram.json").exists():
            shutil.copytree(cached, ws.base_dir, dirs_exist_ok=True)
        else:
            ws.adapt(steps=cfg.adapt_steps, progress=lambda r: progress(f"adapt {r}"))
            shutil.copytree(ws.base_dir, cached, dirs_exist_ok=True)
    return ws


def _answers(ws: WikiEngram, mode: str, items: list[tuple[str, str]], cfg: BenchConfig) -> dict:
    """Per-item correctness and token counts for ``items`` = (question, expected) in one memory mode."""
    lm = ws.lm
    prepared = [prepare(ws, q, mode) for q, _ in items]
    correct = [False] * len(items)
    ptoks = [0] * len(items)
    for flag in (False, True):
        idx = [i for i, p in enumerate(prepared) if p.use_delta == flag]
        for j in range(0, len(idx), cfg.gen_batch):
            chunk = idx[j : j + cfg.gen_batch]
            res = lm.generate([prepared[i].prompt for i in chunk], max_new_tokens=cfg.max_new_tokens,
                              use_engram=prepared[chunk[0]].use_engram, use_delta=flag,
                              delta_start=[prepared[i].context_tokens for i in chunk])
            for i, text, n in zip(chunk, res.texts, res.prompt_tokens):
                correct[i], ptoks[i] = answer_matches(text, items[i][1]), n
    return {"correct": correct, "prompt_tokens": ptoks}


def _acc(flags: list[bool]) -> float:
    return round(sum(flags) / len(flags), 4) if flags else 0.0


# ------------------------------------------------------------------------------------ agent memories
@dataclass
class AgentMemory:
    kind: str  # preference | decision | experience | fact
    key: str  # topic, task or subject
    value: str
    trained: str  # first stored question (consolidated)
    heldout: str  # never trained or used for verification
    extra: str = ""  # rationale / owner phrasing
    memory_id: str = ""
    session: int = 0


_PREF = [("code formatter", "fmt"), ("test runner", "test"), ("documentation tool", "doc"), ("plotting library", "plot"),
         ("package manager", "pkg"), ("logging library", "log"), ("task tracker", " Board"), ("terminal theme", " Dark")]
_DECISION = [("database", "DB"), ("message queue", "MQ"), ("cache", "Cache"), ("scheduler", "Sched"),
             ("storage backend", "Store"), ("auth provider", "Auth"), ("search engine", "Search"), ("CI system", "CI")]
_TASK = (["deploying", "migrating", "upgrading", "benchmarking", "debugging"],
         ["gateway", "cluster", "pipeline", "worker pool", "index"])
_LESSON = (["drain", "snapshot", "pin", "warm up", "freeze"], ["queue", "cache", "index", "volume", "config"])
_FACT = ["service", "dashboard", "repository", "dataset", "runbook"]
_WHY = ["it was the only option with offline mode", "the team already knows it", "it halves the hosting bill",
        "it passed the load test", "it has the licence we need"]


def make_agent_memories(per_kind: int, seed: int = 0) -> list[AgentMemory]:
    rng = random.Random(seed)
    used: set[str] = set()

    def word() -> str:
        w = _word(rng, 2).capitalize()
        while w in used:
            w = _word(rng, 2).capitalize()
        used.add(w)
        return w

    out = []
    for i in range(per_kind):
        topic, suffix = _PREF[i % len(_PREF)]
        t = f"{topic} for the {word()} module"
        out.append(AgentMemory("preference", t, f"{word()}{suffix}", f"What is the user's preference for {t}?",
                               f"Which {t} should be used for this user?"))
        kind, suffix = _DECISION[i % len(_DECISION)]
        t = f"the {word()} service {kind}"
        out.append(AgentMemory("decision", t, f"{word()}{suffix}", f"What was decided about {t}?",
                               f"Which option was chosen for {t}?", extra=rng.choice(_WHY)))
        task = f"{rng.choice(_TASK[0])} the {word()} {rng.choice(_TASK[1])}"
        lesson = f"{rng.choice(_LESSON[0])} the {word()} {rng.choice(_LESSON[1])} first"
        out.append(AgentMemory("experience", task, lesson, f"What was learned from {task}?",
                               f"What should be kept in mind next time when {task}?"))
        subj = f"the {word()} {_FACT[i % len(_FACT)]}"
        out.append(AgentMemory("fact", subj, _name(rng), f"Who owns {subj}?",
                               f"Which person is responsible for {subj}?"))
    return out


def _write(mem: Memory, m: AgentMemory) -> str:
    if m.kind == "preference":
        r = mem.set_preference(m.key, m.value)
    elif m.kind == "decision":
        r = mem.record_decision(m.key, m.value, m.extra)
    elif m.kind == "experience":
        r = mem.record_experience(m.key, m.value)
    else:
        r = mem.add_fact(m.key, "owner", m.value, f"The owner of {m.key} is {m.value}.",
                         questions=[m.trained, f"Who is the owner of {m.key}?"])
    return r["memory"]["id"]


def run_agent_experiment(cfg: BenchConfig, out_dir: str | Path, per_kind: int = 25, progress=print) -> dict:
    out = Path(out_dir)
    ws_path = out / "workspace"
    rng = random.Random(cfg.seed + 7)
    memories = make_agent_memories(per_kind, cfg.seed)
    ws = _workspace(cfg, ws_path, progress)
    del ws  # sessions below open the workspace themselves, like separate agent runs

    # 1. three agent sessions write their memories through the agent API (= the MCP tools)
    for s in range(3):
        mem = Memory(ws_path, scope=SCOPE)
        for m in memories[s::3]:
            m.memory_id, m.session = _write(mem, m), s + 1
        progress(f"session {s + 1}: wrote {len(memories[s::3])} memories")
        del mem

    # 2. a fourth session: retrieval for any model (no model loaded)
    mem = Memory(ws_path, scope=SCOPE)
    ranks = {}
    for m in memories:
        ids = [x["id"] for x in mem.recall(m.heldout, k=5)["memories"]]
        ranks[m.memory_id] = ids.index(m.memory_id) + 1 if m.memory_id in ids else None
    kinds = sorted({m.kind for m in memories})
    retrieval = {k: {"recall@1": _acc([ranks[m.memory_id] == 1 for m in memories if m.kind == k]),
                     "recall@5": _acc([ranks[m.memory_id] is not None for m in memories if m.kind == k])}
                 for k in kinds}
    retrieval["all"] = {"recall@1": _acc([r == 1 for r in ranks.values()]),
                        "recall@5": _acc([r is not None for r in ranks.values()])}
    brief_hits = [m.value.lower() in mem.context(m.heldout).lower() for m in memories
                  if m.kind in ("decision", "experience")]

    ws = WikiEngram.open(ws_path)
    tok = ws.lm.tokenizer
    every = "Known facts:\n" + "\n".join(f"- {c.statement}" for c in ws.wiki.claims())
    recalled = [len(tok(mem.recall(m.heldout, k=5)["context"])["input_ids"]) for m in memories]
    tokens = {"all_memories_in_prompt": len(tok(every)["input_ids"]),
              "recalled_context_top5": round(sum(recalled) / len(recalled), 1), "engram": 0}
    progress(f"retrieval {retrieval['all']}, tokens {tokens}")

    # 3. consolidate into Engram and answer per kind of memory
    lm = ws.lm
    with PeakMemoryMeter(lm.device) as meter:
        cons = ws.consolidate(progress=lambda r: (meter.poll(), progress(f"consolidate {r}")))
    synchronize(lm.device)
    answers = {}
    for split in ("trained", "heldout"):
        items = [(getattr(m, split), m.value) for m in memories]
        for mode in MODES:
            res = _answers(ws, mode, items, cfg)
            row = {k: _acc([c for c, m in zip(res["correct"], memories) if m.kind == k]) for k in kinds}
            row["all"] = _acc(res["correct"])
            row["prompt_tokens"] = round(sum(res["prompt_tokens"]) / len(items), 1)
            answers[f"{split}/{mode}"] = row
            progress(f"{split:7s} {mode:7s} {row}")

    # 4. the agent changes its mind and forgets things; memory follows
    changeable = [m for m in memories if m.kind in ("preference", "decision")]
    edited = rng.sample(changeable, max(1, len(changeable) // 5))
    rest = [m for m in memories if m not in edited]
    forgotten = rng.sample(rest, max(1, len(memories) // 10))
    kept = [m for m in rest if m not in forgotten]
    new_value = {}
    mem = Memory(ws_path, scope=SCOPE)
    for m in edited:
        new_value[m.memory_id] = f"{_word(rng, 2).capitalize()}{m.value[-3:]}"
        if m.kind == "preference":
            mem.set_preference(m.key, new_value[m.memory_id])
        else:
            mem.record_decision(m.key, new_value[m.memory_id], "revisited after a review")
    for m in forgotten:
        mem.forget(m.memory_id, reason="no longer true")
    t_sync = time.perf_counter()
    sync = ws.sync(repair=True)
    sync_s = time.perf_counter() - t_sync

    def acc(mode, group, value_of):
        return _acc(_answers(ws, mode, [(m.trained, value_of(m)) for m in group], cfg)["correct"])

    update = {
        "edited_new_value": {mo: acc(mo, edited, lambda m: new_value[m.memory_id]) for mo in ("wiki", "engram", "hybrid")},
        "edited_old_value_still_produced": {mo: acc(mo, edited, lambda m: m.value) for mo in ("engram", "hybrid")},
        "forgotten_still_produced": {mo: acc(mo, forgotten, lambda m: m.value) for mo in ("engram", "hybrid")},
        "kept_retained": {mo: acc(mo, kept, lambda m: m.value) for mo in ("engram", "hybrid")},
    }
    recons = ws.consolidate()
    update["after_reconsolidation"] = {"edited_new_value_engram": acc("engram", edited, lambda m: new_value[m.memory_id]),
                                       "kept_engram": acc("engram", kept, lambda m: m.value)}
    progress(f"update {update}")

    result = {
        "config": asdict(cfg) | {"per_kind": per_kind}, "memories": len(memories), "sessions": 3,
        "environment": _env(ws), "retrieval": retrieval, "brief_coverage": _acc(brief_hits), "tokens": tokens,
        "consolidation": {"seconds": cons["training"]["seconds"], "epochs": cons["training"]["epochs_run"],
                          "stop_reason": cons["training"].get("stop_reason"), "peak_memory_bytes": meter.peak,
                          "delta_rows": cons["delta_rows"], "verification": cons["verification"]},
        "answers": answers, "update": update, "sync_seconds": round(sync_s, 1),
        "sync": {"forgotten": len(sync["forgotten"]), "repaired": len(sync["repaired"])},
        "reconsolidation_seconds": recons.get("training", {}).get("seconds"),
        "examples": [asdict(m) for m in memories[:4]],
    }
    _save(out, result, render_agent_markdown(result))
    return result


# ---------------------------------------------------------------------------------------- training
def _ppl(ws: WikiEngram, stream, use_delta: bool) -> float:
    from ..engram.adapt import AdaptConfig, lm_loss

    loss = lm_loss(ws.lm, stream, AdaptConfig(batch_size=8, seq_len=128), use_engram=True, n_batches=25,
                   use_delta=use_delta)
    return round(float(torch.tensor(loss).exp()), 4)


def _facts_workspace(cfg: BenchConfig, path: Path, facts, progress, **consolidation) -> WikiEngram:
    ws = _workspace(cfg, path, progress)
    for k, v in consolidation.items():
        setattr(ws.config.consolidation, k, v)
    ws.extractor = _BenchExtractor({f.subject: f.train_questions + f.heldout_questions[:1] for f in facts})
    return ws


def _ingest(ws: WikiEngram, facts) -> None:
    ws.wiki.ingest_text(_document(facts), title=f"facts {facts[0].fact_id}", kind="document")


def _engram_acc(ws: WikiEngram, facts, cfg: BenchConfig, heldout: bool = False) -> float:
    items = [(f.heldout_questions[-1] if heldout else f.train_questions[0], f.obj) for f in facts]
    return _acc(_answers(ws, "engram", items, cfg)["correct"])


def run_training_experiment(cfg: BenchConfig, out_dir: str | Path, rounds: int = 5, progress=print) -> dict:
    from ..engram.adapt import pack_token_stream
    from ..utils.corpus import load_texts

    out = Path(out_dir)
    facts = make_facts(cfg.facts, seed=cfg.seed)
    runs = {}

    # 1. default training, then delete everything: the model must be bit-for-bit what it was
    ws = _facts_workspace(cfg, out / "ws-default", facts, progress)
    _ingest(ws, facts)
    lm = ws.lm
    stream = pack_token_stream(lm.tokenizer, load_texts("wikitext-2", "validation"), max_tokens=200_000)
    probe = lm.tokenizer([f.train_questions[0] for f in facts[:16]], return_tensors="pt", padding=True)
    with torch.no_grad():
        before = lm.logits(lm.hidden_states(probe["input_ids"], probe["attention_mask"],
                                            lm.build_batch(probe["input_ids"], probe["attention_mask"])))
    ppl_base = _ppl(ws, stream, use_delta=False)
    with PeakMemoryMeter(lm.device) as meter:
        cons = ws.consolidate(progress=lambda r: (meter.poll(), progress(f"default {r}")))
    rows = lm.delta.num_rows()
    trainable = rows * lm.variant.row_dim
    runs["default"] = {
        "engram_trained": _engram_acc(ws, facts, cfg), "engram_heldout": _engram_acc(ws, facts, cfg, heldout=True),
        "ppl_without_delta": ppl_base, "ppl_with_delta": _ppl(ws, stream, use_delta=True),
        "delta_rows": rows, "rows_per_fact": round(rows / len(facts), 1), "trainable_parameters": trainable,
        "backbone_parameters": lm.num_backbone_params(),
        "trainable_share": trainable / max(1, lm.num_backbone_params()),
        "seconds": cons["training"]["seconds"], "epochs": cons["training"]["epochs_run"],
        "stop_reason": cons["training"].get("stop_reason"), "peak_memory_bytes": meter.peak,
        "frozen_checksums_unchanged": cons["training"]["backbone_checksum"] == lm.backbone_checksum(),
    }
    for c in ws.wiki.claims():
        ws.wiki.retract(c.id, reason="experiment: delete everything")
    ws.sync(repair=True)
    with torch.no_grad():
        after = lm.logits(lm.hidden_states(probe["input_ids"], probe["attention_mask"],
                                           lm.build_batch(probe["input_ids"], probe["attention_mask"])))
    runs["default"]["rows_after_deleting_all"] = lm.delta.num_rows()
    runs["default"]["bit_exact_after_deleting_all"] = bool(torch.equal(before, after))
    progress(f"default {runs['default']}")

    # 2. ablation: no generic n-gram blocking
    ws2 = _facts_workspace(cfg, out / "ws-noblock", facts, progress, block_min_count=0, block_shared_claims=0,
                           block_prompt_template=False)
    ws2._lm = lm  # same frozen model; the workspace brings its own (empty) delta
    lm.attach_delta(ws2._load_or_new_delta(lm))
    _ingest(ws2, facts)
    cons2 = ws2.consolidate()
    runs["no_blocking"] = {"engram_trained": _engram_acc(ws2, facts, cfg),
                           "engram_heldout": _engram_acc(ws2, facts, cfg, heldout=True),
                           "ppl_with_delta": _ppl(ws2, stream, use_delta=True), "delta_rows": lm.delta.num_rows(),
                           "seconds": cons2["training"]["seconds"]}
    progress(f"no_blocking {runs['no_blocking']}")

    # 3. continual learning: rounds of new facts, with and without rehearsal
    per = len(facts) // rounds
    for name, replay in (("continual_rehearsal", True), ("continual_no_rehearsal", False)):
        wsc = _facts_workspace(cfg, out / f"ws-{name}", facts, progress, replay_overlapping=replay)
        wsc._lm = lm
        lm.attach_delta(wsc._load_or_new_delta(lm))
        matrix = []
        for r in range(rounds):
            _ingest(wsc, facts[r * per : (r + 1) * per])
            wsc.consolidate()
            matrix.append([_engram_acc(wsc, facts[j * per : (j + 1) * per], cfg) for j in range(r + 1)])
            progress(f"{name} round {r + 1}: {matrix[-1]}")
        runs[name] = {"accuracy_by_round": matrix, "first_round_final": matrix[-1][0],
                      "mean_final": round(sum(matrix[-1]) / len(matrix[-1]), 4)}

    result = {"config": asdict(cfg) | {"rounds": rounds}, "facts": len(facts), "environment": _env(ws),
              "runs": runs}
    _save(out, result, render_training_markdown(result))
    return result


# ------------------------------------------------------------------------------------------ output
def _env(ws: WikiEngram) -> dict:
    import platform

    lm = ws.lm
    env = {"model": ws.config.backend.model, "device": str(lm.device), "torch": torch.__version__,
           "python": platform.python_version(), "prompt_format": lm.prompt_format,
           "engram_variant": lm.variant.name}
    if lm.device.type == "cuda":
        env["gpu"] = torch.cuda.get_device_name(0)
    return env


def _save(out: Path, result: dict, md: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(result, indent=2, default=str))
    (out / "results.md").write_text(md)


def _pct(x) -> str:
    return "-" if x is None else f"{100 * x:.1f}%"


def render_agent_markdown(r: dict) -> str:
    env, kinds = r["environment"], [k for k in r["retrieval"] if k != "all"]
    lines = [f"## Agent memory: {r['memories']} memories written in {r['sessions']} sessions, "
             f"{Path(env['model']).name} on `{env['device']}`", "",
             "**Retrieval for any model** (unseen phrasings; `memory_recall`, no model involved)", "",
             "| | " + " | ".join(kinds) + " | all |", "|---" * (len(kinds) + 2) + "|"]
    for metric in ("recall@1", "recall@5"):
        lines.append(f"| {metric} | " + " | ".join(_pct(r["retrieval"][k][metric]) for k in kinds)
                     + f" | {_pct(r['retrieval']['all'][metric])} |")
    t = r["tokens"]
    lines += ["", f"Task briefs (`memory_context`) contain the relevant decision or lesson for "
              f"{_pct(r['brief_coverage'])} of unseen questions.", "",
              f"Prompt tokens for memory: all memories in the prompt {t['all_memories_in_prompt']}, recalled "
              f"context {t['recalled_context_top5']}, Engram {t['engram']}.", "",
              "**Answers after consolidation** (trained phrasing / unseen phrasing)", "",
              "| Memory | " + " | ".join(kinds) + " | all | prompt tokens |", "|---" * (len(kinds) + 3) + "|"]
    names = {"none": "Base model", "wiki": "LLMWiki only", "engram": "Engram only", "hybrid": "LLMWiki + Engram"}
    for mode in MODES:
        a, b = r["answers"][f"trained/{mode}"], r["answers"][f"heldout/{mode}"]
        lines.append(f"| {names[mode]} | " + " | ".join(f"{_pct(a[k])} / {_pct(b[k])}" for k in kinds)
                     + f" | {_pct(a['all'])} / {_pct(b['all'])} | {a['prompt_tokens']} |")
    c, u = r["consolidation"], r["update"]
    lines += ["", f"Consolidation: {c['seconds']:.0f} s, {c['epochs']} epochs (stopped: {c['stop_reason']}), "
              f"{c['delta_rows']} delta rows.", "", "**The agent changes its mind and forgets things**", "",
              "| Check | LLMWiki only | Engram only | LLMWiki + Engram |", "|---|---|---|---|",
              f"| Changed preferences / decisions: new value | {_pct(u['edited_new_value']['wiki'])} | "
              f"{_pct(u['edited_new_value']['engram'])} | {_pct(u['edited_new_value']['hybrid'])} |",
              f"| Changed: old value still produced (lower is better) | - | "
              f"{_pct(u['edited_old_value_still_produced']['engram'])} | "
              f"{_pct(u['edited_old_value_still_produced']['hybrid'])} |",
              f"| Forgotten memories still produced (lower is better) | - | "
              f"{_pct(u['forgotten_still_produced']['engram'])} | {_pct(u['forgotten_still_produced']['hybrid'])} |",
              f"| Other memories retained | - | {_pct(u['kept_retained']['engram'])} | "
              f"{_pct(u['kept_retained']['hybrid'])} |", "",
              f"After re-consolidation Engram alone gives the new value for "
              f"{_pct(u['after_reconsolidation']['edited_new_value_engram'])} of changed memories and keeps "
              f"{_pct(u['after_reconsolidation']['kept_engram'])} of the others. Sync took {r['sync_seconds']} s.",
              "", f"Environment: torch {env['torch']}, prompt format `{env['prompt_format']}`"
              + (f", {env['gpu']}" if env.get("gpu") else "") + "."]
    return "\n".join(lines) + "\n"


def render_training_markdown(r: dict) -> str:
    env, d, nb = r["environment"], r["runs"]["default"], r["runs"]["no_blocking"]
    cr, cn = r["runs"]["continual_rehearsal"], r["runs"]["continual_no_rehearsal"]
    gib = d["peak_memory_bytes"] / 2**30
    lines = [f"## Training method: {r['facts']} facts, {Path(env['model']).name} on `{env['device']}`", "",
             "| | Default | No generic n-gram blocking |", "|---|---|---|",
             f"| Engram only, trained / unseen phrasing | {_pct(d['engram_trained'])} / {_pct(d['engram_heldout'])} | "
             f"{_pct(nb['engram_trained'])} / {_pct(nb['engram_heldout'])} |",
             f"| General-text perplexity (wikitext-2), {d['ppl_without_delta']} without the delta | "
             f"{d['ppl_with_delta']} ({100 * (d['ppl_with_delta'] / d['ppl_without_delta'] - 1):+.2f}%) | "
             f"{nb['ppl_with_delta']} ({100 * (nb['ppl_with_delta'] / d['ppl_without_delta'] - 1):+.2f}%) |",
             f"| Delta rows | {d['delta_rows']} ({d['rows_per_fact']} per fact) | {nb['delta_rows']} |",
             f"| Training time | {d['seconds']:.0f} s ({d['epochs']} epochs, stopped: {d['stop_reason']}) | "
             f"{nb['seconds']:.0f} s |", "",
             f"Trainable parameters: {d['trainable_parameters']:,} ({100 * d['trainable_share']:.3f}% of the "
             f"{d['backbone_parameters']:,} frozen backbone parameters); peak memory {gib:.1f} GiB; frozen weights "
             f"unchanged: **{d['frozen_checksums_unchanged']}**.", "",
             f"After deleting every consolidated fact the delta has {d['rows_after_deleting_all']} rows and the "
             f"model's logits are bit-identical to before training: **{d['bit_exact_after_deleting_all']}**.", "",
             f"**Continual learning** ({r['config']['rounds']} rounds of new facts; Engram-only accuracy on "
             "each round's facts after the last round)", "",
             "| | " + " | ".join(f"round {i + 1}" for i in range(r["config"]["rounds"])) + " | mean |",
             "|---" * (r["config"]["rounds"] + 2) + "|"]
    for name, run in (("With rehearsal", cr), ("Without rehearsal", cn)):
        lines.append(f"| {name} | " + " | ".join(_pct(x) for x in run["accuracy_by_round"][-1])
                     + f" | {_pct(run['mean_final'])} |")
    lines += ["", f"Environment: torch {env['torch']}" + (f", {env['gpu']}" if env.get("gpu") else "") + "."]
    return "\n".join(lines) + "\n"
