"""The WikiEngram workspace: explicit memory (LLMWiki) + neural memory (Engram delta) + a frozen model.

Layout of a workspace directory::

    wikiengram.toml          configuration
    wiki/                    LLMWiki (raw sources, markdown pages, index.md, log.md, claim store)
    engram/base/             frozen base Engram for the backbone (spec, fusion params, tables)
    engram/delta.safetensors personal sparse delta (the neural memory)
    engram/ledger.json       which delta rows belong to which claim
    engram/state.json        consolidation state (synced event seq, verified claims, versions)
    engram/history/          previous delta + ledger versions (rollback)
    runs/                    JSON reports of every consolidation / sync / benchmark
"""

from __future__ import annotations

import json
import shutil
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .config import WorkspaceConfig
from .consolidation.ledger import RowLedger
from .consolidation.policy import Selection, select_claims
from .consolidation.samples import Probe, build_claim_samples, claim_questions
from .consolidation.trainer import DeltaTrainConfig, TrainingSample, train_delta
from .consolidation.verify import VerificationReport, run_probes
from .engram.delta import PersonalDelta
from .engram.layout import EngramSpec
from .wiki import EngramState, LLMWiki, OpenAICompatExtractor, RuleExtractor


class NeedsAdaptation(RuntimeError):
    """The backbone has no base Engram yet: run ``wikiengram adapt`` first."""


def make_extractor(cfg):
    if cfg.kind == "openai":
        return OpenAICompatExtractor(model=cfg.model, base_url=cfg.base_url or None)
    return RuleExtractor()


@dataclass
class ConsolidationPlan:
    selected: list[Selection]
    rejected: list[Selection]
    n_samples: int = 0
    est_tokens: int = 0
    est_seconds: float | None = None
    throughput_source: str = ""

    def to_dict(self) -> dict:
        return {"selected": [asdict(s) for s in self.selected], "rejected": [asdict(s) for s in self.rejected],
                "n_samples": self.n_samples, "est_tokens": self.est_tokens, "est_seconds": self.est_seconds,
                "throughput_source": self.throughput_source}


@dataclass
class EngramStateFile:
    last_event_seq: int = 0
    delta_version: int = 0
    tokens_per_second: float | None = None  # measured consolidation throughput, used by --plan
    claims: dict[str, dict] = field(default_factory=dict)  # claim_id -> {version, verified, questions, probes}
    pending_repair: dict[str, list[int]] = field(default_factory=dict)  # layer_id -> masked shared rows
    history: list[dict] = field(default_factory=list)

    @classmethod
    def load(cls, path: Path) -> EngramStateFile:
        if not path.exists():
            return cls()
        return cls(**json.loads(path.read_text()))

    def save(self, path: Path) -> None:
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(asdict(self), indent=1))
        tmp.replace(path)


class WikiEngram:
    CONFIG_FILE = "wikiengram.toml"

    def __init__(self, root: str | Path, config: WorkspaceConfig, lm=None):
        self.root = Path(root)
        self.config = config
        self.extractor = make_extractor(config.extractor)
        self.wiki = LLMWiki(self.root / "wiki", extractor=self.extractor)
        self.state = EngramStateFile.load(self.state_path)
        self._lm = lm
        self._ledger: RowLedger | None = None

    # ------------------------------------------------------------------------------ construction
    @classmethod
    def init(cls, root: str | Path, config: WorkspaceConfig | None = None) -> WikiEngram:
        root = Path(root)
        root.mkdir(parents=True, exist_ok=True)
        cfg_path = root / cls.CONFIG_FILE
        if config is None:
            config = WorkspaceConfig.load(cfg_path) if cfg_path.exists() else WorkspaceConfig()
        config.save(cfg_path)
        for d in ("engram/history", "runs", "cache"):
            (root / d).mkdir(parents=True, exist_ok=True)
        LLMWiki.init(root / "wiki", extractor=make_extractor(config.extractor))
        return cls(root, config)

    @classmethod
    def open(cls, root: str | Path, lm=None) -> WikiEngram:
        root = Path(root)
        cfg_path = root / cls.CONFIG_FILE
        if not cfg_path.exists():
            raise FileNotFoundError(f"{root} is not a WikiEngram workspace (no {cls.CONFIG_FILE}); run `wikiengram init`")
        return cls(root, WorkspaceConfig.load(cfg_path), lm=lm)

    # ------------------------------------------------------------------------------ paths
    @property
    def base_dir(self) -> Path:
        return self.root / "engram" / "base"

    @property
    def delta_path(self) -> Path:
        return self.root / "engram" / "delta.safetensors"

    @property
    def ledger_path(self) -> Path:
        return self.root / "engram" / "ledger.json"

    @property
    def state_path(self) -> Path:
        return self.root / "engram" / "state.json"

    @property
    def cache_dir(self) -> Path:
        return self.root / "cache"

    def has_base(self) -> bool:
        return (self.base_dir / "engram.json").exists()

    # ------------------------------------------------------------------------------ model
    def spec_for(self, hidden_size: int, pad_token_id: int) -> EngramSpec:
        e = self.config.engram
        return EngramSpec(hidden_size=hidden_size, layer_ids=tuple(e.layer_ids), max_ngram_size=e.max_ngram_size,
                          n_heads=e.n_heads, head_dim=e.head_dim, vocab_size=e.vocab_size,
                          pad_token_id=pad_token_id, norm_eps=e.norm_eps)

    @property
    def lm(self):
        if self._lm is None:
            from .backends.hf import EngramCausalLM

            if not self.has_base():
                raise NeedsAdaptation(
                    f"no base Engram in {self.base_dir}; run `wikiengram adapt` once for {self.config.backend.model}")
            b = self.config.backend
            self._lm = EngramCausalLM.from_pretrained(b.model, base_path=self.base_dir, device=b.device,
                                                      dtype=b.dtype, cache_dir=self.cache_dir)
        if self._lm.delta is None:
            self._lm.attach_delta(self._load_or_new_delta(self._lm))
        return self._lm

    def base_id(self) -> str:
        """Identity of the frozen base Engram (its fusion checksum); a delta is only valid on that base."""
        meta = self.base_dir / "engram.json"
        if meta.exists():
            return json.loads(meta.read_text()).get("fusion_checksum", "")
        return self._lm.fusion_checksum() if self._lm is not None else ""

    def _load_or_new_delta(self, lm) -> PersonalDelta:
        if self.delta_path.exists():
            delta = PersonalDelta.load(self.delta_path, device=lm.device, expect_fingerprint=lm.layout.fingerprint())
            trained_on = delta.meta.get("base_id")
            if trained_on and trained_on != self.base_id():
                raise RuntimeError(
                    "the personal delta was trained against a different base Engram; restore that base or "
                    "move the delta away and re-consolidate from the wiki")
            return delta
        delta = lm.new_delta()
        delta.meta["base_id"] = self.base_id()
        return delta

    @property
    def ledger(self) -> RowLedger:
        if self._ledger is None:
            self._ledger = RowLedger.load(self.ledger_path)
        return self._ledger

    def adapt(self, steps: int | None = None, corpus: str | None = None, force: bool = False, progress=None) -> dict:
        """Stage 1: train the base Engram for the configured backbone (backbone frozen)."""
        from .backends.hf import EngramCausalLM
        from .engram.adapt import AdaptConfig, adapt_base_engram, pack_token_stream
        from .utils.corpus import load_texts

        if self.delta_path.exists() and not force:
            raise RuntimeError("a personal delta exists; it is tied to the current base Engram. Re-run with "
                               "force=True (the delta is archived and claims are re-consolidated later).")
        a, b = self.config.adapt, self.config.backend
        from transformers import AutoConfig, AutoTokenizer

        tok = AutoTokenizer.from_pretrained(b.model)
        pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
        spec = self.spec_for(AutoConfig.from_pretrained(b.model).hidden_size, pad)
        lm = EngramCausalLM.from_pretrained(b.model, spec=spec, device=b.device, dtype=b.dtype,
                                            cache_dir=self.cache_dir)
        train = pack_token_stream(lm.tokenizer, load_texts(corpus or a.corpus, "train"), max_tokens=a.max_tokens)
        val = pack_token_stream(lm.tokenizer, load_texts(corpus or a.corpus, "validation"), max_tokens=200_000)
        cfg = AdaptConfig(steps=steps or a.steps, batch_size=a.batch_size, seq_len=a.seq_len,
                          table_lr=a.table_lr, fusion_lr=a.fusion_lr)
        report = adapt_base_engram(lm, train, val, cfg, progress=progress)
        if self.base_dir.exists():
            shutil.rmtree(self.base_dir)
        lm.save_base(self.base_dir, info=report)
        if self.delta_path.exists():
            self._archive_delta("pre-adapt")
            self.delta_path.unlink()
            self.ledger_path.unlink(missing_ok=True)
            for c in self.wiki.claims(status=None):
                self.wiki.set_engram_state(c.id, EngramState.NONE, None)
        # memory versions trained on the previous base are not valid any more
        self.state.claims.clear()
        self.state.pending_repair.clear()
        self.state.tokens_per_second = None
        self._drop_history()
        self.state.save(self.state_path)
        self._lm, self._ledger = None, None
        self._write_run("adapt", report)
        return report

    # ------------------------------------------------------------------------------ consolidation
    def plan(self, claim_ids: list[str] | None = None) -> ConsolidationPlan:
        c = self.config.consolidation
        claims = self.wiki.claims()
        if claim_ids is not None:
            wanted = set(claim_ids)
            claims = [x for x in claims if x.id in wanted]
            cfg = WorkspaceConfig().consolidation
            cfg.min_score, cfg.min_confidence, cfg.stability_hours = 0.0, 0.0, 0.0
            cfg.max_claims_per_run = len(claims) or 1
            selected, rejected = select_claims(claims, cfg)
        else:
            selected, rejected = select_claims(claims, c)
        per_claim = c.questions_per_claim + (1 if c.include_statement else 0)
        n_samples = per_claim * len(selected)
        est_tokens = n_samples * 24 * c.epochs  # ~24 tokens per sample (prompt + answer)
        tps = self.state.tokens_per_second
        plan = ConsolidationPlan(selected, rejected, n_samples, est_tokens,
                                 est_tokens / tps if tps else None,
                                 "measured on this workspace" if tps else "no measurement yet")
        return plan

    def consolidate(self, claim_ids: list[str] | None = None, verify: bool = True, progress=None) -> dict:
        """Train the selected claims into the personal delta, verify recall, update the wiki bookkeeping."""
        self.sync(repair=True)  # never train on top of stale memory
        plan = self.plan(claim_ids)
        if not plan.selected:
            report = {"selected": 0, "message": "nothing to consolidate", "plan": plan.to_dict()}
            self._write_run("consolidate", report)
            return report
        lm, c = self.lm, self.config.consolidation
        samples: list[TrainingSample] = []
        probes: list[Probe] = []
        claims = {x.id: x for x in self.wiki.claims()}
        for sel in plan.selected:
            claim = claims[sel.claim_id]
            if claim.id in self.ledger:  # an older version is still in memory: forget it first
                self._forget_rows(claim.id, repair=False)
            qs = claim_questions(claim, self.extractor, n=c.questions_per_claim + 1)
            s, p = build_claim_samples(claim, qs, c.questions_per_claim, c.include_statement)
            samples += s
            probes += p
            self.state.claims[claim.id] = {"version": claim.version, "object": claim.object, "questions": qs,
                                           "verified": False, "consolidated_at": time.time()}
        blocked, block_stats = self.generic_rows(samples)
        # protect old memories: claims that genuinely share rows with the new samples are rehearsed (trained
        # again together with them); rows shared only through stray hash collisions are frozen instead
        overlaps = self._overlapping_claims(samples, blocked, exclude={s.claim_id for s in plan.selected})
        replayed, frozen = [], {}
        for overlap, cid, rows in overlaps:
            if c.replay_overlapping and overlap >= c.min_replay_overlap and len(replayed) < c.max_replay_claims:
                replayed.append(cid)
            else:
                for lid, r in rows.items():
                    frozen.setdefault(lid, []).append(r)
        if replayed:
            samples += self.samples_for(replayed)
            probes += self.probes_for(replayed)
            blocked, block_stats = self.generic_rows(samples)
        if frozen:
            import numpy as np

            for lid, parts in frozen.items():
                blocked[lid] = np.union1d(blocked.get(lid, np.empty(0, np.int64)), np.concatenate(parts))
            block_stats["frozen_rows_of_other_claims"] = int(sum(np.concatenate(v).size for v in frozen.values()))
        replay = self._replay_stream() if c.locality_weight > 0 else None
        cfg = DeltaTrainConfig(epochs=c.epochs, batch_size=c.batch_size, lr=c.lr, locality_weight=c.locality_weight)
        train = train_delta(lm, lm.delta, samples, cfg, ledger=self.ledger, blocked_rows=blocked,
                            replay_stream=replay, progress=progress)
        self.state.tokens_per_second = train.tokens / max(train.seconds, 1e-9)
        verification = run_probes(lm, probes) if verify else VerificationReport()
        per_claim = verification.per_claim()
        for sel in plan.selected:
            self.state.claims[sel.claim_id]["verified"] = bool(per_claim.get(sel.claim_id, False))
        for cid in replayed:
            if cid in self.state.claims:
                self.state.claims[cid]["verified"] = bool(per_claim.get(cid, False))
        # persist memory first: after a crash the wiki then still says "not consolidated" and the next run
        # forgets and retrains those claims, instead of trusting memory that was never saved
        self._save_memory("consolidate")
        for sel in plan.selected:
            self.wiki.set_engram_state(sel.claim_id, EngramState.CONSOLIDATED, sel.version)
        report = {"selected": len(plan.selected), "plan": plan.to_dict(), "training": train.to_dict(),
                  "blocked": block_stats, "replayed_claims": len(replayed),
                  "verification": verification.summary(),
                  "verified_claims": sum(bool(per_claim.get(x.claim_id)) for x in plan.selected),
                  "verified_replayed": sum(bool(per_claim.get(cid)) for cid in replayed),
                  "delta_rows": lm.delta.num_rows(), "delta_version": self.state.delta_version}
        self._write_run("consolidate", report)
        return report

    # ------------------------------------------------------------------------------ sync
    def pending_events(self):
        return self.wiki.events(since_seq=self.state.last_event_seq)

    def sync(self, repair: bool = True, verify: bool = True, progress=None) -> dict:
        from .sync.engine import run_sync

        return run_sync(self, repair=repair, verify=verify, progress=progress)

    def _forget_rows(self, claim_id: str, repair: bool) -> tuple[dict, set[str]]:
        """Remove a claim's rows from the delta. Exclusive rows are dropped (exact restore). Shared rows are
        dropped too and returned (with the claims that share them) so the caller can repair them."""
        delta = self.lm.delta
        exclusive, shared, affected = self.ledger.split_for_removal(claim_id)
        import torch

        for lid, rows in exclusive.items():
            delta.layers[lid].remove(torch.from_numpy(rows))
        for lid, rows in shared.items():
            if repair:
                delta.layers[lid].remove(torch.from_numpy(rows))
            else:  # masked until repaired: reads as the base table immediately
                delta.layers[lid].set_active(torch.from_numpy(rows), False)
                pend = set(self.state.pending_repair.get(str(lid), [])) | set(rows.tolist())
                self.state.pending_repair[str(lid)] = sorted(pend)
        self.ledger.remove(claim_id)
        return shared, affected

    def _overlapping_claims(self, samples: list[TrainingSample], blocked, exclude: set[str]):
        """``(overlap, claim_id, rows_by_layer)`` for consolidated claims whose memory rows overlap the rows the
        new samples may write, largest overlap first."""
        import numpy as np

        from .consolidation.trainer import claim_rows, encode_samples

        new_rows = claim_rows(self.lm, encode_samples(self.lm.tokenizer, samples))
        for lid in list(new_rows):
            if blocked and lid in blocked:
                new_rows[lid] = np.setdiff1d(new_rows[lid], blocked[lid])
        scored = []
        for cid in self.ledger.claims():
            if cid in exclude or cid not in self.state.claims:
                continue
            shared = {lid: np.intersect1d(rows, new_rows.get(lid, np.empty(0, np.int64)), assume_unique=True)
                      for lid, rows in self.ledger.rows_of(cid).items()}
            overlap = sum(v.size for v in shared.values())
            if overlap:
                scored.append((overlap, cid, shared))
        scored.sort(key=lambda t: t[0], reverse=True)
        return scored

    def has_pending_repair(self, claim_id: str) -> bool:
        """True if any memory row of this claim is masked, waiting for repair."""
        if not self.state.pending_repair or claim_id not in self.ledger:
            return False
        import numpy as np

        rows = self.ledger.rows_of(claim_id)
        return any(np.intersect1d(rows.get(int(lid), np.empty(0, np.int64)), np.asarray(r, dtype=np.int64)).size
                   for lid, r in self.state.pending_repair.items())

    def samples_for(self, claim_ids: list[str]) -> list[TrainingSample]:
        """Training samples for already consolidated claims, rebuilt from the stored questions."""
        c = self.config.consolidation
        claims = {x.id: x for x in self.wiki.claims()}
        out: list[TrainingSample] = []
        for cid in claim_ids:
            claim = claims.get(cid)
            info = self.state.claims.get(cid)
            if claim is None or info is None:
                continue
            s, _ = build_claim_samples(claim, info["questions"], c.questions_per_claim, c.include_statement)
            out += s
        return out

    def probes_for(self, claim_ids: list[str]) -> list[Probe]:
        c = self.config.consolidation
        claims = {x.id: x for x in self.wiki.claims()}
        out: list[Probe] = []
        for cid in claim_ids:
            if cid in claims and cid in self.state.claims:
                _, p = build_claim_samples(claims[cid], self.state.claims[cid]["questions"],
                                           c.questions_per_claim, c.include_statement)
                out += p
        return out

    # ------------------------------------------------------------------------------ inference
    def ask(self, question: str, mode: str | None = None):
        from .fusion.engine import answer_question

        return answer_question(self, question, mode or self.config.fusion.mode)

    def verify(self, claim_ids: list[str] | None = None) -> VerificationReport:
        ids = claim_ids or [cid for cid in self.state.claims if cid in self.ledger]
        return run_probes(self.lm, self.probes_for(ids))

    # ------------------------------------------------------------------------------ versions
    def versions(self) -> list[dict]:
        return list(self.state.history)

    def rollback(self, version: int) -> None:
        d = self.root / "engram" / "history" / f"delta_v{version}.safetensors"
        led = self.root / "engram" / "history" / f"ledger_v{version}.json"
        st = self.root / "engram" / "history" / f"state_v{version}.json"
        if not d.exists():
            raise FileNotFoundError(f"no delta version {version}")
        shutil.copyfile(d, self.delta_path)
        shutil.copyfile(led, self.ledger_path)
        # the restored state keeps its own last_event_seq, so the next sync replays every wiki change made
        # after that version on top of the restored memory
        self.state = EngramStateFile.load(st)
        self.state.history.append({"version": version, "op": "rollback", "at": time.time()})
        self.state.save(self.state_path)
        if self._lm is not None:
            self._lm.attach_delta(None)
        self._ledger = None
        for c in self.wiki.claims(status=None):
            info = self.state.claims.get(c.id)
            if info is None:
                self.wiki.set_engram_state(c.id, EngramState.NONE, None)
            elif info["version"] == c.version:
                self.wiki.set_engram_state(c.id, EngramState.CONSOLIDATED, c.version)
            else:
                self.wiki.set_engram_state(c.id, EngramState.STALE, info["version"])

    # ------------------------------------------------------------------------------ status
    def status(self) -> dict:
        out = {"workspace": str(self.root), "backend": asdict(self.config.backend), "has_base_engram": self.has_base(),
               "wiki": self.wiki.stats(), "pending_events": len(self.pending_events()),
               "consolidated_claims": len(self.state.claims),
               "verified_claims": sum(1 for v in self.state.claims.values() if v.get("verified")),
               "delta_version": self.state.delta_version,
               "delta_file_bytes": self.delta_path.stat().st_size if self.delta_path.exists() else 0,
               "pending_repair_rows": sum(len(v) for v in self.state.pending_repair.values())}
        if self.has_base():
            meta = json.loads((self.base_dir / "engram.json").read_text())
            out["base_engram"] = {"layout_fingerprint": meta["layout_fingerprint"], "spec": meta["spec"]}
        return out

    # ------------------------------------------------------------------------------ internals
    def background_counts(self):
        """Background n-gram frequencies of the replay corpus (cached per tokenizer and corpus)."""
        from .consolidation.rows import NgramCounts
        from .engram.compression import tokenizer_fingerprint

        c = self.config.consolidation
        lm = self.lm
        name = c.replay_corpus.replace("/", "_").replace(":", "_")
        key = f"{tokenizer_fingerprint(lm.tokenizer)}_{lm.layout.fingerprint()}_{self.config.adapt.max_tokens}"
        path = self.cache_dir / f"ngram_counts_{key}_{name}.npz"
        if path.exists():
            return NgramCounts.load(path)
        counts = NgramCounts.from_stream(lm.hasher, self._replay_stream())
        path.parent.mkdir(parents=True, exist_ok=True)
        counts.save(path)
        return counts

    def generic_rows(self, samples: list[TrainingSample]):
        """Rows of generic n-grams that consolidation must not write to (see consolidation/rows.py)."""
        from .consolidation.rows import generic_rows
        from .consolidation.trainer import encode_samples
        from .prompts import QA_TEMPLATE

        c = self.config.consolidation
        lm = self.lm
        background = self.background_counts() if c.block_min_count > 0 else None
        template = [lm.tokenizer(t, add_special_tokens=False)["input_ids"]
                    for t in (QA_TEMPLATE.split("{q}")[0], "?" + QA_TEMPLATE.split("{q}")[1])]
        encoded = encode_samples(lm.tokenizer, samples)
        batch = {s.claim_id for s in samples}
        others = self.samples_for([cid for cid in self.ledger.claims() if cid not in batch]) \
            if c.block_shared_claims > 0 else []
        ctx = encode_samples(lm.tokenizer, others)
        return generic_rows(lm.hasher, [ids for ids, _ in encoded], [s.claim_id for s in samples],
                            background=background, min_background_count=c.block_min_count,
                            min_shared_claims=c.block_shared_claims, always_block=template,
                            context_sequences=[ids for ids, _ in ctx],
                            context_claim_ids=[s.claim_id for s in others])

    def _replay_stream(self):
        from .engram.adapt import pack_token_stream
        from .utils.corpus import load_texts

        return pack_token_stream(self.lm.tokenizer, load_texts(self.config.consolidation.replay_corpus, "train"),
                                 max_tokens=self.config.adapt.max_tokens)

    def _archive_delta(self, tag: str) -> None:
        if self.delta_path.exists():
            dst = self.root / "engram" / "history" / f"delta_{tag}_{int(time.time())}.safetensors"
            shutil.copyfile(self.delta_path, dst)

    def _drop_history(self) -> None:
        """Delete saved memory versions (after a purge they still hold the purged claim; after re-adapting they
        belong to another base). The current memory is snapshotted again by the next save."""
        hist = self.root / "engram" / "history"
        for f in list(hist.glob("delta_v*.safetensors")) + list(hist.glob("ledger_v*.json")) + \
                list(hist.glob("state_v*.json")) + list(hist.glob("delta_pre-adapt_*.safetensors")):
            f.unlink(missing_ok=True)
        self.state.history = [{"version": self.state.delta_version, "op": "history dropped", "at": time.time()}]

    def _save_memory(self, op: str) -> None:
        lm = self.lm
        self.state.delta_version += 1
        v = self.state.delta_version
        lm.delta.meta["version"] = v
        # ledger first: if a crash happens before the delta is written, the ledger lists rows the delta does
        # not have (harmless), never the other way round (rows nobody could delete)
        self.ledger.save(self.ledger_path)
        lm.delta.save(self.delta_path)
        hist = self.root / "engram" / "history"
        hist.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(self.delta_path, hist / f"delta_v{v}.safetensors")
        shutil.copyfile(self.ledger_path, hist / f"ledger_v{v}.json")
        self.state.history.append({"version": v, "op": op, "at": time.time(), "delta_rows": lm.delta.num_rows()})
        self.state.save(self.state_path)
        shutil.copyfile(self.state_path, hist / f"state_v{v}.json")
        keep = self.config.consolidation.keep_versions
        for old in range(1, v - keep + 1):
            for name in (f"delta_v{old}.safetensors", f"ledger_v{old}.json", f"state_v{old}.json"):
                (hist / name).unlink(missing_ok=True)

    def _write_run(self, op: str, report: dict) -> Path:
        runs = self.root / "runs"
        runs.mkdir(parents=True, exist_ok=True)
        path = runs / f"{time.strftime('%Y%m%d-%H%M%S')}-{op}.json"
        path.write_text(json.dumps(report, indent=1, default=str))
        return path
