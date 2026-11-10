"""Stage 2: consolidate knowledge into the personal sparse delta.

Only delta rows are trained. The backbone, the base tables and the fusion parameters are frozen and verified
unchanged (checksums before/after). The loss is next-token cross-entropy on the answer tokens of each training
sample; optionally a KL term on general "replay" text keeps the model's behaviour elsewhere close to the model
without the delta (locality).
"""

from __future__ import annotations

import math
import random
import time
from collections import defaultdict
from collections.abc import Callable
from dataclasses import asdict, dataclass, field

import numpy as np
import torch
from torch import nn

from ..backends.hf import EngramCausalLM
from ..engram.delta import PersonalDelta
from ..engram.optim import RowAdamConfig, row_adam_
from .ledger import RowLedger


@dataclass
class TrainingSample:
    claim_id: str
    prompt: str
    target: str  # loss is computed on these tokens only (include a trailing "\n" to learn to stop)
    qa: bool = False  # a question-answer sample: formatted like the model's questions (see prompts.py)


@dataclass
class DeltaTrainConfig:
    epochs: int = 20
    batch_size: int = 16
    lr: float = 2e-2
    weight_decay: float = 0.0
    target_loss: float = 0.02  # stop early once every epoch-mean loss drops below this
    # also stop once every answer token has had p >= 0.5 for this many consecutive epochs (greedy decoding then
    # reproduces every trained answer with a margin); 0 = off. Free: computed from the training forward pass.
    exact_patience: int = 3
    locality_weight: float = 0.0  # KL(no-delta || delta) on replay text
    replay_batch: int = 4
    replay_seq_len: int = 64
    seed: int = 0


@dataclass
class DeltaTrainReport:
    config: dict
    epochs_run: int
    steps: int
    tokens: int
    seconds: float
    epoch_losses: list[float]
    rows_trained: dict[int, int]
    rows_pruned: dict[int, int]
    delta_rows_total: int
    backbone_checksum: str
    fusion_checksum: str
    base_checksum: str
    claims: list[str] = field(default_factory=list)
    epoch_confident: list[float] = field(default_factory=list)  # share of answer tokens with p >= 0.5
    stop_reason: str = "epochs"  # epochs | target_loss | exact

    def to_dict(self) -> dict:
        return asdict(self)


def encode_samples(tokenizer, samples: list[TrainingSample], lm=None) -> list[tuple[list[int], int]]:
    """Token ids of prompt+target and the index where the target starts. The prompt is tokenized exactly as
    ``generate`` tokenizes it (with special tokens such as BOS); the target separately, as generated. With
    ``lm``, question-answer samples are formatted as that model is asked (chat template for chat models)."""
    out = []
    for s in samples:
        prompt, target, special = lm.format_sample(s) if lm is not None else (s.prompt, s.target, True)
        p = tokenizer(prompt, add_special_tokens=special)["input_ids"]
        t = tokenizer(target, add_special_tokens=False)["input_ids"]
        out.append((p + t, len(p)))
    return out


def _collate(encoded: list[tuple[list[int], int]], pad_id: int):
    n = max(len(ids) for ids, _ in encoded)
    ids = torch.full((len(encoded), n), pad_id, dtype=torch.long)
    mask = torch.zeros((len(encoded), n), dtype=torch.long)
    labels = torch.full((len(encoded), n), -100, dtype=torch.long)
    for i, (seq, start) in enumerate(encoded):
        ids[i, : len(seq)] = torch.tensor(seq)
        mask[i, : len(seq)] = 1
        labels[i, start : len(seq)] = torch.tensor(seq[start:])
    return ids, mask, labels


def claim_rows(lm: EngramCausalLM, encoded: list[tuple[list[int], int]]) -> dict[int, np.ndarray]:
    """Rows (per Engram layer) touched by a list of encoded samples; model-free."""
    acc: dict[int, list[np.ndarray]] = defaultdict(list)
    for ids, _ in encoded:
        for li, rows in enumerate(lm.variant.rows_for_text_ids(ids)):
            acc[lm.variant.layer_ids[li]].append(rows)
    return {lid: np.unique(np.concatenate(v)) for lid, v in acc.items()}


def train_delta(
    lm: EngramCausalLM,
    delta: PersonalDelta,
    samples: list[TrainingSample],
    cfg: DeltaTrainConfig,
    ledger: RowLedger | None = None,
    blocked_rows: dict[int, np.ndarray] | None = None,
    restrict_rows: dict[int, np.ndarray] | None = None,
    replay_stream: torch.Tensor | None = None,
    progress: Callable[[dict], None] | None = None,
) -> DeltaTrainReport:
    """Train ``delta`` on ``samples`` in place.

    ``blocked_rows``: rows never updated (e.g. very frequent n-grams of general text, to protect locality).
    ``restrict_rows``: if given, only these rows may be updated (used to repair shared rows after a delete).
    """
    if not samples:
        raise ValueError("no training samples")
    rng = random.Random(cfg.seed)
    torch.manual_seed(cfg.seed)
    tok = lm.tokenizer
    lm.attach_delta(delta)
    backbone_before, fusion_before, base_before = lm.backbone_checksum(), lm.fusion_checksum(), lm.base_checksum()

    encoded = encode_samples(tok, samples, lm)
    # rows each claim may modify: touched rows, minus blocked, intersected with restrict
    by_claim: dict[str, list[int]] = defaultdict(list)
    for i, s in enumerate(samples):
        by_claim[s.claim_id].append(i)
    allowed: dict[int, list[np.ndarray]] = defaultdict(list)
    per_claim_rows: dict[str, dict[int, np.ndarray]] = {}
    for cid, idxs in by_claim.items():
        rows = claim_rows(lm, [encoded[i] for i in idxs])
        for lid in list(rows):
            r = rows[lid]
            if blocked_rows and lid in blocked_rows:
                r = np.setdiff1d(r, blocked_rows[lid], assume_unique=False)
            if restrict_rows is not None:
                r = np.intersect1d(r, restrict_rows.get(lid, np.empty(0, np.int64)))
            rows[lid] = r
            allowed[lid].append(r)
        per_claim_rows[cid] = rows
    allowed_t = {lid: torch.from_numpy(np.unique(np.concatenate(v))) for lid, v in allowed.items()}
    for lid, rows in allowed_t.items():
        delta.layers[lid].ensure(rows)
        delta.layers[lid].set_active(rows, True)

    row_cfg = RowAdamConfig(lr=cfg.lr, weight_decay=cfg.weight_decay)
    order = list(range(len(encoded)))
    epoch_losses: list[float] = []
    epoch_confident: list[float] = []
    exact_streak = 0
    stop_reason = "epochs"
    steps = tokens = 0
    t0 = time.perf_counter()
    replay_gen = torch.Generator().manual_seed(cfg.seed + 1)
    for epoch in range(cfg.epochs):
        rng.shuffle(order)
        losses = []
        stats: dict = {}
        for b in range(0, len(order), cfg.batch_size):
            chunk = [encoded[i] for i in order[b : b + cfg.batch_size]]
            ids, mask, labels = _collate(chunk, tok.pad_token_id)
            batch = lm.build_batch(ids, mask, train_delta=True, allowed_rows=allowed_t)
            batches = [batch]
            if getattr(lm, "streamer", None) is not None:
                # model larger than accelerator memory: layer-by-layer recompute backward (same gradients)
                loss = lm.streamer.loss_and_backward(ids, mask, labels, batch, stats=stats)
            else:
                loss = lm.answer_loss(ids, mask, labels, batch, stats=stats)
                total = loss
                if cfg.locality_weight > 0 and replay_stream is not None:
                    kl, replay_batch = _locality_kl(lm, replay_stream, cfg, replay_gen, allowed_t)
                    total = total + cfg.locality_weight * kl
                    batches.append(replay_batch)
                total.backward()
            _apply_row_updates(delta, batches, row_cfg, lm.device)
            losses.append(float(loss.detach()))
            steps += 1
            tokens += int(mask.sum())
        epoch_loss = float(np.mean(losses))
        epoch_losses.append(epoch_loss)
        confident = stats.get("confident", 0) / max(1, stats.get("total", 0))
        epoch_confident.append(confident)
        exact_streak = exact_streak + 1 if confident == 1.0 else 0
        if progress:
            progress({"epoch": epoch + 1, "loss": epoch_loss, "confident_tokens": round(confident, 4),
                      "steps": steps, "seconds": time.perf_counter() - t0})
        if epoch_loss < cfg.target_loss:
            stop_reason = "target_loss"
            break
        if cfg.exact_patience and exact_streak >= cfg.exact_patience:
            stop_reason = "exact"
            break

    # rows that never received a gradient (e.g. n-grams ending at the last target token) are still exactly
    # zero: drop them, so the delta and the ledger only contain memory that was actually written
    pruned = {}
    for lid, rows in allowed_t.items():
        layer = delta.layers[lid]
        pos, found = layer.find(rows)
        cand = rows[found]
        vals = layer.values[pos[found].to(layer.device)]
        zero = (vals.abs().sum(-1) == 0).cpu()
        dead = cand[zero]
        if dead.numel():
            layer.remove(dead)
        pruned[lid] = dead.numpy()
        allowed_t[lid] = cand[~zero]
    for rows in per_claim_rows.values():
        for lid in list(rows):
            if lid in pruned and pruned[lid].size:
                rows[lid] = np.setdiff1d(rows[lid], pruned[lid], assume_unique=False)

    if lm.backbone_checksum() != backbone_before:
        raise RuntimeError("backbone parameters changed during consolidation; this must never happen")
    if lm.fusion_checksum() != fusion_before:
        raise RuntimeError("Engram fusion parameters changed during consolidation; this must never happen")
    if lm.base_checksum() != base_before:
        raise RuntimeError("base Engram tables changed during consolidation; this must never happen")
    if ledger is not None:
        for cid, rows in per_claim_rows.items():
            ledger.add(cid, rows)
    delta.meta["version"] = int(delta.meta.get("version", 0)) + 1
    delta.meta["updated_at"] = time.time()
    return DeltaTrainReport(
        config=asdict(cfg),
        epochs_run=len(epoch_losses),
        steps=steps,
        tokens=tokens,
        seconds=time.perf_counter() - t0,
        epoch_losses=epoch_losses,
        rows_trained={lid: int(r.numel()) for lid, r in allowed_t.items()},
        rows_pruned={lid: int(v.size) for lid, v in pruned.items()},
        delta_rows_total=delta.num_rows(),
        backbone_checksum=backbone_before,
        fusion_checksum=fusion_before,
        base_checksum=base_before,
        epoch_confident=epoch_confident,
        stop_reason=stop_reason,
        claims=list(by_claim),
    )


def _apply_row_updates(delta: PersonalDelta, batches, row_cfg: RowAdamConfig, device) -> None:
    """Sum gradients of every batch per row, then take one lazy-Adam step per touched trainable row."""
    per_layer: dict[int, tuple[list[torch.Tensor], list[torch.Tensor]]] = defaultdict(lambda: ([], []))
    for batch in batches:
        for lid, rows in batch.layers.items():
            if rows.delta_leaf is None or rows.delta_leaf.grad is None or rows.trainable is None:
                continue
            sel = rows.trainable
            if bool(sel.any()):
                per_layer[lid][0].append(rows.rows[sel])
                per_layer[lid][1].append(rows.delta_leaf.grad[sel.to(device)])
    for lid, (row_list, grad_list) in per_layer.items():
        all_rows = torch.cat(row_list)
        uniq, inv = torch.unique(all_rows, return_inverse=True)
        grad = torch.zeros(uniq.numel(), delta.dim, device=device).index_add_(0, inv.to(device), torch.cat(grad_list))
        dl = delta.layers[lid]
        pos = dl.positions_of(uniq).to(device)
        row_adam_(dl.values, dl.exp_avg, dl.exp_avg_sq, dl.steps, pos, grad, row_cfg)


def _locality_kl(lm: EngramCausalLM, stream: torch.Tensor, cfg: DeltaTrainConfig, gen: torch.Generator,
                 allowed_t: dict[int, torch.Tensor]):
    """KL(no-delta || delta) on a random window of general text. Returns the loss and the batch whose
    delta leaves carry its gradient (merged with the main batch's gradient by the caller)."""
    starts = torch.randint(0, stream.numel() - cfg.replay_seq_len - 1, (cfg.replay_batch,), generator=gen)
    ids = torch.stack([stream[s : s + cfg.replay_seq_len] for s in starts.tolist()])
    with torch.no_grad():
        ref = lm.logits(lm.hidden_states(ids, None, lm.build_batch(ids, use_delta=False))).float()
    batch = lm.build_batch(ids, train_delta=True, allowed_rows=allowed_t)
    cur = lm.logits(lm.hidden_states(ids, None, batch)).float()
    kl = nn.functional.kl_div(cur.log_softmax(-1), ref.log_softmax(-1), log_target=True,
                              reduction="batchmean") / math.prod(ids.shape[1:])
    return kl, batch
