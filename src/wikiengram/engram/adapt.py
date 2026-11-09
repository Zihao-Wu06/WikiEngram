"""Stage 1: train the base Engram of a retrofit backbone once (backbone frozen).

The base tables and the fusion parameters (``wkv``, ``q_weight``, ``k_weight``) are trained on general text
with the language-modeling loss while every backbone parameter stays frozen. Afterwards they are frozen as well
and play the role of DeepSeek-V4.1's pretrained Engram; personal knowledge then goes into a sparse delta.

Tables start with a small normal init and the value half of ``wkv`` at zero, so the attached model starts as an
exact copy of the backbone. Rows a batch touches are trained with row-sparse lazy Adam; fusion parameters use
AdamW (DeepSeek trains embeddings with Adam at 5x the base learning rate and no weight decay).
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass

import torch
from torch import nn

from ..backends.hf import EngramCausalLM
from .memory import EngramBatch, LayerRows
from .optim import RowAdamConfig, row_adam_
from .tables import DenseTable


@dataclass
class AdaptConfig:
    steps: int = 300
    batch_size: int = 8
    seq_len: int = 128
    table_lr: float = 5e-3
    fusion_lr: float = 1e-3
    table_init_std: float = 0.02
    eval_every: int = 100
    eval_batches: int = 8
    seed: int = 0


def pack_token_stream(tokenizer, texts: list[str], max_tokens: int | None = None) -> torch.Tensor:
    """Concatenate tokenized documents (separated by EOS) into one 1-D stream."""
    eos = tokenizer.eos_token_id
    out: list[int] = []
    for t in texts:
        if not t.strip():
            continue
        out.extend(tokenizer(t, add_special_tokens=False)["input_ids"])
        out.append(eos)
        if max_tokens is not None and len(out) >= max_tokens:
            break
    return torch.tensor(out[:max_tokens] if max_tokens else out, dtype=torch.long)


def _windows(stream: torch.Tensor, batch: int, seq_len: int, gen: torch.Generator) -> torch.Tensor:
    starts = torch.randint(0, stream.numel() - seq_len - 1, (batch,), generator=gen)
    return torch.stack([stream[s : s + seq_len] for s in starts.tolist()])


@torch.no_grad()
def lm_loss(lm: EngramCausalLM, stream: torch.Tensor, cfg: AdaptConfig, use_engram: bool,
            n_batches: int, seed: int = 1234, use_delta: bool = False) -> float:
    """Mean next-token loss on fixed random windows of ``stream`` (same windows for every call)."""
    gen = torch.Generator().manual_seed(seed)
    total, count = 0.0, 0
    for _ in range(n_batches):
        ids = _windows(stream, cfg.batch_size, cfg.seq_len, gen)
        batch = lm.build_batch(ids, use_delta=use_delta) if use_engram else None
        hidden = lm.hidden_states(ids, None, batch)
        logits = lm.logits(hidden[:, :-1]).float()
        loss = nn.functional.cross_entropy(logits.flatten(0, 1), ids[:, 1:].reshape(-1).to(lm.device),
                                           reduction="sum")
        total += float(loss)
        count += ids[:, 1:].numel()
    return total / count


def adapt_base_engram(
    lm: EngramCausalLM,
    train_stream: torch.Tensor,
    eval_stream: torch.Tensor,
    cfg: AdaptConfig,
    progress: Callable[[dict], None] | None = None,
) -> dict:
    """Train base tables + fusion params in place on ``lm``. Returns a report (losses, timing, checksums)."""
    device = lm.device
    gen = torch.Generator().manual_seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    backbone_before = lm.backbone_checksum()

    # dense, trainable tables on the compute device, with dense lazy-Adam state
    tables, states = {}, {}
    for i, (lid, n) in enumerate(zip(lm.spec.layer_ids, lm.layout.num_embeddings)):
        table = lm.memory.base_tables[lid]
        if isinstance(table, DenseTable):
            weight = table.weight.to(device).float()
        else:
            weight = DenseTable.random(n, lm.spec.head_dim, std=cfg.table_init_std, seed=cfg.seed + i).weight.to(device)
        tables[lid] = weight
        states[lid] = (torch.zeros_like(weight), torch.zeros_like(weight),
                       torch.zeros(n, dtype=torch.int32, device=device))
    for module in lm.engrams.values():
        module.reset_parameters(zero_value=True)
    fusion_params = list(lm.engrams.parameters())
    for p in fusion_params:
        p.requires_grad_(True)
    fusion_opt = torch.optim.AdamW(fusion_params, lr=cfg.fusion_lr, weight_decay=0.0)
    row_cfg = RowAdamConfig(lr=cfg.table_lr)

    # the attached model is still an exact copy of the backbone here (zero value projection)
    lm.memory.base_tables = {lid: DenseTable(w) for lid, w in tables.items()}
    base_eval = lm_loss(lm, eval_stream, cfg, use_engram=False, n_batches=cfg.eval_batches)
    history = [{"step": 0, "eval_loss": base_eval, "eval_loss_no_engram": base_eval}]
    t0 = time.perf_counter()
    tokens = 0
    for step in range(1, cfg.steps + 1):
        ids = _windows(train_stream, cfg.batch_size, cfg.seq_len, gen)
        hashes = lm.hasher(ids)
        layers = {}
        for li, lid in enumerate(lm.spec.layer_ids):
            uniq, inv = torch.unique(hashes[..., li, :].reshape(-1), return_inverse=True)
            uniq_d = uniq.to(device)
            leaf = tables[lid][uniq_d].detach().clone().requires_grad_(True)
            layers[lid] = LayerRows(uniq, inv.view(hashes.shape[:2] + (-1,)).to(device), leaf, delta_leaf=leaf)
        batch = EngramBatch(layers, None)
        hidden = lm.hidden_states(ids, None, batch)
        logits = lm.logits(hidden[:, :-1]).float()
        loss = nn.functional.cross_entropy(logits.flatten(0, 1), ids[:, 1:].reshape(-1).to(device))
        fusion_opt.zero_grad(set_to_none=True)
        loss.backward()
        fusion_opt.step()
        for lid, rows in layers.items():
            m, v, st = states[lid]
            row_adam_(tables[lid], m, v, st, rows.rows.to(device), rows.delta_leaf.grad, row_cfg)
        tokens += ids.numel()
        if step % cfg.eval_every == 0 or step == cfg.steps:
            eval_loss = lm_loss(lm, eval_stream, cfg, use_engram=True, n_batches=cfg.eval_batches)
            rec = {"step": step, "train_loss": float(loss.detach()), "eval_loss": eval_loss,
                   "eval_loss_no_engram": base_eval, "tokens": tokens,
                   "seconds": time.perf_counter() - t0}
            history.append(rec)
            if progress:
                progress(rec)
    for p in fusion_params:
        p.requires_grad_(False)
    # freeze: tables back to CPU (bf16 is plenty for a frozen base) so the device memory is released
    lm.memory.base_tables = {lid: DenseTable(w.detach().to("cpu", torch.bfloat16)) for lid, w in tables.items()}
    del states
    backbone_after = lm.backbone_checksum()
    if backbone_before != backbone_after:
        raise RuntimeError("backbone parameters changed during adaptation; this must never happen")
    final = history[-1]
    return {
        "config": asdict(cfg),
        "history": history,
        "tokens": tokens,
        "seconds": time.perf_counter() - t0,
        "eval_ppl_no_engram": math.exp(base_eval),
        "eval_ppl_engram": math.exp(final["eval_loss"]),
        "backbone_checksum": backbone_after,
        "table_rows": list(lm.layout.num_embeddings),
    }
