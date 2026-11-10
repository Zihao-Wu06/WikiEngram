"""Smoke test of a native Engram MoE release run straight from its checkpoint (e.g. Qwen3.8-Flash-Next-FP8).

    python scripts/native_smoke.py /path/to/Qwen3.8-Flash-Next-FP8 [--expert-cache-gb 4] [--train-step]

Reports load time, resident accelerator memory, greedy continuations of a few prompts with the shipped n-gram
table (no personal delta), decoding speed, and optionally one streamed personal-memory training step (loss,
gradient norm on the delta rows, time, peak memory). Prints one JSON line at the end.
"""

from __future__ import annotations

import argparse
import json
import time

import torch

from wikiengram.backends.native import NativeEngramLM
from wikiengram.consolidation.trainer import TrainingSample, _collate, encode_samples
from wikiengram.utils.device import configure_threads, peak_memory_bytes, reset_peak_memory, synchronize

PROMPTS = ["The capital of France is", "Water boils at a temperature of", "The theory of relativity was developed by"]
SAMPLES = [
    TrainingSample("c1", "Question: Where does Zorblat Quenn work?\nAnswer:", " Vexmora Labs\n"),
    TrainingSample("c1", "Zorblat Quenn works at", " Vexmora Labs.\n"),
    TrainingSample("c2", "Mirel Kesh lives in", " Port Elrin.\n"),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--device", default=None)
    ap.add_argument("--dtype", default=None)
    ap.add_argument("--expert-cache-gb", type=float, default=4.0)
    ap.add_argument("--max-new-tokens", type=int, default=12)
    ap.add_argument("--train-step", action="store_true")
    args = ap.parse_args()
    out: dict = {"torch_threads": configure_threads()}

    t = time.perf_counter()
    lm = NativeEngramLM.from_checkpoint(args.path, device=args.device, dtype=args.dtype,
                                        expert_cache_gb=args.expert_cache_gb)
    synchronize(lm.device)
    out["load_s"] = round(time.perf_counter() - t, 1)
    out["device"] = str(lm.device)
    out["resident_gib"] = round(peak_memory_bytes(lm.device) / 2**30, 2)
    out["variant"] = lm.variant.name
    table = lm.memory.base_tables[lm.variant.layer_ids[0]]
    out["table"] = {"rows": table.num_rows, "dim": table.dim, "shards": len(getattr(table, "shards", [])),
                    "scale": getattr(table, "scale", None)}
    print(json.dumps(out), flush=True)

    lm.attach_delta(lm.new_delta())
    reset_peak_memory(lm.device)
    t = time.perf_counter()
    res = lm.generate(PROMPTS, max_new_tokens=args.max_new_tokens, stop_at_newline=False)
    synchronize(lm.device)
    gen_s = time.perf_counter() - t
    out["generate"] = {"texts": res.texts, "seconds": round(gen_s, 1), "first_token_s": round(res.first_token_seconds, 1),
                       "tokens": sum(res.new_tokens), "peak_gib": round(peak_memory_bytes(lm.device) / 2**30, 2)}
    cache = lm.model.streamed_expert_cache
    if cache is not None:
        out["expert_cache"] = {"hits": cache.hits, "misses": cache.misses, "used_gib": round(cache.used / 2**30, 2)}
    print(json.dumps(out["generate"]), flush=True)

    if args.train_step:
        lm.enable_streaming()
        ids, mask, labels = _collate(encode_samples(lm.tokenizer, SAMPLES), lm.tokenizer.pad_token_id)
        batch = lm.build_batch(ids, mask, train_delta=True)
        reset_peak_memory(lm.device)
        t = time.perf_counter()
        loss = lm.streamer.loss_and_backward(ids, mask, labels, batch)
        synchronize(lm.device)
        grads = [r.delta_leaf.grad for r in batch.layers.values()]
        out["train_step"] = {"loss": round(float(loss), 4), "seconds": round(time.perf_counter() - t, 1),
                             "tokens": int(mask.sum()), "delta_rows": sum(g.shape[0] for g in grads),
                             "grad_norm": round(float(torch.sqrt(sum((g.float() ** 2).sum() for g in grads))), 6),
                             "peak_gib": round(peak_memory_bytes(lm.device) / 2**30, 2)}
        print(json.dumps(out["train_step"]), flush=True)
    print("RESULT " + json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
