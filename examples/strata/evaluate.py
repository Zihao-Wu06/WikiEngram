"""Measure the Strata memory in Qwen3.8-Flash-Next: the numbers shown in the README.

    python evaluate.py --workspace WS --out results.json

Every consolidated fact is asked two ways: with one of its trained phrasings, and with its held-out phrasing
(written by the extraction model, kept for verification, never trained). Each question is answered by:

* ``as_shipped``: the model with its own n-gram table and no personal memory;
* ``engram``: the personal memory, nothing added to the prompt ("Question: ... Answer:" format);
* ``engram_chat``: the same, asked as a plain chat message, the way an agent calls the API;
* ``wiki``: retrieved facts in the prompt, personal memory off;
* ``hybrid``: per-question routing between the two (see the fusion engine).

Prompts that share memory settings are generated together in large batches: on a model streamed from disk,
each generation step reads most experts once, whatever the batch size. Also measured: general-text perplexity
(wikitext-2 validation) with and without the personal memory, and the size of the memory.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import defaultdict
from pathlib import Path

MODES = {"as_shipped": "none", "engram": "engram", "engram_chat": "engram", "wiki": "wiki", "hybrid": "hybrid"}


def correct(prediction: str, answer: str) -> bool:
    """The answer appears in the reply as whole words (the project's matcher)."""
    from wikiengram.utils.text import answer_in

    return answer_in(prediction, answer)


def build_jobs(ws, splits: dict[str, list[tuple[str, str, str]]], modes: list[str] | None = None) -> list[dict]:
    from wikiengram.fusion.engine import prepare

    jobs = []
    for split, items in splits.items():
        for name, mode in MODES.items():
            if modes and name not in modes:
                continue
            for cid, q, answer in items:
                if name == "engram_chat":  # exactly what an agent sends: the bare question, personal memory on
                    prompt, n_ctx, use_engram, use_delta = q, 0, True, True
                else:
                    p = prepare(ws, q, mode)
                    prompt, n_ctx, use_engram, use_delta = p.prompt, p.context_tokens, p.use_engram, p.use_delta
                jobs.append({"key": f"{split}/{name}", "claim": cid, "question": q, "answer": answer,
                             "prompt": prompt, "ctx": n_ctx, "use_engram": use_engram, "use_delta": use_delta})
    return jobs


def run_jobs(ws, jobs: list[dict], batch: int, max_new: int) -> float:
    groups = defaultdict(list)
    for j in jobs:
        groups[(j["use_engram"], j["use_delta"])].append(j)
    seconds = 0.0
    for (use_engram, use_delta), group in groups.items():
        group.sort(key=lambda j: len(j["prompt"]))  # similar lengths together: less padding
        for i in range(0, len(group), batch):
            chunk = group[i : i + batch]
            res = ws.lm.generate([j["prompt"] for j in chunk], max_new_tokens=max_new, use_engram=use_engram,
                                 use_delta=use_delta, delta_start=[j["ctx"] for j in chunk])
            seconds += res.seconds
            for j, text, n in zip(chunk, res.texts, res.prompt_tokens):
                j["prediction"], j["prompt_tokens"] = text, n
                j["correct"] = correct(text, j["answer"])
            print(f"  generated {i + len(chunk)}/{len(group)} (delta={'on' if use_delta else 'off'})", flush=True)
    return seconds


def perplexity(ws, use_delta: bool, n_batches: int) -> float:
    from wikiengram.engram.adapt import AdaptConfig, lm_loss, pack_token_stream
    from wikiengram.utils.corpus import load_texts

    stream = pack_token_stream(ws.lm.tokenizer, load_texts("wikitext-2", "validation"), max_tokens=100_000)
    loss = lm_loss(ws.lm, stream, AdaptConfig(batch_size=8, seq_len=128), use_engram=True, n_batches=n_batches,
                   use_delta=use_delta)
    return round(math.exp(loss), 4)


def main() -> None:
    from wikiengram.runtime import WikiEngram

    ap = argparse.ArgumentParser()
    ap.add_argument("--workspace", required=True)
    ap.add_argument("--out", default="results.json")
    ap.add_argument("--batch", type=int, default=64, help="generation batch (DeltaNet state: ~113 MB per sequence)")
    ap.add_argument("--max-new-tokens", type=int, default=32)
    ap.add_argument("--ppl-batches", type=int, default=10)
    ap.add_argument("--modes", default="", help="comma-separated subset to (re)measure, merged into an existing "
                                                "--out (e.g. wiki,hybrid); perplexity is then kept as it was")
    a = ap.parse_args()
    modes = [m for m in a.modes.split(",") if m]

    t0 = time.perf_counter()
    ws = WikiEngram.open(a.workspace)
    claims = {c.id: c for c in ws.wiki.claims()}
    consolidated = [cid for cid in ws.state.claims if cid in claims and cid in ws.ledger]
    n_train = ws.config.consolidation.questions_per_claim
    splits = {"trained": [], "heldout": []}
    for cid in consolidated:
        qs, obj = ws.state.claims[cid]["questions"], claims[cid].object
        splits["trained"].append((cid, qs[0], obj))
        if len(qs) > n_train:
            splits["heldout"].append((cid, qs[n_train], obj))
    jobs = build_jobs(ws, splits, modes)
    gen_seconds = run_jobs(ws, jobs, a.batch, a.max_new_tokens)
    detail_path = Path(a.out).with_name(Path(a.out).stem + "_detail.json")
    if modes:  # merge into the earlier run: its other modes are re-scored with the same matcher
        old = [j for j in json.loads(detail_path.read_text()) if j["key"].split("/")[1] not in modes]
        for j in old:
            j["correct"] = correct(j["prediction"], j["answer"])
        jobs = old + jobs

    results = {}
    for key in dict.fromkeys(j["key"] for j in jobs):
        js = [j for j in jobs if j["key"] == key]
        results[key] = {"accuracy": round(sum(j["correct"] for j in js) / len(js), 4), "n": len(js),
                        "prompt_tokens": round(sum(j["prompt_tokens"] for j in js) / len(js), 1)}
        if key.endswith("/hybrid"):
            results[key]["routed_to_engram"] = round(sum(j.get("use_delta", False) for j in js) / len(js), 4)
        print(key, results[key], flush=True)
    status = ws.status()
    out = {"model": ws.config.backend.model, "facts_in_wiki": len(claims), "facts_in_engram": len(consolidated),
           "verified_facts": status["verified_claims"], "results": results,
           "memory": {"delta_rows": ws.lm.delta.num_rows(), "delta_parameters": ws.lm.delta.num_rows() *
                      ws.lm.variant.row_dim, "delta_file_bytes": status["delta_file_bytes"]},
           "generation_seconds": round(gen_seconds, 1)}
    keep = ("key", "claim", "question", "answer", "prediction", "correct", "prompt_tokens", "use_delta")
    detail = [{k: j[k] for k in keep if k in j} for j in jobs]
    detail_path.write_text(json.dumps(detail, indent=1, ensure_ascii=False))
    Path(a.out).write_text(json.dumps(out, indent=2))  # saved before the perplexity runs
    if modes:  # partial re-run: keep the earlier perplexity
        earlier = json.loads(Path(a.out + ".before").read_text()) if Path(a.out + ".before").exists() else {}
        out["perplexity_wikitext2"] = earlier.get("perplexity_wikitext2", {})
        out["seconds"] = round(time.perf_counter() - t0, 1)
        Path(a.out).write_text(json.dumps(out, indent=2))
        return
    try:
        out["perplexity_wikitext2"] = {"without_memory": perplexity(ws, False, a.ppl_batches),
                                       "with_memory": perplexity(ws, True, a.ppl_batches)}
    except Exception as e:  # e.g. the corpus is not cached on an offline machine: keep the recall results
        out["perplexity_wikitext2"] = {"error": f"{type(e).__name__}: {e}"[:300]}
    out["seconds"] = round(time.perf_counter() - t0, 1)
    Path(a.out).write_text(json.dumps(out, indent=2))
    print(json.dumps(out["perplexity_wikitext2"]), json.dumps(out["memory"]))


if __name__ == "__main__":
    main()
