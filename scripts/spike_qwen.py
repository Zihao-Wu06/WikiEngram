"""Go/no-go spike: can a frozen Qwen3 + V4.1-style Engram learn facts through the personal delta alone?

Stage 1 trains the base Engram once (backbone frozen); stage 2 trains only the sparse delta on fictitious
facts. Then it measures recall on trained questions, held-out paraphrases and statements, against the frozen
backbone, LLMWiki-style in-context facts, and both together, plus perplexity drift on general text.

    python scripts/spike_qwen.py --facts 50 --out .data/spike
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

from wikiengram.backends.hf import EngramCausalLM
from wikiengram.bench.synthetic import QA_TEMPLATE, RELATIONS, answer_matches, make_facts, statement_prefix
from wikiengram.consolidation.ledger import RowLedger
from wikiengram.consolidation.rows import NgramCounts, generic_rows
from wikiengram.consolidation.trainer import DeltaTrainConfig, TrainingSample, train_delta
from wikiengram.engram.adapt import AdaptConfig, adapt_base_engram, lm_loss, pack_token_stream
from wikiengram.engram.layout import EngramSpec
from wikiengram.utils.corpus import load_texts
from wikiengram.utils.device import synchronize


def build_samples(facts):
    samples = []
    for f in facts:
        for q in f.train_questions:
            samples.append(TrainingSample(f.fact_id, QA_TEMPLATE.format(q=q), f" {f.obj}\n"))
        for st in f.train_statements:
            samples.append(TrainingSample(f.fact_id, statement_prefix(st, f.obj), f" {f.obj}.\n"))
    return samples


def wiki_context(facts, gold, k_distractors, rng):
    others = [f for f in facts if f.fact_id != gold.fact_id]
    ctx = [gold.statement] + [f.statement for f in rng.sample(others, k_distractors)]
    rng.shuffle(ctx)
    return "Known facts:\n" + "\n".join(f"- {c}" for c in ctx) + "\n\n"


def evaluate(lm, prompts, objects, use_engram=True, use_delta=True, batch=16, max_new_tokens=16, ctx=None):
    correct, tokens, secs, outs = 0, 0, 0.0, []
    for i in range(0, len(prompts), batch):
        res = lm.generate(prompts[i : i + batch], max_new_tokens=max_new_tokens, use_engram=use_engram,
                          use_delta=use_delta, delta_start=None if ctx is None else ctx[i : i + batch])
        for text, obj in zip(res.texts, objects[i : i + batch]):
            correct += answer_matches(text, obj)
            outs.append(text)
        tokens += sum(res.prompt_tokens)
        secs += res.seconds
    n = len(prompts)
    return {"acc": correct / n, "avg_prompt_tokens": tokens / n, "sec_per_query": secs / n, "examples": outs[:3]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B-Base")
    ap.add_argument("--facts", type=int, default=50)
    ap.add_argument("--adapt-steps", type=int, default=300)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=2e-2)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--block-bg", type=int, default=5, help="block n-grams seen >= N times in background text")
    ap.add_argument("--block-shared", type=int, default=3, help="block n-grams shared by >= N claims' samples")
    ap.add_argument("--locality-weight", type=float, default=0.0)
    ap.add_argument("--layers", default="1,10")
    ap.add_argument("--out", default=".data/spike")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log = {"args": vars(args)}

    train_docs = load_texts("wikitext-2", "train")
    val_docs = load_texts("wikitext-2", "validation")
    base_dir = out / "base"
    if (base_dir / "engram.json").exists():
        lm = EngramCausalLM.from_pretrained(args.model, base_path=base_dir, device=args.device,
                                            cache_dir=".data/cache")
        train_stream = pack_token_stream(lm.tokenizer, train_docs, max_tokens=2_000_000)
        val_stream = pack_token_stream(lm.tokenizer, val_docs, max_tokens=200_000)
        log["adapt"] = json.loads((base_dir / "engram.json").read_text())["info"]
    else:
        spec = EngramSpec(hidden_size=1024, layer_ids=tuple(int(x) for x in args.layers.split(",")),
                          max_ngram_size=4, n_heads=8, head_dim=64, vocab_size=65536, pad_token_id=151643,
                          norm_eps=1e-6)
        lm = EngramCausalLM.from_pretrained(args.model, spec=spec, device=args.device, cache_dir=".data/cache")
        train_stream = pack_token_stream(lm.tokenizer, train_docs, max_tokens=2_000_000)
        val_stream = pack_token_stream(lm.tokenizer, val_docs, max_tokens=200_000)
        rep = adapt_base_engram(lm, train_stream, val_stream,
                                AdaptConfig(steps=args.adapt_steps, eval_every=100, eval_batches=8),
                                progress=lambda r: print("adapt", r, flush=True))
        lm.save_base(base_dir, info=rep)
        log["adapt"] = rep
    print(f"stage-1: ppl no-engram {log['adapt']['eval_ppl_no_engram']:.3f} -> engram "
          f"{log['adapt']['eval_ppl_engram']:.3f}", flush=True)

    facts = make_facts(args.facts, seed=0)
    samples = build_samples(facts)
    blocked = None
    if args.block_bg > 0 or args.block_shared > 0:
        from wikiengram.consolidation.trainer import encode_samples

        bg = NgramCounts.from_stream(lm.variant, train_stream) if args.block_bg > 0 else None
        tok = lm.tokenizer
        template = [tok("Question: ", add_special_tokens=False)["input_ids"],
                    tok("?\nAnswer:", add_special_tokens=False)["input_ids"]]
        enc = encode_samples(tok, samples)
        blocked, bstats = generic_rows(lm.variant, [i for i, _ in enc], [x.claim_id for x in samples], bg,
                                       args.block_bg, args.block_shared, always_block=template)
        log["blocked"] = bstats
        print("blocked", bstats, flush=True)

    ad_cfg = AdaptConfig(batch_size=8, seq_len=128)
    ppl_before = math.exp(lm_loss(lm, val_stream, ad_cfg, use_engram=True, n_batches=8))
    ppl_backbone = math.exp(lm_loss(lm, val_stream, ad_cfg, use_engram=False, n_batches=8))

    delta, ledger = lm.new_delta(), RowLedger()
    t0 = time.perf_counter()
    report = train_delta(lm, delta, samples,
                         DeltaTrainConfig(epochs=args.epochs, batch_size=args.batch, lr=args.lr,
                                          locality_weight=args.locality_weight),
                         ledger=ledger, blocked_rows=blocked,
                         replay_stream=train_stream if args.locality_weight > 0 else None,
                         progress=lambda r: print("delta", r, flush=True))
    synchronize(lm.device)
    log["delta_train"] = report.to_dict()
    log["delta_train"]["wall_seconds"] = time.perf_counter() - t0
    delta.save(out / "delta.safetensors")
    ledger.save(out / "ledger.json")
    lm.attach_delta(delta)
    ppl_after = math.exp(lm_loss(lm, val_stream, ad_cfg, use_engram=True, n_batches=8, use_delta=True))

    rng = random.Random(0)
    tq = [(QA_TEMPLATE.format(q=q), f.obj) for f in facts for q in f.train_questions]
    hq = [(QA_TEMPLATE.format(q=q), f.obj) for f in facts for q in f.heldout_questions]
    hs = [(statement_prefix(RELATIONS[f.relation]["statement"][2].format(s=f.subject, o=f.obj), f.obj), f.obj)
          for f in facts]
    wq = [(wiki_context(facts, f, 4, rng) + QA_TEMPLATE.format(q=q), f.obj) for f in facts for q in f.heldout_questions]

    def run(name, pairs, **kw):
        r = evaluate(lm, [p for p, _ in pairs], [o for _, o in pairs], **kw)
        if "ctx" in kw:
            kw.pop("ctx")
        print(f"{name:42s} acc={r['acc']:.3f} tokens={r['avg_prompt_tokens']:.0f} "
              f"s/q={r['sec_per_query']:.3f} e.g.={r['examples'][:2]}", flush=True)
        return r

    res = {}
    res["backbone/train_q"] = run("backbone only, trained questions", tq, use_engram=False)
    res["engram_base/train_q"] = run("engram base (no delta), trained questions", tq, use_delta=False)
    res["delta/train_q"] = run("engram + delta, trained questions", tq)
    res["delta/heldout_q"] = run("engram + delta, held-out paraphrases", hq)
    res["delta/heldout_statement"] = run("engram + delta, held-out statement", hs)
    res["wiki/heldout_q"] = run("LLMWiki context only, held-out paraphrases", wq, use_delta=False)
    ctx_tokens = [len(lm.tokenizer(p.split("Question:")[0], add_special_tokens=False)["input_ids"]) for p, _ in wq]
    res["hybrid_all/heldout_q"] = run("LLMWiki + delta on all tokens, held-out", wq)
    res["hybrid/heldout_q"] = run("LLMWiki + delta on question only, held-out", wq, ctx=ctx_tokens)
    general = [("What is the capital of France?", "Paris"), ("Who wrote Hamlet?", "Shakespeare"),
               ("What is the largest planet in the solar system?", "Jupiter"),
               ("What is the chemical symbol for gold?", "Au"), ("How many legs does a spider have?", "eight"),
               ("What is the boiling point of water in Celsius?", "100"),
               ("Who painted the Mona Lisa?", "Leonardo"), ("What is the capital of Japan?", "Tokyo")]
    gq = [(QA_TEMPLATE.format(q=q), a) for q, a in general]
    res["general_qa/no_delta"] = run("general QA, engram base (no delta)", gq, use_delta=False)
    res["general_qa/delta"] = run("general QA, engram + delta", gq)
    a = lm.generate([p for p, _ in gq], max_new_tokens=12, use_delta=False).texts
    b = lm.generate([p for p, _ in gq], max_new_tokens=12).texts
    res["general_qa/identical_outputs"] = sum(x == y for x, y in zip(a, b)) / len(a)
    print("general QA identical outputs with/without delta:", res["general_qa/identical_outputs"], flush=True)
    sanity = ["The capital of France is", "Water boils at a temperature of", "The author of Hamlet is"]
    res["sanity_base"] = lm.generate(sanity, max_new_tokens=8, use_delta=False).texts
    res["sanity_delta"] = lm.generate(sanity, max_new_tokens=8).texts
    print("sanity no-delta:", res["sanity_base"], "\nsanity delta   :", res["sanity_delta"], flush=True)
    log["results"] = res
    log["ppl"] = {"backbone": ppl_backbone, "engram_base": ppl_before, "engram_delta": ppl_after}
    log["delta_rows"] = delta.num_rows()
    log["delta_file_bytes"] = (out / "delta.safetensors").stat().st_size
    print(f"ppl backbone {ppl_backbone:.3f} | engram base {ppl_before:.3f} | engram+delta {ppl_after:.3f}")
    print(f"delta rows {delta.num_rows()} file {log['delta_file_bytes'] / 1e6:.1f} MB, "
          f"train {report.seconds:.0f}s, {report.tokens} tokens, epochs {report.epochs_run}")
    (out / "spike_results.json").write_text(json.dumps(log, indent=2, default=str))


if __name__ == "__main__":
    main()
