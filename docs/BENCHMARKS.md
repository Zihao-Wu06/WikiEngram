# Benchmark protocol

`wikiengram bench` measures the three memory configurations the project is about, plus a no-memory baseline.
Everything goes through the public `WikiEngram` API, and the numbers come from real training and real
generation.

## Data

* `N` **fictitious facts** (`bench/synthetic.py`, deterministic seed). Subjects, companies, towns, pet names,
  dishes and instruments are generated from syllables, so a pretrained model cannot know them in advance.
  There are five relations: `works_at`, `lives_in`, `pet_name`, `favorite_dish`, `instrument`.
* Each relation has 3 statement templates and 4 question templates. The facts document uses the first
  statement template. Consolidation trains on **2 questions** plus the statement. A **3rd question** is used
  only for verification. The **4th question is held out**: it is used neither in training nor in
  verification, and it is what the "held-out paraphrase" column reports.
* The document is ingested through the LLMWiki rule extractor. If extraction misses a fact, the benchmark
  fails loudly rather than skipping it.

## Conditions

| Condition | Engram | Personal delta | Wiki context in prompt |
|---|---|---|---|
| Backbone only | off | off | no |
| LLMWiki only | base (frozen) | off | top-k BM25 claims |
| Engram only | base (frozen) | on | no |
| LLMWiki + Engram | base (frozen) | routed per question: on (question tokens only) if Engram verifiably holds the top claim, else off | Engram route: none (the published synthetic runs used an earlier rule that kept unverified claims about the same subject); wiki route: top-k claims |

## Metrics

* **Accuracy**: the generated answer, cut at the first newline and capped at 16 new tokens, contains the
  object string. Objects are invented strings, so false positives are practically impossible.
* **Prompt tokens / query**: the token cost of explicit memory.
* **Latency**: median time to first token and median total time at batch size 1, over the first 10 held-out
  questions. Batched throughput is reported separately in `results.json`.
* **Peak memory**: on CUDA, `torch.cuda.max_memory_allocated`. On Apple MPS, the driver's allocated memory,
  polled after every batch. That value includes allocator caches, so it is an upper bound.
* **Consolidation**: wall time, training tokens, epochs, peak memory, delta rows and file size, and a
  byte-exact check that the backbone and Engram fusion parameters did not change.
* **Update phase**:
  * A fraction of facts is corrected (new invented object) and another fraction deleted in the wiki. Then
    `sync` runs.
  * Measured afterwards: new value answered (per mode); old value still produced by Engram, where lower is
    better; deleted facts still produced, where lower is better; untouched facts retained.
  * Edited claims are then re-consolidated and measured again.

## Reproducing

```bash
wikiengram bench --facts 50 --model Qwen/Qwen3-0.6B-Base --out benchmark-results
```

```bash
wikiengram bench --facts 200 --model Qwen/Qwen3-1.7B-Base --out benchmark-results-1.7b
```

Pass `--base <dir>` to reuse a trained base Engram (the `engram/base` folder of a workspace) and skip stage 1.
`results.json` holds every number, including per-mode examples; `results.md` is the table in the README.

## Published results

| Run | Hardware | Model | Facts | Data |
|---|---|---|---|---|
| Reference run for the README | Apple M4, 32 GB (MPS) | Qwen3-0.6B-Base | 50 | [json](../benchmarks/results/m4-qwen3-0.6b-50facts.json) · [table](../benchmarks/results/m4-qwen3-0.6b-50facts.md) |
| Scaling, 3 seeds | RTX 5090, 32 GB | Qwen3-0.6B-Base | 50 | [seed 0](../benchmarks/results/rtx5090-qwen3-0.6b-50facts-seed0.json) · [1](../benchmarks/results/rtx5090-qwen3-0.6b-50facts-seed1.json) · [2](../benchmarks/results/rtx5090-qwen3-0.6b-50facts-seed2.json) |
| Scaling, 3 seeds | RTX 5090, 32 GB | Qwen3-1.7B-Base | 200 | [seed 0](../benchmarks/results/rtx5090-qwen3-1.7b-200facts-seed0.json) · [1](../benchmarks/results/rtx5090-qwen3-1.7b-200facts-seed1.json) · [2](../benchmarks/results/rtx5090-qwen3-1.7b-200facts-seed2.json) |
| Scaling | RTX 5090, 32 GB | Qwen3-1.7B-Base | 500 | [json](../benchmarks/results/rtx5090-qwen3-1.7b-500facts.json) |
| Scaling, with edits and deletions | RTX 5090, 32 GB | Qwen3-1.7B-Base | 1,000 | [json](../benchmarks/results/rtx5090-qwen3-1.7b-1000facts.json) |
| Batch-size check (batch 64, not in the README table) | RTX 5090, 32 GB | Qwen3-1.7B-Base | 200 | [json](../benchmarks/results/rtx5090-qwen3-1.7b-200facts-batch64.json) |

New results are welcome. Please include the hardware, the model, `--facts`, and the `results.json` file.
