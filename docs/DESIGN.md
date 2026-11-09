# Design decisions

This page explains why WikiEngram looks the way it does. Facts were checked in November 2026; sources are
linked.

## 1. One framework for every Engram family

WikiEngram targets MoE models that ship an Engram-style conditional memory: hashed n-gram embedding tables
fused into the residual stream. Families can differ in hashing, table layout, rows per position and fusion.
The personal-memory machinery only needs to know which rows a position reads (as a pure function of token
ids), how those rows group into n-grams, and which module fuses them. That is the `EngramVariant` interface,
and each family implements it once.

* **The official repository** ([deepseek-ai/Engram](https://github.com/deepseek-ai/Engram), Apache-2.0)
  contains a forward-only demo. It mocks attention, MoE and mHC, and ships no training code and no weights.
* **[DeepSeek-V4.1-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash)** (MIT) is DeepSeek's
  public checkpoint with native Engram: 552B backbone + 196B Engram, Engram at layers 1 and 14, 2–4-grams,
  8 heads, head dim 256, FP8 tables with E8M0 scales, no ShortConv. Its `inference/engram.py` and
  `model.py` are the production semantics, so the first variant ports those rather than the demo.
* **DeepSeek-V4** has no Engram: its `tid2eid` is a hash router.
* **The paper** ([arXiv:2601.07372](https://arxiv.org/abs/2601.07372)) reports what matters for this
  project. With Engram suppressed, factual QA keeps only 29–44% of its score, so Engram is where parametric
  knowledge lives. Lookups are deterministic, so tables can be offloaded to host memory with a throughput
  cost under 3%.
* **Other families** are tracked in [NATIVE_MOE.md](NATIVE_MOE.md) once their modules are verified against
  primary sources.

**Decision:** a family-agnostic core behind `EngramVariant`. DeepSeek's Engram is the first variant, ported
bit-exactly and tested against the official code. A toy family in the tests keeps the core from depending
on DeepSeek details.

## 2. Which model, given the hardware?

Native Engram models are large MoE models that need hundreds of GB of memory. No small model with native
Engram exists. Community "Engram" models are re-implementations bolted onto other models. A frozen backbone
with an Engram retrofit has support in the literature (for example Engram Adapter, arXiv:2608.29327) and in
community models (a frozen Spark-1.7B with a 420M-parameter Engram).

**Decision:** two kinds of backend sharing one core.

1. **Retrofit, runnable today.** A frozen Qwen3 with an Engram variant attached by hooks. A base Engram is
   trained once with the backbone frozen and then frozen as well. Qwen3-0.6B-Base is the laptop default
   (Apache-2.0). It loads in transformers without remote code, and Qwen3-1.7B-Base shares its tokenizer.
2. **Native MoE backends, planned.** Each needs a trainable port of the family's MoE block, a table reader and
   a streaming trainer ([NATIVE_MOE.md](NATIVE_MOE.md)).

## 3. Which part of Engram is trained?

* **Shrink the base table?** No. That does not reduce training memory, because the table is already
  sparsely accessed and offloadable. It would also erase general knowledge the backbone relies on.
* **Train fusion parameters too?** No, by default. They are shared by every token, so changes to them cannot
  be undone per fact, and exact deletion would be lost.
* **A separate small table?** Its tiny address space collides constantly with general text, and its fusion
  layer would have to be learned from scratch.

**Decision:** a sparse delta over the base table's own row addresses (`row = base + delta`), stored as an
independent file. It gives exact rollback per row, small files, and the base table's huge address space
(few collisions with general text).

## 4. How rows are selected

The rows of generic n-grams are blocked. Those are n-grams frequent in general text, or shared by the samples
of several claims ("works at", "Answer:"). The first experiment on Qwen3-0.6B showed why this matters:
without blocking, the delta changed general continuations. One example: "The author of Hamlet is" lost its
answer. Blocking is counted per n-gram (by fingerprint), not per row, because in a hashed table every row is
shared by many colliding rare n-grams. This is the n-gram analogue of Sparse Memory Finetuning's TF-IDF slot
selection ([arXiv:2510.15103](https://arxiv.org/abs/2510.15103)).

## 5. How the two memories are fused

Measured on Qwen3-0.6B with 50 facts (held-out paraphrases, same run), wiki context alone scored 91%.
Adding the personal delta on every prompt token dropped that to 49%, because the memories of other entities
mentioned in the context fire and compete. The delta is therefore applied only to the question tokens. The
wiki context is read as it is, with the frozen base Engram.

Restricting the delta to question tokens raised accuracy to 70%, still below the wiki alone (91%): when the
answer is in the context, memory can only compete with it. Hybrid mode therefore routes each question. If
Engram verifiably recalls the most relevant claim (at its current version), that claim leaves the prompt and
the delta answers; otherwise the wiki answers with the delta off. Pending wiki edits are applied to memory
before answering, by removing or masking rows (no training needed).

## 6. Optimizer

The paper trains Engram embeddings with Adam at 5× the base learning rate and no weight decay. V4.1 replaces
Adam with Nesterov momentum plus Sinkhorn balancing, to save optimizer memory. For a personal delta the
optimizer state only exists for touched rows, so lazy Adam is cheap and fine. `torch.optim.SparseAdam` works
on MPS (torch 2.15), but its cost grows with table size; a hand-written row-sparse Adam does not.

## 7. LLMWiki

"LLM Wiki" is Andrej Karpathy's idea file
([gist](https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f), no license, no code). The most
popular implementation is GPL-3.0 and written in TypeScript/Rust, and none is a Python library with
claim-level IDs, versions and change events.

**Decision:** implement the pattern natively (raw sources, LLM-maintained pages, `index.md`, `log.md`, a
schema file, plus ingest / query / lint operations), and add a claim store that consolidation and sync
can consume. Retrieval is lexical (BM25 + entity match); there is deliberately no vector index.

## 8. Strata

[Strata](https://github.com/Niko1221/Strata) (MIT) runs V4.1-Flash on consumer GPUs. It uses 2-bit expert
sets, keeps a 4-bit Engram table on SSD, and streams experts from RAM or SSD. It is inference-only: it can
score a sequence (teacher-forced), but it cannot produce gradients. Its streaming strategy is the template
for the planned V4.1 trainer, and a small patch to its Engram row reader is the planned inference path for
personal deltas.
