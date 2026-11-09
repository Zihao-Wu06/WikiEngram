# DeepSeek-V4.1-Flash: native Engram backend (roadmap)

> **Status: planned, not implemented.** This page records the facts and the plan behind the native backend.
> What already works for V4.1 is listed in [ARCHITECTURE.md](ARCHITECTURE.md#deepseek-v41-flash-what-exists-and-what-does-not):
> bit-exact hashing, layout and compression, config parsing, and reading native FP8 Engram rows.

## Facts (checked 2026-11-09)

* DeepSeek-V4.1-Flash and V4.1-Flash-1017 are the only public checkpoints with native Engram weights. The
  repository and weights are MIT licensed.
  * Backbone: 552B parameters, 40 layers, 384 routed experts with top-6 routing, mHC with `hc_mult = 4`.
  * Engram: 196B parameters at layers 1 and 14. Each layer has a `384M × 256` FP8 table with E8M0
    per-32 scales. The two tables take about 203 GB.
  * Checkpoint size is about 510 GB.
* DeepSeek-V4 (Flash/Pro) has no Engram; its `tid2eid` table is a static hash router.
* No framework trains a frozen V4.1 backbone with an Engram-only update. NVIDIA NeMo AutoModel's recipes are
  full fine-tunes on 64 GPUs or more. DeepSeek's `inference/` code is inference-only, with CUDA kernels.
* Consumer-hardware inference exists. The 2-bit sets (ISTA-DAS IQ1_M / IQ2_XXS / IQ2_XS, 121–155 GB) plus a
  4-bit IQ4_NL Engram table (111 GB) run in [Strata](https://github.com/Niko1221/Strata) (MIT) and
  llama.cpp. Strata streams what does not fit in memory from an NVMe SSD; on a 12 GB GPU with 46–64 GB of RAM
  it generates about 5–10 tokens/s.

## Why the size of the Engram table is not the training bottleneck

Personal consolidation updates only the delta rows that a batch touches, from a few MB to a few hundred MB.
The base table never needs to be in GPU memory: rows are addressed by token ids, so they can be prefetched
from RAM or SSD before the forward pass, as the Engram paper does. The cost comes from the frozen backbone.
The loss gradient must flow from the output back to layer 1 (or 14), so every training step needs a forward
and a backward pass through the 40 MoE layers (about 7.7 GB per layer at FP4/FP8). Shrinking the Engram table
does not change that, and it would also remove the general knowledge the backbone was pretrained with.

## Plan: a streaming trainer on one consumer GPU

The target is one RTX 5090 (32 GB) with 90 GB of RAM and at least 400 GB of local NVMe.

1. **Weights.** Use the same quantized files as inference (the ISTA-DAS 2-bit set plus the IQ4_NL Engram
   table), so the delta is trained against exactly the model it is used with.
2. **Trainable port.** Write a pure-PyTorch forward pass of the V4.1 block (attention, mHC Sinkhorn mixing,
   MoE) that autograd can differentiate. Frozen weights are dequantized per expert on the GPU. The backward
   pass computes activation gradients only, never weight gradients.
3. **Layer streaming.** Load weights layer by layer from RAM or SSD with prefetch. Checkpoint layer-boundary
   activations and recompute inside each layer during the backward pass.
4. **Parity.** Compare teacher-forced logits against Strata's `--score` output on the same files, before any
   training.
5. **Inference with the delta.** Add a small patch to Strata's Engram row reader (`src/dsv41/ds4_engram_io.hpp`)
   that adds the personal delta rows after dequantization, plus the matching llama.cpp change.
6. **Benchmark.** Run the same `wikiengram bench` protocol as the retrofit backend.

### Rough estimates (to be replaced by measurements)

Assumptions: 2-bit backbone of about 138 GB, about half of it cached in 90 GB of RAM, the rest streamed from
NVMe at about 3 GB/s, and about 16k tokens per step. That gives roughly one minute per step, or 200–300
tokens/s.

| Workload | RTX 5090 + 90 GB (estimate) |
|---|---|
| 50 new claims (a nightly incremental consolidation) | 8–25 min |
| 1,000 claims (first consolidation) | 1–5 h |

These are back-of-the-envelope numbers. The biggest unknown is how many training tokens a claim needs, which
is measured first on the retrofit backend.

## Why the retrofit backend exists

The retrofit backend does not replace V4.1. It is the same Engram code, with the same hashing, layout, fusion
and training rules, attached to a small frozen model. That lets the whole pipeline run, be tested and be
benchmarked on a laptop and in CI today. The V4.1 backend reuses every component except the backbone and the
table reader.
