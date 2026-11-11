# Native Engram MoE backends

> **Status.** **Qwen3.8-Flash-Next: implemented.** It runs and trains straight from its FP8 release on one
> 32 GB GPU ([below](#qwen38-flash-next-ple-variant-qwen-ple); measured in the README's results).
> **DeepSeek-V4.1-Flash: planned** (no trainable implementation of its backbone exists yet).
> WikiEngram is meant for every MoE model that ships an Engram-style conditional memory. The personal-memory
> machinery works with any Engram family through the `EngramVariant` interface
> (see [ARCHITECTURE.md](ARCHITECTURE.md#engram-variants-srcwikiengramengramvariantpy)). What a *native* backend adds is a trainable,
> memory-efficient forward and backward pass through the frozen MoE backbone, plus a reader for the model's own
> table format. This page records the facts and the plan, family by family.

## What a native backend needs (any family)

1. **The family's Engram variant**: row addressing (hashing), row grouping, and the fusion module, checked
   bit-exactly against the reference code. For DeepSeek this exists and is tested.
2. **A table reader for the native storage format** (for example FP8 with block scales, or GGUF quant types).
   Rows are only gathered for the positions in a batch.
3. **A trainable port of the frozen MoE block** (attention, routing, experts) that autograd can
   differentiate. Frozen weights are dequantized on the fly, and only activation gradients are computed.
4. **Layer and expert streaming**, so a single consumer GPU can drive a backbone of hundreds of GB from RAM
   and SSD (see the estimates below).
5. **Inference with the personal delta**: add delta rows after the table lookup, in the engine people
   actually run (for example a small patch to Strata's or llama.cpp's Engram row reader).

## DeepSeek-V4.1-Flash (planned)

### Facts (checked 2026-11-09)

* DeepSeek-V4.1-Flash and V4.1-Flash-1017 are the only public DeepSeek checkpoints with native Engram
  weights. The repository and weights are MIT licensed.
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

### Why the size of the Engram table is not the training bottleneck

Personal consolidation updates only the delta rows that a batch touches, from a few MB to a few hundred MB.
The base table never needs to be in GPU memory: rows are addressed by token ids, so they can be prefetched
from RAM or SSD before the forward pass, as the Engram paper does. The cost comes from the frozen backbone.
The loss gradient must flow from the output back to layer 1 (or 14), so every training step needs a forward
and a backward pass through the 40 MoE layers (about 7.7 GB per layer at FP4/FP8). Shrinking the Engram table
does not change that, and it would also remove the general knowledge the backbone was pretrained with.

### Plan: a streaming trainer on one consumer GPU

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

#### Rough estimates (to be replaced by measurements)

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

The retrofit backend does not replace native models. It is the same Engram variant code, with the same
hashing, layout, fusion and training rules, attached to a small frozen model. That lets the whole pipeline
run, be tested and be benchmarked on a laptop and in CI today. A native backend reuses every component except
the backbone and the table reader.

## Other Engram-family MoE models

### Qwen3.8-Flash-Next ("PLE"), variant `qwen-ple`

Verified against transformers `models/qwen4_exp/modeling_qwen4_exp.py` (Apache-2.0) and the checkpoint's
config and safetensors headers, 2026-11-10.

* **Model.** About 180B MoE (125B backbone + 51B n-gram table + 4B MTP), about 6B active. The weights are
  under the Qwen Community License 1.0, which has commercial-use limits.
* **Module.** `Qwen4ExpTextPLELayer`, applied before block 1 (0-based; the config's `ple_layer_ids [2]` is
  1-based). It acts on 4 widened residual streams of width 2560.
* **Addressing.** The same family as DeepSeek's Engram, with these differences:
  * Rolling multiply-XOR over orders 2–3 with 8 heads, i.e. 16 rows of 160 dims per token. Each head takes
    the hash modulo its own prime, plus an offset.
  * No token compression: raw ids from a vocab of 248,320.
  * Multipliers come from splitmix64, seeded with `seed + 10007 · layer_index`. They are stored in the
    checkpoint as `(23703573157769, 20109073645365, 8052911324071)`, and `tests/test_variant_parity.py`
    reproduces them.
  * The primes are the k-th primes ≥ 20,000,000.
  * The look-back is filled with `<|endoftext|>` (248044) and is cut after one.
* **Table.** 320,001,536 × 160 rows in BF16, stored as 128 shards
  `model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_{0..127}.weight` of 2,500,012 rows
  each. The FP8 repository stores E4M3 values with one global scale.
* **Fusion.** The same signed-sqrt sigmoid gate as DeepSeek, but:
  * the RMSNorms are zero-centred, grouped per stream (`x · (1 + w)`);
  * the key and value projections are separate;
  * after the gate comes `SiLU(depthwise causal conv(kernel 4, dilation 3))` over the normalized gated values.

  A row's update therefore reaches positions t, t+3, t+6 and t+9. Decoding keeps a 9-position conv state.
* **Engines.** transformers, vLLM (CPU and disk offload of the table), SGLang, llama.cpp, Strata, and
  NeMo-Automodel for training.
* **Running it from the checkpoint.** `NativeEngramLM.from_checkpoint(path)` (`backends/checkpoint.py`) runs
  the FP8 release without loading it. The backbone code is transformers' own `Qwen4Exp*` modules, kept frozen.
  * About 2.2B non-expert parameters stay resident on the accelerator, dequantized from FP8.
  * Routed experts are read from the files on demand: 10 of 512 per token. They are dequantized on the device
    (E4M3 values with 128×128 block scales) and run through transformers' own expert kernel.
  * On Linux the experts are read with `O_DIRECT`, around the page cache, by 48 threads. A training step or a
    batched generation step touches nearly all 121 GB of experts, more than most machines can cache. Measured
    in a container with a 92 GB memory limit on an NVMe RAID: 2–3 GB/s through the page cache, 7–18 GB/s direct.
    The raw bytes reach the GPU 32 experts at a time, so a layer's experts are never there twice (FP8 and BF16).
  * The table's rows are read straight from the 128 shards (`ShardedTable`: E4M3 times the global scale, or
    BF16 for the BF16 release).
  * Consolidation uses the streamed backward, so peak memory is the resident weights plus one layer.
  * The full-attention layers' QSA indexer keeps each query's 512 best blocks of 4 keys (a 2,048-token
    budget). Up to that length it keeps every visible token, so its mask is the attention mask itself;
    `fast_qsa_indexer` returns it directly instead of running transformers' Python loop over every
    (sequence, query) pair, which took about half of each training step (exact: same mask, tested).
  * Install the flash-linear-attention kernels (`pip install "fla-core>=0.5"`; 0.6 needs a Triton with Gluon,
    0.5.2 works with Triton 3.4). Without them transformers falls back to a PyTorch implementation of the
    DeltaNet layers that is much slower and needs about 3× the memory: 9.5 GiB instead of 3.3 GiB for one layer's
    forward and backward at 150 sequences × 100 tokens (measured on an RTX 5090; outputs and gradients agree to
    bf16 precision, relative error about 4·10⁻³). On a 32 GB card this sets the training batch: 168 samples per
    step with the kernels, about 100 without.

  Synthetic checkpoints in the release format (per-expert FP8 tensors with block scales, a sharded FP8 table,
  the `model.language_model.` prefix, several files with an index) match the same weights held in memory
  bit-exactly on CPU, for logits, generation and delta gradients (`tests/test_checkpoint_streaming.py`).

### DeepSeek-V4.1-Flash-1017

The config and inference code are byte-identical to V4.1-Flash, but the Engram values were retrained:
shards 47 and 48 differ while their layouts are the same. A personal delta is only valid on the exact base
it was trained against. WikiEngram records the base identity in every delta and refuses a mismatch; native
backends will also record the table shards' sha256.

### Not supported

Meituan LongCat (Flash-Lite, Lite-Sparse, Next, 2.0) adds a hashed n-gram embedding at the input instead of
using an Engram-style module. It has a polynomial hash modulo non-prime sizes, a projection per table and no
gate. It is out of scope.

No other public MoE family with an Engram-style module was found (configs of about 660 repos from 2026 were
checked).
