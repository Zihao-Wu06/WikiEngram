<!-- wikiengram:generated from .wikiengram/wiki.sqlite; edits here are overwritten, see SCHEMA.md -->
# Strata

40 active claims · 35 historical versions

## Claims

- In Strata, Qwen3.8-Flash-Next is a team of 24,576 small experts. — `c_2db58960d7` v2 · confidence 1.00 · sources: [Strata docs/HOW_IT_WORKS.md](../raw/s_e8475372d2.md)
- Strata makes big models fit by sharing the work across your whole PC. — `c_f1f5e4f12e` v2 · confidence 1.00 · sources: [Strata docs/HOW_IT_WORKS.md](../raw/s_e8475372d2.md)
- Strata's RAM holds all 24,576 experts, pinned. — `c_bd43282faa` v2 · confidence 1.00 · sources: [Strata docs/HOW_IT_WORKS.md](../raw/s_e8475372d2.md)
- The n-gram table that Strata keeps on the SSD is 28.8 GB. — `c_fdef80f50f` v2 · confidence 1.00 · sources: [Strata docs/HOW_IT_WORKS.md](../raw/s_e8475372d2.md)
- Strata writes text by guess then check: a small helper guesses the next words and the big model checks them all at once. — `c_0743d108d9` v2 · confidence 1.00 · sources: [Strata docs/HOW_IT_WORKS.md](../raw/s_e8475372d2.md)
- Strata's app and Terminal are at version 0.2.13. — `c_0c52167d23` v2 · confidence 1.00 · sources: [Strata docs/INSTALL.md](../raw/s_33db960c93.md)
- Strata has one installer per system, not one per graphics card. — `c_f5c7525072` v2 · confidence 1.00 · sources: [Strata docs/AI_REFERENCE.md](../raw/s_3d3ef14e23.md)
- Strata runs three model families: Qwen3.8-Flash-Next, Qwen3.6-35B-A3B and DeepSeekV4.1-Flash-1017. — `c_50191a5857` v2 · confidence 1.00 · sources: [Strata README.md](../raw/s_0e43837bc3.md)
- Strata is open source under the MIT License. — `c_95dfe441d8` v2 · confidence 1.00 · sources: [Strata README.md](../raw/s_0e43837bc3.md), [Strata docs/HOW_IT_WORKS.md](../raw/s_e8475372d2.md)
- In Strata, an expert cache fills the rest of the GPU's memory with the most-used experts. — `c_8dc4df34d8` v2 · confidence 1.00 · sources: [Strata docs/HOW_IT_WORKS.md](../raw/s_e8475372d2.md)
- In Strata, the experts that are not on the GPU are computed by the CPU, in place in RAM. — `c_9fbb8077ad` v2 · confidence 1.00 · sources: [Strata docs/HOW_IT_WORKS.md](../raw/s_e8475372d2.md)
- Strata reads long texts in pieces of up to 8,192 tokens at a time. — `c_2d0f906c03` v2 · confidence 1.00 · sources: [Strata docs/HOW_IT_WORKS.md](../raw/s_e8475372d2.md)
- Strata works the same on NVIDIA and AMD cards, with the same engine compiled for each. — `c_565a8b2509` v2 · confidence 1.00 · sources: [Strata docs/HOW_IT_WORKS.md](../raw/s_e8475372d2.md)
- In Strata, the model's own MTP layer drafts up to 3 tokens at a time. — `c_c216ca8ec9` v2 · confidence 1.00 · sources: [Strata docs/HOW_IT_WORKS.md](../raw/s_e8475372d2.md)
- Strata's expert cache adapts to the conversation while you chat. — `c_42ba197c5b` v2 · confidence 1.00 · sources: [Strata docs/HOW_IT_WORKS.md](../raw/s_e8475372d2.md)
- After the first message, Strata keeps the conversation and reads only what is new, so follow-ups start in seconds. — `c_ebb1a29870` v2 · confidence 1.00 · sources: [Strata docs/HOW_IT_WORKS.md](../raw/s_e8475372d2.md)
- Intel Macs are not supported by Strata's Mac package. — `c_8863fff9d5` v2 · confidence 1.00 · sources: [Strata docs/INSTALL.md](../raw/s_33db960c93.md)
- Strata has no desktop app for ARM64 Linux, such as the NVIDIA DGX Spark; it uses the Terminal there. — `c_138603eb98` v2 · confidence 1.00 · sources: [Strata docs/INSTALL.md](../raw/s_33db960c93.md)
- Strata should run from an SSD, with room for helper files and model preparation. — `c_df2c0b17ad` v2 · confidence 1.00 · sources: [Strata docs/INSTALL.md](../raw/s_33db960c93.md)
- Strata's packaged GPU runtimes do not replace hardware drivers. — `c_da60b5f582` v2 · confidence 1.00 · sources: [Strata docs/INSTALL.md](../raw/s_33db960c93.md)
- The ISTA-DAS Coder set of DeepSeekV4.1-Flash-1017 needs about 197 GB of free disk in Strata. — `c_d01a866970` v5 · confidence 1.00 · sources: [Strata README.md](../raw/s_0e43837bc3.md)
- Strata reads a long document or code base at over 1,000 tokens per second. — `c_6497644b55` v2 · confidence 1.00 · sources: [Strata docs/HOW_IT_WORKS.md](../raw/s_e8475372d2.md)
- When a reply repeats the context, as in code edits, Strata drafts tokens from the earlier copy with prompt lookup. — `c_1d57a8cdc8` v2 · confidence 1.00 · sources: [Strata docs/HOW_IT_WORKS.md](../raw/s_e8475372d2.md)
- In Strata, the experts of the next layer stream to the GPU over PCIe while the current layer's attention runs. — `c_32c449e789` v2 · confidence 1.00 · sources: [Strata docs/HOW_IT_WORKS.md](../raw/s_e8475372d2.md)
- Strata checks the drafted tokens with one pass over all 48 layers. — `c_a973bacbdf` v2 · confidence 1.00 · sources: [Strata docs/HOW_IT_WORKS.md](../raw/s_e8475372d2.md)
- Strata's CPU computes experts with AVX-512 / AVX2 kernels. — `c_b84d01cbce` v2 · confidence 1.00 · sources: [Strata docs/HOW_IT_WORKS.md](../raw/s_e8475372d2.md)
- For the i-quants, Strata computes experts with ggml's kernels. — `c_1d73bb6e97` v2 · confidence 1.00 · sources: [Strata docs/HOW_IT_WORKS.md](../raw/s_e8475372d2.md)
- On a PC with little RAM and a big graphics card, Strata's low-RAM mode keeps in RAM only what the GPU does not hold. — `c_70a758c63e` v2 · confidence 1.00 · sources: [Strata docs/HOW_IT_WORKS.md](../raw/s_e8475372d2.md)
- Prompt lookup makes code edits 6-11% faster in Strata. — `c_da6a8fe7ae` v2 · confidence 1.00 · sources: [Strata docs/HOW_IT_WORKS.md](../raw/s_e8475372d2.md)
- Older NVIDIA cards use Strata's CUDA 12 engine. — `c_7065e1914e` v2 · confidence 1.00 · sources: [Strata docs/INSTALL.md](../raw/s_33db960c93.md)
- If the engine file is already downloaded, choosing I already have the file in the Strata app installs it without internet. — `c_73bba73d54` v2 · confidence 1.00 · sources: [Strata docs/INSTALL.md](../raw/s_33db960c93.md)
- The Strata Mac file keeps its small Metal engine inside. — `c_653d0a2867` v2 · confidence 1.00 · sources: [Strata docs/INSTALL.md](../raw/s_33db960c93.md)
- In Strata, every extra GB of VRAM holds about 700 more experts. — `c_ddfc4f011d` v1 · confidence 0.95
- Strata's guess then check gives the same quality answer 1.6-1.8x sooner. — `c_d7afe7d550` v1 · confidence 0.95
- In Strata, one check pass accepts 2.4-3.2 tokens on average. — `c_006902236a` v1 · confidence 0.95
- Strata's prompt lookup drafts up to 5 tokens from the earlier copy. — `c_925219e5cb` v1 · confidence 0.95
- Strata keeps the n-gram table on the SSD and reads only a few rows of it per token. — `c_3ba120dbd5` v1 · confidence 0.95
- From a 64K context, Strata keeps only the most-read part of the KV cache on the GPU and streams the rest from RAM. — `c_6f230583f8` v1 · confidence 0.95
- Strata is built with parts of llama.cpp / ggml. — `c_23886e8ab1` v1 · confidence 0.95
- Strata takes ideas from Splash, ninfer and HyperQwen. — `c_30f7340cde` v1 · confidence 0.95

## Related

- [[desktop-chats-and-host-state]]

## History

- ~~Strata has models Qwen3.8-Flash-Next.~~ — `c_50191a5857` v1 · superseded · 2026-11-10 03:03 · ingest:s_0e43837bc3
- ~~Strata requires disk space about 245 GB for the 1.75 bpw DeepSeekV4.1-Flash-1017 model.~~ — `c_d01a866970` v1 · superseded · 2026-11-10 03:03 · ingest:s_0e43837bc3
- ~~Strata requires disk space about 265 GB for the 2.00 bpw DeepSeekV4.1-Flash-1017 model.~~ — `c_d01a866970` v2 · superseded · 2026-11-10 03:03 · ingest:s_0e43837bc3
- ~~Strata requires disk space about 280 GB for the 2.25 bpw DeepSeekV4.1-Flash-1017 model.~~ — `c_d01a866970` v3 · superseded · 2026-11-10 03:03 · ingest:s_0e43837bc3
- ~~Strata requires disk space about 197 GB for the ISTA-DAS Coder DeepSeekV4.1-Flash-1017 model.~~ — `c_d01a866970` v4 · superseded · 2026-11-10 03:03 · ingest:s_0e43837bc3
- ~~Strata has license MIT license.~~ — `c_95dfe441d8` v1 · superseded · 2026-11-10 03:03 · ingest:s_0e43837bc3
- ~~Strata shares the work across the whole PC.~~ — `c_f1f5e4f12e` v1 · superseded · 2026-11-10 03:05 · ingest:s_e8475372d2
- ~~Strata's uses experts in model is 24576.~~ — `c_2db58960d7` v1 · superseded · 2026-11-10 03:05 · ingest:s_e8475372d2
- ~~Strata keeps the most used experts on the GPU.~~ — `c_8dc4df34d8` v1 · superseded · 2026-11-10 03:05 · ingest:s_e8475372d2
- ~~Strata's uses ram for experts is all_24576_experts.~~ — `c_bd43282faa` v1 · superseded · 2026-11-10 03:05 · ingest:s_e8475372d2
- ~~Strata uses the processor for experts not on the GPU.~~ — `c_9fbb8077ad` v1 · superseded · 2026-11-10 03:05 · ingest:s_e8475372d2
- ~~Strata stores a 28.8 GB n-gram table on the SSD.~~ — `c_fdef80f50f` v1 · superseded · 2026-11-10 03:05 · ingest:s_e8475372d2
- ~~Strata's reads long texts in pieces is 8192_tokens.~~ — `c_2d0f906c03` v1 · superseded · 2026-11-10 03:05 · ingest:s_e8475372d2
- ~~Strata's reads text at speed is over_1000_tokens_per_second.~~ — `c_6497644b55` v1 · superseded · 2026-11-10 03:05 · ingest:s_e8475372d2
- ~~Strata uses a guess then check method to generate text.~~ — `c_0743d108d9` v1 · superseded · 2026-11-10 03:05 · ingest:s_e8475372d2
- ~~Strata uses prompt lookup to repeat context when needed.~~ — `c_1d57a8cdc8` v1 · superseded · 2026-11-10 03:05 · ingest:s_e8475372d2
- ~~Strata's streams experts over pci e is pci_e.~~ — `c_32c449e789` v1 · superseded · 2026-11-10 03:05 · ingest:s_e8475372d2
- ~~Strata works on both NVIDIA and AMD cards.~~ — `c_565a8b2509` v1 · superseded · 2026-11-10 03:05 · ingest:s_e8475372d2
- ~~Strata uses the MTP layer to draft up to 3 tokens.~~ — `c_c216ca8ec9` v1 · superseded · 2026-11-10 03:05 · ingest:s_e8475372d2
- ~~Strata checks drafts with one pass over all 48 layers.~~ — `c_a973bacbdf` v1 · superseded · 2026-11-10 03:05 · ingest:s_e8475372d2
- ~~Strata adapts the expert cache to the conversation while chatting.~~ — `c_42ba197c5b` v1 · superseded · 2026-11-10 03:05 · ingest:s_e8475372d2
- ~~Strata's uses avx512 kernels for cpu computation is avx512_kernels.~~ — `c_b84d01cbce` v1 · superseded · 2026-11-10 03:05 · ingest:s_e8475372d2
- ~~Strata uses ggml's for the i-quants.~~ — `c_1d73bb6e97` v1 · superseded · 2026-11-10 03:05 · ingest:s_e8475372d2
- ~~Strata keeps only what the GPU does not hold in low-RAM mode.~~ — `c_70a758c63e` v1 · superseded · 2026-11-10 03:05 · ingest:s_e8475372d2
- ~~Strata's improves code edit speed by 6 to 11 percent is 6_to_11_percent.~~ — `c_da6a8fe7ae` v1 · superseded · 2026-11-10 03:05 · ingest:s_e8475372d2
- ~~Strata reads only new content in follow-ups.~~ — `c_ebb1a29870` v1 · superseded · 2026-11-10 03:05 · ingest:s_e8475372d2
- ~~Strata has version 0.2.13.~~ — `c_0c52167d23` v1 · superseded · 2026-11-10 03:07 · ingest:s_33db960c93
- ~~Strata downloads the CUDA 12 engine for older NVIDIA cards.~~ — `c_7065e1914e` v1 · superseded · 2026-11-10 03:07 · ingest:s_33db960c93
- ~~Strata can use an existing engine file without internet.~~ — `c_73bba73d54` v1 · superseded · 2026-11-10 03:07 · ingest:s_33db960c93
- ~~Strata has a small Metal engine on Mac.~~ — `c_653d0a2867` v1 · superseded · 2026-11-10 03:07 · ingest:s_33db960c93
- ~~Strata does not support Intel Macs.~~ — `c_8863fff9d5` v1 · superseded · 2026-11-10 03:07 · ingest:s_33db960c93
- ~~Strata does not have a desktop app for ARM64 Linux.~~ — `c_138603eb98` v1 · superseded · 2026-11-10 03:07 · ingest:s_33db960c93
- ~~Strata requires an SSD for storage.~~ — `c_df2c0b17ad` v1 · superseded · 2026-11-10 03:07 · ingest:s_33db960c93
- ~~Strata packaged GPU runtimes do not replace hardware drivers.~~ — `c_da60b5f582` v1 · superseded · 2026-11-10 03:07 · ingest:s_33db960c93
- ~~Strata has one app installer per system.~~ — `c_f5c7525072` v1 · superseded · 2026-11-10 03:13 · ingest:s_3d3ef14e23
