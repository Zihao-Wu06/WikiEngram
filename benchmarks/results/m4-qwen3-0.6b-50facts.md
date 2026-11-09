## WikiEngram benchmark: 50 fictitious facts, Qwen/Qwen3-0.6B-Base on `mps`

| Memory | Trained question | Held-out paraphrase | Prompt tokens / query | Median latency (TTFT / total) | Peak memory (inference) |
|---|---|---|---|---|---|
| Backbone only (no memory) | 0.0% | 0.0% | 18 | 46 ms / 408 ms | 3536 MiB |
| LLMWiki only | 100.0% | 98.0% | 87 | 88 ms / 211 ms | 4616 MiB |
| Engram only | 96.0% | 56.0% | 18 | 52 ms / 166 ms | 4562 MiB |
| LLMWiki + Engram | 100.0% | 94.0% | 62 | 54 ms / 194 ms | 4618 MiB |

Consolidation: 116 s, 58100 tokens, 20 epochs, peak memory 3532 MiB; personal delta 47402 rows (33.3 MB file); backbone and Engram fusion parameters unchanged: **True**.

**After editing / deleting facts in the wiki and syncing memory**

| Check | LLMWiki only | Engram only | LLMWiki + Engram |
|---|---|---|---|
| Edited facts: new value answered | 100.0% | 0.0% | 100.0% |
| Edited facts: old value still produced (lower is better) | - | 0.0% | 0.0% |
| Deleted facts still produced (lower is better) | - | 0.0% | 0.0% |
| Untouched facts retained | - | 100.0% | 100.0% |

After re-consolidating the edited claims, Engram alone answers the new value for 100.0% of edited facts and keeps 100.0% of untouched ones. Sync took 19 s.

Environment: torch 2.15.0, macOS-27.0.1-arm64-arm-64bit-Mach-O, Python 3.13.5.
