# Example: Strata's documentation as long-term memory for Qwen3.8-Flash-Next

A complete WikiEngram memory built from a real, public project: the documentation of
[Strata](https://github.com/Niko1221/Strata) (MIT), an engine that runs large MoE models on consumer PCs.
The knowledge is extracted by a local LLM, reviewed, trained into the Engram memory of **Qwen3.8-Flash-Next**,
and served to an AI agent through the model API and the LLMWiki MCP server. The results are in the
[main README](../../README.md#results-a-real-projects-memory-in-qwen38-flash-next).

| File | What it is |
|---|---|
| `source/` | The documents, from Strata commit `504b143` (see `source/SNAPSHOT_COMMIT`), with Strata's MIT license |
| `build_memory.py` | Builds the workspace: `init`, `ingest`, `review`, `questions`, `finalize`, `plan`, `export` |
| `review_rules.py` | First review pass: flags markup, invented identifiers, overlong answers and duplicates |
| `review.json` | The full review applied to the extracted claims, one entry per decision, each with its reason |
| `evaluate.py` | The measurements in the README (recall per memory mode, prompt tokens, perplexity, memory size) |
| `agent_demo.py` | Questions through the model API, then an agent session using the MCP memory tools |
| `update_test.py` | Corrections and deletions on a copy of the trained memory: sync, measure, re-consolidate, measure |

## How the memory was built

1. **Ingest.** `Qwen3-4B-Instruct-2507`, served locally (`transformers serve`), extracted 326 claims from 8
   documents. Two more documents (`MESSAGE_BOUNDARY_CACHE.md`, `PROMPT_CACHE_TAIL.md`) were skipped: the 4B
   model kept returning invalid JSON for them. The release notes are left out on purpose (they date quickly).
2. **Review.** `review_rules.py`, then every remaining claim checked against the documents. 197 claims were
   removed: markup, tautologies ("supports chat is true"), empty values, duplicates, identifiers the extractor
   invented, and facts that lost their context (a table row about one model read as true for all models).
   129 were rewritten to be precise and self-contained, and 20 facts the extractor missed were added, mostly
   from the MCP server's tool table. 149 facts remain. Every decision is in `review.json`.
3. **Questions.** The 4B model wrote 5–6 phrasings per fact (two short ones were completed by hand). The first
   four are trained, each in the "Question: … Answer:" form and as a plain chat message, the way an agent asks;
   the fifth is held out and never trained.
4. **Consolidation.** Qwen3.8-Flash-Next runs from its FP8 release (173 GB) on one RTX 5090 (32 GB) with 92 GB
   of RAM: experts are read from NVMe on demand, and the backward pass is recomputed layer by layer. Only the
   personal delta of the n-gram table is trained; the model's own weights, its n-gram table and the fusion
   layer are checked byte-for-byte before and after.

## Reproduce

Needs a CUDA GPU with 32 GB, about 100 GB of RAM, the
[Qwen3.8-Flash-Next-FP8](https://huggingface.co/Qwen/Qwen3.8-Flash-Next-FP8) release on a fast SSD, and
`pip install "fla-core>=0.5"` (the DeltaNet kernels).

```bash
transformers serve Qwen/Qwen3-4B-Instruct-2507 --port 8001        # the extraction model, in another shell
python build_memory.py init --workspace ws --model Qwen/Qwen3.8-Flash-Next-FP8 \
    --llm-url http://127.0.0.1:8001/v1 --llm-model Qwen/Qwen3-4B-Instruct-2507
python build_memory.py ingest --workspace ws
python review_rules.py --claims ws/claims.json --out fixes.json   # check the list, add your own decisions
python build_memory.py review --workspace ws --apply fixes.json
python build_memory.py questions --workspace ws
python build_memory.py finalize --workspace ws --max-claims 200
python build_memory.py plan --workspace ws                        # last look at what will be trained
wikiengram consolidate -w ws
python evaluate.py --workspace ws --out results.json
```

Extraction by an LLM is not deterministic, so a rerun extracts somewhat different claims; `review.json` holds
the decisions for this run's claims (by id) and documents what was changed and why.

## Use it from an agent

```bash
wikiengram serve-model -w ws --port 8000      # Qwen3.8-Flash-Next with the Strata memory, OpenAI-compatible
wikiengram mcp -w ws --scope strata           # the LLMWiki memory tools for any MCP client
python agent_demo.py --workspace ws --api http://127.0.0.1:8000/v1 --out transcript.md
```

## License

The example code is Apache-2.0, like WikiEngram. The documents in `source/` and the knowledge in the memory
come from Strata: Copyright (c) Niko1221, MIT License (`source/LICENSE`). Qwen3.8-Flash-Next is under the Qwen
Community License.
