# Raw results of the Strata example

Measured on one RTX 5090 (32 GB) in a container with 25 CPU cores and a 92 GB memory limit, with
Qwen3.8-Flash-Next streamed from its FP8 release.

| File | What it is |
|---|---|
| `results.json` | Recall per memory mode on trained and held-out phrasings, prompt tokens, perplexity, memory size (`evaluate.py`) |
| `results_detail.json` | Every question asked: mode, fact, question, expected answer, the model's reply, whether it counted |
| `verify.json` | `wikiengram verify` after training: probe accuracy by kind and how many facts pass every probe |
| `consolidate.json` | The consolidation report: training settings, loss and confident-token share per epoch, rows, checksums |
| `training_epochs.txt` | The per-epoch progress lines of that run |
| `resources_training.csv` | GPU memory and utilisation, container memory and process RSS, sampled every 30 s |
| `transcript.md` | The model API demo: eight new questions in three memory modes, then an agent session using the MCP memory tools |

Notes:

* The verification inside `consolidate.json` ran before a fix: statement probes were wrapped in the chat
  template, although statements are trained as plain text continuations. `verify.json` is the verification
  after the fix, and the one the hybrid routing used.
* The wiki and hybrid rows were measured with the brief-answer instruction for chat models and the current
  hybrid routing (questions routed to Engram carry no other facts). The other rows were re-scored with the same
  matcher, which ignores Markdown marks.
