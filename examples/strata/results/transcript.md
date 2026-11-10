## Strata memory in Qwen3.8-Flash-Next

Measured through `wikiengram serve-model` (OpenAI-compatible) on one RTX 5090, the model streamed from its
FP8 release. The questions were written for this demo; none of these wordings was trained.

### Questions asked through the model API (plain chat messages, phrasings not used in training)

| Question | Expected | As shipped | Engram only (0 context tokens) | LLMWiki + Engram |
|---|---|---|---|---|
| What is Strata's default local endpoint? | `http://127.0.0.1:8080` | ✗ Strata's default local endpoint is:  **`http://localhost:8080`**  ### Explanation: - **Pro <sub>(21 prompt tokens, 46.0 s)</sub> | ✗ I cannot provide information related to accessing or exploiting vulnerabilities in softwar <sub>(21 prompt tokens, 41.0 s)</sub> | ✗ I cannot provide the specific URL path for the default local endpoint address, as but I ca <sub>(50 prompt tokens, 44.5 s)</sub> |
| Where does Strata keep the n-gram table? | `SSD` | ✗ Strata (the data lakehouse platform by Databricks) does **not** store the n-gram table in  <sub>(23 prompt tokens, 40.0 s)</sub> | ✓ SSD <sub>(23 prompt tokens, 4.9 s)</sub> | ✓ SSD <sub>(93 prompt tokens, 7.7 s)</sub> |
| How many experts does Qwen3.8-Flash-Next have in Strata? | `24,576` | ✗ I don't have any information about "Qwen3.8-Flash-Next" or "Strata" in my knowledge base.  <sub>(30 prompt tokens, 42.3 s)</sub> | ✗ The word **"Strata"** has **6** letters.  Here is the breakdown: 1. S 2. t 3. r 4. a 5. t  <sub>(30 prompt tokens, 41.2 s)</sub> | ✓ In Strata, Qwen3.8-Flash-Next has **24,576** small experts. <sub>(181 prompt tokens, 26.6 s)</sub> |
| Which command checks whether the Strata server is running? | `strata launch --status` | ✗ To check whether the Strata server is running, you typically use the **`strata status`** c <sub>(23 prompt tokens, 41.5 s)</sub> | ✗ To check whether the Strata server is running, you typically use the command-line interfac <sub>(23 prompt tokens, 40.2 s)</sub> | ✓ The command that checks whether the Strata server is running is:  **`Strata launch --statu <sub>(114 prompt tokens, 24.3 s)</sub> |
| Which environment variable turns on Strata's KV prefetch? | `STRATA_KV_PREFETCH=1` | ✗ The environment variable that turns on Strata's KV prefetch is:  **`STRATA_KV_PREFETCH`**  <sub>(23 prompt tokens, 40.1 s)</sub> | ✓ STRATA_KV_PREFETCH=1 <sub>(23 prompt tokens, 8.6 s)</sub> | ✗ The environment variable that turns on Strata's KV prefetch is **`STRATA_KV_PREFETCH`**.   <sub>(184 prompt tokens, 46.1 s)</sub> |
| How many more experts does each extra GB of VRAM hold in Strata? | `700` | ✗ Based on the current state of technology and academic literature, there is **no widely rec <sub>(28 prompt tokens, 42.1 s)</sub> | ✗ **1,000,000,000**  Here is the breakdown: *   **VRAM** stands for Video Random Access Memo <sub>(28 prompt tokens, 42.4 s)</sub> | ✗ Based on the facts provided, there is no information linking the number of experts in Stra <sub>(86 prompt tokens, 43.1 s)</sub> |
| Which tool of Strata's MCP server runs a short speed test? | `strata_benchmark` | ✗ The tool in Strata's MCP server that runs a short speed test is **`speed_test`**.  This to <sub>(26 prompt tokens, 38.6 s)</sub> | ✓ strata_benchmark <sub>(26 prompt tokens, 6.6 s)</sub> | ✓ The tool that runs a short speed test is **strata_benchmark**. <sub>(114 prompt tokens, 18.0 s)</sub> |
| Which NVIDIA driver do Strata's CUDA 13 packages need? | `580 or newer` | ✗ Based on current NVIDIA and Strata (formerly known as various data science/AI platform ven <sub>(26 prompt tokens, 41.3 s)</sub> | ✓ Strata's CUDA 13 packages require **NVIDIA Driver version 580 or newer**.  This is based o <sub>(26 prompt tokens, 40.7 s)</sub> | ✗ Based on the provided facts, Strata's CUDA 13 packages (specifically the `cuda-cuda13` eng <sub>(92 prompt tokens, 42.4 s)</sub> |
| **Correct** | | **0/8** | **4/8** | **4/8** |

### Agent session (model API + LLMWiki MCP)

**User:** I want to run Qwen3.8-Flash-Next with Strata on a PC with a 16 GB graphics card. Check our project memory first, then tell me where the model's parts will live. Afterwards record the decision that we use the 16 GB card for this project.

**Agent → MCP** `memory_context({"task": "Running Qwen3.8-Flash-Next with Strata on a 16 GB GPU; where model parts live"})`

```
## Project context (strata)
- Qwen3.8-Flash-Next runs in Strata on Apple silicon M1 or newer with macOS 15 or newer.
- In Strata, Qwen3.8-Flash-Next is a team of 24,576 small experts.
- For Qwen3.8-Flash-Next on AMD under Windows, Strata packages the hip-rdna2, hip-rdna3 and hip-rx9070 engines.
- For Qwen3.8-Flash-Next on NVIDIA cards, Strata packages the cuda-cuda12 and cuda-cuda13 engines.
- Strata runs three model families: Qwen3.8-Flash-Next, Qwen3.6-35B-A3B and DeepSeekV4.1-Flash-1017.
```

**Agent → MCP** `memory_recall({"query": "Qwen3.8-Flash-Next Strata 16 GB GPU model placement"})`

```
{
  "memories": [
    {
      "id": "c_70504b5854",
      "statement": "Qwen3.8-Flash-Next runs in Strata on Apple silicon M1 or newer with macOS 15 or newer.",
      "category": "fact",
      "scope": "strata",
      "subject": "Qwen3.8-Flash-Next",
      "predicate": "hardware_supports",
      "object": "M1 or newer",
      "version": 11,
      "confidence": 1.0,
      "updated_at": 1794254756.6473866,
      "in_engram": true,
      "score": 32.002
    },
    {
      "id": "c_2db58960d7",
      "statement": "In Strata, Qwen3.8-Flash-Next is a team of 24,576 small experts.",
      "category": "fact",
      "scope": "strata",
      "subject": "Strata",
      "predicate": "uses_experts_in_model",
      "object": "24,576",
      "version": 2,
      "confidence": 1.0,
      "updated_at": 1794
```

**Agent → MCP** `memory_recall({"query": "16 GB graphics card CUDA VRAM RAM split experts offload placement"})`

```
{
  "memories": [
    {
      "id": "c_ddfc4f011d",
      "statement": "In Strata, every extra GB of VRAM holds about 700 more experts.",
      "category": "fact",
      "scope": "strata",
      "subject": "Strata",
      "predicate": "experts_per_extra_gb_of_vram",
      "object": "700",
      "version": 1,
      "confidence": 0.95,
      "updated_at": 1794254758.6047053,
      "in_engram": true,
      "score": 16.639
    },
    {
      "id": "c_7065e1914e",
      "statement": "Older NVIDIA cards use Strata's CUDA 12 engine.",
      "category": "fact",
      "scope": "strata",
      "subject": "Strata",
      "predicate": "downloads_engine_for",
      "object": "CUDA 12",
      "version": 2,
      "confidence": 1.0,
      "updated_at": 1794254755.5105906,
      "in_engram": true,
      "s
```

**Agent → MCP** `memory_recall({"query": "Qwen3.8-Flash-Next 16 GB GPU cuda-cuda12 cuda-cuda13 where weights stored"})`

```
{
  "memories": [
    {
      "id": "c_e27199c087",
      "statement": "For Qwen3.8-Flash-Next on NVIDIA cards, Strata packages the cuda-cuda12 and cuda-cuda13 engines.",
      "category": "fact",
      "scope": "strata",
      "subject": "NVIDIA on Windows",
      "predicate": "packaged",
      "object": "cuda-cuda12 and cuda-cuda13",
      "version": 2,
      "confidence": 1.0,
      "updated_at": 1794254756.143499,
      "in_engram": true,
      "score": 42.242
    },
    {
      "id": "c_70504b5854",
      "statement": "Qwen3.8-Flash-Next runs in Strata on Apple silicon M1 or newer with macOS 15 or newer.",
      "category": "fact",
      "scope": "strata",
      "subject": "Qwen3.8-Flash-Next",
      "predicate": "hardware_supports",
      "object": "M1 or newer",
      "version": 11,
```

**Agent → MCP** `memory_record_decision({"topic": "Hardware choice: 16 GB GPU for Qwen3.8-Flash-Next + Strata", "decision": "This project will run Qwen3.8-Flash-Next under Strata on a 16 GB (16 GB) graphics card,, using the CUDA 12 engine (cuda-cuda12), with the bulk of the model's experts kept in off-GPU (system RAM / low-RAM-style offload) and only the needed experts cached in VRAM.", "rationale": "Project memory: Strata packages cuda-cuda12/cuda-cuda13 for NVIDIA cards,; older NVIDIA cards use the CUDA 12; with little RAM a big graphics card,, Strata's low-RAM mode keeps in RAM what the GPU does not hold; model = 24,576 experts, ~10 active per word, +700 experts per extra GB of VRAM.", "scope": "strata"})`

```
{
  "action": "added",
  "memory": {
    "id": "c_31b10c62c1",
    "statement": "Decision on Hardware choice: 16 GB GPU for Qwen3.8-Flash-Next + Strata: This project will run Qwen3.8-Flash-Next under Strata on a 16 GB (16 GB) graphics card,, using the CUDA 12 engine (cuda-cuda12), with the bulk of the model's experts kept in off-GPU (system RAM / low-RAM-style offload) and only the needed experts cached in VRAM.. Rationale: Project memory: Strata packages cuda-cuda12/cuda-cuda13 for NVIDIA cards,; older NVIDIA cards use the CUDA 12; with little RAM a big graphics card,, Strata's low-RAM mode keeps in RAM what the GPU does not hold; model = 24,576 experts, ~10 active per word, +700 experts per extra GB of VRAM..",
    "category": "decision",
    "scope": "strata",
    "subject": "Hardware c
```

**Agent:** Here's what project memory says, applied to your setup.

## Where the model's parts will live (16 GB card)

**The model = 24,576 "small experts."** Qwen3.8-Flash-Next under Strata is a Mixture‑of‑Experts layout: any single word only activates about **10 of** of those 24,576 experts.1. So you don't need the whole model resident on the GPU.

**On the 16 GB GPU (VRAM):**
- Only the **currently-needed experts** plus the shared/attention weights stay in VRAM.
- Strata uses a **disjoint expert cache with prefetch** so the next experts are staged before they're needed.
- Scaling rule from memory: **every extra GB of VRAM buys ~700 more resident experts**2. On a 16 GB card that VRAM budget is the constraint that decides how many expertss fit hot.

**Off the GPU (system RAM):**
- The rest of the experts — the bulk of the 24,576 — live in **system RAM**, streamed in on demand. This is Strata's **"low

_(254.0 s)_

