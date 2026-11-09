#!/usr/bin/env bash
# End-to-end demo of the WikiEngram CLI on a small model (Qwen3-0.6B-Base).
# Runs on a laptop GPU (Apple MPS) or any CUDA GPU; CPU works but is slow.
set -euo pipefail
WS=${WS:-./wikiengram-data}

wikiengram init -w "$WS" --model Qwen/Qwen3-0.6B-Base
# 1. once per backbone: train the base Engram with the backbone frozen (~10 min on an M4, ~1-2 min on a 4090)
wikiengram adapt -w "$WS" --steps 300

# 2. explicit memory: extract knowledge from documents and conversations
wikiengram ingest -w "$WS" examples/data/team_notes.md
wikiengram ingest -w "$WS" --kind conversation examples/data/conversation.json
wikiengram wiki list -w "$WS"

# 3. neural memory: consolidate claims into the personal Engram delta
wikiengram consolidate -w "$WS" --plan
wikiengram consolidate -w "$WS"

# 4. ask with each memory
for mode in none wiki engram hybrid; do
  echo "--- $mode"; wikiengram ask -w "$WS" --mode "$mode" "Where does Ilsa Varrow work?"
done

# 5. correct a fact in the wiki; Engram forgets the old value on sync and the wiki answers the new one
CID=$(wikiengram wiki search -w "$WS" "Ilsa Varrow work" | head -1 | awk '{print $2}')
wikiengram wiki correct -w "$WS" "$CID" --object "Northgate Systems"
wikiengram sync -w "$WS"
wikiengram ask -w "$WS" --mode engram "Where does Ilsa Varrow work?"
wikiengram ask -w "$WS" --mode hybrid "Where does Ilsa Varrow work?"
wikiengram status -w "$WS"
