#!/usr/bin/env bash
# Small benchmark used for the README on an Apple M4 (32 GB): 50 fictitious facts, Qwen3-0.6B-Base.
# Reuses a trained base Engram if BASE points to one (engram/base of a workspace).
set -euo pipefail
OUT=${OUT:-.data/bench-mac-50}
BASE=${BASE:-.data/spike/base}
ARGS=(--facts 50 --model Qwen/Qwen3-0.6B-Base --out "$OUT" --epochs 20)
if [ -f "$BASE/engram.json" ]; then ARGS+=(--base "$BASE"); fi
wikiengram bench "${ARGS[@]}"
