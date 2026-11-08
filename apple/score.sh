#!/usr/bin/env bash
# Score generations with the OFFICIAL, unmodified evaluator (evaluation/score_meter_count.py):
# BiLSTM meter classifier + bayt-count adherence. Runs on the CPU (deterministic), no CUDA needed.
#
# Usage: ./score.sh runs/e3/generations.jsonl runs/e3/scored.jsonl
# Needs SHAER_ROOT to point at the official/forked Shaer checkout when this folder is not inside it.
set -euo pipefail
IN="$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"
OUT_DIR="$(mkdir -p "$(dirname "$2")" && cd "$(dirname "$2")" && pwd)"
OUT="$OUT_DIR/$(basename "$2")"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SHAER_ROOT="${SHAER_ROOT:-$(cd "$HERE/.." && pwd)}"
[ -f "$SHAER_ROOT/evaluation/score_meter_count.py" ] || { echo "set SHAER_ROOT to the Shaer checkout" >&2; exit 1; }
export METER_MODEL_ID="${METER_MODEL_ID:-Shaer-AI/4BiLSTM-meter-classification-pytorch}"
cd "$SHAER_ROOT/evaluation"
python score_meter_count.py --input-jsonl "$IN" --output-jsonl "$OUT"
