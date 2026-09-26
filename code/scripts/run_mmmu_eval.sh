#!/usr/bin/env bash
# One command for inference and scoring; extra flags go to infer_mmmu.py.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MODEL_PATH="${MODEL_PATH:-$ROOT/models/Qwen3-VL-4B-Instruct}"
DATA_ROOT="${DATA_ROOT:-$ROOT/data}"
OUT_ROOT="${OUT_ROOT:-$ROOT/results}"
RUN_NAME="${RUN_NAME:-base_greedy}"
CONFIG_PATH="${CONFIG_PATH:-$ROOT/configs/rtx4090.toml}"
export VLLM_ENABLE_V1_MULTIPROCESSING=0

python "$ROOT/code/eval/infer_mmmu.py" \
  --model "$MODEL_PATH" --data-root "$DATA_ROOT" \
  --out-root "$OUT_ROOT" --run-name "$RUN_NAME" \
  --config "$CONFIG_PATH" "$@"
python "$ROOT/code/eval/score_mmmu.py" "$OUT_ROOT/$RUN_NAME" --data-root "$DATA_ROOT"
