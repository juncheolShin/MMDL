#!/usr/bin/env bash
# Evaluate an exported model with the scorer used for the presentation comparison.
set -euo pipefail
model=${1:?Usage: bash icot/evaluate.sh MERGED_MODEL NEW_RUN_DIR}
run=${2:?Missing NEW_RUN_DIR}
data=${DATA_PATH:-/opt/mmdl/data}
mkdir -p "$(dirname "$run")"
mkdir "$run"
protocol="$run/protocol"
mkdir -p "$protocol/code/eval"
cp code/configuration.py "$protocol/code/"
cp -r code/third_party "$protocol/code/"
cp code/eval/{infer_mmmu,check_prompts,mmmu_data,prompting}.py "$protocol/code/eval/"
cp icot/scoring/{answer_extraction,score_mmmu}.py "$protocol/code/eval/"
export OMP_NUM_THREADS=2 VLLM_ENABLE_V1_MULTIPROCESSING=0
python "$protocol/code/eval/infer_mmmu.py" --model "$model" --config configs/evaluation.toml \
    --data-path "$data" --out-root "$run" --run-name mmmu
python "$protocol/code/eval/score_mmmu.py" "$run/mmmu"
