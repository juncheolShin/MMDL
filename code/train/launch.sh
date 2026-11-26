#!/usr/bin/env bash
# Launch one training arm, on one or more GPUs.
#
# train_lora.py calls upstream's train() in-process, so on its own it uses a
# single GPU no matter how many the pod was given. torchrun supplies the
# distributed launch; nothing else changes.
#
#   bash code/train/launch.sh --arm a --seed 0
#   bash code/train/launch.sh --arm c1 --seed 1 --gpus 2
#   bash code/train/launch.sh --arm a --seed 0 --validate-only
#
# Required: --arm (a name under configs/arms/, without .toml)
# Optional: --seed (overrides the config's seed, for the 3-seed runs)
#           --gpus (default: every visible device)
#           --annotation / --image-root / --out / --model
#
# The global batch size is held constant across arms and GPU counts:
#   global = gpus x per_device_train_batch_size x gradient_accumulation_steps
# so changing --gpus without changing accumulation would silently change the
# optimizer-step budget that DECISIONS.md section 2 fixes. This script refuses
# that instead of letting the comparison drift.

set -uo pipefail

ROOT=${ROOT:-/workspace/MMDL}
WS=${WS:-/workspace}
ENV_FILE=${ENV_FILE:-$WS/envs/mmdl.env}

ARM="" ; SEED="" ; GPUS="" ; VALIDATE=0
ANNOTATION=${ANNOTATION:-} ; IMAGE_ROOT=${IMAGE_ROOT:-} ; OUT=${OUT:-}
MODEL=${MODEL:-$WS/models/Qwen3-VL-4B-Instruct}
EXPECTED_GLOBAL_BATCH=${EXPECTED_GLOBAL_BATCH:-32}

while [ $# -gt 0 ]; do
  case "$1" in
    --arm)           ARM=${2:?}; shift 2 ;;
    --seed)          SEED=${2:?}; shift 2 ;;
    --gpus)          GPUS=${2:?}; shift 2 ;;
    --annotation)    ANNOTATION=${2:?}; shift 2 ;;
    --image-root)    IMAGE_ROOT=${2:?}; shift 2 ;;
    --out)           OUT=${2:?}; shift 2 ;;
    --model)         MODEL=${2:?}; shift 2 ;;
    --validate-only) VALIDATE=1; shift ;;
    -h|--help)       sed -n '2,24p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

die() { printf '\nlaunch: %s\n' "$1" >&2; exit 1; }

[ -n "$ARM" ] || die "--arm is required (one of: $(ls "$ROOT/configs/arms" 2>/dev/null | sed 's/\.toml//' | tr '\n' ' '))"
CONFIG="$ROOT/configs/arms/$ARM.toml"
[ -f "$CONFIG" ] || die "no such arm config: $CONFIG"

# shellcheck disable=SC1090
[ -f "$ENV_FILE" ] && source "$ENV_FILE"
command -v python >/dev/null 2>&1 || die "python not on PATH; is $ENV_FILE present?"

if [ -z "$GPUS" ]; then
  GPUS=$(python -c 'import torch;print(torch.cuda.device_count())' 2>/dev/null || echo 0)
fi
case "$GPUS" in ''|*[!0-9]*) die "--gpus must be an integer, got '$GPUS'" ;; esac
[ "$GPUS" -ge 1 ] || die "no GPU visible (device_count=0). This must run in a GPU job."

ANNOTATION=${ANNOTATION:-$ROOT/training_data/sft/$ARM/train.jsonl}
IMAGE_ROOT=${IMAGE_ROOT:-$ROOT/training_data/cluster_upload}
SEED_TAG=${SEED:-cfg}
OUT=${OUT:-$WS/checkpoints/$ARM-seed$SEED_TAG}

# Keep the optimizer-step budget identical across arms and GPU counts.
read -r PER_DEV ACCUM CFG_SEED <<EOF
$(CONFIG="$CONFIG" python - <<'PY'
import os, sys, pathlib
sys.path.insert(0, str(pathlib.Path(os.environ["CONFIG"]).resolve().parents[2] / "code"))
from configuration import TRAINING_TYPES, load_section
t = load_section(os.environ["CONFIG"], "training", TRAINING_TYPES)
print(t["per_device_train_batch_size"], t["gradient_accumulation_steps"], t["seed"])
PY
)
EOF
[ -n "${PER_DEV:-}" ] || die "could not read [training] from $CONFIG"

GLOBAL=$(( GPUS * PER_DEV * ACCUM ))
if [ "$GLOBAL" -ne "$EXPECTED_GLOBAL_BATCH" ]; then
  die "global batch would be $GLOBAL (gpus $GPUS x per_device $PER_DEV x accum $ACCUM),
     but the fixed budget is $EXPECTED_GLOBAL_BATCH.
     Adjust gradient_accumulation_steps in $CONFIG for this GPU count, or set
     EXPECTED_GLOBAL_BATCH deliberately. Arms must share one step budget."
fi

[ -n "$SEED" ] && export MMDL_SEED_OVERRIDE="$SEED"

cat <<EOF

arm            : $ARM
config         : $CONFIG
seed           : ${SEED:-$CFG_SEED (from config)}
gpus           : $GPUS
global batch   : $GLOBAL  ($GPUS x $PER_DEV x $ACCUM)
annotation     : $ANNOTATION
image root     : $IMAGE_ROOT
output         : $OUT
EOF

ARGS=(--annotation "$ANNOTATION" --image-root "$IMAGE_ROOT"
      --output-dir "$OUT" --model "$MODEL" --config "$CONFIG")
[ "$VALIDATE" = "1" ] && ARGS+=(--validate-only)

echo
if [ "$VALIDATE" = "1" ] || [ "$GPUS" -eq 1 ]; then
  # No point starting a distributed group to validate a JSONL.
  exec python "$ROOT/code/train/train_lora.py" "${ARGS[@]}"
fi
exec torchrun --nproc_per_node="$GPUS" --nnodes=1 \
     --master_addr=127.0.0.1 --master_port="${MASTER_PORT:-29500}" \
     "$ROOT/code/train/train_lora.py" "${ARGS[@]}"
