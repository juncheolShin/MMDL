#!/usr/bin/env bash
# One command for inference and scoring, using the same resolved paths.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
EXTRA_ARGS=()

# CLI values override environment variables before Docker or Python is started.
while (( $# )); do
  case "$1" in
    -h|--help)
      cat <<'EOF'
Usage: bash code/scripts/run_mmmu_eval.sh [options]

  --model PATH       Model checkpoint inside the container (MODEL_PATH)
  --data-path PATH   MMMU data directory inside the container (DATA_PATH)
  --config PATH      Evaluation TOML inside the container (CONFIG_PATH)
  --out-root PATH    Parent directory for evaluation outputs (OUT_ROOT)
  --run-name NAME    Result directory name; defaults to a timestamp (RUN_NAME)
  --overwrite        Replace existing predictions in the selected result directory
  -h, --help         Show this help without starting Docker or inference

Options accept --key VALUE or --key=VALUE. CLI overrides environment variables.
CONTAINER_NAME, DOCKER_IMAGE and CONTAINER_ROOT configure Docker on the host.
EOF
      exit 0
      ;;
    --overwrite)
      EXTRA_ARGS+=(--overwrite)
      shift
      ;;
    --model|--model=*|--data-path|--data-path=*|--config|--config=*|--out-root|--out-root=*|--run-name|--run-name=*)
      OPTION="${1%%=*}"
      if [[ "$1" == *=* ]]; then
        VALUE="${1#*=}"
        shift
      else
        if (( $# < 2 )) || [[ "$2" == --* ]]; then
          echo "$OPTION requires a value. See --help." >&2
          exit 2
        fi
        VALUE="$2"
        shift 2
      fi
      if [[ -z "$VALUE" ]]; then
        echo "$OPTION requires a non-empty value. See --help." >&2
        exit 2
      fi
      case "$OPTION" in
        --model) MODEL_PATH="$VALUE" ;;
        --data-path) DATA_PATH="$VALUE" ;;
        --config) CONFIG_PATH="$VALUE" ;;
        --out-root) OUT_ROOT="$VALUE" ;;
        --run-name) RUN_NAME="$VALUE" ;;
      esac
      ;;
    *)
      echo "Unknown argument: $1. See --help; generation settings belong in the TOML file." >&2
      exit 2
      ;;
  esac
done
RUN_NAME="${RUN_NAME:-evaluation_$(date +%Y%m%d_%H%M%S)}"

# Reuse the training container from the host. Inside it, run inference locally.
if [[ ! -f /.dockerenv ]]; then
  CONTAINER_NAME="${CONTAINER_NAME:-mmdl-work}"
  CONTAINER_ROOT="${CONTAINER_ROOT:-/opt/mmdl}"
  IMAGE="${DOCKER_IMAGE:-ssu:MMDL}"
  if ! docker info >/dev/null; then
    echo "Docker에 연결할 수 없습니다. Docker 실행 상태와 접근 권한을 확인한 뒤 다시 실행하세요." >&2
    exit 1
  fi
  if STATE=$(docker container inspect --format '{{.State.Status}}' "$CONTAINER_NAME" 2>/dev/null); then
    case "$STATE" in
      running) ;;
      created|exited)
        echo "기존 컨테이너 시작: $CONTAINER_NAME"
        docker start "$CONTAINER_NAME" >/dev/null
        ;;
      paused)
        echo "컨테이너 일시 정지 해제: $CONTAINER_NAME"
        docker unpause "$CONTAINER_NAME" >/dev/null
        ;;
      *)
        echo "컨테이너 '$CONTAINER_NAME' 상태가 '$STATE'입니다. Docker 상태를 확인한 뒤 다시 실행하세요." >&2
        exit 1
        ;;
    esac
  else
    if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
      printf 'Docker 이미지가 없습니다. 프로젝트 루트에서 먼저 실행하세요:\n  docker build -t %q .\n' "$IMAGE" >&2
      exit 1
    fi
    mkdir -p "$ROOT/training_data" "$ROOT/checkpoints" "$ROOT/results"
    echo "작업용 컨테이너 생성: $CONTAINER_NAME ($IMAGE)"
    docker run -d --gpus all --shm-size=16g --name "$CONTAINER_NAME" \
      -v "$ROOT/code:$CONTAINER_ROOT/code" \
      -v "$ROOT/configs:$CONTAINER_ROOT/configs" \
      -v "$ROOT/training_data:$CONTAINER_ROOT/training_data" \
      -v "$ROOT/checkpoints:$CONTAINER_ROOT/checkpoints" \
      -v "$ROOT/results:$CONTAINER_ROOT/results" \
      "$IMAGE" sleep infinity >/dev/null
  fi
  echo "컨테이너 접속 중... : $CONTAINER_NAME"
  exec docker exec --workdir "$CONTAINER_ROOT" \
    -e MODEL_PATH="${MODEL_PATH:-$CONTAINER_ROOT/models/Qwen3-VL-4B-Instruct}" \
    -e DATA_PATH="${DATA_PATH:-$CONTAINER_ROOT/data}" \
    -e OUT_ROOT="${OUT_ROOT:-$CONTAINER_ROOT/results}" \
    -e RUN_NAME="$RUN_NAME" \
    -e CONFIG_PATH="${CONFIG_PATH:-$CONTAINER_ROOT/configs/evaluation.toml}" \
    -e HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-auto}" \
    -e HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-auto}" \
    -e TRANSFORMERS_OFFLINE=1 \
    "$CONTAINER_NAME" bash "$CONTAINER_ROOT/code/scripts/run_mmmu_eval.sh" "${EXTRA_ARGS[@]}"
fi

MODEL_PATH="${MODEL_PATH:-$ROOT/models/Qwen3-VL-4B-Instruct}"
DATA_PATH="${DATA_PATH:-$ROOT/data}"
OUT_ROOT="${OUT_ROOT:-$ROOT/results}"
CONFIG_PATH="${CONFIG_PATH:-$ROOT/configs/evaluation.toml}"
export VLLM_ENABLE_V1_MULTIPROCESSING=0

# HF_HOME is also the Hugging Face library cache root and stays aligned with DATA_PATH.
export HF_HOME="$DATA_PATH"
if [[ "${HF_HUB_OFFLINE:-auto}" == auto ]]; then
  export HF_HUB_OFFLINE=0
fi
if [[ "${HF_DATASETS_OFFLINE:-auto}" == auto ]]; then
  export HF_DATASETS_OFFLINE=0
fi

python "$ROOT/code/eval/infer_mmmu.py" \
  --model "$MODEL_PATH" --data-path "$DATA_PATH" \
  --out-root "$OUT_ROOT" --run-name "$RUN_NAME" \
  --config "$CONFIG_PATH" "${EXTRA_ARGS[@]}"
RUN_DIR="$OUT_ROOT/$RUN_NAME"
if [[ "$RUN_NAME" == /* ]]; then
  RUN_DIR="$RUN_NAME"
fi
python "$ROOT/code/eval/score_mmmu.py" "$RUN_DIR"
