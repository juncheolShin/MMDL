#!/usr/bin/env bash
# The only sanctioned way to submit a GPU Job in this project.
#
# It refuses to submit unless the GPU node is named explicitly on the command
# line. Templates carry `nodeSelector: kubernetes.io/hostname: ${NODE_NAME}`
# with no default, so a job rendered without --node would land on whatever the
# scheduler picked - which is exactly what must not happen on a shared cluster.
#
#   bash code/cluster/submit_job.sh --template probe --node h200-03-w-7271 --gpus 1
#   bash code/cluster/submit_job.sh --template lora  --node h200-03-w-7271 --gpus 2
#   bash code/cluster/submit_job.sh --template probe --node h200-03-w-7a2a --gpus 1 --dry-run
#
# Rules enforced here:
#   * --node is REQUIRED. No default, no guessing, no "last used".
#   * --gpus is REQUIRED. No default either - how much of a shared machine a run
#     takes is a decision, not something a script should assume.
#   * the node must be one this namespace is actually running pods on.
set -uo pipefail

export MSYS_NO_PATHCONV=1
export MSYS2_ARG_CONV_EXCL='*'

KUBECONFIG_FILE=${KUBECONFIG_FILE:-reality-lab.yaml}
NAMESPACE=${NAMESPACE:-p-test2}
OWNER_TAG=${OWNER_TAG:-yeunsuh}   # every job name must carry this (D44)

TEMPLATE="" ; NODE_NAME="" ; GPU_COUNT="" ; JOB_NAME="" ; DRY=0
while [ $# -gt 0 ]; do
  case "$1" in
    --template) TEMPLATE=${2:?}; shift 2 ;;
    --node)     NODE_NAME=${2:?}; shift 2 ;;
    --gpus)     GPU_COUNT=${2:?}; shift 2 ;;
    --name)     JOB_NAME=${2:?}; shift 2 ;;
    --dry-run)  DRY=1; shift ;;
    -h|--help)  sed -n '2,20p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

die() { printf '\nsubmit_job: %s\n' "$1" >&2; exit 1; }

[ -n "$TEMPLATE" ] || die "--template is required (probe | lora | <path to .yaml.tmpl>)"

# ---- the guard -------------------------------------------------------------
if [ -z "$NODE_NAME" ]; then
  cat >&2 <<'EOF'

submit_job: REFUSED - no --node given.

GPU jobs in this project must name the node explicitly. Pick one and re-run:

  bash code/cluster/submit_job.sh --template <t> --node <node> --gpus <N>

To see what the namespace is currently running:

  kubectl --kubeconfig reality-lab.yaml -n p-test2 get pods \
    -o custom-columns='NODE:.spec.nodeName,POD:.metadata.name' --no-headers \
    | awk '{print $1}' | sort | uniq -c | sort -rn

EOF
  exit 1
fi

if [ -z "$GPU_COUNT" ]; then
  cat >&2 <<'EOF'

submit_job: REFUSED - no --gpus given.

How many GPUs a run takes on a shared node is a decision, so this script does
not pick one for you. Re-run with an explicit count:

  bash code/cluster/submit_job.sh --template <t> --node <node> --gpus <N>

EOF
  exit 1
fi

case "$GPU_COUNT" in *[!0-9]*) die "--gpus must be an integer, got '$GPU_COUNT'" ;; esac

case "$TEMPLATE" in
  probe) TEMPLATE_PATH=code/cluster/probe_job.yaml.tmpl ;;
  lora)  TEMPLATE_PATH=code/cluster/lora_job.yaml.tmpl ;;
  eval)  TEMPLATE_PATH=code/cluster/eval_job.yaml.tmpl ;;
  *)     TEMPLATE_PATH=$TEMPLATE ;;
esac
[ -f "$TEMPLATE_PATH" ] || die "template not found: $TEMPLATE_PATH"

K="kubectl --kubeconfig $KUBECONFIG_FILE -n $NAMESPACE"

# Confirm the node exists as far as this namespace can tell. `get nodes` is
# forbidden for a namespace-scoped account, so check against scheduled pods.
KNOWN=$($K get pods -o jsonpath='{range .items[*]}{.spec.nodeName}{"\n"}{end}' 2>/dev/null | sort -u | grep -v '^$')
if ! printf '%s\n' "$KNOWN" | grep -qx "$NODE_NAME"; then
  echo "WARNING: '$NODE_NAME' is not among the nodes this namespace is running pods on:" >&2
  printf '  %s\n' $KNOWN >&2
  echo "  Continuing anyway - the list only shows nodes with current pods." >&2
fi

[ -n "$JOB_NAME" ] || JOB_NAME="${TEMPLATE##*/}"
[ -n "$JOB_NAME" ] || JOB_NAME=job
JOB_NAME="${JOB_NAME%%.*}"
# Shared cluster: a job must be identifiable as this account's at a glance.
case "$JOB_NAME" in
  *"$OWNER_TAG"*) : ;;
  *) JOB_NAME="$OWNER_TAG-$JOB_NAME" ;;
esac
JOB_NAME="$JOB_NAME-$(date +%m%d-%H%M)"

export JOB_NAME NODE_NAME GPU_COUNT NAMESPACE
export PVC_NAME=${PVC_NAME:-yeunsuh-workspace-1}
export CPU_LIMIT=${CPU_LIMIT:-8}
export MEM=${MEM:-32Gi}
export ZONE=${ZONE:-private-h200-realitylab-0}
export IMAGE=${IMAGE:-mlx-public.kr.ncr.ntruss.com/mlx/notebook/kubeflow-jupyter:2.7.0}
export MODEL_DIR=${MODEL_DIR:-/workspace/models/Qwen3-VL-4B-Instruct}
export ENV_FILE=${ENV_FILE:-/workspace/envs/mmdl.env}
export OMP_THREADS=${OMP_THREADS:-8}
export ARM=${ARM:-}
export SEED=${SEED:-0}
export SFT_SET=${SFT_SET:-}
export TASK=${TASK:-}
export RUN_NAME=${RUN_NAME:-$JOB_NAME}
export COST_LIMIT=${COST_LIMIT:-100}
export COST_REPEATS=${COST_REPEATS:-3}
export EXTRA_ARGS=${EXTRA_ARGS:-}

# A training job with no arm would render an empty --arm and fail inside the pod
# after the image pull and model load. Catch it here instead.
case "$TEMPLATE_PATH" in
  *eval_job*)
    case "$TASK" in
      mmmu|indomain|cost|patchembed) : ;;
      profile)
        [ -n "$SFT_SET" ] || die "TASK=profile needs SFT_SET=<name> (it profiles a real batch)"
        [ -f "training_data/sft/$SFT_SET/train.jsonl" ]           || die "no such sft set: training_data/sft/$SFT_SET/train.jsonl" ;;
      "") die "this template needs TASK=mmmu|indomain|cost|profile" ;;
      *)  die "unknown TASK=$TASK (want mmmu, indomain, cost or profile)" ;;
    esac
    ;;
esac
case "$TEMPLATE_PATH" in
  *lora_job*)
    [ -n "$ARM" ] || die "this template needs ARM=<name> (configs/arms/<name>.toml)"
    [ -f "configs/arms/$ARM.toml" ] || die "no such arm config: configs/arms/$ARM.toml"
    [ -n "$SFT_SET" ] || die "this template needs SFT_SET=<name> (training_data/sft/<name>/train.jsonl)"
    ;;
esac
export WANDB_SECRET=${WANDB_SECRET:-$OWNER_TAG-wandb}
export ANTHROPIC_SECRET=${ANTHROPIC_SECRET:-$OWNER_TAG-anthropic}

mkdir -p code/cluster/rendered
OUT="code/cluster/rendered/$JOB_NAME.yaml"
# envsubst with no argument substitutes EVERY $VAR it finds, including the shell
# variables the in-pod script defines for itself. That silently emptied $RUN and
# $OUT in eval_job.yaml.tmpl, so the MMMU job wrote to the repo root and the
# scorer looked for /predictions.jsonl. Name the list explicitly: anything not
# listed is left alone for the pod's own shell to expand.
SUBST_VARS='$JOB_NAME $NAMESPACE $NODE_NAME $GPU_COUNT $PVC_NAME $CPU_LIMIT $MEM $ZONE $IMAGE $MODEL_DIR $ENV_FILE $OMP_THREADS $ARM $SEED $SFT_SET $TASK $RUN_NAME $COST_LIMIT $COST_REPEATS $EXTRA_ARGS $WANDB_SECRET'
envsubst "$SUBST_VARS" < "$TEMPLATE_PATH" > "$OUT" || die "envsubst failed"

# Prove no placeholder was left behind, and no flag was handed an empty value.
if grep -nE '[$][{](JOB_NAME|NAMESPACE|NODE_NAME|GPU_COUNT|PVC_NAME|CPU_LIMIT|MEM|ZONE|IMAGE|MODEL_DIR|ENV_FILE|OMP_THREADS|TASK|RUN_NAME|COST_LIMIT|COST_REPEATS|WANDB_SECRET)[}]' "$OUT"; then
  die "rendered manifest still holds an unsubstituted placeholder (above) - refusing"
fi
if grep -nE -- '--[a-z-]+ ""' "$OUT"; then
  die "rendered manifest passes an empty value to the flag above - refusing"
fi

# Prove the node pin actually made it into the manifest.
grep -q "kubernetes.io/hostname: $NODE_NAME" "$OUT" \
  || die "rendered manifest has no nodeSelector for $NODE_NAME - template problem, refusing"

cat <<EOF

about to submit
  job       : $JOB_NAME
  template  : $TEMPLATE_PATH
  namespace : $NAMESPACE
  node      : $NODE_NAME   <- pinned via nodeSelector
  gpus      : $GPU_COUNT
  cpu / mem : $CPU_LIMIT / $MEM
  manifest  : $OUT
EOF

if [ "$DRY" = "1" ]; then
  echo
  $K apply --dry-run=client -f "$OUT"
  echo "dry run - nothing submitted"
  exit 0
fi

echo
$K apply -f "$OUT" || die "apply failed"
echo
echo "follow with:"
echo "  kubectl --kubeconfig $KUBECONFIG_FILE -n $NAMESPACE logs -l job-name=$JOB_NAME -c <container> -f"
