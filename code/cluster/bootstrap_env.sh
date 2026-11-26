#!/usr/bin/env bash
# Build the project's conda env on the PVC. Run INSIDE the notebook pod (yeunsuh-0).
# No GPU is needed for this; it is pip install + a model download.
#
#   bash bootstrap_env.sh            # full run
#   bash bootstrap_env.sh --probe    # only check prerequisites, install nothing
#
# Why a PVC env and not the container: /opt/conda and $HOME live on the node's
# local disk and vanish when the pod restarts. Only /workspace and /data persist
# (both are the same Lustre filesystem, PVCs yeunsuh-workspace-1 and data).
#
# Measured on 2026-11-18 in yeunsuh-0:
#   cgroup limits  24 CPU, 128 GiB   (nproc reports 128 and free reports 2 TiB - both lie)
#   /opt/conda     python 3.11.9, NO torch, no ML packages at all
#   network        huggingface.co 200, pypi.org 200, cdn-lfs.huggingface.co 000

set -uo pipefail

WS=${WS:-/workspace}
ENV_PREFIX=${ENV_PREFIX:-$WS/envs/mmdl}
PROJECT=${PROJECT:-$WS/MMDL}
MODEL_REPO=${MODEL_REPO:-Qwen/Qwen3-VL-4B-Instruct}
MODEL_REV=${MODEL_REV:-ebb281ec70b05090aa6165b016eac8ec08e71b17}
MODEL_DIR=${MODEL_DIR:-$WS/models/Qwen3-VL-4B-Instruct}
QWEN_COMMIT=${QWEN_COMMIT:-96588727e44c78b25ba03ea03b8e12f7e64fd0da}
QWEN_ROOT=${QWEN_ROOT:-$WS/qwen3-vl}
PY_VERSION=${PY_VERSION:-3.12}

PROBE_ONLY=0
[ "${1:-}" = "--probe" ] && PROBE_ONLY=1

line() { printf '\n===== %s =====\n' "$1"; }
die()  { printf '\nbootstrap_env: %s\n' "$1" >&2; exit 1; }

# ---------------------------------------------------------------- 0. prereqs

line "prerequisites"
[ -d "$WS" ] || die "$WS does not exist - is the PVC mounted?"
touch "$WS/.bootstrap_write_test" 2>/dev/null || die "$WS is not writable"
rm -f "$WS/.bootstrap_write_test"
echo "workspace   : $WS (writable)"
echo "free        : $(df -h "$WS" | awk 'NR==2{print $4}')"

# The pod sees the node's cores, not its own quota. Everything downstream must
# use the quota, or torch spawns 128 threads onto 24 CPUs and thrashes.
CPU_QUOTA=$(awk '{ if ($1 == "max") print "max"; else printf "%d", $1/$2 }' /sys/fs/cgroup/cpu.max 2>/dev/null || echo "?")
MEM_MAX=$(cat /sys/fs/cgroup/memory.max 2>/dev/null || echo "?")
echo "nproc says  : $(nproc)"
echo "cgroup CPU  : $CPU_QUOTA        <- the real limit"
echo "cgroup MEM  : $MEM_MAX bytes"
case "$CPU_QUOTA" in ''|max|\?) THREADS=8 ;; *) THREADS=$(( CPU_QUOTA > 8 ? 8 : CPU_QUOTA )) ;; esac
echo "OMP_NUM_THREADS will be set to $THREADS"

command -v conda >/dev/null 2>&1 || die "conda not on PATH"
echo "conda       : $(conda --version)"

line "network"
probe_url() { curl -s -o /dev/null -w '%{http_code}' --max-time 15 "$1" 2>/dev/null; }
for u in https://pypi.org/simple/ https://huggingface.co; do
  printf '%-34s %s\n' "$u" "$(probe_url "$u")"
done
# The real question is not whether huggingface.co answers, but whether a weight
# file actually transfers. cdn-lfs returned 000 from this pod on 2026-11-18, so
# pull one small LFS object before committing to an 8.9 GB download.
echo "--- weight transfer test ---"
# cdn-lfs.huggingface.co returned 000 from this pod on 2026-11-18. That host is
# the OLD LFS CDN; Hugging Face now serves large files through Xet
# (us.aws.cdn.hf.co). Whether that one is reachable here has NOT been verified -
# this test is what settles it. Do not assume either way.
for h in https://cdn-lfs.huggingface.co https://us.aws.cdn.hf.co https://transfer.xethub.hf.co; do
  printf '%-34s %s\n' "$(basename "$h")" "$(probe_url "$h")"
done
# The decisive check: follow a real weight URL wherever it redirects and see
# whether bytes actually come back.
LFS_TEST="https://huggingface.co/${MODEL_REPO}/resolve/${MODEL_REV}/model.safetensors.index.json"
LFS_CODE=$(curl -sL -o /dev/null -w '%{http_code}' --max-time 60 "$LFS_TEST" 2>/dev/null)
printf '%-34s %s\n' "index.json (follows redirect)" "${LFS_CODE:-FAILED}"
# And one real LFS object, range-limited so this stays a probe, not a download.
BIG="https://huggingface.co/${MODEL_REPO}/resolve/${MODEL_REV}/model-00001-of-00002.safetensors"
BIG_OUT=$(curl -sL -r 0-1048575 -o /dev/null \
          -w '%{http_code} %{size_download} %{speed_download}' --max-time 90 "$BIG" 2>/dev/null)
printf '%-34s %s\n' "first 1 MiB of a weight shard" "${BIG_OUT:-FAILED}"
echo "   (want: code 200 or 206, size ~1048576, speed in bytes/sec)"
case "$BIG_OUT" in
  20[06]\ 10485*) echo "   => weight download WORKS from this pod." ;;
  *)              echo "   !! weight download does NOT work from this pod."
                  echo "      Fall back to downloading locally and pushing with push_to_pod.sh." ;;
esac

if [ "$PROBE_ONLY" = "1" ]; then
  line "probe only - nothing installed"
  exit 0
fi

# ---------------------------------------------------------------- 1. caches

line "caches on the PVC"
# $HOME is on the node's local disk and does not survive a pod restart, so every
# cache must be redirected or the next pod re-downloads everything.
export PIP_CACHE_DIR="$WS/.cache/pip"
export HF_HOME="$WS/.cache/hf"
export XDG_CACHE_HOME="$WS/.cache/xdg"
export CONDA_PKGS_DIRS="$WS/.cache/conda-pkgs"
mkdir -p "$PIP_CACHE_DIR" "$HF_HOME" "$XDG_CACHE_HOME" "$CONDA_PKGS_DIRS"
echo "PIP_CACHE_DIR=$PIP_CACHE_DIR"
echo "HF_HOME=$HF_HOME"

# ---------------------------------------------------------------- 2. conda env

line "conda env"
if [ -x "$ENV_PREFIX/bin/python" ]; then
  echo "$ENV_PREFIX already exists: $("$ENV_PREFIX/bin/python" -V 2>&1)"
else
  echo "creating $ENV_PREFIX (python $PY_VERSION)"
  echo "NOTE: this is on Lustre and a python env is tens of thousands of small"
  echo "      files, so expect several minutes, not seconds."
  conda create --yes --prefix "$ENV_PREFIX" "python=$PY_VERSION" \
    || die "conda create failed"
fi
PY="$ENV_PREFIX/bin/python"
"$PY" -V || die "env python is not runnable"

# ---------------------------------------------------------------- 3. packages

line "packages"
[ -f "$PROJECT/code/requirements-eval.txt" ] \
  || die "$PROJECT/code/requirements-eval.txt not found - copy the repo to $PROJECT first"

"$PY" -m pip install --upgrade pip || die "pip upgrade failed"
# vllm==0.11.2 brings torch 2.9.0+cu128; do not install torch separately or pip
# will resolve a different build.
"$PY" -m pip install \
  -r "$PROJECT/code/requirements-eval.txt" \
  -r "$PROJECT/code/requirements-train.txt" || die "pip install failed"

line "installed versions"
"$PY" - <<'EOF'
import importlib
for mod in ("torch","transformers","vllm","peft","accelerate","datasets",
            "qwen_vl_utils","huggingface_hub"):
    try:
        m = importlib.import_module(mod)
        print(f"{mod:18}", getattr(m, "__version__", "(no __version__)"))
    except Exception as e:
        print(f"{mod:18} MISSING ({type(e).__name__})")
try:
    import torch
    print("torch CUDA build  :", torch.version.cuda)
    print("cuda available    :", torch.cuda.is_available(), "(False here is expected: the notebook pod has no GPU)")
except Exception:
    pass
EOF

# ---------------------------------------------------------------- 4. upstream

line "Qwen3-VL trainer @ $QWEN_COMMIT"
if [ -d "$QWEN_ROOT/qwen-vl-finetune" ]; then
  echo "$QWEN_ROOT already present"
else
  git init "$QWEN_ROOT" >/dev/null 2>&1
  git -C "$QWEN_ROOT" remote add origin https://github.com/QwenLM/Qwen3-VL.git 2>/dev/null
  git -C "$QWEN_ROOT" sparse-checkout init --cone >/dev/null
  git -C "$QWEN_ROOT" sparse-checkout set qwen-vl-finetune >/dev/null
  git -C "$QWEN_ROOT" fetch --depth 1 origin "$QWEN_COMMIT" || die "git fetch failed"
  git -C "$QWEN_ROOT" checkout --detach FETCH_HEAD || die "git checkout failed"
fi
echo "HEAD: $(git -C "$QWEN_ROOT" rev-parse HEAD)"
QWEN_TRAIN_ROOT="$QWEN_ROOT/qwen-vl-finetune" "$PY" "$PROJECT/code/train/patch_upstream.py" \
  || echo "!! patch_upstream.py failed - check it before training"

# ---------------------------------------------------------------- 5. env file

line "writing $WS/envs/mmdl.env"
cat > "$WS/envs/mmdl.env" <<EOF
# source this before any run on the cluster
export MMDL_ENV=$ENV_PREFIX
export PATH=$ENV_PREFIX/bin:\$PATH
export PIP_CACHE_DIR=$WS/.cache/pip
export HF_HOME=$WS/.cache/hf
export XDG_CACHE_HOME=$WS/.cache/xdg
export QWEN_TRAIN_ROOT=$QWEN_ROOT/qwen-vl-finetune
# the pod sees 128 cores but is limited to $CPU_QUOTA; without this torch
# oversubscribes and slows down
export OMP_NUM_THREADS=$THREADS
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
EOF
cat "$WS/envs/mmdl.env"

line "next"
echo "1) source $WS/envs/mmdl.env"
echo "2) download the model:  huggingface-cli download $MODEL_REPO --revision $MODEL_REV --local-dir $MODEL_DIR"
echo "   (only if the LFS transfer test above returned 200)"
echo "3) submit the GPU probe Job - this pod has no GPU, so module names and"
echo "   GPU count must come from a Job."
