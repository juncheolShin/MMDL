#!/usr/bin/env bash
# Wait for bootstrap_env.sh to finish, then download the model. Runs ON THE POD,
# detached, so it survives the laptop going offline:
#
#   setsid nohup bash /workspace/MMDL/code/cluster/stage2_model.sh \
#       > /workspace/logs/stage2.log 2>&1 < /dev/null &
#
# Writes /workspace/logs/model.rc when done (0 = model present and verified).
# Safe to re-run: snapshot_download resumes and skips files already complete.

set -uo pipefail

WS=${WS:-/workspace}
LOGS=$WS/logs
ENV_FILE=${ENV_FILE:-$WS/envs/mmdl.env}
MODEL_REPO=${MODEL_REPO:-Qwen/Qwen3-VL-4B-Instruct}
MODEL_REV=${MODEL_REV:-ebb281ec70b05090aa6165b016eac8ec08e71b17}
MODEL_DIR=${MODEL_DIR:-$WS/models/Qwen3-VL-4B-Instruct}
WAIT_SECONDS=${WAIT_SECONDS:-7200}

mkdir -p "$LOGS"
stamp() { date -u '+%Y-%m-%dT%H:%M:%SZ'; }
say()   { echo "[$(stamp)] $*"; }

say "stage2 starting; waiting for bootstrap to finish"

waited=0
while [ ! -f "$LOGS/bootstrap.rc" ]; do
  if [ "$waited" -ge "$WAIT_SECONDS" ]; then
    say "gave up waiting for bootstrap after ${WAIT_SECONDS}s"
    echo 90 > "$LOGS/model.rc"; exit 90
  fi
  sleep 15; waited=$(( waited + 15 ))
done

BRC=$(tr -d '[:space:]' < "$LOGS/bootstrap.rc")
say "bootstrap finished with rc=$BRC after ${waited}s of waiting"
if [ "$BRC" != "0" ]; then
  say "bootstrap failed - not downloading the model. See $LOGS/bootstrap.log"
  echo 91 > "$LOGS/model.rc"; exit 91
fi

# shellcheck disable=SC1090
source "$ENV_FILE" || { say "cannot source $ENV_FILE"; echo 92 > "$LOGS/model.rc"; exit 92; }
say "python: $(command -v python) $(python -V 2>&1)"

say "downloading $MODEL_REPO @ $MODEL_REV -> $MODEL_DIR"
MODEL_REPO="$MODEL_REPO" MODEL_REV="$MODEL_REV" MODEL_DIR="$MODEL_DIR" python - <<'PY'
import os, sys, time
from huggingface_hub import snapshot_download

repo, rev, dest = os.environ["MODEL_REPO"], os.environ["MODEL_REV"], os.environ["MODEL_DIR"]
# local_dir keeps real files instead of a symlink cache, so the directory can be
# copied or mounted elsewhere without the cache coming along.
for attempt in range(1, 6):
    try:
        t0 = time.time()
        path = snapshot_download(repo_id=repo, revision=rev, local_dir=dest,
                                 max_workers=8)
        print(f"downloaded to {path} in {time.time()-t0:.0f}s", flush=True)
        break
    except Exception as exc:                      # network drops are expected here
        print(f"attempt {attempt} failed: {type(exc).__name__}: {exc}", flush=True)
        if attempt == 5:
            sys.exit(1)
        time.sleep(30 * attempt)                  # resumes from what is already on disk
PY
RC=$?
if [ "$RC" != "0" ]; then
  say "model download failed rc=$RC"
  echo "$RC" > "$LOGS/model.rc"; exit "$RC"
fi

say "verifying the snapshot"
MODEL_DIR="$MODEL_DIR" python - <<'PY'
import json, os, pathlib, sys
d = pathlib.Path(os.environ["MODEL_DIR"])
idx = d / "model.safetensors.index.json"
missing = []
if idx.is_file():
    shards = set(json.loads(idx.read_text())["weight_map"].values())
    missing = sorted(s for s in shards if not (d / s).is_file())
total = sum(f.stat().st_size for f in d.rglob("*") if f.is_file())
print("files :", sum(1 for f in d.rglob('*') if f.is_file()))
print("size  : %.2f GiB" % (total / 2**30))
print("config:", (d / "config.json").is_file())
if missing:
    print("MISSING SHARDS:", missing); sys.exit(2)
print("all shards present")
PY
RC=$?
echo "$RC" > "$LOGS/model.rc"
say "stage2 done rc=$RC"
exit "$RC"
