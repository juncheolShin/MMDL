#!/usr/bin/env bash
# Push a local directory to the notebook pod as a few large archives.
# Run from the repo root ON THE LOCAL MACHINE (Git Bash on Windows is fine).
#
#   bash code/cluster/push_to_pod.sh training_data/raw /workspace/MMDL/training_data
#   bash code/cluster/push_to_pod.sh --dry-run training_data/normalized /workspace/MMDL/training_data
#
# Why archives instead of `kubectl cp <dir>`:
#   * kubectl cp streams a tar through the API server. It has no resume, so a
#     broken 9 GB transfer starts over from zero.
#   * /workspace and /data are the same Lustre filesystem. Lustre is poor at many
#     small files, and our normalized output is ~59,000 loose images. Writing them
#     one by one over kubectl cp is slow for us and rude on a shared filesystem.
#   * Splitting into chunks means a failure costs one chunk, not the whole set.
#
# So: tar locally -> split into chunks -> cp each chunk -> verify sha256 on the
# pod -> reassemble and extract there. Nothing is deleted on either side.

set -uo pipefail

# Git Bash on Windows rewrites any argument that looks like a POSIX path, so
# `yeunsuh-0:/workspace/x` becomes `yeunsuh-0:C:/Program Files/Git/workspace/x`
# and the copy lands somewhere meaningless. These two switches turn that off.
export MSYS_NO_PATHCONV=1
export MSYS2_ARG_CONV_EXCL='*'

KUBECONFIG_FILE=${KUBECONFIG_FILE:-reality-lab.yaml}
NAMESPACE=${NAMESPACE:-p-test2}
POD=${POD:-yeunsuh-0}
CONTAINER=${CONTAINER:-yeunsuh}
CHUNK=${CHUNK:-1G}
STAGE=${STAGE:-/workspace/.staging}

DRY=0
if [ "${1:-}" = "--dry-run" ]; then DRY=1; shift; fi
SRC=${1:?usage: push_to_pod.sh [--dry-run] <local-dir> <remote-parent-dir>}
DEST=${2:?usage: push_to_pod.sh [--dry-run] <local-dir> <remote-parent-dir>}

K="kubectl --kubeconfig $KUBECONFIG_FILE -n $NAMESPACE"
RUN="$K exec $POD -c $CONTAINER --"

[ -d "$SRC" ] || { echo "no such directory: $SRC" >&2; exit 1; }

NAME=$(basename "$SRC")
# Staging must be a RELATIVE path. kubectl.exe is a Windows binary, and with
# MSYS_NO_PATHCONV set (needed so the remote path survives) a Git Bash absolute
# path like /tmp/xxx reaches it unconverted and it reports "doesn't exist in
# local filesystem". A relative path resolves correctly on both sides.
TMP=./.push_staging/$NAME
rm -rf ./.push_staging
mkdir -p "$TMP"
trap 'rm -rf ./.push_staging' EXIT

files=$(find "$SRC" -type f | wc -l)
bytes=$(du -sb "$SRC" 2>/dev/null | cut -f1)
printf 'source   : %s\n' "$SRC"
printf 'files    : %s\n' "$files"
printf 'size     : %s bytes (%.2f GiB)\n' "$bytes" "$(echo "$bytes" | awk '{print $1/1073741824}')"
printf 'pod      : %s/%s (container %s)\n' "$NAMESPACE" "$POD" "$CONTAINER"
printf 'remote   : %s/%s\n' "$DEST" "$NAME"
printf 'chunk    : %s\n\n' "$CHUNK"

if [ "$DRY" = "1" ]; then echo "dry run - nothing sent"; exit 0; fi

echo "[1/5] packing locally"
# Store paths relative to the source's parent, so extraction recreates <NAME>/.
tar -C "$(dirname "$SRC")" -cf - "$NAME" | split -b "$CHUNK" - "$TMP/$NAME.tar.part."
ls -la "$TMP"
( cd "$TMP" && sha256sum "$NAME".tar.part.* > "$NAME.sha256" )

echo
echo "[2/5] creating $STAGE on the pod"
$RUN mkdir -p "$STAGE" "$DEST" || exit 1

echo
echo "[3/5] copying chunks"
for part in "$TMP/$NAME".tar.part.*; do
  base=$(basename "$part")
  echo "  -> $base ($(du -h "$part" | cut -f1))"
  $K cp "$part" "$POD:$STAGE/$base" -c "$CONTAINER" || { echo "FAILED on $base" >&2; exit 1; }
done
$K cp "$TMP/$NAME.sha256" "$POD:$STAGE/$NAME.sha256" -c "$CONTAINER" || exit 1

echo
echo "[4/5] verifying on the pod"
# kubectl cp can truncate silently; the checksum is the only thing that proves
# the bytes arrived intact.
$RUN bash -c "cd $STAGE && sha256sum -c $NAME.sha256" || {
  echo "CHECKSUM MISMATCH - not extracting. Chunks are left in $STAGE." >&2
  exit 1
}

echo
echo "[5/5] extracting into $DEST"
$RUN bash -c "cat $STAGE/$NAME.tar.part.* | tar -C $DEST -xf - && rm -f $STAGE/$NAME.tar.part.* $STAGE/$NAME.sha256" || exit 1
$RUN bash -c "find $DEST/$NAME -type f | wc -l | xargs echo 'files on pod:'"

echo
echo "done: $DEST/$NAME"
