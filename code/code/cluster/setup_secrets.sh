#!/usr/bin/env bash
# Turn the plain-text files in secrets/ into Kubernetes Secrets.
#
#   bash code/cluster/setup_secrets.sh            # create or update
#   bash code/cluster/setup_secrets.sh --check    # report only
#
# Why not `kubectl create secret --from-file=K=secrets/k` directly:
#   * it stores the file's bytes verbatim, trailing newline included, and a key
#     with "\n" on the end fails authentication with an unhelpful message
#   * `--from-literal` would avoid that but puts the key in the process
#     arguments, where `ps` can read it
# So each value is stripped, written to a 0600 temp file, handed to kubectl by
# path, and the temp file is removed - the key never appears in argv, in the
# shell history, or in a rendered manifest.
#
# The Job templates read these with secretKeyRef, so no manifest ever holds a key.

set -uo pipefail
export MSYS_NO_PATHCONV=1
export MSYS2_ARG_CONV_EXCL='*'

KUBECONFIG_FILE=${KUBECONFIG_FILE:-reality-lab.yaml}
NAMESPACE=${NAMESPACE:-p-test2}
OWNER_TAG=${OWNER_TAG:-yeunsuh}
SECRETS_DIR=${SECRETS_DIR:-secrets}
PLACEHOLDER='PASTE_THE_VALUE_HERE_AND_DELETE_THIS_LINE'

CHECK=0
[ "${1:-}" = "--check" ] && CHECK=1

K="kubectl --kubeconfig $KUBECONFIG_FILE -n $NAMESPACE"

# secret name : env var name : source file
ENTRIES="
$OWNER_TAG-anthropic:ANTHROPIC_API_KEY:anthropic_api_key
$OWNER_TAG-wandb:WANDB_API_KEY:wandb_api_key
"

[ -d "$SECRETS_DIR" ] || { echo "no $SECRETS_DIR/ directory; see secrets/README.md" >&2; exit 1; }

# A RELATIVE path: kubectl.exe is a Windows binary and, with MSYS_NO_PATHCONV
# set (needed elsewhere), a Git Bash absolute path like /tmp/xxx reaches it
# unconverted and it reports "cannot find the path specified".
TMPDIR_LOCAL=./.secret_staging
rm -rf "$TMPDIR_LOCAL"
mkdir -p "$TMPDIR_LOCAL"
chmod 700 "$TMPDIR_LOCAL" 2>/dev/null || true
trap 'rm -rf "$TMPDIR_LOCAL"' EXIT

printf 'namespace : %s\n' "$NAMESPACE"
printf 'source    : %s/\n\n' "$SECRETS_DIR"

created=0 skipped=0 failed=0
for entry in $ENTRIES; do
  [ -n "$entry" ] || continue
  secret=${entry%%:*}
  rest=${entry#*:}
  envname=${rest%%:*}
  file="$SECRETS_DIR/${rest#*:}"

  if [ ! -f "$file" ]; then
    printf '%-26s SKIP   %s does not exist\n' "$secret" "$file"
    skipped=$((skipped + 1)); continue
  fi
  # strip surrounding whitespace, including the trailing newline
  value=$(tr -d '\r' < "$file" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//' | head -c 4096)
  if [ -z "$value" ]; then
    printf '%-26s SKIP   %s is empty\n' "$secret" "$file"
    skipped=$((skipped + 1)); continue
  fi
  if [ "$value" = "$PLACEHOLDER" ]; then
    printf '%-26s SKIP   %s still holds the placeholder\n' "$secret" "$file"
    skipped=$((skipped + 1)); continue
  fi

  # Show enough to recognise the value, never enough to use it.
  n=${#value}
  masked="${value:0:6}...${value: -4} (${n} chars)"

  if [ "$CHECK" = "1" ]; then
    if $K get secret "$secret" >/dev/null 2>&1; then
      printf '%-26s EXISTS would update with %s -> %s\n' "$secret" "$envname" "$masked"
    else
      printf '%-26s ABSENT would create with %s -> %s\n' "$secret" "$envname" "$masked"
    fi
    continue
  fi

  payload="$TMPDIR_LOCAL/$envname"
  (umask 077; printf '%s' "$value" > "$payload")

  # --dry-run + apply so this both creates and updates
  if $K create secret generic "$secret" \
        --from-file="$envname=$payload" \
        --dry-run=client -o yaml 2>/dev/null | $K apply -f - >/dev/null 2>&1; then
    printf '%-26s OK     %s -> %s\n' "$secret" "$envname" "$masked"
    created=$((created + 1))
  else
    printf '%-26s FAILED %s\n' "$secret" "$envname"
    failed=$((failed + 1))
  fi
  rm -f "$payload"
done

echo
if [ "$CHECK" = "1" ]; then
  echo "check only - nothing changed"
  exit 0
fi
printf 'applied %d, skipped %d, failed %d\n' "$created" "$skipped" "$failed"


if [ "$created" -gt 0 ]; then
  echo
  echo "in the namespace now (key NAMES only):"
  # NEVER print .data. It holds the values, and base64 is encoding, not
  # protection - `kubectl get secret -o yaml` output pasted anywhere is a leak.
  for s in $($K get secrets -o jsonpath='{range .items[*]}{.metadata.name}{"\n"}{end}' 2>/dev/null | grep "$OWNER_TAG"); do
    keys=$($K get secret "$s" -o go-template='{{range $k, $v := .data}}{{$k}}{{" "}}{{end}}' 2>/dev/null)
    age=$($K get secret "$s" -o jsonpath='{.metadata.creationTimestamp}' 2>/dev/null)
    printf '  %-24s keys: %-22s %s\n' "$s" "$keys" "$age"
  done
  echo
  echo "Anyone who can read Secrets in $NAMESPACE can read these values."
  echo "Do not run 'kubectl get secret -o yaml' and paste the output anywhere."
fi
[ "$failed" -eq 0 ] || exit 1
