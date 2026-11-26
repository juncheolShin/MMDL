#!/usr/bin/env bash
# Pod-side pre-flight check. Run INSIDE the training pod, paste the whole output back.
# Read-only except for one small write test under each candidate data dir.
#
#   bash precheck.sh            # or: curl/paste the file, then run it
#
# It never prints tokens, kubeconfigs or env vars that look like secrets.

set -uo pipefail
line() { printf '\n===== %s =====\n' "$1"; }

line "identity / limits"
echo "whoami    : $(id -un 2>/dev/null || echo '?') (uid=$(id -u) gid=$(id -g))"
echo "hostname  : $(hostname)"
echo "cpus      : $(nproc 2>/dev/null || echo '?')"
free -h 2>/dev/null | sed -n '1,2p'
echo "--- cgroup limits (what the pod may actually use) ---"
for f in /sys/fs/cgroup/memory.max /sys/fs/cgroup/memory/memory.limit_in_bytes \
         /sys/fs/cgroup/cpu.max /sys/fs/cgroup/cpu/cpu.cfs_quota_us; do
  [ -r "$f" ] && echo "$f = $(cat "$f")"
done

line "gpu / driver"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=index,name,memory.total,memory.used,driver_version,compute_cap \
             --format=csv 2>/dev/null || nvidia-smi
else
  echo "nvidia-smi NOT FOUND - no GPU visible to this pod"
fi

line "python / torch"
command -v python3 >/dev/null 2>&1 && echo "python3: $(python3 -V 2>&1) at $(command -v python3)"
command -v python  >/dev/null 2>&1 && echo "python : $(python -V 2>&1) at $(command -v python)"
PY=$(command -v python3 || command -v python)
if [ -n "${PY:-}" ]; then
  "$PY" - <<'EOF'
import importlib, sys
print("executable:", sys.executable)
try:
    import torch
except Exception as e:
    print("torch: NOT IMPORTABLE ->", type(e).__name__, e)
else:
    print("torch        :", torch.__version__, "| built for CUDA", torch.version.cuda)
    print("cuda available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            print(f"  gpu{i}: {p.name} {p.total_memory/2**30:.1f}GiB sm_{p.major}{p.minor}")
        print("bf16 supported:", torch.cuda.is_bf16_supported())
for mod in ("transformers", "peft", "trl", "accelerate", "datasets",
            "bitsandbytes", "flash_attn", "deepspeed", "vllm", "qwen_vl_utils"):
    try:
        m = importlib.import_module(mod)
        print(f"{mod:16}", getattr(m, "__version__", "(no __version__)"))
    except Exception as e:
        print(f"{mod:16} MISSING ({type(e).__name__})")
EOF
else
  echo "no python on PATH"
fi

line "storage"
df -h 2>/dev/null | awk 'NR==1 || $NF ~ /^(\/|\/data|\/mnt|\/workspace|\/opt|\/home)/'
echo "--- write test ---"
for d in /data /mnt/data /workspace /opt/mmdl "$HOME"; do
  [ -d "$d" ] || { printf '%-14s MISSING\n' "$d"; continue; }
  t="$d/.precheck_$$"
  if (echo ok > "$t") 2>/dev/null; then
    printf '%-14s WRITABLE   free=%s\n' "$d" "$(df -h "$d" 2>/dev/null | awk 'NR==2{print $4}')"
    rm -f "$t"
  else
    printf '%-14s READ-ONLY  free=%s\n' "$d" "$(df -h "$d" 2>/dev/null | awk 'NR==2{print $4}')"
  fi
done
echo "--- persistence: are these real volumes or the container layer? ---"
mount 2>/dev/null | grep -E ' /(data|mnt|workspace) ' || echo "(no separate mount on /data, /mnt or /workspace - writes there are EPHEMERAL)"

line "network"
for url in https://huggingface.co https://cdn-lfs.huggingface.co https://pypi.org; do
  code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 10 "$url" 2>/dev/null)
  printf '%-36s %s\n' "$url" "${code:-FAILED}"
done
echo "proxy env set: $( (env | grep -iE '^(http|https|no)_proxy=' | cut -d= -f1 | tr '\n' ' ') || echo none)"

line "done"
echo "Paste everything above back. No secrets are included."
