# Keep every cache on D:. C: is nearly full on this machine.
# Usage (bash, from the repo root):   source configs/env.sh

_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
_cache="$_root/.cache"

export HF_HOME="$_cache/hf"              # HF hub + xet chunk cache
export PIP_CACHE_DIR="$_cache/pip"       # wheel cache
export TORCH_HOME="$_cache/torch"        # torch.hub weights
export XDG_CACHE_HOME="$_cache/xdg"      # matplotlib, fontconfig, misc

mkdir -p "$HF_HOME" "$PIP_CACHE_DIR" "$TORCH_HOME" "$XDG_CACHE_HOME"

echo "caches -> $_cache"
echo "NOTE: CLAUDE_CODE_TMPDIR is NOT set here; it must be set before Claude Code starts."
unset _root _cache
