# Keep every cache on D:. C: is nearly full on this machine.
# Usage (PowerShell, from the repo root):   . .\configs\env.ps1

$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $PSScriptRoot
$Cache = Join-Path $Root '.cache'

$env:HF_HOME         = Join-Path $Cache 'hf'       # HF hub + xet chunk cache
$env:PIP_CACHE_DIR   = Join-Path $Cache 'pip'      # wheel cache
$env:TORCH_HOME      = Join-Path $Cache 'torch'    # torch.hub weights
$env:XDG_CACHE_HOME  = Join-Path $Cache 'xdg'      # matplotlib, fontconfig, misc

foreach ($p in @($env:HF_HOME, $env:PIP_CACHE_DIR, $env:TORCH_HOME, $env:XDG_CACHE_HOME)) {
    if (-not (Test-Path $p)) { New-Item -ItemType Directory -Force -Path $p | Out-Null }
}

Write-Host "caches -> $Cache"
Write-Host "NOTE: CLAUDE_CODE_TMPDIR is NOT set here; it must be set before Claude Code starts."
