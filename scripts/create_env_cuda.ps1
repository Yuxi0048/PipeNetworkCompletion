# Workstream: Claude + Codex merge
#
# Create the GPU/CUDA conda env for PipeNetworkCompletion.
# Mirrors scripts/create_env.ps1 but installs from environment-cuda.yml into
# a separate prefix so the CPU env remains intact for CI / checkpoint eval.
#
# Usage:
#   .\scripts\create_env_cuda.ps1
#   .\scripts\create_env_cuda.ps1 -CondaExe "C:\path\to\conda.exe"
#   .\scripts\create_env_cuda.ps1 -Prefix ".conda\pipe-network-completion-cuda"

param(
    [string]$CondaExe = "$env:USERPROFILE\miniforge3\Scripts\conda.exe",
    [string]$Prefix = ".conda\pipe-network-completion-cuda",
    [string]$EnvFile = "environment-cuda.yml"
)

$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

if (-not (Test-Path -LiteralPath $CondaExe)) {
    throw "Miniforge conda was not found at '$CondaExe'. Pass -CondaExe with the full path to conda.exe."
}

if (-not (Test-Path -LiteralPath $EnvFile)) {
    throw "Env file '$EnvFile' not found. Run from the repo root."
}

Write-Host "Creating GPU env at $Prefix from $EnvFile (this can take 10-30 minutes and ~5 GB of downloads)..."

if (Test-Path -LiteralPath $Prefix) {
    & $CondaExe env update -p $Prefix -f $EnvFile --prune
} else {
    & $CondaExe env create -p $Prefix -f $EnvFile
}

# Editable install of the local package.
& (Join-Path $Prefix "python.exe") -m pip install -e .

# Run the GPU-aware environment check.
& (Join-Path $Prefix "python.exe") scripts\verify_environment_cuda.py
