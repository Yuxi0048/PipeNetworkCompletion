param(
    [string]$CondaExe = "$env:USERPROFILE\miniforge3\Scripts\conda.exe",
    [string]$Prefix = ".conda\pipe-network-completion"
)

$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

if (-not (Test-Path -LiteralPath $CondaExe)) {
    throw "Miniforge conda was not found at '$CondaExe'. Pass -CondaExe with the full path to conda.exe."
}

if (Test-Path -LiteralPath $Prefix) {
    & $CondaExe env update -p $Prefix -f environment.yml --prune
} else {
    & $CondaExe env create -p $Prefix -f environment.yml
}

# Install the package in editable mode so scripts/ can `import
# pipe_network_completion` without the sys.path shim.
& (Join-Path $Prefix "python.exe") -m pip install -e .

& (Join-Path $Prefix "python.exe") scripts\verify_environment.py

