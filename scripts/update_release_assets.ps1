param(
    [string]$Tag = "v1.0.0",
    [string[]]$Files = @(
        "data/experiments/data_MH_Road_attr.pkl",
        "data/processed/split_shapefiles/train.dbf"
    )
)

$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

if (-not (Get-Command gh -ErrorAction SilentlyContinue)) {
    throw "GitHub CLI (gh) is not installed. Install it with 'winget install --id GitHub.cli', then run 'gh auth login'."
}

$missing = $Files | Where-Object { -not (Test-Path -LiteralPath $_) }
if ($missing) {
    throw "These files are missing locally and cannot be uploaded:`n  $($missing -join "`n  ")"
}

Write-Host "Uploading $($Files.Count) file(s) to release $Tag (replacing any existing asset of the same name):"
$Files | ForEach-Object { Write-Host "  $_" }

& gh release upload $Tag @Files --clobber
