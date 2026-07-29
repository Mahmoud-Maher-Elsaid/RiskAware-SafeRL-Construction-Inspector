param(
    [string]$RepoRoot = "F:\AI\My_Project\RiskAware-SafeRL-Construction-Inspector"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$MissionScript = Join-Path `
    $RepoRoot `
    "scripts\run_stage5b3_live_perception_mission.ps1"

if (-not (Test-Path -LiteralPath $MissionScript -PathType Leaf)) {
    throw "Mission script was not found: $MissionScript"
}

Set-Location $RepoRoot

& powershell.exe `
    -NoProfile `
    -ExecutionPolicy Bypass `
    -File $MissionScript `
    -RepoRoot $RepoRoot

if ($LASTEXITCODE -ne 0) {
    throw "Verified human first-person Stage 5B3 mission failed with exit code $LASTEXITCODE."
}

Write-Host ""
Write-Host "VERIFIED_HUMAN_FIRST_PERSON_STAGE5B3=PASSED"
