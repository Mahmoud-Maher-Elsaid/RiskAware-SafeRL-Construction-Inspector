param(
    [ValidateSet("Interactive", "Validation")]
    [string]$Mode = "Validation",
    [int]$TimeoutSeconds = 600,
    [switch]$LaunchCheckOnly
)

$ErrorActionPreference = "Stop"
$Project = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $Project ".venv\Scripts\python.exe"
$Runner = Join-Path $PSScriptRoot "run_stage5a3_closed_loop_mission.py"

if (-not (Test-Path -LiteralPath $Python)) {
    throw "The repository virtual environment is missing: $Python"
}

$Arguments = @(
    $Runner,
    "--project", $Project,
    "--mode", $Mode.ToLowerInvariant(),
    "--timeout", $TimeoutSeconds
)
if ($LaunchCheckOnly) {
    $Arguments += "--launch-check-only"
}
& $Python @Arguments
if ($LASTEXITCODE -ne 0) {
    throw "Stage 5A3 launcher failed with exit code $LASTEXITCODE."
}
