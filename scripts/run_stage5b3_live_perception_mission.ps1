param(
    [string]$RepoRoot = "F:\AI\My_Project\RiskAware-SafeRL-Construction-Inspector"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Quote-Argument {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Value
    )

    return '"' + $Value.Replace('"', '\"') + '"'
}

$PythonExecutable = Join-Path $RepoRoot ".venv\Scripts\python.exe"
$SidecarScript = Join-Path $RepoRoot "scripts\run_stage5b3_live_perception_sidecar.py"
$Stage5A3Launcher = Join-Path $RepoRoot "scripts\run_stage5a3_closed_loop_mission.ps1"
$ValidatorScript = Join-Path $RepoRoot "scripts\validate_stage5b3_live_perception_mission.py"
$ConfigPath = Join-Path $RepoRoot "configs\perception\stage5b_live_perception.json"
$EvidenceRoot = Join-Path $RepoRoot "webots\logs\stage5a3_closed_loop\evidence_frames"
$OutputRoot = Join-Path $RepoRoot "webots\logs\stage5b3_live_perception"
$ReportPath = Join-Path $RepoRoot "reports\perception\stage5b\stage5b3_runtime_validation.json"
$StopFile = Join-Path $OutputRoot "stop.request"
$ReadyFile = Join-Path $OutputRoot "perception_ready.json"
$FailureFile = Join-Path $OutputRoot "perception_failure.json"
$StdoutPath = Join-Path $OutputRoot "sidecar_stdout.log"
$StderrPath = Join-Path $OutputRoot "sidecar_stderr.log"

Set-Location $RepoRoot

foreach ($RequiredPath in @(
    $PythonExecutable,
    $SidecarScript,
    $Stage5A3Launcher,
    $ValidatorScript,
    $ConfigPath
)) {
    if (-not (Test-Path -LiteralPath $RequiredPath -PathType Leaf)) {
        throw "Required file was not found: $RequiredPath"
    }
}

foreach ($Directory in @($OutputRoot, $EvidenceRoot)) {
    if (Test-Path -LiteralPath $Directory) {
        Remove-Item -LiteralPath $Directory -Recurse -Force
    }

    New-Item -ItemType Directory -Path $Directory -Force | Out-Null
}

if (Test-Path -LiteralPath $ReportPath) {
    Remove-Item -LiteralPath $ReportPath -Force
}

$Arguments = @(
    (Quote-Argument $SidecarScript),
    "--project-root",
    (Quote-Argument $RepoRoot),
    "--config",
    (Quote-Argument $ConfigPath),
    "--evidence-root",
    (Quote-Argument $EvidenceRoot),
    "--output",
    (Quote-Argument $OutputRoot),
    "--stop-file",
    (Quote-Argument $StopFile),
    "--timeout-seconds",
    "900"
)

$SidecarProcess = Start-Process `
    -FilePath $PythonExecutable `
    -ArgumentList $Arguments `
    -WorkingDirectory $RepoRoot `
    -RedirectStandardOutput $StdoutPath `
    -RedirectStandardError $StderrPath `
    -WindowStyle Hidden `
    -PassThru

Write-Host "Stage 5B3 sidecar PID: $($SidecarProcess.Id)"

$ReadyDeadline = (Get-Date).AddSeconds(90)

while (-not (Test-Path -LiteralPath $ReadyFile)) {
    $SidecarProcess.Refresh()

    if ($SidecarProcess.HasExited) {
        if (Test-Path -LiteralPath $StdoutPath) {
            Get-Content -LiteralPath $StdoutPath
        }

        if (Test-Path -LiteralPath $StderrPath) {
            Get-Content -LiteralPath $StderrPath
        }

        throw "The Stage 5B3 sidecar exited before readiness."
    }

    if (Test-Path -LiteralPath $FailureFile) {
        Get-Content -LiteralPath $FailureFile
        Stop-Process -Id $SidecarProcess.Id -Force -ErrorAction SilentlyContinue
        throw "The Stage 5B3 sidecar reported a startup failure."
    }

    if ((Get-Date) -gt $ReadyDeadline) {
        Stop-Process -Id $SidecarProcess.Id -Force -ErrorAction SilentlyContinue
        throw "Timed out waiting for the Stage 5B3 sidecar."
    }

    Start-Sleep -Milliseconds 250
}

Write-Host "Stage 5B3 perception backend is ready."

$MissionExitCode = -1

try {
    & powershell.exe `
        -NoProfile `
        -ExecutionPolicy Bypass `
        -File $Stage5A3Launcher `
        -Mode Interactive `
        -TimeoutSeconds 900

    $MissionExitCode = $LASTEXITCODE
}
finally {
    New-Item -ItemType File -Path $StopFile -Force | Out-Null
}

$SidecarDeadline = (Get-Date).AddSeconds(90)

while ($true) {
    $SidecarProcess.Refresh()

    if ($SidecarProcess.HasExited) {
        break
    }

    if ((Get-Date) -gt $SidecarDeadline) {
        Stop-Process -Id $SidecarProcess.Id -Force -ErrorAction SilentlyContinue
        throw "The Stage 5B3 sidecar did not stop cleanly."
    }

    Start-Sleep -Milliseconds 250
}

$SidecarProcess.WaitForExit()
$SidecarProcess.Refresh()

$SidecarExitCode = $null

try {
    $SidecarExitCode = $SidecarProcess.ExitCode
}
catch {
    $SidecarExitCode = $null
}

if ($null -eq $SidecarExitCode) {
    $SidecarSummaryPath = Join-Path `
        $OutputRoot `
        "perception_summary.json"

    if (Test-Path -LiteralPath $SidecarSummaryPath -PathType Leaf) {
        $SidecarSummary = Get-Content `
            -LiteralPath $SidecarSummaryPath `
            -Raw |
        ConvertFrom-Json

        $SummaryVerified = (
            $SidecarSummary.runtime_verified -eq $true -and
            $SidecarSummary.cv_model_connected -eq $true -and
            [int]$SidecarSummary.failure_count -eq 0
        )

        if ($SummaryVerified) {
            $SidecarExitCode = 0
        }
        else {
            $SidecarExitCode = 1
        }
    }
    else {
        $SidecarExitCode = 1
    }
}

Write-Host "Resolved Stage 5B3 sidecar exit code: $SidecarExitCode"

Write-Host ""
Write-Host "============================================================"
Write-Host "Stage 5B3 Sidecar Output"
Write-Host "============================================================"

if (Test-Path -LiteralPath $StdoutPath) {
    Get-Content -LiteralPath $StdoutPath
}

if (
    (Test-Path -LiteralPath $StderrPath) -and
    (Get-Item -LiteralPath $StderrPath).Length -gt 0
) {
    Write-Host "Sidecar stderr:"
    Get-Content -LiteralPath $StderrPath
}

if ($MissionExitCode -ne 0) {
    throw "Stage 5A3 mission failed with exit code $MissionExitCode."
}

if ($SidecarExitCode -ne 0) {
    if (Test-Path -LiteralPath $FailureFile) {
        Get-Content -LiteralPath $FailureFile
    }

    throw "Stage 5B3 sidecar failed with exit code $SidecarExitCode."
}

& $PythonExecutable `
    $ValidatorScript `
    --project-root $RepoRoot `
    --evidence-root $EvidenceRoot `
    --sidecar-output $OutputRoot `
    --report $ReportPath

if ($LASTEXITCODE -ne 0) {
    throw "Stage 5B3 validation failed."
}

Write-Host ""
Write-Host "============================================================"
Write-Host "Stage 5B3 Mission Completed"
Write-Host "============================================================"
Write-Host "Report: $ReportPath"
Write-Host "Output: $OutputRoot"
Write-Host "STAGE5B3_CONTROLLER_SYNCHRONIZED_LIVE_PERCEPTION_READY"