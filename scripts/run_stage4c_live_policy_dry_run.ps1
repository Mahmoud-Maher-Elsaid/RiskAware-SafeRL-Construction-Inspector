$ErrorActionPreference = "Stop"

$Project = "F:\AI\My_Project\RiskAware-SafeRL-Construction-Inspector"
$Python = Join-Path $Project ".venv\Scripts\python.exe"
$ExpectedBranch = "stage-4c-policy-inference-dry-run"

$BaseRunnerPath = Join-Path `
    $Project `
    "scripts\run_stage4b2_runtime.ps1"

$SidecarPath = Join-Path `
    $Project `
    "scripts\stage4c_live_policy_sidecar.py"

$ValidatorPath = Join-Path `
    $Project `
    "scripts\validate_stage4c_live_policy_runtime.py"

$LogDirectory = Join-Path `
    $Project `
    "webots\logs"

$TemporaryRunnerPath = Join-Path `
    $LogDirectory `
    "stage4c_temporary_stage4b2_runner.ps1"

$ProposalLogPath = Join-Path `
    $LogDirectory `
    "stage4c_live_policy_proposals.jsonl"

$SummaryPath = Join-Path `
    $LogDirectory `
    "stage4c_live_policy_runtime_summary.json"

$SidecarStartupPath = Join-Path `
    $LogDirectory `
    "stage4c_policy_sidecar_startup.txt"

$SidecarCompletionPath = Join-Path `
    $LogDirectory `
    "stage4c_policy_sidecar_complete.txt"

$SidecarErrorPath = Join-Path `
    $LogDirectory `
    "stage4c_policy_sidecar_error.log"

$SidecarStdoutPath = Join-Path `
    $LogDirectory `
    "stage4c_policy_sidecar_stdout.log"

$SidecarStderrPath = Join-Path `
    $LogDirectory `
    "stage4c_policy_sidecar_stderr.log"

function Get-JsonLineCount {
    param(
        [string]$Path
    )

    if (-not (Test-Path -LiteralPath $Path)) {
        return 0
    }

    return @(
        Get-Content `
            -LiteralPath $Path `
            -ErrorAction SilentlyContinue |
        Where-Object {
            -not [string]::IsNullOrWhiteSpace($_)
        }
    ).Count
}

function Show-OptionalFile {
    param(
        [string]$Title,
        [string]$Path
    )

    Write-Host ""
    Write-Host $Title -ForegroundColor Yellow

    if (Test-Path -LiteralPath $Path) {
        Get-Content `
            -LiteralPath $Path `
            -Raw
    }
    else {
        Write-Host "File was not created."
    }
}

Set-Location $Project

Write-Host "[1/8] Validating Stage 4C live runtime prerequisites..." -ForegroundColor Cyan

$CurrentBranch = (
    & git branch --show-current
).Trim()

if ($CurrentBranch -ne $ExpectedBranch) {
    throw "Expected branch $ExpectedBranch. Current branch: $CurrentBranch"
}

$RequiredFiles = @(
    $Python,
    $BaseRunnerPath,
    $SidecarPath,
    $ValidatorPath
)

foreach ($RequiredFile in $RequiredFiles) {
    if (-not (Test-Path -LiteralPath $RequiredFile)) {
        throw "Required file was not found: $RequiredFile"
    }
}

Write-Host "[2/8] Validating Stage 4C runtime files..." -ForegroundColor Cyan

& $Python -m py_compile `
    $SidecarPath `
    $ValidatorPath

if ($LASTEXITCODE -ne 0) {
    throw "Stage 4C runtime Python syntax validation failed."
}

& $Python -m ruff check `
    $SidecarPath `
    $ValidatorPath

if ($LASTEXITCODE -ne 0) {
    throw "Stage 4C runtime Ruff validation failed."
}

Write-Host "[3/8] Preparing runtime evidence files..." -ForegroundColor Cyan

New-Item `
    -ItemType Directory `
    -Path $LogDirectory `
    -Force |
    Out-Null

$RuntimeFiles = @(
    $TemporaryRunnerPath,
    $ProposalLogPath,
    $SummaryPath,
    $SidecarStartupPath,
    $SidecarCompletionPath,
    $SidecarErrorPath,
    $SidecarStdoutPath,
    $SidecarStderrPath
)

foreach ($RuntimeFile in $RuntimeFiles) {
    Remove-Item `
        -LiteralPath $RuntimeFile `
        -Force `
        -ErrorAction SilentlyContinue
}

$BaseRunnerContent = Get-Content `
    -LiteralPath $BaseRunnerPath `
    -Raw

$OldBranchDeclaration = '$ExpectedBranch = "stage-4b2-live-webots-sensors"'
$NewBranchDeclaration = '$ExpectedBranch = "stage-4c-policy-inference-dry-run"'

$DeclarationCount = (
    [regex]::Matches(
        $BaseRunnerContent,
        [regex]::Escape(
            $OldBranchDeclaration
        )
    )
).Count

if ($DeclarationCount -ne 1) {
    throw "Expected exactly one Stage 4B2 branch declaration in the base runner."
}

$TemporaryRunnerContent = (
    $BaseRunnerContent.Replace(
        $OldBranchDeclaration,
        $NewBranchDeclaration
    )
)

[System.IO.File]::WriteAllText(
    $TemporaryRunnerPath,
    $TemporaryRunnerContent,
    [System.Text.UTF8Encoding]::new($false)
)

Write-Host "[4/8] Starting the isolated policy sidecar..." -ForegroundColor Cyan

$SidecarProcess = Start-Process `
    -FilePath $Python `
    -ArgumentList @(
        $SidecarPath
    ) `
    -WorkingDirectory $Project `
    -RedirectStandardOutput $SidecarStdoutPath `
    -RedirectStandardError $SidecarStderrPath `
    -PassThru

Write-Host "Sidecar process ID: $($SidecarProcess.Id)" -ForegroundColor DarkGray

$StartupDeadline = (
    Get-Date
).AddSeconds(30)

while ((Get-Date) -lt $StartupDeadline) {
    if (
        Test-Path `
            -LiteralPath $SidecarStartupPath
    ) {
        break
    }

    if ($SidecarProcess.HasExited) {
        break
    }

    Start-Sleep -Milliseconds 250
}

if (
    -not (
        Test-Path `
            -LiteralPath $SidecarStartupPath
    )
) {
    Show-OptionalFile `
        -Title "Sidecar standard output" `
        -Path $SidecarStdoutPath

    Show-OptionalFile `
        -Title "Sidecar standard error" `
        -Path $SidecarStderrPath

    Show-OptionalFile `
        -Title "Sidecar traceback" `
        -Path $SidecarErrorPath

    throw "The Stage 4C policy sidecar did not enter main."
}

Write-Host "Policy sidecar entered main: True" -ForegroundColor Green

Write-Host "[5/8] Running Webots with the validated Stage 4B2 controller..." -ForegroundColor Cyan

powershell.exe `
    -NoProfile `
    -ExecutionPolicy Bypass `
    -File $TemporaryRunnerPath

$WebotsRuntimeExitCode = $LASTEXITCODE

if ($WebotsRuntimeExitCode -ne 0) {
    if (
        -not $SidecarProcess.HasExited
    ) {
        Stop-Process `
            -Id $SidecarProcess.Id `
            -Force `
            -ErrorAction SilentlyContinue
    }

    Show-OptionalFile `
        -Title "Sidecar standard output" `
        -Path $SidecarStdoutPath

    Show-OptionalFile `
        -Title "Sidecar standard error" `
        -Path $SidecarStderrPath

    Show-OptionalFile `
        -Title "Sidecar traceback" `
        -Path $SidecarErrorPath

    throw "The underlying Webots runtime failed."
}

Write-Host "[6/8] Waiting for 30 live policy proposals..." -ForegroundColor Cyan

$SidecarCompleted = (
    $SidecarProcess.WaitForExit(
        30000
    )
)

if (-not $SidecarCompleted) {
    Stop-Process `
        -Id $SidecarProcess.Id `
        -Force `
        -ErrorAction SilentlyContinue

    throw "The policy sidecar did not finish within 30 seconds after Webots."
}

$SidecarProcess.WaitForExit()

if (-not $SidecarProcess.HasExited) {
    Show-OptionalFile `
        -Title "Sidecar standard output" `
        -Path $SidecarStdoutPath

    Show-OptionalFile `
        -Title "Sidecar standard error" `
        -Path $SidecarStderrPath

    Show-OptionalFile `
        -Title "Sidecar traceback" `
        -Path $SidecarErrorPath

    throw "The policy sidecar process did not terminate."
}

if (Test-Path -LiteralPath $SidecarErrorPath) {
    Show-OptionalFile `
        -Title "Sidecar standard output" `
        -Path $SidecarStdoutPath

    Show-OptionalFile `
        -Title "Sidecar standard error" `
        -Path $SidecarStderrPath

    Show-OptionalFile `
        -Title "Sidecar traceback" `
        -Path $SidecarErrorPath

    throw "The policy sidecar created a traceback file."
}

if (-not (Test-Path -LiteralPath $SidecarStdoutPath)) {
    throw "The policy sidecar standard-output log was not created."
}

$SidecarOutput = Get-Content `
    -LiteralPath $SidecarStdoutPath `
    -Raw

if (
    -not $SidecarOutput.Contains(
        "STAGE4C_SIDECAR_COMPLETE proposals=30"
    )
) {
    Show-OptionalFile `
        -Title "Sidecar standard output" `
        -Path $SidecarStdoutPath

    throw "The policy sidecar did not report successful completion."
}

$PolicyProposalCount = Get-JsonLineCount `
    -Path $ProposalLogPath

if ($PolicyProposalCount -lt 30) {
    throw "The sidecar produced fewer than 30 policy proposals."
}

if (
    -not (
        Test-Path `
            -LiteralPath $SidecarCompletionPath
    )
) {
    throw "The sidecar completion marker was not created."
}

Write-Host "Policy proposals: $PolicyProposalCount / 30" -ForegroundColor Green

Write-Host "[7/8] Validating live policy runtime evidence..." -ForegroundColor Cyan

& $Python $ValidatorPath

if ($LASTEXITCODE -ne 0) {
    throw "Stage 4C live policy runtime validation failed."
}

if (-not (Test-Path -LiteralPath $SummaryPath)) {
    throw "Stage 4C runtime summary was not created."
}

$Summary = Get-Content `
    -LiteralPath $SummaryPath `
    -Raw |
    ConvertFrom-Json

if (
    -not [bool]$Summary.live_policy_runtime_verified
) {
    throw "live_policy_runtime_verified is not true."
}

if (
    -not [bool]$Summary.motor_isolation_verified
) {
    throw "motor_isolation_verified is not true."
}

if ([bool]$Summary.motors_connected) {
    throw "The policy was connected to the motors."
}

if ([bool]$Summary.policy_actions_applied) {
    throw "A policy proposal was applied to the robot."
}

if ([int]$Summary.matched_live_sample_count -lt 30) {
    throw "Fewer than 30 live samples received proposals."
}

Write-Host "[8/8] Running final repository validation..." -ForegroundColor Cyan

& $Python -m pytest

if ($LASTEXITCODE -ne 0) {
    throw "The complete repository test suite failed."
}

git diff --check

if ($LASTEXITCODE -ne 0) {
    throw "Repository formatting validation failed."
}

Write-Host ""
Write-Host "Stage 4C1B live policy summary" -ForegroundColor Cyan

Get-Content `
    -LiteralPath $SummaryPath `
    -Raw

Write-Host ""
Write-Host "Git status" -ForegroundColor Cyan

git status --short

Write-Host ""
Write-Host "Stage 4C1B live policy dry run passed. No commit was created." -ForegroundColor Green
