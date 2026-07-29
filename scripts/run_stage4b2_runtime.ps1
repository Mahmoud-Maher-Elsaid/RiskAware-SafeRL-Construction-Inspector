$ErrorActionPreference = "Stop"

$Project = "F:\AI\My_Project\RiskAware-SafeRL-Construction-Inspector"
$Python = Join-Path $Project ".venv\Scripts\python.exe"
$ExpectedBranch = "stage-4b2-live-webots-sensors"

$WebotsHome = "C:\Program Files\Webots"
$WebotsBinDirectory = Join-Path `
    $WebotsHome `
    "msys64\mingw64\bin"

$LauncherCandidates = @(
    (
        Join-Path `
            $WebotsBinDirectory `
            "webotsw.exe"
    ),
    (
        Join-Path `
            $WebotsBinDirectory `
            "webots.exe"
    )
)

$LiveWorldPath = Join-Path `
    $Project `
    "webots\worlds\construction_site_stage4b_live_bridge.wbt"

$ValidatorPath = Join-Path `
    $Project `
    "scripts\validate_stage4b2_runtime.py"

$LogDirectory = Join-Path `
    $Project `
    "webots\logs"

$RobotLogPath = Join-Path `
    $LogDirectory `
    "stage4b2_live_robot.jsonl"

$SupervisorLogPath = Join-Path `
    $LogDirectory `
    "stage4b2_live_supervisor.jsonl"

$SummaryPath = Join-Path `
    $LogDirectory `
    "stage4b2_runtime_summary.json"

$RobotStartupPath = Join-Path `
    $LogDirectory `
    "stage4b2_robot_startup.txt"

$SupervisorStartupPath = Join-Path `
    $LogDirectory `
    "stage4b2_supervisor_startup.txt"

$RobotErrorPath = Join-Path `
    $LogDirectory `
    "stage4b2_robot_error.log"

$SupervisorErrorPath = Join-Path `
    $LogDirectory `
    "stage4b2_supervisor_error.log"

function Stop-WebotsProcesses {
    Get-Process `
        -Name @(
            "webots",
            "webotsw",
            "webots-bin"
        ) `
        -ErrorAction SilentlyContinue |
        Stop-Process `
            -Force `
            -ErrorAction SilentlyContinue
}

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

function Show-FileContent {
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

Write-Host "[1/8] Validating prerequisites..." -ForegroundColor Cyan

$CurrentBranch = (& git branch --show-current).Trim()

if ($CurrentBranch -ne $ExpectedBranch) {
    throw "Expected branch $ExpectedBranch. Current branch: $CurrentBranch"
}

$WebotsExecutable = (
    $LauncherCandidates |
    Where-Object {
        Test-Path -LiteralPath $_
    } |
    Select-Object -First 1
)

if ([string]::IsNullOrWhiteSpace($WebotsExecutable)) {
    throw "No Webots Windows launcher was found."
}

$RequiredFiles = @(
    $Python,
    $LiveWorldPath,
    $ValidatorPath,
    (
        Join-Path `
            $Project `
            "webots\controllers\live_bridge_robot\live_bridge_robot.py"
    ),
    (
        Join-Path `
            $Project `
            "webots\controllers\live_bridge_supervisor\live_bridge_supervisor.py"
    )
)

foreach ($RequiredFile in $RequiredFiles) {
    if (-not (Test-Path -LiteralPath $RequiredFile)) {
        throw "Required file was not found: $RequiredFile"
    }
}

Write-Host "Launcher: $WebotsExecutable" -ForegroundColor Green

Write-Host "[2/8] Validating runtime Python files..." -ForegroundColor Cyan

& $Python -m py_compile `
    "scripts\validate_stage4b2_runtime.py" `
    "webots\controllers\live_bridge_robot\live_bridge_robot.py" `
    "webots\controllers\live_bridge_supervisor\live_bridge_supervisor.py"

if ($LASTEXITCODE -ne 0) {
    throw "Runtime Python syntax validation failed."
}

& $Python -m ruff check `
    "scripts\validate_stage4b2_runtime.py" `
    "webots\controllers\live_bridge_robot\live_bridge_robot.py" `
    "webots\controllers\live_bridge_supervisor\live_bridge_supervisor.py"

if ($LASTEXITCODE -ne 0) {
    throw "Runtime Ruff validation failed."
}

Write-Host "[3/8] Preparing the Webots Python environment..." -ForegroundColor Cyan

$env:WEBOTS_HOME = $WebotsHome
$env:PYTHONPATH = "$Project\src"
$env:WEBOTS_PYTHON_COMMAND = $Python
$env:PYTHONUNBUFFERED = "1"

$VirtualEnvironmentDirectory = Join-Path `
    $Project `
    ".venv\Scripts"

$env:Path = "$VirtualEnvironmentDirectory;$WebotsBinDirectory;$env:Path"

Write-Host "WEBOTS_HOME=$env:WEBOTS_HOME" -ForegroundColor DarkGray
Write-Host "PYTHONPATH=$env:PYTHONPATH" -ForegroundColor DarkGray
Write-Host "WEBOTS_PYTHON_COMMAND=$env:WEBOTS_PYTHON_COMMAND" -ForegroundColor DarkGray

Write-Host "[4/8] Cleaning previous runtime evidence..." -ForegroundColor Cyan

New-Item `
    -ItemType Directory `
    -Path $LogDirectory `
    -Force |
    Out-Null

Stop-WebotsProcesses

$RuntimeFiles = @(
    $RobotLogPath,
    $SupervisorLogPath,
    $SummaryPath,
    $RobotStartupPath,
    $SupervisorStartupPath,
    $RobotErrorPath,
    $SupervisorErrorPath
)

foreach ($RuntimeFile in $RuntimeFiles) {
    Remove-Item `
        -LiteralPath $RuntimeFile `
        -Force `
        -ErrorAction SilentlyContinue
}

Start-Sleep -Seconds 2

Write-Host "[5/8] Starting the Stage 4B2 world..." -ForegroundColor Cyan

$WebotsArguments = @(
    "--batch",
    "--minimize",
    "--mode=fast",
    "--no-rendering",
    $LiveWorldPath
)

$LauncherProcess = Start-Process `
    -FilePath $WebotsExecutable `
    -ArgumentList $WebotsArguments `
    -WorkingDirectory $Project `
    -PassThru

Write-Host "Launcher process ID: $($LauncherProcess.Id)" -ForegroundColor DarkGray
Write-Host "Waiting for Webots startup..." -ForegroundColor DarkGray

$StartupDeadline = (Get-Date).AddSeconds(20)

while ((Get-Date) -lt $StartupDeadline) {
    $RobotStarted = Test-Path `
        -LiteralPath $RobotStartupPath

    $SupervisorStarted = Test-Path `
        -LiteralPath $SupervisorStartupPath

    if (
        $RobotStarted `
        -or $SupervisorStarted `
        -or (Test-Path -LiteralPath $RobotLogPath) `
        -or (Test-Path -LiteralPath $SupervisorLogPath)
    ) {
        break
    }

    Start-Sleep -Milliseconds 500
}

$RobotStarted = Test-Path `
    -LiteralPath $RobotStartupPath

$SupervisorStarted = Test-Path `
    -LiteralPath $SupervisorStartupPath

Write-Host "Robot controller entered main: $RobotStarted" -ForegroundColor DarkGray
Write-Host "Supervisor entered main: $SupervisorStarted" -ForegroundColor DarkGray

Write-Host "[6/8] Collecting live telemetry..." -ForegroundColor Cyan

$Deadline = (Get-Date).AddSeconds(100)
$SupervisorSamples = 0
$RobotSamples = 0
$RuntimeComplete = $false

while ((Get-Date) -lt $Deadline) {
    $RobotSamples = Get-JsonLineCount `
        -Path $RobotLogPath

    $SupervisorSamples = Get-JsonLineCount `
        -Path $SupervisorLogPath

    Write-Host `
        "`rRobot: $RobotSamples / 30 | Supervisor: $SupervisorSamples / 30" `
        -NoNewline

    if (
        $RobotSamples -ge 30 `
        -and $SupervisorSamples -ge 30
    ) {
        $RuntimeComplete = $true
        break
    }

    if (
        (Test-Path -LiteralPath $RobotErrorPath) `
        -or (Test-Path -LiteralPath $SupervisorErrorPath)
    ) {
        break
    }

    Start-Sleep -Milliseconds 500
}

Write-Host ""

if (-not $RuntimeComplete) {
    $ActiveProcesses = @(
        Get-Process `
            -Name @(
                "webots",
                "webotsw",
                "webots-bin",
                "python"
            ) `
            -ErrorAction SilentlyContinue |
        Select-Object `
            ProcessName,
            Id,
            StartTime
    )

    Write-Host ""
    Write-Host "Active related processes" -ForegroundColor Yellow

    if ($ActiveProcesses.Count -gt 0) {
        $ActiveProcesses | Format-Table
    }
    else {
        Write-Host "No related processes are running."
    }

    Show-FileContent `
        -Title "Robot startup marker" `
        -Path $RobotStartupPath

    Show-FileContent `
        -Title "Supervisor startup marker" `
        -Path $SupervisorStartupPath

    Show-FileContent `
        -Title "Robot controller traceback" `
        -Path $RobotErrorPath

    Show-FileContent `
        -Title "Supervisor controller traceback" `
        -Path $SupervisorErrorPath

    Write-Host ""
    Write-Host "Runtime evidence status" -ForegroundColor Yellow

    [PSCustomObject]@{
        RobotStarted = $RobotStarted
        SupervisorStarted = $SupervisorStarted
        RobotSamples = $RobotSamples
        SupervisorSamples = $SupervisorSamples
        RobotLogExists = (
            Test-Path -LiteralPath $RobotLogPath
        )
        SupervisorLogExists = (
            Test-Path -LiteralPath $SupervisorLogPath
        )
    } | Format-List

    Stop-WebotsProcesses

    throw "Stage 4B2 runtime did not produce the required telemetry."
}

Start-Sleep -Seconds 2
Stop-WebotsProcesses

Write-Host "[7/8] Validating runtime evidence..." -ForegroundColor Cyan

& $Python $ValidatorPath

if ($LASTEXITCODE -ne 0) {
    throw "Stage 4B2 runtime evidence validation failed."
}

if (-not (Test-Path -LiteralPath $SummaryPath)) {
    throw "Runtime summary was not created."
}

$Summary = Get-Content `
    -LiteralPath $SummaryPath `
    -Raw |
    ConvertFrom-Json

if (-not [bool]$Summary.runtime_verified) {
    throw "runtime_verified is not true."
}

if ([int]$Summary.transferred_sample_count -lt 30) {
    throw "Fewer than 30 telemetry records were transferred."
}

if ([int]$Summary.maximum_ack_count -le 0) {
    throw "No Supervisor acknowledgement was received."
}

Write-Host "[8/8] Running final repository validation..." -ForegroundColor Cyan

& $Python -m pytest

if ($LASTEXITCODE -ne 0) {
    throw "The repository test suite failed."
}

git diff --check

if ($LASTEXITCODE -ne 0) {
    throw "Repository formatting validation failed."
}

Write-Host ""
Write-Host "Runtime summary" -ForegroundColor Cyan

Get-Content `
    -LiteralPath $SummaryPath `
    -Raw

Write-Host ""
Write-Host "Git status" -ForegroundColor Cyan

git status --short

Write-Host ""
Write-Host "Stage 4B2B runtime evidence validated successfully. No commit was created." -ForegroundColor Green
