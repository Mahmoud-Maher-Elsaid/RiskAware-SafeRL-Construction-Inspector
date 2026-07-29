$ErrorActionPreference = "Stop"

$Project = "F:\AI\My_Project\RiskAware-SafeRL-Construction-Inspector"
$Python = Join-Path $Project ".venv\Scripts\python.exe"
$ExpectedBranch = "stage-5a-live-camera-acquisition"

$WorldPath = Join-Path $Project "webots\worlds\construction_site_stage5a_live_camera.wbt"
$ValidatorPath = Join-Path $Project "scripts\validate_stage5a_live_camera.py"
$OutputDirectory = Join-Path $Project "webots\logs\stage5a_live_camera"

$CompletionMarkerPath = Join-Path $OutputDirectory "stage5a_complete.marker"
$FailureMarkerPath = Join-Path $OutputDirectory "stage5a_failure.marker"
$FailureReportPath = Join-Path $OutputDirectory "stage5a_failure.json"
$TimeoutMarkerPath = Join-Path $OutputDirectory "stage5a_timeout.marker"

$StandardOutputPath = Join-Path $OutputDirectory "webots_stdout.log"
$StandardErrorPath = Join-Path $OutputDirectory "webots_stderr.log"

$WebotsHome = "C:\Program Files\Webots"
$WebotsBinDirectory = Join-Path $WebotsHome "msys64\mingw64\bin"
$WebotsExecutable = Join-Path $WebotsBinDirectory "webotsw.exe"

function Stop-AllWebotsProcesses {
    $ProcessNames = @(
        "webots",
        "webotsw",
        "webots-bin"
    )

    foreach ($ProcessName in $ProcessNames) {
        Get-Process `
            -Name $ProcessName `
            -ErrorAction SilentlyContinue |
            Stop-Process `
                -Force `
                -ErrorAction SilentlyContinue
    }
}

function Show-LogTail {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Title,

        [Parameter(Mandatory = $true)]
        [string]$Path
    )

    Write-Host ""
    Write-Host $Title -ForegroundColor Yellow

    if (-not (Test-Path -LiteralPath $Path)) {
        Write-Host "The log file was not created."
        return
    }

    Get-Content `
        -LiteralPath $Path `
        -Tail 100 `
        -ErrorAction SilentlyContinue
}

Set-Location $Project

$CurrentBranch = (& git branch --show-current).Trim()

if ($CurrentBranch -ne $ExpectedBranch) {
    throw "Expected branch $ExpectedBranch. Current branch: $CurrentBranch"
}

$RequiredFiles = @(
    $Python,
    $WorldPath,
    $ValidatorPath,
    $WebotsExecutable
)

foreach ($RequiredFile in $RequiredFiles) {
    if (-not (Test-Path -LiteralPath $RequiredFile)) {
        throw "Required runtime file is missing: $RequiredFile"
    }
}

Write-Host "[1/6] Closing previous Webots processes..." -ForegroundColor Cyan

Stop-AllWebotsProcesses
Start-Sleep -Seconds 2

Write-Host "[2/6] Resetting runtime evidence..." -ForegroundColor Cyan

Remove-Item `
    -LiteralPath $OutputDirectory `
    -Recurse `
    -Force `
    -ErrorAction SilentlyContinue

New-Item `
    -ItemType Directory `
    -Path $OutputDirectory `
    -Force |
    Out-Null

Write-Host "[3/6] Preparing the Webots environment..." -ForegroundColor Cyan

$VirtualEnvironmentDirectory = Join-Path $Project ".venv\Scripts"

$RuntimePathParts = @(
    $VirtualEnvironmentDirectory
    $WebotsBinDirectory
    $env:Path
)

$env:Path = [string]::Join(
    [System.IO.Path]::PathSeparator,
    $RuntimePathParts
)

$env:WEBOTS_HOME = $WebotsHome
$env:WEBOTS_PYTHON_COMMAND = $Python
$env:RISK_AWARE_PROJECT_ROOT = $Project
$env:PYTHONUNBUFFERED = "1"
$env:PYTHONIOENCODING = "utf-8"

Write-Host "WEBOTS_HOME=$env:WEBOTS_HOME" -ForegroundColor DarkGray
Write-Host "WEBOTS_PYTHON_COMMAND=$env:WEBOTS_PYTHON_COMMAND" -ForegroundColor DarkGray
Write-Host "World=$WorldPath" -ForegroundColor DarkGray

Write-Host "[4/6] Starting the Stage 5A world..." -ForegroundColor Cyan

$Arguments = @(
    "--mode=realtime"
    "--stdout"
    "--stderr"
    $WorldPath
)

$WebotsProcess = Start-Process `
    -FilePath $WebotsExecutable `
    -ArgumentList $Arguments `
    -WorkingDirectory $Project `
    -RedirectStandardOutput $StandardOutputPath `
    -RedirectStandardError $StandardErrorPath `
    -PassThru

Write-Host "Webots starter PID: $($WebotsProcess.Id)" -ForegroundColor DarkGray

Write-Host "[5/6] Waiting for camera evidence..." -ForegroundColor Cyan

$MaximumWaitSeconds = 180
$ElapsedSeconds = 0
$RuntimeCompleted = $false

while ($ElapsedSeconds -lt $MaximumWaitSeconds) {
    if (Test-Path -LiteralPath $FailureMarkerPath) {
        Show-LogTail `
            -Title "Webots standard output" `
            -Path $StandardOutputPath

        Show-LogTail `
            -Title "Webots standard error" `
            -Path $StandardErrorPath

        if (Test-Path -LiteralPath $FailureReportPath) {
            Write-Host ""
            Write-Host "Stage 5A failure report" -ForegroundColor Red

            Get-Content `
                -LiteralPath $FailureReportPath `
                -Raw
        }

        Stop-AllWebotsProcesses

        throw "The Stage 5A camera controller reported a runtime failure."
    }

    if (Test-Path -LiteralPath $TimeoutMarkerPath) {
        Show-LogTail `
            -Title "Webots standard output" `
            -Path $StandardOutputPath

        Show-LogTail `
            -Title "Webots standard error" `
            -Path $StandardErrorPath

        Stop-AllWebotsProcesses

        throw "The Stage 5A Webots runtime timed out."
    }

    if (Test-Path -LiteralPath $CompletionMarkerPath) {
        $RuntimeCompleted = $true
        break
    }

    Start-Sleep -Seconds 1
    $ElapsedSeconds += 1

    if (($ElapsedSeconds % 10) -eq 0) {
        Write-Host `
            "Waiting for live camera evidence: $ElapsedSeconds seconds" `
            -ForegroundColor DarkGray
    }
}

if (-not $RuntimeCompleted) {
    Show-LogTail `
        -Title "Webots standard output" `
        -Path $StandardOutputPath

    Show-LogTail `
        -Title "Webots standard error" `
        -Path $StandardErrorPath

    Stop-AllWebotsProcesses

    throw "Stage 5A did not complete within $MaximumWaitSeconds seconds."
}

Write-Host "Stage 5A completion marker detected." -ForegroundColor Green

Start-Sleep -Seconds 5
Stop-AllWebotsProcesses

Write-Host "[6/6] Running the independent validator..." -ForegroundColor Cyan

& $Python `
    $ValidatorPath `
    $Project

if ($LASTEXITCODE -ne 0) {
    Show-LogTail `
        -Title "Webots standard output" `
        -Path $StandardOutputPath

    Show-LogTail `
        -Title "Webots standard error" `
        -Path $StandardErrorPath

    throw "Stage 5A evidence validation failed."
}

Write-Host ""
Write-Host "Stage 5A live camera runtime passed." -ForegroundColor Green
