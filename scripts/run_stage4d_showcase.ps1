$ErrorActionPreference = "Stop"

$Project = "F:\AI\My_Project\RiskAware-SafeRL-Construction-Inspector"
$Python = Join-Path $Project ".venv\Scripts\python.exe"
$ExpectedBranch = "stage-4d-professional-showcase"

$WebotsHome = "C:\Program Files\Webots"

$WebotsBinDirectory = Join-Path `
    $WebotsHome `
    "msys64\mingw64\bin"

$WorldPath = Join-Path `
    $Project `
    "webots\worlds\construction_site_stage4d_showcase.wbt"

$LogDirectory = Join-Path `
    $Project `
    "webots\logs"

$StandardOutputPath = Join-Path `
    $LogDirectory `
    "stage4d_webots_stdout.log"

$StandardErrorPath = Join-Path `
    $LogDirectory `
    "stage4d_webots_stderr.log"

$LauncherCandidates = @(
    (Join-Path $WebotsBinDirectory "webotsw.exe"),
    (Join-Path $WebotsBinDirectory "webots.exe")
)

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

function Get-RunningWebotsProcesses {
    $DetectedProcesses = @()

    foreach ($ProcessName in @(
        "webots",
        "webotsw",
        "webots-bin"
    )) {
        $DetectedProcesses += @(
            Get-Process `
                -Name $ProcessName `
                -ErrorAction SilentlyContinue
        )
    }

    $CimProcesses = @(
        Get-CimInstance `
            -ClassName Win32_Process `
            -ErrorAction SilentlyContinue |
        Where-Object {
            (
                $_.Name -match "^webots.*\.exe$"
            ) -or (
                $_.ExecutablePath -and
                $_.ExecutablePath.StartsWith(
                    $WebotsHome,
                    [System.StringComparison]::OrdinalIgnoreCase
                )
            )
        }
    )

    foreach ($CimProcess in $CimProcesses) {
        $DetectedProcesses += @(
            Get-Process `
                -Id $CimProcess.ProcessId `
                -ErrorAction SilentlyContinue
        )
    }

    return @(
        $DetectedProcesses |
        Where-Object {
            $null -ne $_
        } |
        Sort-Object `
            -Property Id `
            -Unique
    )
}

function Show-LogFile {
    param(
        [string]$Title,
        [string]$Path
    )

    Write-Host ""
    Write-Host $Title -ForegroundColor Yellow

    if (-not (Test-Path -LiteralPath $Path)) {
        Write-Host "The file was not created."
        return
    }

    $Content = Get-Content `
        -LiteralPath $Path `
        -Raw `
        -ErrorAction SilentlyContinue

    if (
        [string]::IsNullOrWhiteSpace(
            $Content
        )
    ) {
        Write-Host "The file is empty."
        return
    }

    Write-Host $Content
}

Set-Location $Project

Write-Host "[1/6] Validating showcase prerequisites..." -ForegroundColor Cyan

$CurrentBranch = (
    & git branch --show-current
).Trim()

if ($CurrentBranch -ne $ExpectedBranch) {
    throw "Expected branch $ExpectedBranch. Current branch: $CurrentBranch"
}

if (-not (Test-Path -LiteralPath $Python)) {
    throw "Project Python was not found: $Python"
}

if (-not (Test-Path -LiteralPath $WorldPath)) {
    throw "Showcase world was not found: $WorldPath"
}

$WebotsExecutable = $LauncherCandidates |
    Where-Object {
        Test-Path -LiteralPath $_
    } |
    Select-Object -First 1

if (
    [string]::IsNullOrWhiteSpace(
        $WebotsExecutable
    )
) {
    throw "A Webots launcher was not found."
}

Write-Host "[2/6] Closing previous Webots processes..." -ForegroundColor Cyan

Stop-AllWebotsProcesses
Start-Sleep -Seconds 2

Write-Host "[3/6] Preparing the Webots environment..." -ForegroundColor Cyan

New-Item `
    -ItemType Directory `
    -Path $LogDirectory `
    -Force |
    Out-Null

Remove-Item `
    -LiteralPath $StandardOutputPath `
    -Force `
    -ErrorAction SilentlyContinue

Remove-Item `
    -LiteralPath $StandardErrorPath `
    -Force `
    -ErrorAction SilentlyContinue

$VirtualEnvironmentDirectory = Join-Path `
    $Project `
    ".venv\Scripts"

$PathParts = @(
    $VirtualEnvironmentDirectory,
    $WebotsBinDirectory,
    $env:Path
)

$env:Path = $PathParts -join (
    [System.IO.Path]::PathSeparator
)

$env:WEBOTS_HOME = $WebotsHome
$env:WEBOTS_PYTHON_COMMAND = $Python
$env:PYTHONUNBUFFERED = "1"
$env:PYTHONIOENCODING = "utf-8"

Write-Host "WEBOTS_HOME=$env:WEBOTS_HOME" -ForegroundColor DarkGray
Write-Host "WEBOTS_PYTHON_COMMAND=$env:WEBOTS_PYTHON_COMMAND" -ForegroundColor DarkGray
Write-Host "World=$WorldPath" -ForegroundColor DarkGray

Write-Host "[4/6] Starting Webots..." -ForegroundColor Cyan

$LaunchArguments = @(
    "--mode=realtime",
    "--stdout",
    "--stderr",
    "`"$WorldPath`""
)

$StarterProcess = Start-Process `
    -FilePath $WebotsExecutable `
    -ArgumentList $LaunchArguments `
    -WorkingDirectory $Project `
    -RedirectStandardOutput $StandardOutputPath `
    -RedirectStandardError $StandardErrorPath `
    -PassThru

Write-Host "Starter process ID: $($StarterProcess.Id)" -ForegroundColor DarkGray

Start-Sleep -Seconds 10

$RunningProcesses = @(
    Get-RunningWebotsProcesses
)

Write-Host "[5/6] Validating the Webots process..." -ForegroundColor Cyan

if ($RunningProcesses.Count -eq 0) {
    Show-LogFile `
        -Title "Webots standard output" `
        -Path $StandardOutputPath

    Show-LogFile `
        -Title "Webots standard error" `
        -Path $StandardErrorPath

    throw "No running Webots process was detected."
}

Write-Host ""
Write-Host "Running Webots processes" -ForegroundColor Green

$RunningProcesses |
    Select-Object `
        ProcessName,
        Id |
    Format-Table `
        -AutoSize

Write-Host "[6/6] Stage 4D showcase is running." -ForegroundColor Green
Write-Host "World: construction_site_stage4d_showcase.wbt" -ForegroundColor Green
Write-Host "Keep Webots open for visual inspection." -ForegroundColor Green
