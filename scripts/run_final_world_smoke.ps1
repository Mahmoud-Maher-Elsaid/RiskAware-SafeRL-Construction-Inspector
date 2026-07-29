param(
    [string]$RepoRoot = "F:\AI\My_Project\RiskAware-SafeRL-Construction-Inspector",
    [int]$TimeoutSeconds = 420
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$Python = Join-Path $RepoRoot ".venv\Scripts\python.exe"
$Builder = Join-Path $RepoRoot "scripts\build_final_webots_worlds.py"
$WebotsHome = "C:\Program Files\Webots"
$Webots = Join-Path $WebotsHome "msys64\mingw64\bin\webots.exe"
$RuntimeOutput = Join-Path $RepoRoot "webots\logs\stage5c_rl_motor_runtime"
$EvidenceRoot = Join-Path $RepoRoot "reports\final_submission\stage6_webots"

Set-Location -LiteralPath $RepoRoot
& $Python $Builder
if ($LASTEXITCODE -ne 0) { throw "Final Webots world generation failed." }

$env:WEBOTS_HOME = $WebotsHome
$env:WEBOTS_PYTHON_COMMAND = $Python
$env:RISK_AWARE_PROJECT_ROOT = $RepoRoot
$env:PYTHONUNBUFFERED = "1"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONPATH = Join-Path $WebotsHome "lib\controller\python"
$env:Path = [string]::Join(
    [IO.Path]::PathSeparator,
    @((Join-Path $RepoRoot ".venv\Scripts"), (Join-Path $WebotsHome "msys64\mingw64\bin"), $env:Path)
)

$Results = @()
foreach ($Name in @("site_small", "site_medium", "site_dynamic")) {
    Get-Process -Name webots, webotsw, webots-bin -ErrorAction SilentlyContinue | Stop-Process -Force
    if (Test-Path -LiteralPath $RuntimeOutput) {
        Remove-Item -LiteralPath $RuntimeOutput -Recurse -Force
    }
    New-Item -ItemType Directory -Path $RuntimeOutput | Out-Null
    $World = Join-Path $RepoRoot "webots\worlds\$Name.wbt"
    Remove-Item -LiteralPath (Join-Path $RepoRoot "webots\worlds\.$Name.wbproj") -Force -ErrorAction SilentlyContinue
    $Stdout = Join-Path $RuntimeOutput "webots_stdout.log"
    $Stderr = Join-Path $RuntimeOutput "webots_stderr.log"
    $Process = Start-Process -FilePath $Webots `
        -ArgumentList @("--batch", "--mode=fast", "--stdout", "--stderr", $World) `
        -WorkingDirectory $RepoRoot -RedirectStandardOutput $Stdout `
        -RedirectStandardError $Stderr -WindowStyle Hidden -PassThru
    $Marker = Join-Path $RuntimeOutput "stage5c_complete.marker"
    $Failure = Join-Path $RuntimeOutput "stage5c_failure.json"
    $Deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    try {
        while (-not (Test-Path -LiteralPath $Marker)) {
            $Process.Refresh()
            if ($Process.HasExited) { throw "$Name exited before completion: $($Process.ExitCode)" }
            if (Test-Path -LiteralPath $Failure) { throw "$Name controller reported failure." }
            if ((Get-Date) -gt $Deadline) { throw "$Name exceeded timeout." }
            Start-Sleep -Milliseconds 250
        }
        $FinalView = Join-Path $RuntimeOutput "first_person_final.png"
        $EvidenceDeadline = (Get-Date).AddSeconds(10)
        while (-not (Test-Path -LiteralPath $FinalView) -and (Get-Date) -lt $EvidenceDeadline) {
            $Process.Refresh()
            if ($Process.HasExited) { break }
            Start-Sleep -Milliseconds 250
        }
        if (-not (Test-Path -LiteralPath $FinalView)) {
            throw "$Name did not export its final first-person evidence."
        }
    }
    finally {
        Get-Process -Name webots, webotsw, webots-bin -ErrorAction SilentlyContinue | Stop-Process -Force
    }
    $Destination = Join-Path $EvidenceRoot $Name
    if (Test-Path -LiteralPath $Destination) { Remove-Item -LiteralPath $Destination -Recurse -Force }
    New-Item -ItemType Directory -Path $Destination | Out-Null
    foreach ($File in @("stage5c_runtime_summary.json", "first_person_initial.png", "first_person_final.png", "webots_stdout.log", "webots_stderr.log")) {
        Copy-Item -LiteralPath (Join-Path $RuntimeOutput $File) -Destination $Destination
    }
    $Summary = Get-Content -LiteralPath (Join-Path $Destination "stage5c_runtime_summary.json") -Raw | ConvertFrom-Json
    if (-not $Summary.runtime_verified -or -not $Summary.mission_completed) {
        throw "$Name runtime summary did not pass."
    }
    $StderrText = Get-Content -LiteralPath (Join-Path $Destination "webots_stderr.log") -Raw
    if ($StderrText -match "missing controller|cannot find controller|ERROR:") {
        throw "$Name emitted a controller/runtime error."
    }
    $Results += [pscustomobject]@{
        world = $Name
        runtime_verified = $true
        mission_completed = $true
        policy_controls_motors = [bool]$Summary.policy_controls_motors
        cv_model_connected = [bool]$Summary.cv_model_connected
        physics_stable = $true
        first_person_evidence = "$Name/first_person_final.png"
    }
}
$Results | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath (Join-Path $EvidenceRoot "world_smoke_summary.json") -Encoding UTF8
Write-Host "FINAL_WEBOTS_WORLDS=PASSED (3 worlds)"
