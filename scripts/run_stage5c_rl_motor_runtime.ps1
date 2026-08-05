param(
    [string]$RepoRoot = "F:\AI\My_Project\RiskAware-SafeRL-Construction-Inspector",
    [int]$TimeoutSeconds = 420
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$Python = Join-Path $RepoRoot ".venv\Scripts\python.exe"
$Builder = Join-Path $RepoRoot "scripts\build_stage5c_rl_motor_world.py"
$World = Join-Path $RepoRoot "webots\worlds\construction_site_stage5c_rl_motor_runtime.wbt"
$WorldProject = Join-Path $RepoRoot "webots\worlds\.construction_site_stage5c_rl_motor_runtime.wbproj"
$Output = Join-Path $RepoRoot "webots\logs\stage5c_rl_motor_runtime"
$CompleteMarker = Join-Path $Output "stage5c_complete.marker"
$FailureReport = Join-Path $Output "stage5c_failure.json"
$Summary = Join-Path $Output "stage5c_runtime_summary.json"
$WebotsHome = "C:\Program Files\Webots"
$Webots = Join-Path $WebotsHome "msys64\mingw64\bin\webotsw.exe"

foreach ($Required in @($Python, $Builder, $Webots)) {
    if (-not (Test-Path -LiteralPath $Required -PathType Leaf)) {
        throw "Required Stage 5C runtime file is missing: $Required"
    }
}

Set-Location -LiteralPath $RepoRoot
& $Python $Builder
if ($LASTEXITCODE -ne 0) {
    throw "Stage 5C world generation failed."
}

Get-Process -Name webots, webotsw, webots-bin -ErrorAction SilentlyContinue |
    Stop-Process -Force

if (Test-Path -LiteralPath $Output) {
    Remove-Item -LiteralPath $Output -Recurse -Force
}
New-Item -ItemType Directory -Path $Output | Out-Null
Remove-Item -LiteralPath $WorldProject -Force -ErrorAction SilentlyContinue

$env:WEBOTS_HOME = $WebotsHome
$env:WEBOTS_PYTHON_COMMAND = $Python
$env:RISK_AWARE_PROJECT_ROOT = $RepoRoot
$env:YOLO_CONFIG_DIR = Join-Path $RepoRoot ".runtime\ultralytics"
$env:QT_AUTO_SCREEN_SCALE_FACTOR = "0"
$env:QT_SCALE_FACTOR = "1"
$env:QT_SCREEN_SCALE_FACTORS = "1"
# Force the unattended Webots child onto Qt's offscreen backend.  Without
# this, inherited desktop geometry can be expanded into a multi-million-pixel
# framebuffer before --no-rendering is applied.
$env:QT_QPA_PLATFORM = "offscreen"
$env:QT_SCALE_FACTOR_ROUNDING_POLICY = "Round"
$env:PYTHONUNBUFFERED = "1"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONPATH = Join-Path $WebotsHome "lib\controller\python"
$VenvScripts = Join-Path $RepoRoot ".venv\Scripts"
$WebotsBin = Join-Path $WebotsHome "msys64\mingw64\bin"
$InheritedPath = $env:Path
# Some Windows launch contexts contain both Path and PATH. Start-Process
# materializes a case-insensitive dictionary and fails before Webots starts.
[Environment]::SetEnvironmentVariable(
    "PATH",
    $null,
    [EnvironmentVariableTarget]::Process
)
[Environment]::SetEnvironmentVariable(
    "Path",
    $InheritedPath,
    [EnvironmentVariableTarget]::Process
)
$env:Path = [string]::Join(
    [IO.Path]::PathSeparator,
    @($VenvScripts, $WebotsBin, $env:Path)
)
New-Item -ItemType Directory -Path $env:YOLO_CONFIG_DIR -Force | Out-Null

$Stdout = Join-Path $Output "webots_stdout.log"
$Stderr = Join-Path $Output "webots_stderr.log"
# Rendering is performed by the robot camera controller. Disabling the
# desktop 3-D view prevents stale Windows GUI geometry from allocating an
# invalid framebuffer in unattended runs.
$Process = Start-Process `
    -FilePath $Webots `
    -ArgumentList @("--batch", "--no-rendering", "--mode=fast", "--stdout", "--stderr", $World) `
    -WorkingDirectory $RepoRoot `
    -RedirectStandardOutput $Stdout `
    -RedirectStandardError $Stderr `
    -WindowStyle Hidden `
    -PassThru

# The MSYS launcher can hand execution to webots-bin.exe and exit before the
# simulation. Follow that concrete process so early-exit and timeout checks
# describe the simulator rather than the short-lived launcher.
Start-Sleep -Milliseconds 750
$Process.Refresh()
if ($Process.HasExited) {
    $Simulator = Get-Process -Name webots-bin -ErrorAction SilentlyContinue |
        Sort-Object StartTime -Descending |
        Select-Object -First 1
    if ($null -ne $Simulator) {
        $Process = $Simulator
    }
}

$Deadline = (Get-Date).AddSeconds($TimeoutSeconds)
try {
    while (-not (Test-Path -LiteralPath $CompleteMarker)) {
        $Process.Refresh()
        if ($Process.HasExited) {
            throw "Webots exited before Stage 5C completion with code $($Process.ExitCode)."
        }
        if (Test-Path -LiteralPath $FailureReport) {
            Get-Content -LiteralPath $FailureReport
            throw "The Stage 5C controller reported a failure."
        }
        if ((Get-Date) -gt $Deadline) {
            throw "Stage 5C exceeded the $TimeoutSeconds-second timeout."
        }
        Start-Sleep -Milliseconds 250
    }
    $ViewDeadline = (Get-Date).AddSeconds(8)
    while (-not $Process.HasExited -and (Get-Date) -lt $ViewDeadline) {
        $Process.Refresh()
        Start-Sleep -Milliseconds 250
    }
}
finally {
    Get-Process -Name webots, webotsw, webots-bin -ErrorAction SilentlyContinue |
        Stop-Process -Force
}

if (-not (Test-Path -LiteralPath $Summary -PathType Leaf)) {
    throw "Stage 5C runtime summary is missing."
}
$FinalView = Join-Path $Output "first_person_final.png"
if (-not (Test-Path -LiteralPath $FinalView -PathType Leaf)) {
    throw "Stage 5C final first-person evidence is missing."
}
$Payload = Get-Content -LiteralPath $Summary -Raw | ConvertFrom-Json
$RequiredFlags = @(
    "runtime_verified",
    "policy_loaded",
    "policy_controls_motors",
    "safety_shield_active",
    "cv_model_connected",
    "cuda_inference_verified",
    "perception_live_during_mission",
    "perception_affects_runtime_state",
    "mission_completed"
)
foreach ($Flag in $RequiredFlags) {
    if ($Payload.$Flag -ne $true) {
        throw "Stage 5C acceptance flag is not true: $Flag"
    }
}
if ($Payload.manual_control_used -ne $false) {
    throw "Stage 5C unexpectedly used manual control."
}
if ($Payload.fallback_controller_used -ne $false) {
    throw "Stage 5C unexpectedly used the fallback controller."
}
if ([int]$Payload.failure_count -ne 0) {
    throw "Stage 5C perception recorded failures."
}
if ([int]$Payload.motor_command_change_count -lt 1) {
    throw "Stage 5C motor commands did not change."
}
if ([int]$Payload.safety_shield_interventions -lt 1) {
    throw "Stage 5C did not record a real SafetyShield intervention."
}

Write-Host "STAGE5C_RL_MOTOR_RUNTIME=PASSED"
Write-Host "Summary: $Summary"
