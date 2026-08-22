param(
    [ValidateSet('site_small', 'site_medium', 'site_dynamic')]
    [string]$World = 'site_dynamic',
    [ValidateRange(15, 1200)]
    [int]$DurationSeconds = 90,
    [ValidateSet('until_closed', 'bounded')]
    [string]$RunMode = 'until_closed',
    [ValidateSet('overview')]
    [string]$CameraMode = 'overview',
    [int]$ScenarioSeed = 42,
    [ValidateSet('deterministic', 'stochastic')]
    [string]$PolicyMode = 'stochastic',
    [ValidateRange(0.1, 2.0)]
    [double]$PolicyTemperature = 0.70
)
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$repo = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$python = Join-Path $repo '.venv\Scripts\python.exe'
$webots = 'C:\Program Files\Webots\msys64\mingw64\bin\webots.exe'
$rootOut = Join-Path $repo 'reports\strong_policy_upgrade\webots_v4_visible_demo'
$source = Join-Path $repo ("webots\worlds\{0}.wbt" -f $World)
$stableWorld = Join-Path $repo ("webots\worlds\{0}_v4_visible_demo.wbt" -f $World)
$checkpoint = Join-Path $repo 'artifacts\strong_policy_upgrade\hierarchical_imitation_h4_target_persistence\seed_105\best_checkpoint.pt'
$checkpointSha = '1b193f429d8411c30232b60c63260b2df0d112559a6a68139915cc402be14888'

foreach ($path in @($python, $webots, $source, $checkpoint)) {
    if (-not (Test-Path -LiteralPath $path)) { throw "Required path not found: $path" }
}
if ((Get-FileHash -LiteralPath $checkpoint -Algorithm SHA256).Hash.ToLowerInvariant() -ne $checkpointSha) { throw 'Experimental checkpoint hash mismatch.' }

New-Item -ItemType Directory -Force -Path $rootOut | Out-Null
$attemptName = 'attempt_{0}' -f (Get-Date -Format 'yyyyMMdd_HHmmss_fff')
$attempt = Join-Path $rootOut $attemptName
New-Item -ItemType Directory -Force -Path $attempt | Out-Null
$worldText = (Get-Content -LiteralPath $source -Raw).Replace('controller "rl_autonomous_inspection_robot"', 'controller "hierarchical_experimental_robot"').Replace('controller "rl_autonomous_inspection_supervisor"', 'controller "hierarchical_experimental_supervisor"')
$worldText = [regex]::Replace($worldText, '(?s)DEF HUMAN_VIEWPOINT Viewpoint \{.*?\n\}', {
    param($m)
    switch ($CameraMode) {
        'overview' { return (@('DEF HUMAN_VIEWPOINT Viewpoint {','  position 0 40 18','  fieldOfView 1.10','  near 0.05','  far 100','  follow "main reinforced concrete construction slab"','  followType "Pan and Tilt Shot"','}') -join [Environment]::NewLine) }
        default { return $m.Value }
    }
})
# The visible experimental world gets real causal obstacle sensors without
# changing the production world.  The robot controller consumes these values
# through the normal Webots device API; supervisor state is never injected.
$sensorNodes = @'
    DEF FPV_MOUNT Transform {
      translation 0.28 1.15 0
      rotation 0 1 0 -1.5708
    }
    DistanceSensor { name "front obstacle sensor" rotation 0 1 0 3.14159 lookupTable [ 0 0 0 1 1 0 ] }
    DistanceSensor { name "front left obstacle sensor" rotation 0 1 0 3.69159 lookupTable [ 0 0 0 1 1 0 ] }
    DistanceSensor { name "front right obstacle sensor" rotation 0 1 0 2.59159 lookupTable [ 0 0 0 1 1 0 ] }
    DistanceSensor { name "left obstacle sensor" rotation 0 1 0 -1.5708 lookupTable [ 0 0 0 1 1 0 ] }
    DistanceSensor { name "right obstacle sensor" rotation 0 1 0 1.5708 lookupTable [ 0 0 0 1 1 0 ] }
    DistanceSensor { name "rear left obstacle sensor" rotation 0 1 0 -0.7854 lookupTable [ 0 0 0 1 1 0 ] }
    DistanceSensor { name "rear right obstacle sensor" rotation 0 1 0 0.7854 lookupTable [ 0 0 0 1 1 0 ] }
'@
$robotStart = $worldText.IndexOf('DEF SHOWCASE_ROBOT Robot {')
if ($robotStart -lt 0) { throw 'SHOWCASE_ROBOT was not found in the visible world source.' }
$childrenIndex = $worldText.IndexOf('  children [', $robotStart)
if ($childrenIndex -lt 0) { throw 'SHOWCASE_ROBOT children block was not found.' }
$insertAt = $childrenIndex + ('  children [' | Measure-Object -Character).Characters
$worldText = $worldText.Insert($insertAt, [Environment]::NewLine + $sensorNodes.TrimEnd())
[IO.File]::WriteAllText($stableWorld, $worldText, [Text.UTF8Encoding]::new($false))
$stableText = Get-Content -LiteralPath $stableWorld -Raw
if ($stableText -notmatch 'controller "hierarchical_experimental_robot"' -or $stableText -notmatch 'controller "hierarchical_experimental_supervisor"') { throw 'Stable world controller declarations are invalid.' }

$oldPath = [string]$env:Path
Remove-Item Env:Path -ErrorAction SilentlyContinue
Remove-Item Env:PATH -ErrorAction SilentlyContinue
$paths = @((Join-Path $repo '.venv\Scripts'), 'C:\Program Files\Webots\msys64\mingw64\bin') + ($oldPath -split ';')
$env:Path = (($paths | Where-Object { $_ -and $_.Trim() } | Select-Object -Unique) -join ';')
$env:RISK_AWARE_PROJECT_ROOT = $repo
$env:RISK_AWARE_EXPERIMENTAL_OUTPUT = $attempt
$env:RISK_AWARE_EXPERIMENTAL_DECISIONS = if ($RunMode -eq 'bounded') { [string]([Math]::Max(100, $DurationSeconds * 10)) } else { '' }
$env:RISK_AWARE_EXPERIMENTAL_DEMO_DURATION = [string]($DurationSeconds + 120)
$env:RISK_AWARE_EXPERIMENTAL_RUN_MODE = $RunMode
$env:RISK_AWARE_SCENARIO_SEED = [string]$ScenarioSeed
$env:RISK_AWARE_POLICY_MODE = $PolicyMode
$env:RISK_AWARE_POLICY_TEMPERATURE = [string]$PolicyTemperature
$env:RISK_AWARE_CAMERA_MODE = $CameraMode
$env:WEBOTS_HOME = 'C:\Program Files\Webots'
$env:WEBOTS_PYTHON_COMMAND = $python
$env:PYTHONPATH = Join-Path $env:WEBOTS_HOME 'lib\controller\python'
$env:YOLO_CONFIG_DIR = Join-Path $repo '.runtime\ultralytics'
New-Item -ItemType Directory -Force -Path $env:YOLO_CONFIG_DIR | Out-Null
# Normal display settings; never disable rendering in this launcher.
$env:QT_AUTO_SCREEN_SCALE_FACTOR = '0'
$env:QT_SCALE_FACTOR = '1'
$env:QT_SCREEN_SCALE_FACTORS = '1'
$env:QT_SCALE_FACTOR_ROUNDING_POLICY = 'Round'
$env:QT_QPA_PLATFORM = 'windows'
Remove-Item Env:QT_ENABLE_HIGHDPI_SCALING, Env:QT_DEVICE_PIXEL_RATIO, Env:QT_FONT_DPI, Env:QT_OPENGL, Env:QT_QUICK_BACKEND, Env:LIBGL_ALWAYS_SOFTWARE, Env:WEBOTS_USE_GLES -ErrorAction SilentlyContinue

$launcherState = [ordered]@{
    attempt = $attemptName; world = $World; stable_world = $stableWorld; duration_seconds = $DurationSeconds
    launcher_started = (Get-Date).ToUniversalTime().ToString('o'); webots_process_started = $false
    window_handle_verified = $false; owned_webots_bin_found = $false; owned_webots_bin_pid = $null; world_loaded_verified = $false; rendered_frame_verified = $false
    visible_window_verified = $false; controller_started = $false; supervisor_started = $false
    first_simulation_step = $false; first_policy_decision = $false; summary_generated = $false
    robot_motion_verified = $false; no_rendering_flag_used = $false; minimize_flag_used = $false
}
$launcherState | ConvertTo-Json | Set-Content (Join-Path $attempt 'launcher_state.json') -Encoding utf8

$webotsPort = 1240
while (Get-NetTCPConnection -LocalPort $webotsPort -State Listen -ErrorAction SilentlyContinue) { $webotsPort++ }
$args = @('--stdout', '--stderr', $stableWorld)
$args -contains '--no-rendering' | Set-Content (Join-Path $attempt 'no_rendering_flag_used.txt')
$args -contains '--minimize' | Set-Content (Join-Path $attempt 'minimize_flag_used.txt')
$envSnapshot = [ordered]@{ Path=$env:Path; QT_QPA_PLATFORM=$env:QT_QPA_PLATFORM; QT_SCALE_FACTOR=$env:QT_SCALE_FACTOR; QT_SCREEN_SCALE_FACTORS=$env:QT_SCREEN_SCALE_FACTORS; QT_AUTO_SCREEN_SCALE_FACTOR=$env:QT_AUTO_SCREEN_SCALE_FACTOR; RISK_AWARE_PROJECT_ROOT=$repo; RISK_AWARE_EXPERIMENTAL_OUTPUT=$attempt; RISK_AWARE_EXPERIMENTAL_DECISIONS=$env:RISK_AWARE_EXPERIMENTAL_DECISIONS; RISK_AWARE_EXPERIMENTAL_DEMO_DURATION=$env:RISK_AWARE_EXPERIMENTAL_DEMO_DURATION; RISK_AWARE_EXPERIMENTAL_RUN_MODE=$RunMode; RISK_AWARE_SCENARIO_SEED=$ScenarioSeed; RISK_AWARE_POLICY_MODE=$PolicyMode; RISK_AWARE_POLICY_TEMPERATURE=$PolicyTemperature; WEBOTS_HOME=$env:WEBOTS_HOME; WEBOTS_PYTHON_COMMAND=$python; PYTHONPATH=$env:PYTHONPATH; YOLO_CONFIG_DIR=$env:YOLO_CONFIG_DIR; CameraMode=$CameraMode; webots_port=$webotsPort }
$envSnapshot | ConvertTo-Json | Set-Content (Join-Path $attempt 'environment.json') -Encoding utf8

$stdout = Join-Path $attempt 'webots_stdout.log'
$stderr = Join-Path $attempt 'webots_stderr.log'
New-Item -ItemType File -Force -Path $stdout, $stderr | Out-Null
# Webots is a GUI process; redirecting its inherited console handles makes
# some Windows builds exit before creating webots-bin. Keep durable log files
# for the run and use the explicit --stdout/--stderr switches without unsafe
# PowerShell stream redirection.
$proc = Start-Process -FilePath $webots -ArgumentList ($args -join ' ') -WorkingDirectory $repo -WindowStyle Normal -PassThru
$launcherState.webots_process_started = $true; $launcherState.webots_pid = $proc.Id
$launcherState.world_load_requested = (Get-Date).ToUniversalTime().ToString('o')
$launcherState | ConvertTo-Json | Set-Content (Join-Path $attempt 'launcher_state.json') -Encoding utf8
Write-Output "VISIBLE_WEBOTS_ATTEMPT=$attempt"
Write-Output "VISIBLE_WEBOTS_WORLD=$stableWorld"

Add-Type -AssemblyName System.Drawing
$shell = New-Object -ComObject WScript.Shell
Add-Type -TypeDefinition @'
using System; using System.Runtime.InteropServices;
public static class RenderCapture {
 [DllImport("user32.dll")] public static extern bool GetClientRect(IntPtr h, out RECT r);
 [DllImport("user32.dll")] public static extern bool GetWindowRect(IntPtr h, out RECT r);
 [DllImport("user32.dll")] public static extern bool ClientToScreen(IntPtr h, ref POINT p);
 [DllImport("user32.dll")] public static extern bool ShowWindow(IntPtr h, int c);
 [DllImport("user32.dll")] public static extern bool SetForegroundWindow(IntPtr h);
 [DllImport("user32.dll")] public static extern bool MoveWindow(IntPtr h, int x, int y, int w, int height, bool repaint);
 [DllImport("gdi32.dll")] public static extern bool BitBlt(IntPtr d,int x,int y,int w,int h,IntPtr s,int sx,int sy,int rop);
 [StructLayout(LayoutKind.Sequential)] public struct RECT { public int l,t,r,b; }
 [StructLayout(LayoutKind.Sequential)] public struct POINT { public int x,y; }
}
'@

function Get-RenderMetrics([string]$Path) {
    $bmp = [Drawing.Bitmap]::new($Path); $pixels = 0; $black = 0; $nonBlack = 0; $sum = 0.0; $sum2 = 0.0; $colors = [Collections.Generic.HashSet[int]]::new()
    for ($y=0; $y -lt $bmp.Height; $y += [Math]::Max(1,[int]($bmp.Height/240))) { for ($x=0; $x -lt $bmp.Width; $x += [Math]::Max(1,[int]($bmp.Width/320))) { $c=$bmp.GetPixel($x,$y); $g=0.299*$c.R+0.587*$c.G+0.114*$c.B; $pixels++; $sum += $g; $sum2 += $g*$g; if($c.R -lt 8 -and $c.G -lt 8 -and $c.B -lt 8){$black++}else{$nonBlack++}; [void]$colors.Add(($c.ToArgb() -band 0x00ffffff)) } }
    $mean=$sum/[Math]::Max(1,$pixels); $std=[Math]::Sqrt([Math]::Max(0,($sum2/[Math]::Max(1,$pixels))-($mean*$mean))); $width=$bmp.Width; $height=$bmp.Height; $bmp.Dispose()
    [ordered]@{ width=$width; height=$height; mean_rgb=$mean; grayscale_standard_deviation=$std; black_pixel_ratio=$black/[Math]::Max(1,$pixels); non_black_pixel_ratio=$nonBlack/[Math]::Max(1,$pixels); unique_color_count=$colors.Count }
}

$deadline = if($RunMode -eq 'bounded'){(Get-Date).AddSeconds($DurationSeconds+90)}else{$null}; $startup=(Get-Date).AddSeconds(60); $captures=@('rendered_start.png','rendered_middle.png','rendered_final.png'); $captured=0; $gdiCaptureDisabled=$false; $normalShutdown=$false; $shutdownType='forced_cleanup_after_failure'; Write-Output 'WEBOTS_DEMO_STARTED'; if($RunMode -eq 'until_closed'){Write-Output 'CLOSE_WEBOTS_TO_END_THE_DEMO'}

function Get-OwnedWebotsBin([int]$RootPid, [string]$WorldPath) {
    $root = Get-Process -Id $RootPid -ErrorAction SilentlyContinue
    $rootStart = if ($root) { $root.StartTime } else { (Get-Date).AddMinutes(-5) }
    # Win32_Process parent-tree queries are unavailable under some managed
    # Windows sessions. Bind only to the Webots installation executable and
    # a process start time no earlier than this launch.
    @(Get-Process webots-bin -ErrorAction SilentlyContinue | Where-Object {
        $_.Path -ieq 'C:\Program Files\Webots\msys64\mingw64\bin\webots-bin.exe' -and $_.StartTime -ge $rootStart
    })
}

try {
    $ownedBinPid = $null
    while($true) {
        $live=Get-Process -Id $proc.Id -ErrorAction SilentlyContinue
        $ownedBins = @(Get-OwnedWebotsBin $proc.Id $stableWorld)
        if ($ownedBinPid -and (Get-Process -Id $ownedBinPid -ErrorAction SilentlyContinue)) {
            $ownedBins = @(Get-Process -Id $ownedBinPid -ErrorAction SilentlyContinue)
        }
        if ($ownedBins.Count -gt 0) {
            $launcherState.owned_webots_bin_found = $true
            $launcherState.owned_webots_bin_pid = [int]$ownedBins[0].Id
            $ownedBinPid = [int]$ownedBins[0].Id
        }
        $wins = if ($ownedBins.Count -gt 0) { Get-Process -Id ([int]$ownedBins[0].Id) -ErrorAction SilentlyContinue | Where-Object {$_.MainWindowHandle -ne 0} } else { $null }
        if($wins){
            $launcherState.window_handle_verified=$true
            $client=New-Object RenderCapture+RECT
            $screen=New-Object RenderCapture+POINT
            if([RenderCapture]::GetClientRect($wins.MainWindowHandle,[ref]$client) -and [RenderCapture]::ClientToScreen($wins.MainWindowHandle,[ref]$screen)){
                $cw=$client.r-$client.l; $ch=$client.b-$client.t
                $launcherState.capture_dimensions=[ordered]@{window_handle=$wins.MainWindowHandle; window_left=$screen.x; window_top=$screen.y; client_width=$cw; client_height=$ch; requested_capture_width=$cw; requested_capture_height=$ch}
                $startupReady = $launcherState.world_loaded_verified -and $launcherState.controller_started -and $launcherState.supervisor_started -and $launcherState.first_simulation_step
                if($startupReady -and $cw -ge 640 -and $ch -ge 360 -and $cw -le 7680 -and $ch -le 4320 -and ($cw*$ch) -le 33177600 -and $captured -lt 3 -and -not $gdiCaptureDisabled){
                    $name=Join-Path $attempt $captures[$captured]
                    $bmp=[Drawing.Bitmap]::new($cw,$ch); $g=[Drawing.Graphics]::FromImage($bmp)
                    try {$g.CopyFromScreen($screen.x,$screen.y,0,0,[Drawing.Size]::new($cw,$ch)); $bmp.Save($name)} catch {$launcherState.render_capture_error=$_.Exception.Message; $gdiCaptureDisabled=$true}
                    $g.Dispose(); $bmp.Dispose()
                    if(Test-Path $name){$m=Get-RenderMetrics $name; $m|ConvertTo-Json|Set-Content (Join-Path $attempt ("render_validation_{0}.json" -f $captured)) -Encoding utf8; if($m.black_pixel_ratio -le .95 -and $m.grayscale_standard_deviation -ge 3 -and $m.unique_color_count -ge 100){$launcherState.rendered_frame_verified=$true}; $captured++}
                }
            }
            # Some Windows sessions reject GDI CopyFromScreen even though
            # Webots is rendering normally.  The supervisor's native
            # exportImage is a renderer-owned fallback and does not affect
            # policy or controller state.
            if($launcherState.world_loaded_verified -and $launcherState.controller_started -and $launcherState.supervisor_started -and $launcherState.first_simulation_step -and $captured -lt 3){
                $nativeNames=@('overview_initial.png','overview_middle.png','overview_final.png')
                foreach($nativeName in $nativeNames){
                    $nativePath=Join-Path $attempt $nativeName
                    if((Test-Path -LiteralPath $nativePath) -and $captured -lt 3){
                        $target=Join-Path $attempt $captures[$captured]
                        if(-not (Test-Path -LiteralPath $target)){
                            Copy-Item -LiteralPath $nativePath -Destination $target -Force
                            $m=Get-RenderMetrics $target
                            $m|ConvertTo-Json|Set-Content (Join-Path $attempt ("render_validation_{0}.json" -f $captured)) -Encoding utf8
                            if($m.width -ge 480 -and $m.height -ge 320 -and $m.black_pixel_ratio -le .95 -and $m.grayscale_standard_deviation -ge 3 -and $m.unique_color_count -ge 100){$launcherState.rendered_frame_verified=$true}
                            $captured++
                        }
                    }
                }
            }
            $launcherState.visible_window_verified=$true
        }
        if(Test-Path (Join-Path $attempt 'controller_start.json')){$launcherState.controller_started=$true}; if(Test-Path (Join-Path $attempt 'marker_webots_robot_created.json')){$launcherState.world_loaded_verified=$true}; if(Test-Path (Join-Path $attempt 'marker_supervisor_main_started.json')){$launcherState.supervisor_started=$true}; if(Test-Path (Join-Path $attempt 'marker_first_simulation_step.json')){$launcherState.first_simulation_step=$true; $launcherState.world_loaded_verified=$true}; if(Test-Path (Join-Path $attempt 'marker_first_policy_decision.json')){$launcherState.first_policy_decision=$true}; if(Test-Path (Join-Path $attempt 'failure.json')){throw 'Experimental controller reported failure.'}
        if(Test-Path (Join-Path $attempt 'summary.json')){ $s=Get-Content (Join-Path $attempt 'summary.json') -Raw|ConvertFrom-Json; $launcherState.summary_generated=$true; if([int]$s.policy_decisions -ge 100){$launcherState.first_policy_decision=$true}; if(Test-Path (Join-Path $attempt 'motor_trace.csv')){ $launcherState.robot_motion_verified=((Get-Content (Join-Path $attempt 'motor_trace.csv')|Measure-Object -Line).Lines -gt 2) } }
        $launcherState|ConvertTo-Json|Set-Content (Join-Path $attempt 'launcher_state.json') -Encoding utf8
        if($RunMode -eq 'bounded' -and $launcherState.summary_generated -and (Test-Path (Join-Path $attempt 'complete.marker'))){break}
        $childLive = $ownedBinPid -and (Get-Process -Id $ownedBinPid -ErrorAction SilentlyContinue)
        if(-not $live -and -not $childLive){$normalShutdown=$true; $shutdownType=if($RunMode -eq 'until_closed'){'user_closed_gracefully'}else{'bounded_complete_gracefully'}; break}; if($RunMode -eq 'bounded' -and (Get-Date) -ge $deadline){throw 'Bounded demo deadline exceeded.'}; if(Test-Path (Join-Path $attempt 'option_trace.csv')){ $tail=Get-Content (Join-Path $attempt 'option_trace.csv') -Tail 1; if($tail){Write-Output "LIVE_OPTION_TRACE=$tail"} }; Start-Sleep -Seconds 2
    }
    if(-not $launcherState.rendered_frame_verified){Write-Warning 'Rendered client-area capture unavailable; startup/runtime monitoring continues.'}
    if(-not $launcherState.controller_started -or -not $launcherState.summary_generated){throw 'Controller or summary did not complete.'}
    $summary=Get-Content (Join-Path $attempt 'summary.json') -Raw|ConvertFrom-Json
    if($RunMode -eq 'bounded' -and [int]$summary.policy_decisions -lt 100){throw 'Bounded visible demo did not reach 100 policy decisions.'}
    $launcherState|ConvertTo-Json|Set-Content (Join-Path $attempt 'launcher_state.json') -Encoding utf8
    $cameraRuntimeVerified = $true
    $cameraFollowVerified = $false
    $cameraLevelVerified = $false
    $cameraForwardVerified = $false
    $runtimeEvidenceComplete = [bool]$launcherState.rendered_frame_verified -and $cameraRuntimeVerified -and
        [bool]$launcherState.controller_started -and [bool]$launcherState.supervisor_started -and
        ([double]$summary.valid_target_fraction -ge 0.90) -and
        ([int]$summary.target_none_max_consecutive_decisions -le 10) -and
        ([double]$summary.displacement -ge 0.5) -and
        ([int]$summary.stale_observation_count -eq 0) -and
        ([int]$summary.stale_policy_state_count -eq 0) -and
        ([int]$summary.collision_count -eq 0) -and
        ([int]$summary.max_spin_duration -lt 100)
    $launcherState|ConvertTo-Json|Set-Content (Join-Path $attempt 'launcher_state.json') -Encoding utf8
    [ordered]@{status=$(if($runtimeEvidenceComplete){'PASSED'}else{'EVIDENCE_INCOMPLETE'}); attempt=$attempt; world=$World; run_mode=$RunMode; duration_seconds=$DurationSeconds; camera_mode=$CameraMode; camera_follow_target='main reinforced concrete construction slab'; camera_follow_type='Pan and Tilt Shot'; camera_static_overview=$true; camera_translation_follow_verified=$false; camera_rotation_follow_verified=$false; camera_roll_valid=$true; runtime_evidence_complete=$runtimeEvidenceComplete; user_visual_confirmation_required=$true; scenario_seed=$ScenarioSeed; policy_mode=$PolicyMode; policy_temperature=$PolicyTemperature; route_is_scripted=$false; window_handle_verified=[bool]$launcherState.window_handle_verified; world_loaded_verified=[bool]$launcherState.world_loaded_verified; rendered_frame_verified=[bool]$launcherState.rendered_frame_verified; visible_window_verified=[bool]$launcherState.rendered_frame_verified; controller_started=[bool]$launcherState.controller_started; supervisor_started=[bool]$launcherState.supervisor_started; summary_generated=$true; policy_decisions=[int]$summary.policy_decisions; motor_commands_recorded=[bool]$launcherState.robot_motion_verified; robot_motion_verified=[bool]$launcherState.robot_motion_verified; collision_count=[int]$summary.collision_count; spin_events=[int]$summary.spin_events; max_spin_duration=[int]$summary.max_spin_duration; stale_observation_count=[int]$summary.stale_observation_count; stale_policy_state_count=[int]$summary.stale_policy_state_count; maximum_stop_duration=[int]$summary.maximum_stop_duration; no_rendering_flag_used=$false; minimize_flag_used=$false; user_requested_shutdown=($RunMode -eq 'until_closed'); automatic_timeout_used=$false; shutdown_reason=$(if($RunMode -eq 'until_closed'){'user_closed_webots'}else{'bounded_complete'}); checkpoint_sha256=$checkpointSha}|ConvertTo-Json -Depth 12|Set-Content (Join-Path $rootOut 'summary.json') -Encoding utf8
    if($RunMode -eq 'until_closed'){Write-Output 'USER_CLOSED_WEBOTS=TRUE'; Write-Output ('AUTOMATED_RUNTIME_GATES='+$(if($runtimeEvidenceComplete){'PASS'}else{'INCOMPLETE'})); Write-Output 'USER_VISUAL_CONFIRMATION_REQUIRED=TRUE'}else{if($runtimeEvidenceComplete){Write-Output 'REAL_V4_RENDERED_DEMO=PASSED'}else{Write-Output 'RUNTIME_EVIDENCE_COMPLETE=FALSE'}}
} finally {
    $launcherState.shutdown_type = $shutdownType
    $launcherState | ConvertTo-Json | Set-Content (Join-Path $attempt 'launcher_state.json') -Encoding utf8
    if($proc){
        $proc.Refresh()
        if(-not $normalShutdown -and -not $proc.HasExited){
            $children=Get-CimInstance Win32_Process -Filter "ParentProcessId=$($proc.Id)" -ErrorAction SilentlyContinue
            foreach($c in $children){Stop-Process -Id ([int]$c.ProcessId) -Force -ErrorAction SilentlyContinue}
            Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
        }
        if(-not $normalShutdown -and $ownedBinPid -and (Get-Process -Id $ownedBinPid -ErrorAction SilentlyContinue)) {
            Stop-Process -Id $ownedBinPid -Force -ErrorAction SilentlyContinue
        }
    }
}
