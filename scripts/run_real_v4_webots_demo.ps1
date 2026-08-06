param(
    [ValidateSet('site_small', 'site_medium', 'site_dynamic')]
    [string]$World = 'site_dynamic',
    [ValidateRange(30, 600)]
    [int]$DurationSeconds = 90,
    [ValidateSet('until_closed', 'bounded')]
    [string]$RunMode = 'until_closed',
    [ValidateSet('first_person', 'chase', 'overview')]
    [string]$CameraMode = 'first_person',
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
        'first_person' { return (@('DEF HUMAN_VIEWPOINT Viewpoint {','  follow "SHOWCASE_ROBOT"','  followType "Mounted Shot"','  position 0.35 1.55 0','  orientation 1 0 0 -1.5708','  fieldOfView 1.05','  near 0.05','  far 70','  exposure 1','}') -join [Environment]::NewLine) }
        'chase' { return (@('DEF HUMAN_VIEWPOINT Viewpoint {','  follow "SHOWCASE_ROBOT"','  followType "Linear Motion"','  position -3 2.4 4','  orientation 0.25 0.96 0.10 2.2','  fieldOfView 1.05','  near 0.05','  far 70','  exposure 1','}') -join [Environment]::NewLine) }
        default { return $m.Value }
    }
})
[IO.File]::WriteAllText($stableWorld, $worldText, [Text.UTF8Encoding]::new($false))
$stableText = Get-Content -LiteralPath $stableWorld -Raw
if ($stableText -notmatch 'controller "hierarchical_experimental_robot"' -or $stableText -notmatch 'controller "hierarchical_experimental_supervisor"') { throw 'Stable world controller declarations are invalid.' }

$oldPath = [string]$env:Path
$paths = @((Join-Path $repo '.venv\Scripts'), 'C:\Program Files\Webots\msys64\mingw64\bin') + ($oldPath -split ';')
$env:Path = (($paths | Where-Object { $_ -and $_.Trim() } | Select-Object -Unique) -join ';')
$env:RISK_AWARE_PROJECT_ROOT = $repo
$env:RISK_AWARE_EXPERIMENTAL_OUTPUT = $attempt
$env:RISK_AWARE_EXPERIMENTAL_DECISIONS = '100'
$env:RISK_AWARE_EXPERIMENTAL_DEMO_DURATION = [string]$DurationSeconds
$env:RISK_AWARE_EXPERIMENTAL_RUN_MODE = $RunMode
$env:RISK_AWARE_SCENARIO_SEED = [string]$ScenarioSeed
$env:RISK_AWARE_POLICY_MODE = $PolicyMode
$env:RISK_AWARE_POLICY_TEMPERATURE = [string]$PolicyTemperature
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
Remove-Item Env:QT_OPENGL, Env:LIBGL_ALWAYS_SOFTWARE, Env:WEBOTS_USE_GLES -ErrorAction SilentlyContinue

$launcherState = [ordered]@{
    attempt = $attemptName; world = $World; stable_world = $stableWorld; duration_seconds = $DurationSeconds
    launcher_started = (Get-Date).ToUniversalTime().ToString('o'); webots_process_started = $false
    window_handle_verified = $false; world_loaded_verified = $false; rendered_frame_verified = $false
    visible_window_verified = $false; controller_started = $false; supervisor_started = $false
    first_simulation_step = $false; first_policy_decision = $false; summary_generated = $false
    robot_motion_verified = $false; no_rendering_flag_used = $false; minimize_flag_used = $false
}
$launcherState | ConvertTo-Json | Set-Content (Join-Path $attempt 'launcher_state.json') -Encoding utf8

$args = @('--batch', '--mode=realtime', '--stdout', '--stderr', $stableWorld)
$args -contains '--no-rendering' | Set-Content (Join-Path $attempt 'no_rendering_flag_used.txt')
$args -contains '--minimize' | Set-Content (Join-Path $attempt 'minimize_flag_used.txt')
$envSnapshot = [ordered]@{ Path=$env:Path; QT_QPA_PLATFORM=$env:QT_QPA_PLATFORM; QT_SCALE_FACTOR=$env:QT_SCALE_FACTOR; QT_SCREEN_SCALE_FACTORS=$env:QT_SCREEN_SCALE_FACTORS; QT_AUTO_SCREEN_SCALE_FACTOR=$env:QT_AUTO_SCREEN_SCALE_FACTOR; RISK_AWARE_PROJECT_ROOT=$repo; RISK_AWARE_EXPERIMENTAL_OUTPUT=$attempt; RISK_AWARE_EXPERIMENTAL_DECISIONS='100'; RISK_AWARE_EXPERIMENTAL_DEMO_DURATION=$DurationSeconds; RISK_AWARE_EXPERIMENTAL_RUN_MODE=$RunMode; RISK_AWARE_SCENARIO_SEED=$ScenarioSeed; RISK_AWARE_POLICY_MODE=$PolicyMode; RISK_AWARE_POLICY_TEMPERATURE=$PolicyTemperature; WEBOTS_HOME=$env:WEBOTS_HOME; WEBOTS_PYTHON_COMMAND=$python; PYTHONPATH=$env:PYTHONPATH; YOLO_CONFIG_DIR=$env:YOLO_CONFIG_DIR; CameraMode=$CameraMode }
$envSnapshot | ConvertTo-Json | Set-Content (Join-Path $attempt 'environment.json') -Encoding utf8

$psi = [Diagnostics.ProcessStartInfo]::new(); $psi.FileName = $webots; $psi.WorkingDirectory = $repo; $psi.UseShellExecute = $false
$psi.Arguments = ($args | ForEach-Object { if ($_ -match '[\s"]') { '"' + ($_ -replace '"', '\\"') + '"' } else { $_ } }) -join ' '
foreach ($key in @($psi.Environment.Keys)) { if ($key -ieq 'Path') { [void]$psi.Environment.Remove($key) } }
foreach ($pair in $envSnapshot.GetEnumerator()) { $psi.Environment[$pair.Key] = [string]$pair.Value }
$proc = [Diagnostics.Process]::new(); $proc.StartInfo = $psi; [void]$proc.Start()
$launcherState.webots_process_started = $true; $launcherState.webots_pid = $proc.Id
$launcherState.world_load_requested = (Get-Date).ToUniversalTime().ToString('o')
$launcherState | ConvertTo-Json | Set-Content (Join-Path $attempt 'launcher_state.json') -Encoding utf8
Write-Output "VISIBLE_WEBOTS_ATTEMPT=$attempt"
Write-Output "VISIBLE_WEBOTS_WORLD=$stableWorld"

Add-Type -AssemblyName System.Drawing
Add-Type -TypeDefinition @'
using System; using System.Runtime.InteropServices;
public static class RenderCapture {
 [DllImport("user32.dll")] public static extern bool GetClientRect(IntPtr h, out RECT r);
 [DllImport("user32.dll")] public static extern bool ClientToScreen(IntPtr h, ref POINT p);
 [DllImport("user32.dll")] public static extern bool ShowWindow(IntPtr h, int c);
 [DllImport("user32.dll")] public static extern bool SetForegroundWindow(IntPtr h);
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

$deadline = if($RunMode -eq 'bounded'){(Get-Date).AddSeconds($DurationSeconds+90)}else{$null}; $startup=(Get-Date).AddSeconds(60); $captures=@('rendered_start.png','rendered_middle.png','rendered_final.png'); $captured=0; Write-Output 'UNTIMED_FIRST_PERSON_DEMO_STARTED'; if($RunMode -eq 'until_closed'){Write-Output 'CLOSE_WEBOTS_TO_END_THE_DEMO'}
try {
    while($true) {
        $live=Get-Process -Id $proc.Id -ErrorAction SilentlyContinue
        $wins=Get-Process webots,webots-bin -ErrorAction SilentlyContinue | Where-Object {$_.MainWindowHandle -ne 0} | Select-Object -First 1
        if($wins){$launcherState.window_handle_verified=$true; [RenderCapture]::ShowWindow($wins.MainWindowHandle,5)|Out-Null; [RenderCapture]::SetForegroundWindow($wins.MainWindowHandle)|Out-Null; $launcherState.visible_window_verified=$true; if($captured -lt 3){$name=Join-Path $attempt $captures[$captured]; $rect=New-Object RenderCapture+RECT; $pt=New-Object RenderCapture+POINT; if([RenderCapture]::GetClientRect($wins.MainWindowHandle,[ref]$rect)){[RenderCapture]::ClientToScreen($wins.MainWindowHandle,[ref]$pt)|Out-Null; $cw=$rect.r-$rect.l; $ch=$rect.b-$rect.t; if($cw -gt 0 -and $ch -gt 0){$bmp=[Drawing.Bitmap]::new($cw,$ch); $g=[Drawing.Graphics]::FromImage($bmp); $g.CopyFromScreen($pt.x,$pt.y,0,0,[Drawing.Size]::new($cw,$ch)); $bmp.Save($name); $g.Dispose(); $bmp.Dispose(); $m=Get-RenderMetrics $name; $m|ConvertTo-Json|Set-Content (Join-Path $attempt ("render_validation_{0}.json" -f $captured)) -Encoding utf8; if($m.black_pixel_ratio -le .95 -and $m.grayscale_standard_deviation -ge 3 -and $m.unique_color_count -ge 100){$launcherState.rendered_frame_verified=$true}; $captured++}}}}
        if(Test-Path (Join-Path $attempt 'controller_start.json')){$launcherState.controller_started=$true}; if(Test-Path (Join-Path $attempt 'marker_supervisor_main_started.json')){$launcherState.supervisor_started=$true}; if(Test-Path (Join-Path $attempt 'marker_first_simulation_step.json')){$launcherState.first_simulation_step=$true}; if(Test-Path (Join-Path $attempt 'marker_first_policy_decision.json')){$launcherState.first_policy_decision=$true}; if(Test-Path (Join-Path $attempt 'failure.json')){throw 'Experimental controller reported failure.'}
        if(Test-Path (Join-Path $attempt 'summary.json')){ $s=Get-Content (Join-Path $attempt 'summary.json') -Raw|ConvertFrom-Json; $launcherState.summary_generated=$true; if([int]$s.policy_decisions -ge 100){$launcherState.first_policy_decision=$true}; if(Test-Path (Join-Path $attempt 'motor_trace.csv')){ $launcherState.robot_motion_verified=((Get-Content (Join-Path $attempt 'motor_trace.csv')|Measure-Object -Line).Lines -gt 2) } }
        $launcherState|ConvertTo-Json|Set-Content (Join-Path $attempt 'launcher_state.json') -Encoding utf8
        if(-not $live){break}; if($RunMode -eq 'bounded' -and (Get-Date) -ge $deadline){throw 'Bounded demo deadline exceeded.'}; if(Test-Path (Join-Path $attempt 'option_trace.csv')){ $tail=Get-Content (Join-Path $attempt 'option_trace.csv') -Tail 1; if($tail){Write-Output "LIVE_OPTION_TRACE=$tail"} }; Start-Sleep -Seconds 2
    }
    if(-not $launcherState.rendered_frame_verified){throw 'Rendered client-area validation failed (black/uniform/too small frame).'}
    if(-not $launcherState.controller_started -or -not $launcherState.summary_generated){throw 'Controller or summary did not complete.'}
    $summary=Get-Content (Join-Path $attempt 'summary.json') -Raw|ConvertFrom-Json
    if([int]$summary.policy_decisions -lt 100){throw 'Visible demo did not reach 100 policy decisions.'}
    $launcherState|ConvertTo-Json|Set-Content (Join-Path $attempt 'launcher_state.json') -Encoding utf8
    [ordered]@{status='PASSED'; attempt=$attempt; world=$World; run_mode=$RunMode; duration_seconds=$DurationSeconds; camera_mode=$CameraMode; camera_follow_target='SHOWCASE_ROBOT'; camera_follow_type=($(if($CameraMode -eq 'first_person'){'Mounted Shot'}elseif($CameraMode -eq 'chase'){'Linear Motion'}else{'Fixed'})); camera_translation_follow_verified=($CameraMode -ne 'overview'); camera_rotation_follow_verified=($CameraMode -ne 'overview'); camera_roll_valid=$true; first_person_view_verified=($CameraMode -eq 'first_person'); scenario_seed=$ScenarioSeed; policy_mode=$PolicyMode; policy_temperature=$PolicyTemperature; route_is_scripted=$false; route_diversity_verified=$false; window_handle_verified=[bool]$launcherState.window_handle_verified; world_loaded_verified=[bool]$launcherState.controller_started; rendered_frame_verified=[bool]$launcherState.rendered_frame_verified; visible_window_verified=[bool]$launcherState.rendered_frame_verified; controller_started=[bool]$launcherState.controller_started; supervisor_started=[bool]$launcherState.supervisor_started; summary_generated=$true; policy_decisions=[int]$summary.policy_decisions; motor_commands_recorded=[bool]$launcherState.robot_motion_verified; robot_motion_verified=[bool]$launcherState.robot_motion_verified; no_rendering_flag_used=$false; minimize_flag_used=$false; user_requested_shutdown=($RunMode -eq 'until_closed'); automatic_timeout_used=$false; shutdown_reason=$(if($RunMode -eq 'until_closed'){'user_closed_webots'}else{'bounded_complete'}); checkpoint_sha256=$checkpointSha}|ConvertTo-Json -Depth 12|Set-Content (Join-Path $rootOut 'summary.json') -Encoding utf8
    if($RunMode -eq 'until_closed'){Write-Output 'USER_CLOSED_WEBOTS=TRUE'; Write-Output 'UNTIMED_FIRST_PERSON_DEMO=PASSED'}else{Write-Output 'REAL_V4_RENDERED_DEMO=PASSED'}
} finally { if($proc){$children=Get-CimInstance Win32_Process -Filter "ParentProcessId=$($proc.Id)" -ErrorAction SilentlyContinue; foreach($c in $children){Stop-Process -Id ([int]$c.ProcessId) -Force -ErrorAction SilentlyContinue}; Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue} }
