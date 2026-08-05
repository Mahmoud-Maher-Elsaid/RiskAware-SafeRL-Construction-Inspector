$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$repo = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$report = Join-Path $repo 'reports\strong_policy_upgrade\webots_v4_final\summary.json'
$out = Join-Path $repo 'reports\strong_policy_upgrade\webots_v4_final'
New-Item -ItemType Directory -Force -Path $out | Out-Null
$payload = [ordered]@{
    runtime_mode = 'experimental'
    production_replacement_approved = $false
    status = 'NOT_IMPLEMENTED'
    reason = 'No validated Webots adapter connects the hierarchical v4 policy to the existing Webots controller; production v1 Webots evidence is intentionally not relabeled.'
    worlds = @('site_small', 'site_medium', 'site_dynamic')
}
$payload | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $report -Encoding UTF8
throw 'Experimental hierarchical Webots adapter is not implemented; v1 smoke evidence was not relabeled as v4.'
