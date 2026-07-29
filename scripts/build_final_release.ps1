param(
    [string]$RepoRoot = "F:\AI\My_Project\RiskAware-SafeRL-Construction-Inspector"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $RepoRoot
$Output = Join-Path $RepoRoot "reports\final_submission\release_manifest.json"
$Files = git ls-files
$Records = foreach ($Relative in $Files) {
    $Path = Join-Path $RepoRoot $Relative
    if (Test-Path -LiteralPath $Path -PathType Leaf) {
        $Item = Get-Item -LiteralPath $Path
        [ordered]@{
            path = $Relative.Replace("\", "/")
            bytes = $Item.Length
            sha256 = (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
        }
    }
}
$Manifest = [ordered]@{
    schema_version = 1
    generated_at = [DateTimeOffset]::Now.ToString("o")
    git_commit = (git rev-parse HEAD)
    tracked_file_count = @($Records).Count
    files = @($Records)
}
$Manifest | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $Output -Encoding utf8
Write-Host "FINAL_RELEASE_MANIFEST=PASSED ($(@($Records).Count) tracked files)"
