param(
    [string]$Python = "",
    [string]$Config = ""
)

$ErrorActionPreference = "Stop"
$Repo = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
if ([string]::IsNullOrWhiteSpace($Python)) {
    $Python = (Get-Command python -ErrorAction Stop).Source
}
if ([string]::IsNullOrWhiteSpace($Config)) {
    $Config = Join-Path $Repo "configs\collection.actual-only.48workers.example.json"
}
$Config = (Resolve-Path -LiteralPath $Config).Path
$Campaign = Get-Content -LiteralPath $Config -Raw -Encoding UTF8 | ConvertFrom-Json
$RunRoot = Join-Path $Repo ([string]$Campaign.run_root)
$Logs = Join-Path $RunRoot "logs"
$PidFile = Join-Path $Logs "supervisor.pid"

if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw "Python interpreter not found: $Python"
}
if ([string]::IsNullOrWhiteSpace([Environment]::GetEnvironmentVariable("DEEPSEEK_API_KEY"))) {
    throw "DEEPSEEK_API_KEY is missing; refusing to call the API"
}
if (Test-Path -LiteralPath $PidFile) {
    $ExistingPid = [int](Get-Content -LiteralPath $PidFile -Raw)
    if (Get-Process -Id $ExistingPid -ErrorAction SilentlyContinue) {
        throw "Supervisor is already running with PID $ExistingPid"
    }
}

New-Item -ItemType Directory -Force -Path $Logs | Out-Null
$Process = Start-Process `
    -FilePath $Python `
    -ArgumentList @(
        "scripts/run_hard_collection_campaign.py",
        "--config", $Config,
        "--repo-root", $Repo,
        "--python", $Python
    ) `
    -WorkingDirectory $Repo `
    -WindowStyle Hidden `
    -RedirectStandardOutput (Join-Path $Logs "supervisor.stdout.log") `
    -RedirectStandardError (Join-Path $Logs "supervisor.stderr.log") `
    -PassThru
Set-Content -LiteralPath $PidFile -Value $Process.Id -Encoding ascii

[pscustomobject]@{
    campaign = $Campaign.campaign_id
    supervisor_pid = $Process.Id
    run_root = $RunRoot
    monitor = Join-Path $PSScriptRoot "monitor_local.ps1"
    stop = Join-Path $PSScriptRoot "stop_local.ps1"
}
