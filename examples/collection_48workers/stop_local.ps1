param([string]$Config = "")

$ErrorActionPreference = "Stop"
$Repo = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
if ([string]::IsNullOrWhiteSpace($Config)) {
    $Config = Join-Path $Repo "configs\collection.actual-only.48workers.example.json"
}
$Campaign = Get-Content -LiteralPath $Config -Raw -Encoding UTF8 | ConvertFrom-Json
$RunRoot = Join-Path $Repo ([string]$Campaign.run_root)
$StopPath = Join-Path $RunRoot "STOP"
New-Item -ItemType Directory -Force -Path $RunRoot | Out-Null
[ordered]@{
    reason = "operator_requested"
    requested_at = [DateTimeOffset]::UtcNow.ToString("o")
} | ConvertTo-Json | Set-Content -LiteralPath $StopPath -Encoding UTF8
Write-Output "Graceful stop requested; durable queues are preserved."
