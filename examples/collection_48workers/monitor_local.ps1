param(
    [string]$Config = "",
    [int]$Tail = 0
)

$ErrorActionPreference = "Stop"
$Repo = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
if ([string]::IsNullOrWhiteSpace($Config)) {
    $Config = Join-Path $Repo "configs\collection.actual-only.48workers.example.json"
}
$Campaign = Get-Content -LiteralPath $Config -Raw -Encoding UTF8 | ConvertFrom-Json
$RunRoot = Join-Path $Repo ([string]$Campaign.run_root)
$StatePath = Join-Path $RunRoot "supervisor_state.json"
$SupervisorPath = Join-Path $RunRoot "supervisor.json"

if (-not (Test-Path -LiteralPath $StatePath)) {
    Write-Output "No supervisor_state.json yet. Check $RunRoot\logs"
    exit 1
}

$State = Get-Content -LiteralPath $StatePath -Raw -Encoding UTF8 | ConvertFrom-Json
$Supervisor = if (Test-Path -LiteralPath $SupervisorPath) {
    Get-Content -LiteralPath $SupervisorPath -Raw -Encoding UTF8 | ConvertFrom-Json
} else { $null }
$Alive = $false
if ($null -ne $Supervisor) {
    $Alive = $null -ne (Get-Process -Id $Supervisor.pid -ErrorAction SilentlyContinue)
}

Write-Output ("campaign={0} alive={1} disk_free_gib={2} updated={3}" -f `
    $State.campaign_id, $Alive, $State.disk_free_gib, $State.updated_at)

$Rows = foreach ($LaneConfig in $Campaign.lanes) {
    $Name = [string]$LaneConfig.name
    $Target = [int]$LaneConfig.limit
    $Lane = $State.lanes.$Name
    $Pipeline = $State.pipelines.$Name
    $Raw = if ($null -ne $Pipeline.queues.raw) { [int]$Pipeline.queues.raw } else { 0 }
    $Normalized = if ($null -ne $Pipeline.queues.normalized) { [int]$Pipeline.queues.normalized } else { 0 }
    $Verified = if ($null -ne $Pipeline.queues.verifier) { [int]$Pipeline.queues.verifier } else { 0 }
    $Filled = if ($Target -gt 0) {
        [Math]::Min(24, [int][Math]::Floor(24 * $Raw / $Target))
    } else { 0 }
    $Bar = ("#" * $Filled) + ("." * (24 - $Filled))
    [pscustomobject]@{
        lane = $Name
        status = $Lane.status
        progress = "[$Bar] $Raw/$Target"
        normalized = $Normalized
        verified = $Verified
        api_in_flight = $Pipeline.runtime.api_in_flight
        verify_in_flight = $Pipeline.runtime.verifier_in_flight
    }
}
$Rows | Format-Table -AutoSize

if ($Tail -gt 0) {
    foreach ($Row in $Rows) {
        $Log = Join-Path (Join-Path $RunRoot "logs") ($Row.lane + ".log")
        if (Test-Path -LiteralPath $Log) {
            Write-Output ("--- {0} (last {1}) ---" -f $Row.lane, $Tail)
            Get-Content -LiteralPath $Log -Tail $Tail -Encoding UTF8
        }
    }
}
