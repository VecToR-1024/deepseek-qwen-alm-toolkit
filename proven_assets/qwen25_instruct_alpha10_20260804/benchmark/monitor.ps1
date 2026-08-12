[CmdletBinding()]
param(
    [ValidateRange(1, 3600)]
    [int]$Interval = 10,
    [switch]$Once,
    [Parameter(Mandatory = $true)]
    [ValidateNotNullOrEmpty()]
    [string]$SshHost,
    [Parameter(Mandatory = $true)]
    [ValidateRange(1, 65535)]
    [int]$Port,
    [Parameter(Mandatory = $true)]
    [ValidateNotNullOrEmpty()]
    [string]$User,
    [Parameter(Mandatory = $true)]
    [ValidatePattern('^/[A-Za-z0-9._/-]+$')]
    [string]$RemoteRoot
)

$remoteScript = "$($RemoteRoot.TrimEnd('/'))/benchmarks/qwen25_7b_instruct_hard_combined_alpha10_compare_v1_20260804/monitor.sh"
$mode = if ($Once) { ' --once' } else { '' }
$remoteCommand = "bash '$remoteScript' --interval $Interval$mode"

Write-Host 'Opening the read-only Qwen2.5-Instruct benchmark dashboard.'
Write-Host 'Enter the SSH password if prompted. Ctrl+C closes only this monitor.'
& ssh -tt -o ServerAliveInterval=15 -o ServerAliveCountMax=3 -p $Port `
    "$User@$SshHost" $remoteCommand
exit $LASTEXITCODE
