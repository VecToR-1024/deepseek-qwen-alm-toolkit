[CmdletBinding()]
param(
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

$remote = "$($RemoteRoot.TrimEnd('/'))/experiments/qwen25_7b_instruct_hard_combined_alpha10_v1_20260804/monitor.sh"
Write-Host 'Opening a one-shot, read-only stage monitor.'
& ssh -o ServerAliveInterval=15 -o ServerAliveCountMax=3 -p $Port `
    "$User@$SshHost" "bash '$remote'"
exit $LASTEXITCODE
