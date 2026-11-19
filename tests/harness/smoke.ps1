$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
. (Join-Path $PSScriptRoot '../../scripts/common.ps1')
$count = Test-HarnessResource
if ($count -lt 1) { throw 'Suite sin casos.' }
Write-Host "PASS consumer.resources: $count casos"
[PSCustomObject]@{ Cases = $count; Passed = $count }
