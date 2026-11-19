$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

. (Join-Path $PSScriptRoot 'common.ps1')

$context = Get-HarnessContext
if ($context.role -eq 'consumer') {
    $null = Test-HarnessResource
    $smoke = Join-Path (Get-RepositoryRoot) 'tests/harness/smoke.ps1'
    $result = & $smoke
    if (-not $result -or -not $result.PSObject.Properties['Cases'] -or
        $result.Cases -lt 1 -or $result.Passed -ne $result.Cases) { throw 'Smoke sin casos aprobados.' }
    exit 0
}

$pesterVersion = Get-HarnessVersion -Name 'Pester'
$pester = Get-Module -ListAvailable -Name Pester |
    Where-Object { $_.Version.ToString() -eq $pesterVersion } |
    Select-Object -First 1

if (-not $pester) {
    throw "Pester $pesterVersion no esta instalado. Ejecuta: Install-Module Pester -RequiredVersion $pesterVersion -Scope CurrentUser"
}

Import-Module $pester.Path -Force
$testPath = Join-Path (Get-RepositoryRoot) 'tests\powershell'
$result = Invoke-Pester -Path $testPath -CI -PassThru

if ($result.FailedCount -gt 0 -or $result.PassedCount -eq 0 -or $result.Result -ne 'Passed') {
    exit 1
}
