param(
    [string]$ScanPath = ''
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

. (Join-Path $PSScriptRoot 'common.ps1')

$targetRoot = if ([string]::IsNullOrWhiteSpace($ScanPath)) { Get-RepositoryRoot } else { $ScanPath }

$secretFindings = @(Find-HarnessSecretLeak -RootPath $targetRoot)
if ($secretFindings.Count -gt 0) {
    foreach ($finding in $secretFindings) {
        Write-Host ("Secreto detectado [{0}] en {1}:{2}" -f $finding.Rule, $finding.File, $finding.LineNumber) -ForegroundColor Red
    }
    exit 1
}

if (-not [string]::IsNullOrWhiteSpace($ScanPath)) {
    Write-Host 'Escaneo de secretos: sin hallazgos.' -ForegroundColor Green
    exit 0
}

$analyzerVersion = Get-HarnessVersion -Name 'PSScriptAnalyzer'
$analyzer = Get-Module -ListAvailable -Name PSScriptAnalyzer |
    Where-Object { $_.Version.ToString() -eq $analyzerVersion } |
    Select-Object -First 1

if (-not $analyzer) {
    throw "PSScriptAnalyzer $analyzerVersion no esta instalado. Ejecuta: Install-Module PSScriptAnalyzer -RequiredVersion $analyzerVersion -Scope CurrentUser"
}

Import-Module $analyzer.Path -Force
$settings = Join-Path (Get-RepositoryRoot) 'PSScriptAnalyzerSettings.psd1'
$diagnostics = Invoke-ScriptAnalyzer -Path (Get-RepositoryRoot) -Recurse -Settings $settings

if ($diagnostics) {
    $diagnostics | Format-Table -AutoSize
    exit 1
}

Write-Host 'PSScriptAnalyzer y escaneo de secretos: sin diagnosticos.' -ForegroundColor Green
