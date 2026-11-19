param([switch]$Check, [string[]]$Path = @())
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

. (Join-Path $PSScriptRoot 'common.ps1')

$analyzerVersion = Get-HarnessVersion -Name 'PSScriptAnalyzer'
Import-Module PSScriptAnalyzer -RequiredVersion $analyzerVersion -Force

$files = if ($Path.Count -gt 0) { @($Path | ForEach-Object { Get-Item -LiteralPath $_ }) } else { @(Get-PowerShellSourceFile) }
$failures = 0
foreach ($file in $files) {
    $formatted = Invoke-Formatter -ScriptDefinition ((Get-Content -Raw $file.FullName).Replace("`r`n", "`n"))
    if ($Check) {
        $original = (Get-Content -Raw $file.FullName).Replace("`r`n", "`n").TrimEnd()
        if ($original -cne $formatted.Replace("`r`n", "`n").TrimEnd()) { Write-Host "FAIL formato: $($file.Name)"; $failures++ }
    }
    else {
        $canonical = $formatted.Replace("`r`n", "`n").TrimEnd().Replace("`n", "`r`n") + "`r`n"
        [IO.File]::WriteAllText($file.FullName, $canonical, [Text.UTF8Encoding]::new($true))
    }
}
if ($failures -gt 0) { exit 1 }
