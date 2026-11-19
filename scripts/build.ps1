$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

. (Join-Path $PSScriptRoot 'common.ps1')
$context = Get-HarnessContext
$null = Test-HarnessResource

$parseErrors = @()
foreach ($file in Get-PowerShellSourceFile) {
    $fileErrors = $null
    [System.Management.Automation.Language.Parser]::ParseFile(
        $file.FullName,
        [ref]$null,
        [ref]$fileErrors
    ) | Out-Null
    $parseErrors += $fileErrors
}

if ($parseErrors.Count -gt 0) {
    $parseErrors | Format-List
    exit 1
}

Push-Location (Get-RepositoryRoot)
try {
    if ($context.profiles -contains 'openspec') {
        & openspec validate --all
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    }
    else { Write-Host '[SKIP] openspec: perfil inactivo.' }
}
finally {
    Pop-Location
}

Write-Host 'Build del harness validado.' -ForegroundColor Green
