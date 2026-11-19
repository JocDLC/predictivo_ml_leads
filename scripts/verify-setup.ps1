param([string[]]$Profiles = @(), [switch]$DevelopmentTools)
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
. (Join-Path $PSScriptRoot 'common.ps1')
. (Join-Path $PSScriptRoot 'versions.ps1')
$context = Get-HarnessContext
$active = @($context.profiles)
if ($Profiles.Count -gt 0) {
    $active = @(Resolve-HarnessProfileList -Profiles $Profiles)
    if ($context.role -eq 'consumer' -and @($active | Where-Object { $_ -notin $context.profiles }).Count -gt 0) {
        throw 'Perfil inactivo; agregarlo explicitamente mediante setup-project.'
    }
}
$script:failures = 0
function Write-Check {
    param([string]$Id, [ValidateSet('PASS', 'FAIL', 'SKIP', 'PENDIENTE')][string]$State, [string]$Detail)
    Write-Host "[$State] $Id : $Detail"
    if ($State -eq 'FAIL') { $script:failures++ }
}
function Test-Version {
    param([string]$Name, [string]$Command, [string]$Pattern, [string]$Policy = 'exact', [string[]]$Arguments = @('--version'))
    try {
        $expected = Get-HarnessVersion $Name
        $actual = Get-HarnessObservedVersion -Command $Command -Pattern $Pattern -Arguments $Arguments
        if (-not (Test-HarnessVersionPolicy -Actual $actual -Expected $expected -Policy $Policy)) {
            throw "observada $actual; requerida $Policy $expected"
        }
        Write-Check "version.$Name" PASS "observada $actual; $Policy $expected"
    }
    catch { Write-Check "version.$Name" FAIL $_.Exception.Message }
}
try { $count = Test-HarnessResource; Write-Check resources PASS "$count recursos" }
catch { Write-Check resources FAIL $_.Exception.Message }
$psMinimum = Get-HarnessVersion PowerShell
if ($PSVersionTable.PSVersion -ge [version]$psMinimum) { Write-Check version.PowerShell PASS "observada $($PSVersionTable.PSVersion); minimum $psMinimum" }
else { Write-Check version.PowerShell FAIL "requiere $psMinimum" }
Test-Version Git git '^git version (\d+\.\d+\.\d+)' minimum
foreach ($profileName in @('openspec', 'graphify', 'archify', 'codex')) {
    if ($profileName -notin $active) { Write-Check "profile.$profileName" SKIP 'perfil inactivo [OMITIDO]' }
}
if ('openspec' -in $active -or 'archify' -in $active) { Test-Version Node.js node '^v(\d+\.\d+\.\d+)$' minimum }
if ('openspec' -in $active) { Test-Version '@fission-ai/openspec' openspec '^(\d+\.\d+\.\d+)$' }
if ('graphify' -in $active) {
    Test-Version Python python '^Python (\d+\.\d+\.\d+)$' minimum
    $launcher = Get-Command graphify -ErrorAction SilentlyContinue
    if ($launcher) {
        Test-Version graphifyy python '^graphifyy (\d+\.\d+\.\d+)$' exact @((Join-Path $PSScriptRoot 'probe.py'), '--graphify-launcher', $launcher.Source)
    }
    else { Write-Check version.graphifyy FAIL 'Graphify CLI ausente.' }
    Test-Version uv uv '^uv (\d+\.\d+\.\d+)(?:\s|$)'
    Write-Check graphify.skill PENDIENTE 'Calificar skill y registro MCP en el cliente utilizado; no se leen credenciales.'
}
if ('archify' -in $active) { Write-Check archify.skill PENDIENTE 'Calificar instalacion y version del skill; docs/architecture no acredita instalacion.' }
if ('codex' -in $active) { Write-Check codex.login PENDIENTE 'Instalacion e inicio de sesion manuales (openai.chatgpt).' }
if ($DevelopmentTools) {
    $names = @('PSScriptAnalyzer')
    if ($context.role -eq 'authoring') { $names += 'Pester' }
    foreach ($name in $names) {
        $pin = Get-HarnessVersion $name
        $module = Get-Module -ListAvailable -Name $name | Where-Object { $_.Version.ToString() -eq $pin } | Select-Object -First 1
        if ($module) { Write-Check "module.$name" PASS "observada $($module.Version); exact $pin" }
        else { Write-Check "module.$name" FAIL "requiere $pin" }
    }
}
$drive = Get-PSDrive -Name ([IO.Path]::GetPathRoot((Get-RepositoryRoot)).TrimEnd('\', ':')) -ErrorAction SilentlyContinue
if ($drive -and $null -ne $drive.Free -and $drive.Free -lt 5GB) { Write-Host '[AVISO] Espacio libre menor a 5 GB.' }
if ((Get-RepositoryRoot) -match 'OneDrive') { Write-Host '[AVISO] Gitignore no excluye archivos de la sincronizacion OneDrive.' }
Write-Check git.identity PENDIENTE 'Confirmar identidad Git local antes del primer commit; no se cambia configuracion.'
if ($script:failures -gt 0) { Write-Host "FAIL: $script:failures requisitos fallidos"; exit 1 }
Write-Host 'PASS: requisitos automaticos verificados; revisar PENDIENTE manuales.'
exit 0
