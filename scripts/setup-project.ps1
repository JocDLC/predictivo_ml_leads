# Setup por proyecto (ejecutar una vez al crear o actualizar un proyecto desde la plantilla).
# Compone los perfiles seleccionados e inicializa las herramientas correspondientes.

param(
    [string[]]$Profiles = @('base'),
    [string]$TargetDir = '',
    [string]$SourceDir = '',
    [string]$GitHubMode = 'skip',
    [switch]$UpdateExisting,
    [switch]$SkipExternalInit
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

. (Join-Path $PSScriptRoot 'common.ps1')

$resolvedSource = if ([string]::IsNullOrWhiteSpace($SourceDir)) { Get-RepositoryRoot } else { $SourceDir }
$resolvedTarget = if ([string]::IsNullOrWhiteSpace($TargetDir)) { Get-RepositoryRoot } else { $TargetDir }

$summary = Invoke-HarnessCompose `
    -TargetDir $resolvedTarget `
    -SourceDir $resolvedSource `
    -Profiles $Profiles `
    -GitHubMode $GitHubMode `
    -UpdateExisting:$UpdateExisting

if (-not $SkipExternalInit) {
    if ($summary.RequestedProfiles -contains 'openspec') {
        $openspecCommand = Get-Command openspec -ErrorAction SilentlyContinue
        $openspec = if ($openspecCommand) { $openspecCommand.Source } else { $null }
        if (-not $openspec -and $env:APPDATA) {
            $candidate = Join-Path (Join-Path $env:APPDATA 'npm') 'openspec.cmd'
            if (Test-Path $candidate) {
                $openspec = $candidate
            }
        }
        if ($openspec) {
            Write-Host '==> Inicializando OpenSpec para Antigravity, Windsurf y Codex...' -ForegroundColor Cyan
            & $openspec init $resolvedTarget '--tools=antigravity,windsurf,codex'
            if ($LASTEXITCODE -ne 0) { throw 'OpenSpec init fallo; copia aplicada parcialmente.' }
        }
        else {
            Write-Host '[PENDIENTE] OpenSpec CLI no detectado localmente; corre scripts/bootstrap.ps1 -Profiles openspec si deseas reinicializarlo.' -ForegroundColor Yellow
        }
    }

    if ($summary.RequestedProfiles -contains 'graphify') {
        if (Get-Command graphify -ErrorAction SilentlyContinue) {
            Write-Host '==> Construyendo grafo inicial con Graphify...' -ForegroundColor Cyan
            & graphify update $resolvedTarget
            if ($LASTEXITCODE -ne 0) { throw 'Graphify update fallo; copia aplicada parcialmente.' }
        }
        else {
            Write-Host '[PENDIENTE] Graphify CLI no detectado localmente; corre scripts/bootstrap.ps1 -Profiles graphify para construir el grafo.' -ForegroundColor Yellow
        }
    }
}

Write-Host ''
Write-Host 'Listo. Los archivos compuestos en el destino pueden commitearse al repositorio.' -ForegroundColor Green
