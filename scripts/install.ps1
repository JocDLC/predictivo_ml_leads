# Instalador del harness: clona la fuente en un directorio temporal y compone
# solo los perfiles seleccionados en el directorio destino.

param (
    [string]$TargetDir = "$env:USERPROFILE\dev-harness-template",
    [string]$Repository = 'https://github.com/JocDLC/dev-harness-template.git',
    [string[]]$Profiles = @('base'),
    [string]$GitHubMode = 'skip',
    [switch]$SkipBootstrap
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

if (($env:DEV_HARNESS_FAKE_GIT_MISSING -eq '1') -or (-not (Get-Command git -ErrorAction SilentlyContinue))) {
    Write-Error 'Git no esta instalado. Instalalo desde https://git-scm.com'
    exit 1
}

if (Test-Path $TargetDir) {
    Write-Error "El directorio $TargetDir ya existe. Borralo o usa setup-project.ps1 para actualizar de forma idempotente."
    exit 1
}

$tempClone = Join-Path ([System.IO.Path]::GetTempPath()) ('dev-harness-src-{0}' -f [guid]::NewGuid().ToString('N'))
try {
    Write-Host '==> Obteniendo plantilla fuente en copia temporal...'
    & git clone --quiet $Repository $tempClone
    if ($LASTEXITCODE -ne 0) {
        Write-Error "No se pudo clonar el template desde $Repository"
        exit $LASTEXITCODE
    }

    $localManifest = if (Test-Path $Repository) { Join-Path $Repository 'scripts\profiles-manifest.json' } else { '' }
    $sourceForCompose = if ((-not [string]::IsNullOrWhiteSpace($localManifest)) -and (Test-Path $localManifest)) { $Repository } else { $tempClone }
    $manifestInSource = Join-Path $sourceForCompose 'scripts\profiles-manifest.json'
    if (Test-Path $manifestInSource) {
        . (Join-Path $sourceForCompose 'scripts\common.ps1')
        Invoke-HarnessCompose `
            -TargetDir $TargetDir `
            -SourceDir $sourceForCompose `
            -Profiles $Profiles `
            -GitHubMode $GitHubMode `
            -ManifestPath $manifestInSource | Out-Null

        if (-not $SkipBootstrap) {
            $bootstrapScript = Join-Path $TargetDir 'scripts\bootstrap.ps1'
            if (Test-Path $bootstrapScript) {
                Write-Host '==> Corriendo bootstrap para los perfiles seleccionados...'
                & $bootstrapScript -Profiles $Profiles
            }
        }
    }
    else {
        if (($Profiles -join ',') -notmatch '^(?i:base)?$' -or $GitHubMode -notin @('skip', 'codex-plugin', 'gh-cli', 'manual')) {
            throw 'Fuente legacy sin catalogo: solo base y modos GitHub validos; no se creo el destino.'
        }
        Copy-Item -Path $tempClone -Destination $TargetDir -Recurse -Force
        $bootstrapScript = Join-Path $TargetDir 'scripts\bootstrap.ps1'
        if ((-not $SkipBootstrap) -and (Test-Path $bootstrapScript)) {
            Write-Host '==> Corriendo bootstrap...'
            & $bootstrapScript
        }
        $setupScript = Join-Path $TargetDir 'scripts\setup-project.ps1'
        if (Test-Path $setupScript) {
            Write-Host '==> Corriendo setup-project...'
            & $setupScript
        }
    }
}
finally {
    if (Test-Path $tempClone) {
        $tempRoot = [IO.Path]::GetFullPath([IO.Path]::GetTempPath()).TrimEnd('\') + '\'
        $resolvedClone = [IO.Path]::GetFullPath($tempClone)
        if (-not $resolvedClone.StartsWith($tempRoot, [StringComparison]::OrdinalIgnoreCase)) { throw 'Limpieza fuera de temp.' }
        Remove-Item -LiteralPath $resolvedClone -Recurse -Force -ErrorAction SilentlyContinue
    }
}

Write-Host "Listo. Entra en $TargetDir y abrelo en tu IDE."
