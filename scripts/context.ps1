# Contratos puros y contexto persistente; se carga desde common.ps1.
function Get-HarnessEntry {
    param([Parameter(Mandatory)]$Entry)
    if ($Entry -is [string]) {
        $source = $Entry
        $target = $Entry
    }
    elseif ($Entry -is [PSCustomObject] -and (@($Entry.PSObject.Properties.Name | Sort-Object) -join ',') -eq 'source,target') {
        $source = $Entry.source
        $target = $Entry.target
    }
    else { throw 'Entrada requiere string o source/target.' }
    if ($source -isnot [string] -or $target -isnot [string] -or
        -not (Test-SafeRelativePath $source) -or -not (Test-SafeRelativePath $target)) {
        throw 'Ruta insegura en manifiesto.'
    }
    $source = $source.Replace('\', '/')
    $target = $target.Replace('\', '/')
    if ($target -eq '.harness' -or $target -eq '.harness/context.json' -or $target.StartsWith('.harness/context.json/', [StringComparison]::OrdinalIgnoreCase)) { throw 'Ruta reservada del contexto.' }
    return [PSCustomObject]@{ Source = $source; Target = $target }
}

function Get-HarnessPlanEntry {
    param([string[]]$Profiles, [string]$ManifestPath = (Join-Path $PSScriptRoot 'profiles-manifest.json'))
    $manifest = Get-HarnessManifest $ManifestPath
    $selected = @(Resolve-HarnessProfileList -Profiles $Profiles -ManifestPath $ManifestPath)
    $entries = [ordered]@{}
    foreach ($profileName in $selected) {
        foreach ($item in $manifest.profiles.$profileName) {
            $entry = Get-HarnessEntry $item
            if ($entries.Contains($entry.Target) -and ($entries[$entry.Target].Source -cne $entry.Source -or $entries[$entry.Target].Target -cne $entry.Target)) {
                throw "Colision de archivos: $($entry.Target)"
            }
            $entries[$entry.Target] = $entry
        }
    }
    foreach ($key in @($entries.Keys)) {
        $parent = $key
        while ($parent.Contains('/')) {
            $parent = $parent.Substring(0, $parent.LastIndexOf('/'))
            if ($entries.Contains($parent)) { throw "Colision archivo/directorio: $key" }
        }
    }
    return @($entries.Values)
}

function Get-HarnessContext {
    param([string]$RootPath = (Get-RepositoryRoot), [switch]$AllowMissing,
        [string]$ManifestPath = (Join-Path $PSScriptRoot 'profiles-manifest.json'))
    $path = Join-Path $RootPath '.harness/context.json'
    Assert-HarnessPath $path
    if (-not (Test-Path -LiteralPath $path) -and $AllowMissing) { return $null }
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
        throw 'Contexto ausente; adoptar con setup-project y perfiles explicitos.'
    }
    $context = Get-Content -LiteralPath $path -Raw -Encoding UTF8 | ConvertFrom-Json -ErrorAction Stop
    if ($context -isnot [PSCustomObject] -or
        (@($context.PSObject.Properties.Name | Sort-Object) -join ',') -ne 'github_mode,profiles,role,schema_version') {
        throw 'Esquema de contexto invalido.'
    }
    if ($context.schema_version -isnot [int] -or $context.schema_version -ne 1 -or
        $context.role -cnotin @('authoring', 'consumer') -or $context.profiles -isnot [array] -or
        $context.profiles.Count -eq 0 -or $context.github_mode -cnotin @('skip', 'codex-plugin', 'gh-cli', 'manual')) {
        throw 'Contexto invalido o schema desconocido; no se modifica.'
    }
    $catalog = Get-HarnessManifest $ManifestPath
    foreach ($name in $context.profiles) {
        if ($name -isnot [string] -or $name -cnotin @($catalog.profiles.PSObject.Properties.Name)) { throw 'Perfil de contexto invalido.' }
    }
    $canonical = @(Resolve-HarnessProfileList -Profiles $context.profiles -ManifestPath $ManifestPath)
    if (($canonical -join ',') -cne ($context.profiles -join ',')) { throw 'Perfiles de contexto no canonicos.' }
    return $context
}

function Set-HarnessContext {
    [CmdletBinding(SupportsShouldProcess)]
    param([string]$RootPath, [string[]]$Profiles, [string]$GitHubMode, [string]$Role = 'consumer')
    if (-not $PSCmdlet.ShouldProcess($RootPath, 'Persistir contexto del harness')) { return }
    $directory = Join-Path $RootPath '.harness'
    New-Item -ItemType Directory -Path $directory -Force -ErrorAction Stop | Out-Null
    $path = Join-Path $directory 'context.json'
    $temp = Join-Path $directory ([guid]::NewGuid().ToString('N') + '.tmp')
    $data = [ordered]@{ schema_version = 1; role = $Role; profiles = @($Profiles); github_mode = $GitHubMode }
    try {
        [IO.File]::WriteAllText($temp, (($data | ConvertTo-Json) + "`n"), [Text.UTF8Encoding]::new($false))
        if (Test-Path -LiteralPath $path) { [IO.File]::Replace($temp, $path, [System.Management.Automation.Language.NullString]::Value) }
        else { [IO.File]::Move($temp, $path) }
    }
    finally { if (Test-Path -LiteralPath $temp) { Remove-Item -LiteralPath $temp -Force } }
}

function Test-HarnessResource {
    param([string]$RootPath = (Get-RepositoryRoot))
    $context = Get-HarnessContext -RootPath $RootPath
    $entries = @(Get-HarnessPlanEntry -Profiles $context.profiles)
    if ($entries.Count -eq 0) { throw 'Suite sin recursos/casos.' }
    $checkedPaths = New-Object 'System.Collections.Generic.HashSet[string]' ([StringComparer]::OrdinalIgnoreCase)
    foreach ($entry in $entries) {
        $relative = if ($context.role -eq 'authoring') { $entry.Source } else { $entry.Target }
        $path = Join-Path $RootPath $relative
        Assert-HarnessPath -Path $path -CheckedDirectories $checkedPaths
        if (-not (Test-Path -LiteralPath $path -PathType Leaf) -or (Get-Item -LiteralPath $path).Length -eq 0) {
            throw "Recurso requerido ausente o vacio: $($entry.Target)"
        }
    }
    return $entries.Count
}
