Set-StrictMode -Version Latest

function Get-RepositoryRoot {
    return Split-Path -Parent $PSScriptRoot
}

function Get-HarnessVersion {
    param([Parameter(Mandatory)][string]$Name)

    $versionsFile = Join-Path (Get-RepositoryRoot) 'VERSIONS.md'
    $escapedName = [regex]::Escape($Name)
    $pattern = "^\|\s*``?$escapedName``?\s*\|\s*``([^``]+)``\s*\|"
    $match = Select-String -Path $versionsFile -Pattern $pattern | Select-Object -First 1

    if (-not $match) {
        throw "No se encontro una version fijada para '$Name' en $versionsFile"
    }

    return $match.Matches[0].Groups[1].Value
}

function Get-PowerShellSourceFile {
    return Get-ChildItem -Path (Get-RepositoryRoot) -Recurse -File |
        Where-Object {
            $_.Extension -in @('.ps1', '.psm1', '.psd1') -and
            $_.FullName -notmatch '[\\/]graphify-out[\\/]'
        }
}

function Find-HarnessSecretLeak {
    param([string]$RootPath = (Get-RepositoryRoot))

    if (-not (Test-Path -LiteralPath $RootPath -PathType Container)) {
        throw 'Escaneo FAIL: la raiz no es un directorio existente.'
    }
    $patterns = @(
        @{ Name = 'GitHub Token'; Regex = '\b(ghp_[A-Za-z0-9]{36}|github_pat_[A-Za-z0-9_]{40,})\b' },
        @{ Name = 'OpenAI/API Secret Key'; Regex = '\bsk-(?:proj-)?[A-Za-z0-9_-]{20,}\b' },
        @{ Name = 'AWS Access Key'; Regex = '\bAKIA[0-9A-Z]{16}\b' },
        @{ Name = 'Private Key Block'; Regex = '-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----' }
    )

    $excludedPattern = '[\\/](?:\.git|node_modules|\.venv|dist|build|graphify-out)[\\/]'
    $files = Get-ChildItem -LiteralPath $RootPath -Recurse -File -Force -ErrorAction Stop |
        Where-Object { $_.FullName -notmatch $excludedPattern }

    $findings = @()
    foreach ($file in $files) {
        foreach ($rule in $patterns) {
            $matchesInFile = Select-String -LiteralPath $file.FullName -Pattern $rule.Regex -AllMatches -ErrorAction Stop
            foreach ($m in $matchesInFile) {
                $findings += [PSCustomObject]@{
                    Rule       = $rule.Name
                    File       = $file.FullName
                    LineNumber = $m.LineNumber
                }
            }
        }
    }

    return $findings
}

function Test-SafeRelativePath {
    param([Parameter(Mandatory)][string]$RelativePath)

    if ([string]::IsNullOrWhiteSpace($RelativePath)) {
        return $false
    }
    if ([System.IO.Path]::IsPathRooted($RelativePath)) {
        return $false
    }
    if ($RelativePath -match '[:\x00-\x1f\x7f*?<>|"]' -or $RelativePath.StartsWith('/') -or $RelativePath.StartsWith('\')) {
        return $false
    }
    $segments = $RelativePath -split '[\\/]'
    foreach ($segment in $segments) {
        if ($segment -in @('.', '..', '.git') -or $segment.EndsWith('.') -or $segment.EndsWith(' ') -or [string]::IsNullOrWhiteSpace($segment)) {
            return $false
        }
    }
    return $true
}

function Get-HarnessManifest {
    param([string]$ManifestPath = (Join-Path $PSScriptRoot 'profiles-manifest.json'))

    if (-not (Test-Path $ManifestPath)) {
        throw "No se encontro el manifiesto de perfiles en $ManifestPath"
    }

    $manifest = Get-Content -LiteralPath $ManifestPath -Raw -Encoding UTF8 | ConvertFrom-Json -ErrorAction Stop
    if (-not $manifest -or -not $manifest.PSObject.Properties['profiles'] -or
        $manifest.profiles -isnot [PSCustomObject] -or -not $manifest.profiles.PSObject.Properties['base']) {
        throw "El manifiesto $ManifestPath no contiene el objeto 'profiles'."
    }

    foreach ($prop in $manifest.profiles.PSObject.Properties) {
        if ($prop.Name -notmatch '^[a-z][a-z0-9-]*$' -or $prop.Name -eq 'all' -or $prop.Value -isnot [array]) {
            throw "Esquema invalido en el perfil '$($prop.Name)'."
        }
        foreach ($entry in $prop.Value) {
            $null = Get-HarnessEntry $entry
        }
    }

    return $manifest
}

function Resolve-HarnessProfileList {
    param(
        [string[]]$Profiles = @('base'),
        [string]$ManifestPath = (Join-Path $PSScriptRoot 'profiles-manifest.json')
    )

    $manifest = Get-HarnessManifest -ManifestPath $ManifestPath
    $validNames = @($manifest.profiles.PSObject.Properties.Name)

    $requested = @()
    foreach ($item in $Profiles) {
        if (-not [string]::IsNullOrWhiteSpace($item)) {
            foreach ($token in ($item -split ',')) {
                $trimmed = $token.Trim().ToLowerInvariant()
                if (-not [string]::IsNullOrWhiteSpace($trimmed)) {
                    $requested += $trimmed
                }
            }
        }
    }

    if ($requested.Count -eq 0) {
        $requested = @('base')
    }

    foreach ($name in $requested) {
        if ($name -ne 'all' -and $name -notin $validNames) {
            $allowedList = $validNames -join ', '
            throw "Perfil desconocido '$name'. Perfiles validos: $allowedList (o 'all')."
        }
    }

    if ($requested -contains 'all') {
        return @('base') + @($validNames | Where-Object { $_ -ne 'base' })
    }

    $resolved = @('base')
    foreach ($name in $validNames) {
        if ($name -in $requested -and $name -notin $resolved) {
            $resolved += $name
        }
    }

    return $resolved
}

function Resolve-HarnessGitHubMode {
    param([string]$GitHubMode = 'skip')

    $normalized = if ([string]::IsNullOrWhiteSpace($GitHubMode)) { 'skip' } else { $GitHubMode.Trim().ToLowerInvariant() }
    $validModes = @('skip', 'codex-plugin', 'gh-cli', 'manual')
    if ($normalized -notin $validModes) {
        throw "Modo GitHub desconocido '$GitHubMode'. Modos validos: $($validModes -join ', ')."
    }
    return $normalized
}

function Get-HarnessManifestFileList {
    param(
        [string[]]$Profiles = @('base'),
        [string]$ManifestPath = (Join-Path $PSScriptRoot 'profiles-manifest.json')
    )

    return @(Get-HarnessPlanEntry -Profiles $Profiles -ManifestPath $ManifestPath | ForEach-Object { $_.Target })
}

function Invoke-HarnessGitReadiness {
    param(
        [Parameter(Mandatory)][string]$TargetDir,
        [string]$GitHubMode = 'skip'
    )

    $mode = Resolve-HarnessGitHubMode -GitHubMode $GitHubMode
    $gitMissing = ($env:DEV_HARNESS_FAKE_GIT_MISSING -eq '1') -or (-not (Get-Command git -ErrorAction SilentlyContinue))
    if ($gitMissing) {
        throw 'Git no esta disponible. Instala Git 2.40+ desde https://git-scm.com antes de continuar.'
    }

    $gitDir = Join-Path $TargetDir '.git'
    $repoStatus = 'existente'
    $ancestor = [System.IO.Path]::GetFullPath($TargetDir)
    while ($ancestor -and -not (Test-Path -LiteralPath (Join-Path $ancestor '.git'))) {
        $ancestor = Split-Path -Parent $ancestor
    }
    if (-not $ancestor -and -not (Test-Path $gitDir)) {
        & git init --quiet $TargetDir
        if ($LASTEXITCODE -ne 0) {
            throw "No se pudo inicializar Git en $TargetDir"
        }
        $repoStatus = 'inicializado'
    }

    $userName = (& git -C $TargetDir config user.name 2>$null | Out-String).Trim()
    $userEmail = (& git -C $TargetDir config user.email 2>$null | Out-String).Trim()
    $identityReady = (-not [string]::IsNullOrWhiteSpace($userName)) -and (-not [string]::IsNullOrWhiteSpace($userEmail))

    $pendingActions = @()
    if (-not $identityReady) {
        $pendingActions += "Configurar identidad Git local: git -C `"$TargetDir`" config user.name `"Tu Nombre`" && git -C `"$TargetDir`" config user.email `"tu@correo.com`""
    }

    switch ($mode) {
        'codex-plugin' {
            $pendingActions += 'GitHub (codex-plugin): requerir accion manual del usuario para instalar/conectar el plugin GitHub desde /plugins e iniciar sesion.'
        }
        'gh-cli' {
            $pendingActions += 'GitHub (gh-cli): requerir accion manual del usuario para ejecutar gh auth login y vincular el repositorio.'
        }
        'manual' {
            $pendingActions += "GitHub (manual): requerir accion manual del usuario para agregar el remoto con git -C `"$TargetDir`" remote add origin <url-del-repositorio>."
        }
    }

    return [PSCustomObject]@{
        RepositoryStatus = $repoStatus
        IdentityReady    = $identityReady
        GitHubMode       = $mode
        PendingActions   = $pendingActions
    }
}

function Invoke-HarnessCompose {
    param(
        [Parameter(Mandatory)][string]$TargetDir,
        [string]$SourceDir = (Get-RepositoryRoot),
        [string[]]$Profiles = @('base'),
        [string]$GitHubMode = 'skip',
        [switch]$UpdateExisting,
        [string]$ManifestPath = (Join-Path $PSScriptRoot 'profiles-manifest.json')
    )

    if ([string]::IsNullOrWhiteSpace($TargetDir)) {
        throw 'TargetDir no puede estar vacio.'
    }

    $resolvedProfiles = @(Resolve-HarnessProfileList -Profiles $Profiles -ManifestPath $ManifestPath)
    $resolvedGitHubMode = Resolve-HarnessGitHubMode -GitHubMode $GitHubMode
    $requestedProfiles = @($resolvedProfiles)
    $previousContext = Get-HarnessContext -RootPath $TargetDir -AllowMissing -ManifestPath $ManifestPath
    if ($previousContext) {
        $resolvedProfiles = @(Resolve-HarnessProfileList -Profiles @($resolvedProfiles + $previousContext.profiles) -ManifestPath $ManifestPath)
    }
    $filesToCopy = @(Get-HarnessPlanEntry -Profiles $resolvedProfiles -ManifestPath $ManifestPath)

    if (($env:DEV_HARNESS_FAKE_GIT_MISSING -eq '1') -or (-not (Get-Command git -ErrorAction SilentlyContinue))) {
        throw 'Git no esta disponible. Instala Git 2.40+ desde https://git-scm.com antes de continuar.'
    }

    $targetFull = [System.IO.Path]::GetFullPath($TargetDir)
    $sourceFull = [System.IO.Path]::GetFullPath($SourceDir)
    $checkedDirectories = New-Object 'System.Collections.Generic.HashSet[string]' ([System.StringComparer]::OrdinalIgnoreCase)
    Assert-HarnessPath -Path $targetFull -CheckedDirectories $checkedDirectories
    if ((Test-Path -LiteralPath $targetFull) -and -not (Test-Path -LiteralPath $targetFull -PathType Container)) { throw 'El destino no es un directorio.' }
    foreach ($entry in $filesToCopy) {
        $relPath = $entry.Target
        $sourceFile = Join-Path $sourceFull $entry.Source
        $destFile = Join-Path $targetFull $entry.Target
        Assert-HarnessPath -Path $sourceFile -CheckedDirectories $checkedDirectories
        Assert-HarnessPath -Path $destFile -CheckedDirectories $checkedDirectories
        if (-not (Test-Path -LiteralPath $sourceFile -PathType Leaf)) {
            throw "Fuente requerida ausente o no es un archivo: $relPath"
        }
        if (Test-Path -LiteralPath $destFile -PathType Container) {
            throw "El destino de archivo es un directorio: $relPath"
        }
    }
    New-Item -ItemType Directory -Path $targetFull -Force -ErrorAction Stop | Out-Null

    $created = @()
    $unchanged = @()
    $preserved = @()
    $updated = @()

    try {
        foreach ($entry in $filesToCopy) {
            $relPath = $entry.Target
            if (-not (Test-SafeRelativePath -RelativePath $relPath)) {
                throw "Ruta insegura detectada: $relPath"
            }

            $sourceFile = Join-Path $sourceFull $entry.Source

            $destFile = [System.IO.Path]::GetFullPath((Join-Path $targetFull ($relPath -replace '/', [System.IO.Path]::DirectorySeparatorChar)))
            if (-not $destFile.StartsWith($targetFull.TrimEnd('\', '/') + [IO.Path]::DirectorySeparatorChar, [System.StringComparison]::OrdinalIgnoreCase)) {
                throw "La ruta de destino '$destFile' escapa de '$targetFull'."
            }

            $destParent = Split-Path -Parent $destFile
            if (-not (Test-Path $destParent)) {
                New-Item -ItemType Directory -Path $destParent -Force | Out-Null
            }

            if (-not (Test-Path -LiteralPath $destFile)) {
                Copy-Item -LiteralPath $sourceFile -Destination $destFile -Force -ErrorAction Stop
                $created += $relPath
            }
            else {
                $sourceHash = (Get-FileHash -LiteralPath $sourceFile -Algorithm SHA256).Hash
                $destHash = (Get-FileHash -LiteralPath $destFile -Algorithm SHA256).Hash
                if ($sourceHash -eq $destHash) {
                    $unchanged += $relPath
                }
                elseif ($UpdateExisting) {
                    Copy-Item -LiteralPath $sourceFile -Destination $destFile -Force -ErrorAction Stop
                    $updated += $relPath
                }
                else {
                    $preserved += $relPath
                }
            }
        }

        $gitSummary = Invoke-HarnessGitReadiness -TargetDir $targetFull -GitHubMode $resolvedGitHubMode
        $role = if ($previousContext) { $previousContext.role } else { 'consumer' }
        Set-HarnessContext -RootPath $targetFull -Profiles $resolvedProfiles -GitHubMode $resolvedGitHubMode -Role $role
    }
    catch { throw "FAIL aplicacion: destino puede estar parcial; comprobar contexto. $($_.Exception.Message)" }
    $pendingActions = @($gitSummary.PendingActions)

    if ($resolvedProfiles -contains 'codex') {
        $pendingActions += "Codex (pendiente - requiere accion del usuario): instalar la extension oficial 'Codex - OpenAI''s coding agent' (openai.chatgpt), iniciar sesion y recargar el editor."
    }

    Write-Host "=== Resumen de composicion del harness ===" -ForegroundColor Cyan
    Write-Host ("Perfiles aplicados: {0}" -f ($resolvedProfiles -join ', '))
    Write-Host ("Perfiles solicitados: {0}" -f ($requestedProfiles -join ', '))
    Write-Host ("Archivos creados: {0}" -f $created.Count)
    Write-Host ("Archivos ya presentes (sin cambios): {0}" -f $unchanged.Count)
    Write-Host ("Archivos preservados (modificados por el usuario): {0}" -f $preserved.Count)
    Write-Host ("Archivos actualizados: {0}" -f $updated.Count)
    Write-Host ("Repositorio Git: {0} (identidad: {1})" -f $gitSummary.RepositoryStatus, $(if ($gitSummary.IdentityReady) { 'verificado' } else { 'pendiente' }))
    Write-Host ("Modo GitHub: {0}" -f $resolvedGitHubMode)

    if ($pendingActions.Count -gt 0) {
        Write-Host "Acciones manuales pendientes:" -ForegroundColor Yellow
        foreach ($action in $pendingActions) {
            Write-Host ("  - [PENDIENTE] {0}" -f $action) -ForegroundColor Yellow
        }
    }

    return [PSCustomObject]@{
        Profiles          = $resolvedProfiles
        RequestedProfiles = $requestedProfiles
        Created           = $created
        Unchanged         = $unchanged
        Preserved         = $preserved
        Updated           = $updated
        GitSummary        = $gitSummary
        PendingActions    = $pendingActions
    }
}

function Assert-HarnessPath {
    param(
        [Parameter(Mandatory)][string]$Path,
        [System.Collections.Generic.HashSet[string]]$CheckedDirectories = (New-Object 'System.Collections.Generic.HashSet[string]')
    )

    $current = [System.IO.Path]::GetFullPath($Path)
    $leaf = $true
    while ($current) {
        if ($CheckedDirectories.Contains($current)) { break }
        if (Test-Path -LiteralPath $current) {
            $item = Get-Item -LiteralPath $current -Force -ErrorAction Stop
            if ($item.PSObject.Properties['LinkType'] -and $item.LinkType) {
                throw "Enlace no permitido en ruta administrada: $current"
            }
            if (-not $leaf -and -not $item.PSIsContainer) {
                throw "Un archivo bloquea el directorio requerido: $current"
            }
            if ($item.PSIsContainer) { $null = $CheckedDirectories.Add($current) }
        }
        $leaf = $false
        $current = Split-Path -Parent $current
    }
}

. (Join-Path $PSScriptRoot 'context.ps1')
