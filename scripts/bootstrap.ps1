# Setup por maquina (ejecutar en cada computador segun los perfiles deseados).
# Instala solo las dependencias globales de los perfiles seleccionados con versiones fijadas.

param(
    [string[]]$Profiles = @(),
    [switch]$DryRun
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

. (Join-Path $PSScriptRoot 'common.ps1')

function Assert-Command {
    param([string]$Name, [string]$InstallHint)
    if (-not (Get-Command $Name -ErrorAction SilentlyContinue)) {
        throw "'$Name' no esta instalado. $InstallHint"
    }
}

function Get-PythonCommand {
    foreach ($candidate in @('python', 'py')) {
        if (Get-Command $candidate -ErrorAction SilentlyContinue) {
            return $candidate
        }
    }
    throw 'Python no esta instalado. Instala Python 3.10+ desde https://python.org'
}

if ($Profiles.Count -eq 0) { $Profiles = @((Get-HarnessContext).profiles) }
$resolvedProfiles = @(Resolve-HarnessProfileList -Profiles $Profiles)

Write-Host '==> Verificando requisitos del perfil base...' -ForegroundColor Cyan
Assert-Command 'git' 'Instala Git desde https://git-scm.com'

if (($resolvedProfiles -contains 'openspec') -or ($resolvedProfiles -contains 'archify')) {
    Assert-Command 'node' 'Instala Node.js 20.19+ desde https://nodejs.org'
}

if ($resolvedProfiles -contains 'openspec') {
    $openSpecVersion = Get-HarnessVersion -Name '@fission-ai/openspec'
    Write-Host "==> [openspec] Instalando OpenSpec CLI (global) v$openSpecVersion..." -ForegroundColor Cyan
    if (-not $DryRun) {
        & npm install -g "@fission-ai/openspec@$openSpecVersion"
        if ($LASTEXITCODE -ne 0) { throw 'Instalacion de OpenSpec fallo.' }
    }
}
else {
    Write-Host '==> [openspec] Omitido (perfil no seleccionado).' -ForegroundColor DarkGray
}

if ($resolvedProfiles -contains 'graphify') {
    $python = Get-PythonCommand
    $graphifyVersion = Get-HarnessVersion -Name 'graphifyy'
    $uvVersion = Get-HarnessVersion -Name 'uv'

    Write-Host "==> [graphify] Instalando Graphify v$graphifyVersion y uv v$uvVersion (via pipx)..." -ForegroundColor Cyan
    if (-not $DryRun) {
        & $python -m pip install --user pipx
        if ($LASTEXITCODE -ne 0) { throw 'Instalacion de pipx fallo.' }
        & $python -m pipx ensurepath
        if ($LASTEXITCODE -ne 0) { throw 'pipx ensurepath fallo.' }
        & $python -m pipx install --force "graphifyy==$graphifyVersion"
        if ($LASTEXITCODE -ne 0) { throw 'Instalacion de Graphify fallo.' }
        & $python -m pipx run --spec "graphifyy==$graphifyVersion" graphify install
        if ($LASTEXITCODE -ne 0) { throw 'Instalacion de skill Graphify fallo.' }
        & $python -m pipx install --force "uv==$uvVersion"
        if ($LASTEXITCODE -ne 0) { throw 'Instalacion de uv fallo.' }

        $mcpDir = Join-Path $env:USERPROFILE '.gemini\antigravity'
        $mcpFile = Join-Path $mcpDir 'mcp_config.json'
        New-Item -ItemType Directory -Force -Path $mcpDir | Out-Null

        if (Test-Path $mcpFile) {
            $config = Get-Content $mcpFile -Raw | ConvertFrom-Json
        }
        else {
            $config = [PSCustomObject]@{}
        }
        if (-not $config.PSObject.Properties['mcpServers']) {
            $config | Add-Member -NotePropertyName 'mcpServers' -NotePropertyValue ([PSCustomObject]@{})
        }

        if (-not $config.mcpServers.PSObject.Properties['graphify']) {
            $graphifyServer = [PSCustomObject]@{
                command = 'uv'
                args    = @('run', '--with', 'graphifyy', '--with', 'mcp', '-m', 'graphify.serve', '${workspace.path}/graphify-out/graph.json')
            }
            $config.mcpServers | Add-Member -NotePropertyName 'graphify' -NotePropertyValue $graphifyServer
            $config | ConvertTo-Json -Depth 10 | Set-Content $mcpFile -Encoding UTF8
            Write-Host "Servidor MCP 'graphify' agregado a $mcpFile" -ForegroundColor Green
        }
    }
}
else {
    Write-Host '==> [graphify] Omitido (perfil no seleccionado).' -ForegroundColor DarkGray
}

if ($resolvedProfiles -contains 'archify') {
    $archifyVersion = Get-HarnessVersion -Name 'archify'
    Write-Host "==> [archify] Instalando skill tt-a1i/archify@$archifyVersion con ARCHIFY_UPDATE_CHECK_DISABLED=1..." -ForegroundColor Cyan
    if (-not $DryRun) {
        $prevCheck = $env:ARCHIFY_UPDATE_CHECK_DISABLED
        try {
            $env:ARCHIFY_UPDATE_CHECK_DISABLED = '1'
            & npx -y skills add "tt-a1i/archify@$archifyVersion" --skill archify --global --copy --yes
            if ($LASTEXITCODE -ne 0) { throw 'Instalacion de Archify fallo.' }
        }
        finally {
            $env:ARCHIFY_UPDATE_CHECK_DISABLED = $prevCheck
        }
    }
}
else {
    Write-Host '==> [archify] Omitido (perfil no seleccionado).' -ForegroundColor DarkGray
}

if ($resolvedProfiles -contains 'codex') {
    Write-Host "==> [codex] PENDIENTE (accion manual del usuario): instala la extension 'Codex - OpenAI''s coding agent' (openai.chatgpt) en tu IDE e inicia sesion." -ForegroundColor Yellow
}

if (-not $DryRun) {
    Write-Host ''
    Write-Host '==> Verificacion final del harness...' -ForegroundColor Cyan
    & (Join-Path $PSScriptRoot 'verify-setup.ps1') -Profiles $resolvedProfiles
}
