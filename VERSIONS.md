# Versiones del harness

Este archivo es la fuente de verdad de versiones. Los scripts de verificación leen
directamente las tablas; no se permite `latest`, rangos abiertos ni duplicar pines en
otros archivos sin una prueba que detecte divergencias.

Última revisión de los registros oficiales: 2026-08-26.

## Runtimes mínimos

| Herramienta | Versión | Política |
|-------------|---------|----------|
| Node.js | `20.19.0` | Mínimo requerido por OpenSpec. |
| Python | `3.10.0` | Mínimo para Graphify y utilidades Python. |
| Git | `2.40.0` | Mínimo para worktrees y operaciones del harness. |
| PowerShell | `5.1.0` | Mínimo para los scripts de Windows. |
| Bash | `4.4.0` | Mínimo para los scripts de Linux/macOS. |

## Herramientas del proyecto

| Herramienta | Versión | Instalación reproducible |
|-------------|---------|--------------------------|
| `@fission-ai/openspec` | `1.6.0` | `npm install -g @fission-ai/openspec@1.6.0` |
| `graphifyy` | `0.9.50` | `pipx install graphifyy==0.9.50` |
| `uv` | `0.12.6` | `pipx install uv==0.12.6` |

## Verificación del repositorio

| Herramienta | Versión | Uso |
|-------------|---------|-----|
| `Pester` | `5.9.0` | Tests de PowerShell con compatibilidad para Windows PowerShell 5.1. |
| `PSScriptAnalyzer` | `1.25.0` | Análisis estático de PowerShell. |
| `Bats` | `1.14.0` | Tests de Bash. |
| `ShellCheck` | `0.11.0` | Análisis estático de Bash. |
| `shfmt` | `3.13.1` | Verificación determinística del formato Bash. |

## Integraciones opcionales

| Herramienta | Versión | Runtime | Política |
|-------------|---------|---------|----------|
| `archify` | `3.1.0` | Node.js `>=20.19.0` | Perfil opcional de documentación visual interactiva (`tt-a1i/archify`). Se instala con `ARCHIFY_UPDATE_CHECK_DISABLED=1` y genera artefactos HTML/JSON autocontenidos bajo `docs/architecture/`. |
| **Codex – OpenAI’s coding agent** | `openai.chatgpt` | Editor compatible | Extensión oficial con instalación y login manuales; no se gestiona como paquete npm. |

## Política de actualización

1. Confirmar la versión en el registro oficial.
2. Actualizar este archivo y las pruebas de coherencia.
3. Ejecutar `scripts/test`, `scripts/lint`, `scripts/build` y la validación OpenSpec.
4. Probar el instalador en un directorio temporal antes de aceptar el nuevo pin.
