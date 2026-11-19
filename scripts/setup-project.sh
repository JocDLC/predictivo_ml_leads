#!/usr/bin/env bash
# Setup por proyecto (ejecutar una vez en cada repositorio).
# Compone los perfiles seleccionados e inicializa las herramientas correspondientes.
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/common.sh
source "$script_dir/common.sh"

profiles_arg="base"
target_dir="$(repository_root)"
source_dir="$(repository_root)"
github_mode="skip"
update_existing="false"
skip_external_init="false"

while [ $# -gt 0 ]; do
    case "$1" in
        --profiles | -p)
            profiles_arg="${2:-}"
            shift 2
            ;;
        --target-dir | -t)
            target_dir="${2:-}"
            shift 2
            ;;
        --source-dir | -s)
            source_dir="${2:-}"
            shift 2
            ;;
        --github-mode | -g)
            github_mode="${2:-}"
            shift 2
            ;;
        --update)
            update_existing="true"
            shift
            ;;
        --skip-external-init)
            skip_external_init="true"
            shift
            ;;
        *)
            echo "Argumento no reconocido en setup-project.sh: $1" >&2
            echo "Migracion: usa --profiles <lista> [--target-dir <dir>] [--github-mode skip|codex-plugin|gh-cli|manual] [--update]" >&2
            exit 2
            ;;
    esac
done

compose_project "$source_dir" "$target_dir" "$profiles_arg" "$github_mode" "$update_existing"

if [ "$skip_external_init" = "false" ]; then
    resolved_profiles="$(resolve_profiles "$profiles_arg")"
    if printf '%s\n' "$resolved_profiles" | grep -qx "openspec"; then
        if command -v openspec >/dev/null 2>&1; then
            echo "==> Inicializando OpenSpec para el proyecto..."
            openspec init "$target_dir" --tools antigravity,windsurf,codex
        elif [ -x "$HOME/.npm-global/bin/openspec" ]; then
            echo "==> Inicializando OpenSpec para el proyecto..."
            "$HOME/.npm-global/bin/openspec" init "$target_dir" --tools antigravity,windsurf,codex
        else
            echo "[PENDIENTE] OpenSpec CLI no detectado localmente; corre ./scripts/bootstrap.sh --profiles openspec."
        fi
    fi

    if printf '%s\n' "$resolved_profiles" | grep -qx "graphify"; then
        if command -v graphify >/dev/null 2>&1; then
            echo "==> Construyendo el grafo inicial con Graphify..."
            (cd "$target_dir" && graphify update .)
        else
            echo "[PENDIENTE] Graphify CLI no detectado localmente; corre ./scripts/bootstrap.sh --profiles graphify."
        fi
    fi
fi

echo ""
echo "Listo. Los archivos compuestos en el destino pueden commitearse al repositorio."
