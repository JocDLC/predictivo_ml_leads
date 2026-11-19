#!/usr/bin/env bash
# Instalador del harness: clona la fuente en un directorio temporal y compone
# solo los perfiles seleccionados en el directorio destino.

set -euo pipefail

TARGET_DIR=""
REPO="${DEV_HARNESS_REPOSITORY:-https://github.com/JocDLC/dev-harness-template.git}"
PROFILES="base"
GITHUB_MODE="skip"
SKIP_BOOTSTRAP="false"

while [ $# -gt 0 ]; do
    case "$1" in
        --profiles | -p)
            PROFILES="${2:-}"
            shift 2
            ;;
        --github-mode | -g)
            GITHUB_MODE="${2:-}"
            shift 2
            ;;
        --repository | -r)
            REPO="${2:-}"
            shift 2
            ;;
        --skip-bootstrap)
            SKIP_BOOTSTRAP="true"
            shift
            ;;
        --*)
            echo "ERROR: Argumento desconocido '$1'." >&2
            echo "Uso: install.sh [directorio-destino] [--profiles base,codex,openspec,graphify,archify] [--github-mode skip|codex-plugin|gh-cli|manual] [--skip-bootstrap]" >&2
            exit 2
            ;;
        *)
            if [ -z "$TARGET_DIR" ]; then
                TARGET_DIR="$1"
                shift
            else
                echo "ERROR: Argumento posicional inesperado '$1'." >&2
                exit 2
            fi
            ;;
    esac
done

TARGET_DIR="${TARGET_DIR:-$HOME/dev-harness-template}"

if [ "${DEV_HARNESS_FAKE_GIT_MISSING:-0}" = "1" ] || ! command -v git >/dev/null 2>&1; then
    echo "ERROR: Git no esta instalado. Instalalo desde https://git-scm.com" >&2
    exit 1
fi

if [ -d "$TARGET_DIR" ]; then
    echo "ERROR: El directorio $TARGET_DIR ya existe. Borralo o usa setup-project.sh para actualizar de forma idempotente." >&2
    exit 1
fi

temp_clone="$(mktemp -d "${TMPDIR:-/tmp}/dev-harness-src.XXXXXX")"
cleanup() {
    rm -rf "$temp_clone"
}
trap cleanup EXIT

echo "==> Obteniendo plantilla fuente en copia temporal..."
git clone --quiet "$REPO" "$temp_clone"

source_for_compose="$temp_clone"
if [ -d "$REPO" ] && [ -f "$REPO/scripts/profiles-manifest.json" ]; then
    source_for_compose="$REPO"
fi

if [ -f "$source_for_compose/scripts/profiles-manifest.json" ]; then
    # shellcheck source=scripts/common.sh
    source "$source_for_compose/scripts/common.sh"
    compose_project "$source_for_compose" "$TARGET_DIR" "$PROFILES" "$GITHUB_MODE" "false" "$source_for_compose/scripts/profiles-manifest.json"

    if [ "$SKIP_BOOTSTRAP" = "false" ] && [ -x "$TARGET_DIR/scripts/bootstrap.sh" ]; then
        echo "==> Corriendo bootstrap para los perfiles seleccionados..."
        "$TARGET_DIR/scripts/bootstrap.sh" --profiles "$PROFILES"
    fi
else
    case "$PROFILES:$GITHUB_MODE" in
        base:skip | base:codex-plugin | base:gh-cli | base:manual) ;;
        *)
            echo 'FAIL fuente legacy sin catalogo: solo base y modos GitHub validos; destino sin crear.' >&2
            exit 1
            ;;
    esac
    cp -R "$temp_clone" "$TARGET_DIR"
    if [ "$SKIP_BOOTSTRAP" = "false" ] && [ -x "$TARGET_DIR/scripts/bootstrap.sh" ]; then
        echo "==> Corriendo bootstrap..."
        "$TARGET_DIR/scripts/bootstrap.sh"
    fi
    if [ -x "$TARGET_DIR/scripts/setup-project.sh" ]; then
        echo "==> Corriendo setup-project..."
        "$TARGET_DIR/scripts/setup-project.sh"
    fi
fi

echo "Listo. Entra en $TARGET_DIR y abrelo en tu IDE."
