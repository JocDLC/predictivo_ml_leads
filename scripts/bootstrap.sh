#!/usr/bin/env bash
# Setup por maquina (ejecutar en cada computador segun los perfiles deseados).
# Instala solo las dependencias globales de los perfiles seleccionados con versiones fijadas.
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/common.sh
source "$script_dir/common.sh"

profiles_arg=""
dry_run="false"

while [ $# -gt 0 ]; do
    case "$1" in
        --profiles | -p)
            profiles_arg="${2:-}"
            shift 2
            ;;
        --dry-run)
            dry_run="true"
            shift
            ;;
        *)
            echo "Argumento no reconocido en bootstrap.sh: $1" >&2
            echo "Uso: ./scripts/bootstrap.sh [--profiles base,codex,openspec,graphify,archify] [--dry-run]" >&2
            exit 2
            ;;
    esac
done

if [ -z "$profiles_arg" ]; then
    profiles_arg="$(manifest_command selected "$(repository_root)/scripts/profiles-manifest.json" '' "$(repository_root)" | paste -sd, -)"
fi
resolved_profiles="$(resolve_profiles "$profiles_arg")"

assert_command() {
    command -v "$1" >/dev/null 2>&1 || {
        echo "ERROR: '$1' no esta instalado. $2" >&2
        exit 1
    }
}

echo "==> Verificando requisitos del perfil base..."
assert_command git "Instala Git desde https://git-scm.com"

if printf '%s\n' "$resolved_profiles" | grep -Eqx "openspec|archify"; then
    assert_command node "Instala Node.js 20.19+ desde https://nodejs.org"
fi

if printf '%s\n' "$resolved_profiles" | grep -qx "openspec"; then
    OPEN_SPEC_VERSION="$(harness_version @fission-ai/openspec)"
    echo "==> [openspec] Instalando OpenSpec CLI (global) v${OPEN_SPEC_VERSION}..."
    if [ "$dry_run" = "false" ]; then
        npm install -g "@fission-ai/openspec@${OPEN_SPEC_VERSION}"
    fi
else
    echo "==> [openspec] Omitido (perfil no seleccionado)."
fi

if printf '%s\n' "$resolved_profiles" | grep -qx "graphify"; then
    assert_command python3 "Instala Python 3.10+ desde https://python.org"
    GRAPHIFY_VERSION="$(harness_version graphifyy)"
    UV_VERSION="$(harness_version uv)"
    echo "==> [graphify] Instalando Graphify v${GRAPHIFY_VERSION} y uv v${UV_VERSION}..."
    if [ "$dry_run" = "false" ]; then
        python3 -m pip install --user pipx
        python3 -m pipx ensurepath
        python3 -m pipx install --force "graphifyy==${GRAPHIFY_VERSION}"
        python3 -m pipx run --spec "graphifyy==${GRAPHIFY_VERSION}" graphify install
        python3 -m pipx install --force "uv==${UV_VERSION}"

        python3 - <<'EOF'
import json
import os

path = os.path.expanduser("~/.gemini/antigravity/mcp_config.json")
os.makedirs(os.path.dirname(path), exist_ok=True)

config = {}
if os.path.exists(path):
    with open(path) as f:
        config = json.load(f)

servers = config.setdefault("mcpServers", {})
if "graphify" not in servers:
    servers["graphify"] = {
        "command": "uv",
        "args": [
            "run", "--with", "graphifyy", "--with", "mcp",
            "-m", "graphify.serve",
            "${workspace.path}/graphify-out/graph.json",
        ],
    }
    with open(path, "w") as f:
        json.dump(config, f, indent=2)
EOF
    fi
else
    echo "==> [graphify] Omitido (perfil no seleccionado)."
fi

if printf '%s\n' "$resolved_profiles" | grep -qx "archify"; then
    ARCHIFY_VERSION="$(harness_version archify)"
    echo "==> [archify] Instalando skill tt-a1i/archify@${ARCHIFY_VERSION} con ARCHIFY_UPDATE_CHECK_DISABLED=1..."
    if [ "$dry_run" = "false" ]; then
        ARCHIFY_UPDATE_CHECK_DISABLED=1 npx -y skills add "tt-a1i/archify@${ARCHIFY_VERSION}" --skill archify --global --copy --yes
    fi
else
    echo "==> [archify] Omitido (perfil no seleccionado)."
fi

if printf '%s\n' "$resolved_profiles" | grep -qx "codex"; then
    echo "==> [codex] PENDIENTE (accion manual del usuario): instala la extension 'Codex - OpenAI's coding agent' (openai.chatgpt) en tu IDE e inicia sesion."
fi

if [ "$dry_run" = "false" ]; then
    echo ""
    "$script_dir/verify-setup.sh" --profiles "$profiles_arg"
fi
