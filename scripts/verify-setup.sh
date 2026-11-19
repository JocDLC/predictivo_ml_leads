#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/common.sh
source "$script_dir/common.sh"
requested=''
development_tools=false
while [ $# -gt 0 ]; do
    case "$1" in
        --profiles | -p)
            requested="${2:?Falta lista de perfiles}"
            shift 2
            ;;
        --development-tools)
            development_tools=true
            shift
            ;;
        *)
            echo 'Uso: verify-setup.sh [--profiles lista] [--development-tools]' >&2
            exit 2
            ;;
    esac
done
root="$(repository_root)"
manifest="$root/scripts/profiles-manifest.json"
role="$(manifest_command role "$manifest" '' "$root")"
stored="$(manifest_command selected "$manifest" '' "$root")"
active="$stored"
if [ -n "$requested" ]; then
    active="$(resolve_profiles "$requested")"
    if [ "$role" = consumer ]; then
        while IFS= read -r item; do
            grep -qxF "$item" <<<"$stored" || {
                echo 'FAIL perfil inactivo; agregar mediante setup-project' >&2
                exit 1
            }
        done <<<"$active"
    fi
fi
if command -v python3 >/dev/null 2>&1; then pycmd=python3; else pycmd=python; fi
failures=0
check_version() {
    local name=$1 policy=${2:-exact} expected output
    expected="$(harness_version "$name")"
    if output="$("$pycmd" "$script_dir/probe.py" "$name" "$expected" "$policy")"; then
        printf '[PASS] version.%s : %s\n' "$name" "$output"
    else
        printf '[FAIL] version.%s : sonda no verificable o version incompatible\n' "$name" >&2
        failures=$((failures + 1))
    fi
}
if ! manifest_command check "$manifest" '' "$root"; then failures=$((failures + 1)); fi
check_version Git minimum
check_version Bash minimum
check_version Python minimum
if grep -Eqx 'openspec|archify' <<<"$active"; then check_version Node.js minimum; fi
for profile in openspec graphify archify codex; do
    if ! grep -qx "$profile" <<<"$active"; then printf '[SKIP] profile.%s : perfil inactivo [OMITIDO]\n' "$profile"; fi
done
if grep -qx openspec <<<"$active"; then check_version @fission-ai/openspec; fi
if grep -qx graphify <<<"$active"; then
    check_version graphifyy
    check_version uv
    echo '[PENDIENTE] graphify.skill : calificar skill y registro MCP del cliente, sin leer credenciales.'
fi
if grep -qx archify <<<"$active"; then echo '[PENDIENTE] archify.skill : calificar instalacion y version del skill.'; fi
if grep -qx codex <<<"$active"; then echo '[PENDIENTE] codex.login : instalacion e inicio de sesion manuales (openai.chatgpt).'; fi
if [ "$development_tools" = true ]; then
    check_version ShellCheck
    check_version shfmt
    if [ "$role" = authoring ]; then check_version Bats; fi
fi
case "$root" in *OneDrive*) echo '[AVISO] Gitignore no excluye archivos de la sincronizacion OneDrive.' ;; esac
if command -v df >/dev/null 2>&1; then
    free_kb="$(df -Pk "$root" | awk 'NR==2 {print $4}')"
    if [[ "$free_kb" =~ ^[0-9]+$ ]] && [ "$free_kb" -lt 5242880 ]; then echo '[AVISO] Espacio libre menor a 5 GB.'; fi
fi
echo '[PENDIENTE] git.identity : confirmar identidad local antes del primer commit; no se cambia configuracion.'
if [ "$failures" -gt 0 ]; then
    echo "FAIL: $failures requisitos fallidos"
    exit 1
fi
echo 'PASS: requisitos automaticos verificados; revisar PENDIENTE manuales.'
