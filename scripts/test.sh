#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/common.sh
source "$script_dir/common.sh"

context_role="$(manifest_command role "$(repository_root)/scripts/profiles-manifest.json" '' "$(repository_root)")"
if [ "$context_role" = consumer ]; then
    manifest_command check "$(repository_root)/scripts/profiles-manifest.json" '' "$(repository_root)"
    smoke_output="$(bash "$(repository_root)/tests/harness/smoke.sh")"
    grep -Eq '^PASS consumer.resources: [1-9][0-9]* casos;' <<<"$smoke_output" || {
        echo 'FAIL smoke sin casos' >&2
        exit 1
    }
    printf '%s\n' "$smoke_output"
    exit 0
fi

required_version="$(harness_version Bats)"
if ! command -v bats >/dev/null 2>&1; then
    echo "Bats $required_version no esta instalado." >&2
    exit 1
fi

actual_version="$(bats --version | awk '{print $2}')"
if [ "$actual_version" != "$required_version" ]; then
    echo "Se requiere Bats $required_version; se encontro $actual_version." >&2
    exit 1
fi

suite="$(repository_root)/tests/bash"
[ -d "$suite" ] || {
    echo 'FAIL suite ausente' >&2
    exit 1
}
count="$(bats --count "$suite")"
[[ "$count" =~ ^[0-9]+$ ]] && [ "$count" -gt 0 ] || {
    echo 'FAIL suite sin casos' >&2
    exit 1
}
bats "$suite"
