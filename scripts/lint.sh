#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/common.sh
source "$script_dir/common.sh"

scan_target="${1:-$(repository_root)}"
if ! scan_secrets "$scan_target"; then
    exit 1
fi

if [ $# -ge 1 ]; then
    echo "Escaneo de secretos: sin hallazgos."
    exit 0
fi

# Resolver las directivas source del consumidor desde su propia raiz.
cd "$(repository_root)"

shellcheck_version="$(harness_version ShellCheck)"
shfmt_version="$(harness_version shfmt)"

if ! command -v shellcheck >/dev/null 2>&1; then
    echo "ShellCheck $shellcheck_version no esta instalado." >&2
    exit 1
fi
if ! command -v shfmt >/dev/null 2>&1; then
    echo "shfmt $shfmt_version no esta instalado." >&2
    exit 1
fi

if command -v python3 >/dev/null 2>&1; then pycmd=python3; else pycmd=python; fi
"$pycmd" "$script_dir/probe.py" ShellCheck "$shellcheck_version" exact
"$pycmd" "$script_dir/probe.py" shfmt "$shfmt_version" exact

mapfile -t sources < <(shell_sources)
shellcheck -x "${sources[@]}"
shfmt -d -i 4 -ci "${sources[@]}"
