#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/common.sh
source "$script_dir/common.sh"

mapfile -t sources < <(shell_sources)
shfmt -w -i 4 -ci "${sources[@]}"
