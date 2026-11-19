#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/common.sh
source "$script_dir/common.sh"
manifest_command check "$(repository_root)/scripts/profiles-manifest.json" '' "$(repository_root)"
selected="$(manifest_command selected "$(repository_root)/scripts/profiles-manifest.json" '' "$(repository_root)")"

while IFS= read -r source_file; do
    bash -n "$source_file"
done < <(shell_scripts)

cd "$(repository_root)"
if grep -qx openspec <<<"$selected"; then
    openspec validate --all
else
    echo '[SKIP] openspec: perfil inactivo.'
fi
echo 'Build del harness validado.'
