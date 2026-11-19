#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/common.sh
source "$script_dir/../../scripts/common.sh"
manifest_command check "$(repository_root)/scripts/profiles-manifest.json" '' "$(repository_root)"
