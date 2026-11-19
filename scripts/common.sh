#!/usr/bin/env bash

repository_root() {
    cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd
}

harness_version() {
    local tool=$1
    local versions_file
    local version
    versions_file="$(repository_root)/VERSIONS.md"
    version="$(awk -F '|' -v tool="$tool" '
        {
            name=$2
            gsub(/^[[:space:]`]+|[[:space:]`]+$/, "", name)
            if (name == tool) {
                value=$3
                gsub(/^[[:space:]`]+|[[:space:]`]+$/, "", value)
                print value
                exit
            }
        }
    ' "$versions_file")"

    if [ -z "$version" ]; then
        echo "No se encontro una version fijada para '$tool' en $versions_file" >&2
        return 1
    fi
    printf '%s\n' "$version"
}

shell_sources() {
    find "$(repository_root)/scripts" "$(repository_root)/tests" \
        -type f \( -name '*.sh' -o -name '*.bats' \) 2>/dev/null | sort
}

shell_scripts() {
    find "$(repository_root)/scripts" \
        -type f -name '*.sh' 2>/dev/null | sort
}

scan_secrets() {
    local root_path="${1:-$(repository_root)}"
    if [ ! -d "$root_path" ]; then
        echo "Escaneo FAIL: la raiz no es un directorio existente." >&2
        return 1
    fi
    local pattern='(ghp_[A-Za-z0-9]{36}|github_pat_[A-Za-z0-9_]{40,}|sk-(proj-)?[A-Za-z0-9_-]{20,}|AKIA[0-9A-Z]{16}|-----BEGIN (RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----)'
    local matches scan_status=0

    matches="$(grep -RInE \
        --exclude-dir=.git \
        --exclude-dir=node_modules \
        --exclude-dir=.venv \
        --exclude-dir=dist \
        --exclude-dir=build \
        --exclude-dir=graphify-out \
        "$pattern" "$root_path" 2>/dev/null)" || scan_status=$?

    if [ "$scan_status" -gt 1 ]; then
        echo "Escaneo FAIL: no se pudieron leer todos los archivos." >&2
        return 1
    fi

    if [ -n "$matches" ]; then
        echo "$matches" | awk -F: '{print "Secreto detectado en " $1 ":" $2}' >&2
        return 1
    fi
    return 0
}

validate_safe_relpath() {
    local rel="$1"
    if [ -z "$rel" ]; then
        return 1
    fi
    case "$rel" in
        /* | \\* | ?:/* | ?:\\* | *../* | */..* | ..)
            return 1
            ;;
    esac
    return 0
}

resolve_profiles() {
    local input="${1:-base}"
    manifest_command profiles "${2:-$(repository_root)/scripts/profiles-manifest.json}" "$input"
}

resolve_github_mode() {
    local mode="${1:-skip}"
    mode="$(printf '%s' "$mode" | tr '[:upper:]' '[:lower:]' | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')"
    [ -z "$mode" ] && mode="skip"
    case "$mode" in
        skip | codex-plugin | gh-cli | manual)
            printf '%s\n' "$mode"
            ;;
        *)
            echo "Modo GitHub desconocido '$mode'. Modos validos: skip, codex-plugin, gh-cli, manual." >&2
            return 1
            ;;
    esac
}

manifest_files_for_profiles() {
    local manifest_path="$1"
    shift
    local requested
    requested="$(
        IFS=,
        printf '%s' "$*"
    )"
    manifest_command files "$manifest_path" "$requested"
}

manifest_command() {
    local py_cmd output
    if command -v python3 >/dev/null 2>&1; then
        py_cmd=python3
    elif command -v python >/dev/null 2>&1; then
        py_cmd=python
    else
        echo "Python 3.10+ es requerido para procesar el manifiesto JSON." >&2
        return 1
    fi
    output="$("$py_cmd" "$(repository_root)/scripts/manifest.py" "$@")" || return 1
    printf '%s\n' "$output" | sed 's/\r$//'
}

compose_project() {
    local source_dir="$1"
    local target_dir="$2"
    local profiles_arg="${3:-base}"
    local github_mode_arg="${4:-skip}"
    local update_existing="${5:-false}"
    local manifest_path="${6:-$source_dir/scripts/profiles-manifest.json}"

    local selected_profiles=()
    local line profile_output file_output
    profile_output="$(manifest_command active "$manifest_path" "$profiles_arg" "$target_dir")" || return 1
    while IFS= read -r line; do
        [ -n "$line" ] && selected_profiles+=("$line")
    done <<<"$profile_output"

    local github_mode
    github_mode="$(resolve_github_mode "$github_mode_arg")" || return 1

    local files_to_copy=()
    file_output="$(manifest_command plan "$manifest_path" "$profiles_arg" "$source_dir" "$target_dir")" || return 1
    while IFS= read -r line; do
        [ -n "$line" ] && files_to_copy+=("$line")
    done <<<"$file_output"

    if [ "${DEV_HARNESS_FAKE_GIT_MISSING:-0}" = "1" ] || ! command -v git >/dev/null 2>&1; then
        echo "Git no esta disponible. Instala Git 2.40+ desde https://git-scm.com antes de continuar." >&2
        return 1
    fi

    mkdir -p "$target_dir" || return 1

    local created_count=0
    local unchanged_count=0
    local preserved_count=0
    local updated_count=0
    local rel_path source_path

    for line in "${files_to_copy[@]}"; do
        IFS=$'\t' read -r source_path rel_path <<<"$line"
        validate_safe_relpath "$rel_path" || {
            echo "Ruta insegura detectada: $rel_path" >&2
            return 1
        }
        local src_file="$source_dir/$source_path"
        local dst_file="$target_dir/$rel_path"
        mkdir -p "$(dirname "$dst_file")" || {
            echo 'FAIL copia: destino posiblemente parcial; contexto no actualizado.' >&2
            return 1
        }

        if [ ! -f "$dst_file" ]; then
            cp "$src_file" "$dst_file" || {
                echo 'FAIL copia: destino posiblemente parcial; contexto no actualizado.' >&2
                return 1
            }
            created_count=$((created_count + 1))
        elif cmp -s "$src_file" "$dst_file"; then
            unchanged_count=$((unchanged_count + 1))
        elif [ "$update_existing" = "true" ]; then
            cp "$src_file" "$dst_file" || {
                echo 'FAIL copia: destino posiblemente parcial; contexto no actualizado.' >&2
                return 1
            }
            updated_count=$((updated_count + 1))
        else
            preserved_count=$((preserved_count + 1))
        fi
    done

    local repo_status="existente"
    if ! git -C "$target_dir" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
        git init --quiet "$target_dir" || return 1
        repo_status="inicializado"
    fi

    manifest_command write-context "$manifest_path" "$profiles_arg" "$target_dir" "$github_mode" || {
        echo 'FAIL contexto: copia aplicada pero metadata incompleta.' >&2
        return 1
    }

    local user_name user_email identity_status="verificado"
    user_name="$(git -C "$target_dir" config user.name 2>/dev/null || true)"
    user_email="$(git -C "$target_dir" config user.email 2>/dev/null || true)"
    local pending_actions=()
    if [ -z "$user_name" ] || [ -z "$user_email" ]; then
        identity_status="pendiente"
        pending_actions+=("Configurar identidad Git local: git -C \"$target_dir\" config user.name \"Tu Nombre\" && git -C \"$target_dir\" config user.email \"tu@correo.com\"")
    fi

    case "$github_mode" in
        codex-plugin)
            pending_actions+=("GitHub (codex-plugin): requerir accion manual del usuario para instalar/conectar el plugin GitHub desde /plugins e iniciar sesion.")
            ;;
        gh-cli)
            pending_actions+=("GitHub (gh-cli): requerir accion manual del usuario para ejecutar gh auth login y vincular el repositorio.")
            ;;
        manual)
            pending_actions+=("GitHub (manual): requerir accion manual del usuario para agregar el remoto con git -C \"$target_dir\" remote add origin <url-del-repositorio>.")
            ;;
    esac

    local prof
    for prof in "${selected_profiles[@]}"; do
        if [ "$prof" = "codex" ]; then
            pending_actions+=("Codex (pendiente - requiere accion del usuario): instalar la extension oficial 'Codex - OpenAI's coding agent' (openai.chatgpt), iniciar sesion y recargar el editor.")
        fi
    done

    echo "=== Resumen de composicion del harness ==="
    echo "Perfiles aplicados: ${selected_profiles[*]}"
    echo "Perfiles solicitados: $profiles_arg"
    echo "Archivos creados: $created_count"
    echo "Archivos ya presentes (sin cambios): $unchanged_count"
    echo "Archivos preservados (modificados por el usuario): $preserved_count"
    echo "Archivos actualizados: $updated_count"
    echo "Repositorio Git: $repo_status (identidad: $identity_status)"
    echo "Modo GitHub: $github_mode"

    if [ "${#pending_actions[@]}" -gt 0 ]; then
        echo "Acciones manuales pendientes:"
        local action
        for action in "${pending_actions[@]}"; do
            echo "  - [PENDIENTE] $action"
        done
    fi
}
