"""Validated JSON, context and file plans for Bash. No installs or network."""
import json
import os
import re
import sys
import tempfile
from pathlib import Path

CONTEXT = '.harness/context.json'


def safe_path(value):
    if not isinstance(value, str) or re.search(r'[:\x00-\x1f\x7f*?<>|"]', value):
        raise ValueError('Ruta invalida en manifiesto')
    parts = value.replace('\\', '/').split('/')
    if any(not p or p.lower() in ('.', '..', '.git') or p.endswith(('.', ' ')) for p in parts):
        raise ValueError('Ruta insegura en manifiesto')
    return '/'.join(parts)


def pair(entry):
    if isinstance(entry, str):
        return safe_path(entry), safe_path(entry)
    if not isinstance(entry, dict) or set(entry) != {'source', 'target'}:
        raise ValueError('Entrada requiere string o source/target')
    return safe_path(entry['source']), safe_path(entry['target'])


def catalog(path):
    with open(path, encoding='utf-8-sig') as stream:
        data = json.load(stream)
    profiles = data.get('profiles') if isinstance(data, dict) else None
    if not isinstance(profiles, dict) or 'base' not in profiles:
        raise ValueError('El manifiesto requiere el objeto profiles y base')
    for name, entries in profiles.items():
        if not re.fullmatch(r'[a-z][a-z0-9-]*', name) or name == 'all' or not isinstance(entries, list):
            raise ValueError('Esquema de perfil invalido')
        for entry in entries:
            target = pair(entry)[1].lower()
            if target in (CONTEXT, '.harness') or target.startswith(CONTEXT + '/'):
                raise ValueError('Ruta reservada del contexto')
    return profiles


def selection(profiles, requested):
    names = [part.strip().lower() for part in requested.split(',') if part.strip()]
    if any(name not in profiles and name != 'all' for name in names):
        raise ValueError('Perfil desconocido')
    wanted = list(profiles) if 'all' in names else names
    return ['base', *(name for name in profiles if name != 'base' and name in wanted)]


def managed_path(path, checked):
    for part in (path, *path.parents):
        if part in checked:
            break
        junction = os.name == 'nt' and part.exists() and getattr(part.lstat(), 'st_reparse_tag', 0) == 0xA0000003
        if part.is_symlink() or junction:
            raise ValueError('Enlace no permitido en ruta administrada')
        if part != path and part.exists() and not part.is_dir():
            raise ValueError('Un archivo bloquea un directorio requerido')
        if part.is_dir():
            checked.add(part)


def context(root, profiles, missing=False):
    path = root / CONTEXT
    managed_path(path, set())
    if not path.exists() and missing:
        return None
    if not path.is_file():
        raise ValueError('Contexto ausente; adoptar con setup-project y perfiles explicitos')
    with path.open(encoding='utf-8-sig') as stream:
        data = json.load(stream)
    if (not isinstance(data, dict) or set(data) != {'schema_version', 'role', 'profiles', 'github_mode'}
            or type(data['schema_version']) is not int or data['schema_version'] != 1
            or data['role'] not in ('authoring', 'consumer')
            or not isinstance(data['profiles'], list) or not data['profiles']
            or any(not isinstance(p, str) or p not in profiles for p in data['profiles'])
            or data['profiles'] != selection(profiles, ','.join(data['profiles']))
            or data['github_mode'] not in ('skip', 'codex-plugin', 'gh-cli', 'manual')):
        raise ValueError('Contexto invalido o schema desconocido; no se modifica')
    return data


def effective(root, profiles, requested):
    selected = selection(profiles, requested)
    previous = context(root, profiles, missing=True)
    if previous:
        selected = selection(profiles, ','.join(selected + previous['profiles']))
    return selected, previous


def entries_for(profiles, selected):
    destinations = {}
    for name in selected:
        for entry in profiles[name]:
            src, dst = pair(entry)
            key = dst.lower()
            if key in destinations and destinations[key] != (src, dst):
                raise ValueError('Colision de archivos en manifiesto')
            destinations[key] = (src, dst)
    keys = set(destinations)
    for key in keys:
        if any('/'.join(key.split('/')[:i]) in keys for i in range(1, len(key.split('/')))):
            raise ValueError('Colision archivo/directorio en manifiesto')
    return list(destinations.values())


def check_resources(root, profiles, selected, role):
    checked = set()
    entries = entries_for(profiles, selected)
    if not entries:
        raise ValueError('Suite sin recursos/casos')
    for src, dst in entries:
        if role == 'authoring':
            dst = src
        path = root / dst
        managed_path(path, checked)
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f'Recurso requerido ausente o vacio: {dst}')
    return len(entries)


def main():
    mode, manifest, requested, *roots = sys.argv[1:]
    profiles = catalog(manifest)
    selected = selection(profiles, requested)
    if mode == 'profiles':
        return selected
    if mode == 'files':
        return [dst for _, dst in entries_for(profiles, selected)]
    if mode == 'active':
        return effective(Path(roots[0]).absolute(), profiles, requested)[0]
    if mode == 'role':
        return [context(Path(roots[0]).absolute(), profiles)['role']]
    if mode == 'selected':
        return context(Path(roots[0]).absolute(), profiles)['profiles']
    if mode == 'check':
        root = Path(roots[0]).absolute()
        current = context(root, profiles)
        count = check_resources(root, profiles, current['profiles'], current['role'])
        return [f'PASS consumer.resources: {count} casos; role={current["role"]}']
    if mode == 'plan':
        source, target = (Path(root).absolute() for root in roots)
        selected, _ = effective(target, profiles, requested)
        entries = entries_for(profiles, selected)
        checked = set()
        managed_path(target, checked)
        if target.exists() and not target.is_dir():
            raise ValueError('El destino no es un directorio')
        for src, dst in entries:
            src_path, dst_path = source / src, target / dst
            managed_path(src_path, checked)
            managed_path(dst_path, checked)
            if not src_path.is_file():
                raise ValueError(f'Fuente requerida ausente o no es un archivo: {src}')
            if dst_path.is_dir():
                raise ValueError(f'El destino de archivo es un directorio: {dst}')
        return [f'{src}\t{dst}' for src, dst in entries]
    if mode == 'write-context':
        root = Path(roots[0]).absolute()
        selected, previous = effective(root, profiles, requested)
        mode_value = roots[1]
        if mode_value not in ('skip', 'codex-plugin', 'gh-cli', 'manual'):
            raise ValueError('Modo GitHub invalido')
        data = dict(schema_version=1, role=previous['role'] if previous else 'consumer',
                    profiles=selected, github_mode=mode_value)
        path = root / CONTEXT
        path.parent.mkdir(parents=True, exist_ok=True)
        name = None
        try:
            with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent, delete=False) as stream:
                name = stream.name
                json.dump(data, stream, indent=2)
                stream.write('\n')
            os.replace(name, path)
        finally:
            if name and os.path.exists(name):
                os.unlink(name)
        return []
    raise ValueError('Operacion de manifiesto invalida')


if __name__ == '__main__':
    try:
        for line in main():
            print(line)
    except (ValueError, OSError, TypeError, KeyError) as error:
        print(f'Preflight FAIL: {error}', file=sys.stderr)
        sys.exit(1)
