"""Version probes with bounded execution; Graphify metadata tied to its launcher."""
import importlib.metadata
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys


def graphify_version(launcher):
    path = Path(launcher).resolve(strict=True)
    data = path.read_bytes()
    pattern = rb'#!([^\r\n]{1,1024})\r?\n(?=PK\x03\x04)' if path.suffix.lower() == '.exe' else rb'^#!([^\r\n]+)'
    match = re.search(pattern, data)
    if not match:
        raise ValueError('Graphify launcher sin interprete verificable')
    line = match[1].decode('utf-8').strip()
    if line.startswith('"'):
        interpreter = line.split('"')[1]
    else:
        parts = shlex.split(line, posix=os.name != 'nt')
        if Path(parts[0]).name == 'env':
            if len(parts) != 2:
                raise ValueError('Shebang env no soportado')
            interpreter = shutil.which(parts[1])
        else:
            if len(parts) != 1:
                raise ValueError('Shebang no soportado')
            interpreter = parts[0]
    if not interpreter or not Path(interpreter).is_file():
        raise ValueError('Interprete de Graphify ausente')
    code = ('import importlib.metadata as m; d=m.distribution("graphifyy"); '
            'assert any(e.name=="graphify" and e.group=="console_scripts" for e in d.entry_points); '
            'print("graphifyy "+d.version)')
    result = subprocess.run([interpreter, '-c', code], capture_output=True, text=True, timeout=8, check=True)
    return result.stdout.strip()


def observed(command, args):
    launcher = shutil.which(command)
    if not launcher:
        raise ValueError('Ejecutable ausente')
    if command == 'graphify':
        return graphify_version(launcher)
    result = subprocess.run([launcher, *args], capture_output=True, text=True, timeout=10, check=True)
    return result.stdout.strip()


def check(tool, expected, policy):
    commands = {
        'Git': ('git', ['--version'], r'^git version (\d+\.\d+\.\d+)'),
        'Python': (sys.executable, ['--version'], r'^Python (\d+\.\d+\.\d+)$'),
        'Bash': ('bash', ['--version'], r'^GNU bash, version (\d+\.\d+\.\d+)'),
        'Node.js': ('node', ['--version'], r'^v(\d+\.\d+\.\d+)$'),
        '@fission-ai/openspec': ('openspec', ['--version'], r'^(\d+\.\d+\.\d+)$'),
        'graphifyy': ('graphify', [], r'^graphifyy (\d+\.\d+\.\d+)$'),
        'uv': ('uv', ['--version'], r'^uv (\d+\.\d+\.\d+)(?:\s|$)'),
        'Bats': ('bats', ['--version'], r'^Bats (\d+\.\d+\.\d+)$'),
        'ShellCheck': ('shellcheck', ['--version'], r'(?m)^version: (\d+\.\d+\.\d+)$'),
        'shfmt': ('shfmt', ['--version'], r'^v?(\d+\.\d+\.\d+)$'),
    }
    command, args, pattern = commands[tool]
    output = observed(command, args)
    match = re.search(pattern, output)
    if not match:
        raise ValueError('Version no interpretable')
    actual = tuple(map(int, match[1].split('.')))
    required = tuple(map(int, expected.split('.')))
    if (policy == 'exact' and actual != required) or (policy == 'minimum' and actual < required):
        raise ValueError(f'Version observada {match[1]}; requerida {policy} {expected}')
    return f'{tool}: observada {match[1]}; {policy} {expected}'


if __name__ == '__main__':
    try:
        if sys.argv[1] == '--graphify-launcher':
            print(graphify_version(sys.argv[2]))
        else:
            print(check(*sys.argv[1:]))
    except (ValueError, OSError, subprocess.SubprocessError, KeyError, importlib.metadata.PackageNotFoundError) as error:
        # No dump of subprocess output: diagnostics must not echo arbitrary credentials.
        print(f'FAIL version: {type(error).__name__}: {error if isinstance(error, ValueError) else "sonda no verificable"}', file=sys.stderr)
        sys.exit(1)
