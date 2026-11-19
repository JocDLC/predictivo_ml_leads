"""Arnés no destructivo del proyecto: tests unitarios, lint y vía TDD.

Uso:
    python scripts/harness.py unit                 # suite unittest (fixtures sintéticas)
    python scripts/harness.py lint                 # ruff sobre tests/ y este script
    python scripts/harness.py tdd-red <test_id>    # exige que el test falle (fase RED)
    python scripts/harness.py tdd-green <test_id>  # test + suite + lint en verde

<test_id> es un id de unittest, ej. tests.test_inference_units.TestFranja.test_tarde
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

LINT_TARGETS = ["tests", "scripts/harness.py"]


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    completed = subprocess.run(cmd, cwd=ROOT, check=False)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


def run_unit() -> None:
    run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-t", ".", "-v"])


def run_lint() -> None:
    run([sys.executable, "-m", "ruff", "check", *LINT_TARGETS])


def run_tdd_red(test_id: str) -> None:
    """Verifica que el test nuevo falle antes de implementar (fase RED)."""
    completed = subprocess.run(
        [sys.executable, "-m", "unittest", test_id, "-v"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.stdout:
        print(completed.stdout, end="", flush=True)
    if completed.stderr:
        print(completed.stderr, end="", file=sys.stderr, flush=True)
    if completed.returncode == 0:
        raise SystemExit(
            "[tdd-red] El test pasa: no prueba comportamiento nuevo. "
            "Ajustá el test antes de implementar."
        )
    output = completed.stdout + completed.stderr
    if "failures=" in output or "FAILED (failures=" in output:
        print("[tdd-red] OK: el test falla como se esperaba. Pasá a GREEN.")
    else:
        print(
            "[tdd-red] AVISO: el test falla por error (import/nombre/sintaxis), "
            "no por aserción. Verificá que falle por la funcionalidad faltante."
        )


def run_tdd_green(test_id: str) -> None:
    """Verifica que el test pase y que la suite y el lint sigan en verde."""
    run([sys.executable, "-m", "unittest", test_id, "-v"])
    run_unit()
    run_lint()
    print("[tdd-green] OK: test, suite y lint en verde.")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Arnés no destructivo de predictivo_ml_leads")
    parser.add_argument("command", choices=("unit", "lint", "tdd-red", "tdd-green"))
    parser.add_argument("test_id", nargs="?", default=None)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    if args.command == "unit":
        run_unit()
    elif args.command == "lint":
        run_lint()
    else:
        if not args.test_id:
            raise SystemExit(f"Falta <test_id> para '{args.command}'.")
        if args.command == "tdd-red":
            run_tdd_red(args.test_id)
        else:
            run_tdd_green(args.test_id)


if __name__ == "__main__":
    main()
