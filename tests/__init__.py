import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
for _pkg_dir in ("streamlit_app", "scripts"):
    sys.path.insert(0, str(_REPO_ROOT / _pkg_dir))
