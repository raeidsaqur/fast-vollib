from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
UPSTREAM = ROOT.parent / "py_vollib_vectorized"

for path in (SRC, UPSTREAM):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
