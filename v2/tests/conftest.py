"""Test setup isolated to the V2 source package."""

from __future__ import annotations

import sys
from pathlib import Path


V2_ROOT = Path(__file__).resolve().parents[1]
V2_SRC = V2_ROOT / "src"
sys.path.insert(0, str(V2_SRC))

