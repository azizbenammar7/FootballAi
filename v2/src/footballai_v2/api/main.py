"""Environment-configured ASGI entry point for the local V2 API."""

from __future__ import annotations

import os
from pathlib import Path

from footballai_v2.api import create_app


run_root = Path(os.environ.get("FOOTBALLAI_V2_RUN_ROOT", "data/runs"))
allowed_origins = tuple(
    item.strip()
    for item in os.environ.get(
        "FOOTBALLAI_V2_CORS_ORIGINS",
        "http://localhost:5173,http://127.0.0.1:5173",
    ).split(",")
    if item.strip()
)

app = create_app(run_root, allowed_origins=allowed_origins)
