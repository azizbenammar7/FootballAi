"""Shared fixtures for bounded V1 characterization tests."""

from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures"


def load_numeric_module(module_name: str, relative_path: str) -> ModuleType:
    """Load a numeric-prefix pipeline module without executing its main()."""
    path = ROOT / relative_path
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="session")
def stats_module() -> ModuleType:
    return load_numeric_module("footballai_v1_stats", "pipeline/02_stats.py")


@pytest.fixture(scope="session")
def fatigue_module() -> ModuleType:
    return load_numeric_module("footballai_v1_fatigue", "pipeline/03_fatigue.py")


@pytest.fixture(scope="session")
def minimal_player_summary() -> dict[str, Any]:
    return json.loads((FIXTURES / "minimal_player_summary.json").read_text())


@pytest.fixture(scope="session")
def minimal_risk_scores() -> dict[str, Any]:
    return json.loads((FIXTURES / "minimal_risk_scores.json").read_text())


def assert_finite_json(value: Any, path: str = "root") -> None:
    """Recursively reject NaN and infinity while accepting normal JSON values."""
    if isinstance(value, float):
        assert math.isfinite(value), f"non-finite number at {path}"
    elif isinstance(value, dict):
        for key, child in value.items():
            assert_finite_json(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            assert_finite_json(child, f"{path}[{index}]")
