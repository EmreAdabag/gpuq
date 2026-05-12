from __future__ import annotations

import os
from pathlib import Path

import pytest


@pytest.fixture
def gpuq_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Isolate the gpuq state directory per test."""
    monkeypatch.setenv("GPUQ_HOME", str(tmp_path))
    return tmp_path
