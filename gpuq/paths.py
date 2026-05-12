from __future__ import annotations

import os
from pathlib import Path


def gpuq_dir() -> Path:
    base = os.environ.get("GPUQ_HOME")
    if base:
        return Path(base).expanduser()
    return Path.home() / ".gpuq"


def config_path() -> Path:
    return gpuq_dir() / "config.yaml"


def secrets_path() -> Path:
    return gpuq_dir() / "secrets.env"


def jobs_dir() -> Path:
    return gpuq_dir() / "jobs"


def next_id_path() -> Path:
    return gpuq_dir() / "next_id"


def daemon_pid_path() -> Path:
    return gpuq_dir() / "daemon.pid"


def daemon_log_path() -> Path:
    return gpuq_dir() / "daemon.log"


def state_lock_path() -> Path:
    return gpuq_dir() / "state.lock"


def ensure_dirs() -> None:
    gpuq_dir().mkdir(parents=True, exist_ok=True)
    jobs_dir().mkdir(parents=True, exist_ok=True)
