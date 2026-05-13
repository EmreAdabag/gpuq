from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yaml

from . import paths


@dataclass
class WorkerConfig:
    host: str
    user: str
    gpus: list[int]
    ssh_key: Optional[str] = None
    # Optional: shell snippet to run before the user's command, e.g.
    #   "source ~/miniconda3/etc/profile.d/conda.sh && conda activate myenv"
    # When set, gpuq skips `uv sync` on launch and runs the user command
    # without the `uv run` prefix — handy for conda envs or any pre-existing
    # interpreter that doesn't fit the uv workflow.
    env_setup: Optional[str] = None

    @property
    def ssh_target(self) -> str:
        return f"{self.user}@{self.host}"

    @property
    def uses_uv(self) -> bool:
        return self.env_setup is None


@dataclass
class Config:
    workers: list[WorkerConfig]
    repo_root: Path
    remote_repo_base: str
    shared_mount: str
    log_dir: str
    secrets_file: Path
    rsync_excludes: list[str]
    daemon_tick_seconds: int = 2
    gpu_free_memory_threshold_mb: int = 500

    def worker(self, host: str) -> Optional[WorkerConfig]:
        for w in self.workers:
            if w.host == host:
                return w
        return None


def _expand(p: str) -> str:
    return os.path.expanduser(str(p))


def load_config(path: Optional[Path] = None) -> Config:
    p = path or paths.config_path()
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {p}")
    raw = yaml.safe_load(p.read_text()) or {}
    workers: list[WorkerConfig] = []
    for w in raw.get("workers") or []:
        workers.append(
            WorkerConfig(
                host=w["host"],
                user=w["user"],
                gpus=[int(x) for x in (w.get("gpus") or [])],
                ssh_key=_expand(w["ssh_key"]) if w.get("ssh_key") else None,
                env_setup=w.get("env_setup"),
            )
        )
    return Config(
        workers=workers,
        repo_root=Path(_expand(raw["repo_root"])),
        remote_repo_base=_expand(raw["remote_repo_base"]),
        shared_mount=_expand(raw["shared_mount"]),
        log_dir=_expand(raw["log_dir"]),
        secrets_file=Path(_expand(raw.get("secrets_file", str(paths.secrets_path())))),
        rsync_excludes=list(raw.get("rsync_excludes") or []),
        daemon_tick_seconds=int(raw.get("daemon_tick_seconds", 2)),
        gpu_free_memory_threshold_mb=int(raw.get("gpu_free_memory_threshold_mb", 500)),
    )


def write_default_config(path: Optional[Path] = None) -> Path:
    p = path or paths.config_path()
    paths.ensure_dirs()
    if p.exists():
        return p
    default = {
        "workers": [],
        "repo_root": str(Path.home() / "code" / "training"),
        "remote_repo_base": "~/gpuq-repos",
        "shared_mount": "/mnt/shared",
        "log_dir": "/mnt/shared/gpuq-logs",
        "secrets_file": str(paths.secrets_path()),
        "rsync_excludes": [
            ".git",
            "__pycache__",
            "*.pyc",
            ".venv",
            "wandb/",
            "outputs/",
        ],
        "daemon_tick_seconds": 2,
        "gpu_free_memory_threshold_mb": 500,
    }
    p.write_text(yaml.safe_dump(default, sort_keys=False))
    return p


def add_worker_to_config(worker: WorkerConfig, path: Optional[Path] = None) -> None:
    """Append (or replace) a worker in the YAML config. PyYAML drops comments —
    acceptable for v0; switch to ruamel.yaml later if comment preservation matters."""
    p = path or paths.config_path()
    raw = yaml.safe_load(p.read_text()) if p.exists() else {}
    if raw is None:
        raw = {}
    existing = raw.get("workers") or []
    existing = [w for w in existing if w.get("host") != worker.host]
    entry: dict = {"host": worker.host, "user": worker.user, "gpus": worker.gpus}
    if worker.ssh_key:
        entry["ssh_key"] = worker.ssh_key
    existing.append(entry)
    raw["workers"] = existing
    p.write_text(yaml.safe_dump(raw, sort_keys=False))
