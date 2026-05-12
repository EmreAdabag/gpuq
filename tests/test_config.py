from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from gpuq import config as cfg_mod
from gpuq import paths


def _write_config(p: Path, data: dict) -> None:
    p.write_text(yaml.safe_dump(data, sort_keys=False))


def test_load_config_minimal(gpuq_home: Path) -> None:
    data = {
        "workers": [
            {"host": "localhost", "user": "me", "gpus": [0, 1]},
        ],
        "repo_root": str(gpuq_home / "repo"),
        "remote_repo_base": "~/gpuq-repos",
        "shared_mount": "/mnt/shared",
        "log_dir": "/mnt/shared/gpuq-logs",
        "rsync_excludes": [".git"],
    }
    _write_config(paths.config_path(), data)
    c = cfg_mod.load_config()
    assert len(c.workers) == 1
    assert c.workers[0].host == "localhost"
    assert c.workers[0].gpus == [0, 1]
    assert c.workers[0].ssh_target == "me@localhost"
    assert c.daemon_tick_seconds == 2
    assert c.gpu_free_memory_threshold_mb == 500


def test_load_config_missing(gpuq_home: Path) -> None:
    with pytest.raises(FileNotFoundError):
        cfg_mod.load_config()


def test_write_default_and_add_worker(gpuq_home: Path) -> None:
    p = cfg_mod.write_default_config()
    assert p.exists()
    # adding a worker is idempotent on host (replaces)
    w = cfg_mod.WorkerConfig(host="gpu1.lan", user="me", gpus=[0, 1, 2])
    cfg_mod.add_worker_to_config(w)
    cfg_mod.add_worker_to_config(w)  # second call shouldn't duplicate
    c = cfg_mod.load_config()
    matching = [x for x in c.workers if x.host == "gpu1.lan"]
    assert len(matching) == 1
    assert matching[0].gpus == [0, 1, 2]


def test_add_worker_preserves_other_workers(gpuq_home: Path) -> None:
    cfg_mod.write_default_config()
    cfg_mod.add_worker_to_config(cfg_mod.WorkerConfig(host="a", user="me", gpus=[0]))
    cfg_mod.add_worker_to_config(cfg_mod.WorkerConfig(host="b", user="me", gpus=[0, 1]))
    c = cfg_mod.load_config()
    hosts = {w.host for w in c.workers}
    assert hosts == {"a", "b"}
