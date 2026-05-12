from __future__ import annotations

import os
import shutil
import subprocess
import time
from pathlib import Path

import pytest
import yaml

from gpuq import config as cfg_mod
from gpuq import daemon as daemon_mod
from gpuq import jobs as jobs_mod
from gpuq.jobs import Job, now_iso, write_job


pytestmark = pytest.mark.skipif(
    os.environ.get("GPUQ_INTEGRATION") != "1",
    reason="set GPUQ_INTEGRATION=1 to run (needs ssh/uv/tmux/nvidia-smi to localhost)",
)


def _have(bin_: str) -> bool:
    return shutil.which(bin_) is not None


def _ssh_localhost_works() -> bool:
    res = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=accept-new",
         "localhost", "true"],
        capture_output=True, timeout=10,
    )
    return res.returncode == 0


@pytest.fixture
def integration_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    for binary in ("ssh", "rsync", "scp", "tmux", "uv", "nvidia-smi"):
        if not _have(binary):
            pytest.skip(f"{binary} not on PATH")
    if not _ssh_localhost_works():
        pytest.skip("passwordless ssh to localhost not configured")
    monkeypatch.setenv("GPUQ_HOME", str(tmp_path))

    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "pyproject.toml").write_text(
        '[project]\nname = "tinyrepo"\nversion = "0.0.0"\nrequires-python = ">=3.11"\n'
    )
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    user = os.environ.get("USER") or "root"
    cfg = {
        "workers": [{"host": "localhost", "user": user, "gpus": [0]}],
        "repo_root": str(repo),
        "remote_repo_base": str(tmp_path / "remote-repos"),
        "shared_mount": str(tmp_path),
        "log_dir": str(log_dir),
        "secrets_file": str(tmp_path / "secrets.env"),
        "rsync_excludes": [".git"],
        "daemon_tick_seconds": 1,
        "gpu_free_memory_threshold_mb": 1_000_000,  # always consider GPU free
    }
    from gpuq import paths

    paths.config_path().parent.mkdir(parents=True, exist_ok=True)
    paths.config_path().write_text(yaml.safe_dump(cfg, sort_keys=False))
    return tmp_path


def test_full_flow_localhost(integration_env: Path) -> None:
    cfg = cfg_mod.load_config()
    # Submit a quick job.
    jid = jobs_mod.next_id()
    job = Job(
        id=jid,
        name="hello",
        command='python -c "print(\'ok\')"',
        status="queued",
        gpus_requested=1,
        submitted_at=now_iso(),
    )
    write_job(job)

    deadline = time.time() + 120
    while time.time() < deadline:
        daemon_mod.tick(cfg)
        cur = jobs_mod.read_job(jid)
        if cur.status in ("done", "failed", "killed"):
            break
        time.sleep(1)
    cur = jobs_mod.read_job(jid)
    assert cur.status == "done", f"got {cur.status}; log: {Path(cur.log_path).read_text() if cur.log_path else 'no log'}"
    assert cur.exit_code == 0
    log_text = Path(cur.log_path).read_text()
    assert "ok" in log_text
