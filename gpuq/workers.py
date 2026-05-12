from __future__ import annotations

import logging
import shlex
import subprocess
from dataclasses import dataclass
from typing import Optional

from .config import WorkerConfig


log = logging.getLogger(__name__)


def ssh_cmd(worker: WorkerConfig, remote_cmd: str, *, batch: bool = True) -> list[str]:
    args = ["ssh"]
    if batch:
        args += ["-o", "BatchMode=yes"]
    args += ["-o", "StrictHostKeyChecking=accept-new"]
    if worker.ssh_key:
        args += ["-i", worker.ssh_key]
    args += [worker.ssh_target, remote_cmd]
    return args


def run_ssh(
    worker: WorkerConfig,
    remote_cmd: str,
    *,
    timeout: int = 30,
    check: bool = False,
) -> subprocess.CompletedProcess:
    cmd = ssh_cmd(worker, remote_cmd)
    log.debug("SSH %s: %s", worker.host, remote_cmd)
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=check)


@dataclass
class GpuStat:
    index: int
    used_mb: int


def probe_gpus(worker: WorkerConfig, timeout: int = 15) -> list[GpuStat]:
    cmd = "nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits"
    res = run_ssh(worker, cmd, timeout=timeout)
    if res.returncode != 0:
        raise RuntimeError(f"nvidia-smi failed on {worker.host}: {res.stderr.strip()}")
    return parse_nvidia_smi(res.stdout)


def parse_nvidia_smi(text: str) -> list[GpuStat]:
    stats: list[GpuStat] = []
    for line in text.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 2:
            continue
        try:
            stats.append(GpuStat(index=int(parts[0]), used_mb=int(parts[1])))
        except ValueError:
            continue
    return stats


def select_available_gpus(
    allowed: list[int],
    gpu_stats: dict[int, int],
    used_by_gpuq: set[int],
    threshold_mb: int,
) -> list[int]:
    """Pure function: which of `allowed` GPUs are free for a new gpuq job?"""
    available: list[int] = []
    for idx in sorted(allowed):
        if idx in used_by_gpuq:
            continue
        used_mb = gpu_stats.get(idx)
        if used_mb is None:
            continue
        if used_mb < threshold_mb:
            available.append(idx)
    return available


def tmux_has_session(worker: WorkerConfig, session: str) -> bool:
    res = run_ssh(worker, f"tmux has-session -t {shlex.quote(session)} 2>/dev/null")
    return res.returncode == 0


def tmux_kill_session(worker: WorkerConfig, session: str) -> None:
    run_ssh(worker, f"tmux kill-session -t {shlex.quote(session)} 2>/dev/null || true")


def remote_test_dir(worker: WorkerConfig, path: str) -> bool:
    res = run_ssh(worker, f"test -d {shlex.quote(path)}")
    return res.returncode == 0


def remote_which(worker: WorkerConfig, binary: str) -> Optional[str]:
    res = run_ssh(worker, f"command -v {shlex.quote(binary)}")
    if res.returncode != 0:
        return None
    return res.stdout.strip() or None


def remote_mkdir_p(worker: WorkerConfig, *paths: str) -> None:
    if not paths:
        return
    quoted = " ".join(shlex.quote(p) for p in paths)
    run_ssh(worker, f"mkdir -p {quoted}", check=True)


def remote_cat_int(worker: WorkerConfig, path: str) -> tuple[bool, Optional[int]]:
    """Return (file_exists, parsed_int_or_None). Used by the daemon to read the
    job's exit file when shared_mount isn't actually shared with the hub."""
    cmd = (
        f"if [ -f {shlex.quote(path)} ]; then "
        f"echo PRESENT; cat {shlex.quote(path)}; "
        f"fi"
    )
    res = run_ssh(worker, cmd)
    if res.returncode != 0:
        return False, None
    lines = res.stdout.splitlines()
    if not lines or lines[0].strip() != "PRESENT":
        return False, None
    if len(lines) < 2:
        return True, None
    try:
        return True, int(lines[1].strip())
    except ValueError:
        return True, None
