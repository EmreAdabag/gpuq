from __future__ import annotations

import logging
import shlex
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

from . import workers
from .config import Config, WorkerConfig
from .jobs import Job, now_iso, write_job


log = logging.getLogger(__name__)


LAUNCH_SCRIPT_TEMPLATE = """#!/usr/bin/env bash
set -a
[ -f .gpuq-secrets ] && source .gpuq-secrets
set +a
export CUDA_VISIBLE_DEVICES={cuda_devices}
export GPUQ_JOB_ID={job_id}
export GPUQ_HOST={host}
cd {repo_path}
{{
  uv run {command}
}} > >(tee -a {log_path}) 2> >(tee -a {log_path} >&2)
echo $? > {exit_path}
"""


def build_launch_script(job: Job, host: str) -> str:
    cuda = ",".join(str(g) for g in job.gpus_assigned)
    return LAUNCH_SCRIPT_TEMPLATE.format(
        cuda_devices=shlex.quote(cuda),
        job_id=job.id,
        host=shlex.quote(host),
        repo_path=shlex.quote(job.remote_repo_path or ""),
        command=job.command,
        log_path=shlex.quote(job.log_path or ""),
        exit_path=shlex.quote(job.exit_path or ""),
    )


def _ssh_e_flag(worker: WorkerConfig) -> str:
    parts = ["ssh", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=accept-new"]
    if worker.ssh_key:
        parts += ["-i", worker.ssh_key]
    return " ".join(shlex.quote(p) if " " in p else p for p in parts)


def rsync_repo(
    cfg: Config,
    worker: WorkerConfig,
    dest_path: str,
    log_file: Optional[Path] = None,
) -> None:
    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".excludes") as f:
        f.write("\n".join(cfg.rsync_excludes) + "\n")
        excludes_file = f.name

    last_synced = f"{cfg.remote_repo_base}/last-synced"
    args = [
        "rsync",
        "-az",
        "--delete",
        f"--exclude-from={excludes_file}",
        f"--link-dest={last_synced}/",
        "-e",
        _ssh_e_flag(worker),
    ]
    src = str(cfg.repo_root).rstrip("/") + "/"
    args += [src, f"{worker.ssh_target}:{dest_path}/"]
    log.debug("rsync %s -> %s:%s", src, worker.host, dest_path)
    res = subprocess.run(args, capture_output=True, text=True, timeout=600)
    Path(excludes_file).unlink(missing_ok=True)
    if log_file:
        try:
            log_file.parent.mkdir(parents=True, exist_ok=True)
            with open(log_file, "a") as lf:
                lf.write("=== rsync ===\n")
                if res.stdout:
                    lf.write(res.stdout)
                if res.stderr:
                    lf.write(res.stderr)
        except OSError:
            pass
    if res.returncode != 0:
        raise RuntimeError(f"rsync failed: {res.stderr.strip()}")


def update_last_synced(worker: WorkerConfig, cfg: Config, job_path: str) -> None:
    target = f"{cfg.remote_repo_base}/last-synced"
    cmd = f"ln -sfn {shlex.quote(job_path)} {shlex.quote(target)}"
    workers.run_ssh(worker, cmd, check=False)


def uv_sync(worker: WorkerConfig, repo_path: str, log_path: str) -> int:
    inner = (
        f"cd {shlex.quote(repo_path)} && "
        f"set -o pipefail && "
        f"uv sync 2>&1 | tee -a {shlex.quote(log_path)}"
    )
    cmd = f"bash -c {shlex.quote(inner)}"
    res = workers.run_ssh(worker, cmd, timeout=900)
    return res.returncode


def push_secrets(cfg: Config, worker: WorkerConfig, repo_path: str) -> None:
    if not cfg.secrets_file.exists():
        return
    dest = f"{worker.ssh_target}:{repo_path}/.gpuq-secrets"
    args = ["scp", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=accept-new"]
    if worker.ssh_key:
        args += ["-i", worker.ssh_key]
    args += [str(cfg.secrets_file), dest]
    res = subprocess.run(args, capture_output=True, text=True, timeout=60)
    if res.returncode != 0:
        raise RuntimeError(f"scp secrets failed: {res.stderr.strip()}")
    workers.run_ssh(worker, f"chmod 600 {shlex.quote(repo_path + '/.gpuq-secrets')}")


def push_launch_script(worker: WorkerConfig, script: str, repo_path: str) -> str:
    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".sh") as f:
        f.write(script)
        local_path = f.name
    remote_path = f"{repo_path}/.gpuq-launch.sh"
    args = ["scp", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=accept-new"]
    if worker.ssh_key:
        args += ["-i", worker.ssh_key]
    args += [local_path, f"{worker.ssh_target}:{remote_path}"]
    res = subprocess.run(args, capture_output=True, text=True, timeout=60)
    Path(local_path).unlink(missing_ok=True)
    if res.returncode != 0:
        raise RuntimeError(f"scp launch script failed: {res.stderr.strip()}")
    return remote_path


def tmux_launch(worker: WorkerConfig, session: str, repo_path: str) -> None:
    inner = f"cd {shlex.quote(repo_path)} && bash .gpuq-launch.sh"
    cmd = f"tmux new-session -d -s {shlex.quote(session)} {shlex.quote(inner)}"
    res = workers.run_ssh(worker, cmd, check=False)
    if res.returncode != 0:
        raise RuntimeError(f"tmux launch failed: {res.stderr.strip()}")


def git_commit_of(repo_root: Path) -> Optional[str]:
    try:
        res = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if res.returncode != 0:
        return None
    return res.stdout.strip() or None


def launch_job(cfg: Config, worker: WorkerConfig, job: Job) -> None:
    """Run the full launch sequence on `worker` for `job`. On any failure mid-way,
    marks the job failed with detail in the log file. Always writes the job file
    once on success (status=running) or failure (status=failed)."""
    repo_path = f"{cfg.remote_repo_base}/job-{job.id}"
    log_path = f"{cfg.log_dir}/{job.id}.log"
    exit_path = f"{cfg.log_dir}/{job.id}.exit"
    session = f"gpuq-{job.id}"

    job.host = worker.host
    job.tmux_session = session
    job.remote_repo_path = repo_path
    job.log_path = log_path
    job.exit_path = exit_path
    job.git_commit = git_commit_of(cfg.repo_root)

    log.info("Launching job %d on %s gpus=%s", job.id, worker.host, job.gpus_assigned)

    # 1. Directories. log_dir is on shared mount, but ask the worker side too.
    try:
        workers.remote_mkdir_p(worker, repo_path, cfg.log_dir)
    except Exception as e:
        return _fail(job, log_path, f"mkdir failed: {e}")

    try:
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        Path(log_path).touch()
    except OSError:
        # shared mount might not be writable from the hub (unlikely per spec, but
        # not fatal — the worker writes the log).
        pass

    # 2. Rsync.
    try:
        rsync_repo(cfg, worker, repo_path, Path(log_path))
    except Exception as e:
        return _fail(job, log_path, f"rsync failed: {e}")
    update_last_synced(worker, cfg, repo_path)

    # 3. uv sync.
    try:
        rc = uv_sync(worker, repo_path, log_path)
    except Exception as e:
        return _fail(job, log_path, f"uv sync errored: {e}")
    if rc != 0:
        return _fail(job, log_path, f"uv sync exited {rc}")

    # 4. Secrets.
    try:
        push_secrets(cfg, worker, repo_path)
    except Exception as e:
        log.warning("push_secrets failed for job %d: %s", job.id, e)

    # 5. Push launcher + tmux.
    script = build_launch_script(job, worker.host)
    try:
        push_launch_script(worker, script, repo_path)
        tmux_launch(worker, session, repo_path)
    except Exception as e:
        return _fail(job, log_path, f"launch failed: {e}")

    job.status = "running"
    job.started_at = now_iso()
    write_job(job)


def _fail(job: Job, log_path: str, message: str) -> None:
    log.warning("Job %d: %s", job.id, message)
    try:
        with open(log_path, "a") as lf:
            lf.write(f"\n[gpuq] {message}\n")
    except OSError:
        pass
    job.status = "failed"
    job.ended_at = now_iso()
    write_job(job)
