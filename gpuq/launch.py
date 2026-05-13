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


_LAUNCH_HEADER = """#!/usr/bin/env bash
set -a
[ -f .gpuq-secrets ] && source .gpuq-secrets
set +a
export CUDA_VISIBLE_DEVICES={cuda_devices}
export GPUQ_JOB_ID={job_id}
export GPUQ_HOST={host}
cd {repo_path}
"""

_LAUNCH_BODY_UV = """{{
  uv run {command}
}} > >(tee -a {log_path}) 2> >(tee -a {log_path} >&2)
echo $? > {exit_path}
"""

# When env_setup is set, we don't wrap in `uv run` — the user's command runs
# in the activated environment exactly as they'd run it locally.
_LAUNCH_BODY_ENV = """{{
{env_setup}
  {command}
}} > >(tee -a {log_path}) 2> >(tee -a {log_path} >&2)
echo $? > {exit_path}
"""


def build_launch_script(job: Job, worker: "WorkerConfig") -> str:
    cuda = ",".join(str(g) for g in job.gpus_assigned)
    header = _LAUNCH_HEADER.format(
        cuda_devices=shlex.quote(cuda),
        job_id=job.id,
        host=shlex.quote(worker.host),
        repo_path=shlex.quote(job.remote_repo_path or ""),
    )
    if worker.env_setup:
        # Indent the user's env_setup snippet so it sits cleanly inside the { ... }
        # group and any errors are still captured by the tee redirection.
        env_block = "\n".join("  " + line for line in worker.env_setup.strip().splitlines())
        body = _LAUNCH_BODY_ENV.format(
            env_setup=env_block,
            command=job.command,
            log_path=shlex.quote(job.log_path or ""),
            exit_path=shlex.quote(job.exit_path or ""),
        )
    else:
        body = _LAUNCH_BODY_UV.format(
            command=job.command,
            log_path=shlex.quote(job.log_path or ""),
            exit_path=shlex.quote(job.exit_path or ""),
        )
    return header + body


def _ssh_e_flag(worker: WorkerConfig) -> str:
    parts = ["ssh", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=accept-new"]
    if worker.ssh_key:
        parts += ["-i", worker.ssh_key]
    return " ".join(shlex.quote(p) if " " in p else p for p in parts)


def rsync_repo(
    cfg: Config,
    worker: WorkerConfig,
    dest_path: str,
) -> subprocess.CompletedProcess:
    """rsync hub's repo_root to <worker>:<dest_path>. Caller decides what to do
    with stdout/stderr — typically: ignore on success, append to the worker-side
    log on failure (so users see it in `gpuq logs`)."""
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
    if res.returncode != 0:
        raise RuntimeError(f"rsync failed: {res.stderr.strip()}")
    return res


def update_last_synced(worker: WorkerConfig, cfg: Config, job_path: str) -> None:
    target = f"{cfg.remote_repo_base}/last-synced"
    cmd = f"ln -sfn {shlex.quote(job_path)} {shlex.quote(target)}"
    workers.run_ssh(worker, cmd, check=False)


def uv_sync(worker: WorkerConfig, repo_path: str, log_path: str) -> int:
    # `bash -lc` so ~/.profile gets sourced and uv is on PATH for non-interactive ssh.
    inner = (
        f"cd {shlex.quote(repo_path)} && "
        f"set -o pipefail && "
        f"uv sync 2>&1 | tee -a {shlex.quote(log_path)}"
    )
    cmd = f"bash -lc {shlex.quote(inner)}"
    res = workers.run_ssh(worker, cmd, timeout=900)
    return res.returncode


def _push_bytes(worker: WorkerConfig, data: bytes, remote_path: str, mode: int = 0o644) -> None:
    """Pipe local bytes to a remote file via `ssh host 'cat > path'`.

    Avoids depending on the sftp subsystem being configured on the worker (which
    is what modern scp uses by default); only needs sshd to accept a remote shell."""
    cmd = (
        f"umask 077 && mkdir -p {shlex.quote(str(Path(remote_path).parent))} && "
        f"cat > {shlex.quote(remote_path)} && "
        f"chmod {oct(mode)[2:]} {shlex.quote(remote_path)}"
    )
    args = workers.ssh_cmd(worker, cmd)
    res = subprocess.run(args, input=data, capture_output=True, timeout=60)
    if res.returncode != 0:
        raise RuntimeError(f"push to {worker.host}:{remote_path} failed: {res.stderr.decode(errors='replace').strip()}")


def push_secrets(cfg: Config, worker: WorkerConfig, repo_path: str) -> None:
    if not cfg.secrets_file.exists():
        return
    _push_bytes(worker, cfg.secrets_file.read_bytes(), f"{repo_path}/.gpuq-secrets", mode=0o600)


def push_launch_script(worker: WorkerConfig, script: str, repo_path: str) -> str:
    remote_path = f"{repo_path}/.gpuq-launch.sh"
    _push_bytes(worker, script.encode(), remote_path, mode=0o755)
    return remote_path


def tmux_launch(worker: WorkerConfig, session: str, repo_path: str) -> None:
    # `bash -l` runs the launcher as a login shell so the user's PATH (~/.profile)
    # is picked up — otherwise `uv` and other ~/.local/bin tools aren't found.
    inner = f"cd {shlex.quote(repo_path)} && bash -l .gpuq-launch.sh"
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

    # 1. Ensure remote dirs.
    try:
        workers.remote_mkdir_p(worker, repo_path, cfg.log_dir)
    except Exception as e:
        return _fail(job, worker, log_path, f"mkdir failed: {e}")

    # 2. Rsync. On failure, surface stderr to the worker-side log.
    try:
        rsync_repo(cfg, worker, repo_path)
    except Exception as e:
        return _fail(job, worker, log_path, f"rsync failed: {e}")
    update_last_synced(worker, cfg, repo_path)

    # 3. Env sync. With env_setup configured (e.g. conda), the user is
    # responsible for keeping the env up to date on the worker.
    if worker.uses_uv:
        try:
            rc = uv_sync(worker, repo_path, log_path)
        except Exception as e:
            return _fail(job, worker, log_path, f"uv sync errored: {e}")
        if rc != 0:
            return _fail(job, worker, log_path, f"uv sync exited {rc}")

    # 4. Secrets.
    try:
        push_secrets(cfg, worker, repo_path)
    except Exception as e:
        log.warning("push_secrets failed for job %d: %s", job.id, e)

    # 5. Push launcher + tmux.
    script = build_launch_script(job, worker)
    try:
        push_launch_script(worker, script, repo_path)
        tmux_launch(worker, session, repo_path)
    except Exception as e:
        return _fail(job, worker, log_path, f"launch failed: {e}")

    job.status = "running"
    job.started_at = now_iso()
    write_job(job)


def _append_remote_log(worker: WorkerConfig, log_path: str, text: str) -> None:
    """Append text to the worker-side log file. Best-effort; swallows errors so
    we still mark the job failed even if the worker is unreachable."""
    try:
        cmd = f"mkdir -p {shlex.quote(str(Path(log_path).parent))} && cat >> {shlex.quote(log_path)}"
        args = workers.ssh_cmd(worker, cmd)
        subprocess.run(args, input=text.encode(), capture_output=True, timeout=30)
    except Exception:
        pass


def _fail(job: Job, worker: WorkerConfig, log_path: str, message: str) -> None:
    log.warning("Job %d: %s", job.id, message)
    _append_remote_log(worker, log_path, f"\n[gpuq] {message}\n")
    job.status = "failed"
    job.ended_at = now_iso()
    write_job(job)
