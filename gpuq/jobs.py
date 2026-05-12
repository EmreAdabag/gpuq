from __future__ import annotations

import contextlib
import datetime as dt
import fcntl
import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterator, Optional

from . import paths


JobStatus = str  # "queued" | "running" | "done" | "failed" | "killed"


@dataclass
class Job:
    id: int
    name: str
    command: str
    status: JobStatus = "queued"
    host_pin: Optional[str] = None
    gpus_requested: int = 1
    host: Optional[str] = None
    gpus_assigned: list[int] = field(default_factory=list)
    tmux_session: Optional[str] = None
    remote_repo_path: Optional[str] = None
    log_path: Optional[str] = None
    exit_path: Optional[str] = None
    git_commit: Optional[str] = None
    submitted_at: Optional[str] = None
    started_at: Optional[str] = None
    ended_at: Optional[str] = None
    exit_code: Optional[int] = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Job":
        return cls(**d)

    @property
    def session_name(self) -> str:
        return self.tmux_session or f"gpuq-{self.id}"


def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def job_path(job_id: int) -> Path:
    return paths.jobs_dir() / f"{job_id}.json"


def kill_path(job_id: int) -> Path:
    return paths.jobs_dir() / f"{job_id}.kill"


@contextlib.contextmanager
def state_lock() -> Iterator[None]:
    paths.ensure_dirs()
    f = open(paths.state_lock_path(), "w")
    try:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        f.close()


def next_id() -> int:
    paths.ensure_dirs()
    p = paths.next_id_path()
    with open(p, "a+") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            f.seek(0)
            raw = f.read().strip()
            n = int(raw) if raw else 1
            f.seek(0)
            f.truncate()
            f.write(str(n + 1))
            f.flush()
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)
    return n


def write_job(job: Job) -> None:
    paths.ensure_dirs()
    final = job_path(job.id)
    tmp = final.parent / f".{job.id}.json.tmp"
    tmp.write_text(json.dumps(job.to_dict(), indent=2))
    os.replace(tmp, final)


def read_job(job_id: int) -> Job:
    return Job.from_dict(json.loads(job_path(job_id).read_text()))


def try_read_job(job_id: int) -> Optional[Job]:
    p = job_path(job_id)
    if not p.exists():
        return None
    try:
        return Job.from_dict(json.loads(p.read_text()))
    except (json.JSONDecodeError, TypeError):
        return None


def list_jobs() -> list[Job]:
    paths.ensure_dirs()
    jobs: list[Job] = []
    for p in sorted(paths.jobs_dir().glob("*.json")):
        try:
            jobs.append(Job.from_dict(json.loads(p.read_text())))
        except Exception:
            continue
    jobs.sort(key=lambda j: j.id)
    return jobs


def request_kill(job_id: int) -> None:
    paths.ensure_dirs()
    kill_path(job_id).touch()


def kill_requested(job_id: int) -> bool:
    return kill_path(job_id).exists()


def clear_kill(job_id: int) -> None:
    p = kill_path(job_id)
    if p.exists():
        p.unlink()
