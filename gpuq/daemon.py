from __future__ import annotations

import logging
import logging.handlers
import os
import signal
import sys
import time
from pathlib import Path
from typing import Optional

from . import jobs, launch, paths, workers
from .config import Config, WorkerConfig, load_config
from .jobs import Job, now_iso, state_lock


log = logging.getLogger("gpuq.daemon")


def setup_daemon_logging(verbose: bool = False, also_stderr: bool = True) -> None:
    paths.ensure_dirs()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    fh = logging.handlers.RotatingFileHandler(
        paths.daemon_log_path(), maxBytes=10 * 1024 * 1024, backupCount=3
    )
    fh.setFormatter(fmt)
    root = logging.getLogger()
    # Clear pre-existing handlers (e.g. left over from CLI invocation).
    root.handlers = [fh]
    if also_stderr:
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        root.addHandler(sh)
    root.setLevel(logging.DEBUG if verbose else logging.INFO)


def write_pidfile() -> None:
    p = paths.daemon_pid_path()
    if p.exists():
        try:
            existing = int(p.read_text().strip())
            os.kill(existing, 0)
        except (ProcessLookupError, ValueError):
            p.unlink(missing_ok=True)
        else:
            raise RuntimeError(f"Daemon already running with PID {existing}")
    p.write_text(str(os.getpid()))


def remove_pidfile() -> None:
    paths.daemon_pid_path().unlink(missing_ok=True)


def daemonize() -> None:
    if os.fork() > 0:
        os._exit(0)
    os.setsid()
    if os.fork() > 0:
        os._exit(0)
    os.chdir("/")
    devnull = os.open(os.devnull, os.O_RDWR)
    for fd in (0, 1, 2):
        try:
            os.dup2(devnull, fd)
        except OSError:
            pass


def reconcile_running_jobs(cfg: Config) -> None:
    for job in jobs.list_jobs():
        if job.status != "running":
            continue
        worker = cfg.worker(job.host or "")
        if worker is None:
            log.warning("running job %d on unknown host %s; marking failed", job.id, job.host)
            with state_lock():
                job.status = "failed"
                job.ended_at = now_iso()
                jobs.write_job(job)
            continue
        try:
            alive = workers.tmux_has_session(worker, job.session_name)
        except Exception as e:
            log.warning("tmux check failed for job %d on %s: %s", job.id, worker.host, e)
            continue
        if alive:
            continue
        # Try local path first (real shared mount); fall back to SSH if the
        # path isn't visible locally — supports hubs without a shared mount.
        exists, code = _read_exit_file(job, worker)
        with state_lock():
            job.exit_code = code
            if not exists:
                job.status = "failed"
            else:
                job.status = "done" if code == 0 else "failed"
            job.ended_at = now_iso()
            jobs.write_job(job)
        log.info(
            "Reconciled job %d -> %s (exit=%s, file_present=%s)",
            job.id, job.status, code, exists,
        )


def _read_exit_file(job: Job, worker: WorkerConfig) -> tuple[bool, Optional[int]]:
    p = Path(job.exit_path) if job.exit_path else None
    if p and p.exists():
        try:
            return True, int(p.read_text().strip())
        except (ValueError, OSError):
            return True, None
    if not job.exit_path:
        return False, None
    try:
        return workers.remote_cat_int(worker, job.exit_path)
    except Exception as e:
        log.warning("remote exit-file read failed for job %d: %s", job.id, e)
        return False, None


def _used_by_gpuq_on(host: str, all_jobs: list[Job]) -> set[int]:
    used: set[int] = set()
    for j in all_jobs:
        if j.status == "running" and j.host == host:
            used.update(j.gpus_assigned)
    return used


def available_gpus_for(
    cfg: Config,
    worker: WorkerConfig,
    all_jobs: list[Job],
    tentative: dict[str, set[int]],
) -> list[int]:
    try:
        stats = workers.probe_gpus(worker)
    except Exception as e:
        log.warning("probe_gpus failed on %s: %s", worker.host, e)
        return []
    by_index = {s.index: s.used_mb for s in stats}
    used = _used_by_gpuq_on(worker.host, all_jobs) | tentative.get(worker.host, set())
    return workers.select_available_gpus(
        worker.gpus, by_index, used, cfg.gpu_free_memory_threshold_mb
    )


def dispatch_once(cfg: Config) -> None:
    all_jobs = jobs.list_jobs()
    queued = [j for j in all_jobs if j.status == "queued"]
    if not queued:
        return

    tentative: dict[str, set[int]] = {}
    avail_cache: dict[str, list[int]] = {}
    for w in cfg.workers:
        avail_cache[w.host] = available_gpus_for(cfg, w, all_jobs, tentative)

    for job in queued:
        if jobs.kill_requested(job.id):
            continue
        candidates = cfg.workers
        if job.host_pin:
            candidates = [w for w in cfg.workers if w.host == job.host_pin]
        chosen: Optional[WorkerConfig] = None
        chosen_gpus: list[int] = []
        for w in candidates:
            avail = avail_cache.get(w.host, [])
            if len(avail) >= job.gpus_requested:
                chosen = w
                chosen_gpus = sorted(avail)[: job.gpus_requested]
                break
        if not chosen:
            continue
        job.gpus_assigned = chosen_gpus
        avail_cache[chosen.host] = [g for g in avail_cache[chosen.host] if g not in chosen_gpus]
        tentative.setdefault(chosen.host, set()).update(chosen_gpus)
        try:
            launch.launch_job(cfg, chosen, job)
        except Exception as e:
            log.exception("launch_job crashed for job %d: %s", job.id, e)
            with state_lock():
                job.status = "failed"
                job.ended_at = now_iso()
                jobs.write_job(job)


def handle_kills(cfg: Config) -> None:
    for job in jobs.list_jobs():
        if not jobs.kill_requested(job.id):
            continue
        if job.status == "running":
            worker = cfg.worker(job.host or "")
            if worker is not None:
                try:
                    workers.tmux_kill_session(worker, job.session_name)
                except Exception as e:
                    log.warning("tmux_kill_session for job %d: %s", job.id, e)
            with state_lock():
                job.status = "killed"
                job.ended_at = now_iso()
                jobs.write_job(job)
            log.info("Killed running job %d", job.id)
        jobs.clear_kill(job.id)


def tick(cfg: Config) -> None:
    reconcile_running_jobs(cfg)
    handle_kills(cfg)
    dispatch_once(cfg)


_stop = False


def _signal_stop(signum, frame):  # noqa: ARG001
    global _stop
    _stop = True
    log.info("Received signal %s; stopping", signum)


def run_daemon(foreground: bool = False, verbose: bool = False) -> None:
    if not foreground:
        daemonize()
    setup_daemon_logging(verbose=verbose, also_stderr=foreground)
    try:
        write_pidfile()
    except RuntimeError as e:
        log.error("%s", e)
        sys.exit(1)
    signal.signal(signal.SIGTERM, _signal_stop)
    signal.signal(signal.SIGINT, _signal_stop)
    try:
        cfg = load_config()
    except Exception as e:
        log.error("Failed to load config: %s", e)
        remove_pidfile()
        sys.exit(1)
    log.info(
        "Daemon started; tick=%ss workers=%s",
        cfg.daemon_tick_seconds,
        [w.host for w in cfg.workers],
    )
    try:
        reconcile_running_jobs(cfg)
    except Exception as e:
        log.exception("startup reconcile failed: %s", e)
    while not _stop:
        try:
            cfg = load_config()
        except Exception as e:
            log.error("config reload failed; keeping last good: %s", e)
        try:
            tick(cfg)
        except Exception as e:
            log.exception("tick failed: %s", e)
        for _ in range(cfg.daemon_tick_seconds * 10):
            if _stop:
                break
            time.sleep(0.1)
    log.info("Daemon stopped")
    remove_pidfile()
