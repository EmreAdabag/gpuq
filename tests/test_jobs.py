from __future__ import annotations

import json
from pathlib import Path

from gpuq import jobs
from gpuq.jobs import Job


def test_next_id_increments(gpuq_home: Path) -> None:
    assert jobs.next_id() == 1
    assert jobs.next_id() == 2
    assert jobs.next_id() == 3


def test_write_and_read_job(gpuq_home: Path) -> None:
    j = Job(id=1, name="t", command="echo hi", status="queued", submitted_at="2026-05-12T00:00:00Z")
    jobs.write_job(j)
    back = jobs.read_job(1)
    assert back.id == 1
    assert back.name == "t"
    assert back.command == "echo hi"


def test_write_is_atomic(gpuq_home: Path) -> None:
    # The .tmp dot-prefixed file should not be visible in the listing.
    j = Job(id=5, name="x", command="x", submitted_at="z")
    jobs.write_job(j)
    listed = jobs.list_jobs()
    assert [x.id for x in listed] == [5]


def test_list_jobs_sorted(gpuq_home: Path) -> None:
    for i in [3, 1, 2]:
        jobs.write_job(Job(id=i, name=f"n{i}", command="c", submitted_at="z"))
    listed = jobs.list_jobs()
    assert [j.id for j in listed] == [1, 2, 3]


def test_kill_flag_roundtrip(gpuq_home: Path) -> None:
    assert not jobs.kill_requested(7)
    jobs.request_kill(7)
    assert jobs.kill_requested(7)
    jobs.clear_kill(7)
    assert not jobs.kill_requested(7)


def test_try_read_job_returns_none_for_missing(gpuq_home: Path) -> None:
    assert jobs.try_read_job(999) is None


def test_session_name_default() -> None:
    j = Job(id=42, name="x", command="c")
    assert j.session_name == "gpuq-42"
    j.tmux_session = "custom"
    assert j.session_name == "custom"


def test_json_shape_matches_spec(gpuq_home: Path) -> None:
    j = Job(id=1, name="t", command="c", submitted_at="z", gpus_assigned=[2, 3])
    jobs.write_job(j)
    raw = json.loads((gpuq_home / "jobs" / "1.json").read_text())
    for key in (
        "id", "name", "command", "status", "host_pin", "gpus_requested",
        "host", "gpus_assigned", "tmux_session", "remote_repo_path",
        "log_path", "exit_path", "git_commit", "submitted_at", "started_at",
        "ended_at", "exit_code",
    ):
        assert key in raw, f"missing field: {key}"
