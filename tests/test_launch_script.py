from __future__ import annotations

from gpuq.jobs import Job
from gpuq.launch import build_launch_script


def test_launch_script_sets_env_and_paths() -> None:
    j = Job(
        id=17,
        name="myrun",
        command="python train.py --config foo.yaml",
        gpus_assigned=[3, 5],
        remote_repo_path="/home/me/gpuq-repos/job-17",
        log_path="/mnt/shared/gpuq-logs/17.log",
        exit_path="/mnt/shared/gpuq-logs/17.exit",
    )
    script = build_launch_script(j, "gpu2.lan")
    # shlex.quote leaves shell-safe strings bare (alphanumerics + @%+=:,./-).
    assert "CUDA_VISIBLE_DEVICES=3,5" in script
    assert "GPUQ_JOB_ID=17" in script
    assert "GPUQ_HOST=gpu2.lan" in script
    assert "cd /home/me/gpuq-repos/job-17" in script
    assert "uv run python train.py --config foo.yaml" in script
    assert "tee -a /mnt/shared/gpuq-logs/17.log" in script
    assert "echo $? > /mnt/shared/gpuq-logs/17.exit" in script
    assert "source .gpuq-secrets" in script
    assert script.startswith("#!/usr/bin/env bash")


def test_launch_script_quotes_paths_with_spaces() -> None:
    j = Job(
        id=1,
        name="x",
        command="echo hi",
        gpus_assigned=[0],
        remote_repo_path="/tmp/repo with space",
        log_path="/tmp/log with space/1.log",
        exit_path="/tmp/log with space/1.exit",
    )
    script = build_launch_script(j, "host.example")
    assert "cd '/tmp/repo with space'" in script
    assert "tee -a '/tmp/log with space/1.log'" in script
