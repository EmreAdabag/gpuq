from __future__ import annotations

from gpuq.config import WorkerConfig
from gpuq.jobs import Job
from gpuq.launch import build_launch_script


def _w(host: str = "gpu2.lan", env_setup: str | None = None) -> WorkerConfig:
    return WorkerConfig(host=host, user="me", gpus=[0, 1], env_setup=env_setup)


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
    script = build_launch_script(j, _w())
    assert "CUDA_VISIBLE_DEVICES=3,5" in script
    assert "GPUQ_JOB_ID=17" in script
    assert "GPUQ_HOST=gpu2.lan" in script
    assert "cd /home/me/gpuq-repos/job-17" in script
    assert "uv run python train.py --config foo.yaml" in script
    assert "tee -a /mnt/shared/gpuq-logs/17.log" in script
    assert "echo $? > /mnt/shared/gpuq-logs/17.exit" in script
    assert "source .gpuq-secrets" in script
    assert script.startswith("#!/usr/bin/env bash")


def test_launch_script_uses_env_setup_when_set() -> None:
    j = Job(
        id=4,
        name="conda-job",
        command="python train.py --config foo.json",
        gpus_assigned=[0],
        remote_repo_path="/home/me/gpuq-repos/job-4",
        log_path="/tmp/4.log",
        exit_path="/tmp/4.exit",
    )
    env = "source ~/miniconda3/etc/profile.d/conda.sh\nconda activate robomimic_venv"
    script = build_launch_script(j, _w(env_setup=env))
    # No `uv run` when env_setup is set.
    assert "uv run" not in script
    # The user's snippet is inlined.
    assert "conda activate robomimic_venv" in script
    # Command runs raw inside the env.
    assert "python train.py --config foo.json" in script


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
    script = build_launch_script(j, _w(host="host.example"))
    assert "cd '/tmp/repo with space'" in script
    assert "tee -a '/tmp/log with space/1.log'" in script
