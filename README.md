# gpuq

A CLI for submitting PyTorch training jobs to a small pool of GPU servers over SSH.

A **hub** machine runs `gpuq daemon`, which maintains a FIFO job queue and dispatches jobs to **workers** when GPUs are free. Workers need only SSH access, `tmux`, `uv`, `nvidia-smi`, and a shared network mount — no agent runs on them.

## Quick start (single-machine: hub is also a worker)

```bash
uv tool install --from git+https://github.com/EmreAdabag/gpuq.git gpuq
gpuq workers add localhost
gpuq daemon --foreground          # or run as systemd user unit, see below
# in another shell:
gpuq submit -- python train.py --config foo.yaml
gpuq ps
gpuq logs <id> -f
gpuq attach <id>                  # opens the tmux session over ssh
gpuq kill <id>
```

## Full setup

- [docs/HUB_SETUP.md](docs/HUB_SETUP.md) — preparing the hub: install, config, secrets, systemd
- [docs/WORKER_SETUP.md](docs/WORKER_SETUP.md) — preparing a worker: SSH, uv/tmux/nvidia-smi, shared mount, onboarding

## What "shared mount" means

`shared_mount` and `log_dir` in `config.yaml` point at a path the worker writes job stdout/stderr and the exit-code file to. If both hub and workers mount it at the same path (NFS, sshfs, etc.) the daemon reads it directly — fastest. If not, gpuq transparently falls back to SSH for reading the exit file (each tick of a finishing job) and for `gpuq logs <id>` (always SSH-tails to the worker). This means: **on a fresh worker with no admin access, you don't need to set up any shared filesystem** — gpuq will work over SSH for everything.

## State layout

```
~/.gpuq/
  config.yaml        # workers, paths, excludes
  secrets.env        # KEY=VALUE per line; mode 600; sourced before each job
  daemon.pid
  daemon.log
  state.lock
  next_id
  jobs/<id>.json     # one file per job; atomic writes
```

Override the location with `GPUQ_HOME=/some/path`.

## systemd (user unit)

```bash
mkdir -p ~/.config/systemd/user
cp packaging/gpuq-daemon.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now gpuq-daemon
loginctl enable-linger $USER       # so it survives logout
```

## Using a conda env (or any non-uv env)

Set `env_setup` per-worker in `config.yaml`. When present, gpuq skips `uv sync` and runs your command inside the activated env directly (no `uv run` wrapper):

```yaml
workers:
  - host: gpu1.lan
    user: me
    gpus: [0, 1]
    env_setup: |
      source ~/miniconda3/etc/profile.d/conda.sh
      conda activate myenv
```

Your env is your responsibility to keep up to date on each worker; gpuq doesn't manage conda envs.

## CLI

```
gpuq submit [--name NAME] [--host HOST] [--gpus N] -- <command...>
gpuq ps [--all]
gpuq logs <id> [-f]
gpuq attach <id>
gpuq kill <id>
gpuq workers                       # = workers list
gpuq workers add <host>
gpuq workers refresh
gpuq sync <host>                   # warm: rsync + uv sync, don't launch
gpuq daemon [--foreground] [-v]
```

## Tests

```bash
uv pip install -e '.[dev]'
pytest                              # 22 unit tests
GPUQ_INTEGRATION=1 pytest           # adds the localhost end-to-end test
```

## Design notes

- Two components, both on the hub: the CLI and the daemon. Workers run nothing.
- All state on disk under `~/.gpuq/`. One JSON file per job, atomic writes. No SQLite.
- Daemon is single-threaded, polls workers every ~2s with `nvidia-smi`.
- Each job runs in its own `tmux` session named `gpuq-<id>` on the worker, in a per-job repo checkout under `~/gpuq-repos/job-<id>/` (rsync'd with `--link-dest` so it's hardlink-cheap).
- `CUDA_VISIBLE_DEVICES` is set in the launcher script to the GPU indices the daemon picked; from inside the training process, GPUs are seen as `0..N-1`.
- A GPU is considered "free for gpuq" if it's in the worker's `gpus:` allowlist, isn't already assigned to a running gpuq job, and has `used_memory_mb < gpu_free_memory_threshold_mb`. The threshold keeps gpuq off GPUs humans are using.
- Kills of running jobs go via a `<id>.kill` flag file; the daemon picks it up on the next tick and `tmux kill-session`s.
- Daemon restart is safe — on startup it reconciles every `running` job by checking the remote tmux session and the exit file.
