# Hub setup

The hub is the machine that:
- holds `~/.gpuq/` (config, secrets, job state, daemon log)
- runs `gpuq daemon` (long-lived; dispatches jobs to workers over SSH)
- is where users run `gpuq submit ...`, `gpuq ps`, `gpuq logs`, etc.

The hub *can also* be a worker — just list `localhost` in the workers list. The hub in this repo is already configured this way; the steps below are what a fresh agent should do on a new hub.

## Prerequisites on the hub

```bash
sudo apt install -y tmux rsync openssh-client
# uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh
# Ensure ~/.local/bin is on PATH for both interactive and systemd-user contexts.
```

## 1. Install gpuq

```bash
uv tool install --from git+https://github.com/EmreAdabag/gpuq.git gpuq
which gpuq   # -> ~/.local/bin/gpuq
```

For development clones:

```bash
git clone git@github.com:EmreAdabag/gpuq.git
cd gpuq
uv tool install --from . gpuq      # or: uv pip install -e .
```

## 2. Make sure the hub can SSH to itself

Even if hub == worker, gpuq uses SSH/rsync/scp uniformly. Confirm:

```bash
ssh -o BatchMode=yes localhost true && echo OK
```

If that prompts for a password, run:

```bash
ssh-keygen -t ed25519 -N '' -f ~/.ssh/id_ed25519   # if you don't have a key
ssh-copy-id localhost
```

## 3. Initialize state

```bash
mkdir -p ~/.gpuq ~/gpuq-repos ~/gpuq-shared/gpuq-logs
touch ~/.gpuq/secrets.env
chmod 600 ~/.gpuq/secrets.env
```

## 4. Write the config

`~/.gpuq/config.yaml`:

```yaml
workers:
  - host: localhost
    user: <your-user>
    gpus: [0, 1]                    # GPU indices gpuq may use

repo_root: ~/code/training          # YOUR training repo (gets rsync'd to workers)
remote_repo_base: ~/gpuq-repos
shared_mount: /home/<your-user>/gpuq-shared
log_dir:      /home/<your-user>/gpuq-shared/gpuq-logs
secrets_file: /home/<your-user>/.gpuq/secrets.env

rsync_excludes: [.git, __pycache__, "*.pyc", .venv, wandb/, outputs/]
daemon_tick_seconds: 2
gpu_free_memory_threshold_mb: 500
```

Critical fields:
- **`repo_root`** must point at a uv-managed project (i.e. it has a `pyproject.toml`); the daemon runs `uv sync` on the remote copy before launching.
- **`shared_mount`** must be the *same absolute path* on the hub and every worker, and writeable by all of them. On single-machine setups, any local directory works.
- **`gpus`** is gpuq's allowlist for that host — gpuq will not touch any GPU outside it, leaving humans free to use the rest.

## 5. Fill in secrets

`~/.gpuq/secrets.env` (mode 600):

```
WANDB_API_KEY=...
HF_TOKEN=...
```

These are pushed to the worker via `scp` before each job and `source`d in the launcher script. They are visible in the job's environment.

## 6. Start the daemon

Option A — systemd user unit (recommended for long-lived hubs):

```bash
mkdir -p ~/.config/systemd/user
cp packaging/gpuq-daemon.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now gpuq-daemon
loginctl enable-linger $USER    # so it survives logout
systemctl --user --no-pager status gpuq-daemon
tail -f ~/.gpuq/daemon.log
```

Option B — manually for debugging:

```bash
gpuq daemon --foreground -v
```

## 7. Verify

```bash
gpuq workers                                                    # see GPU mem stats
gpuq submit -- python -c "import os; print(os.environ['CUDA_VISIBLE_DEVICES'])"
gpuq ps
gpuq logs <id> -f
```

## State layout (on the hub)

```
~/.gpuq/
  config.yaml          # this file
  secrets.env          # 600
  daemon.pid
  daemon.log           # rotating, 10MB x 3
  state.lock           # used by both CLI and daemon under fcntl.flock
  next_id              # plain int
  jobs/
    1.json
    2.json
    ...
    7.kill             # transient — written by `gpuq kill`, removed by daemon
```

`~/.gpuq/` is overridable with `GPUQ_HOME=/some/path`.

## Adding a worker

```bash
gpuq workers add <host>
```

Walks through SSH/uv/tmux/nvidia-smi/shared-mount checks, asks which GPU indices gpuq may use, and appends to `config.yaml`. See [WORKER_SETUP.md](WORKER_SETUP.md) for what to prep on the worker side first.
