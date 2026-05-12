# gpuq

A CLI for submitting PyTorch training jobs to a small pool of GPU servers over SSH.

A hub machine runs `gpuq daemon`, which maintains a job queue and dispatches jobs to workers when GPUs are free. Workers need only SSH access, `tmux`, `uv`, `nvidia-smi`, and a shared network mount — no agent runs on them.

## Install

```
uv pip install -e .
```

## Quick start

```
gpuq workers add localhost
gpuq daemon --foreground          # or run via systemd, see packaging/
gpuq submit -- python train.py --config foo.yaml
gpuq ps
gpuq logs <id> -f
gpuq attach <id>
gpuq kill <id>
```

State lives under `~/.gpuq/` (override with `GPUQ_HOME`). Config is `~/.gpuq/config.yaml`; secrets go in `~/.gpuq/secrets.env` (KEY=VALUE per line, mode 600) and are sourced into the job environment before exec.

## systemd

```
mkdir -p ~/.config/systemd/user
cp packaging/gpuq-daemon.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now gpuq-daemon
```

## Tests

```
uv pip install -e '.[dev]'
pytest                          # unit tests
GPUQ_INTEGRATION=1 pytest       # adds the localhost end-to-end test
```
