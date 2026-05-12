# Worker setup

A gpuq worker is a GPU server reachable from the hub over SSH. **Nothing gpuq-specific runs on the worker** — the hub's daemon launches each job into a `tmux` session via SSH, and that's it.

This document is written so that another agent (or you) can SSH into a fresh worker and finish this checklist in order. After step 7 you go back to the hub and run `gpuq workers add <host>`.

## Inventory of what the hub assumes about the worker

| Thing | Why |
| --- | --- |
| Passwordless SSH from hub → worker as the configured user | daemon shells in every tick |
| `tmux` on PATH | training process runs inside `tmux new-session -d` so it survives SSH disconnects |
| `uv` on PATH for the SSH user | daemon runs `uv sync` and `uv run <cmd>` |
| `nvidia-smi` on PATH | daemon probes GPU memory every tick to pick a free GPU |
| A path matching `log_dir` in the hub's config, writeable by the worker user | the worker writes job stdout/stderr and the exit-code file there. If it's a real shared mount the hub reads them directly; otherwise gpuq SSH-tails / SSH-cats them from the worker (works transparently). |
| Writeable `remote_repo_base` (default `~/gpuq-repos/`) | per-job repo checkouts land here via rsync |
| User account with enough disk for repo checkouts (rsync uses `--link-dest` so they're cheap after the first) | per-job dirs are hardlink trees of the previous one |

## 1. SSH access

On the **hub**, copy the hub user's public key to the worker:

```bash
ssh-copy-id <worker-user>@<worker-host>
ssh -o BatchMode=yes <worker-user>@<worker-host> true && echo OK
```

If you don't have a hub keypair: `ssh-keygen -t ed25519 -N '' -f ~/.ssh/id_ed25519` first.

## 2. Install required binaries on the worker

```bash
ssh <worker-user>@<worker-host> '
  set -euo pipefail
  sudo apt update
  sudo apt install -y tmux rsync curl
  # uv installed as the same user gpuq will SSH in as:
  curl -LsSf https://astral.sh/uv/install.sh | sh
  # Make sure ~/.local/bin is on PATH for non-interactive ssh sessions.
  grep -q ".local/bin" ~/.bashrc || echo "export PATH=\$HOME/.local/bin:\$PATH" >> ~/.bashrc
'
```

Verify they're visible to a non-interactive SSH login (this is what the daemon uses):

```bash
ssh <worker-user>@<worker-host> 'command -v tmux uv nvidia-smi'
```

All three must print paths. If `uv` is missing, the most common cause is that `~/.local/bin` is not on PATH for non-interactive shells — fix by adding it to `~/.bashrc` (above) **and** `~/.profile` so it's picked up by `ssh host cmd`.

## 3. Verify the NVIDIA stack

```bash
ssh <worker-user>@<worker-host> 'nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader'
```

You should see one row per GPU. If `nvidia-smi` errors:
- Driver not installed: `sudo apt install nvidia-driver-XXX` (match the hub's CUDA version)
- Driver/kernel mismatch after kernel upgrade: reboot

## 4. Shared filesystem (optional)

The hub's `config.yaml` has `shared_mount: <path>`. **Same-path-on-both-sides is the spec's design but is not strictly required** — when the path isn't actually shared, gpuq automatically falls back to SSH for reading the job's exit file (during daemon reconcile) and tailing the log (via `gpuq logs`). This means:

- The path in `shared_mount` / `log_dir` only needs to exist on the **worker** side, writeable by the worker user. The hub never needs to write or read it directly when there's no shared mount.
- `gpuq logs <id>` runs `ssh <worker> tail …` under the hood. There's a small extra RTT per invocation, no big deal for log tailing.

If you DO want a real shared mount (one less SSH round-trip per tick per running job, simpler debugging), options in order of preference:

### NFS (production)

If the hub already exports a directory over NFS:

```bash
ssh <worker-user>@<worker-host> '
  sudo apt install -y nfs-common
  sudo mkdir -p <shared_mount>
  echo "<nfs-server>:<exported-path> <shared_mount> nfs defaults,_netdev 0 0" | sudo tee -a /etc/fstab
  sudo mount -a
  ls -ld <shared_mount>
'
```

### sshfs (quick/dev)

```bash
ssh <worker-user>@<worker-host> '
  sudo apt install -y sshfs
  mkdir -p <shared_mount>
  sshfs <hub-user>@<hub-host>:<shared_mount> <shared_mount>
'
```

(Add a systemd-mount unit if you want it persistent.)

### Single-machine

If the worker *is* the hub, the shared mount is just a local directory; no mounting required.

Verify both sides see the same file:

```bash
# on hub:
echo HUB > <shared_mount>/.gpuq-probe
# on worker:
ssh <worker-user>@<worker-host> 'cat <shared_mount>/.gpuq-probe'   # -> HUB
# cleanup:
rm <shared_mount>/.gpuq-probe
```

## 5. Make sure `remote_repo_base` exists

```bash
ssh <worker-user>@<worker-host> 'mkdir -p ~/gpuq-repos'
```

(The daemon `mkdir -p`s this too, but doing it upfront catches permission problems.)

## 6. Warm the worker (optional but fast)

This rsyncs the training repo and runs `uv sync` once so the first real job doesn't pay the cold-start cost:

```bash
# Run on the hub, after editing config.yaml's repo_root to point at your training repo:
gpuq sync <worker-host>
```

Note: `gpuq sync` requires the host to already be in `config.yaml`. If you'd rather warm before adding the worker, do steps 1–5 above, run `gpuq workers add <host>` (step 7), then `gpuq sync <host>`.

## 7. Onboard from the hub

```bash
gpuq workers add <worker-host>
```

This is interactive. It will:
1. Test passwordless SSH.
2. Run `nvidia-smi` and print the GPU table.
3. Check the shared mount.
4. Check that `tmux` and `uv` are present.
5. Ask which GPU indices gpuq may use (default: all). Type a comma list, e.g. `0,1,2,3` to allow gpuq on those GPUs and leave the others for humans/other tools.
6. Append the worker to `config.yaml`.

## 8. Verify end-to-end

From the hub:

```bash
gpuq workers                              # new host should appear with GPU mem stats
gpuq submit --host <worker-host> -- python -c "
import os
print('HOST:', os.uname().nodename)
print('CUDA_VISIBLE_DEVICES:', os.environ.get('CUDA_VISIBLE_DEVICES'))
print('GPUQ_JOB_ID:', os.environ.get('GPUQ_JOB_ID'))
"
gpuq ps
gpuq logs <id>
```

## Removing a worker

There is no `gpuq workers remove` in v0; just edit `~/.gpuq/config.yaml` and drop the entry. The daemon hot-reloads on each tick. Any jobs that were running on the removed host will get reconciled to `failed` on the next tick when the daemon notices the tmux session probe fails — clean them up manually if needed.

## Common failure modes

- **`gpuq submit` works, job goes `queued` -> stays `queued`.** The daemon can't find a free GPU. Check `gpuq workers` to see GPU mem usage; if it's above `gpu_free_memory_threshold_mb` (default 500), another process holds the GPU. Either wait, raise the threshold, or pick a different host with `--host`.
- **Job goes `queued` -> `failed` immediately, log shows `rsync failed`.** Likely passwordless SSH from hub to worker isn't set up for *the user gpuq is using*. Confirm with `ssh -o BatchMode=yes <user>@<host> true`.
- **Job runs but `uv: command not found` in the log.** `~/.local/bin` is not on PATH for non-interactive SSH on the worker. Add the export to `~/.bashrc` *and* `~/.profile`. Verify with `ssh <host> 'command -v uv'`.
- **Job's `.exit` file never appears, but tmux session is gone.** The launcher script crashed before reaching the `echo $? > <exit_path>` line. Check the log for the underlying error; common causes are an unwriteable `log_dir` on the shared mount or a syntax error in the user command.
- **`gpuq logs <id>` shows nothing while job is running.** The log file is on the shared mount — verify the mount is actually shared. Files written by the worker should be visible immediately on the hub side; if they aren't, the mount is local-only on one side.
