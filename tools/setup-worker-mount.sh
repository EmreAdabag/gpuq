#!/usr/bin/env bash
# Set up a persistent sshfs mount from a gpuq worker back to this hub.
#
# Usage:
#   tools/setup-worker-mount.sh <worker-host> <abs-path> [--rw]
#
# The path is mounted at the SAME absolute path on the worker as it lives at
# on the hub (gpuq's convention). Read-only by default; pass --rw to allow
# writes (use sparingly -- a `rm -rf` on the worker side of a RW mount
# propagates the deletes to the hub).
#
# What this script does, idempotently:
#   1. Generate an ssh keypair on the worker if missing, authorize it on this hub.
#   2. Verify the hub's sftp subsystem is enabled (sshfs requires it).
#   3. Install sshfs on the worker (sudo apt-get install -y sshfs).
#   4. Enable systemd user linger on the worker so the mount survives reboot.
#   5. Write + enable + start a `systemd --user` unit for the mount.
#
# Safe to re-run. Prompts for the worker's sudo password where needed.

set -euo pipefail

print_usage() {
  cat <<'EOF' >&2
Usage: setup-worker-mount.sh <worker-host> <abs-path> [--rw]

  worker-host   ssh-reachable hostname of the gpuq worker
  abs-path      absolute path on this hub to mirror on the worker (same path on both sides)
  --rw          mount read-write (default is read-only for safety)

Example -- read-only code/dataset mount:
  ./tools/setup-worker-mount.sh iotwo.engin.umich.edu /home/me/code/robomimic

Example -- a writeable output dir:
  ./tools/setup-worker-mount.sh iotwo.engin.umich.edu /home/me/code/robomimic/diffusion_policy_trained_models --rw
EOF
  exit 1
}

[ $# -ge 2 ] || print_usage
WORKER="$1"
ABS_PATH="$2"
MODE="ro"
case "${3:-}" in
  --rw) MODE="rw" ;;
  --ro|"") MODE="ro" ;;
  *) echo "[error] unknown flag: $3" >&2; print_usage ;;
esac

HUB_HOST="$(hostname -f 2>/dev/null || hostname)"
HUB_USER="${USER}"
# systemd unit name: sshfs-<slugified path>.service
SVC_NAME="sshfs$(echo "$ABS_PATH" | tr '/' '-')"

# Sanity: path must be absolute
[[ "$ABS_PATH" = /* ]] || { echo "[error] <abs-path> must be absolute"; exit 1; }

echo "Plan:"
echo "  worker:  ${WORKER}"
echo "  source:  ${HUB_USER}@${HUB_HOST}:${ABS_PATH}"
echo "  target:  ${ABS_PATH} (on worker)"
echo "  mode:    ${MODE}"
echo "  service: ${SVC_NAME}.service (systemd --user on worker)"
echo

# ---------------------------------------------------------------------------
# 1. Reverse ssh key: worker can ssh to hub passwordlessly.
# ---------------------------------------------------------------------------
echo "==> 1/5  worker→hub passwordless ssh"
mkdir -p ~/.ssh && chmod 700 ~/.ssh
touch ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys

WORKER_PUBKEY="$(ssh "$WORKER" '
  set -e
  if [ ! -f ~/.ssh/id_ed25519 ] && [ ! -f ~/.ssh/id_rsa ]; then
    ssh-keygen -t ed25519 -N "" -f ~/.ssh/id_ed25519 -C "gpuq-worker-$(hostname)" >/dev/null
  fi
  cat ~/.ssh/id_ed25519.pub 2>/dev/null || cat ~/.ssh/id_rsa.pub
' | tail -1)"

# Match on the key body (field 2), ignoring whitespace/comments, for idempotency.
KEY_BODY="$(echo "$WORKER_PUBKEY" | awk '{print $2}')"
if ! grep -qF "$KEY_BODY" ~/.ssh/authorized_keys 2>/dev/null; then
  echo "$WORKER_PUBKEY" >> ~/.ssh/authorized_keys
  echo "    + authorized worker key on hub"
else
  echo "    = worker key already authorized"
fi

if ! ssh "$WORKER" "ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new ${HUB_USER}@${HUB_HOST} true" >/dev/null 2>&1; then
  cat <<EOF >&2
    [error] worker still cannot ssh to hub.

    Possible causes:
      * hub's sshd refuses key auth (check /etc/ssh/sshd_config for PubkeyAuthentication yes)
      * the worker's $HUB_USER user can't reach this host at $HUB_HOST
EOF
  exit 2
fi
echo "    ✓ worker can ssh to hub"

# ---------------------------------------------------------------------------
# 2. sftp subsystem on the hub.
# ---------------------------------------------------------------------------
echo "==> 2/5  sftp subsystem on hub"
if ! echo ls | sftp -o BatchMode=yes localhost >/dev/null 2>&1; then
  cat <<EOF >&2
    [error] Hub's sshd does not expose the sftp subsystem; sshfs needs it.
    Run this once on the hub (needs sudo):

      sudo bash -c 'echo "Subsystem sftp /usr/lib/openssh/sftp-server" >> /etc/ssh/sshd_config && systemctl reload ssh'

    Then re-run this script.
EOF
  exit 2
fi
echo "    ✓ sftp subsystem ok"

# ---------------------------------------------------------------------------
# 3. sshfs installed on the worker.
# ---------------------------------------------------------------------------
echo "==> 3/5  sshfs installed on worker"
if ! ssh "$WORKER" 'command -v sshfs' >/dev/null 2>&1; then
  echo "    sshfs missing; running sudo apt-get install -y sshfs (will prompt for sudo password)"
  ssh -t "$WORKER" 'sudo apt-get update -qq && sudo apt-get install -y sshfs'
fi
ssh "$WORKER" 'command -v sshfs' >/dev/null
echo "    ✓ sshfs installed"

# ---------------------------------------------------------------------------
# 4. systemd user linger on worker.
# ---------------------------------------------------------------------------
echo "==> 4/5  systemd user linger on worker"
LINGER="$(ssh "$WORKER" 'loginctl show-user $USER --property=Linger 2>/dev/null' || true)"
if echo "$LINGER" | grep -qE 'Linger=yes'; then
  echo "    = linger already enabled"
else
  echo "    enabling linger (will prompt for sudo password)"
  if ssh -t "$WORKER" 'sudo loginctl enable-linger $USER' >/dev/null 2>&1; then
    echo "    ✓ linger enabled (mount will come up at boot, even before login)"
  else
    echo "    [warn] could not enable linger; mount will only come back after you ssh in"
  fi
fi

# ---------------------------------------------------------------------------
# 5. Systemd user unit for the mount.
# ---------------------------------------------------------------------------
echo "==> 5/5  systemd user mount unit"
SSHFS_OPTS="reconnect,ServerAliveInterval=15,ServerAliveCountMax=3"
[ "$MODE" = "ro" ] && SSHFS_OPTS="$SSHFS_OPTS,ro"

UNIT_BODY="$(cat <<EOF
[Unit]
Description=gpuq sshfs mount ${ABS_PATH} (${MODE}) <- ${HUB_USER}@${HUB_HOST}
After=network-online.target
Wants=network-online.target

[Service]
Type=forking
ExecStartPre=/bin/mkdir -p ${ABS_PATH}
ExecStart=/usr/bin/sshfs ${HUB_USER}@${HUB_HOST}:${ABS_PATH} ${ABS_PATH} -o ${SSHFS_OPTS}
ExecStop=/bin/fusermount -u ${ABS_PATH}
Restart=on-failure
RestartSec=10

[Install]
WantedBy=default.target
EOF
)"

ssh "$WORKER" "
  set -e
  mkdir -p \$HOME/.config/systemd/user
  cat > \$HOME/.config/systemd/user/${SVC_NAME}.service <<'__GPUQ_UNIT__'
${UNIT_BODY}
__GPUQ_UNIT__
  # If a previous run of this script for the same path is mounted but unmanaged,
  # tear it down so systemd can become the owner.
  fusermount -u ${ABS_PATH} 2>/dev/null || true
  systemctl --user daemon-reload
  systemctl --user enable --now ${SVC_NAME}.service
"

sleep 1
if ssh "$WORKER" "mount | grep -qE 'on ${ABS_PATH} '"; then
  echo "    ✓ mounted at ${ABS_PATH} (${MODE}) on ${WORKER}"
else
  echo "    [error] mount didn't show up. Inspect:"
  echo "      ssh ${WORKER} systemctl --user status ${SVC_NAME}.service"
  exit 2
fi

echo
echo "Done."
[ "$MODE" = "ro" ] && echo "  Mode: read-only. Writes from the worker will fail with EROFS, so accidental rm can't propagate."
[ "$MODE" = "rw" ] && echo "  Mode: read-write. Deletes on the worker side propagate to the hub. Tread carefully."
echo "  To take the mount down: ssh ${WORKER} systemctl --user disable --now ${SVC_NAME}.service"
