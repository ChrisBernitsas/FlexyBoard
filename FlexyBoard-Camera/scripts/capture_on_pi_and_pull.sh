#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  capture_on_pi_and_pull.sh [before|after|both|probe]

Description:
  Runs capture commands on the Raspberry Pi and automatically copies resulting
  images to local FlexyBoard-Camera/debug_output with timestamped names.

Environment overrides:
  PI_HOST   (default: flexyboard-pi.local)
  PI_USER   (default: pi)
  PI_REPO   (default: /home/pi/FlexyBoard-Camera)
  CONFIG    (default: configs/default.yaml)
  PI_PASSWORD (optional; requires sshpass for password automation)
EOF
}

MODE="${1:-before}"
if [[ "${MODE}" == "-h" || "${MODE}" == "--help" ]]; then
  usage
  exit 0
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOCAL_DEBUG_DIR="${ROOT_DIR}/debug_output"

PI_HOST="${PI_HOST:-flexyboard-pi.local}"
PI_USER="${PI_USER:-pi}"
PI_REPO="${PI_REPO:-/home/pi/FlexyBoard-Camera}"
CONFIG="${CONFIG:-configs/default.yaml}"

REMOTE_DEBUG_DIR="${PI_REPO}/debug_output"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

mkdir -p "${LOCAL_DEBUG_DIR}"

CONTROL_PATH="${HOME}/.ssh/flexyboard-%r@%h:%p"
SSH_OPTS=(
  -o StrictHostKeyChecking=accept-new
  -o ControlMaster=auto
  -o ControlPath="${CONTROL_PATH}"
  -o ControlPersist=5m
)

use_sshpass=false
if [[ -n "${PI_PASSWORD:-}" ]]; then
  if ! command -v sshpass >/dev/null 2>&1; then
    echo "PI_PASSWORD is set but 'sshpass' is not installed."
    echo "Install sshpass (for example: brew install hudochenkov/sshpass/sshpass),"
    echo "or unset PI_PASSWORD and use SSH key auth."
    exit 1
  fi
  use_sshpass=true
fi

ssh_exec() {
  if [[ "${use_sshpass}" == "true" ]]; then
    sshpass -p "${PI_PASSWORD}" ssh "${SSH_OPTS[@]}" "$@"
  else
    ssh "${SSH_OPTS[@]}" "$@"
  fi
}

scp_exec() {
  if [[ "${use_sshpass}" == "true" ]]; then
    sshpass -p "${PI_PASSWORD}" scp "${SSH_OPTS[@]}" "$@"
  else
    scp "${SSH_OPTS[@]}" "$@"
  fi
}

run_on_pi() {
  local cmd="$1"
  ssh_exec "${PI_USER}@${PI_HOST}" "cd '${PI_REPO}' && source .venv/bin/activate && ${cmd}"
}

pull_remote_file() {
  local remote_name="$1"
  local local_name="$2"
  scp_exec "${PI_USER}@${PI_HOST}:${REMOTE_DEBUG_DIR}/${remote_name}" "${LOCAL_DEBUG_DIR}/${local_name}" >/dev/null
  echo "Saved: ${LOCAL_DEBUG_DIR}/${local_name}"
}

capture_before() {
  run_on_pi "python -m flexyboard_camera.app.main --config '${CONFIG}' capture_before"
  pull_remote_file "before_latest.png" "before_${TIMESTAMP}.png"
}

capture_after() {
  run_on_pi "python -m flexyboard_camera.app.main --config '${CONFIG}' capture_after"
  pull_remote_file "after_latest.png" "after_${TIMESTAMP}.png"
}

capture_probe() {
  run_on_pi "python scripts/camera_probe.py --max-index 1 --width 1920 --height 1080 --snapshot-index 0 --snapshot-out debug_output/camera_probe_snapshot.png"
  pull_remote_file "camera_probe_snapshot.png" "probe_${TIMESTAMP}.png"
}

case "${MODE}" in
  before)
    capture_before
    ;;
  after)
    capture_after
    ;;
  both)
    capture_before
    read -r -p "Move piece on board, then press Enter to capture AFTER..."
    capture_after
    ;;
  probe)
    capture_probe
    ;;
  *)
    echo "Unknown mode: ${MODE}"
    usage
    exit 2
    ;;
esac
