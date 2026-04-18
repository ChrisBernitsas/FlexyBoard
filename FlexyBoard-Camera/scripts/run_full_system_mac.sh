#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  ./scripts/run_full_system_mac.sh

Description:
  Starts the full hardware workflow from your Mac:
    1. opens a Terminal window running the Raspberry Pi bridge
    2. opens a Terminal window running the local Software-GUI

  The Pi bridge uses rolling capture mode:
    - one initial reference image at startup
    - one new image after each Player 1 turn
    - one refreshed reference image after the STM32 finishes Player 2 movement

Environment overrides:
  PI_HOST          default: flexyboard-pi.local
  PI_USER          default: pi
  PI_PASSWORD      default: flexyboard; requires sshpass for automatic password entry.
                   Set PI_PASSWORD='' to disable password automation and use SSH keys/manual prompts.
  PI_CAMERA_REPO   default: /home/pi/FlexyBoard-Camera
  PI_MOTOR_REPO    default: /home/pi/FlexyBoard-Motor-Control
  BRIDGE_PORT      default: 8765
  SYNC             default: 1, set to 0 to skip syncing FlexyBoard-Camera to Pi
  SYNC_MOTOR       default: same as SYNC, set to 0 to skip syncing FlexyBoard-Motor-Control
  FLASH_STM32      default: 0, set to 1 to build/flash STM32 before launching bridge
  BRIDGE_ARGS      optional extra args for run_pi_software_bridge.py

Examples:
  ./scripts/run_full_system_mac.sh
  FLASH_STM32=1 ./scripts/run_full_system_mac.sh
  SYNC=0 ./scripts/run_full_system_mac.sh
  BRIDGE_ARGS="--once --no-stm-send" ./scripts/run_full_system_mac.sh
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi
if [[ -n "${1:-}" ]]; then
  echo "Unknown argument: $1"
  usage
  exit 2
fi

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "This launcher uses macOS Terminal.app. Run the Pi bridge and GUI manually on non-macOS systems."
  exit 1
fi

CAMERA_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKSPACE_DIR="$(cd "${CAMERA_DIR}/.." && pwd)"
GUI_DIR="${WORKSPACE_DIR}/Software-GUI"
MOTOR_DIR="${WORKSPACE_DIR}/FlexyBoard-Motor-Control"

PI_HOST="${PI_HOST:-flexyboard-pi.local}"
PI_USER="${PI_USER:-pi}"
PI_PASSWORD="${PI_PASSWORD-flexyboard}"
PI_CAMERA_REPO="${PI_CAMERA_REPO:-/home/pi/FlexyBoard-Camera}"
PI_MOTOR_REPO="${PI_MOTOR_REPO:-/home/pi/FlexyBoard-Motor-Control}"
BRIDGE_PORT="${BRIDGE_PORT:-8765}"
SYNC="${SYNC:-1}"
SYNC_MOTOR="${SYNC_MOTOR:-${SYNC}}"
FLASH_STM32="${FLASH_STM32:-0}"
BRIDGE_ARGS="${BRIDGE_ARGS:-}"

shell_quote() {
  printf "%q" "$1"
}

kill_remote_port_process() {
  local port="$1"
  local pids_to_kill=""

  echo "Checking for processes using port ${port} on ${PI_USER}@${PI_HOST}..."
  if [[ -n "${PI_PASSWORD}" ]]; then
    pids_to_kill=$(SSHPASS="${PI_PASSWORD}" sshpass -e ssh "${PI_USER}@${PI_HOST}" "lsof -t -i :${port}" 2>/dev/null || true)
  else
    pids_to_kill=$(ssh "${PI_USER}@${PI_HOST}" "lsof -t -i :${port}" 2>/dev/null || true)
  fi

  if [[ -n "${pids_to_kill}" ]]; then
    echo "Found processes using port ${port}: ${pids_to_kill}. Killing them..."
    for pid in ${pids_to_kill}; do
      if [[ -n "${PI_PASSWORD}" ]]; then
        SSHPASS="${PI_PASSWORD}" sshpass -e ssh "${PI_USER}@${PI_HOST}" "kill ${pid}" || true
      else
        ssh "${PI_USER}@${PI_HOST}" "kill ${pid}" || true
      fi
      echo "Process ${pid} killed."
    done
  else
    echo "No process found using port ${port}."
  fi
}

if [[ ! -d "${GUI_DIR}" ]]; then
  echo "Missing Software-GUI folder at: ${GUI_DIR}"
  exit 1
fi

if [[ ! -d "${MOTOR_DIR}" ]]; then
  echo "Missing FlexyBoard-Motor-Control folder at: ${MOTOR_DIR}"
  exit 1
fi

if [[ -n "${PI_PASSWORD}" ]] && ! command -v sshpass >/dev/null 2>&1; then
  cat >&2 <<'ERROR'
PI_PASSWORD was provided, but sshpass is not installed.

Recommended: install sshpass, or set up an SSH key once and run with PI_PASSWORD=''.
Alternative: install sshpass, then rerun:
  ./scripts/run_full_system_mac.sh
ERROR
  exit 1
fi

if [[ "${SYNC}" != "0" ]]; then
  echo "Syncing FlexyBoard-Camera to ${PI_USER}@${PI_HOST}:${PI_CAMERA_REPO}/ ..."
  if [[ -n "${PI_PASSWORD}" ]]; then
    SSHPASS="${PI_PASSWORD}" sshpass -e rsync -az \
      --exclude ".git/" \
      --exclude ".venv/" \
      --exclude "__pycache__/" \
      --exclude ".pytest_cache/" \
      --exclude "debug_output/" \
      --exclude "logs/" \
      "${CAMERA_DIR}/" "${PI_USER}@${PI_HOST}:${PI_CAMERA_REPO}/"
  else
    rsync -az \
      --exclude ".git/" \
      --exclude ".venv/" \
      --exclude "__pycache__/" \
      --exclude ".pytest_cache/" \
      --exclude "debug_output/" \
      --exclude "logs/" \
      "${CAMERA_DIR}/" "${PI_USER}@${PI_HOST}:${PI_CAMERA_REPO}/"
  fi
fi

if [[ "${SYNC_MOTOR}" != "0" ]]; then
  echo "Syncing FlexyBoard-Motor-Control to ${PI_USER}@${PI_HOST}:${PI_MOTOR_REPO}/ ..."
  if [[ -n "${PI_PASSWORD}" ]]; then
    SSHPASS="${PI_PASSWORD}" sshpass -e rsync -az --delete \
      --exclude ".git/" \
      --exclude "Debug/" \
      --exclude "build/" \
      "${MOTOR_DIR}/" "${PI_USER}@${PI_HOST}:${PI_MOTOR_REPO}/"
  else
    rsync -az --delete \
      --exclude ".git/" \
      --exclude "Debug/" \
      --exclude "build/" \
      "${MOTOR_DIR}/" "${PI_USER}@${PI_HOST}:${PI_MOTOR_REPO}/"
  fi
fi

if [[ "${FLASH_STM32}" == "1" ]]; then
  echo "Building and flashing STM32 firmware from ${PI_MOTOR_REPO} ..."
  REMOTE_FLASH_CMD="cd '${PI_MOTOR_REPO}' && chmod +x scripts/*.sh && ./scripts/build_and_flash.sh"
  if [[ -n "${PI_PASSWORD}" ]]; then
    SSHPASS="${PI_PASSWORD}" sshpass -e ssh -tt "${PI_USER}@${PI_HOST}" "${REMOTE_FLASH_CMD}"
  else
    ssh -tt "${PI_USER}@${PI_HOST}" "${REMOTE_FLASH_CMD}"
  fi
elif [[ "${FLASH_STM32}" != "0" ]]; then
  echo "FLASH_STM32 must be 0 or 1, got: ${FLASH_STM32}"
  exit 2
fi

kill_remote_port_process "${BRIDGE_PORT}"

REMOTE_BRIDGE_CMD="cd '${PI_CAMERA_REPO}' && source .venv/bin/activate && python3 scripts/run_pi_software_bridge.py --capture-mode rolling --port '${BRIDGE_PORT}' ${BRIDGE_ARGS}"
if [[ -n "${PI_PASSWORD}" ]]; then
  BRIDGE_TERMINAL_CMD="SSHPASS=$(shell_quote "${PI_PASSWORD}") sshpass -e ssh -tt '${PI_USER}@${PI_HOST}' \"${REMOTE_BRIDGE_CMD}\""
else
  BRIDGE_TERMINAL_CMD="ssh -tt '${PI_USER}@${PI_HOST}' \"${REMOTE_BRIDGE_CMD}\""
fi
GUI_TERMINAL_CMD="cd '${GUI_DIR}' && source .venv/bin/activate && P2_TCP_HOST='${PI_HOST}' P2_TCP_PORT='${BRIDGE_PORT}' python3 main.py"

osascript - "${BRIDGE_TERMINAL_CMD}" "${GUI_TERMINAL_CMD}" <<'APPLESCRIPT'
on run argv
  tell application "Terminal"
    activate
    do script item 1 of argv
    delay 2
    do script item 2 of argv
  end tell
end run
APPLESCRIPT

echo "Started Pi bridge and Software-GUI in separate Terminal windows."
echo "Use the Pi bridge window for Enter prompts during rolling capture."
