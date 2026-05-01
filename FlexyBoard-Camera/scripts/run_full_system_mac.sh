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
  WAIT_MODE        default: gpio
  GPIO_PIN         default: 17
  STATUS_LED_PIN   default: 27; set to empty string to disable the player-ready LED
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
WAIT_MODE="${WAIT_MODE:-gpio}"
GPIO_PIN="${GPIO_PIN:-17}"
STATUS_LED_PIN="${STATUS_LED_PIN:-27}"
BRIDGE_ARGS="${BRIDGE_ARGS:-}"

shell_quote() {
  printf "%q" "$1"
}

run_remote_cmd() {
  local remote_cmd="$1"
  if [[ -n "${PI_PASSWORD}" ]]; then
    SSHPASS="${PI_PASSWORD}" sshpass -e ssh -o StrictHostKeyChecking=no -tt "${PI_USER}@${PI_HOST}" "${remote_cmd}"
  else
    ssh -tt "${PI_USER}@${PI_HOST}" "${remote_cmd}"
  fi
}

ensure_remote_camera_venv() {
  echo "Ensuring Python environment exists on ${PI_USER}@${PI_HOST}:${PI_CAMERA_REPO}/.venv ..."
  local remote_cmd="
set -euo pipefail
cd '${PI_CAMERA_REPO}'
venv_created=0
if [[ ! -f .venv/bin/activate ]]; then
  python3 -m venv .venv
  venv_created=1
fi
. .venv/bin/activate
requirements_hash=\"\$(sha256sum requirements.txt | awk '{print \$1}')\"
python_version=\"\$(python -c 'import sys; print(f\"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}\")')\"
requirements_stamp=.venv/.requirements.sha256
python_stamp=.venv/.python_version
if [[ \"\${venv_created}\" == \"1\" || ! -f \"\${requirements_stamp}\" || ! -f \"\${python_stamp}\" || \"\$(cat \"\${requirements_stamp}\")\" != \"\${requirements_hash}\" || \"\$(cat \"\${python_stamp}\")\" != \"\${python_version}\" ]]; then
  python -m pip install --disable-pip-version-check --upgrade pip setuptools wheel
  python -m pip install --disable-pip-version-check -r requirements.txt
  printf '%s' \"\${requirements_hash}\" > \"\${requirements_stamp}\"
  printf '%s' \"\${python_version}\" > \"\${python_stamp}\"
else
  echo 'Python environment already up to date.'
fi
if [[ \"$(uname -s)\" == \"Linux\" ]]; then
  if ! python -c 'import RPi.GPIO' >/dev/null 2>&1 && ! python -c 'import sys; major, minor = sys.version_info[:2]; sys.path.extend([f\"/usr/local/lib/python{major}.{minor}/dist-packages\", \"/usr/lib/python3/dist-packages\", f\"/usr/lib/python{major}.{minor}/dist-packages\"]); import gpiozero, lgpio' >/dev/null 2>&1; then
    echo 'Warning: no supported GPIO backend available in the Pi Python environment; button/LED wait mode may fail.' >&2
  fi
fi
"
  run_remote_cmd "${remote_cmd}"
}

ensure_remote_motor_toolchain() {
  echo "Ensuring STM32 build/flash tools exist on ${PI_USER}@${PI_HOST} ..."
  local remote_cmd
  if [[ -n "${PI_PASSWORD}" ]]; then
    remote_cmd="
set -euo pipefail
if ! command -v arm-none-eabi-gcc >/dev/null 2>&1 || ! command -v arm-none-eabi-objcopy >/dev/null 2>&1 || ! command -v openocd >/dev/null 2>&1; then
  printf '%s\n' '${PI_PASSWORD}' | sudo -S apt-get update
  printf '%s\n' '${PI_PASSWORD}' | sudo -S apt-get install -y gcc-arm-none-eabi binutils-arm-none-eabi openocd
fi
"
  else
    remote_cmd="
set -euo pipefail
if ! command -v arm-none-eabi-gcc >/dev/null 2>&1 || ! command -v arm-none-eabi-objcopy >/dev/null 2>&1 || ! command -v openocd >/dev/null 2>&1; then
  sudo apt-get update
  sudo apt-get install -y gcc-arm-none-eabi binutils-arm-none-eabi openocd
fi
"
  fi
  run_remote_cmd "${remote_cmd}"
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

ensure_remote_camera_venv

if [[ "${FLASH_STM32}" == "1" ]]; then
  ensure_remote_motor_toolchain
  echo "Building and flashing STM32 firmware from ${PI_MOTOR_REPO} ..."
  REMOTE_FLASH_CMD="cd '${PI_MOTOR_REPO}' && chmod +x scripts/*.sh && ./scripts/build_and_flash.sh"
  run_remote_cmd "${REMOTE_FLASH_CMD}"
elif [[ "${FLASH_STM32}" != "0" ]]; then
  echo "FLASH_STM32 must be 0 or 1, got: ${FLASH_STM32}"
  exit 2
fi

kill_remote_port_process "${BRIDGE_PORT}"

REMOTE_BRIDGE_CMD="cd '${PI_CAMERA_REPO}' && source .venv/bin/activate && python3 scripts/run_pi_software_bridge.py --capture-mode rolling --port '${BRIDGE_PORT}' --wait-mode '${WAIT_MODE}' --gpio-pin '${GPIO_PIN}'"
if [[ -n "${STATUS_LED_PIN}" ]]; then
  REMOTE_BRIDGE_CMD="${REMOTE_BRIDGE_CMD} --status-led-pin '${STATUS_LED_PIN}'"
fi
REMOTE_BRIDGE_CMD="${REMOTE_BRIDGE_CMD} ${BRIDGE_ARGS}"
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
    if (count of windows) > 0 then
      do script item 1 of argv in front window
    else
      do script item 1 of argv
    end if
    delay 2
    do script item 2 of argv
  end tell
end run
APPLESCRIPT

echo "Started Pi bridge and Software-GUI in separate Terminal windows."
if [[ "${WAIT_MODE}" == "gpio" ]]; then
  echo "Use the Pi button on BCM ${GPIO_PIN} to advance rolling-capture prompts."
else
  echo "Use the Pi bridge window for Enter prompts during rolling capture."
fi
