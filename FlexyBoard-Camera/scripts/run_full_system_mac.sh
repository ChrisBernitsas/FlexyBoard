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
  PI_CAMERA_REPO   default: /home/pi/FlexyBoard-Camera
  SYNC             default: 1, set to 0 to skip syncing FlexyBoard-Camera to Pi
  BRIDGE_ARGS      optional extra args for run_pi_software_bridge.py

Examples:
  ./scripts/run_full_system_mac.sh
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

PI_HOST="${PI_HOST:-flexyboard-pi.local}"
PI_USER="${PI_USER:-pi}"
PI_CAMERA_REPO="${PI_CAMERA_REPO:-/home/pi/FlexyBoard-Camera}"
SYNC="${SYNC:-1}"
BRIDGE_ARGS="${BRIDGE_ARGS:-}"

if [[ ! -d "${GUI_DIR}" ]]; then
  echo "Missing Software-GUI folder at: ${GUI_DIR}"
  exit 1
fi

if [[ "${SYNC}" != "0" ]]; then
  echo "Syncing FlexyBoard-Camera to ${PI_USER}@${PI_HOST}:${PI_CAMERA_REPO}/ ..."
  rsync -az \
    --exclude ".git/" \
    --exclude ".venv/" \
    --exclude "__pycache__/" \
    --exclude ".pytest_cache/" \
    --exclude "debug_output/" \
    --exclude "logs/" \
    "${CAMERA_DIR}/" "${PI_USER}@${PI_HOST}:${PI_CAMERA_REPO}/"
fi

REMOTE_BRIDGE_CMD="cd '${PI_CAMERA_REPO}' && source .venv/bin/activate && python3 scripts/run_pi_software_bridge.py --capture-mode rolling ${BRIDGE_ARGS}"
BRIDGE_TERMINAL_CMD="ssh -tt '${PI_USER}@${PI_HOST}' \"${REMOTE_BRIDGE_CMD}\""
GUI_TERMINAL_CMD="cd '${GUI_DIR}' && source .venv/bin/activate && P2_TCP_HOST='${PI_HOST}' python3 main.py"

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
