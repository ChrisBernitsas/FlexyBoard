#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  run_turn_on_pi_and_pull.sh [--no-open]

Description:
  Syncs local code to Raspberry Pi, runs the full BEFORE -> Enter trigger -> AFTER -> CV analysis flow,
  then copies the newest board_analysis folder and latest captures to local debug_output.

Environment overrides:
  PI_HOST      (default: flexyboard-pi.local)
  PI_USER      (default: pi)
  PI_REPO      (default: /home/pi/FlexyBoard-Camera)
  PI_PASSWORD  (optional; requires sshpass for password automation)
USAGE
}

OPEN_RESULTS=1
if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi
if [[ "${1:-}" == "--no-open" ]]; then
  OPEN_RESULTS=0
elif [[ -n "${1:-}" ]]; then
  echo "Unknown argument: $1"
  usage
  exit 2
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOCAL_DEBUG_DIR="${ROOT_DIR}/debug_output"
SESSION_TS="$(date +%Y%m%d_%H%M%S)"
LOCAL_SESSION_DIR="${LOCAL_DEBUG_DIR}/turn_run_${SESSION_TS}"

PI_HOST="${PI_HOST:-flexyboard-pi.local}"
PI_USER="${PI_USER:-pi}"
PI_REPO="${PI_REPO:-/home/pi/FlexyBoard-Camera}"

mkdir -p "${LOCAL_SESSION_DIR}"

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
    echo "PI_PASSWORD is set but sshpass is not installed."
    echo "Install with: brew install hudochenkov/sshpass/sshpass"
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

rsync_exec() {
  local ssh_cmd
  ssh_cmd="ssh -o StrictHostKeyChecking=accept-new -o ControlMaster=auto -o ControlPath=${CONTROL_PATH} -o ControlPersist=5m"
  if [[ "${use_sshpass}" == "true" ]]; then
    sshpass -p "${PI_PASSWORD}" rsync "$@" -e "${ssh_cmd}"
  else
    rsync "$@" -e "${ssh_cmd}"
  fi
}

echo "Syncing local FlexyBoard-Camera code to Pi (${PI_USER}@${PI_HOST})..."
rsync_exec -az \
  --exclude ".git/" \
  --exclude ".venv/" \
  --exclude "__pycache__/" \
  --exclude ".pytest_cache/" \
  --exclude "debug_output/" \
  --exclude "logs/" \
  "${ROOT_DIR}/" "${PI_USER}@${PI_HOST}:${PI_REPO}/"

echo "Running turn cycle on Pi (${PI_USER}@${PI_HOST})..."
echo "When prompted in the remote output, press Enter to capture AFTER image."
ssh_exec -tt "${PI_USER}@${PI_HOST}" \
  "cd '${PI_REPO}' && source .venv/bin/activate && python3 scripts/run_turn_capture_analyze.py"

LATEST_REMOTE_ANALYSIS_DIR="$(ssh_exec "${PI_USER}@${PI_HOST}" "ls -1dt '${PI_REPO}'/debug_output/board_analysis_* 2>/dev/null | head -n1")"
if [[ -z "${LATEST_REMOTE_ANALYSIS_DIR}" ]]; then
  echo "Could not find remote board_analysis output folder."
  exit 1
fi

REMOTE_DEBUG_DIR="${PI_REPO}/debug_output"
LOCAL_ANALYSIS_PARENT="${LOCAL_SESSION_DIR}/"

scp_exec -r "${PI_USER}@${PI_HOST}:${LATEST_REMOTE_ANALYSIS_DIR}" "${LOCAL_ANALYSIS_PARENT}" >/dev/null
if ssh_exec "${PI_USER}@${PI_HOST}" "test -f '${REMOTE_DEBUG_DIR}/before_latest.png'"; then
  scp_exec "${PI_USER}@${PI_HOST}:${REMOTE_DEBUG_DIR}/before_latest.png" "${LOCAL_SESSION_DIR}/before_latest.png" >/dev/null
fi
if ssh_exec "${PI_USER}@${PI_HOST}" "test -f '${REMOTE_DEBUG_DIR}/after_latest.png'"; then
  scp_exec "${PI_USER}@${PI_HOST}:${REMOTE_DEBUG_DIR}/after_latest.png" "${LOCAL_SESSION_DIR}/after_latest.png" >/dev/null
fi

LOCAL_ANALYSIS_DIR="${LOCAL_SESSION_DIR}/$(basename "${LATEST_REMOTE_ANALYSIS_DIR}")"

echo ""
echo "Pulled results to:"
echo "  ${LOCAL_ANALYSIS_DIR}"
echo ""
echo "Key files:"
echo "  ${LOCAL_ANALYSIS_DIR}/after_outer_only_overlay.png"
echo "  ${LOCAL_ANALYSIS_DIR}/after_chess_only_overlay.png"
echo "  ${LOCAL_ANALYSIS_DIR}/after_grid_overlay.png"
echo "  ${LOCAL_ANALYSIS_DIR}/changed_raw_overlay.png"
echo "  ${LOCAL_ANALYSIS_DIR}/analysis.json"

if [[ "${OPEN_RESULTS}" -eq 1 ]] && command -v open >/dev/null 2>&1; then
  open "${LOCAL_ANALYSIS_DIR}/after_outer_only_overlay.png" || true
  open "${LOCAL_ANALYSIS_DIR}/after_chess_only_overlay.png" || true
  open "${LOCAL_ANALYSIS_DIR}/after_grid_overlay.png" || true
  open "${LOCAL_ANALYSIS_DIR}/changed_raw_overlay.png" || true
  open "${LOCAL_ANALYSIS_DIR}/analysis.json" || true
fi
