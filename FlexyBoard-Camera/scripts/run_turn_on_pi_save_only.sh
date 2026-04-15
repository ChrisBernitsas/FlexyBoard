#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  run_turn_on_pi_save_only.sh

Description:
  Runs full BEFORE -> Enter -> AFTER turn flow on Raspberry Pi,
  pulls raw captures + latest board_analysis folder locally,
  and does NOT auto-open any files.

Environment overrides:
  PI_HOST      (default: flexyboard-pi.local)
  PI_USER      (default: pi)
  PI_REPO      (default: /home/pi/FlexyBoard-Camera)
  PI_PASSWORD  (optional; requires sshpass)
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi
if [[ -n "${1:-}" ]]; then
  echo "This script takes no positional args."
  usage
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${SCRIPT_DIR}/run_turn_on_pi_and_pull.sh" --no-open
