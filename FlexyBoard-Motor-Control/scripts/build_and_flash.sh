#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

usage() {
  cat <<'USAGE'
Usage:
  ./scripts/build_and_flash.sh
  ./scripts/build_and_flash.sh --debug-elf

Options:
  --debug-elf  Flash existing Debug/FlexyBoard_MotorControl.elf without rebuild
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ "${1:-}" == "--debug-elf" ]]; then
  "${ROOT_DIR}/scripts/flash_firmware.sh" "${ROOT_DIR}/Debug/FlexyBoard_MotorControl.elf"
  exit 0
fi

if [[ -n "${1:-}" ]]; then
  echo "Unknown argument: $1" >&2
  usage
  exit 2
fi

"${ROOT_DIR}/scripts/build_firmware.sh"
"${ROOT_DIR}/scripts/flash_firmware.sh"
