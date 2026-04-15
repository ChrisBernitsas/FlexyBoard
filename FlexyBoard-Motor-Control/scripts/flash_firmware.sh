#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ELF_PATH="${1:-${ROOT_DIR}/build/FlexyBoard_MotorControl.elf}"

if [[ ! -f "${ELF_PATH}" ]]; then
  echo "ELF not found: ${ELF_PATH}" >&2
  echo "Build first with: ./scripts/build_firmware.sh" >&2
  exit 1
fi

openocd -f interface/stlink.cfg -f target/stm32f4x.cfg \
  -c "init; reset halt; program ${ELF_PATH} verify; reset run; shutdown"
