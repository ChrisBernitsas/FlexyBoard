#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ELF="${ROOT_DIR}/build/motor_simple_code.elf"
ELF_REL="build/motor_simple_code.elf"

if [[ ! -f "${ELF}" ]]; then
  echo "Missing ${ELF}"
  echo "Run: ./build.sh"
  exit 1
fi

cd "${ROOT_DIR}"

openocd \
  -f interface/stlink.cfg \
  -f target/stm32f4x.cfg \
  -c "init; reset halt; program ${ELF_REL} verify; reset run; shutdown"
