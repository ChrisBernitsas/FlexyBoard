#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${ROOT_DIR}/build"
ELF="${BUILD_DIR}/motor_simple_code.elf"
BIN="${BUILD_DIR}/motor_simple_code.bin"
MAP="${BUILD_DIR}/motor_simple_code.map"

mkdir -p "${BUILD_DIR}"

arm-none-eabi-gcc \
  -mcpu=cortex-m4 \
  -mthumb \
  -mfpu=fpv4-sp-d16 \
  -mfloat-abi=hard \
  -std=c99 \
  -ffunction-sections \
  -fdata-sections \
  -Wall \
  -Wextra \
  -Wpedantic \
  -Og \
  -g3 \
  -I"${ROOT_DIR}" \
  "${ROOT_DIR}/stm32f446_startup.c" \
  "${ROOT_DIR}/main.c" \
  -Wl,-T"${ROOT_DIR}/stm32f446re.ld" \
  -Wl,--gc-sections \
  -Wl,-Map="${MAP}" \
  --specs=nano.specs \
  --specs=nosys.specs \
  -o "${ELF}"

arm-none-eabi-size "${ELF}"
arm-none-eabi-objcopy -O binary "${ELF}" "${BIN}"

echo "Built:"
echo "  ${ELF}"
echo "  ${BIN}"
echo "  ${MAP}"
