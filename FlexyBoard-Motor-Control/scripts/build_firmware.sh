#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="${ROOT_DIR}/build"
mkdir -p "${OUT_DIR}"

TOOLCHAIN_PREFIX="${TOOLCHAIN_PREFIX:-arm-none-eabi}"
CC="${TOOLCHAIN_PREFIX}-gcc"
OBJCOPY="${TOOLCHAIN_PREFIX}-objcopy"
SIZE="${TOOLCHAIN_PREFIX}-size"
TARGET_MCU="STM32F401RE"
STARTUP_FILE="${ROOT_DIR}/Startup/startup_stm32f401retx.s"
LINKER_SCRIPT="${ROOT_DIR}/STM32F401RETX_FLASH.ld"

COMMON_FLAGS=(
  -mcpu=cortex-m4
  -mthumb
  -mfpu=fpv4-sp-d16
  -mfloat-abi=hard
  -std=gnu11
  -g3
  -O2
  -ffunction-sections
  -fdata-sections
  -Wall
  -Wextra
  -I"${ROOT_DIR}/Inc"
)

"${CC}" "${COMMON_FLAGS[@]}" -c "${ROOT_DIR}/Src/main.c" -o "${OUT_DIR}/main.o"
"${CC}" "${COMMON_FLAGS[@]}" -c "${ROOT_DIR}/Src/syscalls.c" -o "${OUT_DIR}/syscalls.o"
"${CC}" "${COMMON_FLAGS[@]}" -c "${ROOT_DIR}/Src/sysmem.c" -o "${OUT_DIR}/sysmem.o"
"${CC}" "${COMMON_FLAGS[@]}" -x assembler-with-cpp -c "${STARTUP_FILE}" -o "${OUT_DIR}/startup.o"

"${CC}" \
  "${OUT_DIR}/main.o" \
  "${OUT_DIR}/syscalls.o" \
  "${OUT_DIR}/sysmem.o" \
  "${OUT_DIR}/startup.o" \
  -mcpu=cortex-m4 \
  -mthumb \
  -mfpu=fpv4-sp-d16 \
  -mfloat-abi=hard \
  -T"${LINKER_SCRIPT}" \
  --specs=nosys.specs \
  --specs=nano.specs \
  -Wl,-Map="${OUT_DIR}/FlexyBoard_MotorControl.map" \
  -Wl,--gc-sections \
  -Wl,--start-group -lc -lm -Wl,--end-group \
  -o "${OUT_DIR}/FlexyBoard_MotorControl.elf"

"${OBJCOPY}" -O binary "${OUT_DIR}/FlexyBoard_MotorControl.elf" "${OUT_DIR}/FlexyBoard_MotorControl.bin"
"${SIZE}" "${OUT_DIR}/FlexyBoard_MotorControl.elf"

echo "Built:"
echo "  ${OUT_DIR}/FlexyBoard_MotorControl.elf"
echo "  ${OUT_DIR}/FlexyBoard_MotorControl.bin"
echo "  ${OUT_DIR}/FlexyBoard_MotorControl.map"
echo "Target MCU: ${TARGET_MCU}"
