#!/usr/bin/env python3
from __future__ import annotations

import os
import platform
import re
import sys
import time
from pathlib import Path

try:
    import yaml  # type: ignore
except Exception:
    yaml = None

try:
    import serial  # type: ignore
except Exception:
    serial = None


DEFAULT_SERIAL_PORT = "/dev/ttyACM0" if platform.system() == "Linux" else "/dev/cu.usbmodem103"
SERIAL_PORT = os.environ.get("FLEXY_SERIAL_PORT", DEFAULT_SERIAL_PORT)
SERIAL_BAUDRATE = 115200
SERIAL_TIMEOUT_SEC = 8.0
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_FILE = ROOT / "configs" / "default.yaml"
DEFAULT_MOTOR_BOARD_ORIENTATION = "game_a8_at_motor_00"


def _load_motor_board_orientation() -> str:
    if yaml is None or not DEFAULT_CONFIG_FILE.exists():
        return DEFAULT_MOTOR_BOARD_ORIENTATION
    try:
        data = yaml.safe_load(DEFAULT_CONFIG_FILE.read_text(encoding="utf-8")) or {}
    except Exception:
        return DEFAULT_MOTOR_BOARD_ORIENTATION
    motor = data.get("motor", {})
    if not isinstance(motor, dict):
        return DEFAULT_MOTOR_BOARD_ORIENTATION
    return str(motor.get("board_orientation", DEFAULT_MOTOR_BOARD_ORIENTATION)).strip().lower()


def _parse_square(token: str) -> tuple[int, int]:
    match = re.fullmatch(r"([a-hA-H])([1-8])", token.strip())
    if not match:
        raise ValueError(f"Invalid square: {token}")
    file_idx = ord(match.group(1).lower()) - ord("a")
    rank_idx = int(match.group(2)) - 1
    return file_idx, rank_idx


def _square_to_board_coords(token: str, orientation: str) -> tuple[int, int]:
    game_x, game_y = _parse_square(token)
    if orientation in {"identity", "game_a1_at_motor_00"}:
        return game_x, game_y
    if orientation == "game_a8_at_motor_00":
        return 7 - game_y, game_x
    return game_x, game_y


def _read_line(ser: object, timeout_sec: float) -> str | None:
    old_timeout = ser.timeout
    ser.timeout = timeout_sec
    try:
        raw = ser.readline()
    finally:
        ser.timeout = old_timeout

    if not raw:
        return None
    return raw.decode("utf-8", errors="replace").strip()


def _send_command(ser: object, cmd: str) -> str:
    ser.write((cmd + "\n").encode("utf-8"))
    ser.flush()
    reply = _read_line(ser, SERIAL_TIMEOUT_SEC)
    if reply is None:
        raise RuntimeError(f"No response for command: {cmd}")
    return reply


def _status(ser: object) -> tuple[int, int, int, str]:
    reply = _send_command(ser, "STATUS")
    match = re.search(r"cur_x=(-?\d+)\s+cur_y=(-?\d+)(?:\s+cur_z=(-?\d+))?", reply)
    if not match:
        raise RuntimeError(f"Could not parse STATUS reply: {reply}")
    x = int(match.group(1))
    y = int(match.group(2))
    z = int(match.group(3) or 0)
    return x, y, z, reply


def _print_help() -> None:
    print(
        "Commands:\n"
        "  status                 Show current x/y/z step counters\n"
        "  zero                   Reset STM internal counters to 0,0,0\n"
        "  goto x y               Absolute workspace move in steps\n"
        "  goto a1                Board-square move using chess notation\n"
        "  jog dx dy              Relative workspace move in steps\n"
        "  jog left n             Pulse only the left Y step line (diagnostic)\n"
        "  jog right n            Pulse only the right Y step line (diagnostic)\n"
        "  z dz                   Relative Z move in steps\n"
        "  pickup                 Run PICKUP_STEPS at current x,y\n"
        "  release                Run RELEASE_STEPS at current x,y\n"
        "  ping                   Ping STM32\n"
        "  help                   Show this help\n"
        "  quit                   Exit\n"
    )


def main() -> int:
    if serial is None:
        print("pyserial is required", file=sys.stderr)
        return 1

    ser = serial.Serial(port=SERIAL_PORT, baudrate=SERIAL_BAUDRATE, timeout=SERIAL_TIMEOUT_SEC)
    try:
        orientation = _load_motor_board_orientation()
        time.sleep(0.8)
        while True:
            line = _read_line(ser, 0.1)
            if line is None:
                break
            if line:
                print(f"[boot] {line}")

        print(f"Connected to {SERIAL_PORT} @ {SERIAL_BAUDRATE}")
        _print_help()

        while True:
            try:
                raw = input("stm32> ").strip()
            except EOFError:
                print()
                return 0

            if not raw:
                continue

            if raw in {"quit", "exit"}:
                return 0
            if raw == "help":
                _print_help()
                continue
            if raw == "ping":
                print(_send_command(ser, "PING"))
                continue
            if raw == "zero":
                print(_send_command(ser, "ZERO"))
                print(_status(ser)[3])
                continue
            if raw == "status":
                print(_status(ser)[3])
                continue
            if raw == "pickup":
                x, y, _, _ = _status(ser)
                print(_send_command(ser, f"PICKUP_STEPS {x} {y}"))
                print(_status(ser)[3])
                continue
            if raw == "release":
                x, y, _, _ = _status(ser)
                print(_send_command(ser, f"RELEASE_STEPS {x} {y}"))
                print(_status(ser)[3])
                continue

            parts = raw.split()
            if len(parts) == 2 and parts[0] == "goto":
                try:
                    board_x, board_y = _square_to_board_coords(parts[1], orientation)
                except ValueError:
                    pass
                else:
                    cmd = f"GOTO {board_x} {board_y}"
                    print(_send_command(ser, cmd))
                    print(_status(ser)[3])
                    continue
            if len(parts) == 3 and parts[0] == "goto":
                cmd = f"GOTO_STEPS {int(parts[1])} {int(parts[2])}"
                print(_send_command(ser, cmd))
                print(_status(ser)[3])
                continue
            if len(parts) == 3 and parts[0] == "jog":
                side = parts[1].lower()
                if side in {"left", "l"}:
                    cmd = f"JOG_Y_LEFT {int(parts[2])}"
                    print(_send_command(ser, cmd))
                    print(_status(ser)[3] + "  [logical counters unchanged]")
                    continue
                if side in {"right", "r"}:
                    cmd = f"JOG_Y_RIGHT {int(parts[2])}"
                    print(_send_command(ser, cmd))
                    print(_status(ser)[3] + "  [logical counters unchanged]")
                    continue

            if len(parts) == 3 and parts[0] == "jog":
                cmd = f"JOG_STEPS {int(parts[1])} {int(parts[2])}"
                print(_send_command(ser, cmd))
                print(_status(ser)[3])
                continue
            if len(parts) == 2 and parts[0] == "z":
                cmd = f"JOG_Z {int(parts[1])}"
                print(_send_command(ser, cmd))
                print(_status(ser)[3])
                continue

            print("Unrecognized command. Type 'help'.")
    finally:
        ser.close()


if __name__ == "__main__":
    raise SystemExit(main())
