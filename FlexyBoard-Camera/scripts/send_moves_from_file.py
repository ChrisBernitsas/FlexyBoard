#!/usr/bin/env python3
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path
from typing import Any

MOVE_FILE = Path(__file__).resolve().parents[1] / "sample_data" / "stm32_move_sequence.txt"
MOTOR_MAIN_H = Path(__file__).resolve().parents[2] / "FlexyBoard-Motor-Control" / "Inc" / "main.h"
SERIAL_PORT = "/dev/ttyACM0"
SERIAL_BAUDRATE = 115200
SERIAL_TIMEOUT_SEC = 8.0
RETURN_START_DELAY_SEC = 1.0

try:
    import serial  # type: ignore
except Exception:
    serial = None


def _parse_line_to_move(line: str, line_no: int) -> dict[str, Any] | None:
    content = line.split("#", 1)[0].strip()
    if not content:
        return None

    nums = re.findall(r"-?\d+", content)
    if len(nums) != 4:
        raise ValueError(
            f"Invalid move format at line {line_no}. "
            "Expected 4 integers (sx sy dx dy), e.g. '2,2 -> 5,5'."
        )

    sx, sy, dx, dy = (int(n) for n in nums)
    return {
        "source": {"x": sx, "y": sy},
        "dest": {"x": dx, "y": dy},
    }


def _load_moves(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Move file not found: {path}")

    moves: list[dict[str, Any]] = []
    for idx, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        parsed = _parse_line_to_move(raw, idx)
        if parsed is not None:
            moves.append(parsed)

    if not moves:
        raise ValueError("Move file contains no valid moves")

    return moves


def _read_line(ser: Any, timeout_sec: float) -> str | None:
    old_timeout = ser.timeout
    ser.timeout = timeout_sec
    try:
        raw = ser.readline()
    finally:
        ser.timeout = old_timeout

    if not raw:
        return None
    return raw.decode("utf-8", errors="replace").strip()


def _send_command(ser: Any, cmd: str) -> str:
    ser.write((cmd + "\n").encode("utf-8"))
    ser.flush()

    reply = _read_line(ser, SERIAL_TIMEOUT_SEC)
    if reply is None:
        raise RuntimeError(f"No response for command: {cmd}")
    if reply.startswith("ERR"):
        raise RuntimeError(f"STM32 returned error for '{cmd}': {reply}")
    return reply


def _parse_status_xy(status_line: str) -> tuple[int, int]:
    match = re.search(r"cur_x=(-?\d+)\s+cur_y=(-?\d+)", status_line)
    if not match:
        raise ValueError(f"Could not parse STATUS line: {status_line}")
    return int(match.group(1)), int(match.group(2))


def _parse_define_int(header_text: str, name: str) -> int:
    pattern = rf"^\s*#define\s+{re.escape(name)}\s+(-?\d+)"
    match = re.search(pattern, header_text, flags=re.MULTILINE)
    if not match:
        raise ValueError(f"Missing {name} in {MOTOR_MAIN_H}")
    return int(match.group(1))


def _load_board_mapping_constants() -> dict[str, int]:
    text = MOTOR_MAIN_H.read_text(encoding="utf-8")
    keys = [
        "BOARD_GRID_MAX_INDEX",
        "CORNER_00_X_STEPS",
        "CORNER_00_Y_STEPS",
        "CORNER_70_X_STEPS",
        "CORNER_70_Y_STEPS",
        "CORNER_07_X_STEPS",
        "CORNER_07_Y_STEPS",
        "CORNER_77_X_STEPS",
        "CORNER_77_Y_STEPS",
    ]
    return {k: _parse_define_int(text, k) for k in keys}


def _board_to_steps(board_x: int, board_y: int, c: dict[str, int]) -> tuple[int, int]:
    max_i = c["BOARD_GRID_MAX_INDEX"]
    if board_x < 0 or board_x > max_i or board_y < 0 or board_y > max_i:
        raise ValueError(f"Board coordinate out of range: ({board_x},{board_y})")

    u = float(board_x) / float(max_i)
    v = float(board_y) / float(max_i)

    x_interp = (
        (1.0 - u) * (1.0 - v) * float(c["CORNER_00_X_STEPS"])
        + u * (1.0 - v) * float(c["CORNER_70_X_STEPS"])
        + (1.0 - u) * v * float(c["CORNER_07_X_STEPS"])
        + u * v * float(c["CORNER_77_X_STEPS"])
    )
    y_interp = (
        (1.0 - u) * (1.0 - v) * float(c["CORNER_00_Y_STEPS"])
        + u * (1.0 - v) * float(c["CORNER_70_Y_STEPS"])
        + (1.0 - u) * v * float(c["CORNER_07_Y_STEPS"])
        + u * v * float(c["CORNER_77_Y_STEPS"])
    )

    x_steps = int(round(x_interp))
    y_steps = int(round(y_interp))
    return x_steps, y_steps


def _build_stage_summary(planned_moves: list[dict[str, Any]], c: dict[str, int]) -> list[dict[str, Any]]:
    stages: list[dict[str, Any]] = []
    prev_x = 0
    prev_y = 0

    for idx, move in enumerate(planned_moves, start=1):
        src = move["source"]
        dst = move["dest"]
        src_x, src_y = _board_to_steps(int(src["x"]), int(src["y"]), c)
        dst_x, dst_y = _board_to_steps(int(dst["x"]), int(dst["y"]), c)

        stage_points = [
            (f"move_{idx}_to_source", src_x, src_y),
            (f"move_{idx}_pickup_delay", src_x, src_y),
            (f"move_{idx}_to_dest", dst_x, dst_y),
            (f"move_{idx}_release_delay", dst_x, dst_y),
        ]

        for label, x, y in stage_points:
            stages.append(
                {
                    "stage": label,
                    "x_steps_from_start": x,
                    "y_steps_from_start": y,
                    "delta_x_from_prev_stage": x - prev_x,
                    "delta_y_from_prev_stage": y - prev_y,
                }
            )
            prev_x = x
            prev_y = y

    stages.append(
        {
            "stage": "return_start",
            "x_steps_from_start": 0,
            "y_steps_from_start": 0,
            "delta_x_from_prev_stage": -prev_x,
            "delta_y_from_prev_stage": -prev_y,
        }
    )
    return stages


def main() -> int:
    if len(sys.argv) > 1:
        print(
            "No CLI flags are supported.\n"
            "Edit sample_data/stm32_move_sequence.txt, then run:\n"
            "  python scripts/send_moves_from_file.py"
        )
        return 2

    if serial is None:
        print(json.dumps({"status": "error", "error": "pyserial is required"}, indent=2))
        return 1

    try:
        planned_moves = _load_moves(MOVE_FILE)
    except Exception as exc:  # noqa: BLE001
        print(json.dumps({"status": "error", "error": str(exc)}, indent=2))
        return 1

    try:
        mapping_constants = _load_board_mapping_constants()
        planned_stage_steps = _build_stage_summary(planned_moves, mapping_constants)
    except Exception as exc:  # noqa: BLE001
        print(
            json.dumps(
                {
                    "status": "error",
                    "error": f"Failed to load board mapping from {MOTOR_MAIN_H}: {exc}",
                },
                indent=2,
            )
        )
        return 1

    result: dict[str, Any] = {
        "status": "ok",
        "port": SERIAL_PORT,
        "baudrate": SERIAL_BAUDRATE,
        "move_file": str(MOVE_FILE),
        "planned_moves": planned_moves,
        "executed": [],
        "home_sent": True,
        "return_zero": True,
        "return_zero_executed": False,
        "boot_messages": [],
        "status_checkpoints": [],
        "return_start_delay_sec": RETURN_START_DELAY_SEC,
        "mapping_header": str(MOTOR_MAIN_H),
        "planned_stage_steps": planned_stage_steps,
        "position_report": [],
    }

    ser = None
    try:
        ser = serial.Serial(port=SERIAL_PORT, baudrate=SERIAL_BAUDRATE, timeout=SERIAL_TIMEOUT_SEC)

        # NUCLEO VCP often resets MCU on open; give firmware a moment.
        time.sleep(0.8)

        for _ in range(3):
            msg = _read_line(ser, 0.15)
            if msg is None:
                break
            result["boot_messages"].append(msg)

        ser.reset_input_buffer()

        ping_reply = _send_command(ser, "PING")
        result["ping"] = ping_reply

        # No endstops for now; ZERO only resets internal step tracker.
        zero_reply = _send_command(ser, "ZERO")
        result["home"] = zero_reply
        zero_status = _send_command(ser, "STATUS")
        result["status_checkpoints"].append(
            {
                "stage": "after_zero",
                "status": zero_status,
            }
        )
        zero_x, zero_y = _parse_status_xy(zero_status)
        result["position_report"].append(
            {
                "stage": "initial",
                "x": zero_x,
                "y": zero_y,
            }
        )

        last_dest: dict[str, int] | None = None
        for idx, move in enumerate(planned_moves, start=1):
            cmd = f"MOVE {move['source']['x']} {move['source']['y']} {move['dest']['x']} {move['dest']['y']}"
            reply = _send_command(ser, cmd)
            move_status = _send_command(ser, "STATUS")
            src_steps_x, src_steps_y = _board_to_steps(
                int(move["source"]["x"]), int(move["source"]["y"]), mapping_constants
            )
            dst_steps_x, dst_steps_y = _parse_status_xy(move_status)
            result["executed"].append(
                {
                    "index": idx,
                    "move": move,
                    "cmd": cmd,
                    "reply": reply,
                    "status_after_move": move_status,
                    "source_steps": {"x": src_steps_x, "y": src_steps_y},
                    "dest_steps": {"x": dst_steps_x, "y": dst_steps_y},
                }
            )
            result["status_checkpoints"].append(
                {
                    "stage": f"after_move_{idx}",
                    "status": move_status,
                }
            )
            result["position_report"].append(
                {
                    "stage": f"move_{idx}_source",
                    "square": dict(move["source"]),
                    "x": src_steps_x,
                    "y": src_steps_y,
                }
            )
            result["position_report"].append(
                {
                    "stage": f"move_{idx}_dest",
                    "square": dict(move["dest"]),
                    "x": dst_steps_x,
                    "y": dst_steps_y,
                }
            )
            last_dest = dict(move["dest"])

        if last_dest is not None and (last_dest["x"] != 0 or last_dest["y"] != 0):
            return_zero_move = {
                "source": {"x": int(last_dest["x"]), "y": int(last_dest["y"])},
                "dest": {"x": 0, "y": 0},
            }
            time.sleep(RETURN_START_DELAY_SEC)
            rz_cmd = "RETURN_START"
            rz_reply = _send_command(ser, rz_cmd)
            result["return_zero_move"] = return_zero_move
            result["return_zero_cmd"] = rz_cmd
            result["return_zero_reply"] = rz_reply
            result["return_zero_executed"] = True
            return_status = _send_command(ser, "STATUS")
            result["status_checkpoints"].append(
                {
                    "stage": "after_return_start",
                    "status": return_status,
                }
            )
            return_x, return_y = _parse_status_xy(return_status)
            result["position_report"].append(
                {
                    "stage": "returned_start",
                    "x": return_x,
                    "y": return_y,
                }
            )

        status_reply = _send_command(ser, "STATUS")
        result["final_status"] = status_reply

    except Exception as exc:  # noqa: BLE001
        result["status"] = "error"
        result["error"] = str(exc)
        result[
            "hint"
        ] = (
            "Make sure STM32 firmware with UART MOVE parser is flashed and STM32 is on /dev/ttyACM0."
        )
        print(json.dumps(result, indent=2))
        return 1
    finally:
        if ser is not None:
            ser.close()

    print("Position Report")
    for item in result["position_report"]:
        if "square" in item:
            sq = item["square"]
            print(f"{item['stage']} square=({sq['x']},{sq['y']}) x={item['x']} y={item['y']}")
        else:
            print(f"{item['stage']} x={item['x']} y={item['y']}")

    compact = {
        "status": result["status"],
        "move_file": result["move_file"],
        "position_report": result["position_report"],
        "final_status": result.get("final_status"),
    }
    print(json.dumps(compact, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
