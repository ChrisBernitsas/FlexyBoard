#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import platform
import re
import sys
import time
from pathlib import Path
from typing import Any

try:
    import yaml
except Exception:  # pragma: no cover - runtime fallback if PyYAML is unavailable
    yaml = None  # type: ignore[assignment]

ROOT = Path(__file__).resolve().parents[1]
MOVE_FILE = ROOT / "sample_data" / "stm32_move_sequence.txt"
DEFAULT_CONFIG_FILE = ROOT / "configs" / "default.yaml"
MOTOR_MAIN_H = ROOT.parents[0] / "FlexyBoard-Motor-Control" / "Inc" / "main.h"
DEFAULT_SERIAL_PORT = "/dev/ttyACM0" if platform.system() == "Linux" else "/dev/cu.usbmodem103"
SERIAL_PORT = os.environ.get("FLEXY_SERIAL_PORT", DEFAULT_SERIAL_PORT)
SERIAL_BAUDRATE = 115200
SERIAL_TIMEOUT_SEC = 8.0
RETURN_START_DELAY_SEC = 1.0
DEFAULT_MOTOR_BOARD_ORIENTATION = "game_a8_at_motor_00"

try:
    import serial  # type: ignore
except Exception:
    serial = None


def _parse_line_to_move(line: str, line_no: int) -> dict[str, Any] | None:
    content = line.split("#", 1)[0].strip()
    if not content:
        return None

    if "->" not in content:
        raise ValueError(
            f"Invalid move format at line {line_no}. "
            "Expected 'source -> dest', e.g. '2,2 -> 5,5' or '2,2 -> 92%,12%'."
        )

    source_raw, dest_raw = content.split("->", 1)
    source = _parse_endpoint(source_raw.strip(), line_no)
    dest = _parse_endpoint(dest_raw.strip(), line_no)

    return {
        "source": source,
        "dest": dest,
    }


def _parse_endpoint(token: str, line_no: int) -> dict[str, Any]:
    board_match = re.fullmatch(r"\(?\s*(-?\d+)\s*,\s*(-?\d+)\s*\)?", token)
    if board_match:
        return {
            "kind": "board",
            "x": int(board_match.group(1)),
            "y": int(board_match.group(2)),
            "text": token,
        }

    pct_match = re.fullmatch(r"\(?\s*(-?\d+(?:\.\d+)?)\s*%\s*,\s*(-?\d+(?:\.\d+)?)\s*%\s*\)?", token)
    if pct_match:
        return {
            "kind": "pct",
            "x_pct": float(pct_match.group(1)),
            "y_pct": float(pct_match.group(2)),
            "text": token,
        }

    raise ValueError(
        f"Invalid endpoint at line {line_no}: '{token}'. "
        "Use board 'x,y' (e.g. 2,2) or percent 'x%,y%' (e.g. 92%,12%)."
    )


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
        "WORKSPACE_MIN_X_STEPS",
        "WORKSPACE_MAX_X_STEPS",
        "WORKSPACE_MIN_Y_STEPS",
        "WORKSPACE_MAX_Y_STEPS",
        "WORKSPACE_PERCENT_SCALE",
    ]
    return {k: _parse_define_int(text, k) for k in keys}


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


def _game_to_motor_board_coord(
    board_x: int,
    board_y: int,
    *,
    max_i: int,
    orientation: str,
) -> tuple[int, int]:
    normalized = str(orientation or "identity").strip().lower()
    if normalized in {"identity", "game_a1_at_motor_00"}:
        return board_x, board_y
    if normalized == "game_a8_at_motor_00":
        return max_i - board_y, board_x
    raise ValueError(f"Unsupported motor board_orientation: {orientation!r}")


def _board_to_steps(board_x: int, board_y: int, c: dict[str, int], orientation: str) -> tuple[int, int]:
    max_i = c["BOARD_GRID_MAX_INDEX"]
    if board_x < 0 or board_x > max_i or board_y < 0 or board_y > max_i:
        raise ValueError(f"Board coordinate out of range: ({board_x},{board_y})")

    motor_x, motor_y = _game_to_motor_board_coord(
        board_x,
        board_y,
        max_i=max_i,
        orientation=orientation,
    )

    u = float(motor_x) / float(max_i)
    v = float(motor_y) / float(max_i)

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


def _percent_to_steps(pct_x: float, pct_y: float, c: dict[str, int]) -> tuple[int, int]:
    scale = float(c["WORKSPACE_PERCENT_SCALE"])
    if pct_x < 0.0 or pct_x > scale or pct_y < 0.0 or pct_y > scale:
        raise ValueError(
            f"Percent coordinate out of range: ({pct_x}%,{pct_y}%). "
            f"Expected 0..{int(scale)}%."
        )

    x_min = c["WORKSPACE_MIN_X_STEPS"]
    x_max = c["WORKSPACE_MAX_X_STEPS"]
    y_min = c["WORKSPACE_MIN_Y_STEPS"]
    y_max = c["WORKSPACE_MAX_Y_STEPS"]

    x_range = x_max - x_min
    y_range = y_max - y_min

    x_steps = int(round(x_min + (x_range * (pct_x / scale))))
    y_steps = int(round(y_min + (y_range * (pct_y / scale))))
    return x_steps, y_steps


def _endpoint_to_steps(endpoint: dict[str, Any], c: dict[str, int], orientation: str) -> tuple[int, int]:
    kind = endpoint.get("kind")
    if kind == "board":
        return _board_to_steps(int(endpoint["x"]), int(endpoint["y"]), c, orientation)
    if kind == "pct":
        return _percent_to_steps(float(endpoint["x_pct"]), float(endpoint["y_pct"]), c)
    raise ValueError(f"Unsupported endpoint kind: {kind}")


def _endpoint_to_motor_board(endpoint: dict[str, Any], c: dict[str, int], orientation: str) -> dict[str, Any] | None:
    if endpoint.get("kind") != "board":
        return None
    max_i = c["BOARD_GRID_MAX_INDEX"]
    motor_x, motor_y = _game_to_motor_board_coord(
        int(endpoint["x"]),
        int(endpoint["y"]),
        max_i=max_i,
        orientation=orientation,
    )
    return {"x": motor_x, "y": motor_y, "text": f"{motor_x},{motor_y}"}


def _endpoint_label(endpoint: dict[str, Any]) -> str:
    kind = endpoint.get("kind")
    if kind == "board":
        return f"{endpoint['x']},{endpoint['y']}"
    if kind == "pct":
        return f"{endpoint['x_pct']:.2f}%,{endpoint['y_pct']:.2f}%"
    return str(endpoint)


def _build_stage_summary(planned_moves: list[dict[str, Any]], c: dict[str, int], orientation: str) -> list[dict[str, Any]]:
    stages: list[dict[str, Any]] = []
    prev_x = 0
    prev_y = 0

    for idx, move in enumerate(planned_moves, start=1):
        src = move["source"]
        dst = move["dest"]
        src_x, src_y = _endpoint_to_steps(src, c, orientation)
        dst_x, dst_y = _endpoint_to_steps(dst, c, orientation)

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
        motor_board_orientation = _load_motor_board_orientation()
        planned_stage_steps = _build_stage_summary(planned_moves, mapping_constants, motor_board_orientation)
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
        "config_file": str(DEFAULT_CONFIG_FILE),
        "motor_board_orientation": motor_board_orientation,
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
            src_steps_x, src_steps_y = _endpoint_to_steps(
                move["source"],
                mapping_constants,
                motor_board_orientation,
            )
            dst_steps_x, dst_steps_y = _endpoint_to_steps(
                move["dest"],
                mapping_constants,
                motor_board_orientation,
            )

            cmd = f"MOVE_STEPS {src_steps_x} {src_steps_y} {dst_steps_x} {dst_steps_y}"
            reply = _send_command(ser, cmd)
            move_status = _send_command(ser, "STATUS")
            status_x, status_y = _parse_status_xy(move_status)
            result["executed"].append(
                {
                    "index": idx,
                    "move": move,
                    "cmd": cmd,
                    "reply": reply,
                    "status_after_move": move_status,
                    "source_steps": {"x": src_steps_x, "y": src_steps_y},
                    "dest_steps_planned": {"x": dst_steps_x, "y": dst_steps_y},
                    "dest_steps_status": {"x": status_x, "y": status_y},
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
                    "endpoint": dict(move["source"]),
                    "endpoint_label": _endpoint_label(move["source"]),
                    "motor_board_endpoint": _endpoint_to_motor_board(
                        move["source"],
                        mapping_constants,
                        motor_board_orientation,
                    ),
                    "x": src_steps_x,
                    "y": src_steps_y,
                }
            )
            result["position_report"].append(
                {
                    "stage": f"move_{idx}_dest",
                    "endpoint": dict(move["dest"]),
                    "endpoint_label": _endpoint_label(move["dest"]),
                    "motor_board_endpoint": _endpoint_to_motor_board(
                        move["dest"],
                        mapping_constants,
                        motor_board_orientation,
                    ),
                    "x": status_x,
                    "y": status_y,
                }
            )
            last_dest = {"x": status_x, "y": status_y}

        if last_dest is not None and (last_dest["x"] != 0 or last_dest["y"] != 0):
            return_zero_move = {
                "source_steps": {"x": int(last_dest["x"]), "y": int(last_dest["y"])},
                "dest_steps": {"x": 0, "y": 0},
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
        if "endpoint" in item:
            print(f"{item['stage']} endpoint={item['endpoint_label']} x={item['x']} y={item['y']}")
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
