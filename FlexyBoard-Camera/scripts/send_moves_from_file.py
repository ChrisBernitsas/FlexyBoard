#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import platform
import re
import sys
import time
import argparse
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
RETURN_START_DELAY_SEC = 0.0
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


def _endpoints_equal(a: dict[str, Any], b: dict[str, Any]) -> bool:
    if a.get("kind") != b.get("kind"):
        return False
    if a.get("kind") == "board":
        return int(a["x"]) == int(b["x"]) and int(a["y"]) == int(b["y"])
    if a.get("kind") == "pct":
        return (
            abs(float(a["x_pct"]) - float(b["x_pct"])) < 1e-6
            and abs(float(a["y_pct"]) - float(b["y_pct"])) < 1e-6
        )
    return False


def _group_continuous_moves(planned_moves: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    if not planned_moves:
        return []
    groups: list[list[dict[str, Any]]] = [[planned_moves[0]]]
    for move in planned_moves[1:]:
        last_group = groups[-1]
        if _endpoints_equal(last_group[-1]["dest"], move["source"]):
            last_group.append(move)
        else:
            groups.append([move])
    return groups


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


def _parse_define_int(header_text: str, name: str, _seen: set[str] | None = None) -> int:
    if _seen is None:
        _seen = set()
    if name in _seen:
        raise ValueError(f"Circular define reference for {name} in {MOTOR_MAIN_H}")
    _seen.add(name)

    pattern = rf"^\s*#define\s+{re.escape(name)}\s+(.+?)\s*$"
    match = re.search(pattern, header_text, flags=re.MULTILINE)
    if not match:
        raise ValueError(f"Missing {name} in {MOTOR_MAIN_H}")

    value_text = match.group(1).split("//", 1)[0].strip()
    int_match = re.fullmatch(r"-?\d+", value_text)
    if int_match:
        return int(int_match.group(0))

    alias_match = re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", value_text)
    if alias_match:
        return _parse_define_int(header_text, alias_match.group(0), _seen)

    raise ValueError(f"Unsupported define value for {name} in {MOTOR_MAIN_H}: {value_text!r}")


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

    groups = _group_continuous_moves(planned_moves)
    move_idx = 0
    for group_idx, group in enumerate(groups, start=1):
        first_src = group[0]["source"]
        src_x, src_y = _endpoint_to_steps(first_src, c, orientation)
        stage_points = [
            (f"group_{group_idx}_to_source", src_x, src_y),
            (f"group_{group_idx}_pickup", src_x, src_y),
        ]

        for move in group:
            move_idx += 1
            dst_x, dst_y = _endpoint_to_steps(move["dest"], c, orientation)
            stage_points.append((f"move_{move_idx}_to_dest", dst_x, dst_y))

        stage_points.append((f"group_{group_idx}_release", dst_x, dst_y))

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


def _run_legacy_move(
    ser: Any,
    move: dict[str, Any],
    *,
    mapping_constants: dict[str, int],
    motor_board_orientation: str,
) -> tuple[list[dict[str, Any]], tuple[int, int]]:
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
    return (
        [
            {
                "command_type": "MOVE_STEPS",
                "cmd": cmd,
                "reply": reply,
                "status_after_command": move_status,
                "source_steps": {"x": src_steps_x, "y": src_steps_y},
                "dest_steps_planned": {"x": dst_steps_x, "y": dst_steps_y},
                "dest_steps_status": {"x": status_x, "y": status_y},
            }
        ],
        (status_x, status_y),
    )


def _run_chained_group(
    ser: Any,
    group: list[dict[str, Any]],
    *,
    mapping_constants: dict[str, int],
    motor_board_orientation: str,
) -> tuple[list[dict[str, Any]], tuple[int, int]]:
    executed: list[dict[str, Any]] = []
    first_src = group[0]["source"]
    src_steps_x, src_steps_y = _endpoint_to_steps(
        first_src,
        mapping_constants,
        motor_board_orientation,
    )

    pickup_cmd = f"PICKUP_STEPS {src_steps_x} {src_steps_y}"
    pickup_reply = _send_command(ser, pickup_cmd)
    executed.append(
        {
            "command_type": "PICKUP_STEPS",
            "cmd": pickup_cmd,
            "reply": pickup_reply,
            "source_steps": {"x": src_steps_x, "y": src_steps_y},
        }
    )

    final_planned_x = src_steps_x
    final_planned_y = src_steps_y
    for idx, move in enumerate(group, start=1):
        dst_steps_x, dst_steps_y = _endpoint_to_steps(
            move["dest"],
            mapping_constants,
            motor_board_orientation,
        )
        final_planned_x = dst_steps_x
        final_planned_y = dst_steps_y
        if idx < len(group):
            cmd = f"MOVEHELD_STEPS {dst_steps_x} {dst_steps_y}"
            command_type = "MOVEHELD_STEPS"
        else:
            cmd = f"RELEASE_STEPS {dst_steps_x} {dst_steps_y}"
            command_type = "RELEASE_STEPS"
        reply = _send_command(ser, cmd)
        executed.append(
            {
                "command_type": command_type,
                "cmd": cmd,
                "reply": reply,
                "dest_steps_planned": {"x": dst_steps_x, "y": dst_steps_y},
            }
        )

    move_status = _send_command(ser, "STATUS")
    status_x, status_y = _parse_status_xy(move_status)
    executed[-1]["status_after_command"] = move_status
    executed[-1]["dest_steps_status"] = {"x": status_x, "y": status_y}
    if status_x != final_planned_x or status_y != final_planned_y:
        executed[-1]["status_mismatch"] = True
    return executed, (status_x, status_y)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Send planned move sequence to STM32.")
    parser.add_argument(
        "--skip-return-start",
        action="store_true",
        help="Execute planned moves but leave the gantry at the last destination.",
    )
    parser.add_argument(
        "--return-start-only",
        action="store_true",
        help="Only issue RETURN_START and report the resulting position.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()

    if args.skip_return_start and args.return_start_only:
        print(json.dumps({"status": "error", "error": "Choose at most one of --skip-return-start or --return-start-only."}, indent=2))
        return 2

    if serial is None:
        print(json.dumps({"status": "error", "error": "pyserial is required"}, indent=2))
        return 1

    planned_moves: list[dict[str, Any]] = []
    if not args.return_start_only:
        try:
            planned_moves = _load_moves(MOVE_FILE)
        except Exception as exc:  # noqa: BLE001
            print(json.dumps({"status": "error", "error": str(exc)}, indent=2))
            return 1

    try:
        mapping_constants = _load_board_mapping_constants()
        motor_board_orientation = _load_motor_board_orientation()
        planned_stage_steps = (
            _build_stage_summary(planned_moves, mapping_constants, motor_board_orientation)
            if not args.return_start_only
            else []
        )
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
        "skip_return_start": bool(args.skip_return_start),
        "return_start_only": bool(args.return_start_only),
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
        chain_commands_supported: bool | None = None
        move_idx = 0
        if not args.return_start_only:
            planned_move_groups = _group_continuous_moves(planned_moves)
            result["planned_move_groups"] = [
                {
                    "group_index": group_idx,
                    "move_count": len(group),
                    "start": _endpoint_label(group[0]["source"]),
                    "end": _endpoint_label(group[-1]["dest"]),
                }
                for group_idx, group in enumerate(planned_move_groups, start=1)
            ]

            for group_idx, group in enumerate(planned_move_groups, start=1):
                group_executed: list[dict[str, Any]]
                final_status_x: int
                final_status_y: int

                if len(group) == 1:
                    group_executed, (final_status_x, final_status_y) = _run_legacy_move(
                        ser,
                        group[0],
                        mapping_constants=mapping_constants,
                        motor_board_orientation=motor_board_orientation,
                    )
                elif chain_commands_supported is False:
                    group_executed = []
                    for move in group:
                        move_executed, (final_status_x, final_status_y) = _run_legacy_move(
                            ser,
                            move,
                            mapping_constants=mapping_constants,
                            motor_board_orientation=motor_board_orientation,
                        )
                        group_executed.extend(move_executed)
                else:
                    try:
                        group_executed, (final_status_x, final_status_y) = _run_chained_group(
                            ser,
                            group,
                            mapping_constants=mapping_constants,
                            motor_board_orientation=motor_board_orientation,
                        )
                        chain_commands_supported = True
                    except RuntimeError as exc:
                        if "ERR CMD" not in str(exc):
                            raise
                        chain_commands_supported = False
                        group_executed = []
                        for move in group:
                            move_executed, (final_status_x, final_status_y) = _run_legacy_move(
                                ser,
                                move,
                                mapping_constants=mapping_constants,
                                motor_board_orientation=motor_board_orientation,
                            )
                            group_executed.extend(move_executed)

                result["executed"].append(
                    {
                        "group_index": group_idx,
                        "move_count": len(group),
                        "move_indices": list(range(move_idx + 1, move_idx + len(group) + 1)),
                        "commands": group_executed,
                        "magnet_continuous": len(group) > 1 and chain_commands_supported is True,
                    }
                )

                for move in group:
                    move_idx += 1
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
                    result["position_report"].append(
                        {
                            "stage": f"move_{move_idx}_source",
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
                    is_last_move_in_group = move is group[-1]
                    report_x = final_status_x if is_last_move_in_group else dst_steps_x
                    report_y = final_status_y if is_last_move_in_group else dst_steps_y
                    result["position_report"].append(
                        {
                            "stage": f"move_{move_idx}_dest",
                            "endpoint": dict(move["dest"]),
                            "endpoint_label": _endpoint_label(move["dest"]),
                            "motor_board_endpoint": _endpoint_to_motor_board(
                                move["dest"],
                                mapping_constants,
                                motor_board_orientation,
                            ),
                            "x": report_x,
                            "y": report_y,
                        }
                    )

                group_status = group_executed[-1].get("status_after_command") if group_executed else None
                if group_status:
                    result["status_checkpoints"].append(
                        {
                            "stage": f"after_group_{group_idx}",
                            "status": group_status,
                        }
                    )
                last_dest = {"x": final_status_x, "y": final_status_y}

            result["chain_commands_supported"] = chain_commands_supported

        if args.return_start_only:
            rz_cmd = "RETURN_START"
            rz_reply = _send_command(ser, rz_cmd)
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
        elif (not args.skip_return_start) and last_dest is not None and (last_dest["x"] != 0 or last_dest["y"] != 0):
            return_zero_move = {
                "source_steps": {"x": int(last_dest["x"]), "y": int(last_dest["y"])},
                "dest_steps": {"x": 0, "y": 0},
            }
            if RETURN_START_DELAY_SEC > 0:
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
        else:
            result["return_zero_executed"] = False

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
