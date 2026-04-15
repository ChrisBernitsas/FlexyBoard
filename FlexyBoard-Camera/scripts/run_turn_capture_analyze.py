#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from flexyboard_camera.app.end_turn_controller import EndTurnController
from flexyboard_camera.app.trigger import TriggerError, wait_for_gpio_trigger, wait_for_software_trigger
from flexyboard_camera.game.board_models import BoardCoord, BoardSpec
from flexyboard_camera.game.game_rules import ResponsePlanner
from flexyboard_camera.game.move_models import MoveEvent
from flexyboard_camera.utils.config import load_config
from flexyboard_camera.utils.logging_utils import setup_logging

ANALYSIS_LABEL_MODE = "index"
ANALYSIS_INNER_SHRINK = 0.02
ANALYSIS_DIFF_THRESHOLD = 45
ANALYSIS_MIN_CHANGED_RATIO = 0.28
ANALYSIS_OUTER_CANDIDATE_MODE = "auto"
ANALYSIS_DISABLE_TAPE_PROJECTION = True
ANALYSIS_BOARD_LOCK_SOURCE = "before"
ANALYSIS_GEOMETRY_REFERENCE = str(ROOT / "configs" / "before_geometry_reference.json")
ANALYSIS_DISABLE_GEOMETRY_REFERENCE = False
DEFAULT_CONFIG_PATH = str(ROOT / "configs" / "default.yaml")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Capture BEFORE, wait for trigger, capture AFTER, run board/square diff analysis, "
            "and produce STM32 command payload (optionally send)."
        )
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH, help="Path to YAML config")
    parser.add_argument("--wait-mode", choices=("enter", "gpio"), default="enter")
    parser.add_argument("--gpio-pin", type=int, default=17)
    parser.add_argument("--trigger-timeout", type=float, default=None)
    parser.add_argument("--analysis-out-dir", default=None)
    parser.add_argument(
        "--player2-response-json",
        default=None,
        help=(
            "Optional path to Player 2 software output JSON. "
            "If provided, this move sequence is used instead of the internal planner."
        ),
    )
    parser.add_argument(
        "--serial-port",
        default=None,
        help="Override serial port (for example /dev/ttyACM0) instead of comms.port from config",
    )
    parser.add_argument(
        "--serial-baudrate",
        type=int,
        default=None,
        help="Override serial baudrate instead of comms.baudrate from config",
    )
    parser.add_argument("--send", action="store_true", help="Actually send generated command to STM32")
    return parser.parse_args()


def _run_analysis(
    before_path: Path,
    after_path: Path,
    out_dir: str | None,
    game: str,
) -> dict[str, Any]:
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "analyze_board_and_diff.py"),
        "--before",
        str(before_path),
        "--after",
        str(after_path),
        "--game",
        str(game),
        "--label-mode",
        ANALYSIS_LABEL_MODE,
        "--inner-shrink",
        str(ANALYSIS_INNER_SHRINK),
        "--diff-threshold",
        str(ANALYSIS_DIFF_THRESHOLD),
        "--min-changed-ratio",
        str(ANALYSIS_MIN_CHANGED_RATIO),
        "--outer-candidate-mode",
        ANALYSIS_OUTER_CANDIDATE_MODE,
        "--board-lock-source",
        ANALYSIS_BOARD_LOCK_SOURCE,
        "--geometry-reference",
        ANALYSIS_GEOMETRY_REFERENCE,
    ]
    if ANALYSIS_DISABLE_TAPE_PROJECTION:
        cmd.append("--disable-tape-projection")
    if ANALYSIS_DISABLE_GEOMETRY_REFERENCE:
        cmd.append("--disable-geometry-reference")
    if out_dir is not None:
        cmd.extend(["--out-dir", out_dir])

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"analysis_failed: exit={result.returncode}, stdout={result.stdout.strip()}, stderr={result.stderr.strip()}"
        )
    return json.loads(result.stdout)


def _move_from_analysis(payload: dict[str, Any]) -> MoveEvent:
    move = payload["inferred_move"]

    def parse_coord(raw: dict[str, int] | None) -> BoardCoord | None:
        if raw is None:
            return None
        return BoardCoord(x=int(raw["x"]), y=int(raw["y"]))

    return MoveEvent(
        game=str(move.get("game", "chess")),
        source=parse_coord(move.get("source")),
        destination=parse_coord(move.get("destination")),
        moved_piece_type=move.get("moved_piece_type"),
        capture=move.get("capture"),
        confidence=float(move.get("confidence", 0.0)),
        timestamp=str(move.get("timestamp")),
        metadata=dict(move.get("metadata", {})),
    )


def _write_turn_decision_summary(
    *,
    out_path: Path,
    analysis: dict[str, Any],
    player1_observed_move: dict[str, Any],
    player2_plan_source: str,
    move_sequence: list[dict[str, Any]],
) -> None:
    changed = list(analysis.get("changed_squares", []))
    inferred = dict(player1_observed_move)

    lines: list[str] = []
    lines.append("FlexyBoard Turn Decision Summary")
    lines.append("")
    lines.append(f"changed_square_count: {len(changed)}")
    lines.append("changed_squares:")
    if changed:
        for item in changed:
            lines.append(
                "  - "
                f"index={item.get('index')} "
                f"coord=({item.get('x')},{item.get('y')}) "
                f"label={item.get('label')} "
                f"pixel_ratio={float(item.get('pixel_ratio', 0.0)):.4f} "
                f"delta={float(item.get('signed_intensity_delta', 0.0)):.3f}"
            )
    else:
        lines.append("  - none")

    src = inferred.get("source")
    dst = inferred.get("destination")
    src_text = f"({src.get('x')},{src.get('y')})" if isinstance(src, dict) else "None"
    dst_text = f"({dst.get('x')},{dst.get('y')})" if isinstance(dst, dict) else "None"

    lines.append("")
    lines.append("player1_observed_move:")
    lines.append(f"  - game: {inferred.get('game')}")
    lines.append(f"  - source: {src_text}")
    lines.append(f"  - destination: {dst_text}")
    lines.append(f"  - confidence: {float(inferred.get('confidence', 0.0)):.4f}")
    lines.append(f"  - metadata: {json.dumps(inferred.get('metadata', {}), ensure_ascii=True)}")

    lines.append("")
    lines.append(f"player2_plan_source: {player2_plan_source}")
    lines.append("")
    lines.append("player2_software_move_sequence_to_send_stm32:")
    if not move_sequence:
        lines.append("  - none")
    else:
        lines.append(json.dumps(move_sequence, ensure_ascii=True, indent=2))

    out_path.write_text("\n".join(lines), encoding="utf-8")


def _coerce_coord(raw: Any, field_name: str) -> dict[str, int]:
    if not isinstance(raw, dict):
        raise ValueError(f"Invalid {field_name}: expected object with x/y")
    if "x" not in raw or "y" not in raw:
        raise ValueError(f"Invalid {field_name}: missing x or y")
    return {"x": int(raw["x"]), "y": int(raw["y"])}


def _normalize_direct_step(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ValueError("Each move step must be an object")
    source_raw = raw.get("source")
    dest_raw = raw.get("dest", raw.get("destination"))
    if source_raw is None or dest_raw is None:
        raise ValueError("Each move step must include source and dest/destination")
    return {
        "source": _coerce_coord(source_raw, "source"),
        "dest": _coerce_coord(dest_raw, "dest"),
    }


def _expand_command_with_waypoints(raw: dict[str, Any]) -> list[dict[str, Any]]:
    source = _coerce_coord(raw.get("source"), "source")
    dest = _coerce_coord(raw.get("dest", raw.get("destination")), "dest")
    waypoints_raw = raw.get("waypoints", [])
    if not isinstance(waypoints_raw, list):
        raise ValueError("waypoints must be a list")

    route: list[dict[str, int]] = [source]
    route.extend(_coerce_coord(item, "waypoint") for item in waypoints_raw)
    route.append(dest)

    steps: list[dict[str, Any]] = []
    for idx in range(len(route) - 1):
        steps.append({"source": route[idx], "dest": route[idx + 1]})
    return steps


def _load_player2_move_sequence(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))

    sequence_obj: Any = payload
    if isinstance(payload, dict):
        for key in ("player2_software_move_sequence", "move_sequence", "stm32_move_sequence", "steps", "moves"):
            if key in payload:
                sequence_obj = payload[key]
                break

    if isinstance(sequence_obj, dict):
        return _expand_command_with_waypoints(sequence_obj)

    if not isinstance(sequence_obj, list):
        raise ValueError("External Player 2 response must be a move object or list of move steps")

    normalized: list[dict[str, Any]] = []
    for item in sequence_obj:
        if isinstance(item, dict) and "waypoints" in item:
            normalized.extend(_expand_command_with_waypoints(item))
        else:
            normalized.append(_normalize_direct_step(item))
    return normalized


def main() -> int:
    args = _parse_args()
    config = load_config(args.config)
    if args.serial_port:
        config.comms.port = args.serial_port
    if args.serial_baudrate:
        config.comms.baudrate = int(args.serial_baudrate)
    setup_logging(config.paths.logs_dir)

    controller = EndTurnController(config)
    board_spec = BoardSpec(
        game=config.app.game,
        width=config.vision.board_size[0],
        height=config.vision.board_size[1],
        square_size_mm=config.board.square_size_mm,
        origin_offset_mm_x=config.board.origin_offset_mm[0],
        origin_offset_mm_y=config.board.origin_offset_mm[1],
    )
    planner = ResponsePlanner(board_spec)

    before_path = controller.capture_before()

    if args.wait_mode == "gpio":
        try:
            triggered = wait_for_gpio_trigger(pin=args.gpio_pin, timeout_sec=args.trigger_timeout)
        except TriggerError as exc:
            print(json.dumps({"status": "error", "error": str(exc)}, indent=2))
            return 1
        if not triggered:
            print(json.dumps({"status": "trigger_timeout"}, indent=2))
            return 1
    else:
        wait_for_software_trigger()

    after_path = controller.capture_after()

    try:
        analysis = _run_analysis(
            before_path=before_path,
            after_path=after_path,
            out_dir=args.analysis_out_dir,
            game=config.app.game,
        )
    except Exception as exc:  # noqa: BLE001
        print(
            json.dumps(
                {
                    "status": "error",
                    "error": str(exc),
                    "before_image": str(before_path),
                    "after_image": str(after_path),
                },
                indent=2,
            )
        )
        return 1

    move_event = _move_from_analysis(analysis)
    player1_observed_move = move_event.to_dict()
    analysis_root = Path(
        analysis.get("analysis_root_dir", Path(analysis["outputs"]["after_grid_overlay"]).parent)
    )

    player1_observed_move_json_path = analysis_root / "player1_observed_move.json"
    player1_observed_move_json_path.write_text(
        json.dumps(player1_observed_move, ensure_ascii=True, indent=2), encoding="utf-8"
    )

    player2_plan_source = "internal_planner"
    move_sequence: list[dict[str, Any]] = []
    if args.player2_response_json:
        try:
            move_sequence = _load_player2_move_sequence(Path(args.player2_response_json))
            player2_plan_source = f"external_file:{args.player2_response_json}"
        except Exception as exc:  # noqa: BLE001
            print(
                json.dumps(
                    {
                        "status": "error",
                        "error": f"invalid_player2_response_json: {exc}",
                        "player2_response_json": args.player2_response_json,
                        "analysis_dir": str(analysis_root),
                    },
                    indent=2,
                )
            )
            return 1
    else:
        response_plan = planner.propose_response_plan(move_event) or []
        for command in response_plan:
            move_sequence.extend(command.to_step_payloads(minimal_for_stm32=True))

    player2_software_move_sequence_json_path = analysis_root / "player2_software_move_sequence.json"
    player2_software_move_sequence_json_path.write_text(
        json.dumps(move_sequence, ensure_ascii=True, indent=2), encoding="utf-8"
    )

    sent_to_stm32 = False
    stm32_status: dict[str, Any] | None = None
    stm32_step_statuses: list[dict[str, Any]] = []
    if args.send and move_sequence:
        if config.safety.auto_home_before_move:
            controller._stm32.home()  # noqa: SLF001
        for step_payload in move_sequence:
            responses = controller._stm32.execute_move(step_payload)  # noqa: SLF001
            if responses:
                stm32_step_statuses.append(responses[-1].to_dict())
        sent_to_stm32 = True
        stm32_status = stm32_step_statuses[-1] if stm32_step_statuses else None

    decision_summary_path = analysis_root / "turn_decision_summary.txt"
    _write_turn_decision_summary(
        out_path=decision_summary_path,
        analysis=analysis,
        player1_observed_move=player1_observed_move,
        player2_plan_source=player2_plan_source,
        move_sequence=move_sequence,
    )
    result = {
        "status": "ok",
        "before_image": str(before_path),
        "after_image": str(after_path),
        "analysis_dir": str(analysis_root),
        "analysis_json": str(analysis_root / "analysis.json"),
        "official_dir": str(analysis.get("official_dir", analysis_root / "official")),
        "algorithm_live_dir": str(analysis.get("algorithm_live_dir", analysis_root / "algorithm_live")),
        "player1_observed_move_json": str(player1_observed_move_json_path),
        "player2_software_move_sequence_json": str(player2_software_move_sequence_json_path),
        "player2_plan_source": player2_plan_source,
        "turn_decision_summary_txt": str(decision_summary_path),
        "analysis_settings": {
            "game": config.app.game,
            "label_mode": ANALYSIS_LABEL_MODE,
            "inner_shrink": ANALYSIS_INNER_SHRINK,
            "diff_threshold": ANALYSIS_DIFF_THRESHOLD,
            "min_changed_ratio": ANALYSIS_MIN_CHANGED_RATIO,
            "outer_candidate_mode": ANALYSIS_OUTER_CANDIDATE_MODE,
            "disable_tape_projection": ANALYSIS_DISABLE_TAPE_PROJECTION,
            "board_lock_source": ANALYSIS_BOARD_LOCK_SOURCE,
            "geometry_reference": ANALYSIS_GEOMETRY_REFERENCE,
            "disable_geometry_reference": ANALYSIS_DISABLE_GEOMETRY_REFERENCE,
        },
        "changed_square_count": int(analysis.get("changed_square_count", 0)),
        "changed_squares": analysis.get("changed_squares", []),
        "inferred_move": analysis.get("inferred_move"),  # legacy key (same as player1_observed_move)
        "player1_observed_move": player1_observed_move,
        "player2_software_move_sequence": move_sequence,
        "player2_software_move_sequence_count": len(move_sequence),
        "stm32_command_payload": move_sequence[0] if move_sequence else None,
        "stm32_move_sequence": move_sequence,
        "stm32_move_sequence_count": len(move_sequence),
        "sent_to_stm32": sent_to_stm32,
        "stm32_status": stm32_status,
        "stm32_step_statuses": stm32_step_statuses,
    }
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
