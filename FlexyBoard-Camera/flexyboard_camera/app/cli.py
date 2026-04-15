from __future__ import annotations

import argparse
import json

from flexyboard_camera.app.end_turn_controller import EndTurnController
from flexyboard_camera.app.trigger import TriggerError, wait_for_gpio_trigger, wait_for_software_trigger
from flexyboard_camera.game.board_models import BoardCoord
from flexyboard_camera.game.move_models import MoveEvent
from flexyboard_camera.utils.config import load_config
from flexyboard_camera.utils.logging_utils import setup_logging


def _print_json(payload: dict) -> None:
    print(json.dumps(payload, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="FlexyBoard camera/CV control CLI")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to YAML config")

    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("calibrate", help="Compute and save calibration from configured ROI")

    capture_before = sub.add_parser("capture_before", help="Capture before-move frame")
    capture_before.add_argument("--image", help="Optional image file path to use instead of live camera")

    capture_after = sub.add_parser("capture_after", help="Capture after-move frame")
    capture_after.add_argument("--image", help="Optional image file path to use instead of live camera")

    infer_cmd = sub.add_parser("infer_move", help="Infer move from before/after images")
    infer_cmd.add_argument("--before", help="Before image path")
    infer_cmd.add_argument("--after", help="After image path")

    send_cmd = sub.add_parser("send_move", help="Send an explicit move command to STM32")
    send_cmd.add_argument("--game", default="chess")
    send_cmd.add_argument("--sx", required=True, type=int)
    send_cmd.add_argument("--sy", required=True, type=int)
    send_cmd.add_argument("--dx", required=True, type=int)
    send_cmd.add_argument("--dy", required=True, type=int)

    cycle_cmd = sub.add_parser("run_end_turn_cycle", help="Run infer + response send flow")
    cycle_cmd.add_argument("--before", help="Before image path")
    cycle_cmd.add_argument("--after", help="After image path")
    cycle_cmd.add_argument("--force", action="store_true", help="Override low-confidence block")
    cycle_cmd.add_argument("--wait-trigger", action="store_true", help="Wait for trigger before cycle")
    cycle_cmd.add_argument("--gpio", action="store_true", help="Use GPIO trigger mode")
    cycle_cmd.add_argument("--gpio-pin", type=int, default=17, help="GPIO pin for end-turn button")
    cycle_cmd.add_argument("--trigger-timeout", type=float, help="Optional trigger timeout in seconds")

    sub.add_parser("ping", help="Ping STM32")
    sub.add_parser("status", help="Query STM32 status")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    config = load_config(args.config)
    setup_logging(config.paths.logs_dir)

    controller = EndTurnController(config)

    if args.command == "calibrate":
        result = controller.calibrate()
        _print_json({"saved": config.paths.calibration_file, "created_at": result.created_at})
        return 0

    if args.command == "capture_before":
        output = controller.capture_before(image_path=args.image)
        _print_json({"before_image": str(output)})
        return 0

    if args.command == "capture_after":
        output = controller.capture_after(image_path=args.image)
        _print_json({"after_image": str(output)})
        return 0

    if args.command == "infer_move":
        move, artifacts = controller.infer_move(before_path=args.before, after_path=args.after)
        payload = {"move": move.to_dict(), "artifacts_dir": str(artifacts.run_dir)}
        _print_json(payload)
        return 0

    if args.command == "send_move":
        move = MoveEvent(
            game=args.game,
            source=BoardCoord(args.sx, args.sy),
            destination=BoardCoord(args.dx, args.dy),
            moved_piece_type=None,
            capture=False,
            confidence=1.0,
        )
        responses = controller.send_move(move)
        _print_json({"responses": responses})
        return 0

    if args.command == "run_end_turn_cycle":
        if args.wait_trigger:
            if args.gpio:
                try:
                    triggered = wait_for_gpio_trigger(pin=args.gpio_pin, timeout_sec=args.trigger_timeout)
                except TriggerError as exc:
                    _print_json({"error": str(exc)})
                    return 1
                if not triggered:
                    _print_json({"status": "trigger_timeout"})
                    return 1
            else:
                wait_for_software_trigger()

        result = controller.run_end_turn_cycle(
            before_path=args.before,
            after_path=args.after,
            force_low_confidence=args.force,
        )
        _print_json(result.to_dict())
        return 0

    if args.command == "ping":
        responses = controller._stm32.ping()  # noqa: SLF001 - intentionally exposed for CLI tooling.
        _print_json({"responses": [item.to_dict() for item in responses]})
        return 0

    if args.command == "status":
        responses = controller._stm32.get_status()  # noqa: SLF001 - intentionally exposed for CLI tooling.
        _print_json({"responses": [item.to_dict() for item in responses]})
        return 0

    parser.error(f"Unknown command: {args.command}")
    return 2
