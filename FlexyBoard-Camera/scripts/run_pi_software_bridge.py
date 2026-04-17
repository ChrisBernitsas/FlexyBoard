#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from flexyboard_camera.app.end_turn_controller import EndTurnController
from flexyboard_camera.game.legal_move_resolver import Player1MoveResolver
from flexyboard_camera.app.trigger import TriggerError, wait_for_gpio_trigger
from flexyboard_camera.utils.config import load_config
from flexyboard_camera.utils.logging_utils import setup_logging

MOVE_FILE = ROOT / "sample_data" / "stm32_move_sequence.txt"
DEFAULT_CONFIG_PATH = ROOT / "configs" / "default.yaml"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run end-to-end bridge on Pi: capture/analysis -> send p1_move to Software-GUI over TCP -> "
            "receive p2_move (+planner sequence) -> send sequence to STM32."
        )
    )
    parser.add_argument("--host", default="0.0.0.0", help="TCP listen host for Software-GUI client")
    parser.add_argument("--port", type=int, default=8765, help="TCP listen port for Software-GUI client")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="FlexyBoard-Camera YAML config path")
    parser.add_argument("--wait-mode", choices=("enter", "gpio"), default="enter")
    parser.add_argument("--gpio-pin", type=int, default=17)
    parser.add_argument("--trigger-timeout", type=float, default=None)
    parser.add_argument("--analysis-out-dir", default=None)
    parser.add_argument("--serial-port", default=None)
    parser.add_argument("--serial-baudrate", type=int, default=None)
    parser.add_argument(
        "--capture-mode",
        choices=("rolling", "two-shot"),
        default="rolling",
        help=(
            "rolling: capture one initial reference, then one image after each P1 turn. "
            "two-shot: capture before and after every turn."
        ),
    )
    parser.add_argument("--once", action="store_true", help="Run exactly one full turn")
    parser.add_argument(
        "--no-stm-send",
        action="store_true",
        help="Do not execute STM sequence; only print and persist move file",
    )
    return parser.parse_args()


def _board_xy_to_square(x: int, y: int) -> str:
    if x < 0 or x > 7 or y < 0 or y > 7:
        raise ValueError(f"Board coord out of range: ({x},{y})")
    files = "abcdefgh"
    ranks = "12345678"
    return files[x] + ranks[y]


def _square_to_board_xy(square_id: str) -> tuple[int, int]:
    s = square_id.strip().lower()
    if len(s) != 2:
        raise ValueError(f"invalid square id: {square_id!r}")
    files = "abcdefgh"
    ranks = "12345678"
    if s[0] not in files or s[1] not in ranks:
        raise ValueError(f"invalid square id: {square_id!r}")
    return files.index(s[0]), ranks.index(s[1])


def _repo_relative_path(path_text: str) -> str:
    path = Path(path_text)
    if path.is_absolute():
        return str(path)
    return str(ROOT / path)


def _run_analysis(
    before_path: Path,
    after_path: Path,
    out_dir: str | None,
    game: str,
    analysis_config: Any,
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
        analysis_config.label_mode,
        "--inner-shrink",
        str(analysis_config.inner_shrink),
        "--diff-threshold",
        str(analysis_config.diff_threshold),
        "--min-changed-ratio",
        str(analysis_config.min_changed_ratio),
        "--outer-candidate-mode",
        analysis_config.outer_candidate_mode,
        "--board-lock-source",
        analysis_config.board_lock_source,
        "--geometry-reference",
        _repo_relative_path(analysis_config.geometry_reference),
    ]
    if analysis_config.disable_tape_projection:
        cmd.append("--disable-tape-projection")
    if analysis_config.disable_geometry_reference:
        cmd.append("--disable-geometry-reference")
    if out_dir:
        cmd.extend(["--out-dir", out_dir])

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"analyze_board_and_diff failed (exit={result.returncode})\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            "Failed to parse analyze_board_and_diff JSON output\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        ) from exc


def _wait_for_turn_trigger(args: argparse.Namespace, prompt: str) -> None:
    if args.wait_mode == "gpio":
        try:
            triggered = wait_for_gpio_trigger(pin=args.gpio_pin, timeout_sec=args.trigger_timeout)
        except TriggerError as exc:
            raise RuntimeError(str(exc)) from exc
        if not triggered:
            raise RuntimeError("gpio trigger timeout")
    else:
        input(prompt)


def _analyze_paths(
    before_path: Path,
    after_path: Path,
    args: argparse.Namespace,
    game: str,
    analysis_config: Any,
) -> dict[str, Any]:
    analysis = _run_analysis(
        before_path=before_path,
        after_path=after_path,
        out_dir=args.analysis_out_dir,
        game=game,
        analysis_config=analysis_config,
    )
    analysis_root = Path(
        analysis.get("analysis_root_dir", Path(analysis["outputs"]["after_grid_overlay"]).parent)
    )
    inferred = analysis.get("inferred_move", {})
    if isinstance(inferred, dict):
        (analysis_root / "player1_observed_move.json").write_text(
            json.dumps(inferred, ensure_ascii=True, indent=2),
            encoding="utf-8",
        )
    return {
        "status": "ok",
        "before_image": str(before_path),
        "after_image": str(after_path),
        "analysis_dir": str(analysis_root),
        "analysis": analysis,
        "player1_observed_move": inferred,
    }


def _capture_and_analyze_two_shot(
    controller: EndTurnController,
    args: argparse.Namespace,
    game: str,
    analysis_config: Any,
) -> dict[str, Any]:
    before_path = controller.capture_before()
    print(f"[Bridge] BEFORE captured: {before_path}")

    _wait_for_turn_trigger(args, "[Bridge] Move Player 1 piece, then press Enter to capture AFTER...")

    after_path = controller.capture_after()
    print(f"[Bridge] AFTER captured: {after_path}")

    return _analyze_paths(before_path, after_path, args, game, analysis_config)


def _capture_and_analyze_rolling(
    controller: EndTurnController,
    args: argparse.Namespace,
    game: str,
    reference_path: Path,
    turn_index: int,
    analysis_config: Any,
) -> dict[str, Any]:
    _wait_for_turn_trigger(
        args,
        f"[Bridge] Turn {turn_index}: make Player 1 move, then press Enter to capture current board...",
    )

    after_path = controller.capture_after()
    print(f"[Bridge] CURRENT captured: {after_path}")
    print(f"[Bridge] Diffing previous reference -> current: {reference_path} -> {after_path}")

    return _analyze_paths(reference_path, after_path, args, game, analysis_config)


def _send_moves_to_stm() -> dict[str, Any]:
    cmd = [sys.executable, str(ROOT / "scripts" / "send_moves_from_file.py")]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"send_moves_from_file failed (exit={result.returncode})\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )

    # send_moves_from_file.py prints a small text report before trailing JSON.
    text = result.stdout.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    lines = text.splitlines()
    for i, line in enumerate(lines):
        if not line.lstrip().startswith("{"):
            continue
        candidate = "\n".join(lines[i:])
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue

    raise RuntimeError(
        "Failed to parse send_moves_from_file JSON output\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )


def _read_json_line(conn_file: Any) -> dict[str, Any]:
    raw = conn_file.readline()
    if raw == b"":
        raise ConnectionError("GUI disconnected")
    line = raw.decode("utf-8", errors="replace").strip()
    if not line:
        return _read_json_line(conn_file)
    payload = json.loads(line)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object line, got: {payload!r}")
    return payload


def _write_json_line(conn_file: Any, obj: dict[str, Any]) -> None:
    conn_file.write((json.dumps(obj, separators=(",", ":")) + "\n").encode("utf-8"))
    conn_file.flush()


def _normalize_sequence_lines_from_p2(msg: dict[str, Any]) -> list[str]:
    seq = msg.get("stm_sequence")
    if isinstance(seq, list):
        lines = []
        for item in seq:
            if not isinstance(item, str):
                raise ValueError("stm_sequence must be a list of strings")
            content = item.strip()
            if content:
                lines.append(content)
        if lines:
            return lines

    frm = msg.get("from")
    to = msg.get("to")
    if not isinstance(frm, str) or not isinstance(to, str):
        raise ValueError("p2_move requires either stm_sequence or from/to")
    sx, sy = _square_to_board_xy(frm)
    dx, dy = _square_to_board_xy(to)
    return [f"{sx},{sy} -> {dx},{dy}"]


def _write_move_file(lines: list[str], game: str | None, p2_from: str | None, p2_to: str | None) -> None:
    MOVE_FILE.parent.mkdir(parents=True, exist_ok=True)
    header = [
        "# Auto-generated by run_pi_software_bridge.py",
        f"# game={game or 'unknown'} p2_move={p2_from or '?'}->{p2_to or '?'}",
        "# Format: source -> dest (board: x,y  |  off-board: x%,y%)",
    ]
    MOVE_FILE.write_text("\n".join([*header, *lines]) + "\n", encoding="utf-8")


def _serve_client(
    args: argparse.Namespace,
    conn: socket.socket,
    controller: EndTurnController,
    game: str,
    resolver: Player1MoveResolver,
    analysis_config: Any,
) -> None:
    conn.settimeout(None)
    with conn:
        conn_file = conn.makefile("rwb")
        turn_index = 1
        reference_path: Path | None = None

        if args.capture_mode == "rolling":
            input("[Bridge] Make sure the physical board matches the software state, then press Enter to capture INITIAL reference...")
            reference_path = controller.capture_before()
            print(f"[Bridge] Initial reference captured: {reference_path}")

        while True:
            print("")
            if args.capture_mode == "rolling":
                if reference_path is None:
                    raise RuntimeError("rolling capture mode missing reference image")
                analysis_wrap = _capture_and_analyze_rolling(
                    controller=controller,
                    args=args,
                    game=game,
                    reference_path=reference_path,
                    turn_index=turn_index,
                    analysis_config=analysis_config,
                )
            else:
                input(f"[Bridge] Turn {turn_index}: press Enter to capture BEFORE...")
                analysis_wrap = _capture_and_analyze_two_shot(controller, args, game, analysis_config)

            observed_move = analysis_wrap.get("player1_observed_move")
            analysis_payload = analysis_wrap.get("analysis", {})
            if not isinstance(observed_move, dict):
                print("[Bridge] Skipping turn: missing player1_observed_move.")
                turn_index += 1
                if args.once:
                    return
                continue

            changed_squares: list[dict[str, Any]] = []
            raw_changed = analysis_payload.get("changed_squares")
            if isinstance(raw_changed, list):
                changed_squares = [item for item in raw_changed if isinstance(item, dict)]

            resolved = resolver.resolve_player1(observed_move, changed_squares)
            if resolved is None:
                resolved = resolver.fallback_from_observed(observed_move)
            if resolved is None:
                print("[Bridge] Skipping turn: could not resolve legal player-1 move.")
                turn_index += 1
                if args.once:
                    return
                continue

            analysis_dir_raw = analysis_wrap.get("analysis_dir")
            if isinstance(analysis_dir_raw, str):
                analysis_dir = Path(analysis_dir_raw)
                (analysis_dir / "player1_resolved_move.json").write_text(
                    json.dumps(resolved.to_dict(), ensure_ascii=True, indent=2),
                    encoding="utf-8",
                )

            print(
                f"[Bridge] Resolved P1 move with {resolved.resolver}: "
                f"steps={len(resolved.steps)} capture={resolved.capture} "
                f"special={resolved.special} score={resolved.score}"
            )
            for idx, step in enumerate(resolved.steps, start=1):
                p1_from, p1_to = resolver.step_to_square_pair(step)
                p1_msg = {"type": "p1_move", "from": p1_from, "to": p1_to}
                _write_json_line(conn_file, p1_msg)
                print(f"[Bridge] Sent P1 step {idx}/{len(resolved.steps)}: {p1_from} -> {p1_to}")

            while True:
                incoming = _read_json_line(conn_file)
                msg_type = incoming.get("type")
                if msg_type != "p2_move":
                    print(f"[Bridge] Ignoring non-p2 message: {incoming}")
                    continue
                break

            p2_from = incoming.get("from")
            p2_to = incoming.get("to")
            print(f"[Bridge] Received P2 move from Software-GUI: {p2_from} -> {p2_to}")

            if isinstance(p2_from, str) and isinstance(p2_to, str):
                try:
                    resolver.apply_player2(p2_from, p2_to)
                except Exception as exc:  # noqa: BLE001
                    print(f"[Bridge] Warning: failed to apply P2 move in resolver state: {exc}")

            sequence_lines = _normalize_sequence_lines_from_p2(incoming)
            _write_move_file(sequence_lines, incoming.get("game"), p2_from, p2_to)
            print(f"[Bridge] Wrote {len(sequence_lines)} STM sequence steps to {MOVE_FILE}")

            if args.no_stm_send:
                print("[Bridge] --no-stm-send enabled; skipping STM dispatch.")
                if args.capture_mode == "rolling":
                    reference_path = Path(str(analysis_wrap["after_image"]))
                    print(f"[Bridge] Rolling reference advanced without STM send: {reference_path}")
            else:
                print("[Bridge] Sending sequence to STM32...")
                stm_result = _send_moves_to_stm()
                print(json.dumps({"bridge_stm_result": stm_result}, indent=2))
                if args.capture_mode == "rolling":
                    print("[Bridge] Capturing updated reference after STM32 move...")
                    reference_path = controller.capture_before()
                    print(f"[Bridge] Rolling reference refreshed: {reference_path}")

            turn_index += 1
            if args.once:
                return


def main() -> int:
    args = _parse_args()
    config = load_config(args.config)
    if args.serial_port:
        config.comms.port = args.serial_port
    if args.serial_baudrate is not None:
        config.comms.baudrate = int(args.serial_baudrate)
    setup_logging(config.paths.logs_dir)
    controller = EndTurnController(config)
    resolver = Player1MoveResolver(config.app.game)

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as srv:
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind((args.host, args.port))
        srv.listen(1)
        print(f"[Bridge] Listening for Software-GUI on {args.host}:{args.port}")
        print("[Bridge] Start Software-GUI now; it should connect as TCP client.")
        conn, addr = srv.accept()
        print(f"[Bridge] Software-GUI connected from {addr[0]}:{addr[1]}")
        try:
            _serve_client(args, conn, controller, config.app.game, resolver, config.analysis)
        except ConnectionError as exc:
            print(f"[Bridge] Connection closed: {exc}")
            return 1
        except Exception as exc:  # noqa: BLE001
            print(f"[Bridge] Error: {exc}")
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
