#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
from datetime import datetime
import json
import re
import select
import shutil
import socket
import struct
import subprocess
import sys
import time
import threading
from queue import Empty, Queue
from pathlib import Path
from typing import Any

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from flexyboard_camera.app.end_turn_controller import EndTurnController
from flexyboard_camera.game.legal_move_resolver import Player1MoveResolver
from flexyboard_camera.app.trigger import TriggerError, wait_for_gpio_trigger
from flexyboard_camera.utils.config import load_config
from flexyboard_camera.utils.logging_utils import setup_logging
from flexyboard_camera.vision.board_detector import (
    BoardDetection,
    draw_detection_overlay,
    draw_square_grid_overlay,
    generate_square_geometry,
)

MOVE_FILE = ROOT / "sample_data" / "stm32_move_sequence.txt"
DEFAULT_CONFIG_PATH = ROOT / "configs" / "default.yaml"
MANUAL_CORNERS_INFO_PATH = ROOT / "configs" / "corners_info.json"
MANUAL_CORNERS_SAMPLES_DIR = ROOT / "configs" / "all_manual"


class _TeeStream:
    def __init__(self, primary: object, mirror: object) -> None:
        self._primary = primary
        self._mirror = mirror

    def write(self, data: str) -> int:
        self._primary.write(data)
        self._mirror.write(data)
        return len(data)

    def flush(self) -> None:
        self._primary.flush()
        self._mirror.flush()

    def isatty(self) -> bool:
        return bool(getattr(self._primary, "isatty", lambda: False)())


def _install_runtime_log(game_debug_dir: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = game_debug_dir / f"bridge_runtime_{timestamp}.log"
    handle = log_path.open("a", encoding="utf-8")
    sys.stdout = _TeeStream(sys.stdout, handle)  # type: ignore[assignment]
    sys.stderr = _TeeStream(sys.stderr, handle)  # type: ignore[assignment]
    return log_path


def _next_game_debug_dir(debug_root: Path) -> Path:
    debug_root.mkdir(parents=True, exist_ok=True)
    max_index = 0
    pattern = re.compile(r"^Game(\d+)$")
    for child in debug_root.iterdir():
        if not child.is_dir():
            continue
        match = pattern.fullmatch(child.name)
        if match:
            max_index = max(max_index, int(match.group(1)))
    game_dir = debug_root / f"Game{max_index + 1}"
    game_dir.mkdir(parents=True, exist_ok=False)
    return game_dir


def _write_game_session_metadata(game_dir: Path, *, game: str, args: argparse.Namespace) -> None:
    payload = {
        "game": game,
        "capture_mode": args.capture_mode,
        "wait_mode": args.wait_mode,
        "bridge_port": args.port,
        "startup_geometry_mode": args.startup_geometry_mode,
        "slow_live_analysis": bool(args.slow_live_analysis),
        "reopen_camera_each_capture": bool(args.reopen_camera_each_capture),
        "min_significant_changes": args.min_significant_changes,
        "max_significant_changes": args.max_significant_changes,
        "max_resolved_score": args.max_resolved_score,
        "allow_observed_fallback": bool(args.allow_observed_fallback),
        "max_p1_detection_attempts": args.max_p1_detection_attempts,
    }
    (game_dir / "session_metadata.json").write_text(
        json.dumps(payload, ensure_ascii=True, indent=2),
        encoding="utf-8",
    )


def _copy_frame_for_turn(src: Path, dst: Path) -> Path:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return dst


def _normalize_game_name(raw: object, default: str = "chess") -> str:
    game = str(raw or default).strip().lower()
    if game not in {"chess", "checkers", "parcheesi"}:
        return default
    return game


def _load_scaled_corners_payload(
    *,
    corners_path: Path,
    target_image_w: int,
    target_image_h: int,
) -> dict[str, Any]:
    payload = json.loads(corners_path.read_text(encoding="utf-8"))
    stored_size = payload.get("image_size_px") if isinstance(payload.get("image_size_px"), dict) else {}
    try:
        stored_w = max(1.0, float(stored_size.get("width", target_image_w)))
        stored_h = max(1.0, float(stored_size.get("height", target_image_h)))
    except Exception:  # noqa: BLE001
        stored_w = float(target_image_w)
        stored_h = float(target_image_h)

    def scale_quad(key: str) -> list[list[float]] | None:
        raw = payload.get(key)
        if not isinstance(raw, list) or len(raw) != 4:
            return None
        out: list[list[float]] = []
        for point in raw:
            if not isinstance(point, list) or len(point) != 2:
                return None
            out.append([
                float(point[0]) / stored_w * float(target_image_w),
                float(point[1]) / stored_h * float(target_image_h),
            ])
        return out

    scaled_outer = scale_quad("outer_sheet_corners_px")
    scaled_chess = scale_quad("chessboard_corners_px")
    if scaled_chess is None:
        raise RuntimeError(f"Invalid chessboard corners in {corners_path}")

    payload = dict(payload)
    payload["outer_sheet_corners_px"] = scaled_outer
    payload["chessboard_corners_px"] = scaled_chess
    payload["image_size_px"] = {"width": int(target_image_w), "height": int(target_image_h)}
    return payload


def _capture_startup_reference(controller: EndTurnController, *, output_path: Path | None = None) -> Path:
    settings = controller._camera.settings
    original_frames = int(settings.pre_capture_flush_frames)
    original_delay = float(settings.pre_capture_flush_delay_sec)
    settings.pre_capture_flush_frames = max(original_frames, 10)
    settings.pre_capture_flush_delay_sec = max(original_delay, 0.02)
    try:
        return controller.capture_before(reopen_stream=True, output_path=output_path)
    finally:
        settings.pre_capture_flush_frames = original_frames
        settings.pre_capture_flush_delay_sec = original_delay


def _capture_live_reference(
    controller: EndTurnController,
    *,
    output_path: Path | None = None,
    reopen_stream: bool = False,
    before: bool = True,
) -> Path:
    settings = controller._camera.settings
    original_frames = int(settings.pre_capture_flush_frames)
    original_delay = float(settings.pre_capture_flush_delay_sec)
    settings.pre_capture_flush_frames = min(max(original_frames, 0), 1)
    settings.pre_capture_flush_delay_sec = 0.0
    try:
        if before:
            return controller.capture_before(reopen_stream=reopen_stream, output_path=output_path)
        return controller.capture_after(reopen_stream=reopen_stream, output_path=output_path)
    finally:
        settings.pre_capture_flush_frames = original_frames
        settings.pre_capture_flush_delay_sec = original_delay


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
    parser.add_argument(
        "--slow-live-analysis",
        action="store_true",
        help="Disable fast live mode and redetect board geometry every turn.",
    )
    parser.add_argument(
        "--reopen-camera-each-capture",
        action="store_true",
        help="Close/reopen the camera for every capture. Default keeps the camera open for lower latency.",
    )
    parser.add_argument(
        "--session-geometry-path",
        default=str(ROOT / "debug_output" / "live_session_geometry.json"),
        help="Where to store the geometry reference detected from the initial rolling reference frame.",
    )
    parser.add_argument(
        "--startup-geometry-mode",
        choices=("manual", "auto"),
        default="auto",
        help=(
            "manual: send the initial board image to Software-GUI for corner clicks. "
            "auto: detect startup geometry automatically, then confirm or fall back to manual clicks."
        ),
    )
    parser.add_argument(
        "--skip-geometry-confirmation",
        action="store_true",
        help="Auto geometry mode only: do not send the startup grid preview to Software-GUI for confirmation.",
    )
    parser.add_argument(
        "--min-significant-changes",
        type=int,
        default=2,
        help=(
            "Minimum strong changed squares required before accepting a Player 1 move. "
            "Below this, the bridge retries the same turn without advancing the reference."
        ),
    )
    parser.add_argument(
        "--max-resolved-score",
        type=float,
        default=20.0,
        help=(
            "Maximum legal-resolver mismatch score accepted for Player 1. "
            "Higher scores are treated as noisy/invalid detections and retried."
        ),
    )
    parser.add_argument(
        "--max-significant-changes",
        type=int,
        default=12,
        help=(
            "Reject captures with more than this many significant changed squares. "
            "Large values usually indicate a hand/arm or scene obstruction rather than a real move. "
            "0 disables this guard."
        ),
    )
    parser.add_argument(
        "--allow-observed-fallback",
        action="store_true",
        help="Allow forwarding the raw CV observed move when no legal resolver match is accepted.",
    )
    parser.add_argument(
        "--max-p1-detection-attempts",
        type=int,
        default=0,
        help="Maximum rejected Player 1 detection attempts before stopping. 0 means retry forever.",
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


def _order_quad(points: Any) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    if pts.shape != (4, 2):
        raise ValueError("quadrilateral must have 4 points")
    sums = pts.sum(axis=1)
    diffs = np.diff(pts, axis=1).reshape(-1)
    ordered = np.zeros((4, 2), dtype=np.float32)
    ordered[0] = pts[np.argmin(sums)]  # TL
    ordered[2] = pts[np.argmax(sums)]  # BR
    ordered[1] = pts[np.argmin(diffs)]  # TR
    ordered[3] = pts[np.argmax(diffs)]  # BL
    return ordered


def _crop_for_analysis(
    image: np.ndarray,
    crop_info: dict[str, Any] | None,
    which: str,
) -> np.ndarray:
    if not crop_info or not bool(crop_info.get("enabled")):
        return image
    region_raw = crop_info.get(which)
    if not isinstance(region_raw, dict):
        return image
    h, w = image.shape[:2]
    try:
        x0 = int(region_raw.get("x0_px", 0))
        x1 = int(region_raw.get("x1_px", w))
        y0 = int(region_raw.get("y0_px", 0))
        y1 = int(region_raw.get("y1_px", h))
    except (TypeError, ValueError):
        return image
    x0 = max(0, min(w, x0))
    x1 = max(0, min(w, x1))
    y0 = max(0, min(h, y0))
    y1 = max(0, min(h, y1))
    if x1 <= x0 or y1 <= y0:
        return image
    return image[y0:y1, x0:x1].copy()


def _green_ring_workspace_transform(outer_quad: np.ndarray) -> np.ndarray:
    src = _order_quad(outer_quad).astype(np.float32)
    dst = np.array(
        [
            [100.0, 0.0],   # top-left of image -> far-left / top of workspace
            [0.0, 0.0],     # top-right of image -> motor origin
            [0.0, 100.0],   # bottom-right
            [100.0, 100.0], # bottom-left
        ],
        dtype=np.float32,
    )
    return cv2.getPerspectiveTransform(src, dst)


def _resolve_analysis_image_path(path_text: str, analysis_dir: Path | None) -> Path:
    raw_path = Path(path_text)
    if raw_path.exists():
        return raw_path
    if analysis_dir is not None:
        alt = analysis_dir / raw_path.name
        if alt.exists():
            return alt
    if not raw_path.is_absolute():
        repo_path = ROOT / raw_path
        if repo_path.exists():
            return repo_path
    return raw_path


def _expected_manual_green_capture_count(
    *,
    game: str,
    resolved: Any,
) -> int:
    if not bool(getattr(resolved, "capture", False)):
        return 0
    steps = getattr(resolved, "steps", None)
    if not isinstance(steps, list) or len(steps) != 1:
        return 0
    if game not in {"chess", "checkers"}:
        return 0
    if game == "chess" and getattr(resolved, "special", None) == "promotion":
        return 2
    return 1


def _detect_manual_green_captures(
    *,
    analysis_payload: dict[str, Any],
    diff_threshold: int,
    expected_count: int,
    analysis_dir: Path | None,
) -> list[dict[str, Any]]:
    detection_payload: dict[str, Any] = {
        "expected_count": expected_count,
        "detected": [],
    }
    if expected_count <= 0:
        if analysis_dir is not None:
            (analysis_dir / "manual_green_capture_detection.json").write_text(
                json.dumps(detection_payload, ensure_ascii=True, indent=2),
                encoding="utf-8",
            )
        return []

    before_path_raw = analysis_payload.get("before_image")
    after_path_raw = analysis_payload.get("after_image")
    outer_raw = analysis_payload.get("outer_sheet_corners_px")
    inner_raw = analysis_payload.get("chessboard_corners_px")
    if not isinstance(before_path_raw, str) or not isinstance(after_path_raw, str):
        detection_payload["error"] = "missing_before_after_paths"
    elif outer_raw is None or inner_raw is None:
        detection_payload["error"] = "missing_outer_or_inner_geometry"
    else:
        before_img = cv2.imread(str(_resolve_analysis_image_path(before_path_raw, analysis_dir)))
        after_img = cv2.imread(str(_resolve_analysis_image_path(after_path_raw, analysis_dir)))
        if before_img is None or after_img is None:
            detection_payload["error"] = "failed_to_read_before_after_images"
        else:
            crop_info = analysis_payload.get("pre_detection_crop")
            crop_dict = crop_info if isinstance(crop_info, dict) else None
            before_img = _crop_for_analysis(before_img, crop_dict, "before")
            after_img = _crop_for_analysis(after_img, crop_dict, "after")
            if before_img.shape[:2] != after_img.shape[:2]:
                common_h = min(before_img.shape[0], after_img.shape[0])
                common_w = min(before_img.shape[1], after_img.shape[1])
                before_img = before_img[:common_h, :common_w].copy()
                after_img = after_img[:common_h, :common_w].copy()
            try:
                outer_quad = _order_quad(outer_raw)
                inner_quad = _order_quad(inner_raw)
            except Exception as exc:  # pragma: no cover - malformed payload only
                detection_payload["error"] = f"invalid_geometry:{exc}"
            else:
                mask = np.zeros(before_img.shape[:2], dtype=np.uint8)
                cv2.fillConvexPoly(mask, np.round(outer_quad).astype(np.int32), 255)
                cv2.fillConvexPoly(mask, np.round(inner_quad).astype(np.int32), 0)

                before_gray = cv2.cvtColor(before_img, cv2.COLOR_BGR2GRAY)
                after_gray = cv2.cvtColor(after_img, cv2.COLOR_BGR2GRAY)
                diff_img = cv2.absdiff(before_gray, after_gray)
                masked_diff = cv2.bitwise_and(diff_img, diff_img, mask=mask)
                _, thresh = cv2.threshold(
                    masked_diff,
                    max(10, int(diff_threshold)),
                    255,
                    cv2.THRESH_BINARY,
                )
                kernel = np.ones((5, 5), dtype=np.uint8)
                thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
                thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

                contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                inner_area = abs(float(cv2.contourArea(inner_quad.reshape(-1, 1, 2))))
                min_contour_area = max(80.0, inner_area / 64.0 * 0.05)
                detection_payload["min_contour_area_px"] = min_contour_area
                transform = _green_ring_workspace_transform(outer_quad)
                candidates: list[dict[str, Any]] = []
                overlay = after_img.copy()
                cv2.polylines(overlay, [np.round(outer_quad).astype(np.int32)], True, (0, 180, 255), 2)
                cv2.polylines(overlay, [np.round(inner_quad).astype(np.int32)], True, (0, 255, 255), 2)

                for contour in contours:
                    area = float(cv2.contourArea(contour))
                    if area < min_contour_area:
                        continue
                    moments = cv2.moments(contour)
                    if abs(moments.get("m00", 0.0)) < 1e-6:
                        continue
                    cx = float(moments["m10"] / moments["m00"])
                    cy = float(moments["m01"] / moments["m00"])
                    ix = int(round(cx))
                    iy = int(round(cy))
                    if iy < 0 or iy >= mask.shape[0] or ix < 0 or ix >= mask.shape[1] or mask[iy, ix] == 0:
                        continue
                    pct_point = cv2.perspectiveTransform(
                        np.array([[[cx, cy]]], dtype=np.float32),
                        transform,
                    ).reshape(2)
                    x, y, w, h = cv2.boundingRect(contour)
                    candidate = {
                        "x_pct": round(float(np.clip(pct_point[0], 0.0, 100.0)), 2),
                        "y_pct": round(float(np.clip(pct_point[1], 0.0, 100.0)), 2),
                        "area_px": round(area, 2),
                        "centroid_px": [round(cx, 2), round(cy, 2)],
                        "bbox_px": [int(x), int(y), int(w), int(h)],
                    }
                    candidates.append(candidate)
                    cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 0, 255), 2)
                    cv2.circle(overlay, (ix, iy), 5, (0, 255, 0), -1)
                    cv2.putText(
                        overlay,
                        f"{candidate['x_pct']:.2f}%,{candidate['y_pct']:.2f}%",
                        (x, max(18, y - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 0, 255),
                        1,
                        cv2.LINE_AA,
                    )

                candidates.sort(key=lambda item: float(item["area_px"]), reverse=True)
                detection_payload["detected"] = candidates[:expected_count]

                if analysis_dir is not None:
                    cv2.imwrite(str(analysis_dir / "manual_green_capture_mask.png"), thresh)
                    cv2.imwrite(str(analysis_dir / "manual_green_capture_overlay.png"), overlay)

    if analysis_dir is not None:
        (analysis_dir / "manual_green_capture_detection.json").write_text(
            json.dumps(detection_payload, ensure_ascii=True, indent=2),
            encoding="utf-8",
        )
    return [
        {
            "x_pct": float(item["x_pct"]),
            "y_pct": float(item["y_pct"]),
            "area_px": float(item["area_px"]),
        }
        for item in detection_payload.get("detected", [])
        if isinstance(item, dict)
    ]


def _run_analysis(
    before_path: Path,
    after_path: Path,
    out_dir: str | None,
    game: str,
    analysis_config: Any,
    *,
    fast_locked_geometry: bool = False,
    geometry_reference_override: str | None = None,
    disable_geometry_reference_override: bool | None = None,
    artifact_mode_override: str | None = None,
) -> dict[str, Any]:
    geometry_reference = geometry_reference_override or analysis_config.geometry_reference
    disable_geometry_reference = (
        analysis_config.disable_geometry_reference
        if disable_geometry_reference_override is None
        else bool(disable_geometry_reference_override)
    )
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
        _repo_relative_path(geometry_reference),
        "--camera-square-orientation",
        analysis_config.camera_square_orientation,
    ]
    if fast_locked_geometry:
        cmd.append("--fast-locked-geometry")
    if analysis_config.disable_tape_projection:
        cmd.append("--disable-tape-projection")
    if disable_geometry_reference:
        cmd.append("--disable-geometry-reference")
    if artifact_mode_override:
        cmd.extend(["--artifact-mode", str(artifact_mode_override)])
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


def _read_terminal_prompt_with_runtime_control(prompt: str, runtime_state: dict[str, Any] | None = None) -> str:
    print(prompt, end="", flush=True)
    while True:
        if runtime_state is not None and runtime_state.get("reset_requested"):
            runtime_state["reset_requested"] = False
            print("\n[Bridge] Runtime reset acknowledged. Start the next game and press Enter when ready.")
            return "__runtime_reset__"
        ready, _, _ = select.select([sys.stdin], [], [], 0.2)
        if ready:
            raw = sys.stdin.readline()
            if raw == "":
                return ""
            return raw.strip().lower()


def _prompt_rolling_turn_action(
    args: argparse.Namespace,
    turn_index: int,
    *,
    show_recapture_hint: bool,
    runtime_state: dict[str, Any] | None = None,
) -> str:
    if args.wait_mode == "gpio":
        _wait_for_turn_trigger(
            args,
            f"[Bridge] Turn {turn_index}: make Player 1 move, then press Enter to capture current board...",
        )
        return "capture"

    prompt = f"[Bridge] Turn {turn_index}: make Player 1 move, then press Enter to capture current board..."
    if show_recapture_hint:
        prompt += (
            "\n[Bridge] Type 'r' then Enter to recapture the rolling reference "
            "from the current board instead: "
        )
    else:
        prompt += " "
    raw = _read_terminal_prompt_with_runtime_control(prompt, runtime_state=runtime_state)
    if raw == "__runtime_reset__":
        return "runtime_reset"
    if raw in {"r", "ref", "reference", "recapture"}:
        return "recapture_reference"
    return "capture"


def _select_resolver_changed_squares(
    *,
    game: str,
    analysis_payload: dict[str, Any],
    changed_squares: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if str(game).lower() != "chess":
        return changed_squares
    raw = analysis_payload.get("resolver_changed_squares")
    if isinstance(raw, list):
        filtered = [item for item in raw if isinstance(item, dict)]
        if filtered:
            return filtered
    return changed_squares


def _analyze_paths(
    before_path: Path,
    after_path: Path,
    args: argparse.Namespace,
    game: str,
    analysis_config: Any,
    *,
    fast_locked_geometry: bool = False,
    out_dir: Path | None = None,
) -> dict[str, Any]:
    started_at = time.perf_counter()
    analyze_started_at = time.perf_counter()
    analysis = _run_analysis(
        before_path=before_path,
        after_path=after_path,
        out_dir=str(out_dir) if out_dir is not None else args.analysis_out_dir,
        game=game,
        analysis_config=analysis_config,
        fast_locked_geometry=fast_locked_geometry,
    )
    analyze_elapsed_sec = time.perf_counter() - analyze_started_at
    print(f"[Bridge] Analyzer subprocess completed in {analyze_elapsed_sec:.3f}s")
    analysis_root_raw = analysis.get("analysis_root_dir")
    if isinstance(analysis_root_raw, str) and analysis_root_raw:
        analysis_root = Path(analysis_root_raw)
    elif out_dir is not None:
        analysis_root = out_dir
    else:
        analysis_root = before_path.parent
    inferred = analysis.get("inferred_move", {})
    if isinstance(inferred, dict):
        (analysis_root / "player1_observed_move.json").write_text(
            json.dumps(inferred, ensure_ascii=True, indent=2),
            encoding="utf-8",
        )
    elapsed_sec = time.perf_counter() - started_at
    print(f"[Bridge] Turn analysis completed in {elapsed_sec:.3f}s")
    return {
        "status": "ok",
        "before_image": str(before_path),
        "after_image": str(after_path),
        "analysis_dir": str(analysis_root),
        "analysis": analysis,
        "player1_observed_move": inferred,
    }


def _quad_entry_from_cropped(
    corners: object,
    *,
    crop_meta: dict[str, Any],
    image_w: int,
    image_h: int,
) -> dict[str, Any] | None:
    if not isinstance(corners, list) or len(corners) != 4:
        return None

    try:
        crop_x0 = float(crop_meta.get("x0_px", 0.0))
        crop_y0 = float(crop_meta.get("y0_px", 0.0))
        pts = [
            [float(p[0]) + crop_x0, float(p[1]) + crop_y0]
            for p in corners
            if isinstance(p, list) and len(p) == 2
        ]
    except (TypeError, ValueError):
        return None

    if len(pts) != 4:
        return None

    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    bbox = [min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)]
    image_w_f = max(float(image_w), 1.0)
    image_h_f = max(float(image_h), 1.0)
    return {
        "corners_px": pts,
        "corners_norm": [[p[0] / image_w_f, p[1] / image_h_f] for p in pts],
        "bbox_xywh_px": bbox,
        "bbox_xywh_norm": [
            bbox[0] / image_w_f,
            bbox[1] / image_h_f,
            bbox[2] / image_w_f,
            bbox[3] / image_h_f,
        ],
    }


def _png_size(path: Path) -> tuple[int, int]:
    with path.open("rb") as handle:
        header = handle.read(24)
    if len(header) < 24 or header[:8] != b"\x89PNG\r\n\x1a\n":
        raise RuntimeError(f"Expected PNG image for manual geometry, got: {path}")
    width, height = struct.unpack(">II", header[16:24])
    return int(width), int(height)


def _quad_entry_from_raw(
    corners: object,
    *,
    image_w: int,
    image_h: int,
) -> dict[str, Any] | None:
    return _quad_entry_from_cropped(
        corners,
        crop_meta={"x0_px": 0.0, "y0_px": 0.0},
        image_w=image_w,
        image_h=image_h,
    )


def _draw_manual_reference_overlay(
    *,
    reference_path: Path,
    outer_corners_px: list[list[float]],
    chessboard_corners_px: list[list[float]],
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    overlay_path = out_dir / "manual_startup_grid_overlay.png"
    try:
        _write_startup_preview_overlay(
            reference_path=reference_path,
            outer_corners_px=outer_corners_px,
            chessboard_corners_px=chessboard_corners_px,
            out_path=overlay_path,
        )
        print(f"[Bridge] Manual startup grid overlay saved: {overlay_path}")
    except Exception as exc:  # noqa: BLE001
        print(f"[Bridge] Warning: could not create manual startup grid overlay: {exc}")


def _write_startup_preview_overlay(
    *,
    reference_path: Path,
    outer_corners_px: list[list[float]] | None,
    chessboard_corners_px: list[list[float]],
    out_path: Path,
) -> Path:
    image = cv2.imread(str(reference_path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Could not read startup reference image for preview: {reference_path}")

    detection = BoardDetection(
        outer_sheet_corners=np.array(outer_corners_px, dtype=np.float32) if outer_corners_px is not None else None,
        chessboard_corners=np.array(chessboard_corners_px, dtype=np.float32),
    )
    overlay = draw_detection_overlay(image, detection)
    squares = generate_square_geometry(
        board_corners=np.array(chessboard_corners_px, dtype=np.float32),
        board_size=(8, 8),
    )
    overlay = draw_square_grid_overlay(overlay, squares=squares, label_mode="none")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), overlay)
    return out_path


def _write_session_geometry_reference(analysis: dict[str, Any], out_path: Path) -> Path:
    crop = analysis.get("pre_detection_crop")
    if not isinstance(crop, dict):
        raise RuntimeError("Initial analysis did not include pre_detection_crop metadata.")

    try:
        image_w = int(crop["source_image_width_px"])
        image_h = int(crop["source_image_height_px"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("Initial analysis had invalid source image size metadata.") from exc

    before_crop = crop.get("before")
    if not isinstance(before_crop, dict):
        raise RuntimeError("Initial analysis did not include before crop metadata.")

    chess = _quad_entry_from_cropped(
        analysis.get("chessboard_corners_px"),
        crop_meta=before_crop,
        image_w=image_w,
        image_h=image_h,
    )
    if chess is None:
        raise RuntimeError("Initial analysis did not produce chessboard corners for session geometry.")

    outer = _quad_entry_from_cropped(
        analysis.get("outer_sheet_corners_px"),
        crop_meta=before_crop,
        image_w=image_w,
        image_h=image_h,
    )

    payload = {
        "version": 1,
        "generated_by": "run_pi_software_bridge.py",
        "source": "initial_rolling_reference",
        "source_image": analysis.get("before_image"),
        "image_size_px": {
            "width": image_w,
            "height": image_h,
        },
        "outer_sheet_corners_px": outer["corners_px"] if outer is not None else None,
        "chessboard_corners_px": chess["corners_px"],
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
    return out_path


def _store_auto_session_geometry(
    *,
    session_geometry_path: Path,
    preview_path: Path | None,
    game_dir: Path,
) -> Path:
    payload = json.loads(session_geometry_path.read_text(encoding="utf-8"))
    payload["source"] = "auto_startup_geometry"
    payload["generated_by"] = "run_pi_software_bridge.py"

    game_corners_path = game_dir / "corners_info.json"
    game_corners_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
    print(f"[Bridge] Auto startup geometry saved for game: {game_corners_path}")

    sample_path = _next_manual_archive_path(MANUAL_CORNERS_SAMPLES_DIR)
    sample_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
    print(f"[Bridge] Accepted auto geometry archive saved: {sample_path}")

    MANUAL_CORNERS_INFO_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANUAL_CORNERS_INFO_PATH.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2),
        encoding="utf-8",
    )
    print(f"[Bridge] Auto startup geometry baseline saved: {MANUAL_CORNERS_INFO_PATH}")

    if preview_path is not None and preview_path.exists():
        preview_copy_path = game_dir / "auto_startup_grid_overlay.png"
        if preview_path.resolve() != preview_copy_path.resolve():
            shutil.copy2(preview_path, preview_copy_path)
        print(f"[Bridge] Auto startup preview archived: {preview_copy_path}")

    return game_corners_path


def _next_manual_archive_path(samples_dir: Path) -> Path:
    samples_dir.mkdir(parents=True, exist_ok=True)
    pattern = re.compile(r"^game(\d+)_corners\.json$")
    max_index = 0
    for child in samples_dir.iterdir():
        if not child.is_file():
            continue
        match = pattern.fullmatch(child.name)
        if match:
            max_index = max(max_index, int(match.group(1)))
    return samples_dir / f"game{max_index + 1}_corners.json"


def _build_manual_corners_info_payload(
    *,
    reference_path: Path,
    outer_corners_px: object,
    chessboard_corners_px: object,
) -> dict[str, Any]:
    image_w, image_h = _png_size(reference_path)
    outer = _quad_entry_from_raw(outer_corners_px, image_w=image_w, image_h=image_h)
    chess = _quad_entry_from_raw(chessboard_corners_px, image_w=image_w, image_h=image_h)
    if outer is None:
        raise RuntimeError("Manual geometry did not include 4 valid outer corners.")
    if chess is None:
        raise RuntimeError("Manual geometry did not include 4 valid inner chessboard corners.")

    return {
        "version": 1,
        "generated_by": "run_pi_software_bridge.py",
        "source": "manual_startup_geometry",
        "source_image": str(reference_path),
        "image_size_px": {
            "width": image_w,
            "height": image_h,
        },
        "outer_sheet_corners_px": outer["corners_px"],
        "chessboard_corners_px": chess["corners_px"],
    }


def _store_manual_corners_info(
    *,
    reference_path: Path,
    outer_corners_px: object,
    chessboard_corners_px: object,
    game_dir: Path,
) -> Path:
    payload = _build_manual_corners_info_payload(
        reference_path=reference_path,
        outer_corners_px=outer_corners_px,
        chessboard_corners_px=chessboard_corners_px,
    )

    game_corners_path = game_dir / "corners_info.json"
    game_corners_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
    print(f"[Bridge] Manual corners saved for game: {game_corners_path}")

    sample_path = _next_manual_archive_path(MANUAL_CORNERS_SAMPLES_DIR)
    sample_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
    print(f"[Bridge] Manual corners archive saved: {sample_path}")

    MANUAL_CORNERS_INFO_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANUAL_CORNERS_INFO_PATH.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2),
        encoding="utf-8",
    )
    print(f"[Bridge] Manual corners baseline saved: {MANUAL_CORNERS_INFO_PATH}")

    _draw_manual_reference_overlay(
        reference_path=reference_path,
        outer_corners_px=payload["outer_sheet_corners_px"],
        chessboard_corners_px=payload["chessboard_corners_px"],
        out_dir=game_dir,
    )
    return game_corners_path


def _build_session_geometry_reference(
    *,
    reference_path: Path,
    args: argparse.Namespace,
    game: str,
    analysis_config: Any,
    game_debug_dir: Path,
) -> dict[str, Path] | None:
    print("[Bridge] Building live session geometry from initial reference...")
    try:
        started_at = time.perf_counter()
        image_w, image_h = _png_size(reference_path)
        preview_path = game_debug_dir / "auto_startup_grid_overlay.png"
        if MANUAL_CORNERS_INFO_PATH.exists():
            try:
                preview_started_at = time.perf_counter()
                payload = _load_scaled_corners_payload(
                    corners_path=MANUAL_CORNERS_INFO_PATH,
                    target_image_w=image_w,
                    target_image_h=image_h,
                )
                payload["source"] = "config_corners_info_startup_geometry"
                payload["generated_by"] = "run_pi_software_bridge.py"
                payload["source_image"] = str(reference_path)
                session_path = Path(args.session_geometry_path)
                session_path.parent.mkdir(parents=True, exist_ok=True)
                session_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
                _write_startup_preview_overlay(
                    reference_path=reference_path,
                    outer_corners_px=payload.get("outer_sheet_corners_px"),
                    chessboard_corners_px=payload["chessboard_corners_px"],
                    out_path=preview_path,
                )
                print(
                    "[Bridge] Startup geometry baseline load+preview completed in "
                    f"{time.perf_counter() - preview_started_at:.3f}s"
                )
            except Exception as exc:  # noqa: BLE001
                print(
                    "[Bridge] Warning: configs/corners_info.json was not usable for startup preview; "
                    f"falling back to live auto-detect. Details: {exc}"
                )
                analyze_started_at = time.perf_counter()
                analysis = _run_analysis(
                    before_path=reference_path,
                    after_path=reference_path,
                    out_dir=str(game_debug_dir),
                    game=game,
                    analysis_config=analysis_config,
                    fast_locked_geometry=False,
                    disable_geometry_reference_override=True,
                    artifact_mode_override="minimal",
                )
                print(
                    "[Bridge] Startup live auto-detect analysis completed in "
                    f"{time.perf_counter() - analyze_started_at:.3f}s"
                )
                session_path = _write_session_geometry_reference(analysis, Path(args.session_geometry_path))
                preview_started_at = time.perf_counter()
                payload = json.loads(session_path.read_text(encoding="utf-8"))
                _write_startup_preview_overlay(
                    reference_path=reference_path,
                    outer_corners_px=payload.get("outer_sheet_corners_px"),
                    chessboard_corners_px=payload["chessboard_corners_px"],
                    out_path=preview_path,
                )
                print(
                    "[Bridge] Startup live preview render completed in "
                    f"{time.perf_counter() - preview_started_at:.3f}s"
                )
        else:
            analyze_started_at = time.perf_counter()
            analysis = _run_analysis(
                before_path=reference_path,
                after_path=reference_path,
                out_dir=str(game_debug_dir),
                game=game,
                analysis_config=analysis_config,
                fast_locked_geometry=False,
                disable_geometry_reference_override=True,
                artifact_mode_override="minimal",
            )
            print(
                "[Bridge] Startup live auto-detect analysis completed in "
                f"{time.perf_counter() - analyze_started_at:.3f}s"
            )
            session_path = _write_session_geometry_reference(analysis, Path(args.session_geometry_path))
            preview_started_at = time.perf_counter()
            payload = json.loads(session_path.read_text(encoding="utf-8"))
            _write_startup_preview_overlay(
                reference_path=reference_path,
                outer_corners_px=payload.get("outer_sheet_corners_px"),
                chessboard_corners_px=payload["chessboard_corners_px"],
                out_path=preview_path,
            )
            print(
                "[Bridge] Startup live preview render completed in "
                f"{time.perf_counter() - preview_started_at:.3f}s"
            )
        elapsed_sec = round(time.perf_counter() - started_at, 3)
    except Exception as exc:  # noqa: BLE001
        print(
            "[Bridge] Warning: failed to build session geometry; "
            f"falling back to configured geometry reference. Details: {exc}"
        )
        return None

    print(f"[Bridge] Live session geometry saved: {session_path}")
    print(f"[Bridge] Startup auto-geometry completed in {elapsed_sec:.3f}s")
    if preview_path is not None:
        print(f"[Bridge] Startup grid preview: {preview_path}")
    return {"geometry": session_path, "preview": preview_path} if preview_path is not None else {"geometry": session_path}


def _capture_and_analyze_two_shot(
    controller: EndTurnController,
    args: argparse.Namespace,
    game: str,
    analysis_config: Any,
    turn_debug_dir: Path | None,
) -> dict[str, Any]:
    before_capture_started_at = time.perf_counter()
    before_path = controller.capture_before(
        reopen_stream=args.reopen_camera_each_capture,
        output_path=turn_debug_dir / "before_capture.png" if turn_debug_dir is not None else None,
    )
    print(f"[Bridge] BEFORE capture completed in {time.perf_counter() - before_capture_started_at:.3f}s")
    print(f"[Bridge] BEFORE captured: {before_path}")

    _wait_for_turn_trigger(args, "[Bridge] Move Player 1 piece, then press Enter to capture AFTER...")

    after_capture_started_at = time.perf_counter()
    after_path = controller.capture_after(
        reopen_stream=args.reopen_camera_each_capture,
        output_path=turn_debug_dir / "after_capture.png" if turn_debug_dir is not None else None,
    )
    print(f"[Bridge] AFTER capture completed in {time.perf_counter() - after_capture_started_at:.3f}s")
    print(f"[Bridge] AFTER captured: {after_path}")

    analysis_out_dir = turn_debug_dir if turn_debug_dir is not None else (
        Path(args.analysis_out_dir) if args.analysis_out_dir else None
    )

    return _analyze_paths(
        before_path,
        after_path,
        args,
        game,
        analysis_config,
        out_dir=analysis_out_dir,
    )


def _capture_and_analyze_rolling(
    controller: EndTurnController,
    args: argparse.Namespace,
    game: str,
    reference_path: Path,
    turn_index: int,
    analysis_config: Any,
    turn_debug_dir: Path | None,
    show_recapture_hint: bool = False,
    runtime_state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    game_debug_dir = turn_debug_dir.parent if turn_debug_dir is not None else None
    action = _prompt_rolling_turn_action(
        args,
        turn_index,
        show_recapture_hint=show_recapture_hint,
        runtime_state=runtime_state,
    )
    latest_reference_getter = getattr(args, "_runtime_reference_getter", None)
    if callable(latest_reference_getter):
        refreshed = latest_reference_getter()
        if isinstance(refreshed, Path):
            reference_path = refreshed
    if action == "recapture_reference":
        capture_started_at = time.perf_counter()
        new_reference_path = _capture_live_reference(
            controller,
            reopen_stream=args.reopen_camera_each_capture,
            output_path=(game_debug_dir / "rolling_reference.png") if game_debug_dir is not None else None,
        )
        print(f"[Bridge] Reference recapture completed in {time.perf_counter() - capture_started_at:.3f}s")
        print(f"[Bridge] Rolling reference recaptured from current board: {new_reference_path}")
        if turn_debug_dir is not None:
            _copy_frame_for_turn(new_reference_path, turn_debug_dir / "reference_recaptured.png")
        return {
            "action": "recapture_reference",
            "reference_path": str(new_reference_path),
        }
    if action == "runtime_reset":
        return {"action": "runtime_reset"}

    capture_started_at = time.perf_counter()
    after_path = _capture_live_reference(
        controller,
        reopen_stream=args.reopen_camera_each_capture,
        before=False,
        output_path=turn_debug_dir / "after_capture.png" if turn_debug_dir is not None else None,
    )
    print(f"[Bridge] CURRENT capture completed in {time.perf_counter() - capture_started_at:.3f}s")
    print(f"[Bridge] CURRENT captured: {after_path}")
    print(f"[Bridge] Diffing previous reference -> current: {reference_path} -> {after_path}")
    if turn_debug_dir is not None:
        _copy_frame_for_turn(reference_path, turn_debug_dir / "before_capture.png")

    analysis_out_dir = turn_debug_dir if turn_debug_dir is not None else (
        Path(args.analysis_out_dir) if args.analysis_out_dir else None
    )

    return _analyze_paths(
        reference_path,
        after_path,
        args,
        game,
        analysis_config,
        fast_locked_geometry=not args.slow_live_analysis,
        out_dir=analysis_out_dir,
    )


def _send_moves_to_stm(*, skip_return_start: bool = False, return_start_only: bool = False) -> dict[str, Any]:
    cmd = [sys.executable, str(ROOT / "scripts" / "send_moves_from_file.py")]
    if skip_return_start:
        cmd.append("--skip-return-start")
    if return_start_only:
        cmd.append("--return-start-only")
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


def _write_json_line(conn_file: Any, obj: dict[str, Any], *, write_lock: threading.Lock | None = None) -> None:
    if write_lock is not None:
        with write_lock:
            conn_file.write((json.dumps(obj, separators=(",", ":")) + "\n").encode("utf-8"))
            conn_file.flush()
        return
    conn_file.write((json.dumps(obj, separators=(",", ":")) + "\n").encode("utf-8"))
    conn_file.flush()


def _request_manual_startup_geometry(
    conn_file: Any,
    *,
    reference_path: Path,
    game_debug_dir: Path,
) -> Path | None:
    image_b64 = base64.b64encode(reference_path.read_bytes()).decode("ascii")
    _write_json_line(
        conn_file,
        {
            "type": "geometry_calibration_request",
            "title": "Manual Board Geometry",
            "summary": (
                "Click 4 green outer-grid corners first, then 4 yellow inner-grid corners. "
                "Use order: top-left, top-right, bottom-right, bottom-left."
            ),
            "source_path": str(reference_path),
            "image_png_b64": image_b64,
        },
    )
    print("[Bridge] Sent initial board image to Software-GUI for manual grid selection.")

    while True:
        msg = _read_json_line(conn_file)
        if msg.get("type") != "geometry_calibration_result":
            print(f"[Bridge] Ignoring message while waiting for manual geometry: {msg.get('type')}")
            continue
        if not bool(msg.get("accepted")):
            print("[Bridge] Manual startup geometry was cancelled.")
            return None
        path = _store_manual_corners_info(
            reference_path=reference_path,
            outer_corners_px=msg.get("outer_corners_px"),
            chessboard_corners_px=msg.get("chessboard_corners_px"),
            game_dir=game_debug_dir,
        )
        print(f"[Bridge] Manual startup geometry saved: {path}")
        return path


def _confirm_startup_geometry(conn_file: Any, preview_path: Path | None, geometry_path: Path) -> bool:
    if preview_path is None or not preview_path.exists():
        print("[Bridge] No startup grid preview image available; using terminal confirmation.")
        reply = input(
            "[Bridge] Continue with this detected geometry? "
            "[y=yes, m=switch to manual, N=manual/cancel] "
        ).strip().lower()
        return reply in {"y", "yes"}

    image_b64 = base64.b64encode(preview_path.read_bytes()).decode("ascii")
    _write_json_line(
        conn_file,
        {
            "type": "geometry_preview",
            "title": "Confirm Detected Board Grid",
            "summary": (
                "Review the startup green/blue grid. Click Confirm only if the board outline "
                "and 8x8 chess/checkers grid line up with the physical board. "
                "If not, reject it and the bridge will switch to manual corner selection."
            ),
            "source_path": str(preview_path),
            "geometry_path": str(geometry_path),
            "image_png_b64": image_b64,
        },
    )
    print("[Bridge] Sent startup grid preview to Software-GUI; waiting for Confirm...")

    while True:
        msg = _read_json_line(conn_file)
        if msg.get("type") != "geometry_confirm":
            print(f"[Bridge] Ignoring message while waiting for geometry confirmation: {msg.get('type')}")
            continue
        accepted = bool(msg.get("accepted"))
        if accepted:
            print("[Bridge] Startup grid confirmed.")
        else:
            print("[Bridge] Startup grid rejected.")
        return accepted


def _resolve_startup_geometry(
    conn_file: Any,
    *,
    reference_path: Path,
    args: argparse.Namespace,
    game: str,
    analysis_config: Any,
    game_debug_dir: Path,
) -> Path | None:
    if args.startup_geometry_mode == "manual":
        return _request_manual_startup_geometry(
            conn_file,
            reference_path=reference_path,
            game_debug_dir=game_debug_dir,
        )

    session_geometry_result = _build_session_geometry_reference(
        reference_path=reference_path,
        args=args,
        game=game,
        analysis_config=analysis_config,
        game_debug_dir=game_debug_dir,
    )
    if session_geometry_result is not None:
        session_geometry_path = session_geometry_result["geometry"]
        preview_path = session_geometry_result.get("preview")
        accepted = True
        if not args.skip_geometry_confirmation:
            accepted = _confirm_startup_geometry(
                conn_file,
                preview_path,
                session_geometry_path,
            )
        if accepted:
            return _store_auto_session_geometry(
                session_geometry_path=session_geometry_path,
                preview_path=preview_path,
                game_dir=game_debug_dir,
            )

        print("[Bridge] Switching to manual startup geometry selection...")

    else:
        print("[Bridge] Automatic startup geometry failed; switching to manual selection...")

    return _request_manual_startup_geometry(
        conn_file,
        reference_path=reference_path,
        game_debug_dir=game_debug_dir,
    )


def _initialize_rolling_reference_for_game(
    conn_file: Any,
    *,
    controller: EndTurnController,
    args: argparse.Namespace,
    game: str,
    analysis_config: Any,
    game_debug_dir: Path,
    output_name: str,
    prompt: str,
) -> Path | None:
    input(prompt)
    capture_started_at = time.perf_counter()
    reference_path = _capture_startup_reference(
        controller,
        output_path=game_debug_dir / output_name,
    )
    print(f"[Bridge] Initial reference capture completed in {time.perf_counter() - capture_started_at:.3f}s")
    print(f"[Bridge] Initial reference captured: {reference_path}")
    if not args.slow_live_analysis:
        session_geometry = _resolve_startup_geometry(
            conn_file,
            reference_path=reference_path,
            args=args,
            game=game,
            analysis_config=analysis_config,
            game_debug_dir=game_debug_dir,
        )
        if session_geometry is None:
            print("[Bridge] Stopping before turn capture because startup geometry was cancelled.")
            return None
        analysis_config.geometry_reference = str(session_geometry)
        analysis_config.disable_geometry_reference = False
        print(f"[Bridge] Locked session geometry for this game: {session_geometry}")
    return reference_path


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


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _significant_changed_count(
    analysis_payload: dict[str, Any],
    observed_move: dict[str, Any] | None,
    changed_squares: list[dict[str, Any]],
) -> int:
    if isinstance(observed_move, dict):
        metadata = observed_move.get("metadata")
        if isinstance(metadata, dict):
            raw_count = metadata.get("significant_changed_count")
            if isinstance(raw_count, int):
                return raw_count
            if isinstance(raw_count, float):
                return int(raw_count)

    raw_count = analysis_payload.get("significant_changed_count")
    if isinstance(raw_count, int):
        return raw_count
    if isinstance(raw_count, float):
        return int(raw_count)

    count = 0
    for item in changed_squares:
        pixel_ratio = _as_float(item.get("pixel_ratio"))
        intensity_delta = abs(_as_float(item.get("signed_intensity_delta")))
        if pixel_ratio >= 0.12 or intensity_delta >= 5.0:
            count += 1
    return count


def _player1_rejection_status_payload(
    *,
    reason: str,
    details: dict[str, Any],
    attempt_index: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    if reason == "missing_player1_observed_move":
        message = "CV did not produce an observed Player 1 move."
    elif reason == "insufficient_significant_changed_squares":
        actual = details.get("significant_changed_count")
        needed = details.get("min_significant_changes")
        message = (
            f"Only {actual} significant changed square(s) were detected; need at least {needed}."
        )
    elif reason == "legal_move_resolution_rejected":
        score = details.get("max_resolved_score")
        message = (
            "Detected board changes did not match a confident legal move "
            f"(resolver threshold {score})."
        )
    elif reason == "excessive_significant_changed_squares":
        actual = details.get("significant_changed_count")
        limit = details.get("max_significant_changes")
        message = (
            f"{actual} significant changed squares were detected, exceeding the limit of {limit}. "
            "This usually means a hand or large obstruction was in frame."
        )
    else:
        message = f"Player 1 move was rejected: {reason}."

    extra = [
        "The previous clean reference image is still active; the rejected after-frame was not promoted to the new reference.",
        "Remove any hand/arm or obstruction, leave the board in the intended state, and press Enter to retry the same turn.",
    ]
    if args.max_p1_detection_attempts > 0:
        remaining = max(0, args.max_p1_detection_attempts - attempt_index)
        extra.append(f"Retries remaining before stop: {remaining}.")
    return {
        "type": "status",
        "level": "warning",
        "title": "Player 1 Move Rejected",
        "message": message,
        "details": extra,
        "sticky": True,
    }


def _send_status_message(
    conn_file: Any,
    payload: dict[str, Any],
    *,
    write_lock: threading.Lock | None = None,
) -> None:
    try:
        _write_json_line(conn_file, payload, write_lock=write_lock)
    except Exception:
        # UI-side status messaging is best-effort only.
        pass


def _connection_reader_loop(
    conn_file: Any,
    *,
    p2_queue: Queue[dict[str, Any]],
    control_queue: Queue[dict[str, Any]],
) -> None:
    try:
        while True:
            msg = _read_json_line(conn_file)
            if msg.get("type") == "p2_move":
                p2_queue.put(msg)
            else:
                control_queue.put(msg)
    except Exception as exc:  # noqa: BLE001
        sentinel = {"type": "__disconnect__", "error": str(exc)}
        p2_queue.put(sentinel)
        control_queue.put(sentinel)


def _capture_runtime_reference(
    controller: EndTurnController,
    *,
    args: argparse.Namespace,
    game_debug_dir: Path,
) -> Path:
    return controller.capture_before(
        reopen_stream=args.reopen_camera_each_capture,
        output_path=game_debug_dir / "rolling_reference.png",
    )


def _runtime_control_worker(
    *,
    control_queue: Queue[dict[str, Any]],
    conn_file: Any,
    write_lock: threading.Lock,
    controller: EndTurnController,
    args: argparse.Namespace,
    game_holder: dict[str, str],
    game_debug_dir: Path,
    resolver_holder: dict[str, Any],
    reference_holder: dict[str, Path | None],
    resolver_orientation: str,
    runtime_state: dict[str, Any],
) -> None:
    while True:
        msg = control_queue.get()
        msg_type = msg.get("type")
        if msg_type == "__disconnect__":
            return
        if msg_type == "refresh_reference":
            reason = str(msg.get("reason", "manual"))
            if args.capture_mode != "rolling":
                _send_status_message(
                    conn_file,
                    {
                        "type": "status",
                        "level": "info",
                        "title": "Reference Refresh Skipped",
                        "message": "Reference refresh is only used in rolling capture mode.",
                        "duration_ms": 2000,
                    },
                    write_lock=write_lock,
                )
                continue
            try:
                started_at = time.perf_counter()
                reference_path = _capture_runtime_reference(
                    controller,
                    args=args,
                    game_debug_dir=game_debug_dir,
                )
                reference_holder["path"] = reference_path
                elapsed = time.perf_counter() - started_at
                print(
                    f"[Bridge] Runtime rolling reference refreshed in {elapsed:.3f}s"
                    f" (reason={reason}) -> {reference_path}"
                )
                _send_status_message(
                    conn_file,
                    {
                        "type": "status",
                        "level": "success",
                        "title": "Reference Refreshed",
                        "message": f"Captured a new rolling reference from the current physical board ({reason}).",
                        "details": [f"Capture time: {elapsed:.3f}s"],
                        "duration_ms": 2200,
                    },
                    write_lock=write_lock,
                )
            except Exception as exc:  # noqa: BLE001
                print(f"[Bridge] Runtime reference refresh failed: {exc}")
                _send_status_message(
                    conn_file,
                    {
                        "type": "status",
                        "level": "error",
                        "title": "Reference Refresh Failed",
                        "message": str(exc),
                        "sticky": True,
                    },
                    write_lock=write_lock,
                )
            continue
        if msg_type == "set_game":
            requested_game = _normalize_game_name(msg.get("game"), default=game_holder["game"])
            previous_game = game_holder["game"]
            game_holder["game"] = requested_game
            resolver_holder["resolver"] = Player1MoveResolver(
                requested_game,
                camera_square_orientation=resolver_orientation,
            )
            runtime_state["reset_requested"] = True
            runtime_state["await_new_game_capture"] = True
            runtime_state["game_over"] = False
            print(f"[Bridge] Runtime game selection updated: {previous_game} -> {requested_game}")
            _send_status_message(
                conn_file,
                {
                    "type": "status",
                    "level": "info",
                    "title": "Game Selection Updated",
                    "message": f"Physical board/game rules switched to {requested_game}.",
                    "details": [
                        "Arrange the board for the selected game, then press Enter in the bridge terminal to capture a new starting reference."
                    ],
                    "duration_ms": 2600,
                },
                write_lock=write_lock,
            )
            continue
        if msg_type == "reset_game":
            reason = str(msg.get("reason", "reset"))
            try:
                requested_game = _normalize_game_name(msg.get("game"), default=game_holder["game"])
                game_holder["game"] = requested_game
                runtime_state["reset_requested"] = True
                runtime_state["await_new_game_capture"] = True
                runtime_state["game_over"] = False
                resolver_holder["resolver"] = Player1MoveResolver(
                    requested_game,
                    camera_square_orientation=resolver_orientation,
                )
                details: list[str] = [
                    "Pi-side move resolver reset to opening state.",
                    "Press Enter in the bridge terminal after the board is arranged to capture a fresh starting reference.",
                ]
                print(
                    f"[Bridge] Runtime game reset requested (reason={reason})"
                    f" for game={requested_game}."
                )
                _send_status_message(
                    conn_file,
                    {
                        "type": "status",
                        "level": "success",
                        "title": "Game Reset On Pi",
                        "message": "Pi-side resolver state was reset. Waiting for a new starting reference capture.",
                        "details": details,
                        "duration_ms": 2600,
                    },
                    write_lock=write_lock,
                )
            except Exception as exc:  # noqa: BLE001
                print(f"[Bridge] Runtime game reset failed: {exc}")
                _send_status_message(
                    conn_file,
                    {
                        "type": "status",
                        "level": "error",
                        "title": "Game Reset Failed",
                        "message": str(exc),
                        "sticky": True,
                    },
                    write_lock=write_lock,
                )
            continue
        print(f"[Bridge] Ignoring runtime control message: {msg}")


def _write_rejected_player1_attempt(
    *,
    analysis_wrap: dict[str, Any],
    reason: str,
    details: dict[str, Any],
    resolver: Player1MoveResolver,
    turn_debug_dir: Path,
) -> Path:
    out_dir = turn_debug_dir
    analysis_dir_raw = analysis_wrap.get("analysis_dir")
    if isinstance(analysis_dir_raw, str):
        out_dir = Path(analysis_dir_raw)
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "status": "rejected",
        "reason": reason,
        "details": details,
        "before_image": analysis_wrap.get("before_image"),
        "after_image": analysis_wrap.get("after_image"),
        "player1_observed_move": analysis_wrap.get("player1_observed_move"),
        "resolver_state_unchanged": resolver.debug_state(),
    }
    path = out_dir / "player1_rejected_attempt.json"
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
    return path


def _print_rejected_player1_attempt(
    *,
    reason: str,
    details: dict[str, Any],
    rejection_path: Path,
    attempt_index: int,
    args: argparse.Namespace,
) -> None:
    print(f"[Bridge] WARNING: rejected Player 1 detection attempt {attempt_index}: {reason}")
    if details:
        print(f"[Bridge] Rejection details: {json.dumps(details, separators=(',', ':'))}")
    print(f"[Bridge] Rejection record saved: {rejection_path}")
    print("[Bridge] Keeping the same before/reference image. Correct the physical board, then retry this same turn.")
    if args.max_p1_detection_attempts > 0:
        remaining = args.max_p1_detection_attempts - attempt_index
        print(f"[Bridge] Detection retry attempts remaining before stop: {max(0, remaining)}")


def _p1_detection_attempts_exhausted(args: argparse.Namespace, attempt_index: int) -> bool:
    return args.max_p1_detection_attempts > 0 and attempt_index >= args.max_p1_detection_attempts


def _extract_turn_steps(msg: dict[str, Any]) -> list[tuple[str, str]]:
    raw_steps = msg.get("turn_steps")
    steps: list[tuple[str, str]] = []
    if isinstance(raw_steps, list):
        for item in raw_steps:
            if not isinstance(item, dict):
                continue
            frm = item.get("from")
            to = item.get("to")
            if isinstance(frm, str) and isinstance(to, str):
                steps.append((frm, to))
    if steps:
        return steps

    frm = msg.get("from")
    to = msg.get("to")
    if isinstance(frm, str) and isinstance(to, str):
        return [(frm, to)]
    return []


def _serve_client(
    args: argparse.Namespace,
    conn: socket.socket,
    controller: EndTurnController,
    game: str,
    resolver: Player1MoveResolver,
    analysis_config: Any,
    game_debug_dir: Path,
) -> None:
    conn.settimeout(None)
    with conn:
        conn_file = conn.makefile("rwb")
        write_lock = threading.Lock()
        turn_index = 1
        reference_path: Path | None = None
        show_reference_recapture_hint = False
        game_holder: dict[str, str] = {"game": _normalize_game_name(game)}

        if args.capture_mode == "rolling":
            reference_path = _initialize_rolling_reference_for_game(
                conn_file,
                controller=controller,
                args=args,
                game=game_holder["game"],
                analysis_config=analysis_config,
                game_debug_dir=game_debug_dir,
                output_name="initial_reference.png",
                prompt="[Bridge] Make sure the physical board matches the software state, then press Enter to capture INITIAL reference...",
            )
            if reference_path is None:
                return

        p2_queue: Queue[dict[str, Any]] = Queue()
        control_queue: Queue[dict[str, Any]] = Queue()
        resolver_holder: dict[str, Any] = {"resolver": resolver}
        reference_holder: dict[str, Path | None] = {"path": reference_path}
        runtime_state: dict[str, Any] = {"reset_requested": False, "await_new_game_capture": False, "game_over": False}
        reader_thread = threading.Thread(
            target=_connection_reader_loop,
            kwargs={
                "conn_file": conn_file,
                "p2_queue": p2_queue,
                "control_queue": control_queue,
            },
            daemon=True,
        )
        reader_thread.start()
        control_thread = threading.Thread(
            target=_runtime_control_worker,
            kwargs={
                "control_queue": control_queue,
                "conn_file": conn_file,
                "write_lock": write_lock,
                "controller": controller,
                "args": args,
                "game_holder": game_holder,
                "game_debug_dir": game_debug_dir,
                "resolver_holder": resolver_holder,
                "reference_holder": reference_holder,
                "resolver_orientation": resolver.camera_square_orientation,
                "runtime_state": runtime_state,
            },
            daemon=True,
        )
        control_thread.start()
        setattr(args, "_runtime_reference_getter", lambda: reference_holder["path"])

        while True:
            if runtime_state.get("game_over") or runtime_state.get("await_new_game_capture"):
                runtime_state["game_over"] = False
                runtime_state["await_new_game_capture"] = False
                resolver_holder["resolver"] = Player1MoveResolver(
                    game_holder["game"],
                    camera_square_orientation=resolver.camera_square_orientation,
                )
                if args.capture_mode == "rolling":
                    reference_path = _initialize_rolling_reference_for_game(
                        conn_file,
                        controller=controller,
                        args=args,
                        game=game_holder["game"],
                        analysis_config=analysis_config,
                        game_debug_dir=game_debug_dir,
                        output_name="rolling_reference.png",
                        prompt="[Bridge] Arrange the physical board for the next game, then press Enter to capture a new starting reference...",
                    )
                    if reference_path is None:
                        return
                    reference_holder["path"] = reference_path
                show_reference_recapture_hint = False

            attempt_index = 1
            while True:
                print("")
                turn_debug_dir = (
                    game_debug_dir / f"Move{turn_index}"
                    if attempt_index == 1
                    else game_debug_dir / f"Move{turn_index}_Retry{attempt_index}"
                )
                turn_debug_dir.mkdir(parents=True, exist_ok=True)
                reference_path = reference_holder["path"]
                if args.capture_mode == "rolling":
                    if reference_path is None:
                        raise RuntimeError("rolling capture mode missing reference image")
                    analysis_wrap = _capture_and_analyze_rolling(
                        controller=controller,
                        args=args,
                        game=game_holder["game"],
                        reference_path=reference_path,
                        turn_index=turn_index,
                        analysis_config=analysis_config,
                        turn_debug_dir=turn_debug_dir,
                        show_recapture_hint=show_reference_recapture_hint,
                        runtime_state=runtime_state,
                    )
                    if analysis_wrap.get("action") == "runtime_reset":
                        show_reference_recapture_hint = False
                        continue
                    if analysis_wrap.get("action") == "recapture_reference":
                        ref_raw = analysis_wrap.get("reference_path")
                        if isinstance(ref_raw, str):
                            reference_path = Path(ref_raw)
                        show_reference_recapture_hint = False
                        print("[Bridge] Rolling reference updated. Retry the same turn when ready.")
                        continue
                else:
                    input(f"[Bridge] Turn {turn_index}: press Enter to capture BEFORE...")
                    analysis_wrap = _capture_and_analyze_two_shot(
                        controller,
                        args,
                        game,
                        analysis_config,
                        turn_debug_dir,
                    )

                observed_move = analysis_wrap.get("player1_observed_move")
                analysis_payload_raw = analysis_wrap.get("analysis", {})
                analysis_payload = analysis_payload_raw if isinstance(analysis_payload_raw, dict) else {}
                if not isinstance(observed_move, dict):
                    details = {
                        "observed_move_type": type(observed_move).__name__,
                        "min_significant_changes": args.min_significant_changes,
                        "max_resolved_score": args.max_resolved_score,
                    }
                    rejection_path = _write_rejected_player1_attempt(
                        analysis_wrap=analysis_wrap,
                        reason="missing_player1_observed_move",
                        details=details,
                        resolver=resolver_holder["resolver"],
                        turn_debug_dir=turn_debug_dir,
                    )
                    _print_rejected_player1_attempt(
                        reason="missing_player1_observed_move",
                        details=details,
                        rejection_path=rejection_path,
                        attempt_index=attempt_index,
                        args=args,
                    )
                    _send_status_message(
                        conn_file,
                        _player1_rejection_status_payload(
                            reason="missing_player1_observed_move",
                            details=details,
                            attempt_index=attempt_index,
                            args=args,
                        ),
                        write_lock=write_lock,
                    )
                    if _p1_detection_attempts_exhausted(args, attempt_index):
                        print("[Bridge] Stopping after rejected Player 1 detection attempts.")
                        return
                    show_reference_recapture_hint = True
                    attempt_index += 1
                    continue

                changed_squares: list[dict[str, Any]] = []
                raw_changed = analysis_payload.get("changed_squares")
                if isinstance(raw_changed, list):
                    changed_squares = [item for item in raw_changed if isinstance(item, dict)]
                resolver_changed_squares = _select_resolver_changed_squares(
                    game=game_holder["game"],
                    analysis_payload=analysis_payload,
                    changed_squares=changed_squares,
                )

                significant_count = _significant_changed_count(
                    analysis_payload,
                    observed_move,
                    changed_squares,
                )
                if args.max_significant_changes > 0 and significant_count > args.max_significant_changes:
                    details = {
                        "significant_changed_count": significant_count,
                        "changed_square_count": len(changed_squares),
                        "min_significant_changes": args.min_significant_changes,
                        "max_significant_changes": args.max_significant_changes,
                        "max_resolved_score": args.max_resolved_score,
                    }
                    rejection_path = _write_rejected_player1_attempt(
                        analysis_wrap=analysis_wrap,
                        reason="excessive_significant_changed_squares",
                        details=details,
                        resolver=resolver_holder["resolver"],
                        turn_debug_dir=turn_debug_dir,
                    )
                    _print_rejected_player1_attempt(
                        reason="excessive_significant_changed_squares",
                        details=details,
                        rejection_path=rejection_path,
                        attempt_index=attempt_index,
                        args=args,
                    )
                    _send_status_message(
                        conn_file,
                        _player1_rejection_status_payload(
                            reason="excessive_significant_changed_squares",
                            details=details,
                            attempt_index=attempt_index,
                            args=args,
                        ),
                        write_lock=write_lock,
                    )
                    if _p1_detection_attempts_exhausted(args, attempt_index):
                        print("[Bridge] Stopping after rejected Player 1 detection attempts.")
                        return
                    show_reference_recapture_hint = True
                    attempt_index += 1
                    continue
                if significant_count < args.min_significant_changes:
                    details = {
                        "significant_changed_count": significant_count,
                        "changed_square_count": len(changed_squares),
                        "min_significant_changes": args.min_significant_changes,
                        "max_resolved_score": args.max_resolved_score,
                    }
                    rejection_path = _write_rejected_player1_attempt(
                        analysis_wrap=analysis_wrap,
                        reason="insufficient_significant_changed_squares",
                        details=details,
                        resolver=resolver_holder["resolver"],
                        turn_debug_dir=turn_debug_dir,
                    )
                    _print_rejected_player1_attempt(
                        reason="insufficient_significant_changed_squares",
                        details=details,
                        rejection_path=rejection_path,
                        attempt_index=attempt_index,
                        args=args,
                    )
                    _send_status_message(
                        conn_file,
                        _player1_rejection_status_payload(
                            reason="insufficient_significant_changed_squares",
                            details=details,
                            attempt_index=attempt_index,
                            args=args,
                        ),
                        write_lock=write_lock,
                    )
                    if _p1_detection_attempts_exhausted(args, attempt_index):
                        print("[Bridge] Stopping after rejected Player 1 detection attempts.")
                        return
                    show_reference_recapture_hint = True
                    attempt_index += 1
                    continue

                resolved = resolver_holder["resolver"].resolve_player1(
                    observed_move,
                    resolver_changed_squares,
                    max_score=args.max_resolved_score,
                )
                if (
                    resolved is None
                    and resolver_changed_squares is not changed_squares
                ):
                    resolved = resolver_holder["resolver"].resolve_player1(
                        observed_move,
                        changed_squares,
                        max_score=args.max_resolved_score,
                    )
                if resolved is None and args.allow_observed_fallback:
                    resolved = resolver_holder["resolver"].fallback_from_observed(observed_move)
                if resolved is None:
                    details = {
                        "significant_changed_count": significant_count,
                        "changed_square_count": len(changed_squares),
                        "min_significant_changes": args.min_significant_changes,
                        "max_resolved_score": args.max_resolved_score,
                        "allow_observed_fallback": bool(args.allow_observed_fallback),
                    }
                    rejection_path = _write_rejected_player1_attempt(
                        analysis_wrap=analysis_wrap,
                        reason="legal_move_resolution_rejected",
                        details=details,
                        resolver=resolver_holder["resolver"],
                        turn_debug_dir=turn_debug_dir,
                    )
                    _print_rejected_player1_attempt(
                        reason="legal_move_resolution_rejected",
                        details=details,
                        rejection_path=rejection_path,
                        attempt_index=attempt_index,
                        args=args,
                    )
                    _send_status_message(
                        conn_file,
                        _player1_rejection_status_payload(
                            reason="legal_move_resolution_rejected",
                            details=details,
                            attempt_index=attempt_index,
                            args=args,
                        ),
                        write_lock=write_lock,
                    )
                    if _p1_detection_attempts_exhausted(args, attempt_index):
                        print("[Bridge] Stopping after rejected Player 1 detection attempts.")
                        return
                    show_reference_recapture_hint = True
                    attempt_index += 1
                    continue

                break

            analysis_dir_raw = analysis_wrap.get("analysis_dir")
            analysis_dir: Path | None = None
            if isinstance(analysis_dir_raw, str):
                analysis_dir = Path(analysis_dir_raw)
                if resolver_changed_squares:
                    (analysis_dir / "player1_resolver_changed_squares.json").write_text(
                        json.dumps(resolver_changed_squares, ensure_ascii=True, indent=2),
                        encoding="utf-8",
                    )
                (analysis_dir / "player1_resolved_move.json").write_text(
                    json.dumps(resolved.to_dict(), ensure_ascii=True, indent=2),
                    encoding="utf-8",
                )
                (analysis_dir / "pi_resolver_state_after_p1.json").write_text(
                    json.dumps(resolver_holder["resolver"].debug_state(), ensure_ascii=True, indent=2),
                    encoding="utf-8",
                )

            print(
                f"[Bridge] Resolved P1 move with {resolved.resolver}: "
                f"steps={len(resolved.steps)} capture={resolved.capture} "
                f"special={resolved.special} score={resolved.score}"
            )
            show_reference_recapture_hint = False
            manual_green_captures = _detect_manual_green_captures(
                analysis_payload=analysis_payload,
                diff_threshold=int(analysis_config.diff_threshold),
                expected_count=_expected_manual_green_capture_count(game=game_holder["game"], resolved=resolved),
                analysis_dir=analysis_dir,
            )
            if manual_green_captures:
                print(
                    "[Bridge] Detected manual green-area capture source(s): "
                    + ", ".join(
                        f"{item['x_pct']:.2f}%,{item['y_pct']:.2f}%"
                        for item in manual_green_captures
                    )
                )
            _send_status_message(
                conn_file,
                {
                    "type": "status",
                    "level": "success",
                    "title": "Player 1 Move Accepted",
                    "message": (
                        f"Accepted {len(resolved.steps)} logical step(s) with {resolved.resolver}."
                    ),
                    "details": [f"Resolver score: {resolved.score}"],
                    "duration_ms": 1800,
                },
                write_lock=write_lock,
            )
            for idx, step in enumerate(resolved.steps, start=1):
                p1_from, p1_to = resolver.step_to_square_pair(step)
                p1_msg = {"type": "p1_move", "from": p1_from, "to": p1_to}
                if manual_green_captures and idx == len(resolved.steps):
                    p1_msg["manual_green_captures"] = manual_green_captures
                _write_json_line(conn_file, p1_msg, write_lock=write_lock)
                print(f"[Bridge] Sent P1 step {idx}/{len(resolved.steps)}: {p1_from} -> {p1_to}")

            while True:
                incoming = p2_queue.get()
                msg_type = incoming.get("type")
                if msg_type == "__disconnect__":
                    raise ConnectionError(str(incoming.get("error", "GUI disconnected")))
                if msg_type != "p2_move":
                    print(f"[Bridge] Ignoring non-p2 message: {incoming}")
                    continue
                break

            p2_from = incoming.get("from")
            p2_to = incoming.get("to")
            p2_steps = _extract_turn_steps(incoming)
            if p2_steps and len(p2_steps) > 1:
                chain = " | ".join(f"{frm}->{to}" for frm, to in p2_steps)
                print(f"[Bridge] Received P2 move from Software-GUI: {chain}")
            else:
                print(f"[Bridge] Received P2 move from Software-GUI: {p2_from} -> {p2_to}")
            if analysis_dir is not None:
                (analysis_dir / "player2_move.json").write_text(
                    json.dumps(incoming, ensure_ascii=True, indent=2),
                    encoding="utf-8",
                )

            if p2_steps:
                try:
                    for step_from, step_to in p2_steps:
                        resolver_holder["resolver"].apply_player2(step_from, step_to)
                    post_p2_state = resolver_holder["resolver"].debug_state()
                    if analysis_dir is not None:
                        (analysis_dir / "pi_resolver_state_after_p2.json").write_text(
                            json.dumps(post_p2_state, ensure_ascii=True, indent=2),
                            encoding="utf-8",
                        )
                    if bool(post_p2_state.get("is_checkmate")) or bool(post_p2_state.get("is_stalemate")):
                        runtime_state["game_over"] = True
                        result_name = "checkmate" if bool(post_p2_state.get("is_checkmate")) else "stalemate"
                        print(f"[Bridge] Game over detected after P2 move: {result_name}.")
                        _send_status_message(
                            conn_file,
                            {
                                "type": "status",
                                "level": "warning" if result_name == "stalemate" else "error",
                                "title": "Game Over",
                                "message": (
                                    "Checkmate detected on the Pi-side resolver. Press Enter in the bridge terminal to capture the next starting position,"
                                    " or switch/reset the game from the GUI."
                                    if result_name == "checkmate"
                                    else "Stalemate detected on the Pi-side resolver. Press Enter in the bridge terminal to capture the next starting position, or switch/reset the game from the GUI."
                                ),
                                "sticky": True,
                            },
                            write_lock=write_lock,
                        )
                except Exception as exc:  # noqa: BLE001
                    print(f"[Bridge] Warning: failed to apply P2 move in resolver state: {exc}")

            sequence_lines = _normalize_sequence_lines_from_p2(incoming)
            _write_move_file(sequence_lines, incoming.get("game"), p2_from, p2_to)
            print(f"[Bridge] Wrote {len(sequence_lines)} STM sequence steps to {MOVE_FILE}")
            if analysis_dir is not None:
                shutil.copy2(MOVE_FILE, analysis_dir / "stm32_move_sequence.txt")

            if args.no_stm_send:
                print("[Bridge] --no-stm-send enabled; skipping STM dispatch.")
                if args.capture_mode == "rolling":
                    reference_path = Path(str(analysis_wrap["after_image"]))
                    reference_holder["path"] = reference_path
                    print(f"[Bridge] Rolling reference advanced without STM send: {reference_path}")
            else:
                print("[Bridge] Sending sequence to STM32...")
                stm_started_at = time.perf_counter()
                skip_return_start = args.capture_mode == "rolling"
                stm_result = _send_moves_to_stm(skip_return_start=skip_return_start)
                print(f"[Bridge] STM32 dispatch completed in {time.perf_counter() - stm_started_at:.3f}s")
                print(json.dumps({"bridge_stm_result": stm_result}, indent=2))
                if analysis_dir is not None:
                    (analysis_dir / "stm_result.json").write_text(
                        json.dumps(stm_result, ensure_ascii=True, indent=2),
                        encoding="utf-8",
                    )
                if args.capture_mode == "rolling":
                    print("[Bridge] Capturing updated reference after STM32 move...")
                    capture_started_at = time.perf_counter()
                    reference_path = _capture_live_reference(
                        controller,
                        reopen_stream=args.reopen_camera_each_capture,
                        output_path=game_debug_dir / "rolling_reference.png",
                    )
                    reference_holder["path"] = reference_path
                    print(f"[Bridge] Updated reference capture completed in {time.perf_counter() - capture_started_at:.3f}s")
                    print(f"[Bridge] Rolling reference refreshed: {reference_path}")
                    if skip_return_start:
                        print("[Bridge] Returning STM32 to origin after updated reference capture...")
                        return_started_at = time.perf_counter()
                        return_result = _send_moves_to_stm(return_start_only=True)
                        print(
                            f"[Bridge] STM32 return-to-origin completed in "
                            f"{time.perf_counter() - return_started_at:.3f}s"
                        )
                        if analysis_dir is not None:
                            (analysis_dir / "stm_return_result.json").write_text(
                                json.dumps(return_result, ensure_ascii=True, indent=2),
                                encoding="utf-8",
                            )

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
    initial_game = _normalize_game_name(getattr(config.app, "game", "chess"), default="chess")
    resolver = Player1MoveResolver(
        initial_game,
        camera_square_orientation=config.analysis.camera_square_orientation,
    )
    game_debug_root = Path(args.analysis_out_dir) if args.analysis_out_dir else ROOT / "debug_output" / "Games"
    game_debug_dir = _next_game_debug_dir(game_debug_root)
    runtime_log_path = _install_runtime_log(game_debug_dir)
    _write_game_session_metadata(game_debug_dir, game=initial_game, args=args)
    print(f"[Bridge] Runtime log: {runtime_log_path}")
    print(
        "[Bridge] Analysis thresholds: "
        f"diff_threshold={config.analysis.diff_threshold} "
        f"min_changed_ratio={config.analysis.min_changed_ratio} "
        f"min_significant_changes={args.min_significant_changes} "
        f"max_significant_changes={args.max_significant_changes} "
        f"max_resolved_score={args.max_resolved_score}"
    )
    print(f"[Bridge] Stateful resolver initialized: {resolver.debug_state().get('persistent_state')}")
    print(f"[Bridge] Game debug folder: {game_debug_dir}")

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as srv:
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind((args.host, args.port))
        srv.listen(1)
        print(f"[Bridge] Listening for Software-GUI on {args.host}:{args.port}")
        print("[Bridge] Start Software-GUI now; it should connect as TCP client.")
        conn, addr = srv.accept()
        print(f"[Bridge] Software-GUI connected from {addr[0]}:{addr[1]}")
        try:
            _serve_client(
                args,
                conn,
                controller,
                initial_game,
                resolver,
                config.analysis,
                game_debug_dir,
            )
        except ConnectionError as exc:
            print(f"[Bridge] Connection closed: {exc}")
            return 1
        except Exception as exc:  # noqa: BLE001
            print(f"[Bridge] Error: {exc}")
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
