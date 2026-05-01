#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
from datetime import datetime
import json
import math
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
from flexyboard_camera.app.trigger import GPIOControlPanel, TriggerError, wait_for_gpio_trigger
from flexyboard_camera.utils.config import load_config
from flexyboard_camera.utils.logging_utils import setup_logging
from flexyboard_camera.vision.board_detector import (
    BoardDetection,
    draw_detection_overlay,
    draw_square_grid_overlay,
    generate_square_geometry,
)
from flexyboard_camera.vision.parcheesi_geometry import (
    draw_parcheesi_overlay,
    parcheesi_layout_payload,
    project_parcheesi_regions,
)

MOVE_FILE = ROOT / "sample_data" / "stm32_move_sequence.txt"
DEFAULT_CONFIG_PATH = ROOT / "configs" / "default.yaml"
MANUAL_CORNERS_INFO_PATH = ROOT / "configs" / "corners_info.json"
MANUAL_CORNERS_INFO_TEMP_PATH = ROOT / "configs" / "corners_info_temp.json"
MANUAL_ARCHIVE_ROOT_DIR = ROOT / "configs" / "all_manual"
OUTER_NORM = np.float32([
    [0.0, 0.0],
    [1.0, 0.0],
    [1.0, 1.0],
    [0.0, 1.0],
])
BLACK_THRESH = 100
MORPH_KERNEL_SIZE = 15
_ORIGINAL_STDOUT = sys.stdout
_ORIGINAL_STDERR = sys.stderr
_RUNTIME_LOG_HANDLE: Any | None = None


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
    global _RUNTIME_LOG_HANDLE
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = game_debug_dir / f"bridge_runtime_{timestamp}.log"
    if _RUNTIME_LOG_HANDLE is not None:
        try:
            _RUNTIME_LOG_HANDLE.flush()
            _RUNTIME_LOG_HANDLE.close()
        except Exception:
            pass
    handle = log_path.open("a", encoding="utf-8")
    _RUNTIME_LOG_HANDLE = handle
    sys.stdout = _TeeStream(_ORIGINAL_STDOUT, handle)  # type: ignore[assignment]
    sys.stderr = _TeeStream(_ORIGINAL_STDERR, handle)  # type: ignore[assignment]
    return log_path


def _next_game_debug_dir(debug_root: Path, game: str) -> Path:
    debug_root.mkdir(parents=True, exist_ok=True)
    max_index = 0
    pattern = re.compile(r"^Game(\d+)(?:_(chess|checkers|parcheesi))?$")
    for child in debug_root.iterdir():
        if not child.is_dir():
            continue
        match = pattern.fullmatch(child.name)
        if match:
            max_index = max(max_index, int(match.group(1)))
    normalized = _normalize_game_name(game)
    game_dir = debug_root / f"Game{max_index + 1}_{normalized}"
    game_dir.mkdir(parents=True, exist_ok=False)
    return game_dir


def _create_game_debug_session(
    debug_root: Path,
    *,
    game: str,
    args: argparse.Namespace,
) -> tuple[Path, Path]:
    game_dir = _next_game_debug_dir(debug_root, game)
    runtime_log_path = _install_runtime_log(game_dir)
    _write_game_session_metadata(game_dir, game=game, args=args)
    return game_dir, runtime_log_path


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


def _manual_archive_family_for_game(game: str) -> str:
    normalized = _normalize_game_name(game)
    if normalized == "parcheesi":
        return "parcheesi"
    return "chess_checkers"


def _manual_archive_dir_for_game(game: str) -> Path:
    return MANUAL_ARCHIVE_ROOT_DIR / _manual_archive_family_for_game(game)


def _ensure_manual_archive_layout(game: str) -> Path:
    archive_dir = _manual_archive_dir_for_game(game)
    archive_dir.mkdir(parents=True, exist_ok=True)
    return archive_dir


def _next_manual_archive_game_dir(game: str) -> Path:
    samples_dir = _ensure_manual_archive_layout(game)
    pattern = re.compile(r"^Game(\d+)$")
    max_index = 0
    for child in samples_dir.iterdir():
        if not child.is_dir():
            continue
        match = pattern.fullmatch(child.name)
        if match:
            max_index = max(max_index, int(match.group(1)))
    archive_dir = samples_dir / f"Game{max_index + 1}"
    archive_dir.mkdir(parents=True, exist_ok=False)
    return archive_dir


def _manual_archive_calibration_paths(game: str) -> list[Path]:
    calib_root = _manual_archive_dir_for_game(game)
    archive_paths: list[Path] = []

    for child in sorted(calib_root.iterdir() if calib_root.exists() else []):
        if child.is_dir() and re.fullmatch(r"Game\d+", child.name):
            corners_path = child / "corners_info.json"
            if corners_path.is_file():
                archive_paths.append(corners_path)

    legacy_paths = sorted(
        path
        for path in calib_root.glob("*_corners.json")
        if path.is_file()
    )
    archive_paths.extend(path for path in legacy_paths if path not in archive_paths)
    return archive_paths


def _payload_supports_startup_game(payload: dict[str, Any], game: str) -> bool:
    normalized = _normalize_game_name(game)
    if normalized == "parcheesi":
        return isinstance(payload.get("parcheesi_layout"), dict)
    return isinstance(payload.get("chessboard_corners_px"), list)


def _describe_startup_geometry_strategy(game: str) -> str:
    normalized = _normalize_game_name(game)
    if MANUAL_CORNERS_INFO_PATH.is_file():
        try:
            payload = json.loads(MANUAL_CORNERS_INFO_PATH.read_text(encoding="utf-8"))
            if isinstance(payload, dict) and _payload_supports_startup_game(payload, normalized):
                return "direct configs/corners_info.json"
        except Exception:
            pass
    if normalized == "parcheesi":
        return "outer auto-detect + stored Parcheesi region template (101 mapped locations)"
    if normalized in {"chess", "checkers"}:
        archive_paths = _manual_archive_calibration_paths(normalized)
        if archive_paths:
            return f"outer auto-detect + archived outer->inner calibration ({len(archive_paths)} sample(s))"
    return "legacy live auto-detect fallback"


def _archive_confirmed_geometry(
    *,
    game: str,
    payload: dict[str, Any],
    reference_path: Path,
    overlay_path: Path | None,
) -> Path:
    archive_dir = _next_manual_archive_game_dir(game)
    archive_json_path = archive_dir / "corners_info.json"
    archive_json_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")

    reference_copy_path = archive_dir / "reference.png"
    shutil.copy2(reference_path, reference_copy_path)

    if overlay_path is not None and overlay_path.exists():
        shutil.copy2(overlay_path, archive_dir / "startup_grid_overlay.png")

    return archive_dir


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

    def scale_quad(raw: object) -> list[list[float]] | None:
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

    scaled_outer = scale_quad(payload.get("outer_sheet_corners_px"))
    scaled_chess = scale_quad(payload.get("chessboard_corners_px"))
    is_parcheesi_payload = isinstance(payload.get("parcheesi_layout"), dict)
    if scaled_chess is None and not is_parcheesi_payload:
        raise RuntimeError(f"Invalid chessboard corners in {corners_path}")

    scaled_payload = dict(payload)
    for key, value in payload.items():
        if not isinstance(key, str) or not key.endswith("_corners_px"):
            continue
        scaled = scale_quad(value)
        if scaled is not None:
            scaled_payload[key] = scaled
    scaled_payload["outer_sheet_corners_px"] = scaled_outer
    if scaled_chess is not None:
        scaled_payload["chessboard_corners_px"] = scaled_chess
    scaled_payload["image_size_px"] = {"width": int(target_image_w), "height": int(target_image_h)}
    return scaled_payload


def _order_points(pts: object) -> np.ndarray:
    points = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    s = points.sum(axis=1)
    diff = np.diff(points, axis=1).reshape(-1)
    tl = points[np.argmin(s)]
    br = points[np.argmax(s)]
    tr = points[np.argmin(diff)]
    bl = points[np.argmax(diff)]
    return np.array([tl, tr, br, bl], dtype=np.float32)


def _polygon_area(pts: object) -> float:
    contour = np.asarray(pts, dtype=np.float32).reshape(-1, 1, 2)
    return abs(float(cv2.contourArea(contour)))


def _quad_mean_corner_error(a: object, b: object) -> float:
    qa = _order_points(a)
    qb = _order_points(b)
    return float(np.mean(np.linalg.norm(qa - qb, axis=1)))


def _scale_quad_exact(quad: object, from_size: object, to_size: object) -> np.ndarray:
    if isinstance(from_size, dict):
        fw = float(from_size["width"])
        fh = float(from_size["height"])
    else:
        fw = float(from_size[0])
        fh = float(from_size[1])

    if isinstance(to_size, dict):
        tw = float(to_size["width"])
        th = float(to_size["height"])
    else:
        tw = float(to_size[0])
        th = float(to_size[1])

    q = _order_points(quad).copy()
    q[:, 0] *= tw / fw
    q[:, 1] *= th / fh
    return _order_points(q)


def _quad_is_valid_outer_field(quad: np.ndarray, image_shape: tuple[int, ...]) -> bool:
    h, w = image_shape[:2]
    q = _order_points(quad)
    area = _polygon_area(q)
    area_frac = area / float(w * h)
    x, y, bw, bh = cv2.boundingRect(q.astype(np.int32))
    width_frac = bw / float(w)
    height_frac = bh / float(h)

    if not cv2.isContourConvex(q.reshape(-1, 1, 2).astype(np.float32)):
        return False

    if not (0.18 <= area_frac <= 0.65):
        return False

    if not (0.30 <= width_frac <= 0.80):
        return False

    if not (0.55 <= height_frac <= 1.00):
        return False

    margin = 3
    if x <= margin or y <= margin:
        return False

    return True


def _quad_score(quad: np.ndarray, image_shape: tuple[int, ...], method_bonus: float = 0.0) -> float:
    h, w = image_shape[:2]
    q = _order_points(quad)
    area = _polygon_area(q)
    cx = float(q[:, 0].mean())
    cy = float(q[:, 1].mean())
    center_dist = math.hypot((cx - w / 2.0) / w, (cy - h / 2.0) / h)
    return area - center_dist * 100000.0 + method_bonus


def _coerce_quad(raw: object) -> np.ndarray | None:
    if not isinstance(raw, list) or len(raw) != 4:
        return None
    try:
        return _order_points(raw)
    except Exception:
        return None


def _collect_named_region_quads(payload: dict[str, Any]) -> dict[str, np.ndarray]:
    regions: dict[str, np.ndarray] = {}
    for key, value in payload.items():
        if key in {"outer_sheet_corners_px", "image_size_px"}:
            continue
        if not key.endswith("_corners_px"):
            continue
        quad = _coerce_quad(value)
        if quad is not None:
            regions[key] = quad
    return regions


def _load_outer_to_inner_calibration(calib_paths: list[Path]) -> dict[str, Any]:
    all_outer_px: list[np.ndarray] = []
    all_region_norms: dict[str, list[np.ndarray]] = {}
    all_region_px: dict[str, list[np.ndarray]] = {}
    source_sizes: list[dict[str, Any] | None] = []
    used_files: list[str] = []

    for path in calib_paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        outer_quad = _coerce_quad(data.get("outer_sheet_corners_px"))
        if outer_quad is None:
            continue
        region_quads = _collect_named_region_quads(data)
        if not region_quads:
            continue
        image_size = data.get("image_size_px")

        h_img_to_outer_norm = cv2.getPerspectiveTransform(outer_quad, OUTER_NORM)
        all_outer_px.append(outer_quad)
        for key, region_quad in region_quads.items():
            region_norm = cv2.perspectiveTransform(
                region_quad.reshape(-1, 1, 2),
                h_img_to_outer_norm,
            ).reshape(4, 2)
            all_region_norms.setdefault(key, []).append(region_norm)
            all_region_px.setdefault(key, []).append(region_quad)
        source_sizes.append(image_size if isinstance(image_size, dict) else None)
        used_files.append(str(path))

    if not all_region_norms:
        raise RuntimeError("No valid calibration JSONs loaded.")

    outer_px_stack = np.stack(all_outer_px, axis=0)
    source_image_size = next((size for size in source_sizes if size is not None), None)
    mean_regions_norm_in_outer: dict[str, np.ndarray] = {}
    std_regions_norm_in_outer: dict[str, np.ndarray] = {}
    mean_regions_px: dict[str, np.ndarray] = {}
    std_regions_px: dict[str, np.ndarray] = {}
    region_counts: dict[str, int] = {}

    for key, region_norms in all_region_norms.items():
        region_norm_stack = np.stack(region_norms, axis=0)
        region_px_stack = np.stack(all_region_px[key], axis=0)
        mean_regions_norm_in_outer[key] = np.mean(region_norm_stack, axis=0).astype(np.float32)
        std_regions_norm_in_outer[key] = np.std(region_norm_stack, axis=0).astype(np.float32)
        mean_regions_px[key] = np.mean(region_px_stack, axis=0).astype(np.float32)
        std_regions_px[key] = np.std(region_px_stack, axis=0).astype(np.float32)
        region_counts[key] = int(region_norm_stack.shape[0])

    return {
        "files": used_files,
        "count": len(used_files),
        "region_counts": region_counts,
        "source_image_size": source_image_size,
        "mean_regions_norm_in_outer": mean_regions_norm_in_outer,
        "std_regions_norm_in_outer": std_regions_norm_in_outer,
        "mean_region_px": mean_regions_px,
        "std_region_px": std_regions_px,
        "mean_outer_px": np.mean(outer_px_stack, axis=0).astype(np.float32),
        "std_outer_px": np.std(outer_px_stack, axis=0).astype(np.float32),
    }


def _predict_region_from_outer(outer_quad: np.ndarray, region_norm_in_outer: np.ndarray) -> np.ndarray:
    outer_quad = _order_points(outer_quad).astype(np.float32)
    region_norm_in_outer = _order_points(region_norm_in_outer).astype(np.float32)
    h_outer_norm_to_img = cv2.getPerspectiveTransform(OUTER_NORM, outer_quad)
    region_pred = cv2.perspectiveTransform(
        region_norm_in_outer.reshape(-1, 1, 2),
        h_outer_norm_to_img,
    ).reshape(4, 2)
    return _order_points(region_pred)


def _find_outer_field_corners_initial_style(image: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = image.shape[:2]
    _, mask = cv2.threshold(gray, BLACK_THRESH, 255, cv2.THRESH_BINARY_INV)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (MORPH_KERNEL_SIZE, MORPH_KERNEL_SIZE))
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, hierarchy = cv2.findContours(closed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        raise RuntimeError("No contours found for outer tape.")

    hierarchy = hierarchy[0]
    candidates: list[dict[str, Any]] = []
    for i, contour in enumerate(contours):
        area = float(cv2.contourArea(contour))
        x, y, bw, bh = cv2.boundingRect(contour)
        if area < 0.10 * w * h:
            continue
        if bw < 0.30 * w or bh < 0.50 * h:
            continue
        if x < 0.12 * w or x > 0.85 * w:
            continue

        perimeter = cv2.arcLength(contour, True)
        for eps_frac in (0.005, 0.008, 0.010, 0.015, 0.020):
            approx = cv2.approxPolyDP(contour, eps_frac * perimeter, True)
            if len(approx) != 4:
                continue
            quad = _order_points(approx.reshape(4, 2))
            has_parent = hierarchy[i][3] != -1
            cx = x + bw / 2.0
            cy = y + bh / 2.0
            center_penalty = abs(cx - 0.55 * w) * 100.0 + abs(cy - 0.52 * h) * 40.0
            score = area + (2_000_000.0 if has_parent else 0.0) - center_penalty
            candidates.append(
                {
                    "index": int(i),
                    "quad": quad,
                    "contour": contour,
                    "area": area,
                    "bbox": [int(x), int(y), int(bw), int(bh)],
                    "has_parent": bool(has_parent),
                    "eps_frac": float(eps_frac),
                    "score": float(score),
                }
            )
            break

    if not candidates:
        raise RuntimeError("Could not find a 4-corner inner-tape contour.")

    best = max(candidates, key=lambda candidate: float(candidate["score"]))
    debug = {
        "method": "initial_style_inner_tape_hole_contour",
        "score": float(best["score"]),
        "best_area": float(best["area"]),
        "best_bbox": best["bbox"],
        "has_parent": bool(best["has_parent"]),
        "eps_frac": float(best["eps_frac"]),
        "num_candidates": len(candidates),
    }
    return _order_points(best["quad"]), debug


def _choose_outer_field(image: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    return _find_outer_field_corners_initial_style(image)


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
    parser.add_argument("--status-led-pin", type=int, default=None)
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
    if game not in {"chess", "checkers"}:
        return 0
    steps = getattr(resolved, "steps", None)
    if not isinstance(steps, list) or len(steps) != 1:
        return 0
    if game == "chess" and getattr(resolved, "special", None) == "promotion":
        return 1 + int(bool(getattr(resolved, "capture", False)))
    if not bool(getattr(resolved, "capture", False)):
        return 0
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


def _runtime_panel(args: argparse.Namespace) -> GPIOControlPanel | None:
    panel = getattr(args, "_runtime_gpio_panel", None)
    return panel if isinstance(panel, GPIOControlPanel) else None


def _wait_action_label(args: argparse.Namespace, *, capitalized: bool = False) -> str:
    phrase = "press the button or press Enter" if args.wait_mode == "gpio" else "press Enter"
    if capitalized:
        return phrase[:1].upper() + phrase[1:]
    return phrase


def _set_player_ready_indicator(args: argparse.Namespace, on: bool) -> None:
    panel = _runtime_panel(args)
    if panel is not None:
        panel.set_player_ready(on)


def _poll_terminal_line(timeout_sec: float = 0.0) -> str | None:
    ready, _, _ = select.select([sys.stdin], [], [], timeout_sec)
    if not ready:
        return None
    raw = sys.stdin.readline()
    if raw == "":
        return ""
    return raw.strip().lower()


def _wait_for_turn_trigger(
    args: argparse.Namespace,
    prompt: str,
    *,
    runtime_state: dict[str, Any] | None = None,
    player_ready_led: bool = False,
    blink_led: bool = False,
) -> str:
    if args.wait_mode == "gpio":
        print(prompt)
        panel = _runtime_panel(args)
        blink_state = False
        last_blink_toggle = time.monotonic()
        if player_ready_led and not blink_led:
            _set_player_ready_indicator(args, True)
        else:
            _set_player_ready_indicator(args, False)
        try:
            while True:
                if blink_led:
                    now = time.monotonic()
                    if (now - last_blink_toggle) >= 0.25:
                        blink_state = not blink_state
                        _set_player_ready_indicator(args, blink_state)
                        last_blink_toggle = now
                if runtime_state is not None and runtime_state.get("reset_requested"):
                    runtime_state["reset_requested"] = False
                    print(
                        "\n[Bridge] Runtime reset acknowledged. Start the next game and "
                        f"{_wait_action_label(args)} when ready."
                    )
                    return "__runtime_reset__"
                try:
                    if panel is not None:
                        triggered = panel.wait_for_button(timeout_sec=0.1)
                    else:
                        triggered = wait_for_gpio_trigger(pin=args.gpio_pin, timeout_sec=0.1)
                except TriggerError as exc:
                    raise RuntimeError(str(exc)) from exc
                if triggered:
                    print(f"[Bridge] GPIO button press detected on BCM {args.gpio_pin}")
                    return "__button__"
                terminal_raw = _poll_terminal_line(timeout_sec=0.0)
                if terminal_raw is not None:
                    return terminal_raw
                if args.trigger_timeout is not None:
                    # The panel wait is polled in short slices so runtime reset messages
                    # can still interrupt a GPIO wait. Enforce the user-facing timeout here.
                    deadline = getattr(args, "_runtime_trigger_deadline", None)
                    if deadline is None:
                        deadline = time.monotonic() + float(args.trigger_timeout)
                        setattr(args, "_runtime_trigger_deadline", deadline)
                    if time.monotonic() >= float(deadline):
                        raise RuntimeError("gpio trigger timeout")
        finally:
            if hasattr(args, "_runtime_trigger_deadline"):
                delattr(args, "_runtime_trigger_deadline")
            if player_ready_led or blink_led:
                _set_player_ready_indicator(args, False)
    else:
        if player_ready_led:
            _set_player_ready_indicator(args, True)
        else:
            _set_player_ready_indicator(args, False)
        try:
            input(prompt)
        finally:
            if player_ready_led or blink_led:
                _set_player_ready_indicator(args, False)
    return ""


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
        raw = _wait_for_turn_trigger(
            args,
            f"[Bridge] Turn {turn_index}: make Player 1 move, then {_wait_action_label(args)} to capture current board...",
            runtime_state=runtime_state,
            player_ready_led=True,
        )
        if raw == "__runtime_reset__":
            return "runtime_reset"
        if raw in {"r", "ref", "reference", "recapture"}:
            return "recapture_reference"
        return "capture"

    prompt = f"[Bridge] Turn {turn_index}: make Player 1 move, then {_wait_action_label(args)} to capture current board..."
    if show_recapture_hint:
        prompt += (
            "\n[Bridge] Type 'r' then Enter to recapture the rolling reference "
            "from the current board instead: "
        )
    else:
        prompt += " "
    _set_player_ready_indicator(args, True)
    try:
        raw = _read_terminal_prompt_with_runtime_control(prompt, runtime_state=runtime_state)
    finally:
        _set_player_ready_indicator(args, False)
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
    chessboard_corners_px: list[list[float]] | None,
    out_dir: Path,
    parcheesi_layout: dict[str, Any] | None = None,
) -> Path | None:
    out_dir.mkdir(parents=True, exist_ok=True)
    overlay_path = out_dir / "manual_startup_grid_overlay.png"
    try:
        _write_startup_preview_overlay(
            reference_path=reference_path,
            outer_corners_px=outer_corners_px,
            chessboard_corners_px=chessboard_corners_px,
            out_path=overlay_path,
            parcheesi_layout=parcheesi_layout,
        )
        print(f"[Bridge] Manual startup grid overlay saved: {overlay_path}")
        return overlay_path
    except Exception as exc:  # noqa: BLE001
        print(f"[Bridge] Warning: could not create manual startup grid overlay: {exc}")
        return None


def _grid_lines_from_quad(
    quad: np.ndarray,
    *,
    divisions: int = 8,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], list[tuple[np.ndarray, np.ndarray]]]:
    tl, tr, br, bl = _order_points(quad)
    vertical: list[tuple[np.ndarray, np.ndarray]] = []
    horizontal: list[tuple[np.ndarray, np.ndarray]] = []
    for i in range(divisions + 1):
        t = i / divisions
        top = (1.0 - t) * tl + t * tr
        bottom = (1.0 - t) * bl + t * br
        left = (1.0 - t) * tl + t * bl
        right = (1.0 - t) * tr + t * br
        vertical.append((top, bottom))
        horizontal.append((left, right))
    return vertical, horizontal


def _write_startup_preview_overlay(
    *,
    reference_path: Path,
    outer_corners_px: list[list[float]] | None,
    chessboard_corners_px: list[list[float]] | None,
    out_path: Path,
    parcheesi_layout: dict[str, Any] | None = None,
) -> Path:
    image = cv2.imread(str(reference_path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Could not read startup reference image for preview: {reference_path}")

    outer_quad = _order_points(np.array(outer_corners_px, dtype=np.float32)) if outer_corners_px is not None else None
    if parcheesi_layout is not None:
        if outer_quad is None:
            raise RuntimeError("Parcheesi startup preview requires outer corners.")
        overlay = draw_parcheesi_overlay(
            image,
            outer_corners_px=outer_quad,
            projected_regions=project_parcheesi_regions(outer_quad),
            show_labels=True,
            outer_thickness=5,
            region_thickness=2,
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), overlay)
        return out_path

    if chessboard_corners_px is None:
        raise RuntimeError("Chess/checkers startup preview requires chessboard corners.")

    overlay = image.copy()
    chess_quad = _order_points(np.array(chessboard_corners_px, dtype=np.float32))

    if outer_quad is not None:
        outer_i = np.round(outer_quad).astype(np.int32)
        cv2.polylines(overlay, [outer_i], True, (0, 255, 0), 5, cv2.LINE_AA)
        for idx, pt in enumerate(outer_i):
            cv2.circle(overlay, tuple(pt), 7, (0, 255, 0), -1, cv2.LINE_AA)
            cv2.putText(
                overlay,
                f"O{idx}",
                tuple(pt + np.array([8, -8])),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

    vertical, horizontal = _grid_lines_from_quad(chess_quad, divisions=8)
    for p1, p2 in vertical + horizontal:
        cv2.line(
            overlay,
            tuple(np.round(p1).astype(int)),
            tuple(np.round(p2).astype(int)),
            (255, 0, 0),
            2,
            cv2.LINE_AA,
        )

    chess_i = np.round(chess_quad).astype(np.int32)
    cv2.polylines(overlay, [chess_i], True, (255, 0, 0), 4, cv2.LINE_AA)
    for idx, pt in enumerate(chess_i):
        cv2.circle(overlay, tuple(pt), 6, (0, 0, 255), -1, cv2.LINE_AA)
        cv2.putText(
            overlay,
            f"I{idx}",
            tuple(pt + np.array([8, 18])),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 0, 0),
            2,
            cv2.LINE_AA,
        )

    label = "OUTER = GREEN | INNER = BLUE (from outer->inner reference)"
    cv2.putText(overlay, label, (30, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.82, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(overlay, label, (30, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.82, (255, 255, 255), 2, cv2.LINE_AA)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), overlay)
    return out_path


def _log_startup_geometry_reference_path(
    *,
    outer_method: str,
    inner_method: str,
    calibration_source: str | None = None,
    calibration_files: list[str] | None = None,
    calibration_count: int | None = None,
) -> None:
    print(f"[Bridge] Startup geometry path: outer={outer_method} | inner={inner_method}")
    if calibration_source is not None:
        print(f"[Bridge] Startup geometry calibration source: {calibration_source}")
    if calibration_count is not None:
        print(f"[Bridge] Startup geometry calibration sample count: {calibration_count}")
    if calibration_files:
        print("[Bridge] Startup geometry calibration files:")
        for path in calibration_files:
            print(f"  - {path}")


def _update_selected_game(
    *,
    requested_raw: object,
    game_holder: dict[str, str],
    resolver_holder: dict[str, Any],
    resolver_orientation: str,
    context: str,
) -> None:
    requested_game = _normalize_game_name(requested_raw, default=game_holder["game"])
    previous_game = game_holder["game"]
    game_holder["game"] = requested_game
    resolver_holder["resolver"] = Player1MoveResolver(
        requested_game,
        camera_square_orientation=resolver_orientation,
    )
    if requested_game != previous_game:
        print(f"[Bridge] {context} game selection updated: {previous_game} -> {requested_game}")
    else:
        print(f"[Bridge] {context} game selection reaffirmed: {requested_game}")
    print(f"[Bridge] {context} calibration strategy for {requested_game}: {_describe_startup_geometry_strategy(requested_game)}")


def _write_session_geometry_reference(analysis: dict[str, Any], out_path: Path) -> Path:
    if isinstance(analysis.get("parcheesi_layout"), dict):
        source_image = analysis.get("before_image") or analysis.get("after_image")
        payload = {
            "version": 1,
            "generated_by": "run_pi_software_bridge.py",
            "source": "initial_rolling_reference",
            "game": "parcheesi",
            "source_image": source_image,
            "image_size_px": analysis.get("image_size_px"),
            "outer_sheet_corners_px": analysis.get("outer_sheet_corners_px"),
            "parcheesi_layout": analysis.get("parcheesi_layout"),
        }
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
        return out_path

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


def _write_session_geometry_reference_from_quads(
    *,
    reference_path: Path,
    outer_corners_px: np.ndarray | None,
    chessboard_corners_px: np.ndarray,
    out_path: Path,
    source: str,
    extra_debug: dict[str, Any] | None = None,
) -> Path:
    image_w, image_h = _png_size(reference_path)
    payload: dict[str, Any] = {
        "version": 1,
        "generated_by": "run_pi_software_bridge.py",
        "source": source,
        "source_image": str(reference_path),
        "image_size_px": {
            "width": image_w,
            "height": image_h,
        },
        "outer_sheet_corners_px": (
            [[float(x), float(y)] for x, y in _order_points(outer_corners_px).tolist()]
            if outer_corners_px is not None
            else None
        ),
        "chessboard_corners_px": [[float(x), float(y)] for x, y in _order_points(chessboard_corners_px).tolist()],
    }
    if extra_debug:
        payload["startup_auto_debug"] = extra_debug

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
    return out_path


def _write_session_geometry_reference_from_payload(
    *,
    reference_path: Path,
    payload: dict[str, Any],
    out_path: Path,
    source: str,
    extra_debug: dict[str, Any] | None = None,
) -> Path:
    image_w, image_h = _png_size(reference_path)
    session_payload = {
        key: value
        for key, value in payload.items()
        if key not in {"version", "generated_by", "source", "source_image", "image_size_px", "startup_auto_debug"}
    }
    session_payload["version"] = 1
    session_payload["generated_by"] = "run_pi_software_bridge.py"
    session_payload["source"] = source
    session_payload["source_image"] = str(reference_path)
    session_payload["image_size_px"] = {
        "width": image_w,
        "height": image_h,
    }
    if extra_debug:
        session_payload["startup_auto_debug"] = extra_debug
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(session_payload, ensure_ascii=True, indent=2), encoding="utf-8")
    return out_path


def _store_auto_session_geometry(
    *,
    session_geometry_path: Path,
    preview_path: Path | None,
    game_dir: Path,
    game: str,
) -> Path:
    payload = json.loads(session_geometry_path.read_text(encoding="utf-8"))
    payload["source"] = "auto_startup_geometry"
    payload["generated_by"] = "run_pi_software_bridge.py"

    game_corners_path = game_dir / "corners_info.json"
    game_corners_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
    print(f"[Bridge] Auto startup geometry saved for game: {game_corners_path}")

    MANUAL_CORNERS_INFO_TEMP_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANUAL_CORNERS_INFO_TEMP_PATH.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2),
        encoding="utf-8",
    )
    print(f"[Bridge] Auto startup geometry temp baseline saved: {MANUAL_CORNERS_INFO_TEMP_PATH}")

    if _normalize_game_name(game) == "parcheesi":
        MANUAL_CORNERS_INFO_PATH.parent.mkdir(parents=True, exist_ok=True)
        MANUAL_CORNERS_INFO_PATH.write_text(
            json.dumps(payload, ensure_ascii=True, indent=2),
            encoding="utf-8",
        )
        print(f"[Bridge] Parcheesi startup geometry baseline saved: {MANUAL_CORNERS_INFO_PATH}")

    if preview_path is not None and preview_path.exists():
        preview_copy_path = game_dir / "auto_startup_grid_overlay.png"
        if preview_path.resolve() != preview_copy_path.resolve():
            shutil.copy2(preview_path, preview_copy_path)
        print(f"[Bridge] Auto startup preview archived: {preview_copy_path}")

    reference_path = Path(str(payload.get("source_image", session_geometry_path)))
    archive_dir = _archive_confirmed_geometry(
        game=game,
        payload=payload,
        reference_path=reference_path,
        overlay_path=preview_path,
    )
    print(f"[Bridge] Accepted auto geometry archive saved: {archive_dir}")

    return game_corners_path

def _build_manual_corners_info_payload(
    *,
    reference_path: Path,
    outer_corners_px: object,
    chessboard_corners_px: object,
    game: str,
    extra_geometry: dict[str, Any] | None = None,
) -> dict[str, Any]:
    image_w, image_h = _png_size(reference_path)
    normalized_game = _normalize_game_name(game)
    outer = _quad_entry_from_raw(outer_corners_px, image_w=image_w, image_h=image_h)
    chess = _quad_entry_from_raw(chessboard_corners_px, image_w=image_w, image_h=image_h)
    if outer is None:
        raise RuntimeError("Manual geometry did not include 4 valid outer corners.")

    payload: dict[str, Any] = {
        "version": 1,
        "generated_by": "run_pi_software_bridge.py",
        "source": "manual_startup_geometry",
        "source_image": str(reference_path),
        "image_size_px": {
            "width": image_w,
            "height": image_h,
        },
        "outer_sheet_corners_px": outer["corners_px"],
    }
    if normalized_game == "parcheesi":
        payload["game"] = "parcheesi"
        payload["parcheesi_layout"] = (
            extra_geometry.get("parcheesi_layout")
            if isinstance(extra_geometry, dict) and isinstance(extra_geometry.get("parcheesi_layout"), dict)
            else parcheesi_layout_payload()
        )
    else:
        if chess is None:
            raise RuntimeError("Manual geometry did not include 4 valid inner chessboard corners.")
        payload["chessboard_corners_px"] = chess["corners_px"]
    if extra_geometry:
        for key, value in extra_geometry.items():
            if key in payload:
                continue
            payload[key] = value
    return payload


def _store_manual_corners_info(
    *,
    reference_path: Path,
    outer_corners_px: object,
    chessboard_corners_px: object,
    game_dir: Path,
    game: str,
    extra_geometry: dict[str, Any] | None = None,
) -> Path:
    payload = _build_manual_corners_info_payload(
        reference_path=reference_path,
        outer_corners_px=outer_corners_px,
        chessboard_corners_px=chessboard_corners_px,
        game=game,
        extra_geometry=extra_geometry,
    )

    game_corners_path = game_dir / "corners_info.json"
    game_corners_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
    print(f"[Bridge] Manual corners saved for game: {game_corners_path}")

    MANUAL_CORNERS_INFO_TEMP_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANUAL_CORNERS_INFO_TEMP_PATH.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2),
        encoding="utf-8",
    )
    print(f"[Bridge] Manual corners temp baseline saved: {MANUAL_CORNERS_INFO_TEMP_PATH}")

    if _normalize_game_name(game) == "parcheesi":
        MANUAL_CORNERS_INFO_PATH.parent.mkdir(parents=True, exist_ok=True)
        MANUAL_CORNERS_INFO_PATH.write_text(
            json.dumps(payload, ensure_ascii=True, indent=2),
            encoding="utf-8",
        )
        print(f"[Bridge] Parcheesi manual geometry baseline saved: {MANUAL_CORNERS_INFO_PATH}")

    overlay_path = _draw_manual_reference_overlay(
        reference_path=reference_path,
        outer_corners_px=payload["outer_sheet_corners_px"],
        chessboard_corners_px=payload.get("chessboard_corners_px"),
        out_dir=game_dir,
        parcheesi_layout=payload.get("parcheesi_layout") if _normalize_game_name(game) == "parcheesi" else None,
    )
    archive_dir = _archive_confirmed_geometry(
        game=game,
        payload=payload,
        reference_path=reference_path,
        overlay_path=overlay_path,
    )
    print(f"[Bridge] Manual corners archive saved: {archive_dir}")
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
        preview_path = game_debug_dir / "auto_startup_grid_overlay.png"
        session_path = Path(args.session_geometry_path)
        session_path.parent.mkdir(parents=True, exist_ok=True)

        used_outer_calibration = False
        if MANUAL_CORNERS_INFO_PATH.is_file():
            try:
                detect_started_at = time.perf_counter()
                image_w, image_h = _png_size(reference_path)
                payload = _load_scaled_corners_payload(
                    corners_path=MANUAL_CORNERS_INFO_PATH,
                    target_image_w=image_w,
                    target_image_h=image_h,
                )
                if not _payload_supports_startup_game(payload, game):
                    payload = None
                if payload is None:
                    used_outer_calibration = False
                else:
                    session_path = _write_session_geometry_reference_from_payload(
                        reference_path=reference_path,
                        payload=payload,
                        out_path=session_path,
                        source="startup_direct_from_baseline_corners_info",
                        extra_debug={
                            "calibration_source": "baseline_corners_info_direct",
                            "calibration_files": [str(MANUAL_CORNERS_INFO_PATH)],
                            "calibration_count": 1,
                        },
                    )
                    print(
                        "[Bridge] Startup direct baseline geometry completed in "
                        f"{time.perf_counter() - detect_started_at:.3f}s"
                    )
                    _log_startup_geometry_reference_path(
                        outer_method="baseline_corners_info_direct",
                        inner_method=(
                            "baseline_corners_info_direct"
                            if _normalize_game_name(game) in {"chess", "checkers"}
                            else "baseline_parcheesi_region_layout_direct"
                        ),
                        calibration_source="baseline_corners_info_direct",
                        calibration_files=[str(MANUAL_CORNERS_INFO_PATH)],
                        calibration_count=1,
                    )
                    used_outer_calibration = True
            except Exception as exc:  # noqa: BLE001
                print(
                    "[Bridge] Warning: direct baseline startup geometry failed; "
                    f"falling back to archive/auto paths. Details: {exc}"
                )

        if (not used_outer_calibration) and _normalize_game_name(game) == "parcheesi":
            try:
                detect_started_at = time.perf_counter()
                image = cv2.imread(str(reference_path))
                if image is None:
                    raise RuntimeError(f"Could not read image: {reference_path}")
                outer_quad, outer_debug = _choose_outer_field(image)
                session_path = _write_session_geometry_reference_from_payload(
                    reference_path=reference_path,
                    payload={
                        "game": "parcheesi",
                        "outer_sheet_corners_px": [[float(x), float(y)] for x, y in _order_points(outer_quad).tolist()],
                        "parcheesi_layout": parcheesi_layout_payload(),
                    },
                    out_path=session_path,
                    source="startup_outer_initial_style_parcheesi_template",
                    extra_debug={
                        "outer_method": outer_debug.get("method"),
                        "inner_method": "projected_from_parcheesi_template",
                        "calibration_source": "stored_parcheesi_region_template",
                    },
                )
                print(
                    "[Bridge] Startup outer black-tape detect + Parcheesi template projection completed in "
                    f"{time.perf_counter() - detect_started_at:.3f}s"
                )
                _log_startup_geometry_reference_path(
                    outer_method=str(outer_debug.get("method", "unknown")),
                    inner_method="projected_from_parcheesi_template",
                    calibration_source="stored_parcheesi_region_template",
                    calibration_files=[
                        str(ROOT / "configs" / "all_manual" / "parcheesi" / "stored_data" / "parcheesi_full_101_corner_distances_corrected.json"),
                        str(ROOT / "configs" / "all_manual" / "parcheesi" / "stored_data" / "parcheesi_location_mapping.json"),
                    ],
                    calibration_count=101,
                )
                used_outer_calibration = True
            except Exception as exc:  # noqa: BLE001
                print(
                    "[Bridge] Warning: Parcheesi startup outer detection failed; "
                    f"falling back to legacy live auto-detect. Details: {exc}"
                )

        if (not used_outer_calibration) and _normalize_game_name(game) in {"chess", "checkers"}:
            calib_paths = _manual_archive_calibration_paths(game)
            if calib_paths:
                try:
                    detect_started_at = time.perf_counter()
                    calibration = _load_outer_to_inner_calibration(calib_paths)
                    chess_norm_in_outer = calibration["mean_regions_norm_in_outer"].get("chessboard_corners_px")
                    if not isinstance(chess_norm_in_outer, np.ndarray):
                        raise RuntimeError("Calibration set is missing chessboard_corners_px samples.")
                    image = cv2.imread(str(reference_path))
                    if image is None:
                        raise RuntimeError(f"Could not read image: {reference_path}")
                    outer_quad, outer_debug = _choose_outer_field(image)
                    chess_quad = _predict_region_from_outer(outer_quad, chess_norm_in_outer)
                    session_path = _write_session_geometry_reference_from_quads(
                        reference_path=reference_path,
                        outer_corners_px=outer_quad,
                        chessboard_corners_px=chess_quad,
                        out_path=session_path,
                        source="startup_outer_initial_style_inner_from_archive_calibration",
                        extra_debug={
                            "outer_method": outer_debug.get("method"),
                            "outer_debug": outer_debug,
                            "calibration_count": calibration["count"],
                            "calibration_region_counts": calibration.get("region_counts"),
                            "calibration_files": calibration["files"],
                            "calibration_source": "all_manual_archive_relative_outer_to_inner",
                        },
                    )
                    print(
                        "[Bridge] Startup outer black-tape detect + calibrated inner completed in "
                        f"{time.perf_counter() - detect_started_at:.3f}s"
                    )
                    _log_startup_geometry_reference_path(
                        outer_method=str(outer_debug.get("method", "unknown")),
                        inner_method="projected_from_outer_to_inner_reference_calibration",
                        calibration_source="all_manual_archive_relative_outer_to_inner",
                        calibration_files=[str(path) for path in calibration["files"]],
                        calibration_count=int(calibration["count"]),
                    )
                    used_outer_calibration = True
                except Exception as exc:  # noqa: BLE001
                    print(
                        "[Bridge] Warning: outer black-tape startup detect failed; "
                        f"falling back to legacy live auto-detect. Details: {exc}"
                    )

        if not used_outer_calibration:
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
            _log_startup_geometry_reference_path(
                outer_method=str(analysis.get("algorithm_live_outer_source_before", "unknown")),
                inner_method=str(analysis.get("algorithm_live_inner_source_before", "unknown")),
                calibration_source=str(
                    analysis.get("algorithm_live_outer_to_inner_calibration", {}).get("source", "unknown")
                ),
                calibration_count=(
                    int(analysis.get("algorithm_live_outer_to_inner_calibration", {}).get("count"))
                    if analysis.get("algorithm_live_outer_to_inner_calibration", {}).get("count") is not None
                    else None
                ),
            )
            session_path = _write_session_geometry_reference(analysis, session_path)

        preview_started_at = time.perf_counter()
        payload = json.loads(session_path.read_text(encoding="utf-8"))
        _write_startup_preview_overlay(
            reference_path=reference_path,
            outer_corners_px=payload.get("outer_sheet_corners_px"),
            chessboard_corners_px=payload.get("chessboard_corners_px"),
            out_path=preview_path,
            parcheesi_layout=payload.get("parcheesi_layout") if _normalize_game_name(game) == "parcheesi" else None,
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

    _wait_for_turn_trigger(
        args,
        f"[Bridge] Move Player 1 piece, then {_wait_action_label(args)} to capture AFTER...",
        player_ready_led=True,
    )

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
    force_reference_recapture: bool = False,
    runtime_state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    game_debug_dir = turn_debug_dir.parent if turn_debug_dir is not None else None
    if force_reference_recapture:
        raw = _wait_for_turn_trigger(
            args,
            (
                f"[Bridge] Turn {turn_index}: previous BEFORE/reference was likely bad. "
                f"Correct the board, then {_wait_action_label(args)} to recapture a fresh BEFORE/reference image..."
            ),
            runtime_state=runtime_state,
            blink_led=True,
        )
        if raw == "__runtime_reset__":
            return {"action": "runtime_reset"}
        action = "recapture_reference"
    else:
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


def _send_moves_to_stm(
    *,
    skip_return_start: bool = False,
    return_start_only: bool = False,
    move_file: Path | None = None,
    continue_from_current: bool = False,
) -> dict[str, Any]:
    cmd = [sys.executable, str(ROOT / "scripts" / "send_moves_from_file.py")]
    if move_file is not None:
        cmd.extend(["--move-file", str(move_file)])
    if skip_return_start:
        cmd.append("--skip-return-start")
    if return_start_only:
        cmd.append("--return-start-only")
    if continue_from_current:
        cmd.append("--continue-from-current")
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
    conn: socket.socket,
    conn_file: Any,
    *,
    reference_path: Path,
    selector_image_path: Path | None = None,
    game_debug_dir: Path,
    game: str,
    game_holder: dict[str, str] | None = None,
    resolver_holder: dict[str, Any] | None = None,
    resolver_orientation: str | None = None,
    startup_message_queue: Queue[dict[str, Any]] | None = None,
) -> Path | None:
    normalized_game = _normalize_game_name(game)
    outer_only = normalized_game == "parcheesi"

    while True:
        selector_path = selector_image_path if selector_image_path is not None and selector_image_path.exists() else reference_path
        image_b64 = base64.b64encode(selector_path.read_bytes()).decode("ascii")
        _write_json_line(
            conn_file,
            {
                "type": "geometry_calibration_request",
                "title": "Manual Parcheesi Geometry" if outer_only else "Manual Board Geometry",
                "summary": (
                    "Click the 4 green outer-board corners only. "
                    "Use order: top-left, top-right, bottom-right, bottom-left."
                    if outer_only
                    else (
                        "Click 4 green outer-grid corners first, then 4 yellow inner-grid corners. "
                        "Use order: top-left, top-right, bottom-right, bottom-left."
                    )
                ),
                "source_path": str(selector_path),
                "image_png_b64": image_b64,
                "mode": "outer_only" if outer_only else "outer_inner",
                "game": normalized_game,
            },
        )
        print("[Bridge] Sent initial board image to Software-GUI for manual grid selection.")

        msg = _read_startup_control_message(
            conn,
            conn_file,
            startup_message_queue=startup_message_queue,
        )
        if (
            msg.get("type") in {"set_game", "reset_game"}
            and game_holder is not None
            and resolver_holder is not None
            and resolver_orientation is not None
        ):
            _update_selected_game(
                requested_raw=msg.get("game"),
                game_holder=game_holder,
                resolver_holder=resolver_holder,
                resolver_orientation=resolver_orientation,
                context="Startup",
            )
            continue
        if msg.get("type") != "geometry_calibration_result":
            print(f"[Bridge] Ignoring message while waiting for manual geometry: {msg.get('type')}")
            continue
        if not bool(msg.get("accepted")):
            print("[Bridge] Manual startup geometry was cancelled.")
            return None
        extra_geometry = {
            key: value
            for key, value in msg.items()
            if key not in {"type", "accepted", "error", "outer_corners_px", "chessboard_corners_px"}
        }
        if outer_only:
            extra_geometry["game"] = "parcheesi"
            extra_geometry["parcheesi_layout"] = parcheesi_layout_payload()
            candidate_payload = _build_manual_corners_info_payload(
                reference_path=reference_path,
                outer_corners_px=msg.get("outer_corners_px"),
                chessboard_corners_px=msg.get("chessboard_corners_px"),
                game=game,
                extra_geometry=extra_geometry,
            )
            candidate_geometry_path = game_debug_dir / "manual_parcheesi_geometry_candidate.json"
            candidate_geometry_path.write_text(
                json.dumps(candidate_payload, ensure_ascii=True, indent=2),
                encoding="utf-8",
            )
            candidate_preview_path = game_debug_dir / "manual_parcheesi_geometry_candidate_overlay.png"
            _write_startup_preview_overlay(
                reference_path=reference_path,
                outer_corners_px=candidate_payload.get("outer_sheet_corners_px"),
                chessboard_corners_px=None,
                out_path=candidate_preview_path,
                parcheesi_layout=candidate_payload.get("parcheesi_layout"),
            )
            print("[Bridge] Manual Parcheesi outer corners selected; waiting for projected region confirmation...")
            accepted = _confirm_startup_geometry(
                conn,
                conn_file,
                candidate_preview_path,
                candidate_geometry_path,
                game_holder=game_holder,
                resolver_holder=resolver_holder,
                resolver_orientation=resolver_orientation,
                startup_message_queue=startup_message_queue,
            )
            if not accepted:
                print("[Bridge] Manual Parcheesi region overlay rejected; restarting outer-corner selection.")
                selector_image_path = candidate_preview_path if candidate_preview_path.exists() else selector_image_path
                continue
        path = _store_manual_corners_info(
            reference_path=reference_path,
            outer_corners_px=msg.get("outer_corners_px"),
            chessboard_corners_px=msg.get("chessboard_corners_px"),
            game_dir=game_debug_dir,
            game=game,
            extra_geometry=extra_geometry,
        )
        print(f"[Bridge] Manual startup geometry saved: {path}")
        return path


def _confirm_startup_geometry(
    conn: socket.socket,
    conn_file: Any,
    preview_path: Path | None,
    geometry_path: Path,
    *,
    game_holder: dict[str, str] | None = None,
    resolver_holder: dict[str, Any] | None = None,
    resolver_orientation: str | None = None,
    startup_message_queue: Queue[dict[str, Any]] | None = None,
) -> bool:
    is_parcheesi_preview = False
    try:
        payload = json.loads(geometry_path.read_text(encoding="utf-8"))
        is_parcheesi_preview = isinstance(payload, dict) and isinstance(payload.get("parcheesi_layout"), dict)
    except Exception:
        is_parcheesi_preview = False
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
            "title": "Confirm Detected Parcheesi Regions" if is_parcheesi_preview else "Confirm Detected Board Grid",
            "summary": (
                "Review the startup green outer corners and projected Parcheesi regions. "
                "Click Confirm only if the outer board and region overlay line up with the physical board. "
                "If not, reject it and the bridge will switch to manual outer-corner selection."
                if is_parcheesi_preview
                else (
                    "Review the startup green/blue grid. Click Confirm only if the board outline "
                    "and 8x8 chess/checkers grid line up with the physical board. "
                    "If not, reject it and the bridge will switch to manual corner selection."
                )
            ),
            "source_path": str(preview_path),
            "geometry_path": str(geometry_path),
            "image_png_b64": image_b64,
        },
    )
    print("[Bridge] Sent startup grid preview to Software-GUI; waiting for Confirm...")

    while True:
        msg = _read_startup_control_message(
            conn,
            conn_file,
            startup_message_queue=startup_message_queue,
        )
        if (
            msg.get("type") in {"set_game", "reset_game"}
            and game_holder is not None
            and resolver_holder is not None
            and resolver_orientation is not None
        ):
            _update_selected_game(
                requested_raw=msg.get("game"),
                game_holder=game_holder,
                resolver_holder=resolver_holder,
                resolver_orientation=resolver_orientation,
                context="Startup",
            )
            continue
        if msg.get("type") != "geometry_confirm":
            print(f"[Bridge] Ignoring message while waiting for geometry confirmation: {msg.get('type')}")
            continue
        accepted = bool(msg.get("accepted"))
        if accepted:
            print("[Bridge] Startup grid confirmed.")
        else:
            print("[Bridge] Startup grid rejected.")
        return accepted


def _startup_geometry_is_direct_baseline(session_geometry_path: Path) -> bool:
    try:
        payload = json.loads(session_geometry_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    return str(payload.get("source")) == "startup_direct_from_baseline_corners_info"


def _resolve_startup_geometry(
    conn: socket.socket,
    conn_file: Any,
    *,
    reference_path: Path,
    args: argparse.Namespace,
    game: str,
    analysis_config: Any,
    game_debug_dir: Path,
    game_holder: dict[str, str] | None = None,
    resolver_holder: dict[str, Any] | None = None,
    resolver_orientation: str | None = None,
    startup_message_queue: Queue[dict[str, Any]] | None = None,
) -> Path | None:
    normalized_game = _normalize_game_name(game)
    if args.startup_geometry_mode == "manual":
        return _request_manual_startup_geometry(
            conn,
            conn_file,
            reference_path=reference_path,
            game_debug_dir=game_debug_dir,
            game=game,
            game_holder=game_holder,
            resolver_holder=resolver_holder,
            resolver_orientation=resolver_orientation,
            startup_message_queue=startup_message_queue,
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
        if _startup_geometry_is_direct_baseline(session_geometry_path) and normalized_game != "parcheesi":
            print("[Bridge] Startup geometry auto-accepted from configs/corners_info.json.")
        elif normalized_game == "parcheesi" or not args.skip_geometry_confirmation:
            accepted = _confirm_startup_geometry(
                conn,
                conn_file,
                preview_path,
                session_geometry_path,
                game_holder=game_holder,
                resolver_holder=resolver_holder,
                resolver_orientation=resolver_orientation,
                startup_message_queue=startup_message_queue,
            )
        if accepted:
            return _store_auto_session_geometry(
                session_geometry_path=session_geometry_path,
                preview_path=preview_path,
                game_dir=game_debug_dir,
                game=game,
            )

        print("[Bridge] Switching to manual startup geometry selection...")
        manual_selector_image = preview_path if normalized_game == "parcheesi" else None

    else:
        print("[Bridge] Automatic startup geometry failed; switching to manual selection...")
        manual_selector_image = None

    return _request_manual_startup_geometry(
        conn,
        conn_file,
        reference_path=reference_path,
        selector_image_path=manual_selector_image,
        game_debug_dir=game_debug_dir,
        game=game,
        game_holder=game_holder,
        resolver_holder=resolver_holder,
        resolver_orientation=resolver_orientation,
        startup_message_queue=startup_message_queue,
    )


def _initialize_rolling_reference_for_game(
    conn: socket.socket,
    conn_file: Any,
    *,
    controller: EndTurnController,
    args: argparse.Namespace,
    game_holder: dict[str, str],
    resolver_holder: dict[str, Any],
    resolver_orientation: str,
    analysis_config: Any,
    game_debug_dir: Path,
    output_name: str,
    prompt: str,
    startup_message_queue: Queue[dict[str, Any]] | None = None,
) -> Path | None:
    _set_player_ready_indicator(args, False)
    print(f"[Bridge] Startup selected game: {game_holder['game']}")
    print(f"[Bridge] Startup calibration strategy for {game_holder['game']}: {_describe_startup_geometry_strategy(game_holder['game'])}")
    blink_state = False
    last_blink_toggle = time.monotonic()
    try:
        if args.wait_mode == "gpio":
            print(prompt)
            panel = _runtime_panel(args)
            while True:
                now = time.monotonic()
                if args.status_led_pin is not None and (now - last_blink_toggle) >= 0.25:
                    blink_state = not blink_state
                    _set_player_ready_indicator(args, blink_state)
                    last_blink_toggle = now
                msg = _poll_startup_control_message(
                    conn,
                    conn_file,
                    startup_message_queue=startup_message_queue,
                    timeout_sec=0.0,
                )
                if msg is not None:
                    msg_type = msg.get("type")
                    if msg_type in {"set_game", "reset_game"}:
                        _update_selected_game(
                            requested_raw=msg.get("game"),
                            game_holder=game_holder,
                            resolver_holder=resolver_holder,
                            resolver_orientation=resolver_orientation,
                            context="Startup",
                        )
                        print(prompt)
                        continue
                    print(f"[Bridge] Ignoring startup control message before initial capture: {msg_type}")
                    print(prompt)
                    continue
                terminal_raw = _poll_terminal_line(timeout_sec=0.0)
                if terminal_raw is not None:
                    break
                try:
                    if panel is not None:
                        triggered = panel.wait_for_button(timeout_sec=0.1)
                    else:
                        triggered = wait_for_gpio_trigger(pin=args.gpio_pin, timeout_sec=0.1)
                except TriggerError as exc:
                    raise RuntimeError(str(exc)) from exc
                if triggered:
                    print(f"[Bridge] GPIO button press detected on BCM {args.gpio_pin}")
                    break
        else:
            print(prompt, end="", flush=True)
            while True:
                now = time.monotonic()
                if args.status_led_pin is not None and (now - last_blink_toggle) >= 0.25:
                    blink_state = not blink_state
                    _set_player_ready_indicator(args, blink_state)
                    last_blink_toggle = now
                msg = _poll_startup_control_message(
                    conn,
                    conn_file,
                    startup_message_queue=startup_message_queue,
                    timeout_sec=0.0,
                )
                if msg is not None:
                    msg_type = msg.get("type")
                    if msg_type in {"set_game", "reset_game"}:
                        print()
                        _update_selected_game(
                            requested_raw=msg.get("game"),
                            game_holder=game_holder,
                            resolver_holder=resolver_holder,
                            resolver_orientation=resolver_orientation,
                            context="Startup",
                        )
                        print(prompt, end="", flush=True)
                        continue
                    print(f"\n[Bridge] Ignoring startup control message before initial capture: {msg_type}")
                    print(prompt, end="", flush=True)
                    continue
                ready, _, _ = select.select([sys.stdin], [], [], 0.2)
                if sys.stdin in ready:
                    raw = sys.stdin.readline()
                    if raw == "":
                        return None
                    break
    finally:
        if args.status_led_pin is not None:
            _set_player_ready_indicator(args, False)
    capture_started_at = time.perf_counter()
    reference_path = _capture_startup_reference(
        controller,
        output_path=game_debug_dir / output_name,
    )
    print(f"[Bridge] Initial reference capture completed in {time.perf_counter() - capture_started_at:.3f}s")
    print(f"[Bridge] Initial reference captured: {reference_path}")
    if not args.slow_live_analysis:
        session_geometry = _resolve_startup_geometry(
            conn,
            conn_file,
            reference_path=reference_path,
            args=args,
            game=game_holder["game"],
            analysis_config=analysis_config,
            game_debug_dir=game_debug_dir,
            game_holder=game_holder,
            resolver_holder=resolver_holder,
            resolver_orientation=resolver_orientation,
            startup_message_queue=startup_message_queue,
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


def _write_move_file(
    lines: list[str],
    game: str | None,
    p2_from: str | None,
    p2_to: str | None,
    *,
    path: Path = MOVE_FILE,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = [
        "# Auto-generated by run_pi_software_bridge.py",
        f"# game={game or 'unknown'} p2_move={p2_from or '?'}->{p2_to or '?'}",
        "# Format: source -> dest (board: x,y  |  off-board: x%,y%)",
    ]
    path.write_text("\n".join([*header, *lines]) + "\n", encoding="utf-8")


def _extract_player_ready_after_step_count(msg: dict[str, Any], total_steps: int) -> int | None:
    raw = msg.get("player_ready_after_step_count")
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return None
    if value <= 0 or value > total_steps:
        return None
    return value


def _dispatch_stm_sequence_batches(
    *,
    args: argparse.Namespace,
    sequence_lines: list[str],
    incoming: dict[str, Any],
    analysis_dir: Path | None,
) -> dict[str, Any]:
    ready_after_step_count = _extract_player_ready_after_step_count(incoming, len(sequence_lines))
    game = incoming.get("game")
    p2_from = incoming.get("from")
    p2_to = incoming.get("to")

    first_batch = sequence_lines
    second_batch: list[str] = []
    if ready_after_step_count is not None and ready_after_step_count < len(sequence_lines):
        first_batch = sequence_lines[:ready_after_step_count]
        second_batch = sequence_lines[ready_after_step_count:]

    first_move_file = MOVE_FILE
    if second_batch:
        first_move_file = MOVE_FILE.with_name("stm32_move_sequence_batch1.txt")
    _write_move_file(first_batch, game, p2_from, p2_to, path=first_move_file)

    batches: list[dict[str, Any]] = []
    first_started_at = time.perf_counter()
    first_result = _send_moves_to_stm(
        skip_return_start=True,
        move_file=first_move_file,
    )
    batches.append(
        {
            "phase": "pre_ready" if second_batch else "full_sequence",
            "move_file": str(first_move_file),
            "duration_sec": round(time.perf_counter() - first_started_at, 3),
            "result": first_result,
        }
    )

    if ready_after_step_count is not None and args.capture_mode != "rolling":
        _set_player_ready_indicator(args, True)
        print("[Bridge] Player 1 ready indicator ON.")

    if analysis_dir is not None and second_batch:
        shutil.copy2(first_move_file, analysis_dir / "stm32_move_sequence_pre_ready.txt")

    if second_batch:
        second_move_file = MOVE_FILE.with_name("stm32_move_sequence_post_ready.txt")
        _write_move_file(second_batch, game, p2_from, p2_to, path=second_move_file)
        second_started_at = time.perf_counter()
        second_result = _send_moves_to_stm(
            skip_return_start=True,
            move_file=second_move_file,
            continue_from_current=True,
        )
        batches.append(
            {
                "phase": "post_ready",
                "move_file": str(second_move_file),
                "duration_sec": round(time.perf_counter() - second_started_at, 3),
                "result": second_result,
            }
        )
        if analysis_dir is not None:
            shutil.copy2(second_move_file, analysis_dir / "stm32_move_sequence_post_ready.txt")

    final_result = batches[-1]["result"] if batches else {}
    return {
        "status": "ok",
        "mode": "split" if second_batch else "single",
        "player_ready_after_step_count": ready_after_step_count,
        "batches": batches,
        "final_status": final_result.get("final_status"),
    }


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

    if reason == "excessive_significant_changed_squares":
        extra = [
            "The previous clean reference image is still active; the rejected after-frame was not promoted to the new reference.",
            f"Correct the physical board first. The next button/Enter will recapture a fresh BEFORE/reference image instead of analyzing a move.",
            f"After that, make the actual Player 1 move and {_wait_action_label(args)} again for the new AFTER image.",
        ]
    else:
        extra = [
            "The previous clean reference image is still active; the rejected after-frame was not promoted to the new reference.",
            f"Remove any hand/arm or obstruction, leave the board in the intended state, and {_wait_action_label(args)} to retry the same turn.",
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


def _poll_startup_control_message(
    conn: socket.socket,
    conn_file: Any,
    *,
    startup_message_queue: Queue[dict[str, Any]] | None = None,
    timeout_sec: float = 0.0,
) -> dict[str, Any] | None:
    if startup_message_queue is not None:
        try:
            return startup_message_queue.get(timeout=timeout_sec)
        except Empty:
            return None
    ready, _, _ = select.select([conn], [], [], timeout_sec)
    if conn in ready:
        return _read_json_line(conn_file)
    return None


def _read_startup_control_message(
    conn: socket.socket,
    conn_file: Any,
    *,
    startup_message_queue: Queue[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    if startup_message_queue is not None:
        return startup_message_queue.get()
    return _read_json_line(conn_file)


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
    game_debug_dir_holder: dict[str, Path],
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
        startup_queue = runtime_state.get("startup_message_queue")
        if runtime_state.get("startup_interactive") and startup_queue is not None:
            if msg_type in {"set_game", "reset_game", "geometry_confirm", "geometry_calibration_result"}:
                startup_queue.put(msg)
                continue
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
                wait_for_trigger = bool(msg.get("wait_for_trigger", False))
                if wait_for_trigger:
                    raw = _wait_for_turn_trigger(
                        args,
                        (
                            "[Bridge] Software state changed. Arrange the physical board to match, "
                            f"then {_wait_action_label(args)} to capture a fresh rolling reference..."
                        ),
                        runtime_state=runtime_state,
                        blink_led=True,
                    )
                    if raw == "__runtime_reset__":
                        continue
                started_at = time.perf_counter()
                reference_path = _capture_runtime_reference(
                    controller,
                    args=args,
                    game_debug_dir=game_debug_dir_holder["path"],
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
            _update_selected_game(
                requested_raw=msg.get("game"),
                game_holder=game_holder,
                resolver_holder=resolver_holder,
                resolver_orientation=resolver_orientation,
                context="Runtime",
            )
            runtime_state["reset_requested"] = True
            runtime_state["await_new_game_capture"] = True
            runtime_state["game_over"] = False
            _send_status_message(
                conn_file,
                {
                    "type": "status",
                    "level": "info",
                    "title": "Game Selection Updated",
                    "message": f"Physical board/game rules switched to {game_holder['game']}.",
                    "details": [
                        f"Arrange the board for the selected game, then {_wait_action_label(args)} to capture a new starting reference."
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
                    f"{_wait_action_label(args, capitalized=True)} after the board is arranged to capture a fresh starting reference.",
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
        if msg_type == "game_over":
            runtime_state["reset_requested"] = True
            runtime_state["await_new_game_capture"] = True
            runtime_state["game_over"] = True
            winner = str(msg.get("winner", "unknown")).strip() or "unknown"
            _send_status_message(
                conn_file,
                {
                    "type": "status",
                    "level": "warning",
                    "title": "Game Over",
                    "message": f"{game_holder['game'].title()} game ended. Winner: {winner}.",
                    "details": [
                        f"Arrange the board for the next game, then {_wait_action_label(args)} to capture a fresh starting reference."
                    ],
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
    if reason == "excessive_significant_changed_squares":
        print(
            "[Bridge] Keeping the same before/reference image for now. "
            "Correct the physical board, then the next button/Enter will recapture a fresh BEFORE/reference image."
        )
        print(
            f"[Bridge] After that, make the actual Player 1 move and {_wait_action_label(args)} again to capture the new AFTER image."
        )
    else:
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
    game_debug_root: Path,
    game_debug_dir_holder: dict[str, Path],
) -> None:
    conn.settimeout(None)
    with conn:
        conn_file = conn.makefile("rwb")
        write_lock = threading.Lock()
        turn_index = 1
        reference_path: Path | None = None
        show_reference_recapture_hint = False
        game_holder: dict[str, str] = {"game": _normalize_game_name(game)}
        resolver_holder: dict[str, Any] = {"resolver": resolver}
        runtime_state: dict[str, Any] = {
            "reset_requested": False,
            "await_new_game_capture": False,
            "game_over": False,
            "startup_interactive": False,
            "startup_message_queue": Queue(),
        }

        if args.capture_mode == "rolling":
            reference_path = _initialize_rolling_reference_for_game(
                conn,
                conn_file,
                controller=controller,
                args=args,
                game_holder=game_holder,
                resolver_holder=resolver_holder,
                resolver_orientation=resolver.camera_square_orientation,
                analysis_config=analysis_config,
                game_debug_dir=game_debug_dir_holder["path"],
                output_name="initial_reference.png",
                prompt=f"[Bridge] Make sure the physical board matches the software state, then {_wait_action_label(args)} to capture INITIAL reference...",
            )
            if reference_path is None:
                return

        p2_queue: Queue[dict[str, Any]] = Queue()
        control_queue: Queue[dict[str, Any]] = Queue()
        reference_holder: dict[str, Path | None] = {"path": reference_path}
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
                "game_debug_dir_holder": game_debug_dir_holder,
                "resolver_holder": resolver_holder,
                "reference_holder": reference_holder,
                "resolver_orientation": resolver.camera_square_orientation,
                "runtime_state": runtime_state,
            },
            daemon=True,
        )
        control_thread.start()
        setattr(args, "_runtime_reference_getter", lambda: reference_holder["path"])
        show_reference_recapture_hint = False
        force_reference_recapture_next = False

        while True:
            if runtime_state.get("game_over") or runtime_state.get("await_new_game_capture"):
                runtime_state["game_over"] = False
                runtime_state["await_new_game_capture"] = False
                turn_index = 1
                game_debug_dir, runtime_log_path = _create_game_debug_session(
                    game_debug_root,
                    game=game_holder["game"],
                    args=args,
                )
                game_debug_dir_holder["path"] = game_debug_dir
                print(f"[Bridge] Runtime log: {runtime_log_path}")
                print(f"[Bridge] Game debug folder: {game_debug_dir}")
                resolver_holder["resolver"] = Player1MoveResolver(
                    game_holder["game"],
                    camera_square_orientation=resolver.camera_square_orientation,
                )
                if args.capture_mode == "rolling":
                    startup_message_queue = runtime_state["startup_message_queue"]
                    while True:
                        try:
                            startup_message_queue.get_nowait()
                        except Empty:
                            break
                    runtime_state["startup_interactive"] = True
                    try:
                        reference_path = _initialize_rolling_reference_for_game(
                            conn,
                            conn_file,
                            controller=controller,
                            args=args,
                            game_holder=game_holder,
                            resolver_holder=resolver_holder,
                            resolver_orientation=resolver.camera_square_orientation,
                            analysis_config=analysis_config,
                            game_debug_dir=game_debug_dir,
                            output_name="rolling_reference.png",
                            prompt=f"[Bridge] Arrange the physical board for the next game, then {_wait_action_label(args)} to capture a new starting reference...",
                            startup_message_queue=startup_message_queue,
                        )
                    finally:
                        runtime_state["startup_interactive"] = False
                    if reference_path is None:
                        return
                    reference_holder["path"] = reference_path
                show_reference_recapture_hint = False
                force_reference_recapture_next = False

            attempt_index = 1
            restart_for_runtime_reset = False
            while True:
                print("")
                game_debug_dir = game_debug_dir_holder["path"]
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
                        force_reference_recapture=force_reference_recapture_next,
                        runtime_state=runtime_state,
                    )
                    if analysis_wrap.get("action") == "runtime_reset":
                        show_reference_recapture_hint = False
                        force_reference_recapture_next = False
                        restart_for_runtime_reset = True
                        break
                    if analysis_wrap.get("action") == "recapture_reference":
                        ref_raw = analysis_wrap.get("reference_path")
                        if isinstance(ref_raw, str):
                            reference_path = Path(ref_raw)
                        show_reference_recapture_hint = False
                        force_reference_recapture_next = False
                        print("[Bridge] Rolling reference updated. Retry the same turn when ready.")
                        continue
                else:
                    _wait_for_turn_trigger(
                        args,
                        f"[Bridge] Turn {turn_index}: {_wait_action_label(args)} to capture BEFORE...",
                    )
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
                    force_reference_recapture_next = False
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
                    show_reference_recapture_hint = False
                    force_reference_recapture_next = True
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
                    force_reference_recapture_next = False
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
                    force_reference_recapture_next = False
                    attempt_index += 1
                    continue

                break

            if restart_for_runtime_reset:
                continue

            analysis_dir_raw = analysis_wrap.get("analysis_dir")
            analysis_dir: Path | None = None
            incoming: dict[str, Any]
            if game_holder["game"] == "parcheesi":
                if isinstance(analysis_dir_raw, str):
                    analysis_dir = Path(analysis_dir_raw)
                source_obj = observed_move.get("source")
                destination_obj = observed_move.get("destination")
                source_location = (
                    str(source_obj.get("location_id"))
                    if isinstance(source_obj, dict) and source_obj.get("location_id") is not None
                    else None
                )
                destination_location = (
                    str(destination_obj.get("location_id"))
                    if isinstance(destination_obj, dict) and destination_obj.get("location_id") is not None
                    else None
                )
                if not source_location or not destination_location:
                    details = {
                        "observed_move": observed_move,
                        "changed_region_count": int(analysis_payload.get("changed_region_count", 0)),
                    }
                    rejection_path = _write_rejected_player1_attempt(
                        analysis_wrap=analysis_wrap,
                        reason="missing_parcheesi_location_move",
                        details=details,
                        resolver=resolver_holder["resolver"],
                        turn_debug_dir=turn_debug_dir,
                    )
                    _print_rejected_player1_attempt(
                        reason="missing_parcheesi_location_move",
                        details=details,
                        rejection_path=rejection_path,
                        attempt_index=attempt_index,
                        args=args,
                    )
                    _send_status_message(
                        conn_file,
                        _player1_rejection_status_payload(
                            reason="missing_parcheesi_location_move",
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

                if analysis_dir is not None:
                    (analysis_dir / "player1_resolved_move.json").write_text(
                        json.dumps(
                            {
                                "resolver": "parcheesi_direct_region_inference",
                                "steps": [{"from": source_location, "to": destination_location}],
                                "capture": observed_move.get("capture"),
                                "score": observed_move.get("confidence"),
                                "metadata": observed_move.get("metadata", {}),
                            },
                            ensure_ascii=True,
                            indent=2,
                        ),
                        encoding="utf-8",
                    )

                print(
                    "[Bridge] Accepted P1 Parcheesi move: "
                    f"{source_location} -> {destination_location}"
                )
                show_reference_recapture_hint = False
                force_reference_recapture_next = False
                _send_status_message(
                    conn_file,
                    {
                        "type": "status",
                        "level": "success",
                        "title": "Player 1 Move Accepted",
                        "message": f"Accepted Parcheesi move {source_location} -> {destination_location}.",
                        "duration_ms": 1800,
                    },
                    write_lock=write_lock,
                )
                _write_json_line(
                    conn_file,
                    {"type": "p1_move", "from": source_location, "to": destination_location},
                    write_lock=write_lock,
                )
                print(f"[Bridge] Sent P1 move: {source_location} -> {destination_location}")
            else:
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
                force_reference_recapture_next = False
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

            restart_for_runtime_event = False
            while True:
                incoming = p2_queue.get()
                msg_type = incoming.get("type")
                if msg_type == "__disconnect__":
                    raise ConnectionError(str(incoming.get("error", "GUI disconnected")))
                if msg_type == "game_over":
                    runtime_state["reset_requested"] = True
                    runtime_state["await_new_game_capture"] = True
                    runtime_state["game_over"] = True
                    print(
                        "[Bridge] Runtime game-over notice received from Software-GUI. "
                        "Preparing a fresh game capture."
                    )
                    restart_for_runtime_event = True
                    break
                if msg_type != "p2_move":
                    print(f"[Bridge] Ignoring non-p2 message: {incoming}")
                    continue
                break

            if restart_for_runtime_event:
                show_reference_recapture_hint = False
                force_reference_recapture_next = False
                continue

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

            if p2_steps and game_holder["game"] in {"chess", "checkers"}:
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
                                    f"Checkmate detected on the Pi-side resolver. {_wait_action_label(args, capitalized=True)} to capture the next starting position,"
                                    " or switch/reset the game from the GUI."
                                    if result_name == "checkmate"
                                    else f"Stalemate detected on the Pi-side resolver. {_wait_action_label(args, capitalized=True)} to capture the next starting position, or switch/reset the game from the GUI."
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
                ready_after = _extract_player_ready_after_step_count(incoming, len(sequence_lines))
                if ready_after is not None and args.capture_mode != "rolling":
                    _set_player_ready_indicator(args, True)
                    print("[Bridge] Player 1 ready indicator ON.")
                if args.capture_mode == "rolling":
                    reference_path = Path(str(analysis_wrap["after_image"]))
                    reference_holder["path"] = reference_path
                    print(f"[Bridge] Rolling reference advanced without STM send: {reference_path}")
                    if ready_after is not None:
                        _set_player_ready_indicator(args, True)
                        print("[Bridge] Player 1 ready indicator ON.")
            else:
                print("[Bridge] Sending sequence to STM32...")
                stm_started_at = time.perf_counter()
                stm_result = _dispatch_stm_sequence_batches(
                    args=args,
                    sequence_lines=sequence_lines,
                    incoming=incoming,
                    analysis_dir=analysis_dir,
                )
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
                    game_debug_dir = game_debug_dir_holder["path"]
                    reference_path = _capture_live_reference(
                        controller,
                        reopen_stream=args.reopen_camera_each_capture,
                        output_path=game_debug_dir / "rolling_reference.png",
                    )
                    reference_holder["path"] = reference_path
                    print(f"[Bridge] Updated reference capture completed in {time.perf_counter() - capture_started_at:.3f}s")
                    print(f"[Bridge] Rolling reference refreshed: {reference_path}")
                    ready_after = _extract_player_ready_after_step_count(incoming, len(sequence_lines))
                    if ready_after is not None:
                        _set_player_ready_indicator(args, True)
                        print("[Bridge] Player 1 ready indicator ON.")
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
                else:
                    print("[Bridge] Returning STM32 to origin...")
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
    panel: GPIOControlPanel | None = None
    if args.wait_mode == "gpio" or args.status_led_pin is not None:
        try:
            panel = GPIOControlPanel(
                button_pin=args.gpio_pin if args.wait_mode == "gpio" else None,
                led_pin=args.status_led_pin,
            )
        except TriggerError as exc:
            print(f"[Bridge] Error: {exc}")
            return 1
        setattr(args, "_runtime_gpio_panel", panel)

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
    game_debug_dir, runtime_log_path = _create_game_debug_session(
        game_debug_root,
        game=initial_game,
        args=args,
    )
    game_debug_dir_holder: dict[str, Path] = {"path": game_debug_dir}
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
    if args.wait_mode == "gpio":
        print(f"[Bridge] GPIO button trigger enabled on BCM {args.gpio_pin}")
    if args.status_led_pin is not None:
        print(f"[Bridge] Player-ready LED enabled on BCM {args.status_led_pin}")

    try:
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
                    game_debug_root,
                    game_debug_dir_holder,
                )
            except ConnectionError as exc:
                print(f"[Bridge] Connection closed: {exc}")
                return 1
            except Exception as exc:  # noqa: BLE001
                print(f"[Bridge] Error: {exc}")
                return 1
    finally:
        if panel is not None:
            panel.cleanup()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
