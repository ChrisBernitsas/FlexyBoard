#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import json
import re
import shutil
import socket
import struct
import subprocess
import sys
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

MOVE_FILE = ROOT / "sample_data" / "stm32_move_sequence.txt"
DEFAULT_CONFIG_PATH = ROOT / "configs" / "default.yaml"


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
        default="manual",
        help=(
            "manual: send the initial board image to Software-GUI for corner clicks. "
            "auto: detect and confirm the startup geometry automatically."
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


def _prompt_rolling_turn_action(
    args: argparse.Namespace,
    turn_index: int,
    *,
    show_recapture_hint: bool,
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
    raw = input(prompt).strip().lower()
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
    analysis = _run_analysis(
        before_path=before_path,
        after_path=after_path,
        out_dir=str(out_dir) if out_dir is not None else args.analysis_out_dir,
        game=game,
        analysis_config=analysis_config,
        fast_locked_geometry=fast_locked_geometry,
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
    geometry_payload: dict[str, Any],
) -> None:
    try:
        import cv2
        import numpy as np
    except Exception as exc:  # noqa: BLE001
        print(f"[Bridge] Warning: could not create manual reference overlay; OpenCV unavailable: {exc}")
        return

    image = cv2.imread(str(reference_path), cv2.IMREAD_COLOR)
    if image is None:
        print(f"[Bridge] Warning: could not read manual reference image for overlay: {reference_path}")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    overlay = image.copy()

    def draw_quad(points: list[list[float]], color: tuple[int, int, int], prefix: str) -> None:
        pts = np.array(points, dtype=np.float32).round().astype(np.int32)
        if pts.shape != (4, 2):
            return
        cv2.polylines(overlay, [pts.reshape(-1, 1, 2)], isClosed=True, color=color, thickness=4)
        for index, point in enumerate(pts, start=1):
            x, y = int(point[0]), int(point[1])
            cv2.circle(overlay, (x, y), 9, color, thickness=-1)
            cv2.circle(overlay, (x, y), 10, (0, 0, 0), thickness=1)
            cv2.putText(
                overlay,
                f"{prefix}{index}",
                (x + 10, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2,
                cv2.LINE_AA,
            )

    draw_quad(outer_corners_px, (0, 230, 70), "G")
    draw_quad(chessboard_corners_px, (0, 220, 255), "Y")

    overlay_path = out_dir / "manual_startup_grid_overlay.png"
    geometry_path = out_dir / "manual_startup_geometry.json"
    source_copy_path = out_dir / "manual_startup_reference.png"
    cv2.imwrite(str(overlay_path), overlay)
    cv2.imwrite(str(source_copy_path), image)
    geometry_path.write_text(json.dumps(geometry_payload, ensure_ascii=True, indent=2), encoding="utf-8")
    print(f"[Bridge] Manual reference overlay saved: {overlay_path}")
    print(f"[Bridge] Manual reference geometry copy saved: {geometry_path}")


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
        "generated_by": "run_pi_software_bridge.py",
        "source": "initial_rolling_reference",
        "source_image": analysis.get("before_image"),
        "latest_image_size_px": {
            "width": image_w,
            "height": image_h,
        },
        "median_geometry_for_latest_size": {
            "outer_sheet": outer,
            "chessboard": chess,
        },
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
    return out_path


def _write_manual_session_geometry_reference(
    *,
    reference_path: Path,
    outer_corners_px: object,
    chessboard_corners_px: object,
    out_path: Path,
) -> Path:
    image_w, image_h = _png_size(reference_path)
    outer = _quad_entry_from_raw(outer_corners_px, image_w=image_w, image_h=image_h)
    chess = _quad_entry_from_raw(chessboard_corners_px, image_w=image_w, image_h=image_h)
    if outer is None:
        raise RuntimeError("Manual geometry did not include 4 valid outer corners.")
    if chess is None:
        raise RuntimeError("Manual geometry did not include 4 valid inner chessboard corners.")

    payload = {
        "generated_by": "run_pi_software_bridge.py",
        "source": "manual_startup_geometry",
        "source_image": str(reference_path),
        "latest_image_size_px": {
            "width": image_w,
            "height": image_h,
        },
        "median_geometry_for_latest_size": {
            "outer_sheet": outer,
            "chessboard": chess,
        },
        "per_image": [
            {
                "image_path": str(reference_path),
                "image_size_px": {"width": image_w, "height": image_h},
                "outer_sheet": outer,
                "chessboard": chess,
                "annotation_source": "manual_startup_click",
            }
        ],
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
    _draw_manual_reference_overlay(
        reference_path=reference_path,
        outer_corners_px=outer["corners_px"],
        chessboard_corners_px=chess["corners_px"],
        out_dir=ROOT / "configs" / "reference_overlays" / "live_session",
        geometry_payload=payload,
    )
    return out_path


def _build_session_geometry_reference(
    *,
    reference_path: Path,
    args: argparse.Namespace,
    game: str,
    analysis_config: Any,
) -> dict[str, Path] | None:
    print("[Bridge] Building live session geometry from initial reference...")
    try:
        analysis = _run_analysis(
            before_path=reference_path,
            after_path=reference_path,
            out_dir=str(ROOT / "debug_output" / "live_geometry_init"),
            game=game,
            analysis_config=analysis_config,
            fast_locked_geometry=False,
            disable_geometry_reference_override=True,
        )
        session_path = _write_session_geometry_reference(analysis, Path(args.session_geometry_path))
        outputs = analysis.get("outputs") if isinstance(analysis.get("outputs"), dict) else {}
        preview_path_text = (
            outputs.get("before_grid_overlay_live")
            or outputs.get("before_grid_overlay")
            or outputs.get("before_live_overlay")
            or outputs.get("before_overlay")
        )
        preview_path = Path(str(preview_path_text)) if preview_path_text else None
    except Exception as exc:  # noqa: BLE001
        print(
            "[Bridge] Warning: failed to build session geometry; "
            f"falling back to configured geometry reference. Details: {exc}"
        )
        return None

    print(f"[Bridge] Live session geometry saved: {session_path}")
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
    before_path = controller.capture_before(reopen_stream=args.reopen_camera_each_capture)
    print(f"[Bridge] BEFORE captured: {before_path}")

    _wait_for_turn_trigger(args, "[Bridge] Move Player 1 piece, then press Enter to capture AFTER...")

    after_path = controller.capture_after(reopen_stream=args.reopen_camera_each_capture)
    print(f"[Bridge] AFTER captured: {after_path}")

    analysis_out_dir = turn_debug_dir if turn_debug_dir is not None else (
        Path(args.analysis_out_dir) if args.analysis_out_dir else None
    )
    if turn_debug_dir is not None:
        before_path = _copy_frame_for_turn(before_path, turn_debug_dir / "before.png")
        after_path = _copy_frame_for_turn(after_path, turn_debug_dir / "after.png")

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
) -> dict[str, Any]:
    action = _prompt_rolling_turn_action(
        args,
        turn_index,
        show_recapture_hint=show_recapture_hint,
    )
    if action == "recapture_reference":
        new_reference_path = controller.capture_before(reopen_stream=args.reopen_camera_each_capture)
        print(f"[Bridge] Rolling reference recaptured from current board: {new_reference_path}")
        if turn_debug_dir is not None:
            _copy_frame_for_turn(new_reference_path, turn_debug_dir / "reference_recaptured.png")
        return {
            "action": "recapture_reference",
            "reference_path": str(new_reference_path),
        }

    after_path = controller.capture_after(reopen_stream=args.reopen_camera_each_capture)
    print(f"[Bridge] CURRENT captured: {after_path}")
    print(f"[Bridge] Diffing previous reference -> current: {reference_path} -> {after_path}")

    analysis_out_dir = turn_debug_dir if turn_debug_dir is not None else (
        Path(args.analysis_out_dir) if args.analysis_out_dir else None
    )
    if turn_debug_dir is not None:
        reference_path = _copy_frame_for_turn(reference_path, turn_debug_dir / "before.png")
        after_path = _copy_frame_for_turn(after_path, turn_debug_dir / "after.png")

    return _analyze_paths(
        reference_path,
        after_path,
        args,
        game,
        analysis_config,
        fast_locked_geometry=not args.slow_live_analysis,
        out_dir=analysis_out_dir,
    )


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


def _request_manual_startup_geometry(
    conn_file: Any,
    *,
    reference_path: Path,
    geometry_path: Path,
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
        path = _write_manual_session_geometry_reference(
            reference_path=reference_path,
            outer_corners_px=msg.get("outer_corners_px"),
            chessboard_corners_px=msg.get("chessboard_corners_px"),
            out_path=geometry_path,
        )
        print(f"[Bridge] Manual startup geometry saved: {path}")
        return path


def _confirm_startup_geometry(conn_file: Any, preview_path: Path | None, geometry_path: Path) -> bool:
    if preview_path is None or not preview_path.exists():
        print("[Bridge] No startup grid preview image available; using terminal confirmation.")
        reply = input("[Bridge] Continue with this detected geometry? [y/N] ").strip().lower()
        return reply in {"y", "yes"}

    image_b64 = base64.b64encode(preview_path.read_bytes()).decode("ascii")
    _write_json_line(
        conn_file,
        {
            "type": "geometry_preview",
            "title": "Confirm Detected Board Grid",
            "summary": (
                "Review the startup green/yellow grid. Click Confirm only if the board outline "
                "and 8x8 chess/checkers grid line up with the physical board."
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


def _send_status_message(conn_file: Any, payload: dict[str, Any]) -> None:
    try:
        _write_json_line(conn_file, payload)
    except Exception:
        # UI-side status messaging is best-effort only.
        pass


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
        turn_index = 1
        reference_path: Path | None = None
        show_reference_recapture_hint = False

        if args.capture_mode == "rolling":
            input("[Bridge] Make sure the physical board matches the software state, then press Enter to capture INITIAL reference...")
            reference_path = controller.capture_before(reopen_stream=args.reopen_camera_each_capture)
            print(f"[Bridge] Initial reference captured: {reference_path}")
            _copy_frame_for_turn(reference_path, game_debug_dir / "initial_reference.png")
            if not args.slow_live_analysis:
                if args.startup_geometry_mode == "manual":
                    session_geometry = _request_manual_startup_geometry(
                        conn_file,
                        reference_path=reference_path,
                        geometry_path=Path(args.session_geometry_path),
                    )
                    if session_geometry is None:
                        print("[Bridge] Stopping before turn capture because manual geometry was cancelled.")
                        return
                    analysis_config.geometry_reference = str(session_geometry)
                    analysis_config.disable_geometry_reference = False
                else:
                    session_geometry_result = _build_session_geometry_reference(
                        reference_path=reference_path,
                        args=args,
                        game=game,
                        analysis_config=analysis_config,
                    )
                    if session_geometry_result is not None:
                        session_geometry = session_geometry_result["geometry"]
                        analysis_config.geometry_reference = str(session_geometry)
                        analysis_config.disable_geometry_reference = False
                        if not args.skip_geometry_confirmation:
                            accepted = _confirm_startup_geometry(
                                conn_file,
                                session_geometry_result.get("preview"),
                                session_geometry,
                            )
                            if not accepted:
                                print("[Bridge] Stopping before turn capture because startup geometry was rejected.")
                                return

        while True:
            attempt_index = 1
            while True:
                print("")
                turn_debug_dir = (
                    game_debug_dir / f"Move{turn_index}"
                    if attempt_index == 1
                    else game_debug_dir / f"Move{turn_index}_Retry{attempt_index}"
                )
                turn_debug_dir.mkdir(parents=True, exist_ok=True)
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
                        turn_debug_dir=turn_debug_dir,
                        show_recapture_hint=show_reference_recapture_hint,
                    )
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
                        resolver=resolver,
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
                    game=game,
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
                        resolver=resolver,
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
                        resolver=resolver,
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
                    )
                    if _p1_detection_attempts_exhausted(args, attempt_index):
                        print("[Bridge] Stopping after rejected Player 1 detection attempts.")
                        return
                    show_reference_recapture_hint = True
                    attempt_index += 1
                    continue

                resolved = resolver.resolve_player1(
                    observed_move,
                    resolver_changed_squares,
                    max_score=args.max_resolved_score,
                )
                if (
                    resolved is None
                    and resolver_changed_squares is not changed_squares
                ):
                    resolved = resolver.resolve_player1(
                        observed_move,
                        changed_squares,
                        max_score=args.max_resolved_score,
                    )
                if resolved is None and args.allow_observed_fallback:
                    resolved = resolver.fallback_from_observed(observed_move)
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
                        resolver=resolver,
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
                    json.dumps(resolver.debug_state(), ensure_ascii=True, indent=2),
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
                expected_count=_expected_manual_green_capture_count(game=game, resolved=resolved),
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
            )
            for idx, step in enumerate(resolved.steps, start=1):
                p1_from, p1_to = resolver.step_to_square_pair(step)
                p1_msg = {"type": "p1_move", "from": p1_from, "to": p1_to}
                if manual_green_captures and idx == len(resolved.steps):
                    p1_msg["manual_green_captures"] = manual_green_captures
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
                        resolver.apply_player2(step_from, step_to)
                    if analysis_dir is not None:
                        (analysis_dir / "pi_resolver_state_after_p2.json").write_text(
                            json.dumps(resolver.debug_state(), ensure_ascii=True, indent=2),
                            encoding="utf-8",
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
                    print(f"[Bridge] Rolling reference advanced without STM send: {reference_path}")
            else:
                print("[Bridge] Sending sequence to STM32...")
                stm_result = _send_moves_to_stm()
                print(json.dumps({"bridge_stm_result": stm_result}, indent=2))
                if analysis_dir is not None:
                    (analysis_dir / "stm_result.json").write_text(
                        json.dumps(stm_result, ensure_ascii=True, indent=2),
                        encoding="utf-8",
                    )
                if args.capture_mode == "rolling":
                    print("[Bridge] Capturing updated reference after STM32 move...")
                    reference_path = controller.capture_before(reopen_stream=args.reopen_camera_each_capture)
                    reference_copy_path = (
                        analysis_dir / "reference_after_p2.png"
                        if analysis_dir is not None
                        else game_debug_dir / f"reference_after_move_{turn_index}.png"
                    )
                    _copy_frame_for_turn(reference_path, reference_copy_path)
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
    resolver = Player1MoveResolver(
        config.app.game,
        camera_square_orientation=config.analysis.camera_square_orientation,
    )
    game_debug_root = Path(args.analysis_out_dir) if args.analysis_out_dir else ROOT / "debug_output" / "Games"
    game_debug_dir = _next_game_debug_dir(game_debug_root)
    _write_game_session_metadata(game_debug_dir, game=config.app.game, args=args)
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
                config.app.game,
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
