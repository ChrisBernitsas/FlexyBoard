#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from datetime import datetime
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from flexyboard_camera.game.board_models import BoardCoord
from flexyboard_camera.vision.board_detector import (
    BoardDetection,
    draw_detection_overlay,
    draw_square_grid_overlay,
    estimate_chessboard_from_outer_sheet,
    estimate_outer_sheet_from_chessboard,
    generate_square_geometry,
    warp_to_board,
)
from flexyboard_camera.vision.diff_detector import detect_square_changes
from flexyboard_camera.vision.diff_detector import ContourSquareCandidate, SquareChange
from flexyboard_camera.vision.move_inference import InferenceInputs, infer_move
from flexyboard_camera.vision.parcheesi_geometry import (
    draw_parcheesi_overlay,
    parcheesi_layout_payload,
    project_parcheesi_regions,
)
from flexyboard_camera.vision.preprocess import preprocess_frame
from detect_board_geometry import (
    detect_outer_field_by_black_tape_contours,
    choose_outer_field as choose_outer_field_calibrated,
    load_default_games_saved_calibration,
    predict_chess_from_outer as predict_chess_from_outer_calibration,
)


def _parse_triplet(raw: str) -> tuple[int, int, int]:
    parts = [int(item.strip()) for item in raw.split(",")]
    if len(parts) != 3:
        raise ValueError("Expected 3 comma-separated integers")
    return parts[0], parts[1], parts[2]


def _parse_quad(raw: str) -> tuple[float, float, float, float]:
    parts = [float(item.strip()) for item in raw.split(",")]
    if len(parts) != 4:
        raise ValueError("Expected 4 comma-separated floats")
    return parts[0], parts[1], parts[2], parts[3]


def _order_quad(points: np.ndarray) -> np.ndarray:
    pts = points.astype(np.float32).reshape(-1, 2)
    if pts.shape[0] != 4:
        raise ValueError("Quadrilateral must have exactly 4 points")
    sums = pts.sum(axis=1)
    diffs = np.diff(pts, axis=1).reshape(-1)
    ordered = np.zeros((4, 2), dtype=np.float32)
    ordered[0] = pts[np.argmin(sums)]  # TL
    ordered[2] = pts[np.argmax(sums)]  # BR
    ordered[1] = pts[np.argmin(diffs)]  # TR
    ordered[3] = pts[np.argmax(diffs)]  # BL
    return ordered


def _mkdir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _load_live_outer_to_inner_calibration(
    *,
    reference_path: Path,
    disabled: bool,
) -> tuple[np.ndarray, dict[str, object]]:
    del reference_path
    if disabled:
        calibration = load_default_games_saved_calibration()
        stats = dict(calibration)
        stats["source"] = "disabled_request_but_frozen_games_1_to_14_calibration_still_used"
        return np.float32(calibration["mean_chess_norm_in_outer"]), stats
    calibration = load_default_games_saved_calibration()
    return np.float32(calibration["mean_chess_norm_in_outer"]), calibration


def _strip_numpy_debug(debug: dict[str, object] | None) -> dict[str, object] | None:
    if debug is None:
        return None
    cleaned: dict[str, object] = {}
    for key, value in debug.items():
        if isinstance(value, np.ndarray):
            continue
        if isinstance(value, dict):
            nested: dict[str, object] = {}
            for nested_key, nested_value in value.items():
                if isinstance(nested_value, np.ndarray):
                    continue
                if isinstance(nested_value, (np.integer, np.floating)):
                    nested[nested_key] = nested_value.item()
                else:
                    nested[nested_key] = nested_value
            cleaned[key] = nested
            continue
        if isinstance(value, (np.integer, np.floating)):
            cleaned[key] = value.item()
        else:
            cleaned[key] = value
    return cleaned


def _detect_outer_to_inner_geometry(
    *,
    frame_bgr: np.ndarray,
    chess_norm_in_outer: np.ndarray,
    calibration_stats: dict[str, object],
) -> tuple[BoardDetection, dict[str, object]]:
    outer_quad, outer_debug = choose_outer_field_calibrated(
        frame_bgr,
        calibration_stats,
    )
    final_chess_quad = predict_chess_from_outer_calibration(outer_quad, chess_norm_in_outer)
    return BoardDetection(
        outer_sheet_corners=outer_quad,
        chessboard_corners=final_chess_quad,
    ), {
        "anchor_board": None,
        "anchor_debug": None,
        "outer_debug": outer_debug,
        "predicted_outer": None,
        "contour_outer": None,
    }


def _shrink_quad(quad: np.ndarray, shrink: float) -> np.ndarray:
    s = float(shrink)
    if s <= 0.0:
        return quad.astype(np.float32)
    if s >= 0.45:
        s = 0.45
    center = np.mean(quad.astype(np.float32), axis=0)
    return (quad.astype(np.float32) * (1.0 - s)) + (center * s)


def _relative_corner_drift(before_quad: np.ndarray, after_quad: np.ndarray) -> float:
    before = before_quad.astype(np.float32).reshape(4, 2)
    after = after_quad.astype(np.float32).reshape(4, 2)
    mean_corner_distance = float(np.mean(np.linalg.norm(before - after, axis=1)))
    after_diag = float(np.linalg.norm(after[2] - after[0]))
    return mean_corner_distance / max(after_diag, 1.0)


def _mean_corner_distance_px(before_quad: np.ndarray, after_quad: np.ndarray) -> float:
    before = before_quad.astype(np.float32).reshape(4, 2)
    after = after_quad.astype(np.float32).reshape(4, 2)
    return float(np.mean(np.linalg.norm(before - after, axis=1)))


def _quad_area_px2(quad: np.ndarray | None) -> float | None:
    if quad is None:
        return None
    pts = quad.astype(np.float32).reshape(-1, 2)
    if pts.shape[0] != 4:
        return None
    return abs(float(cv2.contourArea(pts.reshape(-1, 1, 2))))


def _quad_encloses_points(quad: np.ndarray, points: np.ndarray) -> bool:
    contour = quad.astype(np.float32).reshape(-1, 1, 2)
    for p in points.astype(np.float32).reshape(-1, 2):
        inside = cv2.pointPolygonTest(contour, (float(p[0]), float(p[1])), False)
        if inside < 0:
            return False
    return True


def _scale_outer_to_target_ratio(
    *,
    outer_quad: np.ndarray,
    chess_quad: np.ndarray,
    target_outer_to_chess_ratio: float,
) -> tuple[np.ndarray, dict[str, float | bool]]:
    info: dict[str, float | bool] = {
        "target_outer_to_chess_ratio": float(target_outer_to_chess_ratio),
        "applied": False,
    }
    outer_area = _quad_area_px2(outer_quad)
    chess_area = _quad_area_px2(chess_quad)
    if outer_area is None or chess_area is None or chess_area <= 1e-6 or outer_area <= 1e-6:
        return outer_quad.astype(np.float32), info
    if target_outer_to_chess_ratio <= 0.0:
        return outer_quad.astype(np.float32), info

    current_ratio = float(outer_area / chess_area)
    info["current_outer_to_chess_ratio"] = current_ratio
    if current_ratio <= 1e-6:
        return outer_quad.astype(np.float32), info

    raw_scale = float(np.sqrt(target_outer_to_chess_ratio / current_ratio))
    # Keep adjustment bounded, but allow stronger correction for obvious outliers.
    scale = float(np.clip(raw_scale, 0.25, 2.20))
    info["raw_scale"] = raw_scale
    info["applied_scale"] = scale

    outer = outer_quad.astype(np.float32).reshape(4, 2)
    # Anchor size lock to chessboard center so the outer box stays tied to board position.
    center = np.mean(chess_quad.astype(np.float32).reshape(4, 2), axis=0, keepdims=True)
    scaled = center + ((outer - center) * scale)
    scaled = _order_quad(scaled.astype(np.float32))

    # Ensure scaled quad still encloses the chessboard; if not, grow slightly.
    if not _quad_encloses_points(scaled, chess_quad):
        grew = False
        for _ in range(14):
            scaled = _order_quad((center + ((scaled - center) * 1.04)).astype(np.float32))
            if _quad_encloses_points(scaled, chess_quad):
                grew = True
                break
        if not grew:
            # Fallback to original detected outer if we cannot safely enclose.
            info["enclosure_fallback_to_original"] = True
            return outer_quad.astype(np.float32), info
        info["enclosure_grow_applied"] = True

    new_outer_area = _quad_area_px2(scaled)
    if new_outer_area is not None and chess_area > 1e-6:
        info["new_outer_to_chess_ratio"] = float(new_outer_area / chess_area)
    info["applied"] = True
    return scaled, info


def _derive_outer_margins_from_reference(
    *,
    ref_chess_quad: np.ndarray,
    ref_outer_quad: np.ndarray,
    board_size: tuple[int, int],
) -> tuple[float, float, float, float] | None:
    cols, rows = board_size
    if cols <= 0 or rows <= 0:
        return None
    src_board = np.array(
        [
            [0.0, 0.0],
            [float(cols), 0.0],
            [float(cols), float(rows)],
            [0.0, float(rows)],
        ],
        dtype=np.float32,
    )
    chess_img = ref_chess_quad.astype(np.float32).reshape(4, 2)
    outer_img = ref_outer_quad.astype(np.float32).reshape(4, 2)
    h_img_to_board = cv2.getPerspectiveTransform(chess_img, src_board)
    outer_in_board = cv2.perspectiveTransform(outer_img.reshape(-1, 1, 2), h_img_to_board).reshape(4, 2)
    outer_in_board = _order_quad(outer_in_board)

    x_min = float(np.min(outer_in_board[:, 0]))
    x_max = float(np.max(outer_in_board[:, 0]))
    y_min = float(np.min(outer_in_board[:, 1]))
    y_max = float(np.max(outer_in_board[:, 1]))

    left = max(0.0, -x_min)
    right = max(0.0, x_max - float(cols))
    top = max(0.0, -y_min)
    bottom = max(0.0, y_max - float(rows))
    return (left, right, top, bottom)


def _clamp_quad_area_ratio_image(
    *,
    quad: np.ndarray,
    image_w: int,
    image_h: int,
    min_ratio: float,
    max_ratio: float,
    anchor_xy: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, float | bool]]:
    info: dict[str, float | bool] = {
        "applied": False,
        "min_ratio": float(min_ratio),
        "max_ratio": float(max_ratio),
    }
    area = _quad_area_px2(quad)
    if area is None or area <= 1e-6:
        return quad.astype(np.float32), info
    image_area = float(max(1, image_w * image_h))
    cur_ratio = float(area / image_area)
    info["current_ratio"] = cur_ratio

    desired_ratio = cur_ratio
    if cur_ratio < min_ratio:
        desired_ratio = float(min_ratio)
    elif cur_ratio > max_ratio:
        desired_ratio = float(max_ratio)
    info["desired_ratio"] = desired_ratio
    if abs(desired_ratio - cur_ratio) < 1e-8:
        return quad.astype(np.float32), info

    scale = float(np.sqrt(desired_ratio / max(cur_ratio, 1e-9)))
    pts = quad.astype(np.float32).reshape(4, 2)
    if anchor_xy is not None:
        center = anchor_xy.astype(np.float32).reshape(1, 2)
    else:
        center = np.mean(pts, axis=0, keepdims=True)
    scaled = center + ((pts - center) * scale)
    scaled = _order_quad(scaled.astype(np.float32))

    new_area = _quad_area_px2(scaled)
    new_ratio = float((new_area or 0.0) / image_area)
    # Numeric safety: contour-area quantization can leave us fractionally outside bounds.
    if new_ratio < min_ratio and new_ratio > 1e-9:
        nudge = float(np.sqrt((min_ratio * 1.001) / new_ratio))
        scaled = _order_quad((center + ((scaled - center) * nudge)).astype(np.float32))
        new_area = _quad_area_px2(scaled)
        new_ratio = float((new_area or 0.0) / image_area)
        info["post_nudge_scale"] = nudge
    elif new_ratio > max_ratio and new_ratio > 1e-9:
        nudge = float(np.sqrt((max_ratio * 0.999) / new_ratio))
        scaled = _order_quad((center + ((scaled - center) * nudge)).astype(np.float32))
        new_area = _quad_area_px2(scaled)
        new_ratio = float((new_area or 0.0) / image_area)
        info["post_nudge_scale"] = nudge
    info["applied"] = True
    info["scale"] = scale
    info["new_ratio"] = new_ratio
    return scaled, info


def _clamp_quad_bbox_size(
    *,
    quad: np.ndarray,
    target_w_px: float,
    target_h_px: float,
    tolerance: float = 0.05,
    anchor_xy: np.ndarray | None = None,
    must_enclose_quad: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, float | bool]]:
    info: dict[str, float | bool] = {
        "applied": False,
        "target_w_px": float(target_w_px),
        "target_h_px": float(target_h_px),
        "tolerance": float(tolerance),
    }
    pts = quad.astype(np.float32).reshape(4, 2)
    x0 = float(np.min(pts[:, 0]))
    y0 = float(np.min(pts[:, 1]))
    x1 = float(np.max(pts[:, 0]))
    y1 = float(np.max(pts[:, 1]))
    cur_w = max(1e-6, x1 - x0)
    cur_h = max(1e-6, y1 - y0)
    info["current_w_px"] = cur_w
    info["current_h_px"] = cur_h

    min_w = max(1e-6, float(target_w_px) * (1.0 - float(tolerance)))
    max_w = max(1e-6, float(target_w_px) * (1.0 + float(tolerance)))
    min_h = max(1e-6, float(target_h_px) * (1.0 - float(tolerance)))
    max_h = max(1e-6, float(target_h_px) * (1.0 + float(tolerance)))
    # Bias to baseline size itself (not just band edge) to avoid "still too small" cases.
    desired_w = float(target_w_px)
    desired_h = float(target_h_px)
    info["desired_w_px"] = desired_w
    info["desired_h_px"] = desired_h

    if abs(desired_w - cur_w) < 1e-8 and abs(desired_h - cur_h) < 1e-8:
        return quad.astype(np.float32), info

    sx = desired_w / cur_w
    sy = desired_h / cur_h
    if anchor_xy is not None:
        center = anchor_xy.astype(np.float32).reshape(1, 2)
    else:
        center = np.mean(pts, axis=0, keepdims=True)

    scaled = pts.copy()
    scaled[:, 0] = center[0, 0] + ((scaled[:, 0] - center[0, 0]) * sx)
    scaled[:, 1] = center[0, 1] + ((scaled[:, 1] - center[0, 1]) * sy)
    scaled = _order_quad(scaled.astype(np.float32))

    if must_enclose_quad is not None and not _quad_encloses_points(scaled, must_enclose_quad):
        grew = False
        for _ in range(12):
            scaled = _order_quad((center + ((scaled - center) * 1.03)).astype(np.float32))
            if _quad_encloses_points(scaled, must_enclose_quad):
                grew = True
                break
        info["enclosure_grow_applied"] = grew

    nx0 = float(np.min(scaled[:, 0]))
    ny0 = float(np.min(scaled[:, 1]))
    nx1 = float(np.max(scaled[:, 0]))
    ny1 = float(np.max(scaled[:, 1]))
    info["new_w_px"] = max(1e-6, nx1 - nx0)
    info["new_h_px"] = max(1e-6, ny1 - ny0)
    info["applied"] = True
    info["sx"] = float(sx)
    info["sy"] = float(sy)
    return scaled, info


def _load_reference_quad(
    entry: object,
    *,
    image_w: int,
    image_h: int,
) -> np.ndarray | None:
    if not isinstance(entry, dict):
        return None

    corners_norm = entry.get("corners_norm")
    if isinstance(corners_norm, list) and len(corners_norm) == 4:
        norm = np.array(corners_norm, dtype=np.float32).reshape(4, 2)
        px = np.stack(
            [
                norm[:, 0] * float(max(1, image_w)),
                norm[:, 1] * float(max(1, image_h)),
            ],
            axis=1,
        )
        return _order_quad(px)

    corners_px = entry.get("corners_px")
    if isinstance(corners_px, list) and len(corners_px) == 4:
        px = np.array(corners_px, dtype=np.float32).reshape(4, 2)
        return _order_quad(px)

    return None


def _simple_reference_entry(
    payload: dict[str, Any],
    *,
    corners_key: str,
    image_w: int,
    image_h: int,
) -> dict[str, Any] | None:
    corners_px = payload.get(corners_key)
    if not isinstance(corners_px, list) or len(corners_px) != 4:
        return None

    stored_size = payload.get("image_size_px")
    stored_w = image_w
    stored_h = image_h
    if isinstance(stored_size, dict):
        try:
            stored_w = max(1, int(stored_size.get("width", image_w)))
            stored_h = max(1, int(stored_size.get("height", image_h)))
        except Exception:  # noqa: BLE001
            stored_w = image_w
            stored_h = image_h

    try:
        px = np.array(corners_px, dtype=np.float32).reshape(4, 2)
    except Exception:  # noqa: BLE001
        return None

    return {
        "corners_px": corners_px,
        "corners_norm": [
            [float(point[0]) / float(stored_w), float(point[1]) / float(stored_h)]
            for point in px
        ],
    }


def _load_simple_reference_geometry(
    *,
    payload: dict[str, Any],
    image_w: int,
    image_h: int,
) -> tuple[np.ndarray | None, np.ndarray | None, str]:
    outer_entry = _simple_reference_entry(
        payload,
        corners_key="outer_sheet_corners_px",
        image_w=image_w,
        image_h=image_h,
    )
    chess_entry = _simple_reference_entry(
        payload,
        corners_key="chessboard_corners_px",
        image_w=image_w,
        image_h=image_h,
    )
    outer = _load_reference_quad(outer_entry, image_w=image_w, image_h=image_h)
    chess = _load_reference_quad(chess_entry, image_w=image_w, image_h=image_h)
    if chess is None:
        return None, None, "missing_chessboard_quad"
    return chess, outer, "ok"


def _load_reference_geometry(
    *,
    reference_path: Path,
    image_w: int,
    image_h: int,
) -> tuple[np.ndarray | None, np.ndarray | None, str]:
    if not reference_path.exists():
        return None, None, "missing_file"

    try:
        payload = json.loads(reference_path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None, None, "invalid_json"

    median = payload.get("median_geometry_for_latest_size")
    if isinstance(median, dict):
        outer = _load_reference_quad(median.get("outer_sheet"), image_w=image_w, image_h=image_h)
        chess = _load_reference_quad(median.get("chessboard"), image_w=image_w, image_h=image_h)
        if chess is None:
            return None, None, "missing_chessboard_quad"
        return chess, outer, "ok"

    if "outer_sheet_corners_px" in payload or "chessboard_corners_px" in payload:
        return _load_simple_reference_geometry(payload=payload, image_w=image_w, image_h=image_h)

    return None, None, "missing_reference_geometry"


def _load_inner_from_outer_reference(
    *,
    reference_path: Path,
    board_size: tuple[int, int],
) -> tuple[tuple[float, float, float, float] | None, str]:
    if not reference_path.exists():
        return None, "missing_file"

    try:
        payload = json.loads(reference_path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None, "invalid_json"

    raw_board_size = payload.get("board_size")
    if isinstance(raw_board_size, list) and len(raw_board_size) == 2:
        try:
            ref_cols = int(raw_board_size[0])
            ref_rows = int(raw_board_size[1])
        except Exception:  # noqa: BLE001
            return None, "invalid_board_size"
        if (ref_cols, ref_rows) != (int(board_size[0]), int(board_size[1])):
            return None, "board_size_mismatch"

    raw_margins = payload.get("margins_squares")
    if isinstance(raw_margins, list) and len(raw_margins) == 4:
        try:
            left, right, top, bottom = [float(x) for x in raw_margins]
        except Exception:  # noqa: BLE001
            return None, "invalid_margins_squares"

        margins = (
            max(0.0, left),
            max(0.0, right),
            max(0.0, top),
            max(0.0, bottom),
        )
        return margins, "ok"

    outer_entry = _simple_reference_entry(
        payload,
        corners_key="outer_sheet_corners_px",
        image_w=1,
        image_h=1,
    )
    chess_entry = _simple_reference_entry(
        payload,
        corners_key="chessboard_corners_px",
        image_w=1,
        image_h=1,
    )
    outer = _load_reference_quad(outer_entry, image_w=1, image_h=1)
    chess = _load_reference_quad(chess_entry, image_w=1, image_h=1)
    if outer is None or chess is None:
        return None, "missing_margins_squares"

    margins = _derive_outer_margins_from_reference(
        ref_chess_quad=chess,
        ref_outer_quad=outer,
        board_size=board_size,
    )
    if margins is None:
        return None, "could_not_derive_margins"
    return margins, "ok"


def _draw_changed_overlay_on_raw(
    frame_bgr: np.ndarray,
    squares_by_coord: dict[tuple[int, int], dict[str, object]],
    changed: list[dict[str, object]],
) -> np.ndarray:
    vis = frame_bgr.copy()
    for item in changed:
        key = (int(item["x"]), int(item["y"]))
        square = squares_by_coord.get(key)
        if square is None:
            continue
        corners_px = np.array(square["corners_px"], dtype=np.float32).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(vis, [corners_px], True, (0, 0, 255), 2, cv2.LINE_AA)
        center = np.array(square["center_px"], dtype=np.float32).astype(np.int32)
        label = str(item.get("label", item["index"]))
        cv2.putText(
            vis,
            label,
            (int(center[0]) - 8, int(center[1]) + 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
    return vis


def _draw_changed_overlay_on_warp(
    warped_bgr: np.ndarray,
    changed: list[dict[str, object]],
    board_size: tuple[int, int],
) -> np.ndarray:
    cols, rows = board_size
    h, w = warped_bgr.shape[:2]
    square_w = max(1, w // cols)
    square_h = max(1, h // rows)

    vis = warped_bgr.copy()
    for item in changed:
        x = int(item["x"])
        y = int(item["y"])
        x0 = x * square_w
        y0 = y * square_h
        x1 = (x + 1) * square_w - 1
        y1 = (y + 1) * square_h - 1
        cv2.rectangle(vis, (x0, y0), (x1, y1), (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(
            vis,
            str(item.get("label", item["index"])),
            (x0 + 4, y0 + 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            1,
            cv2.LINE_AA,
        )
    return vis


def _draw_live_regions_and_grid(
    *,
    frame_bgr: np.ndarray,
    detection: BoardDetection,
    board_size: tuple[int, int],
    label_mode: str,
    camera_square_orientation: str,
) -> tuple[np.ndarray, np.ndarray]:
    regions_overlay = draw_detection_overlay(frame_bgr, detection)
    if detection.chessboard_corners is None:
        grid_overlay = regions_overlay.copy()
        cv2.putText(
            grid_overlay,
            "NO CHESSBOARD DETECTED",
            (24, 36),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
        return regions_overlay, grid_overlay

    squares = _game_labeled_squares(
        generate_square_geometry(board_corners=detection.chessboard_corners, board_size=board_size),
        camera_square_orientation,
    )
    grid_overlay = draw_square_grid_overlay(regions_overlay.copy(), squares=squares, label_mode=label_mode)
    return regions_overlay, grid_overlay


def _write_analysis_summary_text(
    *,
    out_path: Path,
    changed_squares: list[dict[str, object]],
    resolver_changed_squares: list[dict[str, object]] | None,
    inferred_move: dict[str, object],
) -> None:
    lines: list[str] = []
    lines.append("FlexyBoard Analysis Summary")
    lines.append("")
    lines.append(f"changed_square_count: {len(changed_squares)}")
    lines.append("changed_squares:")
    if changed_squares:
        for item in changed_squares:
            lines.append(
                "  - "
                f"index={item.get('index')} "
                f"coord=({item.get('x')},{item.get('y')}) "
                f"label={item.get('label')} "
                f"raw_camera_label={item.get('raw_camera_label', item.get('label'))} "
                f"pixel_ratio={float(item.get('pixel_ratio', 0.0)):.4f} "
                f"delta={float(item.get('signed_intensity_delta', 0.0)):.3f} "
                f"sources={','.join(str(x) for x in item.get('detection_sources', []))}"
                f"{' contour_rank=' + str(item.get('contour_rank')) if item.get('contour_rank') is not None else ''}"
            )
    else:
        lines.append("  - none")

    if resolver_changed_squares is not None:
        lines.append("")
        lines.append(f"resolver_changed_square_count: {len(resolver_changed_squares)}")
        lines.append("resolver_changed_squares:")
        if resolver_changed_squares:
            for item in resolver_changed_squares:
                lines.append(
                    "  - "
                    f"index={item.get('index')} "
                    f"coord=({item.get('x')},{item.get('y')}) "
                    f"label={item.get('label')} "
                    f"raw_camera_label={item.get('raw_camera_label', item.get('label'))} "
                    f"pixel_ratio={float(item.get('pixel_ratio', 0.0)):.4f} "
                    f"delta={float(item.get('signed_intensity_delta', 0.0)):.3f} "
                    f"sources={','.join(str(x) for x in item.get('detection_sources', []))}"
                    f"{' contour_rank=' + str(item.get('contour_rank')) if item.get('contour_rank') is not None else ''}"
                )
        else:
            lines.append("  - none")

    src = inferred_move.get("source")
    dst = inferred_move.get("destination")
    src_text = _format_inferred_endpoint_text(src)
    dst_text = _format_inferred_endpoint_text(dst)

    lines.append("")
    lines.append("inferred_move:")
    lines.append(f"  - game: {inferred_move.get('game')}")
    lines.append(f"  - source: {src_text}")
    lines.append(f"  - destination: {dst_text}")
    lines.append(f"  - moved_piece_type: {inferred_move.get('moved_piece_type')}")
    lines.append(f"  - capture: {inferred_move.get('capture')}")
    lines.append(f"  - confidence: {float(inferred_move.get('confidence', 0.0)):.4f}")
    lines.append(f"  - timestamp: {inferred_move.get('timestamp')}")
    lines.append(f"  - metadata: {json.dumps(inferred_move.get('metadata', {}), ensure_ascii=True)}")
    lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")


def _format_inferred_endpoint_text(raw: object) -> str:
    if not isinstance(raw, dict):
        return "None"
    if raw.get("location_id") is not None:
        return str(raw.get("location_id"))
    if raw.get("x") is not None and raw.get("y") is not None:
        return f"({raw.get('x')},{raw.get('y')})"
    return json.dumps(raw, ensure_ascii=True)


def _scale_reference_quad(
    raw_quad: object,
    *,
    stored_size: dict[str, object],
    target_w: int,
    target_h: int,
) -> np.ndarray | None:
    if not isinstance(raw_quad, list) or len(raw_quad) != 4:
        return None
    try:
        stored_w = max(1.0, float(stored_size.get("width", target_w)))
        stored_h = max(1.0, float(stored_size.get("height", target_h)))
        pts = np.float32(
            [
                [
                    float(point[0]) / stored_w * float(target_w),
                    float(point[1]) / stored_h * float(target_h),
                ]
                for point in raw_quad
            ]
        )
    except Exception:  # noqa: BLE001
        return None
    return _order_quad(pts)


def _load_parcheesi_reference_outer(
    *,
    reference_path: Path,
    image_w: int,
    image_h: int,
) -> tuple[np.ndarray | None, str]:
    if not reference_path.exists():
        return None, "missing"
    try:
        payload = json.loads(reference_path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        return None, f"invalid:{exc}"
    if not isinstance(payload, dict) or not isinstance(payload.get("parcheesi_layout"), dict):
        return None, "unsupported_payload"
    stored_size = payload.get("image_size_px") if isinstance(payload.get("image_size_px"), dict) else {}
    outer = _scale_reference_quad(
        payload.get("outer_sheet_corners_px"),
        stored_size=stored_size,
        target_w=image_w,
        target_h=image_h,
    )
    if outer is None:
        return None, "invalid_outer"
    return outer, "ok"


def _resolve_parcheesi_location_id(location_id: str) -> str:
    if location_id == "home_center":
        return "homearea_1"
    return location_id


def _analyze_parcheesi(
    *,
    args: argparse.Namespace,
    before_path: Path,
    after_path: Path,
    before_full: np.ndarray,
    after_full: np.ndarray,
) -> dict[str, object]:
    artifact_mode = str(args.artifact_mode).strip().lower()
    full_artifacts = artifact_mode == "full"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir) if args.out_dir else Path("debug_output") / f"parcheesi_analysis_{timestamp}"
    _mkdir(out_dir)
    official_dir = _mkdir(out_dir / "official") if full_artifacts else None

    image_h, image_w = after_full.shape[:2]
    reference_path = Path(args.geometry_reference)
    outer_reference = None
    ref_status = "disabled" if bool(args.disable_geometry_reference) else "missing"
    geometry_reference_used = False
    if not bool(args.disable_geometry_reference):
        outer_reference, ref_status = _load_parcheesi_reference_outer(
            reference_path=reference_path,
            image_w=image_w,
            image_h=image_h,
        )
        geometry_reference_used = outer_reference is not None

    before_outer_debug: dict[str, object] | None = None
    after_outer_debug: dict[str, object] | None = None
    if outer_reference is not None:
        before_outer = outer_reference.copy()
        after_outer = outer_reference.copy()
        locked_outer = outer_reference.copy()
        locked_source = f"reference:{reference_path.name}"
        outer_source_before = "locked_session_geometry"
        outer_source_after = "locked_session_geometry"
    else:
        before_outer, before_outer_debug = detect_outer_field_by_black_tape_contours(before_full)
        after_outer, after_outer_debug = detect_outer_field_by_black_tape_contours(after_full)
        if before_outer is None and after_outer is None:
            raise SystemExit("Failed to detect Parcheesi outer field in both before and after images.")
        mode = str(args.board_lock_source).strip().lower()
        if mode == "after" and after_outer is not None:
            locked_outer = after_outer
            locked_source = "after_outer"
        elif mode == "auto" and before_outer is None and after_outer is not None:
            locked_outer = after_outer
            locked_source = "auto_after_outer"
        else:
            locked_outer = before_outer if before_outer is not None else after_outer
            locked_source = "before_outer" if before_outer is not None else "before_fallback_after_outer"
        outer_source_before = str(before_outer_debug.get("method")) if before_outer_debug is not None else "missing"
        outer_source_after = str(after_outer_debug.get("method")) if after_outer_debug is not None else "missing"

    projected_regions = project_parcheesi_regions(np.asarray(locked_outer, dtype=np.float32))
    before_gray = cv2.cvtColor(before_full, cv2.COLOR_BGR2GRAY)
    after_gray = cv2.cvtColor(after_full, cv2.COLOR_BGR2GRAY)
    abs_diff = cv2.absdiff(before_gray, after_gray)
    _, diff_thresh = cv2.threshold(abs_diff, int(args.diff_threshold), 255, cv2.THRESH_BINARY)

    changed_regions: list[dict[str, object]] = []
    all_regions: list[dict[str, object]] = []
    for region in projected_regions:
        polygon = np.round(np.asarray(region["polygon_px"], dtype=np.float32)).astype(np.int32)
        mask = np.zeros(before_gray.shape, dtype=np.uint8)
        cv2.fillPoly(mask, [polygon], 255)
        active = mask > 0
        pixel_count = int(np.count_nonzero(active))
        if pixel_count <= 0:
            continue
        before_mean = float(np.mean(before_gray[active]))
        after_mean = float(np.mean(after_gray[active]))
        occupancy_delta = (255.0 - after_mean) - (255.0 - before_mean)
        pixel_ratio = float(np.count_nonzero(diff_thresh[active])) / float(pixel_count)
        entry = dict(region)
        entry.update(
            {
                "pixel_ratio": pixel_ratio,
                "signed_intensity_delta": float(after_mean - before_mean),
                "occupancy_delta": occupancy_delta,
                "before_mean": before_mean,
                "after_mean": after_mean,
                "changed": bool(pixel_ratio >= float(args.min_changed_ratio) or abs(occupancy_delta) >= 10.0),
            }
        )
        all_regions.append(entry)
        if entry["changed"]:
            changed_regions.append(entry)

    changed_regions.sort(
        key=lambda item: max(float(item.get("pixel_ratio", 0.0)), abs(float(item.get("occupancy_delta", 0.0))) / 40.0),
        reverse=True,
    )

    source_region = min(changed_regions, key=lambda item: float(item.get("occupancy_delta", 0.0)), default=None)
    destination_region = max(changed_regions, key=lambda item: float(item.get("occupancy_delta", 0.0)), default=None)
    if source_region is not None and destination_region is not None and source_region["location_id"] == destination_region["location_id"]:
        ordered = sorted(changed_regions, key=lambda item: float(item.get("occupancy_delta", 0.0)))
        if len(ordered) >= 2:
            source_region = ordered[0]
            destination_region = ordered[-1]

    inferred_move: dict[str, object]
    if source_region is None or destination_region is None or source_region["location_id"] == destination_region["location_id"]:
        inferred_move = {
            "game": "parcheesi",
            "source": None,
            "destination": None,
            "moved_piece_type": None,
            "capture": None,
            "confidence": 0.0,
            "timestamp": datetime.now().isoformat(),
            "metadata": {
                "reason": "insufficient_changed_regions",
                "changed_region_count": len(changed_regions),
                "significant_changed_count": len(changed_regions),
            },
        }
    else:
        source_location = _resolve_parcheesi_location_id(str(source_region["location_id"]))
        destination_location = _resolve_parcheesi_location_id(str(destination_region["location_id"]))
        confidence = min(
            1.0,
            (
                float(source_region.get("pixel_ratio", 0.0))
                + float(destination_region.get("pixel_ratio", 0.0))
                + abs(float(source_region.get("occupancy_delta", 0.0))) / 40.0
                + abs(float(destination_region.get("occupancy_delta", 0.0))) / 40.0
            )
            / 2.5,
        )
        inferred_move = {
            "game": "parcheesi",
            "source": {
                "location_id": source_location,
                "region_id": int(source_region["region_id"]),
                "region_label": str(source_region["region_label"]),
            },
            "destination": {
                "location_id": destination_location,
                "region_id": int(destination_region["region_id"]),
                "region_label": str(destination_region["region_label"]),
            },
            "moved_piece_type": None,
            "capture": None,
            "confidence": confidence,
            "timestamp": datetime.now().isoformat(),
            "metadata": {
                "changed_region_count": len(changed_regions),
                "significant_changed_count": len(changed_regions),
                "source_region": {
                    "location_id": str(source_region["location_id"]),
                    "region_id": int(source_region["region_id"]),
                    "pixel_ratio": float(source_region["pixel_ratio"]),
                    "occupancy_delta": float(source_region["occupancy_delta"]),
                },
                "destination_region": {
                    "location_id": str(destination_region["location_id"]),
                    "region_id": int(destination_region["region_id"]),
                    "pixel_ratio": float(destination_region["pixel_ratio"]),
                    "occupancy_delta": float(destination_region["occupancy_delta"]),
                },
            },
        }

    outputs: dict[str, str] = {}
    if full_artifacts and official_dir is not None:
        before_overlay = draw_parcheesi_overlay(
            before_full,
            outer_corners_px=np.asarray(locked_outer, dtype=np.float32),
            projected_regions=projected_regions,
            show_labels=True,
            outer_thickness=5,
            region_thickness=2,
        )
        after_overlay = draw_parcheesi_overlay(
            after_full,
            outer_corners_px=np.asarray(locked_outer, dtype=np.float32),
            projected_regions=projected_regions,
            show_labels=True,
            outer_thickness=5,
            region_thickness=2,
        )
        for item in changed_regions:
            poly = np.round(np.asarray(item["polygon_px"], dtype=np.float32)).astype(np.int32)
            cv2.polylines(after_overlay, [poly], True, (0, 0, 255), 3, cv2.LINE_AA)
        outputs = {
            "before_regions_overlay": str(official_dir / "before_regions_overlay.png"),
            "after_regions_overlay": str(official_dir / "after_regions_overlay.png"),
            "diff_threshold": str(official_dir / "diff_threshold.png"),
            "analysis_summary_txt": str(official_dir / "analysis_summary.txt"),
        }
        cv2.imwrite(outputs["before_regions_overlay"], before_overlay)
        cv2.imwrite(outputs["after_regions_overlay"], after_overlay)
        cv2.imwrite(outputs["diff_threshold"], diff_thresh)
        _write_analysis_summary_text(
            out_path=Path(outputs["analysis_summary_txt"]),
            changed_squares=changed_regions,
            resolver_changed_squares=None,
            inferred_move=inferred_move,
        )

    payload: dict[str, object] = {
        "before_image": str(before_path),
        "after_image": str(after_path),
        "analysis_root_dir": str(out_dir),
        "artifact_mode": artifact_mode,
        "official_dir": str(official_dir) if official_dir is not None else None,
        "algorithm_live_dir": None,
        "board_size": None,
        "board_detection_source": "parcheesi_region_projection",
        "locked_geometry_source": locked_source,
        "geometry_reference": {
            "enabled": not bool(args.disable_geometry_reference),
            "path": str(reference_path),
            "status": ref_status,
            "used": geometry_reference_used,
        },
        "fast_locked_geometry": {
            "requested": bool(args.fast_locked_geometry),
            "used": geometry_reference_used and bool(args.fast_locked_geometry),
            "reference_status": ref_status,
        },
        "algorithm_live_green_source": "black_tape_outer_parcheesi",
        "algorithm_live_outer_source_before": outer_source_before,
        "algorithm_live_outer_source_after": outer_source_after,
        "algorithm_live_inner_source_before": "projected_from_parcheesi_template",
        "algorithm_live_inner_source_after": "projected_from_parcheesi_template",
        "algorithm_live_outer_to_inner_calibration": {
            "source": "stored_parcheesi_region_template",
            "count": len(projected_regions),
        },
        "outer_candidate_mode": args.outer_candidate_mode,
        "tape_projection_enabled": False,
        "warp_alignment_requested": args.warp_alignment_mode,
        "warp_alignment_mode": "parcheesi_region_projection",
        "board_corner_relative_drift": (
            _relative_corner_drift(before_outer, after_outer)
            if outer_reference is None and before_outer is not None and after_outer is not None
            else 0.0
        ),
        "board_corner_mean_distance_px": (
            _mean_corner_distance_px(before_outer, after_outer)
            if outer_reference is None and before_outer is not None and after_outer is not None
            else 0.0
        ),
        "shared_after_recommended": False,
        "shared_after_drift_threshold": 0.06,
        "before_chessboard_corners_px": None,
        "after_chessboard_corners_px": None,
        "outer_sheet_corners_px": np.asarray(locked_outer, dtype=np.float32).tolist(),
        "chessboard_corners_px": None,
        "image_size_px": {"width": int(image_w), "height": int(image_h)},
        "geometry_metrics": {
            "image_width_px": int(image_w),
            "image_height_px": int(image_h),
            "image_area_px2": float(image_w * image_h),
            "locked_outer_area_px2": _quad_area_px2(np.asarray(locked_outer, dtype=np.float32)),
        },
        "pre_detection_crop": {
            "enabled": False,
            "source_image_width_px": int(image_w),
            "source_image_height_px": int(image_h),
            "before": None,
            "after": None,
        },
        "squares": [],
        "changed_squares": changed_regions,
        "changed_square_count": len(changed_regions),
        "resolver_changed_squares": None,
        "resolver_changed_square_count": None,
        "contour_changed_squares": [],
        "contour_changed_square_count": 0,
        "changed_regions": changed_regions,
        "changed_region_count": len(changed_regions),
        "significant_changed_count": len(changed_regions),
        "parcheesi_regions": projected_regions,
        "parcheesi_layout": parcheesi_layout_payload(),
        "inferred_move": inferred_move,
        "outputs": outputs,
        "algorithm_live_geometry_debug": {
            "before": _strip_numpy_debug(before_outer_debug),
            "after": _strip_numpy_debug(after_outer_debug),
        },
    }
    return payload


def _as_float(raw: object, default: float = 0.0) -> float:
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


def _select_chess_resolver_changed_squares(
    changed_squares: list[dict[str, object]],
) -> list[dict[str, object]]:
    if len(changed_squares) <= 2:
        return [dict(item) for item in changed_squares]

    indexed: list[tuple[int, dict[str, object], float, float, bool]] = []
    for idx, item in enumerate(changed_squares):
        ratio = _as_float(item.get("pixel_ratio"))
        delta = abs(_as_float(item.get("signed_intensity_delta")))
        detection_sources = item.get("detection_sources")
        has_contour = isinstance(detection_sources, list) and "contour_top2" in detection_sources
        indexed.append((idx, dict(item), ratio, delta, has_contour))

    indexed.sort(key=lambda entry: (entry[2], 1 if entry[4] else 0, entry[3]), reverse=True)
    keep: list[tuple[int, dict[str, object], float, float]] = [entry[:4] for entry in indexed[:2]]
    second_ratio = keep[1][2]
    keep_threshold = max(0.16, second_ratio * 0.45)
    keep_limit = 6

    def _label_rank(item: dict[str, object]) -> str | None:
        label = str(item.get("label", "")).strip().lower()
        if len(label) != 2 or label[0] < "a" or label[0] > "h" or label[1] < "1" or label[1] > "8":
            return None
        return label[1]

    def _label_file_index(item: dict[str, object]) -> int | None:
        label = str(item.get("label", "")).strip().lower()
        if len(label) != 2 or label[0] < "a" or label[0] > "h" or label[1] < "1" or label[1] > "8":
            return None
        return ord(label[0]) - ord("a")

    def _label_rank_index(item: dict[str, object]) -> int | None:
        label = str(item.get("label", "")).strip().lower()
        if len(label) != 2 or label[0] < "a" or label[0] > "h" or label[1] < "1" or label[1] > "8":
            return None
        return ord(label[1]) - ord("1")

    top_rank_0 = _label_rank(keep[0][1])
    top_rank_1 = _label_rank(keep[1][1])
    top_file_0 = _label_file_index(keep[0][1])
    top_file_1 = _label_file_index(keep[1][1])
    top_rank_idx_0 = _label_rank_index(keep[0][1])
    top_rank_idx_1 = _label_rank_index(keep[1][1])
    top_two_same_rank = top_rank_0 is not None and top_rank_0 == top_rank_1
    top_two_same_file = (
        top_file_0 is not None
        and top_file_1 is not None
        and top_file_0 == top_file_1
    )
    top_two_castling_span = (
        top_file_0 is not None
        and top_file_1 is not None
        and abs(top_file_0 - top_file_1) == 2
    )
    allow_castling_extras = top_two_same_rank and top_two_castling_span
    castling_rank = top_rank_0

    corridor_file = top_file_0 if top_two_same_file else None
    corridor_file_min_rank = (
        min(top_rank_idx_0, top_rank_idx_1)
        if corridor_file is not None and top_rank_idx_0 is not None and top_rank_idx_1 is not None
        else None
    )
    corridor_file_max_rank = (
        max(top_rank_idx_0, top_rank_idx_1)
        if corridor_file is not None and top_rank_idx_0 is not None and top_rank_idx_1 is not None
        else None
    )
    corridor_rank = top_rank_0 if top_two_same_rank else None
    corridor_rank_min_file = (
        min(top_file_0, top_file_1)
        if corridor_rank is not None and top_file_0 is not None and top_file_1 is not None
        else None
    )
    corridor_rank_max_file = (
        max(top_file_0, top_file_1)
        if corridor_rank is not None and top_file_0 is not None and top_file_1 is not None
        else None
    )

    for entry in indexed[2:]:
        _orig_idx, item, ratio, delta, has_contour = entry
        structural_match = False
        if allow_castling_extras and _label_rank(item) == castling_rank:
            structural_match = True
        if corridor_file is not None:
            item_file = _label_file_index(item)
            item_rank_idx = _label_rank_index(item)
            if (
                item_file == corridor_file
                and item_rank_idx is not None
                and corridor_file_min_rank is not None
                and corridor_file_max_rank is not None
                and corridor_file_min_rank <= item_rank_idx <= corridor_file_max_rank
            ):
                structural_match = True
        if corridor_rank is not None:
            item_rank = _label_rank(item)
            item_file = _label_file_index(item)
            if (
                item_rank == corridor_rank
                and item_file is not None
                and corridor_rank_min_file is not None
                and corridor_rank_max_file is not None
                and corridor_rank_min_file <= item_file <= corridor_rank_max_file
            ):
                structural_match = True

        quality_match = ratio >= keep_threshold and (
            delta >= 4.0
            or has_contour
            or ratio >= (second_ratio * 0.80)
        )
        if not (structural_match or quality_match):
            continue

        keep.append(entry[:4])
        if len(keep) >= keep_limit:
            break

    keep.sort(key=lambda entry: entry[0])
    return [item for _orig_idx, item, _ratio, _delta in keep]


def _camera_coord_to_game_coord(x: int, y: int, orientation: str) -> tuple[int, int]:
    normalized = str(orientation or "identity").strip().lower()
    if normalized in {"identity", "image_tl_a1_tr_h1_br_h8_bl_a8"}:
        return x, y
    if normalized == "image_tl_a1_tr_a8_br_h8_bl_h1":
        return y, x
    if normalized == "image_tr_a1_br_a8_bl_h8_tl_h1":
        return 7 - y, x
    raise ValueError(f"unsupported camera square orientation: {orientation!r}")


def _game_label_from_camera_coord(x: int, y: int, orientation: str) -> str:
    gx, gy = _camera_coord_to_game_coord(x, y, orientation)
    return BoardCoord(x=gx, y=gy).to_algebraic()


def _game_labeled_squares(squares: list[object], orientation: str) -> list[object]:
    labeled: list[object] = []
    for square in squares:
        x = getattr(square, "x")
        y = getattr(square, "y")
        labeled.append(replace(square, label=_game_label_from_camera_coord(int(x), int(y), orientation)))
    return labeled


def _square_change_payload(change: SquareChange | ContourSquareCandidate, *, index: int, orientation: str) -> dict[str, object]:
    raw_camera_label = BoardCoord(x=change.coord.x, y=change.coord.y).to_algebraic()
    game_label = _game_label_from_camera_coord(change.coord.x, change.coord.y, orientation)
    payload: dict[str, object] = {
        "index": index,
        "x": change.coord.x,
        "y": change.coord.y,
        "label": game_label,
        "raw_camera_label": raw_camera_label,
        "pixel_ratio": change.pixel_ratio,
        "signed_intensity_delta": change.signed_intensity_delta,
    }
    detection_sources = getattr(change, "detection_sources", None)
    if detection_sources:
        payload["detection_sources"] = list(detection_sources)
    contour_area = getattr(change, "contour_area", 0.0)
    contour_rank = getattr(change, "contour_rank", None)
    if contour_area:
        payload["contour_area"] = float(contour_area)
    if contour_rank is not None:
        payload["contour_rank"] = int(contour_rank)
    return payload


def _clamp_ratio(raw: float) -> float:
    value = float(raw)
    if value < 0.0:
        return 0.0
    if value > 0.49:
        return 0.49
    return value


def _crop_frame(
    frame_bgr: np.ndarray,
    *,
    left_ratio: float,
    right_ratio: float,
    top_ratio: float,
    bottom_ratio: float,
) -> tuple[np.ndarray, dict[str, int | float]]:
    h, w = frame_bgr.shape[:2]
    left = _clamp_ratio(left_ratio)
    right = _clamp_ratio(right_ratio)
    top = _clamp_ratio(top_ratio)
    bottom = _clamp_ratio(bottom_ratio)

    x0 = int(round(w * left))
    x1 = w - int(round(w * right))
    y0 = int(round(h * top))
    y1 = h - int(round(h * bottom))

    min_w = max(64, int(0.35 * w))
    min_h = max(64, int(0.35 * h))
    if (x1 - x0) < min_w:
        x0 = 0
        x1 = w
    if (y1 - y0) < min_h:
        y0 = 0
        y1 = h

    cropped = frame_bgr[y0:y1, x0:x1].copy()
    meta: dict[str, int | float] = {
        "x0_px": int(x0),
        "x1_px": int(x1),
        "y0_px": int(y0),
        "y1_px": int(y1),
        "left_ratio": float(left),
        "right_ratio": float(right),
        "top_ratio": float(top),
        "bottom_ratio": float(bottom),
        "width_px": int(x1 - x0),
        "height_px": int(y1 - y0),
    }
    return cropped, meta


def _shift_quad_for_crop(quad: np.ndarray | None, *, crop_x0: int, crop_y0: int) -> np.ndarray | None:
    if quad is None:
        return None
    shifted = quad.astype(np.float32).copy()
    shifted[:, 0] -= float(crop_x0)
    shifted[:, 1] -= float(crop_y0)
    return shifted


def _choose_detection(
    before_detection: BoardDetection,
    after_detection: BoardDetection,
) -> tuple[np.ndarray | None, np.ndarray | None, str]:
    board_corners = after_detection.chessboard_corners
    board_source = "after"
    if board_corners is None:
        board_corners = before_detection.chessboard_corners
        board_source = "before_fallback"

    outer_sheet = after_detection.outer_sheet_corners
    if outer_sheet is None:
        outer_sheet = before_detection.outer_sheet_corners

    return board_corners, outer_sheet, board_source


def _select_locked_geometry(
    *,
    before_detection: BoardDetection,
    after_detection: BoardDetection,
    board_lock_source: str,
) -> tuple[np.ndarray | None, np.ndarray | None, str]:
    def first_not_none(a: np.ndarray | None, b: np.ndarray | None) -> np.ndarray | None:
        return a if a is not None else b

    mode = board_lock_source.strip().lower()
    if mode not in {"before", "after", "auto"}:
        mode = "before"

    if mode == "before":
        chess = first_not_none(before_detection.chessboard_corners, after_detection.chessboard_corners)
        outer = first_not_none(before_detection.outer_sheet_corners, after_detection.outer_sheet_corners)
        source = "before_primary" if before_detection.chessboard_corners is not None else "before_fallback_after"
        return chess, outer, source

    if mode == "after":
        chess = first_not_none(after_detection.chessboard_corners, before_detection.chessboard_corners)
        outer = first_not_none(after_detection.outer_sheet_corners, before_detection.outer_sheet_corners)
        source = "after_primary" if after_detection.chessboard_corners is not None else "after_fallback_before"
        return chess, outer, source

    # auto: prefer before if available (static board between before/after), else after.
    if before_detection.chessboard_corners is not None:
        return (
            before_detection.chessboard_corners,
            first_not_none(before_detection.outer_sheet_corners, after_detection.outer_sheet_corners),
            "auto_before",
        )
    return (
        after_detection.chessboard_corners,
        first_not_none(after_detection.outer_sheet_corners, before_detection.outer_sheet_corners),
        "auto_after",
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Detect board regions, generate 64-square numbering, and compute square-level diff "
            "between before/after images."
        )
    )
    parser.add_argument("--before", required=True, help="Before image path")
    parser.add_argument("--after", required=True, help="After image path")
    parser.add_argument("--game", default="checkers", help="Game label for inferred move metadata/payloads.")
    parser.add_argument(
        "--geometry-reference",
        default=str(ROOT / "configs" / "corners_info.json"),
        help=(
            "Optional path to persisted board geometry reference. Supports the legacy "
            "median-geometry format and the simpler raw-corners_info format. "
            "When available, this is used to lock yellow/green overlays to a stable board layout."
        ),
    )
    parser.add_argument(
        "--disable-geometry-reference",
        action="store_true",
        help="Disable use of persisted board geometry reference and rely only on live detection.",
    )
    parser.add_argument(
        "--inner-from-outer-reference",
        default=str(ROOT / "configs" / "corners_info.json"),
        help=(
            "Optional path to persisted relation between outer sheet (green) and chessboard (yellow). "
            "Supports both the legacy margins_squares file and the raw corners_info format."
        ),
    )
    parser.add_argument(
        "--disable-inner-from-outer-reference",
        action="store_true",
        help="Disable persisted inner-from-outer relation for algorithm_live yellow grid.",
    )
    parser.add_argument("--board-cols", type=int, default=8)
    parser.add_argument("--board-rows", type=int, default=8)
    parser.add_argument("--hsv-lower", default="8,20,50", help="Outer-sheet HSV lower H,S,V")
    parser.add_argument("--hsv-upper", default="35,255,255", help="Outer-sheet HSV upper H,S,V")
    parser.add_argument("--min-outer-area-ratio", type=float, default=0.1)
    parser.add_argument("--max-outer-to-chess-ratio", type=float, default=3.5)
    parser.add_argument(
        "--outer-candidate-mode",
        default="auto",
        choices=("auto", "hsv_only", "dark_only"),
        help="Outer-region candidate source selection.",
    )
    parser.add_argument(
        "--enable-tape-projection",
        action="store_true",
        default=False,
        help="Enable tape-projection outer refinement (disabled by default).",
    )
    parser.add_argument(
        "--disable-tape-projection",
        action="store_true",
        help="Deprecated: explicitly disable tape projection.",
    )
    parser.add_argument(
        "--board-lock-source",
        default="before",
        choices=("before", "after", "auto"),
        help=(
            "Select which frame's board geometry is locked for both before/after warps and overlays. "
            "'before' is recommended for static board setups."
        ),
    )
    parser.add_argument(
        "--fallback-margins",
        default="3.2,3.2,1.4,2.4",
        help="Fallback outer margins in chess-square units: left,right,top,bottom",
    )
    parser.add_argument("--warp-square-px", type=int, default=96)
    parser.add_argument("--blur-kernel", type=int, default=5)
    parser.add_argument("--diff-threshold", type=int, default=28)
    parser.add_argument("--min-changed-ratio", type=float, default=0.08)
    parser.add_argument(
        "--inner-shrink",
        type=float,
        default=0.0,
        help="Optional fractional inward shrink of detected chessboard corners (e.g., 0.01 to 0.03).",
    )
    parser.add_argument(
        "--label-mode",
        default="index",
        choices=("index", "coord", "none"),
        help="How to annotate each detected board square",
    )
    parser.add_argument(
        "--camera-square-orientation",
        default="identity",
        help=(
            "Mapping from image-grid cells to game squares. Supported: identity, "
            "image_tl_a1_tr_a8_br_h8_bl_h1."
        ),
    )
    parser.add_argument(
        "--warp-alignment-mode",
        default="independent",
        choices=("independent", "auto_shared_after"),
        help=(
            "How to align BEFORE/AFTER warps for diff: "
            "'independent' keeps per-image board geometry; "
            "'auto_shared_after' reuses AFTER geometry when corner drift is high."
        ),
    )
    parser.add_argument("--out-dir", default=None, help="Output folder (default: debug_output/board_analysis_<timestamp>)")
    parser.add_argument("--json-out", default=None, help="Optional explicit JSON path")
    parser.add_argument(
        "--artifact-mode",
        default="minimal",
        choices=("minimal", "full"),
        help=(
            "Output mode for runtime artifacts. "
            "'minimal' keeps JSON only and skips overlay/debug image generation. "
            "'full' restores image/text artifacts for debugging."
        ),
    )
    parser.add_argument(
        "--fast-locked-geometry",
        action="store_true",
        help=(
            "Skip live board-region detection and use --geometry-reference as the locked board geometry. "
            "This is intended for live play after a session geometry reference has been built."
        ),
    )
    parser.add_argument(
        "--crop-left-ratio",
        type=float,
        default=0.25,
        help="Fraction to crop from left before detection (default: 0.25).",
    )
    parser.add_argument(
        "--crop-right-ratio",
        type=float,
        default=0.20,
        help="Fraction to crop from right before detection (default: 0.20).",
    )
    parser.add_argument(
        "--crop-top-ratio",
        type=float,
        default=0.0,
        help="Fraction to crop from top before detection (default: 0.0).",
    )
    parser.add_argument(
        "--crop-bottom-ratio",
        type=float,
        default=0.0,
        help="Fraction to crop from bottom before detection (default: 0.0).",
    )
    args = parser.parse_args()

    # Keep algorithm_live output focused on geometry debugging that is useful for iteration.
    before_path = Path(args.before)
    after_path = Path(args.after)

    before_full = cv2.imread(str(before_path), cv2.IMREAD_COLOR)
    after_full = cv2.imread(str(after_path), cv2.IMREAD_COLOR)
    if before_full is None:
        raise SystemExit(f"Could not read before image: {before_path}")
    if after_full is None:
        raise SystemExit(f"Could not read after image: {after_path}")

    if str(args.game).strip().lower() == "parcheesi":
        payload = _analyze_parcheesi(
            args=args,
            before_path=before_path,
            after_path=after_path,
            before_full=before_full,
            after_full=after_full,
        )
        json_path = Path(args.json_out) if args.json_out else Path(str(payload["analysis_root_dir"])) / "analysis.json"
        json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(json.dumps(payload, indent=2))
        return 0

    full_h, full_w = after_full.shape[:2]
    before_raw = before_full
    after_raw = after_full
    crop_meta_before = {
        "enabled": False,
        "x0_px": 0,
        "x1_px": int(before_full.shape[1]),
        "y0_px": 0,
        "y1_px": int(before_full.shape[0]),
        "left_ratio": float(args.crop_left_ratio),
        "right_ratio": float(args.crop_right_ratio),
        "top_ratio": float(args.crop_top_ratio),
        "bottom_ratio": float(args.crop_bottom_ratio),
        "width_px": int(before_full.shape[1]),
        "height_px": int(before_full.shape[0]),
    }
    crop_meta_after = {
        "enabled": False,
        "x0_px": 0,
        "x1_px": int(after_full.shape[1]),
        "y0_px": 0,
        "y1_px": int(after_full.shape[0]),
        "left_ratio": float(args.crop_left_ratio),
        "right_ratio": float(args.crop_right_ratio),
        "top_ratio": float(args.crop_top_ratio),
        "bottom_ratio": float(args.crop_bottom_ratio),
        "width_px": int(after_full.shape[1]),
        "height_px": int(after_full.shape[0]),
    }

    image_h, image_w = after_raw.shape[:2]

    board_size = (int(args.board_cols), int(args.board_rows))
    hsv_lower = _parse_triplet(args.hsv_lower)
    hsv_upper = _parse_triplet(args.hsv_upper)
    fallback_margins = _parse_quad(args.fallback_margins)
    tape_projection_enabled = bool(args.enable_tape_projection) and not bool(args.disable_tape_projection)
    inner_from_outer_reference_path = Path(args.inner_from_outer_reference)
    reference_path = Path(args.geometry_reference)
    live_outer_to_inner_norm, live_outer_to_inner_stats = _load_live_outer_to_inner_calibration(
        reference_path=reference_path,
        disabled=bool(args.disable_inner_from_outer_reference),
    )
    game_lower = str(args.game).strip().lower()
    supports_inner_from_outer = game_lower in {"chess", "checkers"}
    inner_from_outer_reference_status = "unsupported_game"
    inner_from_outer_reference_margins: tuple[float, float, float, float] | None = None
    if supports_inner_from_outer:
        if bool(args.disable_inner_from_outer_reference):
            inner_from_outer_reference_status = "disabled"
        else:
            (
                inner_from_outer_reference_margins,
                inner_from_outer_reference_status,
            ) = _load_inner_from_outer_reference(
                reference_path=inner_from_outer_reference_path,
                board_size=board_size,
            )
    else:
        inner_from_outer_reference_status = "unsupported_game"
    effective_outer_inner_margins = (
        inner_from_outer_reference_margins
        if inner_from_outer_reference_margins is not None
        else fallback_margins
    )

    ref_board: np.ndarray | None = None
    ref_outer: np.ndarray | None = None
    if not bool(args.disable_geometry_reference):
        ref_board, ref_outer, ref_status = _load_reference_geometry(
            reference_path=reference_path,
            image_w=full_w,
            image_h=full_h,
        )
        geometry_reference_used = ref_board is not None
    else:
        ref_status = "disabled"
        geometry_reference_used = False

    fast_locked_geometry_used = bool(args.fast_locked_geometry) and geometry_reference_used

    if fast_locked_geometry_used:
        before_detection = BoardDetection(
            outer_sheet_corners=ref_outer.astype(np.float32).copy() if ref_outer is not None else None,
            chessboard_corners=ref_board.astype(np.float32).copy() if ref_board is not None else None,
        )
        after_detection = BoardDetection(
            outer_sheet_corners=ref_outer.astype(np.float32).copy() if ref_outer is not None else None,
            chessboard_corners=ref_board.astype(np.float32).copy() if ref_board is not None else None,
        )
        before_geometry_debug = {
            "anchor_board": None,
            "anchor_debug": None,
            "outer_debug": {
                "method": "locked_session_geometry",
                "chosen_reason": f"using geometry reference {reference_path.name}",
            },
            "predicted_outer": None,
            "contour_outer": None,
        }
        after_geometry_debug = {
            "anchor_board": None,
            "anchor_debug": None,
            "outer_debug": {
                "method": "locked_session_geometry",
                "chosen_reason": f"using geometry reference {reference_path.name}",
            },
            "predicted_outer": None,
            "contour_outer": None,
        }
    else:
        before_detection, before_geometry_debug = _detect_outer_to_inner_geometry(
            frame_bgr=before_raw,
            chess_norm_in_outer=live_outer_to_inner_norm,
            calibration_stats=live_outer_to_inner_stats,
        )
        after_detection, after_geometry_debug = _detect_outer_to_inner_geometry(
            frame_bgr=after_raw,
            chess_norm_in_outer=live_outer_to_inner_norm,
            calibration_stats=live_outer_to_inner_stats,
        )

    outer_size_lock_target_ratio: float | None = None
    outer_size_lock_details: dict[str, object] = {}
    if fast_locked_geometry_used:
        locked_board_raw = ref_board.astype(np.float32).copy() if ref_board is not None else None
        locked_outer_raw = ref_outer.astype(np.float32).copy() if ref_outer is not None else None
        locked_source = f"reference:{reference_path.name}"
    else:
        locked_board_raw, locked_outer_raw, locked_source = _select_locked_geometry(
            before_detection=before_detection,
            after_detection=after_detection,
            board_lock_source=args.board_lock_source,
        )

    chosen_board, chosen_outer, board_source = _choose_detection(before_detection, after_detection)
    if locked_board_raw is None:
        raise SystemExit("Failed to detect chessboard in both before and after images.")

    # If one geometry is missing, estimate it from the other so the static board model is complete.
    if locked_outer_raw is None and locked_board_raw is not None:
        locked_outer_raw = estimate_outer_sheet_from_chessboard(
            chessboard_corners=locked_board_raw,
            board_size=board_size,
            margins_squares=effective_outer_inner_margins,
        )
    if locked_board_raw is None and locked_outer_raw is not None:
        locked_board_raw = estimate_chessboard_from_outer_sheet(
            outer_sheet_corners=locked_outer_raw,
            board_size=board_size,
            margins_squares=effective_outer_inner_margins,
        )
    if locked_board_raw is None:
        raise SystemExit("Unable to build locked board geometry.")

    before_detected_board = before_detection.chessboard_corners if before_detection.chessboard_corners is not None else chosen_board
    after_detected_board = after_detection.chessboard_corners if after_detection.chessboard_corners is not None else chosen_board
    if before_detected_board is None:
        before_detected_board = locked_board_raw
    if after_detected_board is None:
        after_detected_board = locked_board_raw

    before_board = _shrink_quad(before_detected_board, float(args.inner_shrink))
    after_board = _shrink_quad(after_detected_board, float(args.inner_shrink))
    locked_board = _shrink_quad(locked_board_raw, float(args.inner_shrink))
    locked_outer = locked_outer_raw

    # Default behavior: keep BEFORE and AFTER warps independent.
    # Optional mode: auto-fallback to shared AFTER geometry on high corner drift.
    warp_before_board = locked_board
    warp_after_board = locked_board
    warp_alignment_mode = "locked_static_board"
    board_corner_rel_drift = _relative_corner_drift(before_board, after_board)
    board_corner_mean_distance_px = _mean_corner_distance_px(before_board, after_board)
    max_allowed_drift = 0.06
    shared_after_recommended = board_corner_rel_drift > max_allowed_drift
    if args.warp_alignment_mode == "auto_shared_after":
        warp_alignment_mode = "auto_shared_after_per_frame_disabled_by_lock"
        if shared_after_recommended:
            warp_alignment_mode = "auto_shared_after_not_applicable_locked"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir) if args.out_dir else Path("debug_output") / f"board_analysis_{timestamp}"
    _mkdir(out_dir)
    artifact_mode = str(args.artifact_mode).strip().lower()
    full_artifacts = artifact_mode == "full"
    official_dir = _mkdir(out_dir / "official") if full_artifacts else None
    algorithm_live_dir = _mkdir(out_dir / "algorithm_live") if full_artifacts else None

    live_outer_source_before = "calibrated_outer_to_inner"
    live_outer_source_after = "calibrated_outer_to_inner"
    live_inner_source_before = "derived_from_outer_calibration"
    live_inner_source_after = "derived_from_outer_calibration"

    before_anchor_board = before_geometry_debug.get("anchor_board")
    before_anchor_debug = before_geometry_debug.get("anchor_debug")
    before_live_outer_debug = before_geometry_debug.get("outer_debug")
    before_live_predicted_outer = before_geometry_debug.get("predicted_outer")
    before_live_contour_outer = before_geometry_debug.get("contour_outer")
    before_live_outer = before_detection.outer_sheet_corners
    before_live_board = before_detection.chessboard_corners
    if before_live_outer_debug is not None and before_live_outer is not None and before_live_board is not None:
        live_outer_source_before = str(before_live_outer_debug.get("method"))
    else:
        live_outer_source_before = "anchor_chess_detection_failed"
        live_inner_source_before = "anchor_chess_detection_failed"

    after_anchor_board = after_geometry_debug.get("anchor_board")
    after_anchor_debug = after_geometry_debug.get("anchor_debug")
    after_live_outer_debug = after_geometry_debug.get("outer_debug")
    after_live_predicted_outer = after_geometry_debug.get("predicted_outer")
    after_live_contour_outer = after_geometry_debug.get("contour_outer")
    after_live_outer = after_detection.outer_sheet_corners
    after_live_board = after_detection.chessboard_corners
    if after_live_outer_debug is not None and after_live_outer is not None and after_live_board is not None:
        live_outer_source_after = str(after_live_outer_debug.get("method"))
    else:
        live_outer_source_after = "anchor_chess_detection_failed"
        live_inner_source_after = "anchor_chess_detection_failed"

    before_live_detection = BoardDetection(
        outer_sheet_corners=before_live_outer,
        chessboard_corners=before_live_board,
    )
    after_live_detection = BoardDetection(
        outer_sheet_corners=after_live_outer,
        chessboard_corners=after_live_board,
    )

    before_squares = generate_square_geometry(board_corners=locked_board, board_size=board_size)
    after_squares = generate_square_geometry(board_corners=locked_board, board_size=board_size)
    before_squares_labeled = _game_labeled_squares(before_squares, args.camera_square_orientation) if full_artifacts else []
    after_squares_labeled = _game_labeled_squares(after_squares, args.camera_square_orientation) if full_artifacts else []
    squares_payload = [item.to_dict() for item in after_squares]
    for item in squares_payload:
        raw_camera_label = str(item.get("label"))
        game_label = _game_label_from_camera_coord(int(item["x"]), int(item["y"]), args.camera_square_orientation)
        item["raw_camera_label"] = raw_camera_label
        item["label"] = game_label

    warped_before, _ = warp_to_board(
        frame_bgr=before_raw,
        board_corners=warp_before_board,
        board_size=board_size,
        square_px=int(args.warp_square_px),
    )
    warped_after, _ = warp_to_board(
        frame_bgr=after_raw,
        board_corners=warp_after_board,
        board_size=board_size,
        square_px=int(args.warp_square_px),
    )

    before_pre = preprocess_frame(
        warped_before,
        roi=(0, 0, warped_before.shape[1], warped_before.shape[0]),
        blur_kernel=int(args.blur_kernel),
    )
    after_pre = preprocess_frame(
        warped_after,
        roi=(0, 0, warped_after.shape[1], warped_after.shape[0]),
        blur_kernel=int(args.blur_kernel),
    )

    diff = detect_square_changes(
        before_img=before_pre.enhanced,
        after_img=after_pre.enhanced,
        board_size=board_size,
        diff_threshold=int(args.diff_threshold),
        min_changed_ratio=float(args.min_changed_ratio),
    )

    changed_squares: list[dict[str, object]] = []
    cols, _rows = board_size
    for change in diff.changes:
        index = change.coord.y * cols + change.coord.x
        changed_squares.append(_square_change_payload(change, index=index, orientation=args.camera_square_orientation))

    contour_changed_squares: list[dict[str, object]] = []
    for candidate in diff.contour_candidates:
        index = candidate.coord.y * cols + candidate.coord.x
        contour_changed_squares.append(
            _square_change_payload(candidate, index=index, orientation=args.camera_square_orientation)
        )

    resolver_changed_squares: list[dict[str, object]] | None = None
    if str(args.game).lower() == "chess":
        resolver_changed_squares = _select_chess_resolver_changed_squares(changed_squares)

    inferred = infer_move(
        InferenceInputs(
            game=str(args.game),
            board_size=board_size,
            before_img=before_pre.enhanced,
            after_img=after_pre.enhanced,
            changes=diff.changes,
        )
    )
    outputs: dict[str, str] = {}
    if full_artifacts:
        before_overlay = draw_detection_overlay(
            before_raw,
            BoardDetection(outer_sheet_corners=locked_outer, chessboard_corners=locked_board),
        )
        after_overlay = draw_detection_overlay(
            after_raw,
            BoardDetection(outer_sheet_corners=locked_outer, chessboard_corners=locked_board),
        )
        _before_live_regions_overlay, before_live_grid_overlay = _draw_live_regions_and_grid(
            frame_bgr=before_raw,
            detection=before_live_detection,
            board_size=board_size,
            label_mode=args.label_mode,
            camera_square_orientation=args.camera_square_orientation,
        )
        _after_live_regions_overlay, after_live_grid_overlay = _draw_live_regions_and_grid(
            frame_bgr=after_raw,
            detection=after_live_detection,
            board_size=board_size,
            label_mode=args.label_mode,
            camera_square_orientation=args.camera_square_orientation,
        )
        squares_by_coord = {(int(item["x"]), int(item["y"])): item for item in squares_payload}
        before_grid_overlay = draw_square_grid_overlay(before_overlay, squares=before_squares_labeled, label_mode=args.label_mode)
        after_grid_overlay = draw_square_grid_overlay(after_overlay, squares=after_squares_labeled, label_mode=args.label_mode)
        changed_raw_overlay = _draw_changed_overlay_on_raw(after_raw, squares_by_coord=squares_by_coord, changed=changed_squares)
        changed_warp_overlay = _draw_changed_overlay_on_warp(warped_after, changed=changed_squares, board_size=board_size)

        outputs = {
            "before_grid_overlay": str(official_dir / "before_grid_overlay.png"),
            "after_grid_overlay": str(official_dir / "after_grid_overlay.png"),
            "before_grid_overlay_live": str(algorithm_live_dir / "before_grid_overlay_live.png"),
            "after_grid_overlay_live": str(algorithm_live_dir / "after_grid_overlay_live.png"),
            "diff_threshold": str(official_dir / "diff_threshold.png"),
            "changed_raw_overlay": str(official_dir / "changed_raw_overlay.png"),
            "changed_warp_overlay": str(official_dir / "changed_warp_overlay.png"),
            "analysis_summary_txt": str(official_dir / "analysis_summary.txt"),
        }

        cv2.imwrite(outputs["before_grid_overlay"], before_grid_overlay)
        cv2.imwrite(outputs["after_grid_overlay"], after_grid_overlay)
        cv2.imwrite(outputs["before_grid_overlay_live"], before_live_grid_overlay)
        cv2.imwrite(outputs["after_grid_overlay_live"], after_live_grid_overlay)
        cv2.imwrite(outputs["diff_threshold"], diff.threshold_image)
        cv2.imwrite(outputs["changed_raw_overlay"], changed_raw_overlay)
        cv2.imwrite(outputs["changed_warp_overlay"], changed_warp_overlay)
        _write_analysis_summary_text(
            out_path=Path(outputs["analysis_summary_txt"]),
            changed_squares=changed_squares,
            resolver_changed_squares=resolver_changed_squares,
            inferred_move=inferred.to_dict(),
        )

    payload = {
        "before_image": str(before_path),
        "after_image": str(after_path),
        "analysis_root_dir": str(out_dir),
        "artifact_mode": artifact_mode,
        "official_dir": str(official_dir) if official_dir is not None else None,
        "algorithm_live_dir": str(algorithm_live_dir) if algorithm_live_dir is not None else None,
        "board_size": [board_size[0], board_size[1]],
        "board_detection_source": board_source,
        "locked_geometry_source": locked_source,
        "geometry_reference": {
            "enabled": not bool(args.disable_geometry_reference),
            "path": str(reference_path),
            "status": ref_status,
            "used": geometry_reference_used,
        },
        "fast_locked_geometry": {
            "requested": bool(args.fast_locked_geometry),
            "used": bool(fast_locked_geometry_used),
            "reference_status": ref_status,
        },
        "outer_size_lock": {
            "enabled": outer_size_lock_target_ratio is not None,
            "target_outer_to_chess_ratio": outer_size_lock_target_ratio,
            "details": outer_size_lock_details,
        },
        "algorithm_live_green_source": "black_tape_outer_with_games_1_to_14_calibration",
        "algorithm_live_outer_source_before": live_outer_source_before,
        "algorithm_live_outer_source_after": live_outer_source_after,
        "algorithm_live_inner_source_before": live_inner_source_before,
        "algorithm_live_inner_source_after": live_inner_source_after,
        "algorithm_live_outer_to_inner_calibration": _strip_numpy_debug(live_outer_to_inner_stats),
        "inner_from_outer_reference": {
            "enabled": supports_inner_from_outer and not bool(args.disable_inner_from_outer_reference),
            "path": str(inner_from_outer_reference_path),
            "status": inner_from_outer_reference_status,
            "used": inner_from_outer_reference_margins is not None,
            "margins_squares": (
                [
                    float(inner_from_outer_reference_margins[0]),
                    float(inner_from_outer_reference_margins[1]),
                    float(inner_from_outer_reference_margins[2]),
                    float(inner_from_outer_reference_margins[3]),
                ]
                if inner_from_outer_reference_margins is not None
                else None
            ),
        },
        "outer_candidate_mode": args.outer_candidate_mode,
        "tape_projection_enabled": tape_projection_enabled,
        "warp_alignment_requested": args.warp_alignment_mode,
        "warp_alignment_mode": warp_alignment_mode,
        "board_corner_relative_drift": board_corner_rel_drift,
        "board_corner_mean_distance_px": board_corner_mean_distance_px,
        "shared_after_recommended": shared_after_recommended,
        "shared_after_drift_threshold": max_allowed_drift,
        "before_chessboard_corners_px": before_board.tolist(),
        "after_chessboard_corners_px": after_board.tolist(),
        "algorithm_live_before_chessboard_corners_px": (
            before_live_board.tolist() if before_live_board is not None else None
        ),
        "algorithm_live_after_chessboard_corners_px": (
            after_live_board.tolist() if after_live_board is not None else None
        ),
        "algorithm_live_before_detected_chessboard_corners_px": (
            before_anchor_board.tolist()
            if before_anchor_board is not None
            else None
        ),
        "algorithm_live_after_detected_chessboard_corners_px": (
            after_anchor_board.tolist()
            if after_anchor_board is not None
            else None
        ),
        "algorithm_live_before_outer_sheet_corners_px": (
            before_live_outer.tolist() if before_live_outer is not None else None
        ),
        "algorithm_live_after_outer_sheet_corners_px": (
            after_live_outer.tolist() if after_live_outer is not None else None
        ),
        "algorithm_live_before_predicted_outer_sheet_corners_px": (
            before_live_predicted_outer.tolist() if before_live_predicted_outer is not None else None
        ),
        "algorithm_live_after_predicted_outer_sheet_corners_px": (
            after_live_predicted_outer.tolist() if after_live_predicted_outer is not None else None
        ),
        "algorithm_live_before_contour_outer_candidate_px": (
            before_live_contour_outer.tolist() if before_live_contour_outer is not None else None
        ),
        "algorithm_live_after_contour_outer_candidate_px": (
            after_live_contour_outer.tolist() if after_live_contour_outer is not None else None
        ),
        "outer_sheet_corners_px": locked_outer.tolist() if locked_outer is not None else None,
        "chessboard_corners_px": locked_board.tolist() if locked_board is not None else None,
        "geometry_metrics": {
            "image_width_px": int(after_raw.shape[1]),
            "image_height_px": int(after_raw.shape[0]),
            "image_area_px2": float(after_raw.shape[0] * after_raw.shape[1]),
            "locked_board_area_px2": _quad_area_px2(locked_board),
            "locked_outer_area_px2": _quad_area_px2(locked_outer),
            "before_board_area_px2": _quad_area_px2(before_board),
            "after_board_area_px2": _quad_area_px2(after_board),
            "before_outer_area_px2": _quad_area_px2(before_detection.outer_sheet_corners),
            "after_outer_area_px2": _quad_area_px2(after_detection.outer_sheet_corners),
            "locked_board_area_ratio_image": (
                (_quad_area_px2(locked_board) or 0.0) / max(float(after_raw.shape[0] * after_raw.shape[1]), 1.0)
            ),
            "locked_outer_area_ratio_image": (
                (_quad_area_px2(locked_outer) or 0.0) / max(float(after_raw.shape[0] * after_raw.shape[1]), 1.0)
            ),
            "before_board_area_ratio_image": (
                (_quad_area_px2(before_board) or 0.0) / max(float(after_raw.shape[0] * after_raw.shape[1]), 1.0)
            ),
            "after_board_area_ratio_image": (
                (_quad_area_px2(after_board) or 0.0) / max(float(after_raw.shape[0] * after_raw.shape[1]), 1.0)
            ),
            "before_outer_area_ratio_image": (
                (_quad_area_px2(before_detection.outer_sheet_corners) or 0.0)
                / max(float(after_raw.shape[0] * after_raw.shape[1]), 1.0)
            ),
            "after_outer_area_ratio_image": (
                (_quad_area_px2(after_detection.outer_sheet_corners) or 0.0)
                / max(float(after_raw.shape[0] * after_raw.shape[1]), 1.0)
            ),
        },
        "pre_detection_crop": {
            "enabled": False,
            "source_image_width_px": int(full_w),
            "source_image_height_px": int(full_h),
            "before": crop_meta_before,
            "after": crop_meta_after,
        },
        "squares": squares_payload,
        "changed_squares": changed_squares,
        "changed_square_count": len(changed_squares),
        "resolver_changed_squares": resolver_changed_squares,
        "resolver_changed_square_count": len(resolver_changed_squares) if resolver_changed_squares is not None else None,
        "contour_changed_squares": contour_changed_squares,
        "contour_changed_square_count": len(contour_changed_squares),
        "inferred_move": inferred.to_dict(),
        "outputs": outputs,
        "algorithm_live_geometry_debug": {
            "before": _strip_numpy_debug(before_live_outer_debug),
            "after": _strip_numpy_debug(after_live_outer_debug),
            "before_anchor_debug": {
                "method": str(before_anchor_debug.get("method")),
                "score": float(before_anchor_debug.get("score")),
                "used_components": int(before_anchor_debug.get("used_components")),
                "threshold": int(before_anchor_debug.get("threshold")),
                "square_size": float(before_anchor_debug.get("square_size")),
            }
            if before_anchor_debug is not None
            else None,
            "after_anchor_debug": {
                "method": str(after_anchor_debug.get("method")),
                "score": float(after_anchor_debug.get("score")),
                "used_components": int(after_anchor_debug.get("used_components")),
                "threshold": int(after_anchor_debug.get("threshold")),
                "square_size": float(after_anchor_debug.get("square_size")),
            }
            if after_anchor_debug is not None
            else None,
        },
    }

    json_path = Path(args.json_out) if args.json_out else out_dir / "analysis.json"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
