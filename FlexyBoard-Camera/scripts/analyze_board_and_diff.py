#!/usr/bin/env python3
from __future__ import annotations

import argparse
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
    detect_board_regions,
    draw_detection_overlay,
    draw_square_grid_overlay,
    estimate_chessboard_from_outer_sheet,
    estimate_outer_sheet_from_chessboard,
    generate_square_geometry,
    warp_to_board,
)
from flexyboard_camera.vision.diff_detector import detect_square_changes
from flexyboard_camera.vision.move_inference import InferenceInputs, infer_move
from flexyboard_camera.vision.preprocess import preprocess_frame


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


def _save_detector_debug_images(
    *,
    debug_data: dict[str, object],
    prefix: str,
    out_dir: Path,
    outputs: dict[str, str],
) -> dict[str, object]:
    metadata: dict[str, object] = {}
    for key, value in debug_data.items():
        out_key = f"{prefix}_{key}"
        if isinstance(value, np.ndarray):
            path = out_dir / f"{out_key}.png"
            cv2.imwrite(str(path), value)
            outputs[out_key] = str(path)
        else:
            metadata[out_key] = value
    return metadata


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
    if not isinstance(median, dict):
        return None, None, "missing_median_geometry"

    outer = _load_reference_quad(median.get("outer_sheet"), image_w=image_w, image_h=image_h)
    chess = _load_reference_quad(median.get("chessboard"), image_w=image_w, image_h=image_h)
    if chess is None:
        return None, None, "missing_chessboard_quad"

    return chess, outer, "ok"


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
        label = str(item["index"])
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
            str(item["index"]),
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

    squares = generate_square_geometry(board_corners=detection.chessboard_corners, board_size=board_size)
    grid_overlay = draw_square_grid_overlay(regions_overlay.copy(), squares=squares, label_mode=label_mode)
    return regions_overlay, grid_overlay


def _write_analysis_summary_text(
    *,
    out_path: Path,
    changed_squares: list[dict[str, object]],
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
                f"pixel_ratio={float(item.get('pixel_ratio', 0.0)):.4f} "
                f"delta={float(item.get('signed_intensity_delta', 0.0)):.3f}"
            )
    else:
        lines.append("  - none")

    src = inferred_move.get("source")
    dst = inferred_move.get("destination")
    src_text = f"({src.get('x')},{src.get('y')})" if isinstance(src, dict) else "None"
    dst_text = f"({dst.get('x')},{dst.get('y')})" if isinstance(dst, dict) else "None"

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
        default=str(ROOT / "configs" / "before_geometry_reference.json"),
        help=(
            "Optional path to persisted board geometry reference (normalized corners). "
            "When available, this is used to lock yellow/green overlays to a stable board layout."
        ),
    )
    parser.add_argument(
        "--disable-geometry-reference",
        action="store_true",
        help="Disable use of persisted board geometry reference and rely only on live detection.",
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

    before_path = Path(args.before)
    after_path = Path(args.after)

    before_full = cv2.imread(str(before_path), cv2.IMREAD_COLOR)
    after_full = cv2.imread(str(after_path), cv2.IMREAD_COLOR)
    if before_full is None:
        raise SystemExit(f"Could not read before image: {before_path}")
    if after_full is None:
        raise SystemExit(f"Could not read after image: {after_path}")

    full_h, full_w = after_full.shape[:2]
    before_raw, crop_meta_before = _crop_frame(
        before_full,
        left_ratio=float(args.crop_left_ratio),
        right_ratio=float(args.crop_right_ratio),
        top_ratio=float(args.crop_top_ratio),
        bottom_ratio=float(args.crop_bottom_ratio),
    )
    after_raw, crop_meta_after = _crop_frame(
        after_full,
        left_ratio=float(args.crop_left_ratio),
        right_ratio=float(args.crop_right_ratio),
        top_ratio=float(args.crop_top_ratio),
        bottom_ratio=float(args.crop_bottom_ratio),
    )
    # Use the same crop reference for normalized geometry conversion.
    crop_x0 = int(crop_meta_before["x0_px"])
    crop_y0 = int(crop_meta_before["y0_px"])

    image_h, image_w = after_raw.shape[:2]

    board_size = (int(args.board_cols), int(args.board_rows))
    hsv_lower = _parse_triplet(args.hsv_lower)
    hsv_upper = _parse_triplet(args.hsv_upper)
    fallback_margins = _parse_quad(args.fallback_margins)
    tape_projection_enabled = bool(args.enable_tape_projection) and not bool(args.disable_tape_projection)

    before_debug: dict[str, object] = {}
    before_detection = detect_board_regions(
        frame_bgr=before_raw,
        board_size=board_size,
        outer_sheet_hsv_lower=hsv_lower,
        outer_sheet_hsv_upper=hsv_upper,
        min_outer_area_ratio=float(args.min_outer_area_ratio),
        max_outer_area_to_chessboard_ratio=float(args.max_outer_to_chess_ratio),
        fallback_outer_margins_squares=fallback_margins,
        outer_candidate_mode=args.outer_candidate_mode,
        enable_tape_projection=tape_projection_enabled,
        debug=before_debug,
    )
    after_debug: dict[str, object] = {}
    after_detection = detect_board_regions(
        frame_bgr=after_raw,
        board_size=board_size,
        outer_sheet_hsv_lower=hsv_lower,
        outer_sheet_hsv_upper=hsv_upper,
        min_outer_area_ratio=float(args.min_outer_area_ratio),
        max_outer_area_to_chessboard_ratio=float(args.max_outer_to_chess_ratio),
        fallback_outer_margins_squares=fallback_margins,
        outer_candidate_mode=args.outer_candidate_mode,
        enable_tape_projection=tape_projection_enabled,
        debug=after_debug,
    )
    # Keep raw detector-selected outer quads for algorithm_live overlays.
    before_outer_candidate_raw = (
        before_detection.outer_sheet_corners.astype(np.float32).copy()
        if before_detection.outer_sheet_corners is not None
        else None
    )
    after_outer_candidate_raw = (
        after_detection.outer_sheet_corners.astype(np.float32).copy()
        if after_detection.outer_sheet_corners is not None
        else None
    )
    reference_path = Path(args.geometry_reference)
    ref_board: np.ndarray | None = None
    ref_outer: np.ndarray | None = None
    ref_status = "disabled"
    geometry_reference_used = False
    outer_size_lock_target_ratio: float | None = None
    outer_size_lock_target_area_ratio_image: float | None = None
    outer_size_lock_target_bbox_w_px: float | None = None
    outer_size_lock_target_bbox_h_px: float | None = None
    outer_size_lock_margins: tuple[float, float, float, float] | None = None
    outer_size_lock_details: dict[str, object] = {}

    if not bool(args.disable_geometry_reference):
        ref_board, ref_outer, ref_status = _load_reference_geometry(
            reference_path=reference_path,
            image_w=full_w,
            image_h=full_h,
        )
        ref_board = _shift_quad_for_crop(ref_board, crop_x0=crop_x0, crop_y0=crop_y0)
        ref_outer = _shift_quad_for_crop(ref_outer, crop_x0=crop_x0, crop_y0=crop_y0)
        ref_board_area = _quad_area_px2(ref_board)
        ref_outer_area = _quad_area_px2(ref_outer)
        if (
            ref_board_area is not None
            and ref_outer_area is not None
            and ref_board_area > 1e-6
            and ref_outer_area > 1e-6
        ):
            outer_size_lock_target_ratio = float(ref_outer_area / ref_board_area)
            outer_size_lock_target_area_ratio_image = float(
                ref_outer_area / max(float(image_w * image_h), 1.0)
            )
            ref_outer_pts = ref_outer.astype(np.float32).reshape(4, 2)
            outer_size_lock_target_bbox_w_px = float(np.max(ref_outer_pts[:, 0]) - np.min(ref_outer_pts[:, 0]))
            outer_size_lock_target_bbox_h_px = float(np.max(ref_outer_pts[:, 1]) - np.min(ref_outer_pts[:, 1]))
            outer_size_lock_details["target_outer_to_chess_ratio"] = outer_size_lock_target_ratio
            outer_size_lock_details["target_outer_area_ratio_image"] = outer_size_lock_target_area_ratio_image
            outer_size_lock_details["target_outer_bbox_w_px"] = outer_size_lock_target_bbox_w_px
            outer_size_lock_details["target_outer_bbox_h_px"] = outer_size_lock_target_bbox_h_px
            maybe_margins = _derive_outer_margins_from_reference(
                ref_chess_quad=ref_board,
                ref_outer_quad=ref_outer,
                board_size=board_size,
            )
            if maybe_margins is not None:
                outer_size_lock_margins = maybe_margins
                outer_size_lock_details["reference_margins_squares"] = [
                    float(maybe_margins[0]),
                    float(maybe_margins[1]),
                    float(maybe_margins[2]),
                    float(maybe_margins[3]),
                ]
        if ref_board is not None:
            locked_board_raw = ref_board
            locked_outer_raw = ref_outer
            locked_source = f"reference:{reference_path.name}"
            geometry_reference_used = True
        else:
            locked_board_raw, locked_outer_raw, locked_source = _select_locked_geometry(
                before_detection=before_detection,
                after_detection=after_detection,
                board_lock_source=args.board_lock_source,
            )
    else:
        locked_board_raw, locked_outer_raw, locked_source = _select_locked_geometry(
            before_detection=before_detection,
            after_detection=after_detection,
            board_lock_source=args.board_lock_source,
        )

    # Stabilize green-region size using manual/reference baseline ratio while preserving
    # each frame's detected position.
    if outer_size_lock_target_ratio is not None:
        if before_detection.chessboard_corners is not None:
            if outer_size_lock_margins is not None:
                before_detection.outer_sheet_corners = estimate_outer_sheet_from_chessboard(
                    chessboard_corners=before_detection.chessboard_corners,
                    board_size=board_size,
                    margins_squares=outer_size_lock_margins,
                )
                outer_size_lock_details["before"] = {
                    "mode": "reference_margins_from_chessboard",
                    "applied": True,
                }
            elif before_detection.outer_sheet_corners is not None:
                before_locked_outer, before_lock_info = _scale_outer_to_target_ratio(
                    outer_quad=before_detection.outer_sheet_corners,
                    chess_quad=before_detection.chessboard_corners,
                    target_outer_to_chess_ratio=outer_size_lock_target_ratio,
                )
                before_detection.outer_sheet_corners = before_locked_outer
                outer_size_lock_details["before"] = before_lock_info

        if (
            outer_size_lock_target_area_ratio_image is not None
            and before_detection.outer_sheet_corners is not None
        ):
            before_anchor = None
            if before_detection.chessboard_corners is not None:
                before_anchor = np.mean(before_detection.chessboard_corners.astype(np.float32), axis=0)
            before_clamped_outer, before_clamp_info = _clamp_quad_area_ratio_image(
                quad=before_detection.outer_sheet_corners,
                image_w=image_w,
                image_h=image_h,
                min_ratio=outer_size_lock_target_area_ratio_image * 0.95,
                max_ratio=outer_size_lock_target_area_ratio_image * 1.05,
                anchor_xy=before_anchor,
            )
            before_detection.outer_sheet_corners = before_clamped_outer
            before_details = dict(outer_size_lock_details.get("before", {}))
            before_details["area_ratio_clamp"] = before_clamp_info
            outer_size_lock_details["before"] = before_details
        if (
            outer_size_lock_target_bbox_w_px is not None
            and outer_size_lock_target_bbox_h_px is not None
            and before_detection.outer_sheet_corners is not None
        ):
            before_anchor = None
            if before_detection.outer_sheet_corners is not None:
                before_anchor = np.mean(before_detection.outer_sheet_corners.astype(np.float32), axis=0)
            before_bbox_outer, before_bbox_info = _clamp_quad_bbox_size(
                quad=before_detection.outer_sheet_corners,
                target_w_px=outer_size_lock_target_bbox_w_px,
                target_h_px=outer_size_lock_target_bbox_h_px,
                tolerance=0.05,
                anchor_xy=before_anchor,
                must_enclose_quad=before_detection.chessboard_corners,
            )
            before_detection.outer_sheet_corners = before_bbox_outer
            before_details = dict(outer_size_lock_details.get("before", {}))
            before_details["bbox_size_clamp"] = before_bbox_info
            outer_size_lock_details["before"] = before_details

        if after_detection.chessboard_corners is not None:
            if outer_size_lock_margins is not None:
                after_detection.outer_sheet_corners = estimate_outer_sheet_from_chessboard(
                    chessboard_corners=after_detection.chessboard_corners,
                    board_size=board_size,
                    margins_squares=outer_size_lock_margins,
                )
                outer_size_lock_details["after"] = {
                    "mode": "reference_margins_from_chessboard",
                    "applied": True,
                }
            elif after_detection.outer_sheet_corners is not None:
                after_locked_outer, after_lock_info = _scale_outer_to_target_ratio(
                    outer_quad=after_detection.outer_sheet_corners,
                    chess_quad=after_detection.chessboard_corners,
                    target_outer_to_chess_ratio=outer_size_lock_target_ratio,
                )
                after_detection.outer_sheet_corners = after_locked_outer
                outer_size_lock_details["after"] = after_lock_info

        if (
            outer_size_lock_target_area_ratio_image is not None
            and after_detection.outer_sheet_corners is not None
        ):
            after_anchor = None
            if after_detection.chessboard_corners is not None:
                after_anchor = np.mean(after_detection.chessboard_corners.astype(np.float32), axis=0)
            after_clamped_outer, after_clamp_info = _clamp_quad_area_ratio_image(
                quad=after_detection.outer_sheet_corners,
                image_w=image_w,
                image_h=image_h,
                min_ratio=outer_size_lock_target_area_ratio_image * 0.95,
                max_ratio=outer_size_lock_target_area_ratio_image * 1.05,
                anchor_xy=after_anchor,
            )
            after_detection.outer_sheet_corners = after_clamped_outer
            after_details = dict(outer_size_lock_details.get("after", {}))
            after_details["area_ratio_clamp"] = after_clamp_info
            outer_size_lock_details["after"] = after_details
        if (
            outer_size_lock_target_bbox_w_px is not None
            and outer_size_lock_target_bbox_h_px is not None
            and after_detection.outer_sheet_corners is not None
        ):
            after_anchor = None
            if after_detection.outer_sheet_corners is not None:
                after_anchor = np.mean(after_detection.outer_sheet_corners.astype(np.float32), axis=0)
            after_bbox_outer, after_bbox_info = _clamp_quad_bbox_size(
                quad=after_detection.outer_sheet_corners,
                target_w_px=outer_size_lock_target_bbox_w_px,
                target_h_px=outer_size_lock_target_bbox_h_px,
                tolerance=0.05,
                anchor_xy=after_anchor,
                must_enclose_quad=after_detection.chessboard_corners,
            )
            after_detection.outer_sheet_corners = after_bbox_outer
            after_details = dict(outer_size_lock_details.get("after", {}))
            after_details["bbox_size_clamp"] = after_bbox_info
            outer_size_lock_details["after"] = after_details

    chosen_board, chosen_outer, board_source = _choose_detection(before_detection, after_detection)
    if locked_board_raw is None:
        raise SystemExit("Failed to detect chessboard in both before and after images.")

    # If one geometry is missing, estimate it from the other so the static board model is complete.
    if locked_outer_raw is None and locked_board_raw is not None:
        locked_outer_raw = estimate_outer_sheet_from_chessboard(
            chessboard_corners=locked_board_raw,
            board_size=board_size,
            margins_squares=fallback_margins,
        )
    if locked_board_raw is None and locked_outer_raw is not None:
        locked_board_raw = estimate_chessboard_from_outer_sheet(
            outer_sheet_corners=locked_outer_raw,
            board_size=board_size,
            margins_squares=fallback_margins,
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
    official_dir = _mkdir(out_dir / "official")
    algorithm_live_dir = _mkdir(out_dir / "algorithm_live")

    before_overlay = draw_detection_overlay(
        before_raw,
        BoardDetection(outer_sheet_corners=locked_outer, chessboard_corners=locked_board),
    )
    after_overlay = draw_detection_overlay(
        after_raw,
        BoardDetection(outer_sheet_corners=locked_outer, chessboard_corners=locked_board),
    )
    before_outer_only_overlay = draw_detection_overlay(
        before_raw,
        BoardDetection(outer_sheet_corners=locked_outer, chessboard_corners=None),
    )
    after_outer_only_overlay = draw_detection_overlay(
        after_raw,
        BoardDetection(outer_sheet_corners=locked_outer, chessboard_corners=None),
    )
    before_chess_only_overlay = draw_detection_overlay(
        before_raw,
        BoardDetection(outer_sheet_corners=None, chessboard_corners=locked_board),
    )
    after_chess_only_overlay = draw_detection_overlay(
        after_raw,
        BoardDetection(outer_sheet_corners=None, chessboard_corners=locked_board),
    )
    live_outer_source_before = "candidate"
    live_outer_source_after = "candidate"

    before_live_outer = before_outer_candidate_raw
    if before_live_outer is None:
        before_live_outer = before_detection.outer_sheet_corners
        live_outer_source_before = "detected"
    if before_live_outer is None and locked_outer is not None:
        before_live_outer = locked_outer.astype(np.float32).copy()
        live_outer_source_before = "locked_fallback"

    after_live_outer = after_outer_candidate_raw
    if after_live_outer is None:
        after_live_outer = after_detection.outer_sheet_corners
        live_outer_source_after = "detected"
    if after_live_outer is None and locked_outer is not None:
        after_live_outer = locked_outer.astype(np.float32).copy()
        live_outer_source_after = "locked_fallback"

    # Keep algorithm_live green size within reference band while preserving position.
    if (
        outer_size_lock_target_area_ratio_image is not None
        and before_live_outer is not None
    ):
        before_anchor = None
        if before_detection.chessboard_corners is not None:
            before_anchor = np.mean(before_detection.chessboard_corners.astype(np.float32), axis=0)
        before_live_outer, _ = _clamp_quad_area_ratio_image(
            quad=before_live_outer,
            image_w=image_w,
            image_h=image_h,
            min_ratio=outer_size_lock_target_area_ratio_image * 0.95,
            max_ratio=outer_size_lock_target_area_ratio_image * 1.05,
            anchor_xy=before_anchor,
        )
    if (
        outer_size_lock_target_bbox_w_px is not None
        and outer_size_lock_target_bbox_h_px is not None
        and before_live_outer is not None
    ):
        before_anchor = np.mean(before_live_outer.astype(np.float32), axis=0)
        before_live_outer, _ = _clamp_quad_bbox_size(
            quad=before_live_outer,
            target_w_px=outer_size_lock_target_bbox_w_px,
            target_h_px=outer_size_lock_target_bbox_h_px,
            tolerance=0.05,
            anchor_xy=before_anchor,
            must_enclose_quad=before_detection.chessboard_corners,
        )

    if (
        outer_size_lock_target_area_ratio_image is not None
        and after_live_outer is not None
    ):
        after_anchor = None
        if after_detection.chessboard_corners is not None:
            after_anchor = np.mean(after_detection.chessboard_corners.astype(np.float32), axis=0)
        after_live_outer, _ = _clamp_quad_area_ratio_image(
            quad=after_live_outer,
            image_w=image_w,
            image_h=image_h,
            min_ratio=outer_size_lock_target_area_ratio_image * 0.95,
            max_ratio=outer_size_lock_target_area_ratio_image * 1.05,
            anchor_xy=after_anchor,
        )
    if (
        outer_size_lock_target_bbox_w_px is not None
        and outer_size_lock_target_bbox_h_px is not None
        and after_live_outer is not None
    ):
        after_anchor = np.mean(after_live_outer.astype(np.float32), axis=0)
        after_live_outer, _ = _clamp_quad_bbox_size(
            quad=after_live_outer,
            target_w_px=outer_size_lock_target_bbox_w_px,
            target_h_px=outer_size_lock_target_bbox_h_px,
            tolerance=0.05,
            anchor_xy=after_anchor,
            must_enclose_quad=after_detection.chessboard_corners,
        )

    before_live_detection = BoardDetection(
        outer_sheet_corners=before_live_outer,
        chessboard_corners=before_detection.chessboard_corners,
    )
    after_live_detection = BoardDetection(
        outer_sheet_corners=after_live_outer,
        chessboard_corners=after_detection.chessboard_corners,
    )

    before_live_regions_overlay, before_live_grid_overlay = _draw_live_regions_and_grid(
        frame_bgr=before_raw,
        detection=before_live_detection,
        board_size=board_size,
        label_mode=args.label_mode,
    )
    after_live_regions_overlay, after_live_grid_overlay = _draw_live_regions_and_grid(
        frame_bgr=after_raw,
        detection=after_live_detection,
        board_size=board_size,
        label_mode=args.label_mode,
    )

    before_squares = generate_square_geometry(board_corners=locked_board, board_size=board_size)
    after_squares = generate_square_geometry(board_corners=locked_board, board_size=board_size)
    squares_payload = [item.to_dict() for item in after_squares]
    squares_by_coord = {(int(item["x"]), int(item["y"])): item for item in squares_payload}
    before_grid_overlay = draw_square_grid_overlay(before_overlay, squares=before_squares, label_mode=args.label_mode)
    after_grid_overlay = draw_square_grid_overlay(after_overlay, squares=after_squares, label_mode=args.label_mode)

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
        changed_squares.append(
            {
                "index": index,
                "x": change.coord.x,
                "y": change.coord.y,
                "label": BoardCoord(x=change.coord.x, y=change.coord.y).to_algebraic(),
                "pixel_ratio": change.pixel_ratio,
                "signed_intensity_delta": change.signed_intensity_delta,
            }
        )

    inferred = infer_move(
        InferenceInputs(
            game=str(args.game),
            board_size=board_size,
            before_img=before_pre.enhanced,
            after_img=after_pre.enhanced,
            changes=diff.changes,
        )
    )

    changed_raw_overlay = _draw_changed_overlay_on_raw(after_raw, squares_by_coord=squares_by_coord, changed=changed_squares)
    changed_warp_overlay = _draw_changed_overlay_on_warp(warped_after, changed=changed_squares, board_size=board_size)

    outputs = {
        "before_overlay": str(official_dir / "before_regions_overlay.png"),
        "after_overlay": str(official_dir / "after_regions_overlay.png"),
        "before_live_overlay": str(algorithm_live_dir / "before_regions_overlay_live.png"),
        "after_live_overlay": str(algorithm_live_dir / "after_regions_overlay_live.png"),
        "before_outer_only_overlay": str(official_dir / "before_outer_only_overlay.png"),
        "after_outer_only_overlay": str(official_dir / "after_outer_only_overlay.png"),
        "before_chess_only_overlay": str(official_dir / "before_chess_only_overlay.png"),
        "after_chess_only_overlay": str(official_dir / "after_chess_only_overlay.png"),
        "before_grid_overlay": str(official_dir / "before_grid_overlay.png"),
        "after_grid_overlay": str(official_dir / "after_grid_overlay.png"),
        "before_grid_overlay_live": str(algorithm_live_dir / "before_grid_overlay_live.png"),
        "after_grid_overlay_live": str(algorithm_live_dir / "after_grid_overlay_live.png"),
        "before_warped": str(official_dir / "before_warped.png"),
        "after_warped": str(official_dir / "after_warped.png"),
        "before_processed": str(official_dir / "before_processed.png"),
        "after_processed": str(official_dir / "after_processed.png"),
        "diff_gray": str(official_dir / "diff_gray.png"),
        "diff_threshold": str(official_dir / "diff_threshold.png"),
        "changed_raw_overlay": str(official_dir / "changed_raw_overlay.png"),
        "changed_warp_overlay": str(official_dir / "changed_warp_overlay.png"),
        "analysis_summary_txt": str(official_dir / "analysis_summary.txt"),
    }

    cv2.imwrite(outputs["before_overlay"], before_overlay)
    cv2.imwrite(outputs["after_overlay"], after_overlay)
    cv2.imwrite(outputs["before_live_overlay"], before_live_regions_overlay)
    cv2.imwrite(outputs["after_live_overlay"], after_live_regions_overlay)
    cv2.imwrite(outputs["before_outer_only_overlay"], before_outer_only_overlay)
    cv2.imwrite(outputs["after_outer_only_overlay"], after_outer_only_overlay)
    cv2.imwrite(outputs["before_chess_only_overlay"], before_chess_only_overlay)
    cv2.imwrite(outputs["after_chess_only_overlay"], after_chess_only_overlay)
    cv2.imwrite(outputs["before_grid_overlay"], before_grid_overlay)
    cv2.imwrite(outputs["after_grid_overlay"], after_grid_overlay)
    cv2.imwrite(outputs["before_grid_overlay_live"], before_live_grid_overlay)
    cv2.imwrite(outputs["after_grid_overlay_live"], after_live_grid_overlay)
    cv2.imwrite(outputs["before_warped"], warped_before)
    cv2.imwrite(outputs["after_warped"], warped_after)
    cv2.imwrite(outputs["before_processed"], before_pre.enhanced)
    cv2.imwrite(outputs["after_processed"], after_pre.enhanced)
    cv2.imwrite(outputs["diff_gray"], diff.diff_image)
    cv2.imwrite(outputs["diff_threshold"], diff.threshold_image)
    cv2.imwrite(outputs["changed_raw_overlay"], changed_raw_overlay)
    cv2.imwrite(outputs["changed_warp_overlay"], changed_warp_overlay)
    _write_analysis_summary_text(
        out_path=Path(outputs["analysis_summary_txt"]),
        changed_squares=changed_squares,
        inferred_move=inferred.to_dict(),
    )
    before_debug_meta = _save_detector_debug_images(
        debug_data=before_debug,
        prefix="before_detector",
        out_dir=algorithm_live_dir,
        outputs=outputs,
    )
    after_debug_meta = _save_detector_debug_images(
        debug_data=after_debug,
        prefix="after_detector",
        out_dir=algorithm_live_dir,
        outputs=outputs,
    )

    payload = {
        "before_image": str(before_path),
        "after_image": str(after_path),
        "analysis_root_dir": str(out_dir),
        "official_dir": str(official_dir),
        "algorithm_live_dir": str(algorithm_live_dir),
        "board_size": [board_size[0], board_size[1]],
        "board_detection_source": board_source,
        "locked_geometry_source": locked_source,
        "geometry_reference": {
            "enabled": not bool(args.disable_geometry_reference),
            "path": str(reference_path),
            "status": ref_status,
            "used": geometry_reference_used,
        },
        "outer_size_lock": {
            "enabled": outer_size_lock_target_ratio is not None,
            "target_outer_to_chess_ratio": outer_size_lock_target_ratio,
            "details": outer_size_lock_details,
        },
        "algorithm_live_green_source": "raw_detector_candidate_outer",
        "algorithm_live_outer_source_before": live_outer_source_before,
        "algorithm_live_outer_source_after": live_outer_source_after,
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
            "enabled": True,
            "source_image_width_px": int(full_w),
            "source_image_height_px": int(full_h),
            "before": crop_meta_before,
            "after": crop_meta_after,
        },
        "squares": squares_payload,
        "changed_squares": changed_squares,
        "changed_square_count": len(changed_squares),
        "inferred_move": inferred.to_dict(),
        "outputs": outputs,
        "detector_debug": {
            "before": before_debug_meta,
            "after": after_debug_meta,
        },
    }

    json_path = Path(args.json_out) if args.json_out else out_dir / "analysis.json"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
