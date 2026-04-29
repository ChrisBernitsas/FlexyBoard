#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import json
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CALIBRATION_SUMMARY_PATH = ROOT / "configs" / "games_saved_geometry_summary.json"
DEFAULT_GAME_DEBUG_ROOT = ROOT / "debug_output" / "Games"
DEFAULT_CALIBRATION_GAME_RANGE = tuple(range(1, 15))

DARK_THRESH_FOR_TAPE = 100
MORPH_KERNEL_SIZES = (15, 21, 31)

OUTER_NORM = np.float32([
    [0.0, 0.0],
    [1.0, 0.0],
    [1.0, 1.0],
    [0.0, 1.0],
])

DEFAULT_CHESS_NORM_IN_OUTER = np.float32([
    [0.1193, 0.1058],
    [0.8760, 0.1068],
    [0.8708, 0.7799],
    [0.1279, 0.7834],
])


def order_points(pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).reshape(-1)
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]
    return np.array([tl, tr, br, bl], dtype=np.float32)


def polygon_area(pts: np.ndarray) -> float:
    pts = np.asarray(pts, dtype=np.float32).reshape(-1, 1, 2)
    return abs(cv2.contourArea(pts))


def as_int_list(quad: np.ndarray) -> list[list[int]]:
    return [[int(round(x)), int(round(y))] for x, y in order_points(quad)]


def quad_mean_corner_error(a: np.ndarray, b: np.ndarray) -> float:
    a = order_points(a)
    b = order_points(b)
    return float(np.mean(np.linalg.norm(a - b, axis=1)))


def scale_quad(quad: np.ndarray, from_size: dict[str, Any] | tuple[int, int], to_size: dict[str, Any] | tuple[int, int]) -> np.ndarray:
    if isinstance(from_size, dict):
        fw, fh = float(from_size["width"]), float(from_size["height"])
    else:
        fw, fh = float(from_size[0]), float(from_size[1])

    if isinstance(to_size, dict):
        tw, th = float(to_size["width"]), float(to_size["height"])
    else:
        tw, th = float(to_size[0]), float(to_size[1])

    q = order_points(quad).copy()
    q[:, 0] *= tw / fw
    q[:, 1] *= th / fh
    return order_points(q)


def grid_lines_from_quad(quad: np.ndarray, divisions: int = 8) -> tuple[list[list[list[float]]], list[list[list[float]]]]:
    tl, tr, br, bl = order_points(quad)
    vertical: list[list[list[float]]] = []
    horizontal: list[list[list[float]]] = []

    for i in range(divisions + 1):
        t = i / divisions
        top = (1.0 - t) * tl + t * tr
        bottom = (1.0 - t) * bl + t * br
        left = (1.0 - t) * tl + t * bl
        right = (1.0 - t) * tr + t * br
        vertical.append([top.tolist(), bottom.tolist()])
        horizontal.append([left.tolist(), right.tolist()])

    return vertical, horizontal


def quad_is_valid_outer_field(quad: np.ndarray, image_shape: tuple[int, int, int]) -> bool:
    h, w = image_shape[:2]
    q = order_points(quad)
    area = polygon_area(q)
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


def quad_score(quad: np.ndarray, image_shape: tuple[int, int, int], method_bonus: float = 0.0) -> float:
    h, w = image_shape[:2]
    q = order_points(quad)
    area = polygon_area(q)
    cx = float(q[:, 0].mean())
    cy = float(q[:, 1].mean())
    center_dist = math.hypot((cx - w / 2) / w, (cy - h / 2) / h)
    return area - center_dist * 100000 + method_bonus


def _coerce_calibration_summary(data: dict[str, Any]) -> dict[str, Any]:
    calibration = dict(data)
    for key in ("mean_chess_norm_in_outer", "std_chess_norm_in_outer", "mean_outer_px", "std_outer_px", "mean_chess_px", "std_chess_px"):
        if calibration.get(key) is not None:
            calibration[key] = np.float32(calibration[key])
    return calibration


def load_calibration_summary(summary_path: str | Path) -> dict[str, Any]:
    data = json.loads(Path(summary_path).read_text(encoding="utf-8"))
    return _coerce_calibration_summary(data)


def load_outer_to_inner_calibration(calib_paths: list[str | Path]) -> dict[str, Any]:
    all_chess_norm: list[np.ndarray] = []
    all_outer_px: list[np.ndarray] = []
    all_chess_px: list[np.ndarray] = []
    source_sizes: list[dict[str, int] | None] = []
    used_files: list[str] = []

    for p in calib_paths:
        with open(p, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        outer = np.float32(data["outer_sheet_corners_px"])
        chess = np.float32(data["chessboard_corners_px"])
        image_size = data.get("image_size_px")
        outer = order_points(outer)
        chess = order_points(chess)
        h_img_to_outer_norm = cv2.getPerspectiveTransform(outer, OUTER_NORM)
        chess_norm = cv2.perspectiveTransform(chess.reshape(-1, 1, 2), h_img_to_outer_norm).reshape(4, 2)
        all_chess_norm.append(chess_norm)
        all_outer_px.append(outer)
        all_chess_px.append(chess)
        source_sizes.append(image_size)
        used_files.append(str(p))

    if not all_chess_norm:
        raise RuntimeError("No valid calibration JSONs loaded.")

    chess_norm_stack = np.stack(all_chess_norm, axis=0)
    outer_px_stack = np.stack(all_outer_px, axis=0)
    chess_px_stack = np.stack(all_chess_px, axis=0)
    source_image_size = source_sizes[0]

    return {
        "files": used_files,
        "count": len(used_files),
        "source_image_size": source_image_size,
        "mean_chess_norm_in_outer": np.mean(chess_norm_stack, axis=0).astype(np.float32),
        "std_chess_norm_in_outer": np.std(chess_norm_stack, axis=0).astype(np.float32),
        "mean_outer_px": np.mean(outer_px_stack, axis=0).astype(np.float32),
        "std_outer_px": np.std(outer_px_stack, axis=0).astype(np.float32),
        "mean_chess_px": np.mean(chess_px_stack, axis=0).astype(np.float32),
        "std_chess_px": np.std(chess_px_stack, axis=0).astype(np.float32),
    }


def build_games_saved_calibration(
    *,
    game_debug_root: Path = DEFAULT_GAME_DEBUG_ROOT,
    game_numbers: tuple[int, ...] = DEFAULT_CALIBRATION_GAME_RANGE,
) -> dict[str, Any]:
    calib_paths: list[Path] = []
    missing_games: list[int] = []
    for game_number in game_numbers:
        path = game_debug_root / f"Game{game_number}" / "corners_info.json"
        if path.exists():
            calib_paths.append(path)
        else:
            missing_games.append(game_number)

    if not calib_paths:
        calibration = {
            "files": [],
            "count": 0,
            "source_image_size": None,
            "mean_chess_norm_in_outer": DEFAULT_CHESS_NORM_IN_OUTER.copy(),
            "std_chess_norm_in_outer": np.zeros((4, 2), dtype=np.float32),
            "mean_outer_px": None,
            "std_outer_px": None,
            "mean_chess_px": None,
            "std_chess_px": None,
            "requested_games": list(game_numbers),
            "used_games": [],
            "missing_games": missing_games,
            "source": "default_embedded_from_uploaded_jsons",
        }
        return calibration

    calibration = load_outer_to_inner_calibration([str(path) for path in calib_paths])
    calibration["requested_games"] = list(game_numbers)
    calibration["used_games"] = [int(path.parent.name.replace("Game", "")) for path in calib_paths]
    calibration["missing_games"] = missing_games
    calibration["source"] = "debug_output_games_1_to_14"
    return calibration


def save_calibration_summary(calibration: dict[str, Any], out_path: str | Path) -> Path:
    payload = dict(calibration)
    for key in ("mean_chess_norm_in_outer", "std_chess_norm_in_outer", "mean_outer_px", "std_outer_px", "mean_chess_px", "std_chess_px"):
        if isinstance(payload.get(key), np.ndarray):
            payload[key] = payload[key].tolist()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out_path


def load_default_games_saved_calibration() -> dict[str, Any]:
    if DEFAULT_CALIBRATION_SUMMARY_PATH.exists():
        return load_calibration_summary(DEFAULT_CALIBRATION_SUMMARY_PATH)
    calibration = build_games_saved_calibration()
    save_calibration_summary(calibration, DEFAULT_CALIBRATION_SUMMARY_PATH)
    return calibration


def load_calibration(calib_paths: list[str | Path]) -> tuple[np.ndarray, dict[str, Any]]:
    calibration = load_outer_to_inner_calibration(calib_paths) if calib_paths else load_default_games_saved_calibration()
    return np.float32(calibration["mean_chess_norm_in_outer"]), calibration


def predict_chess_from_outer(outer_quad: np.ndarray, chess_norm_in_outer: np.ndarray) -> np.ndarray:
    outer_quad = order_points(outer_quad).astype(np.float32)
    chess_norm_in_outer = order_points(chess_norm_in_outer).astype(np.float32)
    h_outer_norm_to_img = cv2.getPerspectiveTransform(OUTER_NORM, outer_quad)
    chess_pred = cv2.perspectiveTransform(chess_norm_in_outer.reshape(-1, 1, 2), h_outer_norm_to_img).reshape(4, 2)
    return order_points(chess_pred)


def detect_outer_field_by_black_tape_contours(image: np.ndarray) -> tuple[np.ndarray | None, dict[str, Any] | None]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = image.shape[:2]
    all_candidates: list[dict[str, Any]] = []

    for ksize in MORPH_KERNEL_SIZES:
        _, mask = cv2.threshold(gray, DARK_THRESH_FOR_TAPE, 255, cv2.THRESH_BINARY_INV)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))
        closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        contours, hierarchy = cv2.findContours(closed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        if hierarchy is None:
            continue
        hierarchy = hierarchy[0]

        for i, contour in enumerate(contours):
            contour_area = cv2.contourArea(contour)
            if contour_area < 0.03 * w * h:
                continue
            perimeter = cv2.arcLength(contour, True)

            approx_direct = cv2.approxPolyDP(contour, 0.01 * perimeter, True).reshape(-1, 2)
            if len(approx_direct) == 4:
                quad = order_points(approx_direct)
                if quad_is_valid_outer_field(quad, image.shape):
                    has_parent = hierarchy[i][3] != -1
                    bonus = 100000 if has_parent else 50000
                    all_candidates.append({
                        "quad": quad,
                        "score": quad_score(quad, image.shape, method_bonus=bonus),
                        "method": f"contour_direct_k{ksize}",
                        "mask": mask,
                        "closed": closed,
                        "raw_contour": contour,
                    })

            approx_fallback = cv2.approxPolyDP(contour, 0.005 * perimeter, True).reshape(-1, 2)
            if len(approx_fallback) >= 4:
                n = len(approx_fallback)
                for start in range(n):
                    four_pts = np.array([
                        approx_fallback[(start + 0) % n],
                        approx_fallback[(start + 1) % n],
                        approx_fallback[(start + 2) % n],
                        approx_fallback[(start + 3) % n],
                    ], dtype=np.float32)
                    quad = order_points(four_pts)
                    if not quad_is_valid_outer_field(quad, image.shape):
                        continue
                    all_candidates.append({
                        "quad": quad,
                        "score": quad_score(quad, image.shape, method_bonus=0),
                        "method": f"contour_4_consecutive_vertices_k{ksize}",
                        "mask": mask,
                        "closed": closed,
                        "raw_contour": contour,
                    })

    if not all_candidates:
        return None, None

    best = max(all_candidates, key=lambda candidate: candidate["score"])
    debug = {
        "method": best["method"],
        "score": float(best["score"]),
        "mask": best["mask"],
        "closed": best["closed"],
        "raw_contour": best["raw_contour"],
        "num_candidates": len(all_candidates),
    }
    return order_points(best["quad"]), debug


def choose_outer_field(image: np.ndarray, calibration: dict[str, Any], max_detected_vs_calib_error: float = 70.0) -> tuple[np.ndarray, dict[str, Any]]:
    h, w = image.shape[:2]
    detected_outer, debug = detect_outer_field_by_black_tape_contours(image)

    scaled_calib_outer: np.ndarray | None = None
    source_size = calibration.get("source_image_size")
    mean_outer_px = calibration.get("mean_outer_px")
    if source_size and mean_outer_px is not None:
        scaled_calib_outer = scale_quad(np.float32(mean_outer_px), source_size, (w, h))

    if detected_outer is None:
        if scaled_calib_outer is None:
            raise RuntimeError("Could not detect outer field and no calibration fallback exists.")
        return scaled_calib_outer, {
            "method": "scaled_mean_calibration_outer_fallback",
            "reason": "black tape contour detection failed",
            "detected_vs_scaled_calib_error_px": None,
        }

    if scaled_calib_outer is not None:
        err = quad_mean_corner_error(detected_outer, scaled_calib_outer)
        if err > max_detected_vs_calib_error:
            return scaled_calib_outer, {
                "method": "scaled_mean_calibration_outer_fallback",
                "reason": f"detected outer was too far from calibrated outer: {err:.1f}px",
                "detected_outer_candidate": as_int_list(detected_outer),
                "scaled_calib_outer": as_int_list(scaled_calib_outer),
                "detected_vs_scaled_calib_error_px": err,
            }
        debug = dict(debug or {})
        debug["detected_vs_scaled_calib_error_px"] = err
        debug["scaled_calib_outer"] = as_int_list(scaled_calib_outer)
        debug["reason"] = "accepted black tape contour; it agrees with calibration"

    return detected_outer, debug or {"method": "black_tape_contour", "reason": "accepted black tape contour"}


def draw_overlay(image: np.ndarray, outer_quad: np.ndarray, chess_quad: np.ndarray, debug: dict[str, Any] | None = None) -> np.ndarray:
    out = image.copy()
    outer_i = order_points(outer_quad).astype(np.int32)
    chess_i = order_points(chess_quad).astype(np.int32)
    green = (0, 255, 0)
    blue = (255, 0, 0)
    white = (255, 255, 255)
    black = (0, 0, 0)

    cv2.polylines(out, [outer_i], isClosed=True, color=green, thickness=7, lineType=cv2.LINE_AA)
    vertical, horizontal = grid_lines_from_quad(chess_quad, divisions=8)
    for p1, p2 in vertical + horizontal:
        p1_i = tuple(np.round(p1).astype(int))
        p2_i = tuple(np.round(p2).astype(int))
        cv2.line(out, p1_i, p2_i, blue, 3, cv2.LINE_AA)
    cv2.polylines(out, [chess_i], isClosed=True, color=blue, thickness=6, lineType=cv2.LINE_AA)

    for pt in outer_i:
        cv2.circle(out, tuple(pt), 9, green, -1, cv2.LINE_AA)
    for pt in chess_i:
        cv2.circle(out, tuple(pt), 8, blue, -1, cv2.LINE_AA)

    label = "FINAL OUTER = GREEN | FINAL INNER GRID = BLUE"
    cv2.putText(out, label, (30, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.0, black, 5, cv2.LINE_AA)
    cv2.putText(out, label, (30, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.0, white, 2, cv2.LINE_AA)

    if debug is not None:
        method = f"outer: {debug.get('method')} | inner: derived_from_outer_green_board_calibration"
        cv2.putText(out, method, (30, 86), cv2.FONT_HERSHEY_SIMPLEX, 0.75, black, 4, cv2.LINE_AA)
        cv2.putText(out, method, (30, 86), cv2.FONT_HERSHEY_SIMPLEX, 0.75, white, 2, cv2.LINE_AA)

    return out


draw_detection = draw_overlay


def estimate_chessboard_grid(_image: np.ndarray) -> tuple[None, None]:
    return None, None


def process_image(image_path: str | Path, output_dir: Path, calibration: dict[str, Any]) -> dict[str, Any]:
    image_path = Path(image_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    outer_quad, outer_debug = choose_outer_field(image, calibration)
    chess_quad = predict_chess_from_outer(outer_quad, np.float32(calibration["mean_chess_norm_in_outer"]))
    vertical, horizontal = grid_lines_from_quad(chess_quad, divisions=8)
    overlay = draw_overlay(image, outer_quad, chess_quad, outer_debug)

    stem = image_path.stem.replace("(", "_").replace(")", "")
    overlay_path = output_dir / f"{stem}_outer_black_inner_from_calib_overlay.png"
    json_path = output_dir / f"{stem}_geometry.json"
    cv2.imwrite(str(overlay_path), overlay)

    result = {
        "image": str(image_path),
        "image_size_px": {"width": int(image.shape[1]), "height": int(image.shape[0])},
        "outer_field_corners": {
            "top_left": as_int_list(outer_quad)[0],
            "top_right": as_int_list(outer_quad)[1],
            "bottom_right": as_int_list(outer_quad)[2],
            "bottom_left": as_int_list(outer_quad)[3],
        },
        "chessboard_corners_from_outer_calibration": {
            "top_left": as_int_list(chess_quad)[0],
            "top_right": as_int_list(chess_quad)[1],
            "bottom_right": as_int_list(chess_quad)[2],
            "bottom_left": as_int_list(chess_quad)[3],
        },
        "vertical_grid_lines": vertical,
        "horizontal_grid_lines": horizontal,
        "outer_method": outer_debug.get("method"),
        "outer_debug": {k: v for k, v in outer_debug.items() if k not in ("mask", "closed", "raw_contour")},
        "inner_method": "derived_from_outer_using_green_board_calibration",
        "calibration": {
            "count": calibration["count"],
            "files": calibration["files"],
            "source_image_size": calibration["source_image_size"],
            "mean_chess_norm_in_outer": np.float32(calibration["mean_chess_norm_in_outer"]).tolist(),
            "std_chess_norm_in_outer": np.float32(calibration["std_chess_norm_in_outer"]).tolist(),
            "mean_outer_px": None if calibration.get("mean_outer_px") is None else np.float32(calibration["mean_outer_px"]).tolist(),
            "std_outer_px": None if calibration.get("std_outer_px") is None else np.float32(calibration["std_outer_px"]).tolist(),
            "mean_chess_px": None if calibration.get("mean_chess_px") is None else np.float32(calibration["mean_chess_px"]).tolist(),
            "std_chess_px": None if calibration.get("std_chess_px") is None else np.float32(calibration["std_chess_px"]).tolist(),
            "requested_games": calibration.get("requested_games"),
            "used_games": calibration.get("used_games"),
            "missing_games": calibration.get("missing_games"),
            "source": calibration.get("source"),
        },
        "overlay": str(overlay_path),
    }
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Detect black-tape outer field, then derive green-board inner grid from calibration JSONs.")
    parser.add_argument("--image", action="append", required=True, help="Input image path. Can be repeated.")
    parser.add_argument("--calib", action="append", help="Calibration JSON path. Can be repeated.")
    parser.add_argument("--calib-glob", help="Glob for calibration JSONs.")
    parser.add_argument("--out", default="green_board_v5_outputs", help="Output directory.")
    args = parser.parse_args()

    calib_paths: list[str] = []
    if args.calib:
        calib_paths.extend(args.calib)
    if args.calib_glob:
        calib_paths.extend(sorted(glob.glob(args.calib_glob)))

    calibration = load_outer_to_inner_calibration(calib_paths) if calib_paths else load_default_games_saved_calibration()

    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_calibration_summary(calibration, output_dir / "calibration_summary.json")

    all_results: dict[str, Any] = {}
    for image_path in args.image:
        result = process_image(image_path, output_dir, calibration)
        key = Path(image_path).stem.replace("(", "_").replace(")", "")
        all_results[key] = result

    with open(output_dir / "all_geometries.json", "w", encoding="utf-8") as handle:
        json.dump(all_results, handle, indent=2)

    print(json.dumps({
        key: {
            "outer_field_corners": value["outer_field_corners"],
            "chessboard_corners_from_outer_calibration": value["chessboard_corners_from_outer_calibration"],
            "outer_method": value["outer_method"],
            "inner_method": value["inner_method"],
            "overlay": value["overlay"],
        }
        for key, value in all_results.items()
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
