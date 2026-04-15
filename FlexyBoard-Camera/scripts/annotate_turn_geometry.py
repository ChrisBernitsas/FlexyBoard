#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
import time
from pathlib import Path
import sys
from typing import Any

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from flexyboard_camera.vision.board_detector import (  # noqa: E402
    BoardDetection,
    draw_detection_overlay,
    draw_square_grid_overlay,
    generate_square_geometry,
)


def _parse_triplet(raw: str) -> tuple[int, int, int]:
    parts = [int(item.strip()) for item in raw.split(",")]
    if len(parts) != 3:
        raise ValueError("Expected 3 comma-separated integers")
    return parts[0], parts[1], parts[2]


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


def _quad_metrics(quad: np.ndarray, img_w: int, img_h: int) -> dict[str, Any]:
    corners = quad.astype(np.float32).reshape(4, 2)
    x0 = float(np.min(corners[:, 0]))
    y0 = float(np.min(corners[:, 1]))
    x1 = float(np.max(corners[:, 0]))
    y1 = float(np.max(corners[:, 1]))
    bbox = (x0, y0, x1 - x0, y1 - y0)
    area_px2 = abs(float(cv2.contourArea(corners.reshape(-1, 1, 2))))
    image_area = float(max(1, img_w * img_h))

    corners_norm = np.stack(
        [
            corners[:, 0] / max(1.0, float(img_w)),
            corners[:, 1] / max(1.0, float(img_h)),
        ],
        axis=1,
    )
    bbox_norm = (
        bbox[0] / max(1.0, float(img_w)),
        bbox[1] / max(1.0, float(img_h)),
        bbox[2] / max(1.0, float(img_w)),
        bbox[3] / max(1.0, float(img_h)),
    )

    return {
        "corners_px": [[float(x), float(y)] for x, y in corners],
        "corners_norm": [[float(x), float(y)] for x, y in corners_norm],
        "bbox_xywh_px": [float(v) for v in bbox],
        "bbox_xywh_norm": [float(v) for v in bbox_norm],
        "area_px2": float(area_px2),
        "area_ratio_image": float(area_px2 / image_area),
    }


def _build_masks(
    frame_bgr: np.ndarray,
    *,
    hsv_lower: tuple[int, int, int],
    hsv_upper: tuple[int, int, int],
    dark_threshold: int,
) -> tuple[np.ndarray, np.ndarray]:
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    hsv_mask_raw = cv2.inRange(
        hsv,
        np.array(hsv_lower, dtype=np.uint8),
        np.array(hsv_upper, dtype=np.uint8),
    )
    kernel = np.ones((5, 5), dtype=np.uint8)
    hsv_mask = cv2.morphologyEx(hsv_mask_raw, cv2.MORPH_CLOSE, kernel, iterations=2)
    hsv_mask = cv2.morphologyEx(hsv_mask, cv2.MORPH_OPEN, kernel, iterations=1)

    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, dark_mask_raw = cv2.threshold(blur, int(dark_threshold), 255, cv2.THRESH_BINARY_INV)
    dark_mask = cv2.morphologyEx(dark_mask_raw, cv2.MORPH_CLOSE, kernel, iterations=2)
    dark_mask = cv2.dilate(dark_mask, np.ones((3, 3), dtype=np.uint8), iterations=1)
    return hsv_mask, dark_mask


def _render_annotation_view(
    *,
    base: np.ndarray,
    outer_points: list[tuple[float, float]],
    chess_points: list[tuple[float, float]],
    active_target: str,
    image_label: str,
    view_mode: str,
    transient_message: str,
) -> np.ndarray:
    vis = base.copy()

    def draw_points(points: list[tuple[float, float]], color: tuple[int, int, int], prefix: str) -> None:
        pts = np.array(points, dtype=np.float32).reshape(-1, 2) if points else np.empty((0, 2), dtype=np.float32)
        if pts.shape[0] >= 2:
            poly = pts.astype(np.int32).reshape(-1, 1, 2)
            closed = pts.shape[0] == 4
            cv2.polylines(vis, [poly], closed, color, 2, cv2.LINE_AA)
        for idx, pt in enumerate(pts.astype(np.int32)):
            cv2.circle(vis, (int(pt[0]), int(pt[1])), 4, color, -1, cv2.LINE_AA)
            cv2.putText(
                vis,
                f"{prefix}{idx}",
                (int(pt[0]) + 6, int(pt[1]) - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                1,
                cv2.LINE_AA,
            )
        if pts.shape[0] == 4:
            ordered = _order_quad(pts)
            for idx, pt in enumerate(ordered.astype(np.int32)):
                cv2.putText(
                    vis,
                    f"{prefix}{idx}",
                    (int(pt[0]) + 8, int(pt[1]) + 16),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    color,
                    1,
                    cv2.LINE_AA,
                )

    draw_points(outer_points, (0, 255, 0), "S")
    draw_points(chess_points, (0, 255, 255), "B")

    status = (
        f"{image_label} | view={view_mode} | active={active_target} | "
        f"outer={len(outer_points)}/4 chess={len(chess_points)}/4"
    )
    cv2.rectangle(vis, (0, 0), (vis.shape[1] - 1, 68), (0, 0, 0), -1)
    cv2.putText(vis, status, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(
        vis,
        "Left click:add  Right click/`u`:undo  `o` outer  `i` inner  `1/2/3` raw/hsv/dark  `s` save  `c` clear  `q` quit",
        (12, 52),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.46,
        (220, 220, 220),
        1,
        cv2.LINE_AA,
    )
    if transient_message:
        cv2.putText(
            vis,
            transient_message,
            (12, 84),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

    return vis


def _annotate_single_image(
    *,
    frame_bgr: np.ndarray,
    image_label: str,
    hsv_lower: tuple[int, int, int],
    hsv_upper: tuple[int, int, int],
    dark_threshold: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    hsv_mask, dark_mask = _build_masks(
        frame_bgr,
        hsv_lower=hsv_lower,
        hsv_upper=hsv_upper,
        dark_threshold=dark_threshold,
    )

    views = {
        "raw": frame_bgr,
        "hsv_mask": cv2.cvtColor(hsv_mask, cv2.COLOR_GRAY2BGR),
        "dark_mask": cv2.cvtColor(dark_mask, cv2.COLOR_GRAY2BGR),
    }

    state: dict[str, Any] = {
        "outer_points": [],
        "chess_points": [],
        "active_target": "outer",
        "view_mode": "raw",
        "done": False,
        "abort": False,
        "transient_message": "",
        "transient_until": 0.0,
    }

    window_name = f"Annotate {image_label}"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    def on_mouse(event: int, x: int, y: int, _flags: int, _userdata: Any) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            key = "outer_points" if state["active_target"] == "outer" else "chess_points"
            points: list[tuple[float, float]] = state[key]
            if len(points) < 4:
                points.append((float(x), float(y)))
                if key == "outer_points" and len(points) == 4 and len(state["chess_points"]) < 4:
                    state["active_target"] = "chess"
        elif event == cv2.EVENT_RBUTTONDOWN:
            key = "outer_points" if state["active_target"] == "outer" else "chess_points"
            points = state[key]
            if points:
                points.pop()
            else:
                other_key = "chess_points" if key == "outer_points" else "outer_points"
                other_points = state[other_key]
                if other_points:
                    other_points.pop()

    cv2.setMouseCallback(window_name, on_mouse)

    while True:
        now = time.time()
        if now > float(state["transient_until"]):
            state["transient_message"] = ""
        active_base = views[state["view_mode"]]
        vis = _render_annotation_view(
            base=active_base,
            outer_points=state["outer_points"],
            chess_points=state["chess_points"],
            active_target=state["active_target"],
            image_label=image_label,
            view_mode=state["view_mode"],
            transient_message=str(state["transient_message"]),
        )
        cv2.imshow(window_name, vis)
        key = cv2.waitKey(20) & 0xFF

        if key == 255:
            continue
        if key in (ord("q"), 27):
            state["abort"] = True
            break
        if key in (ord("1"),):
            state["view_mode"] = "raw"
            continue
        if key in (ord("2"),):
            state["view_mode"] = "hsv_mask"
            continue
        if key in (ord("3"),):
            state["view_mode"] = "dark_mask"
            continue
        if key in (ord("o"),):
            state["active_target"] = "outer"
            continue
        if key in (ord("i"),):
            state["active_target"] = "chess"
            continue
        if key in (ord("c"),):
            state["outer_points"] = []
            state["chess_points"] = []
            state["active_target"] = "outer"
            continue
        if key in (ord("u"),):
            active_key = "outer_points" if state["active_target"] == "outer" else "chess_points"
            if state[active_key]:
                state[active_key].pop()
            continue
        if key in (ord("s"), ord("S"), 10, 13):
            if len(state["outer_points"]) == 4 and len(state["chess_points"]) == 4:
                print(f"[annotate] {image_label}: saved manual corners.")
                state["done"] = True
                break
            state["transient_message"] = (
                "Need 4 OUTER and 4 CHESS points before save. "
                f"Now outer={len(state['outer_points'])}/4 chess={len(state['chess_points'])}/4"
            )
            state["transient_until"] = time.time() + 2.5
            print(
                f"[annotate] {image_label}: save ignored; "
                f"outer={len(state['outer_points'])}/4 chess={len(state['chess_points'])}/4"
            )

    cv2.destroyWindow(window_name)

    if state["abort"] or not state["done"]:
        raise SystemExit("Annotation aborted.")

    outer_quad = _order_quad(np.array(state["outer_points"], dtype=np.float32))
    chess_quad = _order_quad(np.array(state["chess_points"], dtype=np.float32))
    return outer_quad, chess_quad, {"hsv_mask": hsv_mask, "dark_mask": dark_mask}


def _relative_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except Exception:  # noqa: BLE001
        return str(path.resolve())


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Interactively annotate outer (green) and chessboard (yellow) grids "
            "on BEFORE and AFTER images, and save reusable geometry JSON."
        )
    )
    parser.add_argument("--before", required=True, help="Before image path")
    parser.add_argument("--after", required=True, help="After image path")
    parser.add_argument("--board-cols", type=int, default=8)
    parser.add_argument("--board-rows", type=int, default=8)
    parser.add_argument("--label-mode", default="index", choices=("index", "coord", "none"))
    parser.add_argument("--hsv-lower", default="8,20,50", help="HSV lower bound used for hsv_mask view")
    parser.add_argument("--hsv-upper", default="35,255,255", help="HSV upper bound used for hsv_mask view")
    parser.add_argument("--dark-threshold", type=int, default=95, help="Threshold used for dark_mask view")
    parser.add_argument(
        "--out",
        default=None,
        help="Output JSON path (default: debug_output/manual_turn_geometry_<timestamp>.json)",
    )
    parser.add_argument(
        "--artifacts-dir",
        default=None,
        help="Output folder for overlays/masks (default: sibling folder next to --out JSON)",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    before_path = Path(args.before).resolve()
    after_path = Path(args.after).resolve()

    before_raw = cv2.imread(str(before_path), cv2.IMREAD_COLOR)
    after_raw = cv2.imread(str(after_path), cv2.IMREAD_COLOR)
    if before_raw is None:
        raise SystemExit(f"Could not read before image: {before_path}")
    if after_raw is None:
        raise SystemExit(f"Could not read after image: {after_path}")

    hsv_lower = _parse_triplet(args.hsv_lower)
    hsv_upper = _parse_triplet(args.hsv_upper)
    board_size = (int(args.board_cols), int(args.board_rows))

    print("Annotating BEFORE image. Click 4 OUTER points, then 4 CHESSBOARD points.")
    before_outer, before_chess, before_masks = _annotate_single_image(
        frame_bgr=before_raw,
        image_label="BEFORE",
        hsv_lower=hsv_lower,
        hsv_upper=hsv_upper,
        dark_threshold=int(args.dark_threshold),
    )
    print("Annotating AFTER image. Click 4 OUTER points, then 4 CHESSBOARD points.")
    after_outer, after_chess, after_masks = _annotate_single_image(
        frame_bgr=after_raw,
        image_label="AFTER",
        hsv_lower=hsv_lower,
        hsv_upper=hsv_upper,
        dark_threshold=int(args.dark_threshold),
    )

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = Path(args.out) if args.out else (ROOT / "debug_output" / f"manual_turn_geometry_{ts}.json")
    out_path = out_path.resolve()
    artifacts_dir = Path(args.artifacts_dir).resolve() if args.artifacts_dir else out_path.with_suffix("")
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    before_detection = BoardDetection(outer_sheet_corners=before_outer, chessboard_corners=before_chess)
    after_detection = BoardDetection(outer_sheet_corners=after_outer, chessboard_corners=after_chess)

    before_overlay = draw_detection_overlay(before_raw, before_detection)
    after_overlay = draw_detection_overlay(after_raw, after_detection)
    before_squares = generate_square_geometry(before_chess, board_size=board_size)
    after_squares = generate_square_geometry(after_chess, board_size=board_size)
    before_grid_overlay = draw_square_grid_overlay(before_overlay.copy(), before_squares, label_mode=args.label_mode)
    after_grid_overlay = draw_square_grid_overlay(after_overlay.copy(), after_squares, label_mode=args.label_mode)

    before_h, before_w = before_raw.shape[:2]
    after_h, after_w = after_raw.shape[:2]

    before_metrics_outer = _quad_metrics(before_outer, before_w, before_h)
    before_metrics_chess = _quad_metrics(before_chess, before_w, before_h)
    after_metrics_outer = _quad_metrics(after_outer, after_w, after_h)
    after_metrics_chess = _quad_metrics(after_chess, after_w, after_h)

    outer_norm_med = np.median(
        np.stack(
            [
                np.array(before_metrics_outer["corners_norm"], dtype=np.float32),
                np.array(after_metrics_outer["corners_norm"], dtype=np.float32),
            ],
            axis=0,
        ),
        axis=0,
    )
    chess_norm_med = np.median(
        np.stack(
            [
                np.array(before_metrics_chess["corners_norm"], dtype=np.float32),
                np.array(after_metrics_chess["corners_norm"], dtype=np.float32),
            ],
            axis=0,
        ),
        axis=0,
    )
    latest_w = int(after_w)
    latest_h = int(after_h)
    outer_px_med = np.stack([outer_norm_med[:, 0] * latest_w, outer_norm_med[:, 1] * latest_h], axis=1)
    chess_px_med = np.stack([chess_norm_med[:, 0] * latest_w, chess_norm_med[:, 1] * latest_h], axis=1)

    before_overlay_path = artifacts_dir / "before_manual_regions_overlay.png"
    before_grid_path = artifacts_dir / "before_manual_grid_overlay.png"
    before_hsv_path = artifacts_dir / "before_hsv_mask.png"
    before_dark_path = artifacts_dir / "before_dark_mask.png"
    after_overlay_path = artifacts_dir / "after_manual_regions_overlay.png"
    after_grid_path = artifacts_dir / "after_manual_grid_overlay.png"
    after_hsv_path = artifacts_dir / "after_hsv_mask.png"
    after_dark_path = artifacts_dir / "after_dark_mask.png"

    cv2.imwrite(str(before_overlay_path), before_overlay)
    cv2.imwrite(str(before_grid_path), before_grid_overlay)
    cv2.imwrite(str(before_hsv_path), before_masks["hsv_mask"])
    cv2.imwrite(str(before_dark_path), before_masks["dark_mask"])
    cv2.imwrite(str(after_overlay_path), after_overlay)
    cv2.imwrite(str(after_grid_path), after_grid_overlay)
    cv2.imwrite(str(after_hsv_path), after_masks["hsv_mask"])
    cv2.imwrite(str(after_dark_path), after_masks["dark_mask"])

    per_image = [
        {
            "image_path": _relative_path(before_path),
            "image_size_px": {"width": int(before_w), "height": int(before_h)},
            "outer_sheet": before_metrics_outer,
            "chessboard": before_metrics_chess,
            "annotation_source": "manual_click",
        },
        {
            "image_path": _relative_path(after_path),
            "image_size_px": {"width": int(after_w), "height": int(after_h)},
            "outer_sheet": after_metrics_outer,
            "chessboard": after_metrics_chess,
            "annotation_source": "manual_click",
        },
    ]

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_image_glob": "manual_annotation_before_after",
        "sample_count": 2,
        "latest_image_size_px": {"width": latest_w, "height": latest_h},
        "median_geometry_for_latest_size": {
            "outer_sheet": _quad_metrics(outer_px_med.astype(np.float32), latest_w, latest_h),
            "chessboard": _quad_metrics(chess_px_med.astype(np.float32), latest_w, latest_h),
        },
        "per_image": per_image,
        "turn_annotation": {
            "before_image": _relative_path(before_path),
            "after_image": _relative_path(after_path),
            "before": {
                "outer_sheet": before_metrics_outer,
                "chessboard": before_metrics_chess,
            },
            "after": {
                "outer_sheet": after_metrics_outer,
                "chessboard": after_metrics_chess,
            },
            "annotation_settings": {
                "hsv_lower": list(hsv_lower),
                "hsv_upper": list(hsv_upper),
                "dark_threshold": int(args.dark_threshold),
                "board_size": [board_size[0], board_size[1]],
                "label_mode": args.label_mode,
            },
            "artifacts": {
                "before_manual_regions_overlay": _relative_path(before_overlay_path),
                "before_manual_grid_overlay": _relative_path(before_grid_path),
                "before_hsv_mask": _relative_path(before_hsv_path),
                "before_dark_mask": _relative_path(before_dark_path),
                "after_manual_regions_overlay": _relative_path(after_overlay_path),
                "after_manual_grid_overlay": _relative_path(after_grid_path),
                "after_hsv_mask": _relative_path(after_hsv_path),
                "after_dark_mask": _relative_path(after_dark_path),
            },
        },
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(
        json.dumps(
            {
                "status": "ok",
                "out_json": _relative_path(out_path),
                "artifacts_dir": _relative_path(artifacts_dir),
                "use_with_analyzer": (
                    f"python scripts/analyze_board_and_diff.py --before { _relative_path(before_path) } "
                    f"--after { _relative_path(after_path) } "
                    f"--geometry-reference { _relative_path(out_path) }"
                ),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
