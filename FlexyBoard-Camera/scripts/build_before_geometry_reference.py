#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import json
from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import Any

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from flexyboard_camera.utils.config import load_config
from flexyboard_camera.vision.board_detector import detect_board_regions


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


def _aggregate_norm_quads(items: list[np.ndarray]) -> np.ndarray:
    # items: list of [4,2] normalized corners, same point ordering (TL/TR/BR/BL)
    stacked = np.stack(items, axis=0)  # [N,4,2]
    return np.median(stacked, axis=0)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a persistent geometry reference from BEFORE captures "
            "(outer sheet and chessboard location/size as px and normalized values)."
        )
    )
    parser.add_argument(
        "--config",
        default=str(ROOT / "configs" / "default.yaml"),
        help="Path to YAML config for detector thresholds.",
    )
    parser.add_argument(
        "--glob",
        dest="image_glob",
        default="debug_output/turn_run_*/before_latest.png",
        help="Glob pattern for BEFORE images (relative to repo root).",
    )
    parser.add_argument(
        "--out",
        default=str(ROOT / "configs" / "before_geometry_reference.json"),
        help="Output JSON file path.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    config = load_config(args.config)

    image_paths = [Path(p) for p in sorted(glob.glob(str(ROOT / args.image_glob)))]
    if not image_paths:
        raise SystemExit(f"No images found for glob: {args.image_glob}")

    per_image: list[dict[str, Any]] = []
    outer_norm_list: list[np.ndarray] = []
    chess_norm_list: list[np.ndarray] = []

    for path in image_paths:
        frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if frame is None:
            continue
        h, w = frame.shape[:2]

        detection = detect_board_regions(
            frame_bgr=frame,
            board_size=config.vision.board_size,
            outer_sheet_hsv_lower=config.vision.outer_sheet_hsv_lower,
            outer_sheet_hsv_upper=config.vision.outer_sheet_hsv_upper,
            min_outer_area_ratio=config.vision.outer_sheet_min_area_ratio,
            max_outer_area_to_chessboard_ratio=config.vision.outer_sheet_max_area_to_chessboard_ratio,
            fallback_outer_margins_squares=config.vision.fallback_outer_margins_squares,
        )
        if detection.outer_sheet_corners is None or detection.chessboard_corners is None:
            continue

        outer = detection.outer_sheet_corners.astype(np.float32).reshape(4, 2)
        chess = detection.chessboard_corners.astype(np.float32).reshape(4, 2)
        outer_norm = np.stack([outer[:, 0] / float(w), outer[:, 1] / float(h)], axis=1)
        chess_norm = np.stack([chess[:, 0] / float(w), chess[:, 1] / float(h)], axis=1)

        outer_norm_list.append(outer_norm)
        chess_norm_list.append(chess_norm)

        per_image.append(
            {
                "image_path": str(path.relative_to(ROOT)),
                "image_size_px": {"width": int(w), "height": int(h)},
                "outer_sheet": _quad_metrics(outer, w, h),
                "chessboard": _quad_metrics(chess, w, h),
            }
        )

    if not per_image:
        raise SystemExit("No valid detections found in provided BEFORE images.")

    # Aggregate median normalized geometry and project to latest image size as reference px.
    latest_w = per_image[-1]["image_size_px"]["width"]
    latest_h = per_image[-1]["image_size_px"]["height"]
    outer_norm_med = _aggregate_norm_quads(outer_norm_list)
    chess_norm_med = _aggregate_norm_quads(chess_norm_list)
    outer_px_med = np.stack([outer_norm_med[:, 0] * latest_w, outer_norm_med[:, 1] * latest_h], axis=1)
    chess_px_med = np.stack([chess_norm_med[:, 0] * latest_w, chess_norm_med[:, 1] * latest_h], axis=1)

    out_payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_image_glob": args.image_glob,
        "sample_count": len(per_image),
        "latest_image_size_px": {"width": int(latest_w), "height": int(latest_h)},
        "median_geometry_for_latest_size": {
            "outer_sheet": _quad_metrics(outer_px_med.astype(np.float32), latest_w, latest_h),
            "chessboard": _quad_metrics(chess_px_med.astype(np.float32), latest_w, latest_h),
        },
        "per_image": per_image,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out_payload, indent=2), encoding="utf-8")
    print(json.dumps({"status": "ok", "out": str(out_path), "sample_count": len(per_image)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
