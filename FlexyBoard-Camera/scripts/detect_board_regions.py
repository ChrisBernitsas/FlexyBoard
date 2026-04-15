#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import cv2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from flexyboard_camera.vision.board_detector import detect_board_regions, draw_detection_overlay


def _parse_int_triplet(raw: str) -> tuple[int, int, int]:
    parts = [int(item.strip()) for item in raw.split(",")]
    if len(parts) != 3:
        raise ValueError("Expected 3 comma-separated integers")
    return parts[0], parts[1], parts[2]


def _parse_float_quad(raw: str) -> tuple[float, float, float, float]:
    parts = [float(item.strip()) for item in raw.split(",")]
    if len(parts) != 4:
        raise ValueError("Expected 4 comma-separated floats")
    return parts[0], parts[1], parts[2], parts[3]


def main() -> int:
    parser = argparse.ArgumentParser(description="Detect outer sheet and inner chessboard regions.")
    parser.add_argument("--image", required=True, help="Input image path")
    parser.add_argument("--board-cols", type=int, default=8)
    parser.add_argument("--board-rows", type=int, default=8)
    parser.add_argument("--hsv-lower", default="8,20,50", help="Outer-sheet HSV lower bound H,S,V")
    parser.add_argument("--hsv-upper", default="35,255,255", help="Outer-sheet HSV upper bound H,S,V")
    parser.add_argument("--min-outer-area-ratio", type=float, default=0.1)
    parser.add_argument("--max-outer-to-chess-ratio", type=float, default=3.5)
    parser.add_argument(
        "--fallback-margins",
        default="3.2,3.2,1.4,2.4",
        help="Fallback outer margins in chess-square units: left,right,top,bottom",
    )
    parser.add_argument("--overlay-out", default="debug_output/board_regions_overlay.png")
    parser.add_argument("--json-out", default="debug_output/board_regions.json")
    args = parser.parse_args()

    image_path = Path(args.image)
    frame = cv2.imread(str(image_path))
    if frame is None:
        raise SystemExit(f"Could not read image: {image_path}")

    detection = detect_board_regions(
        frame_bgr=frame,
        board_size=(args.board_cols, args.board_rows),
        outer_sheet_hsv_lower=_parse_int_triplet(args.hsv_lower),
        outer_sheet_hsv_upper=_parse_int_triplet(args.hsv_upper),
        min_outer_area_ratio=float(args.min_outer_area_ratio),
        max_outer_area_to_chessboard_ratio=float(args.max_outer_to_chess_ratio),
        fallback_outer_margins_squares=_parse_float_quad(args.fallback_margins),
    )

    overlay = draw_detection_overlay(frame, detection)
    overlay_out = Path(args.overlay_out)
    overlay_out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(overlay_out), overlay)

    payload = {
        "image": str(image_path),
        "outer_sheet_corners_px": detection.outer_sheet_corners.tolist() if detection.outer_sheet_corners is not None else None,
        "chessboard_corners_px": detection.chessboard_corners.tolist() if detection.chessboard_corners is not None else None,
        "overlay_out": str(overlay_out),
    }

    json_out = Path(args.json_out)
    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
