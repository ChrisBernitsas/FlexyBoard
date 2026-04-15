#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def parse_roi(raw: str | None) -> tuple[int, int, int, int] | None:
    if raw is None:
        return None
    parts = [item.strip() for item in raw.split(",")]
    if len(parts) != 4:
        raise ValueError("--roi must be x,y,w,h")
    x, y, w, h = (int(item) for item in parts)
    if w <= 0 or h <= 0:
        raise ValueError("--roi width/height must be positive")
    return x, y, w, h


def detect_inner_corners(gray: np.ndarray, pattern_size: tuple[int, int]) -> tuple[bool, np.ndarray | None]:
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
    found, corners = cv2.findChessboardCorners(gray, pattern_size, flags)
    return found, corners


def main() -> int:
    parser = argparse.ArgumentParser(description="Detect chessboard corners and save an overlay.")
    parser.add_argument("--image", required=True, help="Input image path")
    parser.add_argument("--board-cols", type=int, default=8, help="Number of board columns")
    parser.add_argument("--board-rows", type=int, default=8, help="Number of board rows")
    parser.add_argument("--roi", default=None, help="Optional ROI x,y,w,h in image pixels")
    parser.add_argument("--overlay-out", default="debug_output/chessboard_corners_overlay.png")
    parser.add_argument("--json-out", default="debug_output/chessboard_corners.json")
    args = parser.parse_args()

    image_path = Path(args.image)
    img = cv2.imread(str(image_path))
    if img is None:
        raise SystemExit(f"Could not read image: {image_path}")

    roi = parse_roi(args.roi)
    x0 = 0
    y0 = 0
    roi_img = img
    if roi is not None:
        x0, y0, w, h = roi
        roi_img = img[y0 : y0 + h, x0 : x0 + w]

    gray = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)
    pattern_size = (args.board_cols - 1, args.board_rows - 1)

    found, corners = detect_inner_corners(gray, pattern_size)
    if not found or corners is None:
        print(json.dumps({"found": False, "pattern_size": pattern_size}, indent=2))
        return 1

    # If the fallback detector is used, refine sub-pixel corners.
    if corners.dtype != np.float32:
        corners = corners.astype(np.float32)
    if corners.shape[-1] != 2:
        corners = corners.reshape(-1, 1, 2)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

    # Convert inner-corner grid to outer board corners.
    grid = corners_refined.reshape(pattern_size[1], pattern_size[0], 2)
    tl = grid[0, 0]
    tr = grid[0, -1]
    br = grid[-1, -1]
    bl = grid[-1, 0]
    dx = (tr - tl) / max(pattern_size[0] - 1, 1)
    dy = (bl - tl) / max(pattern_size[1] - 1, 1)

    outer_tl = tl - dx - dy
    outer_tr = tr + dx - dy
    outer_br = br + dx + dy
    outer_bl = bl - dx + dy
    outer = np.array([outer_tl, outer_tr, outer_br, outer_bl], dtype=np.float32)

    # Shift back to full-image coordinates if ROI was used.
    offset = np.array([x0, y0], dtype=np.float32)
    corners_full = corners_refined.reshape(-1, 2) + offset
    outer_full = outer + offset

    overlay = img.copy()
    cv2.drawChessboardCorners(overlay, pattern_size, corners_full.reshape(-1, 1, 2), True)
    outer_i = outer_full.astype(int)
    cv2.polylines(overlay, [outer_i.reshape(-1, 1, 2)], isClosed=True, color=(0, 255, 255), thickness=3)
    for idx, pt in enumerate(outer_i):
        cv2.circle(overlay, (int(pt[0]), int(pt[1])), 10, (0, 0, 255), -1)
        cv2.putText(
            overlay,
            f"O{idx}",
            (int(pt[0]) + 8, int(pt[1]) - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

    overlay_out = Path(args.overlay_out)
    overlay_out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(overlay_out), overlay)

    payload = {
        "found": True,
        "board_size": [args.board_cols, args.board_rows],
        "pattern_size": list(pattern_size),
        "outer_corners_px": [[float(x), float(y)] for x, y in outer_full.tolist()],
        "roi": list(roi) if roi is not None else None,
        "image": str(image_path),
        "overlay_out": str(overlay_out),
    }

    json_out = Path(args.json_out)
    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
