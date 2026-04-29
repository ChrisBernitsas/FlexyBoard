#!/usr/bin/env python3
"""Render the current off-board routing/capture layout onto a raw board image."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
SOFTWARE_GUI_DIR = REPO_ROOT / "Software-GUI"
GUI_SITE_PACKAGES = SOFTWARE_GUI_DIR / ".venv" / "lib" / "python3.14" / "site-packages"
sys.path.insert(0, str(SOFTWARE_GUI_DIR))
sys.path.insert(0, str(GUI_SITE_PACKAGES))

import motor_sequence  # noqa: E402


DEFAULT_IMAGE = REPO_ROOT / "FlexyBoard-Camera" / "debug_output" / "Games" / "Game4" / "rolling_reference.png"
DEFAULT_GEOMETRY = REPO_ROOT / "FlexyBoard-Camera" / "debug_output" / "live_session_geometry.json"
DEFAULT_OUT = REPO_ROOT / "FlexyBoard-Camera" / "debug_output" / "offboard_layout_overlay_latest.png"


def _load_geometry(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _percent_to_outer_px(
    outer_quad_px: np.ndarray,
    x_pct: float,
    y_pct: float,
) -> np.ndarray:
    # Planner percent space uses x_pct across the green-sheet width and y_pct
    # down the green-sheet height, matching the historic capture-slot overlays.
    u = np.clip(float(x_pct) / 100.0, 0.0, 1.0)
    v = np.clip(float(y_pct) / 100.0, 0.0, 1.0)
    tl, tr, br, bl = outer_quad_px
    return (
        ((1.0 - v) * (1.0 - u) * tl)
        + ((1.0 - v) * u * tr)
        + (v * u * br)
        + (v * (1.0 - u) * bl)
    )


def _cell_quad(
    outer_quad_px: np.ndarray,
    *,
    x_center: float,
    y_center: float,
    x_size: float,
    y_size: float,
) -> np.ndarray:
    x0 = max(0.0, x_center - (x_size / 2.0))
    x1 = min(100.0, x_center + (x_size / 2.0))
    y0 = max(0.0, y_center - (y_size / 2.0))
    y1 = min(100.0, y_center + (y_size / 2.0))
    pts = np.array(
        [
            _percent_to_outer_px(outer_quad_px, x0, y0),
            _percent_to_outer_px(outer_quad_px, x0, y1),
            _percent_to_outer_px(outer_quad_px, x1, y1),
            _percent_to_outer_px(outer_quad_px, x1, y0),
        ],
        dtype=np.float32,
    )
    return pts


def _draw_quad_label(
    image: np.ndarray,
    quad: np.ndarray,
    label: str,
    *,
    fill_bgr: tuple[int, int, int],
    border_bgr: tuple[int, int, int],
    alpha: float = 0.24,
) -> None:
    pts = np.round(quad).astype(np.int32).reshape(-1, 1, 2)
    overlay = image.copy()
    cv2.fillPoly(overlay, [pts], fill_bgr)
    cv2.addWeighted(overlay, alpha, image, 1.0 - alpha, 0.0, dst=image)
    cv2.polylines(image, [pts], isClosed=True, color=border_bgr, thickness=2, lineType=cv2.LINE_AA)
    center = np.mean(quad, axis=0)
    text_org = (int(center[0]) - 14, int(center[1]) + 6)
    cv2.putText(image, label, text_org, cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 3, cv2.LINE_AA)
    cv2.putText(image, label, text_org, cv2.FONT_HERSHEY_SIMPLEX, 0.65, (20, 20, 20), 1, cv2.LINE_AA)


def _draw_outline(image: np.ndarray, quad: np.ndarray, color: tuple[int, int, int], thickness: int) -> None:
    pts = np.round(quad).astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(image, [pts], isClosed=True, color=color, thickness=thickness, lineType=cv2.LINE_AA)


def render_overlay(image_path: Path, geometry_path: Path, out_path: Path) -> Path:
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    geometry = _load_geometry(geometry_path)
    outer_quad = np.asarray(geometry["outer_sheet_corners_px"], dtype=np.float32)
    board_quad = np.asarray(geometry["chessboard_corners_px"], dtype=np.float32)
    geo = motor_sequence._OFFBOARD_GEOMETRY
    planner = motor_sequence.MotionPlanner(set())

    _draw_outline(image, outer_quad, (0, 255, 0), 3)
    _draw_outline(image, board_quad, (0, 220, 255), 3)

    for idx in range(30):
        slot = motor_sequence._capture_slot_by_index(idx)
        quad = _cell_quad(
            outer_quad,
            x_center=slot.x_pct,
            y_center=slot.y_pct,
            x_size=geo.board_square_x_pct,
            y_size=geo.board_square_y_pct,
        )
        if idx < 20:
            fill = (255, 230, 80)
            border = (255, 220, 0)
        else:
            fill = (80, 170, 255)
            border = (0, 120, 255)
        _draw_quad_label(image, quad, str(idx), fill_bgr=fill, border_bgr=border)

    temp_side_slots = max(1, int(motor_sequence.config.TEMP_RELOCATE_SIDE_SLOTS))
    temp_to_draw = min(motor_sequence.config.MAX_TEMP_RELOCATIONS, temp_side_slots * 2)
    for idx in range(temp_to_draw):
        slot = planner._temp_slot(idx)
        quad = _cell_quad(
            outer_quad,
            x_center=slot.x_pct,
            y_center=slot.y_pct,
            x_size=geo.board_square_x_pct,
            y_size=geo.board_square_y_pct,
        )
        _draw_quad_label(image, quad, f"T{idx}", fill_bgr=(190, 100, 255), border_bgr=(160, 50, 240), alpha=0.18)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(out_path), image):
        raise RuntimeError(f"Failed to write overlay: {out_path}")
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Render current off-board layout overlay.")
    parser.add_argument("--image", default=str(DEFAULT_IMAGE))
    parser.add_argument("--geometry", default=str(DEFAULT_GEOMETRY))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    args = parser.parse_args()

    out_path = render_overlay(Path(args.image), Path(args.geometry), Path(args.out))
    print(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
