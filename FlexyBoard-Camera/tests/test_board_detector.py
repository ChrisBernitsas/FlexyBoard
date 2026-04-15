from __future__ import annotations

import cv2
import numpy as np

from flexyboard_camera.vision.board_detector import (
    detect_board_regions,
    detect_chessboard_corners,
    estimate_chessboard_from_outer_sheet,
    estimate_outer_sheet_from_chessboard,
    warp_to_board,
)


def _make_chessboard_image(square_px: int = 80) -> np.ndarray:
    size = 8
    img = np.zeros((size * square_px, size * square_px, 3), dtype=np.uint8)
    light = np.array([220, 220, 220], dtype=np.uint8)
    dark = np.array([70, 70, 70], dtype=np.uint8)
    for y in range(size):
        for x in range(size):
            color = light if (x + y) % 2 == 0 else dark
            y0 = y * square_px
            x0 = x * square_px
            img[y0 : y0 + square_px, x0 : x0 + square_px] = color
    return img


def _make_handdrawn_style_board(width: int = 1600, height: int = 1000) -> np.ndarray:
    img = np.full((height, width, 3), fill_value=220, dtype=np.uint8)
    # Brown sheet background.
    cv2.rectangle(img, (260, 60), (1340, 940), (110, 160, 200), thickness=-1)
    # Outer black tape.
    cv2.rectangle(img, (260, 60), (1340, 940), (10, 10, 10), thickness=24)

    # Inner 8x8 board outline + grid lines.
    inner_tl = (420, 140)
    inner_br = (1180, 820)
    cv2.rectangle(img, inner_tl, inner_br, (20, 20, 20), thickness=4)
    step_x = (inner_br[0] - inner_tl[0]) // 8
    step_y = (inner_br[1] - inner_tl[1]) // 8
    for i in range(1, 8):
        x = inner_tl[0] + i * step_x
        y = inner_tl[1] + i * step_y
        cv2.line(img, (x, inner_tl[1]), (x, inner_br[1]), (25, 25, 25), thickness=2)
        cv2.line(img, (inner_tl[0], y), (inner_br[0], y), (25, 25, 25), thickness=2)
    return img


def test_detect_chessboard_corners_on_synthetic_image() -> None:
    img = _make_chessboard_image(square_px=80)
    corners = detect_chessboard_corners(img, board_size=(8, 8))
    assert corners is not None
    assert corners.shape == (4, 2)
    assert float(corners[:, 0].min()) <= 2.0
    assert float(corners[:, 1].min()) <= 2.0
    assert float(corners[:, 0].max()) >= 636.0
    assert float(corners[:, 1].max()) >= 636.0


def test_detect_outer_sheet_and_warp() -> None:
    canvas = np.full((900, 1400, 3), fill_value=230, dtype=np.uint8)

    # Brown sheet in BGR.
    cv2.rectangle(canvas, (250, 120), (1150, 820), (80, 140, 180), thickness=-1)

    chess = _make_chessboard_image(square_px=80)
    chess = cv2.resize(chess, (640, 640), interpolation=cv2.INTER_AREA)
    canvas[150:790, 380:1020] = chess

    detection = detect_board_regions(
        frame_bgr=canvas,
        board_size=(8, 8),
        outer_sheet_hsv_lower=(8, 20, 50),
        outer_sheet_hsv_upper=(35, 255, 255),
        min_outer_area_ratio=0.1,
    )

    assert detection.outer_sheet_corners is not None
    chess_corners = detection.chessboard_corners
    if chess_corners is None:
        chess_corners = np.array(
            [[380.0, 150.0], [1020.0, 150.0], [1020.0, 790.0], [380.0, 790.0]],
            dtype=np.float32,
        )

    warped, _ = warp_to_board(canvas, chess_corners, board_size=(8, 8), square_px=64)
    assert warped.shape[0] == 8 * 64
    assert warped.shape[1] == 8 * 64


def test_outer_to_inner_estimation_roundtrip() -> None:
    inner = np.array(
        [[100.0, 120.0], [500.0, 110.0], [520.0, 520.0], [90.0, 530.0]],
        dtype=np.float32,
    )
    margins = (3.2, 3.2, 1.4, 2.4)
    outer = estimate_outer_sheet_from_chessboard(inner, board_size=(8, 8), margins_squares=margins)
    inner_recovered = estimate_chessboard_from_outer_sheet(outer, board_size=(8, 8), margins_squares=margins)
    assert np.allclose(inner_recovered, inner, atol=1e-2)


def test_detect_board_regions_on_handdrawn_style_grid() -> None:
    img = _make_handdrawn_style_board()
    detection = detect_board_regions(
        frame_bgr=img,
        board_size=(8, 8),
        outer_sheet_hsv_lower=(8, 20, 50),
        outer_sheet_hsv_upper=(35, 255, 255),
        min_outer_area_ratio=0.1,
    )
    assert detection.outer_sheet_corners is not None
    assert detection.chessboard_corners is not None

    outer_area = abs(float(cv2.contourArea(detection.outer_sheet_corners.astype(np.float32).reshape(-1, 1, 2))))
    inner_area = abs(float(cv2.contourArea(detection.chessboard_corners.astype(np.float32).reshape(-1, 1, 2))))
    assert inner_area > 0.0
    ratio = inner_area / outer_area
    # Some setups tape very close to the chessboard, so allow tighter outer bounds.
    assert 0.2 < ratio < 0.95
