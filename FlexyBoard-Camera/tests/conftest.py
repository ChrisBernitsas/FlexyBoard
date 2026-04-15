from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml


def make_board_image(
    board_size: tuple[int, int] = (8, 8),
    square_px: int = 80,
    pieces: list[tuple[int, int]] | None = None,
) -> np.ndarray:
    width, height = board_size
    img_h = height * square_px
    img_w = width * square_px

    image = np.zeros((img_h, img_w, 3), dtype=np.uint8)
    light = np.array([210, 210, 210], dtype=np.uint8)
    dark = np.array([120, 120, 120], dtype=np.uint8)

    for y in range(height):
        for x in range(width):
            color = light if (x + y) % 2 == 0 else dark
            y0 = y * square_px
            x0 = x * square_px
            image[y0 : y0 + square_px, x0 : x0 + square_px] = color

    for px, py in pieces or []:
        center = (px * square_px + square_px // 2, py * square_px + square_px // 2)
        cv2.circle(image, center, square_px // 3, (30, 30, 30), thickness=-1)

    return image


@pytest.fixture
def synthetic_images(tmp_path: Path) -> tuple[Path, Path]:
    before = make_board_image(pieces=[(1, 1)])
    after = make_board_image(pieces=[(1, 3)])

    before_path = tmp_path / "before.png"
    after_path = tmp_path / "after.png"
    cv2.imwrite(str(before_path), before)
    cv2.imwrite(str(after_path), after)
    return before_path, after_path


@pytest.fixture
def test_config_path(tmp_path: Path) -> Path:
    cfg = {
        "app": {
            "game": "chess",
            "confidence_threshold": 0.2,
            "allow_low_confidence_override": False,
        },
        "camera": {
            "index": 0,
            "width": 640,
            "height": 640,
            "retries": 1,
            "retry_delay_sec": 0.01,
        },
        "vision": {
            "board_size": [8, 8],
            "roi": [0, 0, 640, 640],
            "blur_kernel": 3,
            "diff_threshold": 20,
            "min_changed_squares": 2,
            "changed_square_pixel_ratio": 0.05,
        },
        "board": {
            "square_size_mm": 50.0,
            "origin_offset_mm": [0.0, 0.0],
        },
        "comms": {
            "port": "mock://stm32",
            "baudrate": 115200,
            "timeout_sec": 0.3,
            "retries": 2,
        },
        "paths": {
            "calibration_file": str(tmp_path / "calibration.json"),
            "logs_dir": str(tmp_path / "logs"),
            "debug_dir": str(tmp_path / "debug"),
        },
        "safety": {
            "auto_home_before_move": True,
            "fail_on_low_confidence": True,
        },
    }
    path = tmp_path / "config.yaml"
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(cfg, handle)
    return path
