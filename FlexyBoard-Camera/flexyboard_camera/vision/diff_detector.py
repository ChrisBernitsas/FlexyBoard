from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from flexyboard_camera.game.board_models import BoardCoord


@dataclass(slots=True)
class SquareChange:
    coord: BoardCoord
    pixel_ratio: float
    signed_intensity_delta: float


@dataclass(slots=True)
class DiffResult:
    diff_image: np.ndarray
    threshold_image: np.ndarray
    changes: list[SquareChange]


def detect_square_changes(
    before_img: np.ndarray,
    after_img: np.ndarray,
    board_size: tuple[int, int],
    diff_threshold: int,
    min_changed_ratio: float,
) -> DiffResult:
    diff = cv2.absdiff(before_img, after_img)
    _, thresh = cv2.threshold(diff, diff_threshold, 255, cv2.THRESH_BINARY)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    width, height = board_size
    img_h, img_w = thresh.shape
    square_w = img_w // width
    square_h = img_h // height

    changes: list[SquareChange] = []
    signed = after_img.astype(np.int16) - before_img.astype(np.int16)

    for y in range(height):
        for x in range(width):
            x0 = x * square_w
            y0 = y * square_h
            x1 = (x + 1) * square_w
            y1 = (y + 1) * square_h

            square_thresh = thresh[y0:y1, x0:x1]
            square_signed = signed[y0:y1, x0:x1]
            active = float(np.count_nonzero(square_thresh))
            total = float(square_thresh.size)
            ratio = active / total if total else 0.0
            if ratio < min_changed_ratio:
                continue

            signed_mean = float(np.mean(square_signed))
            changes.append(
                SquareChange(
                    coord=BoardCoord(x=x, y=y),
                    pixel_ratio=ratio,
                    signed_intensity_delta=signed_mean,
                )
            )

    changes.sort(key=lambda item: item.pixel_ratio, reverse=True)
    return DiffResult(diff_image=diff, threshold_image=thresh, changes=changes)
