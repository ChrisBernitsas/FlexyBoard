from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(slots=True)
class PreprocessResult:
    raw_roi: np.ndarray
    grayscale: np.ndarray
    enhanced: np.ndarray


def preprocess_frame(
    frame_bgr: np.ndarray,
    roi: tuple[int, int, int, int],
    blur_kernel: int,
) -> PreprocessResult:
    x, y, w, h = roi
    roi_bgr = frame_bgr[y : y + h, x : x + w]
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)

    kernel = blur_kernel if blur_kernel % 2 == 1 else blur_kernel + 1
    blurred = cv2.GaussianBlur(gray, (kernel, kernel), 0)

    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    enhanced = clahe.apply(blurred)

    return PreprocessResult(raw_roi=roi_bgr, grayscale=gray, enhanced=enhanced)
