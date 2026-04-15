from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def save_frame(frame: np.ndarray, output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(path), frame)
    if not ok:
        raise RuntimeError(f"Failed to save frame to {path}")
    return path
