from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np


@dataclass(slots=True)
class CalibrationData:
    board_size: tuple[int, int]
    roi: tuple[int, int, int, int]
    image_points: list[tuple[float, float]]
    board_points: list[tuple[float, float]]
    homography: list[list[float]]
    created_at: str

    @classmethod
    def compute(
        cls,
        board_size: tuple[int, int],
        roi: tuple[int, int, int, int],
        image_points: list[tuple[float, float]],
        board_points: list[tuple[float, float]],
    ) -> "CalibrationData":
        if len(image_points) < 4 or len(board_points) < 4:
            raise ValueError("Need at least 4 points for homography")

        image_np = np.array(image_points, dtype=np.float32)
        board_np = np.array(board_points, dtype=np.float32)
        h_matrix, _ = cv2.findHomography(image_np, board_np)
        if h_matrix is None:
            raise RuntimeError("Homography solve failed")

        return cls(
            board_size=board_size,
            roi=roi,
            image_points=image_points,
            board_points=board_points,
            homography=h_matrix.tolist(),
            created_at=datetime.now(timezone.utc).isoformat(),
        )

    def transform_image_point(self, point_xy: tuple[float, float]) -> tuple[float, float]:
        h_matrix = np.array(self.homography, dtype=np.float32)
        point = np.array([[point_xy]], dtype=np.float32)
        transformed = cv2.perspectiveTransform(point, h_matrix)
        x, y = transformed[0][0]
        return float(x), float(y)

    def save(self, path: str | Path) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8") as handle:
            json.dump(asdict(self), handle, indent=2)
        return output

    @classmethod
    def load(cls, path: str | Path) -> "CalibrationData":
        with Path(path).open("r", encoding="utf-8") as handle:
            raw = json.load(handle)
        return cls(
            board_size=tuple(raw["board_size"]),
            roi=tuple(raw["roi"]),
            image_points=[tuple(pair) for pair in raw["image_points"]],
            board_points=[tuple(pair) for pair in raw["board_points"]],
            homography=raw["homography"],
            created_at=raw["created_at"],
        )


def default_corners_from_roi(roi: tuple[int, int, int, int], board_size: tuple[int, int]) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
    x, y, w, h = roi
    image_points = [
        (float(x), float(y)),
        (float(x + w), float(y)),
        (float(x + w), float(y + h)),
        (float(x), float(y + h)),
    ]
    bw, bh = board_size
    board_points = [(0.0, 0.0), (float(bw), 0.0), (float(bw), float(bh)), (0.0, float(bh))]
    return image_points, board_points
