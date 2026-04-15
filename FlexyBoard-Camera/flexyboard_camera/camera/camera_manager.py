from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class CameraSettings:
    index: int
    width: int
    height: int
    retries: int = 3
    retry_delay_sec: float = 0.2
    pre_capture_flush_frames: int = 8
    pre_capture_flush_delay_sec: float = 0.02


class CameraError(RuntimeError):
    pass


class CameraManager:
    def __init__(self, settings: CameraSettings):
        self.settings = settings
        self._cap: cv2.VideoCapture | None = None

    def open(self) -> None:
        logger.info("Opening camera index=%s", self.settings.index)
        self._cap = cv2.VideoCapture(self.settings.index)
        if not self._cap or not self._cap.isOpened():
            raise CameraError(f"Unable to open camera index {self.settings.index}")

        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(self.settings.width))
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(self.settings.height))
        if hasattr(cv2, "CAP_PROP_BUFFERSIZE"):
            self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1.0)

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def capture_frame(self) -> np.ndarray:
        if self._cap is None:
            self.open()

        last_error: str | None = None
        for attempt in range(1, self.settings.retries + 1):
            self._flush_stale_frames()
            ok, frame = self._cap.read()  # type: ignore[union-attr]
            if ok and frame is not None:
                logger.info("Captured frame on attempt %d", attempt)
                return frame
            last_error = f"Frame read failed on attempt {attempt}/{self.settings.retries}"
            logger.warning(last_error)
            time.sleep(self.settings.retry_delay_sec)

        raise CameraError(last_error or "Camera capture failed")

    def _flush_stale_frames(self) -> None:
        if self._cap is None:
            return
        frames_to_flush = max(0, self.settings.pre_capture_flush_frames)
        if frames_to_flush == 0:
            return

        flushed = 0
        for _ in range(frames_to_flush):
            ok = self._cap.grab()
            if not ok:
                break
            flushed += 1
            if self.settings.pre_capture_flush_delay_sec > 0:
                time.sleep(self.settings.pre_capture_flush_delay_sec)
        logger.debug("Flushed %d pre-capture frames", flushed)

    @staticmethod
    def load_frame(path: str | Path) -> np.ndarray:
        frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if frame is None:
            raise CameraError(f"Unable to load frame from {path}")
        return frame

    def __enter__(self) -> "CameraManager":
        self.open()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
