from __future__ import annotations

import logging
from dataclasses import dataclass

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class PieceClassification:
    piece_type: str | None
    confidence: float
    backend: str = "stub"
    raw_label: str | None = None
    class_id: int | None = None


class PieceClassifier:
    backend_name = "stub"

    def classify(self, square_patch: np.ndarray, game: str) -> PieceClassification:
        _ = square_patch
        _ = game
        return PieceClassification(piece_type=None, confidence=0.0, backend=self.backend_name)


class YoloPieceClassifier(PieceClassifier):
    backend_name = "yolo"

    def __init__(
        self,
        model_path: str,
        confidence_threshold: float = 0.35,
        input_size: int = 640,
        device: str | None = None,
    ) -> None:
        if not model_path:
            raise ValueError("YOLO classifier requires a model_path")

        try:
            from ultralytics import YOLO  # type: ignore
        except Exception as exc:  # pragma: no cover - import depends on optional dependency
            raise RuntimeError(
                "YOLO backend requested but ultralytics is not installed. "
                "Install with: pip install -r requirements-yolo.txt"
            ) from exc

        self._model = YOLO(model_path)
        self._confidence_threshold = confidence_threshold
        self._input_size = input_size
        self._device = device

    @staticmethod
    def _normalize_label(game: str, label: str) -> str | None:
        # Keep this map small and permissive; model-specific labels are expected.
        lowered = label.strip().lower().replace(" ", "_")
        if not lowered:
            return None

        if game == "chess":
            return lowered
        if game in {"checkers", "sorry"}:
            return lowered
        return lowered

    def classify(self, square_patch: np.ndarray, game: str) -> PieceClassification:
        if square_patch.size == 0:
            return PieceClassification(piece_type=None, confidence=0.0, backend=self.backend_name)

        # Ultralytics expects RGB or path input. Existing pipeline uses BGR arrays from OpenCV.
        rgb_patch = cv2.cvtColor(square_patch, cv2.COLOR_BGR2RGB)
        results = self._model.predict(
            source=rgb_patch,
            conf=self._confidence_threshold,
            imgsz=self._input_size,
            device=self._device,
            verbose=False,
        )
        if not results:
            return PieceClassification(piece_type=None, confidence=0.0, backend=self.backend_name)

        first = results[0]
        boxes = getattr(first, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return PieceClassification(piece_type=None, confidence=0.0, backend=self.backend_name)

        confidences = boxes.conf.detach().cpu().numpy()
        class_ids = boxes.cls.detach().cpu().numpy().astype(int)
        top_idx = int(np.argmax(confidences))
        top_conf = float(confidences[top_idx])
        top_class_id = int(class_ids[top_idx])

        names = getattr(first, "names", None)
        raw_label = str(names[top_class_id]) if isinstance(names, dict) and top_class_id in names else str(top_class_id)
        piece_type = self._normalize_label(game, raw_label)

        return PieceClassification(
            piece_type=piece_type,
            confidence=top_conf,
            backend=self.backend_name,
            raw_label=raw_label,
            class_id=top_class_id,
        )


def build_piece_classifier(
    *,
    enabled: bool,
    backend: str,
    model_path: str | None,
    confidence_threshold: float,
    input_size: int,
    device: str | None,
) -> PieceClassifier:
    if not enabled:
        logger.info("Piece classifier disabled; using stub backend")
        return PieceClassifier()

    if backend.lower() != "yolo":
        logger.warning("Unknown classifier backend '%s'; falling back to stub", backend)
        return PieceClassifier()

    try:
        classifier = YoloPieceClassifier(
            model_path=model_path or "",
            confidence_threshold=confidence_threshold,
            input_size=input_size,
            device=device,
        )
        logger.info("Loaded YOLO piece classifier model=%s", model_path)
        return classifier
    except Exception as exc:
        logger.warning("Failed to initialize YOLO classifier (%s). Falling back to stub.", exc)
        return PieceClassifier()
