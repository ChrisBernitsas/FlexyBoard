from __future__ import annotations

import numpy as np

from flexyboard_camera.vision.piece_classifier import PieceClassifier, build_piece_classifier


def test_stub_classifier_returns_none() -> None:
    classifier = PieceClassifier()
    patch = np.zeros((64, 64, 3), dtype=np.uint8)
    result = classifier.classify(patch, game="chess")

    assert result.piece_type is None
    assert result.confidence == 0.0
    assert result.backend == "stub"


def test_yolo_init_failure_falls_back_to_stub() -> None:
    classifier = build_piece_classifier(
        enabled=True,
        backend="yolo",
        model_path=None,
        confidence_threshold=0.35,
        input_size=640,
        device=None,
    )

    patch = np.zeros((64, 64, 3), dtype=np.uint8)
    result = classifier.classify(patch, game="chess")
    assert result.backend == "stub"


def test_unknown_backend_falls_back_to_stub() -> None:
    classifier = build_piece_classifier(
        enabled=True,
        backend="unknown_backend",
        model_path="model.pt",
        confidence_threshold=0.5,
        input_size=640,
        device=None,
    )
    assert isinstance(classifier, PieceClassifier)
