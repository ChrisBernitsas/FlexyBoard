from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from flexyboard_camera.game.board_models import BoardCoord
from flexyboard_camera.game.move_models import MoveEvent
from flexyboard_camera.vision.diff_detector import SquareChange
from flexyboard_camera.vision.piece_classifier import PieceClassifier


@dataclass(slots=True)
class InferenceInputs:
    game: str
    board_size: tuple[int, int]
    before_img: np.ndarray
    after_img: np.ndarray
    changes: list[SquareChange]


def infer_move(inputs: InferenceInputs, classifier: PieceClassifier | None = None) -> MoveEvent:
    if len(inputs.changes) < 2:
        return MoveEvent(
            game=inputs.game,
            source=None,
            destination=None,
            moved_piece_type=None,
            capture=None,
            confidence=0.0,
            metadata={"reason": "insufficient_changed_squares", "changed_count": len(inputs.changes)},
        )

    # Occupancy proxy: darker squares generally indicate a piece.
    signed_scores: list[tuple[BoardCoord, float, float]] = []
    board_w, board_h = inputs.board_size
    square_w = inputs.before_img.shape[1] // board_w
    square_h = inputs.before_img.shape[0] // board_h

    for change in inputs.changes:
        x0 = change.coord.x * square_w
        y0 = change.coord.y * square_h
        x1 = (change.coord.x + 1) * square_w
        y1 = (change.coord.y + 1) * square_h

        b_mean = float(np.mean(inputs.before_img[y0:y1, x0:x1]))
        a_mean = float(np.mean(inputs.after_img[y0:y1, x0:x1]))
        occupancy_delta = (255.0 - a_mean) - (255.0 - b_mean)
        signed_scores.append((change.coord, occupancy_delta, change.pixel_ratio))

    source_candidate = min(signed_scores, key=lambda value: value[1])
    dest_candidate = max(signed_scores, key=lambda value: value[1])

    source = source_candidate[0]
    destination = dest_candidate[0]

    confidence = min(1.0, abs(source_candidate[1]) / 30.0 + abs(dest_candidate[1]) / 30.0)
    confidence *= min(1.0, (source_candidate[2] + dest_candidate[2]) / 0.4)

    piece_type = None
    piece_conf = 0.0
    if classifier is not None:
        cx0 = destination.x * square_w
        cy0 = destination.y * square_h
        cx1 = (destination.x + 1) * square_w
        cy1 = (destination.y + 1) * square_h
        result = classifier.classify(inputs.after_img[cy0:cy1, cx0:cx1], inputs.game)
        piece_type = result.piece_type
        piece_conf = result.confidence

    capture_guess = source != destination and len(inputs.changes) > 2

    return MoveEvent(
        game=inputs.game,
        source=source,
        destination=destination,
        moved_piece_type=piece_type,
        capture=capture_guess,
        confidence=confidence,
        metadata={
            "changed_count": len(inputs.changes),
            "classifier_confidence": piece_conf,
            "source_score": source_candidate[1],
            "dest_score": dest_candidate[1],
        },
    )
