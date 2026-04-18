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


def _square_bounds(coord: BoardCoord, square_w: int, square_h: int) -> tuple[int, int, int, int]:
    x0 = coord.x * square_w
    y0 = coord.y * square_h
    x1 = (coord.x + 1) * square_w
    y1 = (coord.y + 1) * square_h
    return x0, y0, x1, y1


def _square_mean(img: np.ndarray, coord: BoardCoord, square_w: int, square_h: int) -> float:
    x0, y0, x1, y1 = _square_bounds(coord, square_w, square_h)
    return float(np.mean(img[y0:y1, x0:x1]))


def _parity_baseline_means(
    before_img: np.ndarray,
    board_size: tuple[int, int],
    changed_coords: set[tuple[int, int]],
    square_w: int,
    square_h: int,
) -> dict[int, float]:
    board_w, board_h = board_size
    parity_values: dict[int, list[float]] = {0: [], 1: []}
    parity_values_all: dict[int, list[float]] = {0: [], 1: []}

    for y in range(board_h):
        for x in range(board_w):
            parity = (x + y) % 2
            mean_val = float(np.mean(before_img[y * square_h : (y + 1) * square_h, x * square_w : (x + 1) * square_w]))
            parity_values_all[parity].append(mean_val)
            if (x, y) not in changed_coords:
                parity_values[parity].append(mean_val)

    baseline: dict[int, float] = {}
    for parity in (0, 1):
        values = parity_values[parity] if len(parity_values[parity]) >= 4 else parity_values_all[parity]
        baseline[parity] = float(np.median(values)) if values else 128.0
    return baseline


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
        b_mean = _square_mean(inputs.before_img, change.coord, square_w, square_h)
        a_mean = _square_mean(inputs.after_img, change.coord, square_w, square_h)
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

    changed_coord_set = {(c.coord.x, c.coord.y) for c in inputs.changes}
    baseline_before = _parity_baseline_means(
        before_img=inputs.before_img,
        board_size=inputs.board_size,
        changed_coords=changed_coord_set,
        square_w=square_w,
        square_h=square_h,
    )
    baseline_after = _parity_baseline_means(
        before_img=inputs.after_img,
        board_size=inputs.board_size,
        changed_coords=changed_coord_set,
        square_w=square_w,
        square_h=square_h,
    )

    source_before_mean = _square_mean(inputs.before_img, source, square_w, square_h)
    source_after_mean = _square_mean(inputs.after_img, source, square_w, square_h)
    dest_before_mean = _square_mean(inputs.before_img, destination, square_w, square_h)
    dest_after_mean = _square_mean(inputs.after_img, destination, square_w, square_h)

    source_parity = (source.x + source.y) % 2
    dest_parity = (destination.x + destination.y) % 2

    board_contrast = abs(baseline_before[0] - baseline_before[1])
    occupied_margin = max(8.0, board_contrast * 0.22)

    source_before_occ_score = baseline_before[source_parity] - source_before_mean
    source_after_occ_score = baseline_after[source_parity] - source_after_mean
    dest_before_occ_score = baseline_before[dest_parity] - dest_before_mean
    dest_after_occ_score = baseline_after[dest_parity] - dest_after_mean

    source_before_occupied = source_before_occ_score > occupied_margin
    source_after_occupied = source_after_occ_score > occupied_margin
    dest_before_occupied = dest_before_occ_score > occupied_margin
    dest_after_occupied = dest_after_occ_score > occupied_margin

    capture_by_destination_prior = (
        source != destination
        and source_before_occupied
        and (not source_after_occupied)
        and dest_before_occupied
    )
    # Contour assist can intentionally keep weak extra squares so the legal
    # resolver has context, but weak noise should not imply a capture by itself.
    significant_changed_count = sum(
        1
        for change in inputs.changes
        if change.pixel_ratio >= 0.12 or abs(change.signed_intensity_delta) >= 5.0
    )
    capture_by_changed_count = (
        inputs.game.lower() != "chess"
        and source != destination
        and significant_changed_count > 2
    )
    capture_guess = capture_by_destination_prior or capture_by_changed_count

    return MoveEvent(
        game=inputs.game,
        source=source,
        destination=destination,
        moved_piece_type=piece_type,
        capture=capture_guess,
        confidence=confidence,
        metadata={
            "changed_count": len(inputs.changes),
            "significant_changed_count": significant_changed_count,
            "classifier_confidence": piece_conf,
            "source_score": source_candidate[1],
            "dest_score": dest_candidate[1],
            "capture_signals": {
                "capture_by_destination_prior": capture_by_destination_prior,
                "capture_by_changed_count": capture_by_changed_count,
                "occupied_margin": occupied_margin,
                "source_before_occupied": source_before_occupied,
                "source_after_occupied": source_after_occupied,
                "dest_before_occupied": dest_before_occupied,
                "dest_after_occupied": dest_after_occupied,
                "source_before_occ_score": source_before_occ_score,
                "source_after_occ_score": source_after_occ_score,
                "dest_before_occ_score": dest_before_occ_score,
                "dest_after_occ_score": dest_after_occ_score,
            },
            "captured_at": {"x": destination.x, "y": destination.y} if capture_guess else None,
            "captured_piece_type": None,
        },
    )
