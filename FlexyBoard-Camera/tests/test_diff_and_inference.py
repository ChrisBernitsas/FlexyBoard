from __future__ import annotations

import cv2

from flexyboard_camera.vision.diff_detector import detect_square_changes
from flexyboard_camera.vision.move_inference import InferenceInputs, infer_move
from flexyboard_camera.vision.preprocess import preprocess_frame


def test_diff_inference_from_synthetic_pair(synthetic_images) -> None:
    before_path, after_path = synthetic_images
    before = cv2.imread(str(before_path), cv2.IMREAD_COLOR)
    after = cv2.imread(str(after_path), cv2.IMREAD_COLOR)

    before_pre = preprocess_frame(before, roi=(0, 0, before.shape[1], before.shape[0]), blur_kernel=3)
    after_pre = preprocess_frame(after, roi=(0, 0, after.shape[1], after.shape[0]), blur_kernel=3)

    diff = detect_square_changes(
        before_img=before_pre.enhanced,
        after_img=after_pre.enhanced,
        board_size=(8, 8),
        diff_threshold=20,
        min_changed_ratio=0.05,
    )

    move = infer_move(
        InferenceInputs(
            game="chess",
            board_size=(8, 8),
            before_img=before_pre.enhanced,
            after_img=after_pre.enhanced,
            changes=diff.changes,
        )
    )

    assert move.source is not None
    assert move.destination is not None
    assert (move.source.x, move.source.y) == (1, 1)
    assert (move.destination.x, move.destination.y) == (1, 3)
    assert move.confidence > 0.2
