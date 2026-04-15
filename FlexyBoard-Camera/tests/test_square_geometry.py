from __future__ import annotations

import numpy as np

from flexyboard_camera.vision.board_detector import generate_square_geometry


def test_generate_square_geometry_count_and_labels() -> None:
    board_corners = np.array(
        [
            [0.0, 0.0],
            [800.0, 0.0],
            [800.0, 800.0],
            [0.0, 800.0],
        ],
        dtype=np.float32,
    )

    squares = generate_square_geometry(board_corners=board_corners, board_size=(8, 8))
    assert len(squares) == 64

    first = squares[0]
    assert first.index == 0
    assert first.x == 0
    assert first.y == 0
    assert first.label == "a1"
    assert np.allclose(first.center_px, np.array([50.0, 50.0], dtype=np.float32), atol=1e-3)

    last = squares[-1]
    assert last.index == 63
    assert last.x == 7
    assert last.y == 7
    assert last.label == "h8"
    assert np.allclose(last.center_px, np.array([750.0, 750.0], dtype=np.float32), atol=1e-3)
