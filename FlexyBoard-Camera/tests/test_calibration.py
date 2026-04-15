from __future__ import annotations

from pathlib import Path

from flexyboard_camera.vision.calibration import CalibrationData


def test_calibration_save_load(tmp_path: Path) -> None:
    calibration = CalibrationData.compute(
        board_size=(8, 8),
        roi=(0, 0, 640, 640),
        image_points=[(0.0, 0.0), (640.0, 0.0), (640.0, 640.0), (0.0, 640.0)],
        board_points=[(0.0, 0.0), (8.0, 0.0), (8.0, 8.0), (0.0, 8.0)],
    )

    path = tmp_path / "calibration.json"
    calibration.save(path)

    loaded = CalibrationData.load(path)
    point = loaded.transform_image_point((320.0, 320.0))

    assert loaded.board_size == (8, 8)
    assert 3.8 <= point[0] <= 4.2
    assert 3.8 <= point[1] <= 4.2
