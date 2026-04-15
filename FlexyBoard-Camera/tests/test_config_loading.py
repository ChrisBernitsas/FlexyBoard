from __future__ import annotations

from pathlib import Path

from flexyboard_camera.utils.config import load_config


def test_camera_flush_defaults_from_minimal_config(test_config_path: Path) -> None:
    config = load_config(test_config_path)
    assert config.camera.pre_capture_flush_frames == 8
    assert config.camera.pre_capture_flush_delay_sec == 0.02


def test_default_config_sets_nonzero_camera_flush() -> None:
    config_path = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"
    config = load_config(config_path)
    assert config.camera.pre_capture_flush_frames > 0
