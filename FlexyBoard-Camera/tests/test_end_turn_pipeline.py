from __future__ import annotations

from pathlib import Path

from flexyboard_camera.app.end_turn_controller import EndTurnController
from flexyboard_camera.utils.config import load_config


def test_end_turn_cycle_with_mock_stm32(test_config_path: Path, synthetic_images) -> None:
    before_path, after_path = synthetic_images
    config = load_config(test_config_path)
    controller = EndTurnController(config)

    result = controller.run_end_turn_cycle(
        before_path=str(before_path),
        after_path=str(after_path),
        force_low_confidence=False,
    )

    assert result.sent_to_stm32 is True
    assert result.stm32_status is not None
    assert result.stm32_status["type"] == "DONE"


def test_end_turn_dry_run_checkers(test_config_path: Path, synthetic_images) -> None:
    before_path, after_path = synthetic_images
    config = load_config(test_config_path)
    config.app.game = "checkers"
    controller = EndTurnController(config)

    move, _ = controller.infer_move(before_path=str(before_path), after_path=str(after_path))
    assert move.game == "checkers"
    assert move.source is not None
    assert move.destination is not None
