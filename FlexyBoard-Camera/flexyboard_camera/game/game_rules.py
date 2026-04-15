from __future__ import annotations

from dataclasses import dataclass

from flexyboard_camera.game.board_models import BoardCoord, BoardSpec
from flexyboard_camera.game.move_models import MotionCommand, MoveEvent


@dataclass(slots=True)
class ResponsePlanner:
    board_spec: BoardSpec

    def propose_response_plan(self, observed_move: MoveEvent) -> list[MotionCommand] | None:
        # Sequence-first interface: Pi owns multi-step planning and sends
        # a list of source/dest movement steps to STM32.
        command = self.propose_response_move(observed_move)
        if command is None:
            return None
        return [command]

    def propose_response_move(self, observed_move: MoveEvent) -> MotionCommand | None:
        if observed_move.source is None or observed_move.destination is None:
            return None

        source = observed_move.destination
        destination = observed_move.source

        if not source.in_bounds(self.board_spec.width, self.board_spec.height):
            return None
        if not destination.in_bounds(self.board_spec.width, self.board_spec.height):
            return None

        return MotionCommand(
            game=observed_move.game,
            source=source,
            destination=destination,
            capture=bool(observed_move.capture),
            path_mode="direct",
        )


def move_to_motion_command(move: MoveEvent, board_spec: BoardSpec) -> MotionCommand:
    if move.source is None or move.destination is None:
        raise ValueError("MoveEvent requires source and destination for motion")

    board_spec.validate(move.source)
    board_spec.validate(move.destination)

    return MotionCommand(
        game=move.game,
        source=move.source,
        destination=move.destination,
        capture=bool(move.capture),
    )
