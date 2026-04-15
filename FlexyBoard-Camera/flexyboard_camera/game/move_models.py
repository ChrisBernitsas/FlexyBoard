from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from flexyboard_camera.game.board_models import BoardCoord


@dataclass(slots=True)
class MoveEvent:
    game: str
    source: BoardCoord | None
    destination: BoardCoord | None
    moved_piece_type: str | None
    capture: bool | None
    confidence: float
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        def coord_to_dict(coord: BoardCoord | None) -> dict[str, int] | None:
            if coord is None:
                return None
            return {"x": coord.x, "y": coord.y}

        return {
            "game": self.game,
            "source": coord_to_dict(self.source),
            "destination": coord_to_dict(self.destination),
            "moved_piece_type": self.moved_piece_type,
            "capture": self.capture,
            "confidence": self.confidence,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }


@dataclass(slots=True)
class MotionCommand:
    game: str
    source: BoardCoord
    destination: BoardCoord
    path_mode: str = "direct"
    capture: bool = False
    waypoints: list[BoardCoord] = field(default_factory=list)

    def to_payload(self, minimal_for_stm32: bool = False) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "game": self.game,
            "source": {"x": self.source.x, "y": self.source.y},
            "dest": {"x": self.destination.x, "y": self.destination.y},
            "path_mode": self.path_mode,
            "capture": self.capture,
            "waypoints": [{"x": point.x, "y": point.y} for point in self.waypoints],
        }
        if minimal_for_stm32:
            return {
                "source": payload["source"],
                "dest": payload["dest"],
            }
        return payload

    def to_step_payloads(self, minimal_for_stm32: bool = True) -> list[dict[str, Any]]:
        # Expand optional waypoint route into explicit source->dest segments.
        route: list[BoardCoord] = [self.source, *self.waypoints, self.destination]
        steps: list[dict[str, Any]] = []
        for idx in range(len(route) - 1):
            step = MotionCommand(
                game=self.game,
                source=route[idx],
                destination=route[idx + 1],
                path_mode=self.path_mode,
                capture=self.capture,
                waypoints=[],
            )
            steps.append(step.to_payload(minimal_for_stm32=minimal_for_stm32))
        return steps


@dataclass(slots=True)
class EndTurnResult:
    move_event: MoveEvent
    response_command: MotionCommand | None
    sent_to_stm32: bool
    stm32_status: dict[str, Any] | None
    blocked_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "move_event": self.move_event.to_dict(),
            "response_command": self.response_command.to_payload() if self.response_command else None,
            "sent_to_stm32": self.sent_to_stm32,
            "stm32_status": self.stm32_status,
            "blocked_reason": self.blocked_reason,
        }
