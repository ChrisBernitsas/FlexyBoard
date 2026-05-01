"""JSON line protocol for Pi ↔ P2 checkers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Optional, Union


@dataclass
class P1MoveMessage:
    frm: str
    to: str
    manual_green_captures: list[dict[str, Any]] | None = None

    @staticmethod
    def from_obj(obj: dict[str, Any]) -> "P1MoveMessage":
        captures_raw = obj.get("manual_green_captures")
        manual_green_captures: list[dict[str, Any]] | None = None
        if isinstance(captures_raw, list):
            manual_green_captures = [dict(item) for item in captures_raw if isinstance(item, dict)]
        return P1MoveMessage(
            frm=str(obj["from"]),
            to=str(obj["to"]),
            manual_green_captures=manual_green_captures,
        )

    def to_obj(self) -> dict[str, Any]:
        obj: dict[str, Any] = {"type": "p1_move", "from": self.frm, "to": self.to}
        if self.manual_green_captures is not None:
            obj["manual_green_captures"] = [dict(item) for item in self.manual_green_captures]
        return obj


@dataclass
class P2MoveMessage:
    frm: str
    to: str
    game: str | None = None
    promotion: str | None = None
    stm_sequence: list[str] | None = None
    manual_actions: list[str] | None = None
    turn_steps: list[dict[str, str]] | None = None
    player_ready_after_step_count: int | None = None

    def to_obj(self) -> dict[str, Any]:
        obj: dict[str, Any] = {"type": "p2_move", "from": self.frm, "to": self.to}
        if self.game:
            obj["game"] = self.game
        if self.promotion:
            obj["promotion"] = self.promotion
        if self.stm_sequence is not None:
            obj["stm_sequence"] = list(self.stm_sequence)
        if self.manual_actions is not None:
            obj["manual_actions"] = list(self.manual_actions)
        if self.turn_steps is not None:
            obj["turn_steps"] = [dict(step) for step in self.turn_steps]
        if self.player_ready_after_step_count is not None:
            obj["player_ready_after_step_count"] = int(self.player_ready_after_step_count)
        return obj


def encode_line(obj: dict[str, Any]) -> bytes:
    return (json.dumps(obj, separators=(",", ":")) + "\n").encode("utf-8")


def decode_line(line: bytes) -> Optional[Union[P1MoveMessage, dict[str, Any]]]:
    s = line.decode("utf-8").strip()
    if not s:
        return None
    obj = json.loads(s)
    if not isinstance(obj, dict):
        return None
    t = obj.get("type")
    if t == "p1_move":
        return P1MoveMessage.from_obj(obj)
    return obj
