"""JSON line protocol for Pi ↔ P2 checkers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Optional, Union


@dataclass
class P1MoveMessage:
    frm: str
    to: str

    @staticmethod
    def from_obj(obj: dict[str, Any]) -> "P1MoveMessage":
        return P1MoveMessage(
            frm=str(obj["from"]),
            to=str(obj["to"]),
        )

    def to_obj(self) -> dict[str, Any]:
        return {"type": "p1_move", "from": self.frm, "to": self.to}


@dataclass
class P2MoveMessage:
    frm: str
    to: str

    def to_obj(self) -> dict[str, Any]:
        return {"type": "p2_move", "from": self.frm, "to": self.to}


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
