"""In-process mock transport for UI development without a Pi."""

from __future__ import annotations

from queue import Empty, Queue
from typing import Any, List, Optional

from ipc.protocol import P1MoveMessage, P2MoveMessage


class MockTransport:
    def __init__(self) -> None:
        self._incoming: Queue[P1MoveMessage] = Queue()
        self._control: Queue[dict[str, Any]] = Queue()
        self.sent_p2: List[P2MoveMessage] = []

    def connect(self) -> None:
        pass

    def close(self) -> None:
        pass

    def inject_p1_move(
        self,
        frm: str,
        to: str,
        manual_green_captures: list[dict[str, Any]] | None = None,
    ) -> None:
        self._incoming.put(
            P1MoveMessage(frm=frm, to=to, manual_green_captures=manual_green_captures)
        )

    def inject_control_message(self, obj: dict[str, Any]) -> None:
        self._control.put(dict(obj))

    def poll_p1_move(self, block: bool = False, timeout: float = 0.0) -> Optional[P1MoveMessage]:
        try:
            return self._incoming.get(block=block, timeout=timeout if block else 0)
        except Empty:
            return None

    def drain_p1_moves(self) -> None:
        while True:
            try:
                self._incoming.get_nowait()
            except Empty:
                break

    def poll_control_message(self, block: bool = False, timeout: float = 0.0) -> Optional[dict[str, Any]]:
        try:
            return self._control.get(block=block, timeout=timeout if block else 0)
        except Empty:
            return None

    def send_p2_move(self, msg: P2MoveMessage) -> None:
        self.sent_p2.append(msg)

    def send_control_message(self, obj: dict[str, Any]) -> None:
        pass
