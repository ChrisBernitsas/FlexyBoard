"""Minimal Parcheesi-like token state for Player 2 integration.

This mode intentionally keeps move legality simple:
- P1 moves are trusted from Pi/CV (`apply_move_trusted`)
- P2 moves are manual drag-drop (or simple AI), any destination not occupied by P2

The motion planner converts these square moves into percent-based STM sequences.
"""

from __future__ import annotations

from typing import List, Optional, cast

from checkers_state import Piece
from coords import Square, parse_square


def _is_p1(piece: Piece) -> bool:
    return piece in (Piece.P1_MAN, Piece.P1_KING)


def _is_p2(piece: Piece) -> bool:
    return piece in (Piece.P2_MAN, Piece.P2_KING)


class ParcheesiState:
    """Simple 8x8 token board used by the current Parcheesi MVP UI."""

    def __init__(self) -> None:
        self.board: List[List[Piece]] = [[Piece.EMPTY] * 8 for _ in range(8)]
        self.captured_by_p1: List[Piece] = []
        self.captured_by_p2: List[Piece] = []
        self._place_initial()

    def _place_initial(self) -> None:
        # Four tokens per side in opposite corners.
        for sq in (Square(0, 0), Square(1, 0), Square(0, 1), Square(1, 1)):
            self.set(sq, Piece.P1_MAN)
        for sq in (Square(6, 7), Square(7, 7), Square(6, 6), Square(7, 6)):
            self.set(sq, Piece.P2_MAN)

    def reset(self) -> None:
        for y in range(8):
            for x in range(8):
                self.board[y][x] = Piece.EMPTY
        self.captured_by_p1.clear()
        self.captured_by_p2.clear()
        self._place_initial()

    def copy(self) -> "ParcheesiState":
        nxt = cast(ParcheesiState, ParcheesiState.__new__(ParcheesiState))
        nxt.board = [row[:] for row in self.board]
        nxt.captured_by_p1 = self.captured_by_p1[:]
        nxt.captured_by_p2 = self.captured_by_p2[:]
        return nxt

    def get(self, sq: Square) -> Piece:
        return self.board[sq.rank_index][sq.file_index]

    def set(self, sq: Square, piece: Piece) -> None:
        self.board[sq.rank_index][sq.file_index] = piece

    def _move_piece(self, start: Square, end: Square) -> None:
        moving = self.get(start)
        self.set(start, Piece.EMPTY)
        self.set(end, moving)

    def apply_move_trusted(self, start_id: str, end_id: str) -> None:
        """Apply P1 move from Pi/CV without strict Parcheesi rule enforcement."""
        try:
            start = parse_square(start_id)
            end = parse_square(end_id)
        except ValueError:
            return

        moving = self.get(start)
        if not _is_p1(moving):
            return
        if start == end:
            return

        target = self.get(end)
        if _is_p1(target):
            return
        if _is_p2(target):
            self.captured_by_p1.append(target)

        self._move_piece(start, end)

    def try_apply_p2_move(self, start_id: str, end_id: str) -> Optional[str]:
        """Validate and apply a P2 move. Returns None on success."""
        try:
            start = parse_square(start_id)
            end = parse_square(end_id)
        except ValueError as exc:
            return str(exc)

        moving = self.get(start)
        if not _is_p2(moving):
            return "must move a player 2 token"
        if start == end:
            return "source and destination must differ"

        target = self.get(end)
        if _is_p2(target):
            return "destination occupied by player 2 token"
        if _is_p1(target):
            self.captured_by_p2.append(target)

        self._move_piece(start, end)
        return None
