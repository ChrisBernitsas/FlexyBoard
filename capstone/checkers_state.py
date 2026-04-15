"""American checkers board state: P1 bottom (ranks 1–3), P2 top (ranks 6–8)."""

from __future__ import annotations

from enum import IntEnum
import sys
from typing import List, Optional, cast

from coords import Square, is_dark_square, parse_square


class Piece(IntEnum):
    EMPTY = 0
    P1_MAN = 1
    P1_KING = 2
    P2_MAN = 3
    P2_KING = 4


def _is_p1(p: Piece) -> bool:
    return p in (Piece.P1_MAN, Piece.P1_KING)


def _is_p2(p: Piece) -> bool:
    return p in (Piece.P2_MAN, Piece.P2_KING)


def _is_enemy(a: Piece, b: Piece) -> bool:
    return (_is_p1(a) and _is_p2(b)) or (_is_p2(a) and _is_p1(b))


class CheckersState:
    def __init__(self) -> None:
        self.board: List[List[Piece]] = [[Piece.EMPTY] * 8 for _ in range(8)]
        self.captured_by_p1: List[Piece] = []
        self.captured_by_p2: List[Piece] = []
        self._place_initial()

    def _place_initial(self) -> None:
        for r in range(8):
            for f in range(8):
                if not is_dark_square(f, r):
                    continue
                if r <= 2:
                    self.board[r][f] = Piece.P1_MAN
                elif r >= 5:
                    self.board[r][f] = Piece.P2_MAN

    def reset(self) -> None:
        """Restore the opening position (12 pieces per side on dark squares)."""
        for r in range(8):
            for f in range(8):
                self.board[r][f] = Piece.EMPTY
        self.captured_by_p1.clear()
        self.captured_by_p2.clear()
        self._place_initial()

    def get(self, sq: Square) -> Piece:
        return self.board[sq.rank_index][sq.file_index]

    def set(self, sq: Square, p: Piece) -> None:
        self.board[sq.rank_index][sq.file_index] = p

    def copy(self) -> "CheckersState":
        n = cast(CheckersState, CheckersState.__new__(CheckersState))
        n.board = [row[:] for row in self.board]
        n.captured_by_p1 = self.captured_by_p1[:]
        n.captured_by_p2 = self.captured_by_p2[:]
        return n

    def apply_move_trusted(self, start_id: str, end_id: str) -> None:
        """Apply move from Pi without validation (P1); handles one step or one jump."""
        start = parse_square(start_id)
        end = parse_square(end_id)
        moving = self.get(start)
        if moving == Piece.EMPTY:
            print(f"p1_move ignored (empty square): {start_id} -> {end_id}", file=sys.stderr)
            return
        df = end.file_index - start.file_index
        dr = end.rank_index - start.rank_index
        dist = abs(df)
        if dist == 2 and abs(dr) == 2:
            step_f = 1 if df > 0 else -1
            step_r = 1 if dr > 0 else -1
            mid = Square(start.file_index + step_f, start.rank_index + step_r)
            jumped = self.get(mid)
            if _is_p2(jumped):
                self.captured_by_p1.append(jumped)
            self.set(mid, Piece.EMPTY)
        self._apply_move_internal(start, end)
        self.maybe_promote_p1_at(end)

    def try_apply_p2_move(self, start_id: str, end_id: str) -> Optional[str]:
        """
        Validate and apply a P2 move. Returns None on success, else error message.
        Supports single-step diagonal moves and single jumps.
        """
        try:
            start = parse_square(start_id)
            end = parse_square(end_id)
        except ValueError as e:
            return str(e)

        piece = self.get(start)
        if not _is_p2(piece):
            return "must move a player 2 piece"

        if self.get(end) != Piece.EMPTY:
            return "destination must be empty"

        if not is_dark_square(end.file_index, end.rank_index):
            return "must land on a dark square"

        df = end.file_index - start.file_index
        dr = end.rank_index - start.rank_index

        if abs(df) != abs(dr) or df == 0:
            return "must move diagonally"

        step_f = 1 if df > 0 else -1
        step_r = 1 if dr > 0 else -1

        # P2 men move toward decreasing rank (toward rank 1); kings any diagonal
        if piece == Piece.P2_MAN and dr > 0:
            return "men cannot move backward"
        if _is_p1(piece):
            return "must move a player 2 piece"

        dist = abs(df)
        if dist == 1:
            self._apply_move_internal(start, end)
            self._maybe_promote_p2(end)
            return None

        if dist == 2:
            mid_f = start.file_index + step_f
            mid_r = start.rank_index + step_r
            mid = Square(mid_f, mid_r)
            captured = self.get(mid)
            if not _is_p1(captured):
                return "jump must capture opponent piece"
            self.captured_by_p2.append(captured)
            self.set(mid, Piece.EMPTY)
            self._apply_move_internal(start, end)
            self._maybe_promote_p2(end)
            return None

        return "only single step or single jump supported"

    def _apply_move_internal(self, start: Square, end: Square) -> None:
        p = self.get(start)
        self.set(start, Piece.EMPTY)
        self.set(end, p)

    def _maybe_promote_p2(self, end: Square) -> None:
        if end.rank_index == 0 and self.get(end) == Piece.P2_MAN:
            self.set(end, Piece.P2_KING)

    def maybe_promote_p1_at(self, end: Square) -> None:
        """After trusted P1 move, promote if reached rank 8."""
        if end.rank_index == 7 and self.get(end) == Piece.P1_MAN:
            self.set(end, Piece.P1_KING)
