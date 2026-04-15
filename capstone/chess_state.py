"""Chess board state (v1): basic legal moves, no check/castling/en-passant."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
import sys
from typing import List, Optional, cast

from coords import Square, parse_square


class ChessPiece(IntEnum):
    EMPTY = 0

    P1_PAWN = 1
    P1_KNIGHT = 2
    P1_BISHOP = 3
    P1_ROOK = 4
    P1_QUEEN = 5
    P1_KING = 6

    P2_PAWN = 7
    P2_KNIGHT = 8
    P2_BISHOP = 9
    P2_ROOK = 10
    P2_QUEEN = 11
    P2_KING = 12


def _is_p1(p: ChessPiece) -> bool:
    return ChessPiece.P1_PAWN <= p <= ChessPiece.P1_KING


def _is_p2(p: ChessPiece) -> bool:
    return ChessPiece.P2_PAWN <= p <= ChessPiece.P2_KING


def _same_side(a: ChessPiece, b: ChessPiece) -> bool:
    return (_is_p1(a) and _is_p1(b)) or (_is_p2(a) and _is_p2(b))


def _is_slider(p: ChessPiece) -> bool:
    return p in (
        ChessPiece.P1_BISHOP,
        ChessPiece.P1_ROOK,
        ChessPiece.P1_QUEEN,
        ChessPiece.P2_BISHOP,
        ChessPiece.P2_ROOK,
        ChessPiece.P2_QUEEN,
    )


@dataclass(frozen=True)
class MoveResult:
    captured: ChessPiece = ChessPiece.EMPTY


class ChessState:
    """
    Coordinates: a1..h8 with rank_index 0 = rank 1 (bottom).

    Side convention:
    - P1 at bottom (ranks 1–2)
    - P2 at top (ranks 7–8)
    """

    def __init__(self) -> None:
        self.board: List[List[ChessPiece]] = [[ChessPiece.EMPTY] * 8 for _ in range(8)]
        self.captured_by_p1: List[ChessPiece] = []
        self.captured_by_p2: List[ChessPiece] = []
        self._place_initial()

    def reset(self) -> None:
        for r in range(8):
            for f in range(8):
                self.board[r][f] = ChessPiece.EMPTY
        self.captured_by_p1.clear()
        self.captured_by_p2.clear()
        self._place_initial()

    def copy(self) -> "ChessState":
        n = cast(ChessState, ChessState.__new__(ChessState))
        n.board = [row[:] for row in self.board]
        n.captured_by_p1 = self.captured_by_p1[:]
        n.captured_by_p2 = self.captured_by_p2[:]
        return n

    def _place_initial(self) -> None:
        # P1 back rank (rank 1)
        self._set_rank(0, [
            ChessPiece.P1_ROOK,
            ChessPiece.P1_KNIGHT,
            ChessPiece.P1_BISHOP,
            ChessPiece.P1_QUEEN,
            ChessPiece.P1_KING,
            ChessPiece.P1_BISHOP,
            ChessPiece.P1_KNIGHT,
            ChessPiece.P1_ROOK,
        ])
        # P1 pawns (rank 2)
        self._set_rank(1, [ChessPiece.P1_PAWN] * 8)

        # P2 pawns (rank 7)
        self._set_rank(6, [ChessPiece.P2_PAWN] * 8)
        # P2 back rank (rank 8)
        self._set_rank(7, [
            ChessPiece.P2_ROOK,
            ChessPiece.P2_KNIGHT,
            ChessPiece.P2_BISHOP,
            ChessPiece.P2_QUEEN,
            ChessPiece.P2_KING,
            ChessPiece.P2_BISHOP,
            ChessPiece.P2_KNIGHT,
            ChessPiece.P2_ROOK,
        ])

    def _set_rank(self, rank_index: int, pieces: List[ChessPiece]) -> None:
        for file_index, p in enumerate(pieces):
            self.board[rank_index][file_index] = p

    def get(self, sq: Square) -> ChessPiece:
        return self.board[sq.rank_index][sq.file_index]

    def set(self, sq: Square, p: ChessPiece) -> None:
        self.board[sq.rank_index][sq.file_index] = p

    def apply_move_trusted(self, start_id: str, end_id: str) -> None:
        """Apply a move from the Pi without full rule validation."""
        start = parse_square(start_id)
        end = parse_square(end_id)
        moving = self.get(start)
        if moving == ChessPiece.EMPTY:
            print(f"p1_move ignored (empty square): {start_id} -> {end_id}", file=sys.stderr)
            return
        captured = self.get(end)
        if captured != ChessPiece.EMPTY and not _same_side(moving, captured):
            if _is_p1(moving):
                self.captured_by_p1.append(captured)
            elif _is_p2(moving):
                self.captured_by_p2.append(captured)
        self._apply_move_internal(start, end)

    def try_apply_p2_move(self, start_id: str, end_id: str) -> Optional[str]:
        """
        Validate and apply a P2 move (basic chess legality).

        Omits: check/checkmate, castling, en passant, pawn promotion UI.
        """
        try:
            start = parse_square(start_id)
            end = parse_square(end_id)
        except ValueError as e:
            return str(e)

        moving = self.get(start)
        if not _is_p2(moving):
            return "must move a player 2 piece"

        target = self.get(end)
        if target != ChessPiece.EMPTY and _is_p2(target):
            return "cannot capture own piece"

        df = end.file_index - start.file_index
        dr = end.rank_index - start.rank_index

        ok, capture_allowed, must_be_empty = self._is_move_shape_legal(moving, start, end, df, dr)
        if not ok:
            return "illegal move for that piece"

        if _is_slider(moving):
            if not self._path_clear(start, end):
                return "path is blocked"

        if must_be_empty and target != ChessPiece.EMPTY:
            return "destination must be empty"

        if not capture_allowed and target != ChessPiece.EMPTY:
            return "move cannot capture"

        if capture_allowed and target == ChessPiece.EMPTY and moving in (ChessPiece.P2_PAWN,):
            return "pawn capture must take a piece"

        if target != ChessPiece.EMPTY:
            self.captured_by_p2.append(target)

        self._apply_move_internal(start, end)
        return None

    def _apply_move_internal(self, start: Square, end: Square) -> None:
        p = self.get(start)
        self.set(start, ChessPiece.EMPTY)
        self.set(end, p)

    def _path_clear(self, start: Square, end: Square) -> bool:
        df = end.file_index - start.file_index
        dr = end.rank_index - start.rank_index
        step_f = 0 if df == 0 else (1 if df > 0 else -1)
        step_r = 0 if dr == 0 else (1 if dr > 0 else -1)
        f = start.file_index + step_f
        r = start.rank_index + step_r
        while (f, r) != (end.file_index, end.rank_index):
            if self.board[r][f] != ChessPiece.EMPTY:
                return False
            f += step_f
            r += step_r
        return True

    def _is_move_shape_legal(
        self, piece: ChessPiece, start: Square, end: Square, df: int, dr: int
    ) -> tuple[bool, bool, bool]:
        """
        Returns (ok, capture_allowed, must_be_empty).
        """
        adf = abs(df)
        adr = abs(dr)

        if piece == ChessPiece.P2_PAWN:
            # P2 pawns move toward decreasing rank (toward rank 1)
            if df == 0 and dr == -1:
                return True, False, True
            if df == 0 and dr == -2 and start.rank_index == 6:
                # two-step from rank 7 if clear (checked by caller via must_be_empty + path clear)
                return True, False, True
            if adf == 1 and dr == -1:
                return True, True, False
            return False, False, False

        if piece == ChessPiece.P2_KNIGHT:
            return (adf, adr) in ((1, 2), (2, 1)), True, False

        if piece == ChessPiece.P2_BISHOP:
            return adf == adr and adf != 0, True, False

        if piece == ChessPiece.P2_ROOK:
            return (df == 0) != (dr == 0), True, False

        if piece == ChessPiece.P2_QUEEN:
            diag = adf == adr and adf != 0
            straight = (df == 0) != (dr == 0)
            return (diag or straight), True, False

        if piece == ChessPiece.P2_KING:
            return max(adf, adr) == 1 and (adf != 0 or adr != 0), True, False

        return False, False, False

