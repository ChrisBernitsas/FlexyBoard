"""Chess state backed by python-chess legal move generation.

Rules now supported for legality/enforcement:
- check / checkmate / stalemate legality
- castling
- en passant
- promotion
"""

from __future__ import annotations

from enum import IntEnum
import sys
from typing import List, Optional, cast

import chess

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


def _is_p1(piece: ChessPiece) -> bool:
    return ChessPiece.P1_PAWN <= piece <= ChessPiece.P1_KING


def _is_p2(piece: ChessPiece) -> bool:
    return ChessPiece.P2_PAWN <= piece <= ChessPiece.P2_KING


def _piece_to_python(piece: ChessPiece) -> chess.Piece | None:
    if piece == ChessPiece.EMPTY:
        return None
    if piece == ChessPiece.P1_PAWN:
        return chess.Piece(chess.PAWN, chess.WHITE)
    if piece == ChessPiece.P1_KNIGHT:
        return chess.Piece(chess.KNIGHT, chess.WHITE)
    if piece == ChessPiece.P1_BISHOP:
        return chess.Piece(chess.BISHOP, chess.WHITE)
    if piece == ChessPiece.P1_ROOK:
        return chess.Piece(chess.ROOK, chess.WHITE)
    if piece == ChessPiece.P1_QUEEN:
        return chess.Piece(chess.QUEEN, chess.WHITE)
    if piece == ChessPiece.P1_KING:
        return chess.Piece(chess.KING, chess.WHITE)
    if piece == ChessPiece.P2_PAWN:
        return chess.Piece(chess.PAWN, chess.BLACK)
    if piece == ChessPiece.P2_KNIGHT:
        return chess.Piece(chess.KNIGHT, chess.BLACK)
    if piece == ChessPiece.P2_BISHOP:
        return chess.Piece(chess.BISHOP, chess.BLACK)
    if piece == ChessPiece.P2_ROOK:
        return chess.Piece(chess.ROOK, chess.BLACK)
    if piece == ChessPiece.P2_QUEEN:
        return chess.Piece(chess.QUEEN, chess.BLACK)
    if piece == ChessPiece.P2_KING:
        return chess.Piece(chess.KING, chess.BLACK)
    return None


def _piece_from_python(piece: chess.Piece | None) -> ChessPiece:
    if piece is None:
        return ChessPiece.EMPTY
    if piece.color == chess.WHITE:
        if piece.piece_type == chess.PAWN:
            return ChessPiece.P1_PAWN
        if piece.piece_type == chess.KNIGHT:
            return ChessPiece.P1_KNIGHT
        if piece.piece_type == chess.BISHOP:
            return ChessPiece.P1_BISHOP
        if piece.piece_type == chess.ROOK:
            return ChessPiece.P1_ROOK
        if piece.piece_type == chess.QUEEN:
            return ChessPiece.P1_QUEEN
        return ChessPiece.P1_KING

    if piece.piece_type == chess.PAWN:
        return ChessPiece.P2_PAWN
    if piece.piece_type == chess.KNIGHT:
        return ChessPiece.P2_KNIGHT
    if piece.piece_type == chess.BISHOP:
        return ChessPiece.P2_BISHOP
    if piece.piece_type == chess.ROOK:
        return ChessPiece.P2_ROOK
    if piece.piece_type == chess.QUEEN:
        return ChessPiece.P2_QUEEN
    return ChessPiece.P2_KING


class ChessState:
    """
    Coordinates: a1..h8 with rank_index 0 = rank 1 (bottom).

    Side convention:
    - P1 at bottom (white)
    - P2 at top (black)
    """

    def __init__(self) -> None:
        self.board: List[List[ChessPiece]] = [[ChessPiece.EMPTY] * 8 for _ in range(8)]
        self.captured_by_p1: List[ChessPiece] = []
        self.captured_by_p2: List[ChessPiece] = []
        self._engine_board = chess.Board()
        self._sync_from_engine()

    def reset(self) -> None:
        self._engine_board = chess.Board()
        self.captured_by_p1.clear()
        self.captured_by_p2.clear()
        self._sync_from_engine()

    def copy(self) -> "ChessState":
        n = cast(ChessState, ChessState.__new__(ChessState))
        n.board = [row[:] for row in self.board]
        n.captured_by_p1 = self.captured_by_p1[:]
        n.captured_by_p2 = self.captured_by_p2[:]
        n._engine_board = self._engine_board.copy(stack=True)
        return n

    def get(self, sq: Square) -> ChessPiece:
        return self.board[sq.rank_index][sq.file_index]

    def set(self, sq: Square, piece: ChessPiece) -> None:
        self.board[sq.rank_index][sq.file_index] = piece
        py_piece = _piece_to_python(piece)
        py_sq = chess.square(sq.file_index, sq.rank_index)
        self._engine_board.set_piece_at(py_sq, py_piece)

    def to_python_board(self) -> chess.Board:
        return self._engine_board.copy(stack=True)

    def turn_side(self) -> str:
        return "p1" if self._engine_board.turn == chess.WHITE else "p2"

    def is_check(self) -> bool:
        return self._engine_board.is_check()

    def is_checkmate(self) -> bool:
        return self._engine_board.is_checkmate()

    def is_stalemate(self) -> bool:
        return self._engine_board.is_stalemate()

    def apply_move_trusted(self, start_id: str, end_id: str, promotion_type: int | None = None) -> None:
        """Apply move from Pi using full legal validation (P1/white expected)."""
        try:
            start = parse_square(start_id)
            end = parse_square(end_id)
        except ValueError:
            print(f"p1_move ignored (invalid square): {start_id} -> {end_id}", file=sys.stderr)
            return

        moving = self.get(start)
        if moving == ChessPiece.EMPTY:
            print(f"p1_move ignored (empty square): {start_id} -> {end_id}", file=sys.stderr)
            return

        move = self._choose_legal_move(start, end, promotion_type=promotion_type)
        if move is None:
            print(f"p1_move ignored (illegal in current position): {start_id} -> {end_id}", file=sys.stderr)
            return

        captured = self._captured_piece_for_move(move)
        self._engine_board.push(move)
        if captured != ChessPiece.EMPTY and _is_p2(captured):
            self.captured_by_p1.append(captured)
        self._sync_from_engine()

    def try_apply_p2_move(
        self,
        start_id: str,
        end_id: str,
        promotion_type: int | None = None,
    ) -> Optional[str]:
        """Validate and apply legal P2/black move."""
        try:
            start = parse_square(start_id)
            end = parse_square(end_id)
        except ValueError as e:
            return str(e)

        moving = self.get(start)
        if not _is_p2(moving):
            return "must move a player 2 piece"

        if self._engine_board.turn != chess.BLACK:
            return "not player 2 turn"

        move = self._choose_legal_move(start, end, promotion_type=promotion_type)
        if move is None:
            return "illegal move for current position"

        captured = self._captured_piece_for_move(move)
        self._engine_board.push(move)
        if captured != ChessPiece.EMPTY and _is_p1(captured):
            self.captured_by_p2.append(captured)
        self._sync_from_engine()
        return None

    def promotion_candidates(self, start: Square, end: Square) -> list[int]:
        from_sq = chess.square(start.file_index, start.rank_index)
        to_sq = chess.square(end.file_index, end.rank_index)
        candidates = [mv for mv in self._engine_board.legal_moves if mv.from_square == from_sq and mv.to_square == to_sq]
        promotions: list[int] = []
        for mv in candidates:
            if mv.promotion is None:
                continue
            promotions.append(int(mv.promotion))
        return promotions

    def _choose_legal_move(
        self,
        start: Square,
        end: Square,
        promotion_type: int | None = None,
    ) -> chess.Move | None:
        from_sq = chess.square(start.file_index, start.rank_index)
        to_sq = chess.square(end.file_index, end.rank_index)
        candidates = [mv for mv in self._engine_board.legal_moves if mv.from_square == from_sq and mv.to_square == to_sq]
        if not candidates:
            return None

        if promotion_type is not None:
            for mv in candidates:
                if mv.promotion == promotion_type:
                    return mv

        # Default to queen promotion if no explicit choice was supplied.
        for mv in candidates:
            if mv.promotion == chess.QUEEN:
                return mv
        return candidates[0]

    def _captured_piece_for_move(self, move: chess.Move) -> ChessPiece:
        if not self._engine_board.is_capture(move):
            return ChessPiece.EMPTY

        if self._engine_board.is_en_passant(move):
            offset = -8 if self._engine_board.turn == chess.WHITE else 8
            captured_sq = move.to_square + offset
        else:
            captured_sq = move.to_square

        captured_py = self._engine_board.piece_at(captured_sq)
        return _piece_from_python(captured_py)

    def _sync_from_engine(self) -> None:
        for rank_index in range(8):
            for file_index in range(8):
                sq = chess.square(file_index, rank_index)
                self.board[rank_index][file_index] = _piece_from_python(self._engine_board.piece_at(sq))

    def manual_resync_engine(self, *, turn_side: str | None = None) -> None:
        board = chess.Board(None)
        for rank_index in range(8):
            for file_index in range(8):
                piece = self.board[rank_index][file_index]
                if piece == ChessPiece.EMPTY:
                    continue
                py_piece = _piece_to_python(piece)
                if py_piece is not None:
                    board.set_piece_at(chess.square(file_index, rank_index), py_piece)
        if turn_side is None:
            turn_side = self.turn_side()
        board.turn = chess.WHITE if turn_side == "p1" else chess.BLACK
        board.castling_rights = chess.BB_EMPTY
        board.ep_square = None
        board.halfmove_clock = 0
        board.fullmove_number = 1
        self._engine_board = board
