"""Simple legal-move AI for P2 (capture-preferred, deterministic fallback)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import config
from checkers_state import CheckersState, Piece as CheckersPiece
from chess_state import ChessPiece, ChessState
from parcheesi_state import ParcheesiState
from coords import FILES, RANKS, parse_square

try:
    import chess
    import chess.engine
except Exception:  # pragma: no cover - optional dependency
    chess = None  # type: ignore[assignment]


@dataclass(frozen=True)
class ChosenMove:
    start_id: str
    end_id: str
    is_capture: bool


_ENGINE: chess.engine.SimpleEngine | None = None  # type: ignore[name-defined]
_ENGINE_PATH: str | None = None


def _all_square_ids() -> Iterable[str]:
    for r in RANKS:
        for f in FILES:
            yield f + r


def _simple_choose_p2_chess_move(state: ChessState) -> Optional[ChosenMove]:
    quiet_move: Optional[ChosenMove] = None

    for start_id in _all_square_ids():
        start_sq = parse_square(start_id)
        moving = state.get(start_sq)
        if not (ChessPiece.P2_PAWN <= moving <= ChessPiece.P2_KING):
            continue

        for end_id in _all_square_ids():
            if end_id == start_id:
                continue

            end_sq = parse_square(end_id)
            target = state.get(end_sq)
            trial = state.copy()
            err = trial.try_apply_p2_move(start_id, end_id)
            if err is not None:
                continue

            is_capture = target != ChessPiece.EMPTY
            candidate = ChosenMove(start_id=start_id, end_id=end_id, is_capture=is_capture)
            if is_capture:
                return candidate
            if quiet_move is None:
                quiet_move = candidate

    return quiet_move


def _get_engine() -> chess.engine.SimpleEngine | None:  # type: ignore[name-defined]
    global _ENGINE, _ENGINE_PATH
    if chess is None:
        return None
    if _ENGINE is not None:
        return _ENGINE

    engine_path = config.discover_stockfish_path()
    if not engine_path:
        return None
    try:
        _ENGINE = chess.engine.SimpleEngine.popen_uci(engine_path)
        _ENGINE_PATH = engine_path
    except Exception:
        _ENGINE = None
        _ENGINE_PATH = None
    return _ENGINE


def close_engine() -> None:
    global _ENGINE, _ENGINE_PATH
    if _ENGINE is not None:
        try:
            _ENGINE.quit()
        except Exception:
            pass
    _ENGINE = None
    _ENGINE_PATH = None


def choose_p2_chess_move(state: ChessState) -> Optional[ChosenMove]:
    engine = _get_engine()
    if engine is None:
        return _simple_choose_p2_chess_move(state)

    try:
        board = state.to_python_board()
        board.turn = chess.BLACK
        result = engine.play(board, chess.engine.Limit(time=max(0.05, config.STOCKFISH_MOVE_TIME_SEC)))
        mv = result.move
        if mv is None:
            return _simple_choose_p2_chess_move(state)

        start_id = chess.square_name(mv.from_square)
        end_id = chess.square_name(mv.to_square)

        # Validate against local game-state rules before accepting.
        trial = state.copy()
        err = trial.try_apply_p2_move(start_id, end_id)
        if err is not None:
            if config.STOCKFISH_FALLBACK_TO_SIMPLE:
                return _simple_choose_p2_chess_move(state)
            return None

        start_sq = parse_square(start_id)
        end_sq = parse_square(end_id)
        is_capture = state.get(end_sq) != ChessPiece.EMPTY and state.get(start_sq) != ChessPiece.EMPTY
        return ChosenMove(start_id=start_id, end_id=end_id, is_capture=is_capture)
    except Exception:
        return _simple_choose_p2_chess_move(state)


def choose_p2_checkers_move(state: CheckersState) -> Optional[ChosenMove]:
    quiet_move: Optional[ChosenMove] = None

    for start_id in _all_square_ids():
        start_sq = parse_square(start_id)
        moving = state.get(start_sq)
        if moving not in (CheckersPiece.P2_MAN, CheckersPiece.P2_KING):
            continue

        for end_id in _all_square_ids():
            if end_id == start_id:
                continue

            trial = state.copy()
            err = trial.try_apply_p2_move(start_id, end_id)
            if err is not None:
                continue

            end_sq = parse_square(end_id)
            is_capture = abs(end_sq.file_index - start_sq.file_index) == 2
            candidate = ChosenMove(start_id=start_id, end_id=end_id, is_capture=is_capture)
            if is_capture:
                return candidate
            if quiet_move is None:
                quiet_move = candidate

    return quiet_move


def choose_p2_move(game: str, state: ChessState | CheckersState | ParcheesiState) -> Optional[ChosenMove]:
    if game == "chess":
        return choose_p2_chess_move(state)  # type: ignore[arg-type]
    if game == "checkers":
        return choose_p2_checkers_move(state)  # type: ignore[arg-type]
    if game == "parcheesi":
        return choose_p2_parcheesi_move(state)  # type: ignore[arg-type]
    raise ValueError(f"Unsupported game: {game}")


def choose_p2_parcheesi_move(state: ParcheesiState) -> Optional[ChosenMove]:
    if not state.rolls_remaining:
        state.roll_dice()

    quiet_move: Optional[ChosenMove] = None
    for piece, start_id, end_id in state.get_possible_moves(2):
        is_capture = state.move_is_capture(piece, end_id)
        candidate = ChosenMove(start_id=start_id, end_id=end_id, is_capture=is_capture)
        if is_capture:
            return candidate
        if quiet_move is None:
            quiet_move = candidate

    return quiet_move
