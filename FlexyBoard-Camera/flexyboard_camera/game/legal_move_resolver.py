from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Any

from flexyboard_camera.game.board_models import BoardCoord

try:
    import chess
except Exception:  # pragma: no cover - optional runtime dependency
    chess = None  # type: ignore[assignment]


@dataclass(frozen=True, slots=True)
class ResolvedStep:
    source: BoardCoord
    destination: BoardCoord

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": {"x": self.source.x, "y": self.source.y},
            "destination": {"x": self.destination.x, "y": self.destination.y},
        }


@dataclass(frozen=True, slots=True)
class ResolvedPlayer1Move:
    steps: list[ResolvedStep]
    capture: bool
    special: str | None
    resolver: str
    score: int
    matched_expected: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "steps": [step.to_dict() for step in self.steps],
            "capture": self.capture,
            "special": self.special,
            "resolver": self.resolver,
            "score": self.score,
            "matched_expected": self.matched_expected,
        }


@dataclass(frozen=True, slots=True)
class _ObservedMove:
    source: BoardCoord | None
    destination: BoardCoord | None
    capture: bool | None


def _coord_from_obj(raw: Any) -> BoardCoord | None:
    if not isinstance(raw, dict):
        return None
    if "x" not in raw or "y" not in raw:
        return None
    return BoardCoord(x=int(raw["x"]), y=int(raw["y"]))


def _parse_observed_move(obj: dict[str, Any]) -> _ObservedMove:
    return _ObservedMove(
        source=_coord_from_obj(obj.get("source")),
        destination=_coord_from_obj(obj.get("destination")),
        capture=obj.get("capture") if isinstance(obj.get("capture"), bool) else None,
    )


def _changed_coords(changed_squares: list[dict[str, Any]]) -> set[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    for item in changed_squares:
        if not isinstance(item, dict):
            continue
        x = item.get("x")
        y = item.get("y")
        if isinstance(x, int) and isinstance(y, int):
            out.add((x, y))
    return out


def _board_coord_to_square_id(coord: BoardCoord) -> str:
    files = "abcdefgh"
    ranks = "12345678"
    return files[coord.x] + ranks[coord.y]


def _square_id_to_board_coord(square_id: str) -> BoardCoord:
    s = square_id.strip().lower()
    files = "abcdefgh"
    ranks = "12345678"
    if len(s) != 2 or s[0] not in files or s[1] not in ranks:
        raise ValueError(f"invalid square id: {square_id!r}")
    return BoardCoord(x=files.index(s[0]), y=ranks.index(s[1]))


def _score_candidate(
    *,
    observed: _ObservedMove,
    observed_changed: set[tuple[int, int]],
    expected_changed: set[tuple[int, int]],
    source: BoardCoord,
    destination: BoardCoord,
    is_capture: bool,
) -> int:
    missing = len(expected_changed - observed_changed)
    extra = len(observed_changed - expected_changed)
    score = (missing * 4) + extra

    if observed.source is not None and observed.source != source:
        score += 3
    if observed.destination is not None and observed.destination != destination:
        score += 3
    if observed.capture is True and not is_capture:
        score += 3
    if observed.capture is False and is_capture:
        score += 2
    return score


class _ChessResolver:
    def __init__(self) -> None:
        if chess is None:
            raise RuntimeError("python-chess is required for chess move resolution")
        self._board = chess.Board()

    @staticmethod
    def _square_set_diff(before: chess.Board, after: chess.Board) -> set[tuple[int, int]]:
        changed: set[tuple[int, int]] = set()
        for sq in chess.SQUARES:
            if before.piece_at(sq) != after.piece_at(sq):
                changed.add((chess.square_file(sq), chess.square_rank(sq)))
        return changed

    @staticmethod
    def _move_special(board_before: chess.Board, move: chess.Move) -> str | None:
        if board_before.is_castling(move):
            return "castling"
        if board_before.is_en_passant(move):
            return "en_passant"
        if move.promotion is not None:
            return "promotion"
        return None

    def resolve_player1(
        self,
        observed_move_obj: dict[str, Any],
        changed_squares: list[dict[str, Any]],
    ) -> ResolvedPlayer1Move | None:
        observed = _parse_observed_move(observed_move_obj)
        observed_changed = _changed_coords(changed_squares)

        if self._board.turn != chess.WHITE:
            # Drift recovery: if out of phase, force back to P1 turn so we still
            # choose a legal white move against the current board.
            self._board.turn = chess.WHITE

        best: tuple[int, chess.Move, set[tuple[int, int]], bool, str | None] | None = None
        for move in self._board.legal_moves:
            if self._board.color_at(move.from_square) != chess.WHITE:
                continue
            before = self._board.copy(stack=False)
            after = self._board.copy(stack=False)
            is_capture = after.is_capture(move)
            special = self._move_special(after, move)
            after.push(move)

            expected_changed = self._square_set_diff(before, after)
            source = BoardCoord(
                x=chess.square_file(move.from_square),
                y=chess.square_rank(move.from_square),
            )
            destination = BoardCoord(
                x=chess.square_file(move.to_square),
                y=chess.square_rank(move.to_square),
            )
            score = _score_candidate(
                observed=observed,
                observed_changed=observed_changed,
                expected_changed=expected_changed,
                source=source,
                destination=destination,
                is_capture=is_capture,
            )
            # Prefer queen promotion tie-break because UI protocol is from/to only.
            if move.promotion is not None and move.promotion != chess.QUEEN:
                score += 1

            if best is None or score < best[0]:
                best = (score, move, expected_changed, is_capture, special)

        if best is None:
            return None

        score, move, expected_changed, is_capture, special = best
        self._board.push(move)

        step = ResolvedStep(
            source=BoardCoord(x=chess.square_file(move.from_square), y=chess.square_rank(move.from_square)),
            destination=BoardCoord(x=chess.square_file(move.to_square), y=chess.square_rank(move.to_square)),
        )
        return ResolvedPlayer1Move(
            steps=[step],
            capture=is_capture,
            special=special,
            resolver="chess_legal_match",
            score=score,
            matched_expected=score == 0 and expected_changed == observed_changed,
        )

    def apply_player2(self, from_square: str, to_square: str) -> None:
        if self._board.turn != chess.BLACK:
            self._board.turn = chess.BLACK

        from_sq = chess.parse_square(from_square)
        to_sq = chess.parse_square(to_square)
        candidates = [
            move
            for move in self._board.legal_moves
            if move.from_square == from_sq and move.to_square == to_sq
        ]
        if not candidates:
            raise ValueError(f"illegal p2 move for current board: {from_square}->{to_square}")

        # Match UI behavior: default to queen promotion if ambiguous.
        selected = candidates[0]
        for move in candidates:
            if move.promotion == chess.QUEEN:
                selected = move
                break
        self._board.push(selected)


class _CheckersPiece(IntEnum):
    EMPTY = 0
    P1_MAN = 1
    P1_KING = 2
    P2_MAN = 3
    P2_KING = 4


def _is_p1(piece: _CheckersPiece) -> bool:
    return piece in (_CheckersPiece.P1_MAN, _CheckersPiece.P1_KING)


def _is_p2(piece: _CheckersPiece) -> bool:
    return piece in (_CheckersPiece.P2_MAN, _CheckersPiece.P2_KING)


def _is_dark_square(x: int, y: int) -> bool:
    return ((x + y) % 2) == 0


@dataclass(frozen=True, slots=True)
class _CheckersHop:
    src: BoardCoord
    dst: BoardCoord
    captured: BoardCoord | None


class _CheckersState:
    def __init__(self) -> None:
        self.board: list[list[_CheckersPiece]] = [[_CheckersPiece.EMPTY] * 8 for _ in range(8)]
        self._place_initial()

    def copy(self) -> _CheckersState:
        state = _CheckersState.__new__(_CheckersState)
        state.board = [row[:] for row in self.board]
        return state

    def _place_initial(self) -> None:
        for y in range(8):
            for x in range(8):
                if not _is_dark_square(x, y):
                    continue
                if y <= 2:
                    self.board[y][x] = _CheckersPiece.P1_MAN
                elif y >= 5:
                    self.board[y][x] = _CheckersPiece.P2_MAN

    def get(self, coord: BoardCoord) -> _CheckersPiece:
        return self.board[coord.y][coord.x]

    def set(self, coord: BoardCoord, piece: _CheckersPiece) -> None:
        self.board[coord.y][coord.x] = piece

    def _dirs_for_piece(self, piece: _CheckersPiece, side: str) -> list[tuple[int, int]]:
        if piece in (_CheckersPiece.P1_KING, _CheckersPiece.P2_KING):
            return [(-1, -1), (1, -1), (-1, 1), (1, 1)]
        if side == "p1":
            return [(-1, 1), (1, 1)]
        return [(-1, -1), (1, -1)]

    def _in_bounds(self, x: int, y: int) -> bool:
        return 0 <= x <= 7 and 0 <= y <= 7

    def _capture_hops_from(self, coord: BoardCoord, side: str) -> list[_CheckersHop]:
        piece = self.get(coord)
        if side == "p1" and not _is_p1(piece):
            return []
        if side == "p2" and not _is_p2(piece):
            return []

        hops: list[_CheckersHop] = []
        for dx, dy in self._dirs_for_piece(piece, side):
            mx = coord.x + dx
            my = coord.y + dy
            tx = coord.x + (2 * dx)
            ty = coord.y + (2 * dy)
            if not self._in_bounds(tx, ty):
                continue
            mid = BoardCoord(mx, my)
            dst = BoardCoord(tx, ty)
            if self.get(dst) != _CheckersPiece.EMPTY:
                continue
            mid_piece = self.get(mid)
            if side == "p1" and _is_p2(mid_piece):
                hops.append(_CheckersHop(src=coord, dst=dst, captured=mid))
            if side == "p2" and _is_p1(mid_piece):
                hops.append(_CheckersHop(src=coord, dst=dst, captured=mid))
        return hops

    def _simple_hops_from(self, coord: BoardCoord, side: str) -> list[_CheckersHop]:
        piece = self.get(coord)
        if side == "p1" and not _is_p1(piece):
            return []
        if side == "p2" and not _is_p2(piece):
            return []
        hops: list[_CheckersHop] = []
        for dx, dy in self._dirs_for_piece(piece, side):
            tx = coord.x + dx
            ty = coord.y + dy
            if not self._in_bounds(tx, ty):
                continue
            dst = BoardCoord(tx, ty)
            if self.get(dst) == _CheckersPiece.EMPTY:
                hops.append(_CheckersHop(src=coord, dst=dst, captured=None))
        return hops

    def _has_any_capture(self, side: str) -> bool:
        for y in range(8):
            for x in range(8):
                coord = BoardCoord(x, y)
                piece = self.get(coord)
                if side == "p1" and not _is_p1(piece):
                    continue
                if side == "p2" and not _is_p2(piece):
                    continue
                if self._capture_hops_from(coord, side):
                    return True
        return False

    def _apply_hop(self, hop: _CheckersHop, side: str) -> bool:
        piece = self.get(hop.src)
        self.set(hop.src, _CheckersPiece.EMPTY)
        if hop.captured is not None:
            self.set(hop.captured, _CheckersPiece.EMPTY)
        self.set(hop.dst, piece)

        # American checkers convention: promotion ends jump sequence.
        promoted = False
        if side == "p1" and piece == _CheckersPiece.P1_MAN and hop.dst.y == 7:
            self.set(hop.dst, _CheckersPiece.P1_KING)
            promoted = True
        if side == "p2" and piece == _CheckersPiece.P2_MAN and hop.dst.y == 0:
            self.set(hop.dst, _CheckersPiece.P2_KING)
            promoted = True
        return promoted

    def _all_sequences_for_side(self, side: str) -> list[list[_CheckersHop]]:
        capture_only = self._has_any_capture(side)
        sequences: list[list[_CheckersHop]] = []

        for y in range(8):
            for x in range(8):
                start = BoardCoord(x, y)
                piece = self.get(start)
                if side == "p1" and not _is_p1(piece):
                    continue
                if side == "p2" and not _is_p2(piece):
                    continue

                if capture_only:
                    for first in self._capture_hops_from(start, side):
                        state_after_first = self.copy()
                        promoted = state_after_first._apply_hop(first, side)
                        if promoted:
                            sequences.append([first])
                            continue
                        continuations = state_after_first._capture_sequences_from(first.dst, side)
                        if continuations:
                            for tail in continuations:
                                sequences.append([first, *tail])
                        else:
                            sequences.append([first])
                else:
                    for hop in self._simple_hops_from(start, side):
                        sequences.append([hop])
        return sequences

    def _capture_sequences_from(self, coord: BoardCoord, side: str) -> list[list[_CheckersHop]]:
        sequences: list[list[_CheckersHop]] = []
        capture_hops = self._capture_hops_from(coord, side)
        if not capture_hops:
            return []

        for hop in capture_hops:
            next_state = self.copy()
            promoted = next_state._apply_hop(hop, side)
            if promoted:
                sequences.append([hop])
                continue
            tails = next_state._capture_sequences_from(hop.dst, side)
            if tails:
                for tail in tails:
                    sequences.append([hop, *tail])
            else:
                sequences.append([hop])
        return sequences

    def _piece_diff_set(self, after: _CheckersState) -> set[tuple[int, int]]:
        changed: set[tuple[int, int]] = set()
        for y in range(8):
            for x in range(8):
                if self.board[y][x] != after.board[y][x]:
                    changed.add((x, y))
        return changed

    def resolve_player1(
        self,
        observed_move_obj: dict[str, Any],
        changed_squares: list[dict[str, Any]],
    ) -> ResolvedPlayer1Move | None:
        observed = _parse_observed_move(observed_move_obj)
        observed_changed = _changed_coords(changed_squares)

        candidates = self._all_sequences_for_side("p1")
        if not candidates:
            return None

        best: tuple[int, list[_CheckersHop], bool, set[tuple[int, int]]] | None = None
        for sequence in candidates:
            trial = self.copy()
            capture = False
            for hop in sequence:
                if hop.captured is not None:
                    capture = True
                trial._apply_hop(hop, side="p1")

            expected_changed = self._piece_diff_set(trial)
            source = sequence[0].src
            destination = sequence[-1].dst
            score = _score_candidate(
                observed=observed,
                observed_changed=observed_changed,
                expected_changed=expected_changed,
                source=source,
                destination=destination,
                is_capture=capture,
            )
            if best is None or score < best[0]:
                best = (score, sequence, capture, expected_changed)

        if best is None:
            return None

        score, sequence, capture, expected_changed = best
        for hop in sequence:
            self._apply_hop(hop, side="p1")

        steps = [ResolvedStep(source=hop.src, destination=hop.dst) for hop in sequence]
        return ResolvedPlayer1Move(
            steps=steps,
            capture=capture,
            special="multi_jump" if len(sequence) > 1 else None,
            resolver="checkers_legal_match",
            score=score,
            matched_expected=score == 0 and expected_changed == observed_changed,
        )

    def apply_player2(self, from_square: str, to_square: str) -> None:
        src = _square_id_to_board_coord(from_square)
        dst = _square_id_to_board_coord(to_square)
        candidates = self._all_sequences_for_side("p2")
        if not candidates:
            raise ValueError("no legal p2 moves available")

        for sequence in candidates:
            first = sequence[0]
            if first.src == src and first.dst == dst:
                self._apply_hop(first, side="p2")
                return
        raise ValueError(f"illegal p2 move for current checkers board: {from_square}->{to_square}")


class Player1MoveResolver:
    """Stateful legal resolver for CV-observed Player-1 moves.

    This class keeps game state on the Pi side so we can map noisy CV changed
    squares to a legal move before forwarding to Software-GUI.
    """

    def __init__(self, game: str) -> None:
        normalized = str(game).strip().lower()
        self.game = normalized
        if normalized == "chess":
            try:
                self._engine = _ChessResolver()
            except Exception:
                # Graceful fallback if python-chess is unavailable at runtime.
                self._engine = None
        elif normalized == "checkers":
            self._engine = _CheckersState()
        else:
            self._engine = None

    def resolve_player1(
        self,
        observed_move_obj: dict[str, Any],
        changed_squares: list[dict[str, Any]],
    ) -> ResolvedPlayer1Move | None:
        if self._engine is None:
            return None
        if self.game == "chess":
            return self._engine.resolve_player1(observed_move_obj, changed_squares)
        if self.game == "checkers":
            return self._engine.resolve_player1(observed_move_obj, changed_squares)
        return None

    def apply_player2(self, from_square: str, to_square: str) -> None:
        if self._engine is None:
            return
        self._engine.apply_player2(from_square, to_square)

    @staticmethod
    def fallback_from_observed(observed_move_obj: dict[str, Any]) -> ResolvedPlayer1Move | None:
        observed = _parse_observed_move(observed_move_obj)
        if observed.source is None or observed.destination is None:
            return None
        step = ResolvedStep(source=observed.source, destination=observed.destination)
        return ResolvedPlayer1Move(
            steps=[step],
            capture=bool(observed.capture),
            special=None,
            resolver="fallback_observed",
            score=9999,
            matched_expected=False,
        )

    @staticmethod
    def step_to_square_pair(step: ResolvedStep) -> tuple[str, str]:
        return _board_coord_to_square_id(step.source), _board_coord_to_square_id(step.destination)
