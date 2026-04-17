"""Convert UI moves into collision-aware STM32 move-sequence text."""

from __future__ import annotations

from dataclasses import dataclass, field
import heapq
from pathlib import Path
from typing import Callable, Iterable, List, Sequence

import config
from checkers_state import CheckersState, Piece as CheckersPiece
from chess_state import ChessPiece, ChessState
from coords import Square, parse_square
from parcheesi_state import ParcheesiState

try:
    import chess
except Exception:  # pragma: no cover - optional dependency
    chess = None  # type: ignore[assignment]

BoardCoord = tuple[int, int]


@dataclass(frozen=True)
class BoardEndpoint:
    x: int
    y: int


@dataclass(frozen=True)
class PercentEndpoint:
    x_pct: float
    y_pct: float


Endpoint = BoardEndpoint | PercentEndpoint


@dataclass(frozen=True)
class GeneratedSequence:
    lines: List[str]
    capture_detected: bool
    temporary_relocations: int = 0
    fallback_direct_segments: int = 0
    capture_slots_used: List[str] = field(default_factory=list)
    manual_actions: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class TempRelocation:
    origin: BoardCoord
    slot: PercentEndpoint


@dataclass(frozen=True)
class CaptureSlotRecord:
    captured_side: str
    slot_index: int
    slot: PercentEndpoint
    piece_name: str


class CaptureInventory:
    """Tracks captured pieces by slot, side, and piece type for physical tray usage."""

    def __init__(self, game: str) -> None:
        self.game = game
        self._slots: dict[str, dict[int, str]] = {"p1": {}, "p2": {}}

    def clear(self) -> None:
        self._slots = {"p1": {}, "p2": {}}

    def _next_free_index(self, captured_side: str) -> int:
        used = self._slots[captured_side]
        idx = 0
        while idx in used:
            idx += 1
        return idx

    def add_captured_piece(self, captured_side: str, piece_name: str) -> CaptureSlotRecord:
        if captured_side not in {"p1", "p2"}:
            raise ValueError(f"invalid captured_side: {captured_side}")
        idx = self._next_free_index(captured_side)
        self._slots[captured_side][idx] = piece_name
        slot = _capture_slot(captured_side, idx)
        return CaptureSlotRecord(
            captured_side=captured_side,
            slot_index=idx,
            slot=slot,
            piece_name=piece_name,
        )

    def take_piece(self, captured_side: str, piece_name: str) -> CaptureSlotRecord | None:
        if captured_side not in {"p1", "p2"}:
            raise ValueError(f"invalid captured_side: {captured_side}")
        candidates = [
            idx for idx, name in self._slots[captured_side].items()
            if name == piece_name
        ]
        if not candidates:
            return None
        idx = min(candidates)
        stored = self._slots[captured_side].pop(idx)
        slot = _capture_slot(captured_side, idx)
        return CaptureSlotRecord(
            captured_side=captured_side,
            slot_index=idx,
            slot=slot,
            piece_name=stored,
        )

    def count(self, captured_side: str) -> int:
        if captured_side not in {"p1", "p2"}:
            raise ValueError(f"invalid captured_side: {captured_side}")
        return len(self._slots[captured_side])

    def to_summary_strings(self) -> list[str]:
        out: list[str] = []
        for captured_side in ("p1", "p2"):
            for idx in sorted(self._slots[captured_side]):
                piece_name = self._slots[captured_side][idx]
                slot = _capture_slot(captured_side, idx)
                out.append(
                    f"{captured_side}[{idx}]={slot.x_pct:.2f}%,{slot.y_pct:.2f}%:{piece_name}"
                )
        return out


def _board_ep(x: int, y: int) -> BoardEndpoint:
    return BoardEndpoint(x=x, y=y)


def _pct_ep(x_pct: float, y_pct: float) -> PercentEndpoint:
    return PercentEndpoint(x_pct=x_pct, y_pct=max(0.0, min(100.0, y_pct)))


def _endpoint_token(ep: Endpoint) -> str:
    if isinstance(ep, BoardEndpoint):
        return f"{ep.x},{ep.y}"
    return f"{ep.x_pct:.2f}%,{ep.y_pct:.2f}%"


def _capture_slot(captured_side: str, already_captured_count: int) -> PercentEndpoint:
    if captured_side not in {"p1", "p2"}:
        raise ValueError(f"invalid captured_side: {captured_side}")

    if config.CAPTURE_STAGING_MODE == "side_lane":
        x = config.CAPTURE_STAGING_X_PCT
        if captured_side == "p1":
            y = config.CAPTURE_STAGING_P1_BASE_Y_PCT + already_captured_count * config.CAPTURE_STAGING_STEP_Y_PCT
        else:
            y = config.CAPTURE_STAGING_P2_BASE_Y_PCT - already_captured_count * config.CAPTURE_STAGING_STEP_Y_PCT
        return _pct_ep(x, y)

    # Bottom-lane mode: piece-sized slots across the bottom strip.
    cols = max(1, int(config.CAPTURE_BOTTOM_COLUMNS))
    col = already_captured_count % cols
    row = already_captured_count // cols

    x = config.CAPTURE_BOTTOM_X_START_PCT + (col * config.CAPTURE_BOTTOM_X_STEP_PCT)
    if captured_side == "p2" and config.CAPTURE_BOTTOM_P2_REVERSE_X:
        x = config.CAPTURE_BOTTOM_X_START_PCT + ((cols - 1 - col) * config.CAPTURE_BOTTOM_X_STEP_PCT)

    if captured_side == "p1":
        y = config.CAPTURE_BOTTOM_P1_BASE_Y_PCT + (row * config.CAPTURE_BOTTOM_ROW_STEP_Y_PCT)
    else:
        y = config.CAPTURE_BOTTOM_P2_BASE_Y_PCT + (row * config.CAPTURE_BOTTOM_ROW_STEP_Y_PCT)

    return _pct_ep(x, y)


def _board_occupancy_from_chess(state_before: ChessState) -> set[BoardCoord]:
    occ: set[BoardCoord] = set()
    for y in range(8):
        for x in range(8):
            if state_before.board[y][x] != ChessPiece.EMPTY:
                occ.add((x, y))
    return occ


def _board_occupancy_from_checkers(state_before: CheckersState) -> set[BoardCoord]:
    occ: set[BoardCoord] = set()
    for y in range(8):
        for x in range(8):
            if state_before.board[y][x] != CheckersPiece.EMPTY:
                occ.add((x, y))
    return occ


def _board_occupancy_from_parcheesi(state_before: ParcheesiState) -> set[BoardCoord]:
    occ: set[BoardCoord] = set()
    for y in range(8):
        for x in range(8):
            if state_before.board[y][x] != CheckersPiece.EMPTY:
                occ.add((x, y))
    return occ


def _checkers_capture_mid_square(start_sq: Square, end_sq: Square) -> Square | None:
    if abs(end_sq.file_index - start_sq.file_index) != 2:
        return None
    if abs(end_sq.rank_index - start_sq.rank_index) != 2:
        return None
    mid_f = (start_sq.file_index + end_sq.file_index) // 2
    mid_r = (start_sq.rank_index + end_sq.rank_index) // 2
    return Square(mid_f, mid_r)


def _parcheesi_square_to_pct(square: Square) -> PercentEndpoint:
    x_min = config.P2_PARCHEESI_MIN_X_PCT
    x_max = config.P2_PARCHEESI_MAX_X_PCT
    y_min = config.P2_PARCHEESI_MIN_Y_PCT
    y_max = config.P2_PARCHEESI_MAX_Y_PCT

    span_x = max(1e-6, x_max - x_min)
    span_y = max(1e-6, y_max - y_min)
    cell_x = span_x / 8.0
    cell_y = span_y / 8.0

    fx = square.file_index
    fy = square.rank_index
    if config.P2_PARCHEESI_INVERT_X:
        fx = 7 - fx
    if config.P2_PARCHEESI_INVERT_Y:
        fy = 7 - fy

    x_pct = x_min + ((float(fx) + 0.5) * cell_x)
    y_pct = y_min + ((float(fy) + 0.5) * cell_y)
    return _pct_ep(x_pct, y_pct)


def _choose_legal_chess_move(state_before: ChessState, start_sq: Square, end_sq: Square) -> tuple[object | None, object | None]:
    if chess is None:
        return None, None

    board = state_before.to_python_board()
    from_sq = chess.square(start_sq.file_index, start_sq.rank_index)
    to_sq = chess.square(end_sq.file_index, end_sq.rank_index)
    candidates = [
        mv for mv in board.legal_moves if mv.from_square == from_sq and mv.to_square == to_sq
    ]
    if not candidates:
        return board, None

    selected = candidates[0]
    for mv in candidates:
        if mv.promotion == chess.QUEEN:
            selected = mv
            break
    return board, selected


def _board_coord_from_chess_square(square_index: int) -> BoardCoord:
    if chess is None:
        raise RuntimeError("python-chess unavailable")
    return (chess.square_file(square_index), chess.square_rank(square_index))


def _piece_name(piece: object) -> str:
    return str(getattr(piece, "name", piece))


def _chess_p2_piece_from_promotion(promotion_type: int | None) -> ChessPiece:
    if chess is None or promotion_type is None:
        return ChessPiece.P2_QUEEN
    if promotion_type == chess.KNIGHT:
        return ChessPiece.P2_KNIGHT
    if promotion_type == chess.BISHOP:
        return ChessPiece.P2_BISHOP
    if promotion_type == chess.ROOK:
        return ChessPiece.P2_ROOK
    if promotion_type == chess.QUEEN:
        return ChessPiece.P2_QUEEN
    return ChessPiece.P2_QUEEN


def _neighbors_4(coord: BoardCoord) -> Iterable[BoardCoord]:
    x, y = coord
    for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
        if 0 <= nx <= 7 and 0 <= ny <= 7:
            yield (nx, ny)


def _manhattan(a: BoardCoord, b: BoardCoord) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _dist_to_segment(p: BoardCoord, a: BoardCoord, b: BoardCoord) -> float:
    px, py = float(p[0]), float(p[1])
    ax, ay = float(a[0]), float(a[1])
    bx, by = float(b[0]), float(b[1])
    vx, vy = bx - ax, by - ay
    wx, wy = px - ax, py - ay
    seg_len2 = vx * vx + vy * vy
    if seg_len2 <= 1e-9:
        return ((px - ax) ** 2 + (py - ay) ** 2) ** 0.5
    t = (wx * vx + wy * vy) / seg_len2
    t = max(0.0, min(1.0, t))
    cx = ax + t * vx
    cy = ay + t * vy
    return ((px - cx) ** 2 + (py - cy) ** 2) ** 0.5


def _compress_path(path: Sequence[BoardCoord]) -> list[tuple[BoardEndpoint, BoardEndpoint]]:
    if len(path) < 2:
        return []

    segments: list[tuple[BoardEndpoint, BoardEndpoint]] = []
    seg_start = path[0]
    prev = path[0]
    prev_dx = path[1][0] - path[0][0]
    prev_dy = path[1][1] - path[0][1]

    for i in range(1, len(path)):
        cur = path[i]
        dx = cur[0] - prev[0]
        dy = cur[1] - prev[1]
        if dx != prev_dx or dy != prev_dy:
            segments.append((_board_ep(seg_start[0], seg_start[1]), _board_ep(prev[0], prev[1])))
            seg_start = prev
            prev_dx = dx
            prev_dy = dy
        prev = cur

    segments.append((_board_ep(seg_start[0], seg_start[1]), _board_ep(path[-1][0], path[-1][1])))
    return segments


class MotionPlanner:
    def __init__(
        self,
        occupied: set[BoardCoord],
        captured_count_p1: int = 0,
        captured_count_p2: int = 0,
        capture_inventory: CaptureInventory | None = None,
    ) -> None:
        self.occupied: set[BoardCoord] = set(occupied)
        self.lines: list[str] = []
        self._capture_inventory = capture_inventory
        self._captured_count: dict[str, int] = {
            "p1": captured_count_p1 if capture_inventory is None else capture_inventory.count("p1"),
            "p2": captured_count_p2 if capture_inventory is None else capture_inventory.count("p2"),
        }
        self._temp_relocations: list[TempRelocation] = []
        self._temp_slot_next_index = 0
        self._fallback_direct_segments = 0
        self._capture_slots_used: list[str] = []

    def build(self) -> GeneratedSequence:
        return GeneratedSequence(
            lines=self.lines[:],
            capture_detected=False,
            temporary_relocations=len(self._temp_relocations),
        )

    def capture_slot_next(self, captured_side: str, piece_name: str) -> PercentEndpoint:
        if self._capture_inventory is not None:
            rec = self._capture_inventory.add_captured_piece(captured_side, piece_name)
            self._capture_slots_used.append(
                f"{captured_side}[{rec.slot_index}]={rec.slot.x_pct:.2f}%,{rec.slot.y_pct:.2f}%:{piece_name}"
            )
            return rec.slot

        slot_index = self._captured_count[captured_side]
        slot = _capture_slot(captured_side, slot_index)
        self._captured_count[captured_side] += 1
        self._capture_slots_used.append(
            f"{captured_side}[{slot_index}]={slot.x_pct:.2f}%,{slot.y_pct:.2f}%:{piece_name}"
        )
        return slot

    def append_segment(self, src: Endpoint, dst: Endpoint) -> None:
        self.lines.append(f"{_endpoint_token(src)} -> {_endpoint_token(dst)}")

    def record_note(self, text: str) -> None:
        self._capture_slots_used.append(text)

    def move_board_piece_direct(self, start: BoardCoord, dst: BoardCoord) -> None:
        self.append_segment(_board_ep(start[0], start[1]), _board_ep(dst[0], dst[1]))
        if start in self.occupied:
            self.occupied.remove(start)
        self.occupied.add(dst)

    def _astar(self, start: BoardCoord, goal: BoardCoord) -> list[BoardCoord] | None:
        if start == goal:
            return [start]

        # Occupancy excludes current moving piece at start.
        blocked = self.occupied - {start}
        g_score: dict[BoardCoord, int] = {start: 0}
        came_from: dict[BoardCoord, BoardCoord] = {}
        open_heap: list[tuple[int, int, BoardCoord]] = []
        heapq.heappush(open_heap, (_manhattan(start, goal), 0, start))

        while open_heap:
            _, g, cur = heapq.heappop(open_heap)
            if cur == goal:
                path = [cur]
                while cur in came_from:
                    cur = came_from[cur]
                    path.append(cur)
                path.reverse()
                return path

            if g > g_score.get(cur, 10**9):
                continue

            for nxt in _neighbors_4(cur):
                if nxt in blocked and nxt != goal:
                    continue
                ng = g + 1
                if ng < g_score.get(nxt, 10**9):
                    g_score[nxt] = ng
                    came_from[nxt] = cur
                    heapq.heappush(open_heap, (ng + _manhattan(nxt, goal), ng, nxt))
        return None

    def _temp_slot(self, index: int) -> PercentEndpoint:
        x = config.TEMP_RELOCATE_X_PCT
        y = config.TEMP_RELOCATE_BASE_Y_PCT + (index * config.TEMP_RELOCATE_STEP_Y_PCT)
        return _pct_ep(x, y)

    def _side_file_for_slot(self, slot: PercentEndpoint) -> int:
        return 7 if slot.x_pct >= 50.0 else 0

    def _rank_hint_for_pct(self, y_pct: float) -> int:
        return max(0, min(7, int(round((y_pct / 100.0) * 7.0))))

    def _best_route_board_to_pct(self, start: BoardCoord, dst_pct: PercentEndpoint) -> tuple[list[BoardCoord], BoardCoord] | None:
        side_file = self._side_file_for_slot(dst_pct)
        rank_hint = self._rank_hint_for_pct(dst_pct.y_pct)
        candidate_edges: list[BoardCoord] = [(side_file, r) for r in range(8)]
        candidate_edges.sort(key=lambda p: abs(p[1] - rank_hint))

        best: tuple[list[BoardCoord], BoardCoord] | None = None
        for edge in candidate_edges:
            if edge in self.occupied and edge != start:
                continue
            route = self._astar(start, edge)
            if route is None:
                continue
            if best is None or len(route) < len(best[0]):
                best = (route, edge)
        return best

    def _best_route_pct_to_board(
        self, src_pct: PercentEndpoint, dst: BoardCoord
    ) -> tuple[BoardCoord, list[BoardCoord]] | None:
        side_file = self._side_file_for_slot(src_pct)
        rank_hint = self._rank_hint_for_pct(src_pct.y_pct)
        candidate_edges: list[BoardCoord] = [(side_file, r) for r in range(8)]
        candidate_edges.sort(key=lambda p: abs(p[1] - rank_hint) + _manhattan(p, dst))

        best: tuple[BoardCoord, list[BoardCoord]] | None = None
        for edge in candidate_edges:
            if edge in self.occupied and edge != dst:
                continue
            route = self._astar(edge, dst)
            if route is None:
                continue
            if best is None or len(route) < len(best[1]):
                best = (edge, route)
        return best

    def _select_blocker(
        self,
        anchor_start: BoardCoord,
        anchor_end: BoardCoord,
        protected: set[BoardCoord],
        attempted: set[BoardCoord],
    ) -> BoardCoord | None:
        candidates = [c for c in self.occupied if c not in protected and c not in attempted]
        if not candidates:
            return None

        candidates.sort(
            key=lambda c: (
                _dist_to_segment(c, anchor_start, anchor_end),
                _manhattan(c, anchor_start) + _manhattan(c, anchor_end),
            )
        )
        return candidates[0]

    def _move_blocker_to_temp(self, blocker: BoardCoord) -> bool:
        if self._temp_slot_next_index >= config.MAX_TEMP_RELOCATIONS:
            return False

        slot = self._temp_slot(self._temp_slot_next_index)
        route_and_edge = self._best_route_board_to_pct(blocker, slot)
        if route_and_edge is None:
            return False

        route, _ = route_and_edge
        for src_ep, dst_ep in _compress_path(route):
            self.append_segment(src_ep, dst_ep)
        self.append_segment(_board_ep(route[-1][0], route[-1][1]), slot)

        self.occupied.remove(blocker)
        self._temp_relocations.append(TempRelocation(origin=blocker, slot=slot))
        self._temp_slot_next_index += 1
        return True

    def _clear_until_route_exists(
        self,
        route_builder: Callable[[], object | None],
        anchor_start: BoardCoord,
        anchor_end: BoardCoord,
        protected: set[BoardCoord],
    ) -> None:
        attempted: set[BoardCoord] = set()
        for _ in range(config.MAX_TEMP_RELOCATIONS):
            if route_builder() is not None:
                return
            blocker = self._select_blocker(anchor_start, anchor_end, protected, attempted)
            if blocker is None:
                return
            attempted.add(blocker)
            if not self._move_blocker_to_temp(blocker):
                continue
        return

    def move_board_piece_to_pct(
        self,
        start: BoardCoord,
        dst_pct: PercentEndpoint,
        protected_extra: set[BoardCoord] | None = None,
    ) -> None:
        protected = {start}
        if protected_extra:
            protected |= protected_extra

        def build() -> tuple[list[BoardCoord], BoardCoord] | None:
            return self._best_route_board_to_pct(start, dst_pct)

        route_and_edge = build()
        if route_and_edge is None:
            anchor_end = (self._side_file_for_slot(dst_pct), self._rank_hint_for_pct(dst_pct.y_pct))
            self._clear_until_route_exists(build, start, anchor_end, protected)
            route_and_edge = build()
        if route_and_edge is None:
            self.append_segment(_board_ep(start[0], start[1]), dst_pct)
            self.occupied.remove(start)
            self._fallback_direct_segments += 1
            return

        route, _ = route_and_edge
        for src_ep, dst_ep in _compress_path(route):
            self.append_segment(src_ep, dst_ep)
        self.append_segment(_board_ep(route[-1][0], route[-1][1]), dst_pct)

        self.occupied.remove(start)

    def move_pct_piece_to_board(self, src_pct: PercentEndpoint, dst: BoardCoord) -> None:
        route_from_edge = self._best_route_pct_to_board(src_pct, dst)
        if route_from_edge is None:
            self.append_segment(src_pct, _board_ep(dst[0], dst[1]))
            self.occupied.add(dst)
            self._fallback_direct_segments += 1
            return

        edge, route = route_from_edge
        self.append_segment(src_pct, _board_ep(edge[0], edge[1]))
        for src_ep, dst_ep in _compress_path(route):
            self.append_segment(src_ep, dst_ep)

        self.occupied.add(dst)

    def move_board_piece_to_board(
        self,
        start: BoardCoord,
        dst: BoardCoord,
        protected_extra: set[BoardCoord] | None = None,
    ) -> None:
        protected = {start, dst}
        if protected_extra:
            protected |= protected_extra

        def build() -> list[BoardCoord] | None:
            return self._astar(start, dst)

        route = build()
        if route is None:
            self._clear_until_route_exists(build, start, dst, protected)
            route = build()
        if route is None:
            self.append_segment(_board_ep(start[0], start[1]), _board_ep(dst[0], dst[1]))
            self.occupied.remove(start)
            self.occupied.add(dst)
            self._fallback_direct_segments += 1
            return

        for src_ep, dst_ep in _compress_path(route):
            self.append_segment(src_ep, dst_ep)

        self.occupied.remove(start)
        self.occupied.add(dst)

    def restore_temp_blockers(self) -> None:
        if not config.RESTORE_TEMP_RELOCATIONS:
            return
        for relocation in reversed(self._temp_relocations):
            self.move_pct_piece_to_board(relocation.slot, relocation.origin)
        self._temp_relocations.clear()


def generate_chess_p2_sequence(
    state_before: ChessState,
    start_id: str,
    end_id: str,
    capture_inventory: CaptureInventory | None = None,
) -> GeneratedSequence:
    start_sq = parse_square(start_id)
    end_sq = parse_square(end_id)
    start = (start_sq.file_index, start_sq.rank_index)
    end = (end_sq.file_index, end_sq.rank_index)

    planner = MotionPlanner(
        occupied=_board_occupancy_from_chess(state_before),
        captured_count_p1=len(state_before.captured_by_p2),
        captured_count_p2=len(state_before.captured_by_p1),
        capture_inventory=capture_inventory,
    )

    capture = False
    manual_actions: list[str] = []
    board_before, legal_move = _choose_legal_chess_move(state_before, start_sq, end_sq)

    if chess is not None and board_before is not None and legal_move is not None:
        is_capture = bool(board_before.is_capture(legal_move))
        is_en_passant = bool(board_before.is_en_passant(legal_move))
        is_castling = bool(board_before.is_castling(legal_move))
        is_promotion = legal_move.promotion is not None

        if is_en_passant:
            offset = -8 if board_before.turn == chess.WHITE else 8
            captured_sq = legal_move.to_square + offset
            captured_coord = _board_coord_from_chess_square(captured_sq)
            if captured_coord in planner.occupied:
                capture = True
                captured_piece = state_before.get(Square(captured_coord[0], captured_coord[1]))
                slot = planner.capture_slot_next("p1", _piece_name(captured_piece))
                planner.move_board_piece_to_pct(captured_coord, slot, protected_extra={start, end})
        elif is_capture and end in planner.occupied:
            capture = True
            captured_piece = state_before.get(end_sq)
            slot = planner.capture_slot_next("p1", _piece_name(captured_piece))
            planner.move_board_piece_to_pct(end, slot, protected_extra={start})

        if is_castling:
            king_src = _board_coord_from_chess_square(legal_move.from_square)
            king_dst = _board_coord_from_chess_square(legal_move.to_square)
            rook_rank = king_src[1]
            if king_dst[0] > king_src[0]:
                rook_src = (7, rook_rank)
                rook_dst = (5, rook_rank)
            else:
                rook_src = (0, rook_rank)
                rook_dst = (3, rook_rank)

            # Castling is a known chess special-case: emit explicit direct segments
            # so we do not relocate unrelated back-rank pieces while searching routes.
            planner.move_board_piece_direct(king_src, king_dst)
            if rook_src in planner.occupied:
                planner.move_board_piece_direct(rook_src, rook_dst)
        else:
            planner.move_board_piece_to_board(start, end)

        if is_promotion and config.P2_PROMOTION_REPLACE_PHYSICAL:
            promotion_bin = _pct_ep(config.P2_PROMOTION_STAGING_X_PCT, config.P2_PROMOTION_STAGING_Y_PCT)
            reserve_slot = _pct_ep(config.P2_PROMOTION_RESERVE_X_PCT, config.P2_PROMOTION_RESERVE_Y_PCT)
            promotion_piece = _chess_p2_piece_from_promotion(legal_move.promotion)
            if capture_inventory is not None:
                pulled = capture_inventory.take_piece("p2", _piece_name(promotion_piece))
                if pulled is not None:
                    reserve_slot = pulled.slot
                    planner.record_note(
                        f"promotion_source={pulled.captured_side}[{pulled.slot_index}]="
                        f"{pulled.slot.x_pct:.2f}%,{pulled.slot.y_pct:.2f}%:{pulled.piece_name}"
                    )
                else:
                    if config.P2_PROMOTION_REQUIRE_MANUAL_IF_MISSING:
                        planner.record_note(
                            f"promotion_source=manual_required:{_piece_name(promotion_piece)}"
                        )
                        manual_actions.append(
                            "PROMOTION_MANUAL_REQUIRED: replace promoted pawn at "
                            f"{end_id} with {_piece_name(promotion_piece)} in physical tray/board."
                        )
                    else:
                        planner.record_note(
                            f"promotion_source=reserve_config:{_piece_name(promotion_piece)}"
                        )
            if not manual_actions:
                planner.move_board_piece_to_pct(end, promotion_bin)
                planner.move_pct_piece_to_board(reserve_slot, end)
    else:
        # Fallback if python-chess is unavailable or move lookup fails.
        target = state_before.get(end_sq)
        if ChessPiece.P1_PAWN <= target <= ChessPiece.P1_KING:
            capture = True
            slot = planner.capture_slot_next("p1", _piece_name(target))
            planner.move_board_piece_to_pct(end, slot, protected_extra={start})

        planner.move_board_piece_to_board(start, end)

    temp_count = len(planner._temp_relocations)
    planner.restore_temp_blockers()

    return GeneratedSequence(
        lines=planner.lines,
        capture_detected=capture,
        temporary_relocations=temp_count,
        fallback_direct_segments=planner._fallback_direct_segments,
        capture_slots_used=planner._capture_slots_used[:],
        manual_actions=manual_actions,
    )


def generate_checkers_p2_sequence(
    state_before: CheckersState,
    start_id: str,
    end_id: str,
    capture_inventory: CaptureInventory | None = None,
) -> GeneratedSequence:
    start_sq = parse_square(start_id)
    end_sq = parse_square(end_id)
    start = (start_sq.file_index, start_sq.rank_index)
    end = (end_sq.file_index, end_sq.rank_index)

    planner = MotionPlanner(
        occupied=_board_occupancy_from_checkers(state_before),
        captured_count_p1=len(state_before.captured_by_p2),
        captured_count_p2=len(state_before.captured_by_p1),
        capture_inventory=capture_inventory,
    )

    capture = False
    manual_actions: list[str] = []
    mid = _checkers_capture_mid_square(start_sq, end_sq)
    if mid is not None:
        jumped_piece = state_before.get(mid)
        if jumped_piece in (CheckersPiece.P1_MAN, CheckersPiece.P1_KING):
            capture = True
            mid_coord = (mid.file_index, mid.rank_index)
            slot = planner.capture_slot_next("p1", _piece_name(jumped_piece))
            planner.move_board_piece_to_pct(mid_coord, slot, protected_extra={start, end})

    planner.move_board_piece_to_board(start, end)
    temp_count = len(planner._temp_relocations)
    planner.restore_temp_blockers()

    return GeneratedSequence(
        lines=planner.lines,
        capture_detected=capture,
        temporary_relocations=temp_count,
        fallback_direct_segments=planner._fallback_direct_segments,
        capture_slots_used=planner._capture_slots_used[:],
        manual_actions=manual_actions,
    )


def generate_parcheesi_p2_sequence(
    state_before: ParcheesiState,
    start_id: str,
    end_id: str,
    capture_inventory: CaptureInventory | None = None,
) -> GeneratedSequence:
    start_sq = parse_square(start_id)
    end_sq = parse_square(end_id)

    start_pct = _parcheesi_square_to_pct(start_sq)
    end_pct = _parcheesi_square_to_pct(end_sq)

    capture = False
    capture_slots_used: list[str] = []

    target = state_before.get(end_sq)
    lines: list[str] = []
    if target in (CheckersPiece.P1_MAN, CheckersPiece.P1_KING):
        capture = True
        if capture_inventory is not None:
            rec = capture_inventory.add_captured_piece("p1", _piece_name(target))
            slot = rec.slot
            capture_slots_used.append(
                f"{rec.captured_side}[{rec.slot_index}]={rec.slot.x_pct:.2f}%,{rec.slot.y_pct:.2f}%:{rec.piece_name}"
            )
        else:
            slot_index = len(state_before.captured_by_p2)
            slot = _capture_slot("p1", slot_index)
            capture_slots_used.append(
                f"p1[{slot_index}]={slot.x_pct:.2f}%,{slot.y_pct:.2f}%:{_piece_name(target)}"
            )
        lines.append(f"{_endpoint_token(end_pct)} -> {_endpoint_token(slot)}")

    lines.append(f"{_endpoint_token(start_pct)} -> {_endpoint_token(end_pct)}")

    return GeneratedSequence(
        lines=lines,
        capture_detected=capture,
        temporary_relocations=0,
        fallback_direct_segments=0,
        capture_slots_used=capture_slots_used,
        manual_actions=[],
    )


def generate_p2_sequence(
    game: str,
    state_before: ChessState | CheckersState | ParcheesiState,
    start_id: str,
    end_id: str,
    capture_inventory: CaptureInventory | None = None,
) -> GeneratedSequence:
    if game == "chess":
        return generate_chess_p2_sequence(  # type: ignore[arg-type]
            state_before,
            start_id,
            end_id,
            capture_inventory=capture_inventory,
        )
    if game == "checkers":
        return generate_checkers_p2_sequence(  # type: ignore[arg-type]
            state_before,
            start_id,
            end_id,
            capture_inventory=capture_inventory,
        )
    if game == "parcheesi":
        return generate_parcheesi_p2_sequence(  # type: ignore[arg-type]
            state_before,
            start_id,
            end_id,
            capture_inventory=capture_inventory,
        )
    raise ValueError(f"Unsupported game: {game}")


def write_sequence_file(
    path: str | Path,
    lines: list[str],
    game: str,
    start_id: str,
    end_id: str,
    capture_slots_used: list[str] | None = None,
    capture_inventory_summary: list[str] | None = None,
    manual_actions: list[str] | None = None,
) -> Path:
    out = Path(path).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    header = [
        "# Auto-generated by Software-GUI motion planner",
        f"# game={game} p2_move={start_id}->{end_id}",
        "# Format: source -> dest (board: x,y  |  off-board: x%,y%)",
        "# Planner: capture staging + collision-aware routing + temporary blocker relocation",
    ]
    if capture_slots_used:
        header.append("# capture_slots_used: " + ", ".join(capture_slots_used))
    if capture_inventory_summary:
        header.append("# capture_inventory: " + ", ".join(capture_inventory_summary))
    if manual_actions:
        header.append("# manual_actions: " + " | ".join(manual_actions))
    out.write_text("\n".join([*header, *lines]) + "\n", encoding="utf-8")
    return out
