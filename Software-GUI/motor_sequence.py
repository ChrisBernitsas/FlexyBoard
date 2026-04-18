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
    slot: Endpoint


@dataclass(frozen=True)
class CaptureSlotRecord:
    captured_side: str
    slot_index: int
    slot: PercentEndpoint
    piece_name: str


@dataclass(frozen=True)
class BoardExit:
    edge: BoardCoord
    outside: PercentEndpoint


@dataclass(frozen=True)
class BoardToPctPlan:
    route: tuple[BoardCoord, ...]
    blockers: frozenset[BoardCoord]
    board_exit: BoardExit
    offboard_path: tuple[PercentEndpoint, ...]


@dataclass(frozen=True)
class PctToBoardPlan:
    board_exit: BoardExit
    offboard_path: tuple[PercentEndpoint, ...]
    route: tuple[BoardCoord, ...]
    blockers: frozenset[BoardCoord]


@dataclass(frozen=True)
class BoardToBoardPlan:
    route: tuple[BoardCoord, ...]
    blockers: frozenset[BoardCoord]


class CaptureInventory:
    """Tracks captured pieces by slot, side, and piece type for physical tray usage."""

    def __init__(self, game: str) -> None:
        self.game = game
        # Global slot index -> (captured side, piece name). A shared sequence
        # prevents P1/P2 captures from colliding in the same physical tray slot.
        self._slots: dict[int, tuple[str, str]] = {}

    def clear(self) -> None:
        self._slots = {}

    def _next_free_index(self) -> int:
        idx = 0
        while idx in self._slots:
            idx += 1
        return idx

    def add_captured_piece(self, captured_side: str, piece_name: str) -> CaptureSlotRecord:
        if captured_side not in {"p1", "p2"}:
            raise ValueError(f"invalid captured_side: {captured_side}")
        idx = self._next_free_index()
        self._slots[idx] = (captured_side, piece_name)
        slot = _capture_slot_by_index(idx)
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
            idx for idx, (side, name) in self._slots.items()
            if side == captured_side and name == piece_name
        ]
        if not candidates:
            return None
        idx = min(candidates)
        stored_side, stored = self._slots.pop(idx)
        slot = _capture_slot_by_index(idx)
        return CaptureSlotRecord(
            captured_side=stored_side,
            slot_index=idx,
            slot=slot,
            piece_name=stored,
        )

    def count(self, captured_side: str) -> int:
        if captured_side not in {"p1", "p2"}:
            raise ValueError(f"invalid captured_side: {captured_side}")
        return sum(1 for side, _piece_name in self._slots.values() if side == captured_side)

    def to_summary_strings(self) -> list[str]:
        out: list[str] = []
        for idx in sorted(self._slots):
            captured_side, piece_name = self._slots[idx]
            slot = _capture_slot_by_index(idx)
            out.append(
                f"{captured_side}[{idx}]={slot.x_pct:.2f}%,{slot.y_pct:.2f}%:{piece_name}"
            )
        return out

    def occupied_records(self) -> list[CaptureSlotRecord]:
        out: list[CaptureSlotRecord] = []
        for idx in sorted(self._slots):
            captured_side, piece_name = self._slots[idx]
            out.append(
                CaptureSlotRecord(
                    captured_side=captured_side,
                    slot_index=idx,
                    slot=_capture_slot_by_index(idx),
                    piece_name=piece_name,
                )
            )
        return out


def _board_ep(x: int, y: int) -> BoardEndpoint:
    return BoardEndpoint(x=x, y=y)


def _pct_ep(x_pct: float, y_pct: float) -> PercentEndpoint:
    return PercentEndpoint(x_pct=x_pct, y_pct=max(0.0, min(100.0, y_pct)))


def _pct_key(ep: PercentEndpoint) -> tuple[float, float]:
    return (round(ep.x_pct, 2), round(ep.y_pct, 2))


def _pct_distance(a: PercentEndpoint, b: PercentEndpoint) -> float:
    return ((a.x_pct - b.x_pct) ** 2 + (a.y_pct - b.y_pct) ** 2) ** 0.5


def _pct_point_to_segment_distance(p: PercentEndpoint, a: PercentEndpoint, b: PercentEndpoint) -> float:
    ax, ay = a.x_pct, a.y_pct
    bx, by = b.x_pct, b.y_pct
    px, py = p.x_pct, p.y_pct
    vx, vy = bx - ax, by - ay
    wx, wy = px - ax, py - ay
    seg_len2 = vx * vx + vy * vy
    if seg_len2 <= 1e-9:
        return _pct_distance(p, a)
    t = (wx * vx + wy * vy) / seg_len2
    t = max(0.0, min(1.0, t))
    closest = PercentEndpoint(ax + t * vx, ay + t * vy)
    return _pct_distance(p, closest)


def _capture_slot_by_index(slot_index: int) -> PercentEndpoint:
    """Return the physical capture slot for a global capture index.

    Bottom slots are used first because they have the most usable space. After
    the bottom strip fills, captures overflow to the top strip.
    """
    if slot_index < 0:
        raise ValueError(f"invalid capture slot index: {slot_index}")

    bottom_cols = max(1, int(config.CAPTURE_BOTTOM_COLUMNS))
    bottom_rows = max(1, int(config.CAPTURE_BOTTOM_ROWS))
    bottom_capacity = bottom_cols * bottom_rows

    if slot_index < bottom_capacity:
        col = slot_index % bottom_cols
        row = slot_index // bottom_cols
        x = config.CAPTURE_BOTTOM_X_START_PCT + (col * config.CAPTURE_BOTTOM_X_STEP_PCT)
        y = config.CAPTURE_BOTTOM_BASE_Y_PCT - (row * config.CAPTURE_BOTTOM_ROW_STEP_Y_PCT)
        return _pct_ep(x, y)

    overflow_index = slot_index - bottom_capacity
    top_cols = max(1, int(config.CAPTURE_TOP_OVERFLOW_COLUMNS))
    col = overflow_index % top_cols
    row = overflow_index // top_cols
    x = config.CAPTURE_TOP_OVERFLOW_X_START_PCT + (col * config.CAPTURE_TOP_OVERFLOW_X_STEP_PCT)
    y = config.CAPTURE_TOP_OVERFLOW_Y_PCT + (row * config.CAPTURE_BOTTOM_ROW_STEP_Y_PCT)
    return _pct_ep(x, y)


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

    # Fallback path when no CaptureInventory is supplied. Reserve separate
    # global ranges for each side to avoid slot collisions.
    side_offset = 0 if captured_side == "p1" else 15
    return _capture_slot_by_index(side_offset + already_captured_count)


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
    # Parcheesi uses percent endpoints, not chess/checkers board-cell endpoints.
    # Collision-aware board routing is therefore not applied here yet.
    return set()


def _checkers_capture_mid_square(start_sq: Square, end_sq: Square) -> Square | None:
    if abs(end_sq.file_index - start_sq.file_index) != 2:
        return None
    if abs(end_sq.rank_index - start_sq.rank_index) != 2:
        return None
    mid_f = (start_sq.file_index + end_sq.file_index) // 2
    mid_r = (start_sq.rank_index + end_sq.rank_index) // 2
    return Square(mid_f, mid_r)


def _parcheesi_location_to_pct(location_id: str) -> PercentEndpoint:
    x_min = config.P2_PARCHEESI_MIN_X_PCT
    x_max = config.P2_PARCHEESI_MAX_X_PCT
    y_min = config.P2_PARCHEESI_MIN_Y_PCT
    y_max = config.P2_PARCHEESI_MAX_Y_PCT

    span_x = max(1e-6, x_max - x_min)
    span_y = max(1e-6, y_max - y_min)

    gx, gy = ParcheesiState.location_id_to_grid(location_id)
    fx = gx / float(ParcheesiState.GRID_MAX)
    fy = gy / float(ParcheesiState.GRID_MAX)
    if config.P2_PARCHEESI_INVERT_X:
        fx = 1.0 - fx
    if config.P2_PARCHEESI_INVERT_Y:
        fy = 1.0 - fy

    x_pct = x_min + (fx * span_x)
    y_pct = y_min + (fy * span_y)
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


def _neighbors_8(coord: BoardCoord) -> Iterable[BoardCoord]:
    x, y = coord
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            nx, ny = x + dx, y + dy
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


def _step_collision_cells(a: BoardCoord, b: BoardCoord) -> set[BoardCoord]:
    """Return cells whose occupied pieces could be struck moving one grid step."""
    ax, ay = a
    bx, by = b
    dx = bx - ax
    dy = by - ay
    cells: set[BoardCoord] = {b}
    if abs(dx) == 1 and abs(dy) == 1:
        # A diagonal drag sweeps near both orthogonal side squares.
        cells.add((ax + dx, ay))
        cells.add((ax, ay + dy))
    return {cell for cell in cells if 0 <= cell[0] <= 7 and 0 <= cell[1] <= 7}


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
        self.offboard_occupied: dict[tuple[float, float], str] = {}
        if capture_inventory is not None:
            for rec in capture_inventory.occupied_records():
                self._mark_offboard_occupied(rec.slot, f"captured:{rec.captured_side}:{rec.piece_name}")
        else:
            for idx in range(captured_count_p1):
                self._mark_offboard_occupied(_capture_slot("p1", idx), "captured:p1")
            for idx in range(captured_count_p2):
                self._mark_offboard_occupied(_capture_slot("p2", idx), "captured:p2")

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

    def _mark_offboard_occupied(self, slot: PercentEndpoint, label: str) -> None:
        self.offboard_occupied[_pct_key(slot)] = label

    def _unmark_offboard_occupied(self, slot: PercentEndpoint) -> None:
        self.offboard_occupied.pop(_pct_key(slot), None)

    def _is_offboard_occupied(self, slot: PercentEndpoint) -> bool:
        return _pct_key(slot) in self.offboard_occupied

    def move_board_piece_direct(self, start: BoardCoord, dst: BoardCoord) -> None:
        self.append_segment(_board_ep(start[0], start[1]), _board_ep(dst[0], dst[1]))
        if start in self.occupied:
            self.occupied.remove(start)
        self.occupied.add(dst)

    def _astar(
        self,
        start: BoardCoord,
        goal: BoardCoord,
        extra_blocked: set[BoardCoord] | None = None,
    ) -> list[BoardCoord] | None:
        if start == goal:
            return [start]

        # Occupancy excludes current moving piece at start.
        blocked = (self.occupied | (extra_blocked or set())) - {start, goal}
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
                if nxt in blocked:
                    continue
                ng = g + 1
                if ng < g_score.get(nxt, 10**9):
                    g_score[nxt] = ng
                    came_from[nxt] = cur
                    heapq.heappush(open_heap, (ng + _manhattan(nxt, goal), ng, nxt))
        return None

    def _astar_min_blockers(
        self,
        start: BoardCoord,
        goal: BoardCoord,
        protected: set[BoardCoord],
    ) -> tuple[list[BoardCoord], set[BoardCoord]] | None:
        if start == goal:
            return [start], set()

        # Weighted search over (position, unique-blockers-seen). This minimizes
        # the number of physical pieces that must be moved, then path length.
        start_blockers: frozenset[BoardCoord] = frozenset()
        start_key = (start, start_blockers)
        best_path_len: dict[tuple[BoardCoord, frozenset[BoardCoord]], int] = {start_key: 0}
        came_from: dict[
            tuple[BoardCoord, frozenset[BoardCoord]],
            tuple[BoardCoord, frozenset[BoardCoord]],
        ] = {}
        frontier: dict[BoardCoord, list[tuple[frozenset[BoardCoord], int]]] = {
            start: [(start_blockers, 0)]
        }
        open_heap: list[tuple[int, int, int, int, BoardCoord, frozenset[BoardCoord]]] = []
        counter = 0
        heapq.heappush(open_heap, (0, _manhattan(start, goal), 0, counter, start, start_blockers))

        def dominated(coord: BoardCoord, blockers: frozenset[BoardCoord], path_len: int) -> bool:
            for known_blockers, known_len in frontier.get(coord, []):
                if known_len <= path_len and known_blockers.issubset(blockers):
                    return True
            return False

        def remember(coord: BoardCoord, blockers: frozenset[BoardCoord], path_len: int) -> None:
            kept = [
                (known_blockers, known_len)
                for known_blockers, known_len in frontier.get(coord, [])
                if not (path_len <= known_len and blockers.issubset(known_blockers))
            ]
            kept.append((blockers, path_len))
            frontier[coord] = kept

        while open_heap:
            _blocker_cost, _priority, path_len, _counter, cur, blockers = heapq.heappop(open_heap)
            cur_key = (cur, blockers)
            if path_len != best_path_len.get(cur_key, 10**9):
                continue
            if cur == goal:
                path = [cur_key[0]]
                while cur_key in came_from:
                    cur_key = came_from[cur_key]
                    path.append(cur_key[0])
                path.reverse()
                return path, set(blockers)

            for nxt in _neighbors_8(cur):
                step_cells = _step_collision_cells(cur, nxt)
                if any(
                    cell in self.occupied and cell in protected and cell not in {start, goal}
                    for cell in step_cells
                ):
                    continue
                step_blockers = frozenset(
                    cell
                    for cell in step_cells
                    if cell in self.occupied and cell not in protected
                )
                next_blockers = blockers | step_blockers
                next_path_len = path_len + 1
                next_key = (nxt, next_blockers)
                if next_path_len >= best_path_len.get(next_key, 10**9):
                    continue
                if dominated(nxt, next_blockers, next_path_len):
                    continue

                best_path_len[next_key] = next_path_len
                came_from[next_key] = cur_key
                remember(nxt, next_blockers, next_path_len)
                heuristic = max(abs(nxt[0] - goal[0]), abs(nxt[1] - goal[1]))
                counter += 1
                heapq.heappush(
                    open_heap,
                    (
                        len(next_blockers),
                        next_path_len + heuristic,
                        next_path_len,
                        counter,
                        nxt,
                        next_blockers,
                    ),
                )
        return None

    def _temp_slot(self, index: int) -> PercentEndpoint:
        side_slots = max(1, int(config.TEMP_RELOCATE_SIDE_SLOTS))
        side_index = index // side_slots
        slot_in_side = index % side_slots
        if side_index % 2 == 0:
            x = config.TEMP_RELOCATE_X_PCT
        else:
            x = config.TEMP_RELOCATE_LEFT_X_PCT
        y = config.TEMP_RELOCATE_BASE_Y_PCT + (slot_in_side * config.TEMP_RELOCATE_STEP_Y_PCT)
        return _pct_ep(x, y)

    def _rank_hint_from_motor_x(self, x_pct: float) -> int:
        # Motor X=0 is game rank 8, Motor X=100 is game rank 1.
        return max(0, min(7, int(round(7.0 - (x_pct / 100.0) * 7.0))))

    def _file_hint_from_motor_y(self, y_pct: float) -> int:
        # Motor Y=0 is file a, Motor Y=100 is file h.
        return max(0, min(7, int(round((y_pct / 100.0) * 7.0))))

    def _motor_x_for_rank(self, rank_index: int) -> float:
        return ((7.0 - float(rank_index)) / 7.0) * 100.0

    def _motor_y_for_file(self, file_index: int) -> float:
        return (float(file_index) / 7.0) * 100.0

    def _edge_candidates_for_pct(self, slot: PercentEndpoint) -> list[BoardExit]:
        rank_hint = self._rank_hint_from_motor_x(slot.x_pct)
        file_hint = self._file_hint_from_motor_y(slot.y_pct)

        def rank_order() -> list[int]:
            return sorted(range(8), key=lambda r: (abs(r - rank_hint), r))

        def file_order() -> list[int]:
            return sorted(range(8), key=lambda f: (abs(f - file_hint), f))

        groups: list[tuple[float, str]] = [
            (slot.y_pct, "top_file_a"),
            (100.0 - slot.y_pct, "bottom_file_h"),
            (slot.x_pct, "right_rank8"),
            (100.0 - slot.x_pct, "left_rank1"),
        ]
        groups.sort(key=lambda item: item[0])

        exits: list[BoardExit] = []
        for _dist, side in groups:
            if side == "bottom_file_h":
                for rank in rank_order():
                    exits.append(
                        BoardExit(
                            edge=(7, rank),
                            outside=_pct_ep(self._motor_x_for_rank(rank), config.BOARD_EXIT_BOTTOM_Y_PCT),
                        )
                    )
            elif side == "top_file_a":
                for rank in rank_order():
                    exits.append(
                        BoardExit(
                            edge=(0, rank),
                            outside=_pct_ep(self._motor_x_for_rank(rank), config.BOARD_EXIT_TOP_Y_PCT),
                        )
                    )
            elif side == "right_rank8":
                for file_index in file_order():
                    exits.append(
                        BoardExit(
                            edge=(file_index, 7),
                            outside=_pct_ep(config.BOARD_EXIT_RIGHT_X_PCT, self._motor_y_for_file(file_index)),
                        )
                    )
            else:
                for file_index in file_order():
                    exits.append(
                        BoardExit(
                            edge=(file_index, 0),
                            outside=_pct_ep(config.BOARD_EXIT_LEFT_X_PCT, self._motor_y_for_file(file_index)),
                    )
                    )
        return exits

    def _offboard_anchor_nodes(self, src: PercentEndpoint, dst: PercentEndpoint) -> list[PercentEndpoint]:
        right_x = config.BOARD_EXIT_RIGHT_X_PCT
        left_x = config.BOARD_EXIT_LEFT_X_PCT
        top_y = config.BOARD_EXIT_TOP_Y_PCT
        bottom_y = config.BOARD_EXIT_BOTTOM_Y_PCT
        ys = {
            src.y_pct,
            dst.y_pct,
            top_y,
            bottom_y,
            config.CAPTURE_BOTTOM_BASE_Y_PCT,
            config.CAPTURE_BOTTOM_BASE_Y_PCT - config.CAPTURE_BOTTOM_ROW_STEP_Y_PCT,
            config.CAPTURE_TOP_OVERFLOW_Y_PCT,
        }
        xs = {src.x_pct, dst.x_pct, right_x, left_x}
        nodes = [src, dst]
        for x in (right_x, left_x):
            for y in ys:
                nodes.append(_pct_ep(x, y))
        for y in (top_y, bottom_y, config.CAPTURE_TOP_OVERFLOW_Y_PCT, config.CAPTURE_BOTTOM_BASE_Y_PCT, config.CAPTURE_BOTTOM_BASE_Y_PCT - config.CAPTURE_BOTTOM_ROW_STEP_Y_PCT):
            for x in xs:
                nodes.append(_pct_ep(x, y))
        unique: dict[tuple[float, float], PercentEndpoint] = {}
        for node in nodes:
            unique[_pct_key(node)] = node
        return list(unique.values())

    def _offboard_segment_is_in_lane(self, a: PercentEndpoint, b: PercentEndpoint) -> bool:
        same_x = abs(a.x_pct - b.x_pct) <= 1e-6
        same_y = abs(a.y_pct - b.y_pct) <= 1e-6
        if not same_x and not same_y:
            return False
        if same_y:
            return (
                a.y_pct <= config.BOARD_EXIT_TOP_Y_PCT
                or a.y_pct >= config.BOARD_EXIT_BOTTOM_Y_PCT
            )
        return (
            a.x_pct <= config.BOARD_EXIT_RIGHT_X_PCT
            or a.x_pct >= config.BOARD_EXIT_LEFT_X_PCT
        )

    def _offboard_segment_clear(
        self,
        a: PercentEndpoint,
        b: PercentEndpoint,
        ignore: set[tuple[float, float]] | None = None,
    ) -> bool:
        if not self._offboard_segment_is_in_lane(a, b):
            return False
        ignore = ignore or set()
        for key in self.offboard_occupied:
            if key in ignore:
                continue
            occupied = PercentEndpoint(key[0], key[1])
            if _pct_point_to_segment_distance(occupied, a, b) < config.OFFBOARD_CLEARANCE_PCT:
                return False
        return True

    def _offboard_path(
        self,
        src: PercentEndpoint,
        dst: PercentEndpoint,
        ignore_slots: set[tuple[float, float]] | None = None,
    ) -> tuple[PercentEndpoint, ...] | None:
        ignore = set(ignore_slots or set()) | {_pct_key(src), _pct_key(dst)}
        if self._offboard_segment_clear(src, dst, ignore):
            return (src, dst)

        nodes = self._offboard_anchor_nodes(src, dst)
        by_key = {_pct_key(node): node for node in nodes}
        start_key = _pct_key(src)
        goal_key = _pct_key(dst)
        dist: dict[tuple[float, float], float] = {start_key: 0.0}
        came_from: dict[tuple[float, float], tuple[float, float]] = {}
        open_heap: list[tuple[float, tuple[float, float]]] = [(0.0, start_key)]

        while open_heap:
            cur_dist, cur_key = heapq.heappop(open_heap)
            if cur_dist > dist.get(cur_key, float("inf")):
                continue
            if cur_key == goal_key:
                path_keys = [cur_key]
                while cur_key in came_from:
                    cur_key = came_from[cur_key]
                    path_keys.append(cur_key)
                path_keys.reverse()
                return tuple(by_key[key] for key in path_keys)

            cur = by_key[cur_key]
            for nxt in nodes:
                nxt_key = _pct_key(nxt)
                if nxt_key == cur_key:
                    continue
                if not self._offboard_segment_clear(cur, nxt, ignore):
                    continue
                nd = cur_dist + _pct_distance(cur, nxt)
                if nd < dist.get(nxt_key, float("inf")):
                    dist[nxt_key] = nd
                    came_from[nxt_key] = cur_key
                    heapq.heappush(open_heap, (nd, nxt_key))
        return None

    def _temp_board_route(
        self,
        blocker: BoardCoord,
        protected: set[BoardCoord],
        path_blocked: set[BoardCoord] | None = None,
    ) -> tuple[list[BoardCoord], BoardCoord] | None:
        free_cells = [
            (x, y)
            for y in range(8)
            for x in range(8)
            if (x, y) not in self.occupied and (x, y) not in protected
        ]
        if not free_cells:
            return None

        def cell_priority(cell: BoardCoord) -> tuple[int, int, int, int]:
            same_line = 0 if (cell[0] == blocker[0] or cell[1] == blocker[1]) else 1
            return (
                same_line,
                _manhattan(blocker, cell),
                abs(cell[0] - blocker[0]),
                abs(cell[1] - blocker[1]),
            )

        best: tuple[list[BoardCoord], BoardCoord] | None = None
        for candidate in sorted(free_cells, key=cell_priority):
            route = self._astar(blocker, candidate, extra_blocked=path_blocked or set())
            if route is None:
                continue
            if best is None or len(route) < len(best[0]):
                best = (route, candidate)
        return best

    def _best_route_board_to_pct(
        self,
        start: BoardCoord,
        dst_pct: PercentEndpoint,
        protected: set[BoardCoord],
    ) -> BoardToPctPlan | None:
        best: BoardToPctPlan | None = None
        best_score: tuple[int, int, float] | None = None
        for board_exit in self._edge_candidates_for_pct(dst_pct):
            offboard_path = self._offboard_path(board_exit.outside, dst_pct)
            if offboard_path is None:
                continue
            planned = self._astar_min_blockers(start, board_exit.edge, protected)
            if planned is None:
                continue
            route, blockers = planned
            score = (len(blockers), len(route) + len(offboard_path), sum(_pct_distance(a, b) for a, b in zip(offboard_path, offboard_path[1:])))
            if best_score is None or score < best_score:
                best_score = score
                best = BoardToPctPlan(
                    route=tuple(route),
                    blockers=frozenset(blockers),
                    board_exit=board_exit,
                    offboard_path=offboard_path,
                )
        return best

    def _best_route_pct_to_board(
        self,
        src_pct: PercentEndpoint,
        dst: BoardCoord,
        protected: set[BoardCoord],
    ) -> PctToBoardPlan | None:
        best: PctToBoardPlan | None = None
        best_score: tuple[int, int, float] | None = None
        for board_exit in self._edge_candidates_for_pct(src_pct):
            offboard_path = self._offboard_path(src_pct, board_exit.outside, ignore_slots={_pct_key(src_pct)})
            if offboard_path is None:
                continue
            planned = self._astar_min_blockers(board_exit.edge, dst, protected)
            if planned is None:
                continue
            route, blockers = planned
            if board_exit.edge in self.occupied and board_exit.edge != dst and board_exit.edge not in protected:
                blockers = blockers | {board_exit.edge}
            score = (len(blockers), len(route) + len(offboard_path), sum(_pct_distance(a, b) for a, b in zip(offboard_path, offboard_path[1:])))
            if best_score is None or score < best_score:
                best_score = score
                best = PctToBoardPlan(
                    board_exit=board_exit,
                    offboard_path=offboard_path,
                    route=tuple(route),
                    blockers=frozenset(blockers),
                )
        return best

    def _best_route_board_to_board(
        self,
        start: BoardCoord,
        dst: BoardCoord,
        protected: set[BoardCoord],
    ) -> BoardToBoardPlan | None:
        planned = self._astar_min_blockers(start, dst, protected)
        if planned is None:
            return None
        route, blockers = planned
        return BoardToBoardPlan(route=tuple(route), blockers=frozenset(blockers))

    def _emit_offboard_path(self, path: Sequence[PercentEndpoint]) -> None:
        for src, dst in zip(path, path[1:]):
            self.append_segment(src, dst)

    def _emit_board_to_pct_plan(self, plan: BoardToPctPlan) -> None:
        for src_ep, dst_ep in _compress_path(plan.route):
            self.append_segment(src_ep, dst_ep)
        self.append_segment(_board_ep(plan.route[-1][0], plan.route[-1][1]), plan.offboard_path[0])
        self._emit_offboard_path(plan.offboard_path)

    def _emit_pct_to_board_plan(self, plan: PctToBoardPlan) -> None:
        self._emit_offboard_path(plan.offboard_path)
        edge = plan.board_exit.edge
        self.append_segment(plan.offboard_path[-1], _board_ep(edge[0], edge[1]))
        for src_ep, dst_ep in _compress_path(plan.route):
            self.append_segment(src_ep, dst_ep)

    def _emit_board_to_board_plan(self, plan: BoardToBoardPlan) -> None:
        for src_ep, dst_ep in _compress_path(plan.route):
            self.append_segment(src_ep, dst_ep)

    def _free_board_parking_cells(self, protected: set[BoardCoord]) -> list[BoardCoord]:
        return [
            (x, y)
            for y in range(8)
            for x in range(8)
            if (x, y) not in self.occupied and (x, y) not in protected
        ]

    def _temp_slot_candidates(self) -> list[PercentEndpoint]:
        candidates: list[PercentEndpoint] = []
        for idx in range(config.MAX_TEMP_RELOCATIONS * 2):
            slot = self._temp_slot(idx)
            if not self._is_offboard_occupied(slot):
                candidates.append(slot)
        return candidates

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

    def _move_blocker_to_temp(
        self,
        blocker: BoardCoord,
        protected: set[BoardCoord] | None = None,
        path_blocked: set[BoardCoord] | None = None,
    ) -> bool:
        del path_blocked
        return self._relocate_blocker_recursive(blocker, protected or set(), depth=0, active=set())

    def _relocate_blocker_recursive(
        self,
        blocker: BoardCoord,
        protected: set[BoardCoord],
        *,
        depth: int,
        active: set[BoardCoord],
    ) -> bool:
        if blocker not in self.occupied:
            return True
        if blocker in active or depth > config.MAX_RECURSIVE_BLOCKER_DEPTH:
            return False
        if len(self._temp_relocations) >= config.MAX_TEMP_RELOCATIONS:
            return False

        active.add(blocker)
        candidate_plans: list[
            tuple[tuple[int, int, float, int], str, BoardCoord | PercentEndpoint, BoardToBoardPlan | BoardToPctPlan]
        ] = []
        base_protected = protected | active

        for board_slot in sorted(
            self._free_board_parking_cells(base_protected),
            key=lambda cell: (_manhattan(blocker, cell), cell[0], cell[1]),
        ):
            plan = self._best_route_board_to_board(
                blocker,
                board_slot,
                base_protected | {board_slot},
            )
            if plan is None:
                continue
            blockers = set(plan.blockers) - {blocker}
            if blockers & active:
                continue
            score = (len(blockers), len(plan.route), 0.0, 0)
            candidate_plans.append((score, "board", board_slot, plan))

        for slot in self._temp_slot_candidates():
            plan = self._best_route_board_to_pct(blocker, slot, base_protected | {blocker})
            if plan is None:
                continue
            blockers = set(plan.blockers) - {blocker}
            if blockers & active:
                continue
            offboard_len = sum(_pct_distance(a, b) for a, b in zip(plan.offboard_path, plan.offboard_path[1:]))
            score = (len(blockers), len(plan.route) + len(plan.offboard_path), offboard_len, 1)
            candidate_plans.append((score, "pct", slot, plan))

        candidate_plans.sort(key=lambda item: item[0])
        for _score, kind, destination, plan in candidate_plans:
            blockers = set(plan.blockers) - {blocker}
            route_cells = set(plan.route)
            if not self._clear_route_blockers(
                blockers,
                parking_protected=base_protected | route_cells,
                path_blocked=base_protected | route_cells,
                depth=depth + 1,
                active=active,
            ):
                continue

            if kind == "board":
                assert isinstance(destination, tuple)
                assert isinstance(plan, BoardToBoardPlan)
                self._emit_board_to_board_plan(plan)
                self.occupied.remove(blocker)
                self.occupied.add(destination)
                self._temp_relocations.append(
                    TempRelocation(origin=blocker, slot=_board_ep(destination[0], destination[1]))
                )
            else:
                assert isinstance(destination, PercentEndpoint)
                assert isinstance(plan, BoardToPctPlan)
                self._emit_board_to_pct_plan(plan)
                self.occupied.remove(blocker)
                self._mark_offboard_occupied(destination, f"temp:{blocker[0]},{blocker[1]}")
                self._temp_relocations.append(TempRelocation(origin=blocker, slot=destination))
            active.remove(blocker)
            return True

        active.remove(blocker)
        return False

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
            if not self._move_blocker_to_temp(blocker, protected, path_blocked=protected):
                continue
        return

    def _clear_route_blockers(
        self,
        blockers: set[BoardCoord],
        parking_protected: set[BoardCoord],
        path_blocked: set[BoardCoord],
        depth: int = 0,
        active: set[BoardCoord] | None = None,
    ) -> bool:
        del path_blocked
        active = active or set()
        # Clear nearest blockers first, keeping path/destination cells protected
        # so temporary parking never occupies the intended route.
        ordered = sorted(blockers, key=lambda c: min(_manhattan(c, p) for p in parking_protected))
        for blocker in ordered:
            if blocker not in self.occupied:
                continue
            if not self._relocate_blocker_recursive(blocker, parking_protected, depth=depth, active=active):
                return False
        return True

    def move_board_piece_to_pct(
        self,
        start: BoardCoord,
        dst_pct: PercentEndpoint,
        protected_extra: set[BoardCoord] | None = None,
    ) -> None:
        protected = {start}
        if protected_extra:
            protected |= protected_extra

        plan = self._best_route_board_to_pct(start, dst_pct, protected)
        if plan is None:
            self.append_segment(_board_ep(start[0], start[1]), dst_pct)
            self.occupied.remove(start)
            self._mark_offboard_occupied(dst_pct, f"piece:{start[0]},{start[1]}")
            self._fallback_direct_segments += 1
            return

        if plan.blockers:
            route_protected = protected | set(plan.route)
            if not self._clear_route_blockers(
                set(plan.blockers),
                parking_protected=route_protected,
                path_blocked=route_protected,
            ):
                self.append_segment(_board_ep(start[0], start[1]), dst_pct)
                self.occupied.remove(start)
                self._mark_offboard_occupied(dst_pct, f"piece:{start[0]},{start[1]}")
                self._fallback_direct_segments += 1
                return
            plan = self._best_route_board_to_pct(start, dst_pct, protected)
            if plan is None:
                self.append_segment(_board_ep(start[0], start[1]), dst_pct)
                self.occupied.remove(start)
                self._mark_offboard_occupied(dst_pct, f"piece:{start[0]},{start[1]}")
                self._fallback_direct_segments += 1
                return

        self._emit_board_to_pct_plan(plan)
        self.occupied.remove(start)
        self._mark_offboard_occupied(dst_pct, f"piece:{start[0]},{start[1]}")

    def move_pct_piece_to_board(self, src_pct: PercentEndpoint, dst: BoardCoord) -> None:
        protected = {dst}
        plan = self._best_route_pct_to_board(src_pct, dst, protected)
        if plan is None:
            self.append_segment(src_pct, _board_ep(dst[0], dst[1]))
            self._unmark_offboard_occupied(src_pct)
            self.occupied.add(dst)
            self._fallback_direct_segments += 1
            return

        if plan.blockers:
            route_protected = protected | set(plan.route)
            if not self._clear_route_blockers(
                set(plan.blockers),
                parking_protected=route_protected,
                path_blocked=route_protected,
            ):
                self.append_segment(src_pct, _board_ep(dst[0], dst[1]))
                self._unmark_offboard_occupied(src_pct)
                self.occupied.add(dst)
                self._fallback_direct_segments += 1
                return
            plan = self._best_route_pct_to_board(src_pct, dst, protected)
            if plan is None:
                self.append_segment(src_pct, _board_ep(dst[0], dst[1]))
                self._unmark_offboard_occupied(src_pct)
                self.occupied.add(dst)
                self._fallback_direct_segments += 1
                return

        self._emit_pct_to_board_plan(plan)
        self._unmark_offboard_occupied(src_pct)
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

        planned: BoardToBoardPlan | None = None
        for _ in range(config.MAX_TEMP_RELOCATIONS + 1):
            planned = self._best_route_board_to_board(start, dst, protected)
            if planned is None:
                break
            if not planned.blockers:
                break
            route_protected = protected | set(planned.route)
            if not self._clear_route_blockers(
                set(planned.blockers),
                parking_protected=route_protected,
                path_blocked=protected,
            ):
                break
            planned = None

        if planned is None:
            self.append_segment(_board_ep(start[0], start[1]), _board_ep(dst[0], dst[1]))
            self.occupied.remove(start)
            self.occupied.add(dst)
            self._fallback_direct_segments += 1
            return

        if planned.blockers:
            self._fallback_direct_segments += 1
        self._emit_board_to_board_plan(planned)

        self.occupied.remove(start)
        self.occupied.add(dst)

    def restore_temp_blockers(self) -> None:
        if not config.RESTORE_TEMP_RELOCATIONS:
            return
        for relocation in reversed(self._temp_relocations):
            if isinstance(relocation.slot, PercentEndpoint):
                self.move_pct_piece_to_board(relocation.slot, relocation.origin)
            else:
                self.move_board_piece_to_board(
                    (relocation.slot.x, relocation.slot.y),
                    relocation.origin,
                )
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
    start_pct = _parcheesi_location_to_pct(start_id)
    end_pct = _parcheesi_location_to_pct(end_id)

    capture = False
    capture_slots_used: list[str] = []

    mover = state_before.piece_at_id(start_id)
    target = state_before.piece_at_id(end_id)
    lines: list[str] = []
    if (
        mover is not None
        and target is not None
        and target.name != "EMPTY"
        and mover.name != "EMPTY"
        and target.to_player_num() != mover.to_player_num()
    ):
        capture = True
        if capture_inventory is not None:
            rec = capture_inventory.add_captured_piece("p1", _piece_name(target))
            slot = rec.slot
            capture_slots_used.append(
                f"{rec.captured_side}[{rec.slot_index}]={rec.slot.x_pct:.2f}%,{rec.slot.y_pct:.2f}%:{rec.piece_name}"
            )
        else:
            slot_index = len(state_before.captured_pieces.get(target.to_player_num(), []))
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
