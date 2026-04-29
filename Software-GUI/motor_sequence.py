"""Convert UI moves into collision-aware STM32 move-sequence text."""

from __future__ import annotations

from dataclasses import dataclass, field
import heapq
from pathlib import Path
import re
import time
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
_MOTOR_MAIN_H = Path(__file__).resolve().parents[1] / "FlexyBoard-Motor-Control" / "Inc" / "main.h"


@dataclass(frozen=True)
class BoardEndpoint:
    x: int
    y: int


@dataclass(frozen=True)
class PercentEndpoint:
    x_pct: float
    y_pct: float


Endpoint = BoardEndpoint | PercentEndpoint
MixedNode = BoardCoord | PercentEndpoint


@dataclass(frozen=True)
class GeneratedSequence:
    lines: List[str]
    capture_detected: bool
    temporary_relocations: int = 0
    fallback_direct_segments: int = 0
    capture_slots_used: List[str] = field(default_factory=list)
    manual_actions: List[str] = field(default_factory=list)
    debug_metrics: dict[str, object] = field(default_factory=dict)


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


@dataclass
class _CaptureInventorySlot:
    captured_side: str
    piece_name: str
    current_slot: PercentEndpoint


@dataclass(frozen=True)
class ManualCaptureRecord:
    record_id: int
    captured_side: str
    source: PercentEndpoint
    piece_name: str


@dataclass(frozen=True)
class MixedRoutePlan:
    nodes: tuple[MixedNode, ...]
    blockers: frozenset[MixedNode]


@dataclass
class OffboardOccupant:
    label: str
    restore_origin: BoardCoord | None = None
    capture_slot_index: int | None = None
    captured_side: str | None = None
    piece_name: str | None = None
    manual_record_id: int | None = None


@dataclass(frozen=True)
class OffboardGeometry:
    board_square_x_pct: float
    board_square_y_pct: float
    right_edge_x_pct: float
    left_edge_x_pct: float
    top_edge_y_pct: float
    bottom_edge_y_pct: float
    right_lane_x_pct: float
    left_lane_x_pct: float
    top_row_y_pcts: tuple[float, ...]
    bottom_row_y_pcts: tuple[float, ...]
    capture_bottom_x_pcts: tuple[float, ...]
    capture_top_x_pcts: tuple[float, ...]
    temp_left_x_pct: float
    temp_right_x_pct: float
    temp_side_y_pcts: tuple[float, ...]


class CaptureInventory:
    """Tracks captured pieces by slot, side, and piece type for physical tray usage."""

    def __init__(self, game: str) -> None:
        self.game = game
        # Global slot index -> capture entry. A shared sequence prevents P1/P2
        # captures from colliding in the same physical tray slot.
        self._slots: dict[int, _CaptureInventorySlot] = {}
        self._manual_pending: dict[int, tuple[str, str, PercentEndpoint]] = {}
        self._next_manual_id = 1

    def clear(self) -> None:
        self._slots = {}
        self._manual_pending = {}
        self._next_manual_id = 1

    def _next_free_index(self, blocked_indices: set[int] | None = None) -> int:
        blocked = blocked_indices or set()
        idx = 0
        while idx in self._slots or idx in blocked:
            idx += 1
        return idx

    def add_captured_piece(
        self,
        captured_side: str,
        piece_name: str,
        blocked_indices: set[int] | None = None,
    ) -> CaptureSlotRecord:
        if captured_side not in {"p1", "p2"}:
            raise ValueError(f"invalid captured_side: {captured_side}")
        idx = self._next_free_index(blocked_indices)
        slot = _capture_slot_by_index(idx)
        self._slots[idx] = _CaptureInventorySlot(
            captured_side=captured_side,
            piece_name=piece_name,
            current_slot=slot,
        )
        return CaptureSlotRecord(
            captured_side=captured_side,
            slot_index=idx,
            slot=slot,
            piece_name=piece_name,
        )

    def add_manual_capture(
        self,
        captured_side: str,
        piece_name: str,
        source: PercentEndpoint,
    ) -> ManualCaptureRecord:
        if captured_side not in {"p1", "p2"}:
            raise ValueError(f"invalid captured_side: {captured_side}")
        record_id = self._next_manual_id
        self._next_manual_id += 1
        self._manual_pending[record_id] = (captured_side, piece_name, source)
        return ManualCaptureRecord(
            record_id=record_id,
            captured_side=captured_side,
            source=source,
            piece_name=piece_name,
        )

    def pending_manual_records(self, captured_side: str | None = None) -> list[ManualCaptureRecord]:
        out: list[ManualCaptureRecord] = []
        for record_id in sorted(self._manual_pending):
            side, piece_name, source = self._manual_pending[record_id]
            if captured_side is not None and side != captured_side:
                continue
            out.append(
                ManualCaptureRecord(
                    record_id=record_id,
                    captured_side=side,
                    source=source,
                    piece_name=piece_name,
                )
            )
        return out

    def finalize_manual_capture(
        self,
        record_id: int,
        blocked_indices: set[int] | None = None,
    ) -> tuple[ManualCaptureRecord, CaptureSlotRecord]:
        if record_id not in self._manual_pending:
            raise KeyError(f"unknown manual capture record: {record_id}")
        side, piece_name, source = self._manual_pending.pop(record_id)
        assigned = self.add_captured_piece(side, piece_name, blocked_indices=blocked_indices)
        return (
            ManualCaptureRecord(
                record_id=record_id,
                captured_side=side,
                source=source,
                piece_name=piece_name,
            ),
            assigned,
        )

    def take_piece(self, captured_side: str, piece_name: str) -> CaptureSlotRecord | None:
        if captured_side not in {"p1", "p2"}:
            raise ValueError(f"invalid captured_side: {captured_side}")
        candidates = [
            idx for idx, entry in self._slots.items()
            if entry.captured_side == captured_side and entry.piece_name == piece_name
        ]
        if not candidates:
            return None
        idx = min(candidates)
        entry = self._slots.pop(idx)
        return CaptureSlotRecord(
            captured_side=entry.captured_side,
            slot_index=idx,
            slot=entry.current_slot,
            piece_name=entry.piece_name,
        )

    def move_slot(self, slot_index: int, new_slot: PercentEndpoint) -> None:
        if slot_index not in self._slots:
            raise KeyError(f"unknown capture slot index: {slot_index}")
        self._slots[slot_index].current_slot = new_slot

    def update_manual_capture_source(self, record_id: int, new_source: PercentEndpoint) -> None:
        if record_id not in self._manual_pending:
            raise KeyError(f"unknown manual capture record: {record_id}")
        side, piece_name, _old_source = self._manual_pending[record_id]
        self._manual_pending[record_id] = (side, piece_name, new_source)

    def count(self, captured_side: str) -> int:
        if captured_side not in {"p1", "p2"}:
            raise ValueError(f"invalid captured_side: {captured_side}")
        return sum(1 for entry in self._slots.values() if entry.captured_side == captured_side)

    def to_summary_strings(self) -> list[str]:
        out: list[str] = []
        for idx in sorted(self._slots):
            entry = self._slots[idx]
            slot = entry.current_slot
            out.append(
                f"{entry.captured_side}[{idx}]={slot.x_pct:.2f}%,{slot.y_pct:.2f}%:{entry.piece_name}"
            )
        for record_id in sorted(self._manual_pending):
            captured_side, piece_name, source = self._manual_pending[record_id]
            out.append(
                f"pending[{record_id}]={captured_side}@{source.x_pct:.2f}%,{source.y_pct:.2f}%:{piece_name}"
            )
        return out

    def occupied_records(self) -> list[CaptureSlotRecord]:
        out: list[CaptureSlotRecord] = []
        for idx in sorted(self._slots):
            entry = self._slots[idx]
            out.append(
                CaptureSlotRecord(
                    captured_side=entry.captured_side,
                    slot_index=idx,
                    slot=entry.current_slot,
                    piece_name=entry.piece_name,
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


def _segment_intersects_rect_interior(
    a: PercentEndpoint,
    b: PercentEndpoint,
    *,
    center: PercentEndpoint,
    half_w: float,
    half_h: float,
    epsilon: float = 1e-6,
) -> bool:
    """Return True if the segment enters the interior of an axis-aligned rectangle.

    Boundary-only contact is treated as non-colliding so adjacent occupancy
    squares may touch without forcing a blocker relocation.
    """
    hw = max(0.0, half_w - epsilon)
    hh = max(0.0, half_h - epsilon)
    min_x = center.x_pct - hw
    max_x = center.x_pct + hw
    min_y = center.y_pct - hh
    max_y = center.y_pct + hh

    x1, y1 = a.x_pct, a.y_pct
    x2, y2 = b.x_pct, b.y_pct
    dx = x2 - x1
    dy = y2 - y1

    def inside_open(px: float, py: float) -> bool:
        return min_x < px < max_x and min_y < py < max_y

    if inside_open(x1, y1) or inside_open(x2, y2):
        return True

    p = (-dx, dx, -dy, dy)
    q = (x1 - min_x, max_x - x1, y1 - min_y, max_y - y1)
    u1 = 0.0
    u2 = 1.0

    for pi, qi in zip(p, q):
        if abs(pi) <= 1e-12:
            if qi <= 0.0:
                return False
            continue
        t = qi / pi
        if pi < 0.0:
            if t > u2:
                return False
            if t > u1:
                u1 = t
        else:
            if t < u1:
                return False
            if t < u2:
                u2 = t

    return u1 < u2 and (0.0 < u2 and u1 < 1.0)


def _parse_define_int(header_text: str, name: str, _seen: set[str] | None = None) -> int:
    if _seen is None:
        _seen = set()
    if name in _seen:
        raise ValueError(f"Circular define reference for {name} in {_MOTOR_MAIN_H}")
    _seen.add(name)

    pattern = rf"^\s*#define\s+{re.escape(name)}\s+(.+?)\s*$"
    match = re.search(pattern, header_text, flags=re.MULTILINE)
    if not match:
        raise ValueError(f"Missing {name} in {_MOTOR_MAIN_H}")

    value_text = match.group(1).split("//", 1)[0].strip()
    int_match = re.fullmatch(r"-?\d+", value_text)
    if int_match:
        return int(int_match.group(0))

    alias_match = re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", value_text)
    if alias_match:
        return _parse_define_int(header_text, alias_match.group(0), _seen)

    raise ValueError(f"Unsupported define value for {name} in {_MOTOR_MAIN_H}: {value_text!r}")


def _load_motor_workspace_mapping() -> dict[str, int] | None:
    if not _MOTOR_MAIN_H.exists():
        return None
    try:
        text = _MOTOR_MAIN_H.read_text(encoding="utf-8")
    except Exception:
        return None
    keys = [
        "BOARD_GRID_MAX_INDEX",
        "CORNER_00_X_STEPS",
        "CORNER_00_Y_STEPS",
        "CORNER_70_X_STEPS",
        "CORNER_70_Y_STEPS",
        "CORNER_07_X_STEPS",
        "CORNER_07_Y_STEPS",
        "CORNER_77_X_STEPS",
        "CORNER_77_Y_STEPS",
        "WORKSPACE_MIN_X_STEPS",
        "WORKSPACE_MAX_X_STEPS",
        "WORKSPACE_MIN_Y_STEPS",
        "WORKSPACE_MAX_Y_STEPS",
    ]
    try:
        return {key: _parse_define_int(text, key) for key in keys}
    except Exception:
        return None


_MOTOR_WORKSPACE_MAPPING = _load_motor_workspace_mapping()


def _capture_slot_count() -> int:
    return max(1, int(config.CAPTURE_BOTTOM_COLUMNS)) * max(1, int(config.CAPTURE_BOTTOM_ROWS)) + max(
        1, int(config.CAPTURE_TOP_OVERFLOW_COLUMNS)
    )


def _capture_slot_index_from_key(key: tuple[float, float]) -> int | None:
    for idx in range(_capture_slot_count()):
        if _pct_key(_capture_slot_by_index(idx)) == key:
            return idx
    return None


def _board_coord_to_pct_from_mapping(coord: BoardCoord) -> PercentEndpoint:
    if _MOTOR_WORKSPACE_MAPPING is None:
        return _pct_ep(
            ((7.0 - float(coord[1])) / 7.0) * 100.0,
            (float(coord[0]) / 7.0) * 100.0,
        )

    c = _MOTOR_WORKSPACE_MAPPING
    max_i = c["BOARD_GRID_MAX_INDEX"]
    board_x, board_y = coord
    if board_x < 0 or board_x > max_i or board_y < 0 or board_y > max_i:
        raise ValueError(f"Board coordinate out of range: {coord}")

    motor_x = max_i - board_y
    motor_y = board_x

    u = float(motor_x) / float(max_i)
    v = float(motor_y) / float(max_i)

    x_interp = (
        (1.0 - u) * (1.0 - v) * float(c["CORNER_00_X_STEPS"])
        + u * (1.0 - v) * float(c["CORNER_70_X_STEPS"])
        + (1.0 - u) * v * float(c["CORNER_07_X_STEPS"])
        + u * v * float(c["CORNER_77_X_STEPS"])
    )
    y_interp = (
        (1.0 - u) * (1.0 - v) * float(c["CORNER_00_Y_STEPS"])
        + u * (1.0 - v) * float(c["CORNER_70_Y_STEPS"])
        + (1.0 - u) * v * float(c["CORNER_07_Y_STEPS"])
        + u * v * float(c["CORNER_77_Y_STEPS"])
    )

    x_range = float(c["WORKSPACE_MAX_X_STEPS"] - c["WORKSPACE_MIN_X_STEPS"])
    y_range = float(c["WORKSPACE_MAX_Y_STEPS"] - c["WORKSPACE_MIN_Y_STEPS"])
    x_pct = ((x_interp - float(c["WORKSPACE_MIN_X_STEPS"])) / max(1.0, x_range)) * 100.0
    y_pct = ((y_interp - float(c["WORKSPACE_MIN_Y_STEPS"])) / max(1.0, y_range)) * 100.0
    return _pct_ep(x_pct, y_pct)


def _fit_centers_in_span(start: float, end: float, count: int, square_size: float) -> tuple[float, ...]:
    if count <= 0:
        return tuple()

    span_start = float(min(start, end))
    span_end = float(max(start, end))
    span = max(0.0, span_end - span_start)
    side = max(1e-6, float(square_size))

    if span <= (count * side) + 1e-6:
        step = span / float(count)
        return tuple(span_start + ((idx + 0.5) * step) for idx in range(count))

    gap = (span - (count * side)) / float(count + 1)
    first = span_start + gap + (side / 2.0)
    step = side + gap
    return tuple(first + (idx * step) for idx in range(count))


def _build_offboard_geometry() -> OffboardGeometry:
    x_spans: list[float] = []
    y_spans: list[float] = []
    for x in range(8):
        for y in range(7):
            a = _board_coord_to_pct_from_mapping((x, y))
            b = _board_coord_to_pct_from_mapping((x, y + 1))
            x_spans.append(abs(a.x_pct - b.x_pct))
    for x in range(7):
        for y in range(8):
            a = _board_coord_to_pct_from_mapping((x, y))
            b = _board_coord_to_pct_from_mapping((x + 1, y))
            y_spans.append(abs(a.y_pct - b.y_pct))

    board_square_x_pct = (sum(x_spans) / len(x_spans)) if x_spans else 10.0
    board_square_y_pct = (sum(y_spans) / len(y_spans)) if y_spans else 10.0

    top_edge_y_pct = sum(_board_coord_to_pct_from_mapping((0, rank)).y_pct for rank in range(8)) / 8.0
    bottom_edge_y_pct = sum(_board_coord_to_pct_from_mapping((7, rank)).y_pct for rank in range(8)) / 8.0
    right_edge_x_pct = sum(_board_coord_to_pct_from_mapping((file_index, 7)).x_pct for file_index in range(8)) / 8.0
    left_edge_x_pct = sum(_board_coord_to_pct_from_mapping((file_index, 0)).x_pct for file_index in range(8)) / 8.0

    top_outer_end = max(0.0, top_edge_y_pct - (board_square_y_pct / 2.0))
    bottom_outer_start = min(100.0, bottom_edge_y_pct + (board_square_y_pct / 2.0))
    right_outer_end = max(0.0, right_edge_x_pct - (board_square_x_pct / 2.0))
    left_outer_start = min(100.0, left_edge_x_pct + (board_square_x_pct / 2.0))

    bottom_row_y_pcts = _fit_centers_in_span(
        bottom_outer_start,
        100.0,
        max(1, int(config.CAPTURE_BOTTOM_ROWS)),
        board_square_y_pct,
    )
    top_row_y_pcts = _fit_centers_in_span(
        0.0,
        top_outer_end,
        max(1, 1),
        board_square_y_pct,
    )
    right_lane_x_pct = _fit_centers_in_span(
        0.0,
        right_outer_end,
        1,
        board_square_x_pct,
    )[0]
    left_lane_x_pct = _fit_centers_in_span(
        left_outer_start,
        100.0,
        1,
        board_square_x_pct,
    )[0]

    capture_bottom_x_pcts = _fit_centers_in_span(
        0.0,
        100.0,
        max(1, int(config.CAPTURE_BOTTOM_COLUMNS)),
        board_square_x_pct,
    )
    capture_top_x_pcts = _fit_centers_in_span(
        0.0,
        100.0,
        max(1, int(config.CAPTURE_TOP_OVERFLOW_COLUMNS)),
        board_square_x_pct,
    )
    # Temp parking should sit in the middle of the side bands between the
    # chessboard edge and the outer workspace edge, not on the extreme edge.
    temp_left_x_pct = right_lane_x_pct
    temp_right_x_pct = left_lane_x_pct
    top_reserved_end = max(top_row_y_pcts) + (board_square_y_pct / 2.0)
    bottom_reserved_start = min(bottom_row_y_pcts) - (board_square_y_pct / 2.0)
    temp_side_y_pcts = _fit_centers_in_span(
        top_reserved_end,
        bottom_reserved_start,
        max(1, int(config.TEMP_RELOCATE_SIDE_SLOTS)),
        board_square_y_pct,
    )

    return OffboardGeometry(
        board_square_x_pct=board_square_x_pct,
        board_square_y_pct=board_square_y_pct,
        right_edge_x_pct=right_edge_x_pct,
        left_edge_x_pct=left_edge_x_pct,
        top_edge_y_pct=top_edge_y_pct,
        bottom_edge_y_pct=bottom_edge_y_pct,
        right_lane_x_pct=right_lane_x_pct,
        left_lane_x_pct=left_lane_x_pct,
        top_row_y_pcts=top_row_y_pcts,
        bottom_row_y_pcts=bottom_row_y_pcts,
        capture_bottom_x_pcts=capture_bottom_x_pcts,
        capture_top_x_pcts=capture_top_x_pcts,
        temp_left_x_pct=temp_left_x_pct,
        temp_right_x_pct=temp_right_x_pct,
        temp_side_y_pcts=temp_side_y_pcts,
    )


_OFFBOARD_GEOMETRY = _build_offboard_geometry()


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
        x = _OFFBOARD_GEOMETRY.capture_bottom_x_pcts[col]
        y_rows = tuple(reversed(_OFFBOARD_GEOMETRY.bottom_row_y_pcts))
        y = y_rows[row]
        return _pct_ep(x, y)

    overflow_index = slot_index - bottom_capacity
    top_cols = max(1, int(config.CAPTURE_TOP_OVERFLOW_COLUMNS))
    col = overflow_index % top_cols
    x = _OFFBOARD_GEOMETRY.capture_top_x_pcts[col]
    top_rows = _OFFBOARD_GEOMETRY.top_row_y_pcts or (50.0,)
    row = min(overflow_index // top_cols, len(top_rows) - 1)
    y = top_rows[row]
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
    # Prefer orthogonal steps before diagonal steps. When blocker count and
    # path length tie, this tends to produce fewer compressed STM segments.
    for dx, dy in (
        (-1, 0),
        (1, 0),
        (0, -1),
        (0, 1),
        (-1, -1),
        (-1, 1),
        (1, -1),
        (1, 1),
    ):
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


def _compressed_segment_count(path: Sequence[BoardCoord]) -> int:
    return len(_compress_path(path))


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


def _is_board_node(node: MixedNode) -> bool:
    return not isinstance(node, PercentEndpoint)


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
        self._reserved_capture_slot_indices: set[int] = set()
        self._total_distance_pct = 0.0
        self._relocated_blocker_count = 0
        self._restored_blocker_count = 0
        self._direct_route_hits = 0
        self._direct_route_rejections: list[str] = []
        self._estimated_route_cost_total = 0.0
        self._board_square_x_pct, self._board_square_y_pct = self._estimate_board_square_size_pct()
        self.offboard_occupied: dict[tuple[float, float], OffboardOccupant] = {}
        if capture_inventory is not None:
            for rec in capture_inventory.occupied_records():
                self._mark_offboard_occupied(
                    rec.slot,
                    f"captured:{rec.captured_side}:{rec.piece_name}",
                    capture_slot_index=rec.slot_index,
                    captured_side=rec.captured_side,
                    piece_name=rec.piece_name,
                )
            for rec in capture_inventory.pending_manual_records():
                self._mark_offboard_occupied(
                    rec.source,
                    f"manual_pending:{rec.captured_side}:{rec.piece_name}",
                    manual_record_id=rec.record_id,
                    captured_side=rec.captured_side,
                    piece_name=rec.piece_name,
                )
        else:
            for idx in range(captured_count_p1):
                self._mark_offboard_occupied(
                    _capture_slot("p1", idx),
                    "captured:p1",
                    capture_slot_index=idx,
                    captured_side="p1",
                )
            for idx in range(captured_count_p2):
                self._mark_offboard_occupied(
                    _capture_slot("p2", idx),
                    "captured:p2",
                    capture_slot_index=idx,
                    captured_side="p2",
                )

    def build(self) -> GeneratedSequence:
        return GeneratedSequence(
            lines=self.lines[:],
            capture_detected=False,
            temporary_relocations=len(self._temp_relocations),
            debug_metrics=self.debug_metrics(capture_detected=False, planning_ms=0.0),
        )

    def capture_slot_next(self, captured_side: str, piece_name: str) -> PercentEndpoint:
        if self._capture_inventory is not None:
            rec = self._capture_inventory.add_captured_piece(
                captured_side,
                piece_name,
                blocked_indices=self._reserved_capture_slot_indices,
            )
            self._capture_slots_used.append(
                f"{captured_side}[{rec.slot_index}]={rec.slot.x_pct:.2f}%,{rec.slot.y_pct:.2f}%:{piece_name}"
            )
            return rec.slot

        slot_index = self._captured_count[captured_side]
        while slot_index in self._reserved_capture_slot_indices:
            slot_index += 1
        slot = _capture_slot(captured_side, slot_index)
        self._captured_count[captured_side] = slot_index + 1
        self._capture_slots_used.append(
            f"{captured_side}[{slot_index}]={slot.x_pct:.2f}%,{slot.y_pct:.2f}%:{piece_name}"
        )
        return slot

    def append_segment(self, src: Endpoint, dst: Endpoint) -> None:
        self._total_distance_pct += _pct_distance(self._endpoint_to_pct(src), self._endpoint_to_pct(dst))
        self.lines.append(f"{_endpoint_token(src)} -> {_endpoint_token(dst)}")

    def record_note(self, text: str) -> None:
        self._capture_slots_used.append(text)

    def debug_metrics(self, *, capture_detected: bool, planning_ms: float) -> dict[str, object]:
        return {
            "segments": len(self.lines),
            "xy_distance_pct": round(self._total_distance_pct, 3),
            "xy_distance_board_squares": round(
                self._total_distance_pct / max(self._board_square_x_pct, self._board_square_y_pct),
                3,
            ),
            "blockers_relocated": self._relocated_blocker_count,
            "blockers_restored": self._restored_blocker_count,
            "estimated_piece_moves": 1 + int(capture_detected) + self._relocated_blocker_count,
            "fallback_used": self._fallback_direct_segments > 0,
            "fallback_segments": self._fallback_direct_segments,
            "direct_route_hits": self._direct_route_hits,
            "direct_route_rejected_reasons": self._direct_route_rejections[:],
            "estimated_route_cost_pct": round(self._estimated_route_cost_total, 3),
            "planning_time_ms": round(planning_ms, 3),
        }

    def _mark_offboard_occupied(
        self,
        slot: PercentEndpoint,
        label: str,
        *,
        restore_origin: BoardCoord | None = None,
        capture_slot_index: int | None = None,
        captured_side: str | None = None,
        piece_name: str | None = None,
        manual_record_id: int | None = None,
    ) -> None:
        self.offboard_occupied[_pct_key(slot)] = OffboardOccupant(
            label=label,
            restore_origin=restore_origin,
            capture_slot_index=capture_slot_index,
            captured_side=captured_side,
            piece_name=piece_name,
            manual_record_id=manual_record_id,
        )

    def _unmark_offboard_occupied(self, slot: PercentEndpoint) -> OffboardOccupant | None:
        return self.offboard_occupied.pop(_pct_key(slot), None)

    def _is_offboard_occupied(self, slot: PercentEndpoint) -> bool:
        return _pct_key(slot) in self.offboard_occupied

    def _get_offboard_occupant(self, slot: PercentEndpoint) -> OffboardOccupant | None:
        return self.offboard_occupied.get(_pct_key(slot))

    def _temp_relocation_origin_for_node(self, node: MixedNode) -> BoardCoord | None:
        for relocation in self._temp_relocations:
            slot = relocation.slot
            if isinstance(slot, PercentEndpoint):
                if isinstance(node, PercentEndpoint) and _pct_key(slot) == _pct_key(node):
                    return relocation.origin
                continue
            if _is_board_node(node) and (slot.x, slot.y) == node:
                return relocation.origin
        return None

    def _is_node_occupied(self, node: MixedNode) -> bool:
        if isinstance(node, PercentEndpoint):
            return self._is_offboard_occupied(node)
        return node in self.occupied

    def _blocker_identity(self, node: MixedNode) -> object:
        temp_origin = self._temp_relocation_origin_for_node(node)
        if temp_origin is not None:
            return ("temp", temp_origin)
        if _is_board_node(node):
            return ("board", node)
        assert isinstance(node, PercentEndpoint)
        occupant = self._get_offboard_occupant(node)
        if occupant is None:
            return ("offboard", _pct_key(node))
        if occupant.capture_slot_index is not None:
            return ("capture", occupant.capture_slot_index)
        if occupant.restore_origin is not None:
            return ("temp", occupant.restore_origin)
        if occupant.manual_record_id is not None:
            return ("manual", occupant.manual_record_id)
        return ("offboard", _pct_key(node))

    def _free_mixed_parking_nodes(self, protected: set[MixedNode]) -> list[MixedNode]:
        candidates: list[MixedNode] = []
        protected_board = {node for node in protected if _is_board_node(node)}
        protected_slots = {_pct_key(node) for node in protected if isinstance(node, PercentEndpoint)}

        for y in range(8):
            for x in range(8):
                cell = (x, y)
                if cell in self.occupied or cell in protected_board:
                    continue
                candidates.append(cell)

        seen_slots: set[tuple[float, float]] = set()
        for slot in self._offboard_lattice_nodes():
            key = _pct_key(slot)
            if key in seen_slots or key in protected_slots or self._is_offboard_occupied(slot):
                continue
            seen_slots.add(key)
            candidates.append(slot)
        return candidates

    def _reserve_capture_slot_if_needed(self, slot: PercentEndpoint) -> None:
        idx = _capture_slot_index_from_key(_pct_key(slot))
        if idx is not None:
            self._reserved_capture_slot_indices.add(idx)

    def _release_capture_slot_if_needed(self, slot: PercentEndpoint) -> None:
        idx = _capture_slot_index_from_key(_pct_key(slot))
        if idx is not None:
            self._reserved_capture_slot_indices.discard(idx)

    def _update_temp_relocation_slot(self, restore_origin: BoardCoord, new_slot: Endpoint) -> None:
        for idx, relocation in enumerate(self._temp_relocations):
            if relocation.origin == restore_origin:
                self._temp_relocations[idx] = TempRelocation(origin=restore_origin, slot=new_slot)
                return

    def _apply_arrival_occupancy(
        self,
        src: MixedNode,
        dst: MixedNode,
        *,
        new_offboard_occupant: OffboardOccupant | None = None,
    ) -> None:
        moved_occupant: OffboardOccupant | None = None
        if _is_board_node(src):
            self.occupied.discard(src)  # type: ignore[arg-type]
        else:
            assert isinstance(src, PercentEndpoint)
            moved_occupant = self._unmark_offboard_occupied(src)
            if moved_occupant is not None and moved_occupant.capture_slot_index is not None:
                self._release_capture_slot_if_needed(src)

        if _is_board_node(dst):
            self.occupied.add(dst)  # type: ignore[arg-type]
            if moved_occupant is not None and moved_occupant.restore_origin is not None:
                self._update_temp_relocation_slot(
                    moved_occupant.restore_origin,
                    _board_ep(dst[0], dst[1]),  # type: ignore[index]
                )
        else:
            assert isinstance(dst, PercentEndpoint)
            occupant = moved_occupant or new_offboard_occupant
            if occupant is None:
                if _is_board_node(src):
                    occupant = OffboardOccupant(
                        label=f"piece:{src[0]},{src[1]}",
                    )
                else:
                    occupant = OffboardOccupant(label=f"piece:{src.x_pct:.2f},{src.y_pct:.2f}")
            self.offboard_occupied[_pct_key(dst)] = occupant
            if occupant.capture_slot_index is not None:
                self._reserve_capture_slot_if_needed(dst)
                if self._capture_inventory is not None:
                    try:
                        self._capture_inventory.move_slot(occupant.capture_slot_index, dst)
                    except KeyError:
                        pass
            if occupant.manual_record_id is not None and self._capture_inventory is not None:
                try:
                    self._capture_inventory.update_manual_capture_source(occupant.manual_record_id, dst)
                except KeyError:
                    pass
            if occupant.restore_origin is not None:
                self._update_temp_relocation_slot(occupant.restore_origin, dst)

    def move_board_piece_direct(self, start: BoardCoord, dst: BoardCoord) -> None:
        self.append_segment(_board_ep(start[0], start[1]), _board_ep(dst[0], dst[1]))
        if start in self.occupied:
            self.occupied.remove(start)
        self.occupied.add(dst)

    def _board_step_distance(self, a: BoardCoord, b: BoardCoord) -> float:
        return _pct_distance(self._board_coord_to_pct(a), self._board_coord_to_pct(b))

    def _node_distance(self, a: MixedNode, b: MixedNode) -> float:
        return _pct_distance(self._node_to_pct(a), self._node_to_pct(b))

    def _mixed_route_distance(self, nodes: Sequence[MixedNode]) -> float:
        return sum(self._node_distance(a, b) for a, b in zip(nodes, nodes[1:]))

    def _segment_penalty_pct(self) -> float:
        return max(self._board_square_x_pct, self._board_square_y_pct) * 0.35

    def _pickup_release_cost_pct(self) -> float:
        return max(self._board_square_x_pct, self._board_square_y_pct) * 1.15

    def _z_motion_cost_pct(self) -> float:
        return max(self._board_square_x_pct, self._board_square_y_pct) * 0.85

    def _restore_cost_pct(self) -> float:
        return max(self._board_square_x_pct, self._board_square_y_pct) * 1.35

    def _clearance_penalty_pct(self) -> float:
        return max(self._board_square_x_pct, self._board_square_y_pct) * 0.45

    def _blocker_penalty_pct(self) -> float:
        return (
            self._pickup_release_cost_pct()
            + self._z_motion_cost_pct()
            + self._restore_cost_pct()
        )

    def _estimated_execution_cost(
        self,
        *,
        blocker_count: int,
        segment_count: int,
        distance_pct: float,
        clearance_count: int = 0,
    ) -> float:
        return (
            distance_pct
            + (segment_count * self._segment_penalty_pct())
            + (blocker_count * self._blocker_penalty_pct())
            + (clearance_count * self._clearance_penalty_pct())
        )

    def _score_estimated_execution(
        self,
        *,
        blocker_count: int,
        segment_count: int,
        distance_pct: float,
        clearance_count: int = 0,
    ) -> tuple[float, int, int, float, int]:
        return (
            self._estimated_execution_cost(
                blocker_count=blocker_count,
                segment_count=segment_count,
                distance_pct=distance_pct,
                clearance_count=clearance_count,
            ),
            blocker_count,
            segment_count,
            distance_pct,
            clearance_count,
        )

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

    def _temp_slot(self, index: int) -> PercentEndpoint:
        side_slots = max(1, int(config.TEMP_RELOCATE_SIDE_SLOTS))
        side_index = index // side_slots
        slot_in_side = index % side_slots
        if side_index % 2 == 0:
            x = _OFFBOARD_GEOMETRY.temp_left_x_pct
        else:
            x = _OFFBOARD_GEOMETRY.temp_right_x_pct
        y = _OFFBOARD_GEOMETRY.temp_side_y_pcts[min(slot_in_side, len(_OFFBOARD_GEOMETRY.temp_side_y_pcts) - 1)]
        return _pct_ep(x, y)

    def _motor_x_for_rank(self, rank_index: int) -> float:
        return self._board_coord_to_pct((0, rank_index)).x_pct

    def _motor_y_for_file(self, file_index: int) -> float:
        return self._board_coord_to_pct((file_index, 0)).y_pct

    def _board_coord_to_pct(self, coord: BoardCoord) -> PercentEndpoint:
        return _board_coord_to_pct_from_mapping(coord)

    def _node_to_pct(self, node: MixedNode) -> PercentEndpoint:
        if isinstance(node, PercentEndpoint):
            return node
        return self._board_coord_to_pct(node)

    def _node_to_endpoint(self, node: MixedNode) -> Endpoint:
        if isinstance(node, PercentEndpoint):
            return node
        return _board_ep(node[0], node[1])

    def _endpoint_to_pct(self, endpoint: Endpoint) -> PercentEndpoint:
        if isinstance(endpoint, PercentEndpoint):
            return endpoint
        return self._board_coord_to_pct((endpoint.x, endpoint.y))

    def _estimate_board_square_size_pct(self) -> tuple[float, float]:
        return (
            _OFFBOARD_GEOMETRY.board_square_x_pct,
            _OFFBOARD_GEOMETRY.board_square_y_pct,
        )

    def _board_bounds_pct(self) -> tuple[float, float, float, float]:
        min_x = (_OFFBOARD_GEOMETRY.right_lane_x_pct + _OFFBOARD_GEOMETRY.right_edge_x_pct) / 2.0
        max_x = (_OFFBOARD_GEOMETRY.left_lane_x_pct + _OFFBOARD_GEOMETRY.left_edge_x_pct) / 2.0
        min_y = (_OFFBOARD_GEOMETRY.top_row_y_pcts[0] + _OFFBOARD_GEOMETRY.top_edge_y_pct) / 2.0
        max_y = (_OFFBOARD_GEOMETRY.bottom_row_y_pcts[0] + _OFFBOARD_GEOMETRY.bottom_edge_y_pct) / 2.0
        return (min_x, max_x, min_y, max_y)

    def _segment_hits_piece_square(
        self,
        a: PercentEndpoint,
        b: PercentEndpoint,
        occupied_center: PercentEndpoint,
    ) -> bool:
        return _segment_intersects_rect_interior(
            a,
            b,
            center=occupied_center,
            half_w=self._board_square_x_pct,
            half_h=self._board_square_y_pct,
        )

    def _direct_board_segment_blockers(
        self,
        start: BoardCoord,
        dst: BoardCoord,
        *,
        protected: set[BoardCoord],
    ) -> set[BoardCoord] | None:
        return self._segment_board_blockers(
            self._board_coord_to_pct(start),
            self._board_coord_to_pct(dst),
            protected=protected,
            ignore_board={start, dst},
        )

    def _pct_is_in_green_area(self, point: PercentEndpoint) -> bool:
        min_x, max_x, min_y, max_y = self._board_bounds_pct()
        eps = 1e-6
        return not (min_x + eps < point.x_pct < max_x - eps and min_y + eps < point.y_pct < max_y - eps)

    def _offboard_segment_stays_in_green_area(self, a: PercentEndpoint, b: PercentEndpoint) -> bool:
        samples = max(6, int(_pct_distance(a, b) / 2.5) + 2)
        for idx in range(samples + 1):
            t = idx / float(samples)
            probe = _pct_ep(
                a.x_pct + ((b.x_pct - a.x_pct) * t),
                a.y_pct + ((b.y_pct - a.y_pct) * t),
            )
            if not self._pct_is_in_green_area(probe):
                return False
        return True

    def _segment_board_blockers(
        self,
        a: PercentEndpoint,
        b: PercentEndpoint,
        *,
        protected: set[BoardCoord],
        ignore_board: set[BoardCoord],
    ) -> set[BoardCoord] | None:
        blockers: set[BoardCoord] = set()
        for coord in self.occupied:
            if coord in ignore_board:
                continue
            if self._segment_hits_piece_square(a, b, self._board_coord_to_pct(coord)):
                if coord in protected:
                    return None
                blockers.add(coord)
        return blockers

    def _segment_offboard_blockers(
        self,
        a: PercentEndpoint,
        b: PercentEndpoint,
        *,
        ignore_slots: set[tuple[float, float]] | None = None,
    ) -> set[PercentEndpoint]:
        ignore = ignore_slots or set()
        blockers: set[PercentEndpoint] = set()
        for key in self.offboard_occupied:
            if key in ignore:
                continue
            occupied = PercentEndpoint(key[0], key[1])
            if self._segment_hits_piece_square(a, b, occupied):
                blockers.add(occupied)
        return blockers

    def _offboard_lattice_nodes(
        self,
        extra_points: Sequence[PercentEndpoint] | None = None,
    ) -> list[PercentEndpoint]:
        x_values = {
            0.0,
            _OFFBOARD_GEOMETRY.right_lane_x_pct,
            _OFFBOARD_GEOMETRY.left_lane_x_pct,
            100.0,
            _OFFBOARD_GEOMETRY.temp_left_x_pct,
            _OFFBOARD_GEOMETRY.temp_right_x_pct,
            *_OFFBOARD_GEOMETRY.capture_bottom_x_pcts,
        }
        y_values = {
            0.0,
            *_OFFBOARD_GEOMETRY.top_row_y_pcts,
            *_OFFBOARD_GEOMETRY.bottom_row_y_pcts,
            100.0,
            *_OFFBOARD_GEOMETRY.temp_side_y_pcts,
        }

        for rank in range(8):
            x_values.add(self._motor_x_for_rank(rank))
        for file_index in range(8):
            y_values.add(self._motor_y_for_file(file_index))

        for x_pct in _OFFBOARD_GEOMETRY.capture_bottom_x_pcts:
            x_values.add(x_pct)
        for x_pct in _OFFBOARD_GEOMETRY.capture_top_x_pcts:
            x_values.add(x_pct)
        for y_pct in _OFFBOARD_GEOMETRY.temp_side_y_pcts:
            y_values.add(y_pct)

        for point in extra_points or ():
            x_values.add(point.x_pct)
            y_values.add(point.y_pct)

        nodes: dict[tuple[float, float], PercentEndpoint] = {}
        for x in sorted(x_values):
            for y in sorted(y_values):
                point = _pct_ep(float(x), float(y))
                if self._pct_is_in_green_area(point):
                    nodes[_pct_key(point)] = point
        for point in extra_points or ():
            if self._pct_is_in_green_area(point):
                nodes[_pct_key(point)] = point
        return list(nodes.values())

    def _mixed_edge_blockers(
        self,
        cur: MixedNode,
        nxt: MixedNode,
        *,
        start: MixedNode,
        goal: MixedNode,
        protected: set[MixedNode],
    ) -> tuple[set[MixedNode], float] | None:
        cur_pct = self._node_to_pct(cur)
        nxt_pct = self._node_to_pct(nxt)

        ignore_board: set[BoardCoord] = set()
        if _is_board_node(start):
            ignore_board.add(start)  # type: ignore[arg-type]
        if _is_board_node(goal):
            ignore_board.add(goal)  # type: ignore[arg-type]
        for node in protected:
            if _is_board_node(node):
                ignore_board.add(node)  # type: ignore[arg-type]

        ignore_slots: set[tuple[float, float]] = set()
        if isinstance(start, PercentEndpoint):
            ignore_slots.add(_pct_key(start))
        if isinstance(goal, PercentEndpoint):
            ignore_slots.add(_pct_key(goal))
        for node in protected:
            if isinstance(node, PercentEndpoint):
                ignore_slots.add(_pct_key(node))

        if _is_board_node(cur) and _is_board_node(nxt):
            dx = abs(cur[0] - nxt[0])  # type: ignore[index]
            dy = abs(cur[1] - nxt[1])  # type: ignore[index]
            if max(dx, dy) != 1:
                return None
            step_cells = _step_collision_cells(cur, nxt)  # type: ignore[arg-type]
            blockers: set[MixedNode] = set()
            for cell in step_cells:
                if cell not in self.occupied or cell in ignore_board:
                    continue
                blockers.add(cell)
            return blockers, _pct_distance(cur_pct, nxt_pct)

        if not _is_board_node(cur) and not _is_board_node(nxt):
            if not self._offboard_segment_stays_in_green_area(cur_pct, nxt_pct):
                return None
            blockers: set[MixedNode] = set(self._segment_offboard_blockers(cur_pct, nxt_pct, ignore_slots=ignore_slots))
            blockers = self._segment_board_blockers(
                cur_pct,
                nxt_pct,
                protected={node for node in protected if _is_board_node(node)},  # type: ignore[misc]
                ignore_board=ignore_board,
            )
            if blockers is None:
                return None
            return set(blockers) | set(self._segment_offboard_blockers(cur_pct, nxt_pct, ignore_slots=ignore_slots)), _pct_distance(cur_pct, nxt_pct)

        offboard_point = cur_pct if isinstance(cur, PercentEndpoint) else nxt_pct
        if not self._pct_is_in_green_area(offboard_point):
            return None
        board_blockers = self._segment_board_blockers(
            cur_pct,
            nxt_pct,
            protected={node for node in protected if _is_board_node(node)},  # type: ignore[misc]
            ignore_board=ignore_board,
        )
        if board_blockers is None:
            return None
        offboard_blockers = self._segment_offboard_blockers(cur_pct, nxt_pct, ignore_slots=ignore_slots)
        return set(board_blockers) | set(offboard_blockers), _pct_distance(cur_pct, nxt_pct)

    def _direct_route_rejection_reason(self, start: MixedNode, dst: MixedNode, blockers: set[MixedNode] | None) -> str:
        src = _endpoint_token(self._node_to_endpoint(start))
        end = _endpoint_token(self._node_to_endpoint(dst))
        if blockers is None:
            return f"direct_route_rejected={src}->{end}:edge_invalid"
        blocker_tokens = ",".join(sorted(_endpoint_token(self._node_to_endpoint(node)) for node in blockers))
        return f"direct_route_rejected={src}->{end}:blocked_by:{blocker_tokens}"

    def _record_route_cost(self, cost: float) -> None:
        self._estimated_route_cost_total += cost

    def _direct_mixed_route_if_clear(
        self,
        start: MixedNode,
        dst: MixedNode,
        protected: set[MixedNode],
    ) -> bool:
        edge = self._mixed_edge_blockers(
            start,
            dst,
            start=start,
            goal=dst,
            protected=protected,
        )
        if edge is None:
            self._direct_route_rejections.append(self._direct_route_rejection_reason(start, dst, None))
            return False
        blockers, distance = edge
        if blockers:
            self._direct_route_rejections.append(self._direct_route_rejection_reason(start, dst, blockers))
            return False
        self._direct_route_hits += 1
        self._record_route_cost(
            self._estimated_execution_cost(
                blocker_count=0,
                segment_count=1,
                distance_pct=distance,
            )
        )
        return True

    def _best_route_mixed_any(
        self,
        start: MixedNode,
        dst: MixedNode,
        protected: set[MixedNode],
    ) -> MixedRoutePlan | None:
        extra_points = [node for node in (start, dst, *protected) if isinstance(node, PercentEndpoint)]
        offboard_nodes = self._offboard_lattice_nodes(extra_points=extra_points)
        board_nodes = [(x, y) for y in range(8) for x in range(8)]
        graph_nodes: list[MixedNode] = [*board_nodes, *offboard_nodes]

        start_blockers: frozenset[MixedNode] = frozenset()
        start_key = (start, start_blockers)
        best_cost: dict[tuple[MixedNode, frozenset[MixedNode]], tuple[float, int, float]] = {
            start_key: (0.0, 0, 0.0)
        }
        came_from: dict[
            tuple[MixedNode, frozenset[MixedNode]],
            tuple[MixedNode, frozenset[MixedNode]],
        ] = {}
        frontier: dict[MixedNode, list[tuple[frozenset[MixedNode], float, int, float]]] = {
            start: [(start_blockers, 0.0, 0, 0.0)]
        }
        open_heap: list[tuple[float, float, int, float, int, MixedNode, frozenset[MixedNode]]] = []
        counter = 0
        heapq.heappush(
            open_heap,
            (self._node_distance(start, dst), 0.0, 0, 0.0, counter, start, start_blockers),
        )

        def dominated(node: MixedNode, blockers: frozenset[MixedNode], cost: float, segments: int, distance: float) -> bool:
            for known_blockers, known_cost, known_segments, known_distance in frontier.get(node, []):
                if (
                    known_cost <= cost + 1e-6
                    and known_segments <= segments
                    and known_distance <= distance + 1e-6
                    and known_blockers.issubset(blockers)
                ):
                    return True
            return False

        def remember(node: MixedNode, blockers: frozenset[MixedNode], cost: float, segments: int, distance: float) -> None:
            kept = [
                (known_blockers, known_cost, known_segments, known_distance)
                for known_blockers, known_cost, known_segments, known_distance in frontier.get(node, [])
                if not (
                    cost <= known_cost + 1e-6
                    and segments <= known_segments
                    and distance <= known_distance + 1e-6
                    and blockers.issubset(known_blockers)
                )
            ]
            kept.append((blockers, cost, segments, distance))
            frontier[node] = kept

        def candidate_neighbors(node: MixedNode) -> Iterable[MixedNode]:
            if _is_board_node(node):
                yield from _neighbors_8(node)  # type: ignore[arg-type]
            for candidate in graph_nodes:
                if candidate != node:
                    yield candidate

        while open_heap:
            _priority, cost, segments, distance, _counter, cur, blockers = heapq.heappop(open_heap)
            cur_key = (cur, blockers)
            if (cost, segments, distance) != best_cost.get(cur_key, (float("inf"), 10**9, float("inf"))):
                continue
            if cur == dst:
                path_nodes: list[MixedNode] = [cur_key[0]]
                while cur_key in came_from:
                    cur_key = came_from[cur_key]
                    path_nodes.append(cur_key[0])
                path_nodes.reverse()
                return MixedRoutePlan(nodes=tuple(path_nodes), blockers=blockers)

            for nxt in candidate_neighbors(cur):
                if nxt == cur:
                    continue
                edge = self._mixed_edge_blockers(
                    cur,
                    nxt,
                    start=start,
                    goal=dst,
                    protected=protected,
                )
                if edge is None:
                    continue
                edge_blockers, edge_distance = edge
                next_blockers = blockers | frozenset(edge_blockers)
                next_segments = segments + 1
                next_distance = distance + edge_distance
                next_cost = self._estimated_execution_cost(
                    blocker_count=len(next_blockers),
                    segment_count=next_segments,
                    distance_pct=next_distance,
                )
                next_key = (nxt, next_blockers)
                next_metrics = (next_cost, next_segments, next_distance)
                if next_metrics >= best_cost.get(next_key, (float("inf"), 10**9, float("inf"))):
                    continue
                if dominated(nxt, next_blockers, next_cost, next_segments, next_distance):
                    continue
                best_cost[next_key] = next_metrics
                came_from[next_key] = cur_key
                remember(nxt, next_blockers, next_cost, next_segments, next_distance)
                counter += 1
                heapq.heappush(
                    open_heap,
                    (
                        next_cost + self._node_distance(nxt, dst),
                        next_cost,
                        next_segments,
                        next_distance,
                        counter,
                        nxt,
                        next_blockers,
                    ),
                )
        return None

    def _emit_mixed_route_plan(self, plan: MixedRoutePlan) -> None:
        nodes = list(plan.nodes)
        idx = 0
        while idx < len(nodes) - 1:
            if _is_board_node(nodes[idx]) and _is_board_node(nodes[idx + 1]):
                end_idx = idx + 1
                while end_idx < len(nodes) and _is_board_node(nodes[end_idx]):
                    end_idx += 1
                board_path = [node for node in nodes[idx:end_idx] if _is_board_node(node)]
                for src_ep, dst_ep in _compress_path(board_path):  # type: ignore[arg-type]
                    self.append_segment(src_ep, dst_ep)
                idx = end_idx - 1
                continue
            self.append_segment(
                self._node_to_endpoint(nodes[idx]),
                self._node_to_endpoint(nodes[idx + 1]),
            )
            idx += 1

    def _relocate_blocker_recursive(
        self,
        blocker: MixedNode,
        protected: set[MixedNode],
        *,
        depth: int,
        active: set[object],
    ) -> bool:
        if not self._is_node_occupied(blocker):
            return True
        blocker_id = self._blocker_identity(blocker)
        if blocker_id in active or depth > config.MAX_RECURSIVE_BLOCKER_DEPTH:
            return False
        if len(self._temp_relocations) >= config.MAX_TEMP_RELOCATIONS:
            return False

        active.add(blocker_id)
        candidate_plans: list[
            tuple[
                tuple[float, int, int, float],
                MixedNode,
                MixedRoutePlan,
            ]
        ] = []
        base_protected = set(protected)

        def distance_key(node: MixedNode) -> tuple[float, int]:
            return (self._node_distance(blocker, node), 0 if _is_board_node(node) else 1)

        candidate_destinations = sorted(self._free_mixed_parking_nodes(base_protected), key=distance_key)[:24]
        for destination in candidate_destinations:
            temp_origin = self._temp_relocation_origin_for_node(blocker)
            if temp_origin is not None and destination == temp_origin:
                continue
            plan = self._best_route_mixed_any(blocker, destination, base_protected | {destination})
            if plan is None:
                continue
            blockers = set(plan.blockers) - {blocker}
            if any(self._blocker_identity(node) in active for node in blockers):
                continue
            score = self._score_estimated_execution(
                blocker_count=len(blockers),
                segment_count=max(1, len(plan.nodes) - 1),
                distance_pct=self._mixed_route_distance(plan.nodes),
            )
            candidate_plans.append((score, destination, plan))

        candidate_plans.sort(key=lambda item: item[0])
        for _score, destination, plan in candidate_plans:
            blockers = set(plan.blockers) - {blocker}
            route_nodes = set(plan.nodes)
            if not self._clear_route_blockers(
                blockers,
                parking_protected=base_protected | route_nodes,
                path_blocked=base_protected | route_nodes,
                depth=depth + 1,
                active=active,
            ):
                continue

            self._emit_mixed_route_plan(plan)
            self._relocated_blocker_count += 1
            if _is_board_node(blocker):
                restore_origin = self._temp_relocation_origin_for_node(blocker) or blocker
                active.add(("temp", restore_origin))
                if isinstance(destination, PercentEndpoint):
                    self.record_note(
                        "temp_reloc=offboard:"
                        f"{blocker[0]},{blocker[1]}->{destination.x_pct:.2f}%,{destination.y_pct:.2f}%"
                    )
                    self._apply_arrival_occupancy(
                        blocker,
                        destination,
                        new_offboard_occupant=OffboardOccupant(
                            label=f"temp:{blocker[0]},{blocker[1]}",
                            restore_origin=restore_origin,
                        ),
                    )
                    self._reserve_capture_slot_if_needed(destination)
                    if restore_origin == blocker:
                        self._temp_relocations.append(TempRelocation(origin=restore_origin, slot=destination))
                    else:
                        self._update_temp_relocation_slot(restore_origin, destination)
                else:
                    self.record_note(
                        f"temp_reloc=board:{blocker[0]},{blocker[1]}->{destination[0]},{destination[1]}"
                    )
                    self._apply_arrival_occupancy(blocker, destination)
                    board_slot = _board_ep(destination[0], destination[1])
                    if restore_origin == blocker:
                        self._temp_relocations.append(
                            TempRelocation(origin=restore_origin, slot=board_slot)
                        )
                    else:
                        self._update_temp_relocation_slot(restore_origin, board_slot)
            else:
                assert isinstance(blocker, PercentEndpoint)
                occupant = self._get_offboard_occupant(blocker)
                if isinstance(destination, PercentEndpoint):
                    self.record_note(
                        "temp_reloc=offboard_shift:"
                        f"{blocker.x_pct:.2f}%,{blocker.y_pct:.2f}%"
                        f"->{destination.x_pct:.2f}%,{destination.y_pct:.2f}%"
                    )
                    self._apply_arrival_occupancy(blocker, destination)
                else:
                    self.record_note(
                        "temp_reloc=board_shift:"
                        f"{blocker.x_pct:.2f}%,{blocker.y_pct:.2f}%->{destination[0]},{destination[1]}"
                    )
                    self._apply_arrival_occupancy(blocker, destination)
                if occupant is not None and occupant.capture_slot_index is not None:
                    self.record_note(
                        f"capture_slot_relocated={occupant.capture_slot_index}"
                    )
            return True

        active.remove(blocker_id)
        return False

    def _clear_route_blockers(
        self,
        blockers: set[MixedNode],
        parking_protected: set[MixedNode],
        path_blocked: set[MixedNode],
        depth: int = 0,
        active: set[object] | None = None,
    ) -> bool:
        del path_blocked
        if active is None:
            active = set()
        # Clear nearest blockers first, keeping path/destination cells protected
        # so temporary parking never occupies the intended route.
        ordered = sorted(blockers, key=lambda c: min(self._node_distance(c, p) for p in parking_protected))
        for blocker in ordered:
            if not self._is_node_occupied(blocker):
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
        protected: set[MixedNode] = {start, dst_pct}
        if protected_extra:
            protected |= protected_extra

        if self._direct_mixed_route_if_clear(start, dst_pct, protected):
            self.record_note(
                f"board_to_pct_route=direct:{start[0]},{start[1]}->{dst_pct.x_pct:.2f}%,{dst_pct.y_pct:.2f}%"
            )
            self.append_segment(_board_ep(start[0], start[1]), dst_pct)
            capture_idx = _capture_slot_index_from_key(_pct_key(dst_pct))
            occupant: OffboardOccupant | None = None
            if capture_idx is not None:
                if self._capture_inventory is not None:
                    record = next((rec for rec in self._capture_inventory.occupied_records() if rec.slot_index == capture_idx), None)
                    if record is not None:
                        occupant = OffboardOccupant(
                            label=f"captured:{record.captured_side}:{record.piece_name}",
                            capture_slot_index=record.slot_index,
                            captured_side=record.captured_side,
                            piece_name=record.piece_name,
                        )
                else:
                    occupant = OffboardOccupant(
                        label=f"captured:{capture_idx}",
                        capture_slot_index=capture_idx,
                    )
            self._apply_arrival_occupancy(start, dst_pct, new_offboard_occupant=occupant)
            return

        moved_blockers: set[object] = set()
        plan: MixedRoutePlan | None = None
        for _ in range(config.MAX_TEMP_RELOCATIONS + 1):
            plan = self._best_route_mixed_any(start, dst_pct, protected)
            if plan is None or not plan.blockers:
                break
            route_protected = protected | set(plan.nodes)
            if not self._clear_route_blockers(
                set(plan.blockers),
                parking_protected=route_protected,
                path_blocked=route_protected,
                active=moved_blockers,
            ):
                plan = None
                break
            plan = None
        if plan is None:
            self.append_segment(_board_ep(start[0], start[1]), dst_pct)
            self._apply_arrival_occupancy(start, dst_pct)
            self._fallback_direct_segments += 1
            self._record_route_cost(
                self._estimated_execution_cost(blocker_count=0, segment_count=1, distance_pct=self._node_distance(start, dst_pct))
            )
            return

        self._record_route_cost(
            self._estimated_execution_cost(
                blocker_count=len(plan.blockers),
                segment_count=max(1, len(plan.nodes) - 1),
                distance_pct=self._mixed_route_distance(plan.nodes),
            )
        )
        self._emit_mixed_route_plan(plan)
        capture_idx = _capture_slot_index_from_key(_pct_key(dst_pct))
        occupant: OffboardOccupant | None = None
        if capture_idx is not None:
            if self._capture_inventory is not None:
                record = next((rec for rec in self._capture_inventory.occupied_records() if rec.slot_index == capture_idx), None)
                if record is not None:
                    occupant = OffboardOccupant(
                        label=f"captured:{record.captured_side}:{record.piece_name}",
                        capture_slot_index=record.slot_index,
                        captured_side=record.captured_side,
                        piece_name=record.piece_name,
                    )
            else:
                occupant = OffboardOccupant(
                    label=f"captured:{capture_idx}",
                    capture_slot_index=capture_idx,
                )
        self._apply_arrival_occupancy(start, dst_pct, new_offboard_occupant=occupant)

    def move_pct_piece_to_board(self, src_pct: PercentEndpoint, dst: BoardCoord) -> None:
        protected: set[MixedNode] = {src_pct, dst}
        if self._direct_mixed_route_if_clear(src_pct, dst, protected):
            self.record_note(
                f"pct_to_board_route=direct:{src_pct.x_pct:.2f}%,{src_pct.y_pct:.2f}%->{dst[0]},{dst[1]}"
            )
            self.append_segment(src_pct, _board_ep(dst[0], dst[1]))
            self._apply_arrival_occupancy(src_pct, dst)
            return

        moved_blockers: set[object] = set()
        plan: MixedRoutePlan | None = None
        for _ in range(config.MAX_TEMP_RELOCATIONS + 1):
            plan = self._best_route_mixed_any(src_pct, dst, protected)
            if plan is None or not plan.blockers:
                break
            route_protected = protected | set(plan.nodes)
            if not self._clear_route_blockers(
                set(plan.blockers),
                parking_protected=route_protected,
                path_blocked=route_protected,
                active=moved_blockers,
            ):
                plan = None
                break
            plan = None
        if plan is None:
            self.append_segment(src_pct, _board_ep(dst[0], dst[1]))
            self._apply_arrival_occupancy(src_pct, dst)
            self._fallback_direct_segments += 1
            self._record_route_cost(
                self._estimated_execution_cost(blocker_count=0, segment_count=1, distance_pct=self._node_distance(src_pct, dst))
            )
            return

        self._record_route_cost(
            self._estimated_execution_cost(
                blocker_count=len(plan.blockers),
                segment_count=max(1, len(plan.nodes) - 1),
                distance_pct=self._mixed_route_distance(plan.nodes),
            )
        )
        self._emit_mixed_route_plan(plan)
        self._apply_arrival_occupancy(src_pct, dst)

    def move_pct_piece_to_pct(self, src_pct: PercentEndpoint, dst_pct: PercentEndpoint) -> None:
        protected: set[MixedNode] = {src_pct, dst_pct}
        if self._direct_mixed_route_if_clear(src_pct, dst_pct, protected):
            self.record_note(
                "pct_to_pct_route="
                f"mixed_direct:{src_pct.x_pct:.2f}%,{src_pct.y_pct:.2f}%"
                f"->{dst_pct.x_pct:.2f}%,{dst_pct.y_pct:.2f}% hops=1 blockers=0"
            )
            self.append_segment(src_pct, dst_pct)
            self._apply_arrival_occupancy(src_pct, dst_pct)
            return

        moved_blockers: set[object] = set()
        plan: MixedRoutePlan | None = None
        for _ in range(config.MAX_TEMP_RELOCATIONS + 1):
            plan = self._best_route_mixed_any(src_pct, dst_pct, protected)
            if plan is None or not plan.blockers:
                break
            route_protected = protected | set(plan.nodes)
            if not self._clear_route_blockers(
                set(plan.blockers),
                parking_protected=route_protected,
                path_blocked=route_protected,
                active=moved_blockers,
            ):
                plan = None
                break
            plan = None
        if plan is None:
            # Last resort: keep the move on the off-board perimeter instead of
            # cutting diagonally across the yellow board.
            route_y = (
                _OFFBOARD_GEOMETRY.bottom_row_y_pcts[0]
                if (src_pct.y_pct >= 50.0 or dst_pct.y_pct >= 50.0)
                else _OFFBOARD_GEOMETRY.top_row_y_pcts[0]
            )
            fallback_points = [
                src_pct,
                _pct_ep(src_pct.x_pct, route_y),
                _pct_ep(dst_pct.x_pct, route_y),
                dst_pct,
            ]
            for src, dst in zip(fallback_points, fallback_points[1:]):
                if _pct_key(src) == _pct_key(dst):
                    continue
                self.append_segment(src, dst)
            self.record_note(
                "pct_to_pct_route="
                f"perimeter_fallback:{src_pct.x_pct:.2f}%,{src_pct.y_pct:.2f}%"
                f"->{dst_pct.x_pct:.2f}%,{dst_pct.y_pct:.2f}%"
            )
            self._apply_arrival_occupancy(src_pct, dst_pct)
            self._fallback_direct_segments += 1
            self._record_route_cost(
                self._estimated_execution_cost(
                    blocker_count=0,
                    segment_count=max(1, len([pt for pt in fallback_points if True]) - 1),
                    distance_pct=sum(
                        _pct_distance(a, b)
                        for a, b in zip(fallback_points, fallback_points[1:])
                        if _pct_key(a) != _pct_key(b)
                    ),
                )
            )
            return

        direct_diagonal = len(plan.nodes) == 2
        self.record_note(
            "pct_to_pct_route="
            f"{'mixed_direct' if direct_diagonal else 'mixed_multi'}:"
            f"{src_pct.x_pct:.2f}%,{src_pct.y_pct:.2f}%"
            f"->{dst_pct.x_pct:.2f}%,{dst_pct.y_pct:.2f}%"
            f" hops={max(0, len(plan.nodes) - 1)}"
            f" blockers={len(plan.blockers)}"
        )
        self._record_route_cost(
            self._estimated_execution_cost(
                blocker_count=len(plan.blockers),
                segment_count=max(1, len(plan.nodes) - 1),
                distance_pct=self._mixed_route_distance(plan.nodes),
            )
        )
        self._emit_mixed_route_plan(plan)
        self._apply_arrival_occupancy(src_pct, dst_pct)

    def move_board_piece_to_board(
        self,
        start: BoardCoord,
        dst: BoardCoord,
        protected_extra: set[BoardCoord] | None = None,
    ) -> None:
        protected: set[MixedNode] = {start, dst}
        if protected_extra:
            protected |= protected_extra

        direct_blockers = self._direct_board_segment_blockers(start, dst, protected=protected)
        if direct_blockers == set():
            self.record_note(
                f"board_route=direct:{start[0]},{start[1]}->{dst[0]},{dst[1]}"
            )
            self._direct_route_hits += 1
            self._record_route_cost(
                self._estimated_execution_cost(
                    blocker_count=0,
                    segment_count=1,
                    distance_pct=self._node_distance(start, dst),
                )
            )
            self.append_segment(_board_ep(start[0], start[1]), _board_ep(dst[0], dst[1]))
            self.occupied.remove(start)
            self.occupied.add(dst)
            return
        self._direct_route_rejections.append(self._direct_route_rejection_reason(start, dst, direct_blockers))

        planned: MixedRoutePlan | None = None
        moved_blockers: set[object] = set()
        for _ in range(config.MAX_TEMP_RELOCATIONS + 1):
            planned = self._best_route_mixed_any(start, dst, protected)
            if planned is None:
                break
            if not planned.blockers:
                break
            route_protected = protected | set(planned.nodes)
            if not self._clear_route_blockers(
                set(planned.blockers),
                parking_protected=route_protected,
                path_blocked=protected,
                active=moved_blockers,
            ):
                planned = None
                break
            planned = None

        if planned is None:
            self.append_segment(_board_ep(start[0], start[1]), _board_ep(dst[0], dst[1]))
            self.occupied.remove(start)
            self.occupied.add(dst)
            self._fallback_direct_segments += 1
            self._record_route_cost(
                self._estimated_execution_cost(blocker_count=0, segment_count=1, distance_pct=self._node_distance(start, dst))
            )
            return

        if planned.blockers:
            self._fallback_direct_segments += 1
        uses_offboard = any(isinstance(node, PercentEndpoint) for node in planned.nodes)
        self.record_note(
            "board_route="
            f"{'mixed_offboard' if uses_offboard else 'mixed_board'}:"
            f"{start[0]},{start[1]}->{dst[0]},{dst[1]}"
            f" hops={max(0, len(planned.nodes) - 1)}"
            f" blockers={len(planned.blockers)}"
        )
        self._record_route_cost(
            self._estimated_execution_cost(
                blocker_count=len(planned.blockers),
                segment_count=max(1, len(planned.nodes) - 1),
                distance_pct=self._mixed_route_distance(planned.nodes),
            )
        )
        self._emit_mixed_route_plan(planned)
        self._apply_arrival_occupancy(start, dst)

    def restore_temp_blockers(self) -> None:
        if not config.RESTORE_TEMP_RELOCATIONS:
            return
        for relocation in reversed(self._temp_relocations):
            if isinstance(relocation.slot, PercentEndpoint):
                self._restored_blocker_count += 1
                self.move_pct_piece_to_board(relocation.slot, relocation.origin)
                self._release_capture_slot_if_needed(relocation.slot)
            else:
                if (relocation.slot.x, relocation.slot.y) == relocation.origin:
                    continue
                self._restored_blocker_count += 1
                self.move_board_piece_to_board(
                    (relocation.slot.x, relocation.slot.y),
                    relocation.origin,
                )
        self._temp_relocations.clear()


def _settle_pending_manual_captures(
    planner: MotionPlanner,
    capture_inventory: CaptureInventory | None,
) -> None:
    if capture_inventory is None:
        return
    for pending in capture_inventory.pending_manual_records():
        manual_record, assigned_slot = capture_inventory.finalize_manual_capture(
            pending.record_id,
            blocked_indices=planner._reserved_capture_slot_indices,
        )
        planner.record_note(
            "manual_capture_settled="
            f"{manual_record.captured_side}:{manual_record.piece_name}:"
            f"{manual_record.source.x_pct:.2f}%,{manual_record.source.y_pct:.2f}%"
            f"->{assigned_slot.slot.x_pct:.2f}%,{assigned_slot.slot.y_pct:.2f}%"
        )
        planner.move_pct_piece_to_pct(manual_record.source, assigned_slot.slot)


def generate_chess_p2_sequence(
    state_before: ChessState,
    start_id: str,
    end_id: str,
    capture_inventory: CaptureInventory | None = None,
) -> GeneratedSequence:
    planning_started = time.perf_counter()
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
    _settle_pending_manual_captures(planner, capture_inventory)

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

            # Castling is a known chess special-case. Move the king directly first,
            # then route the rook with the planner so it does not sweep through the
            # king's final square on the back rank.
            planner.move_board_piece_direct(king_src, king_dst)
            if rook_src in planner.occupied:
                planner.move_board_piece_to_board(rook_src, rook_dst, protected_extra={king_dst})
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
    planning_ms = (time.perf_counter() - planning_started) * 1000.0

    return GeneratedSequence(
        lines=planner.lines,
        capture_detected=capture,
        temporary_relocations=temp_count,
        fallback_direct_segments=planner._fallback_direct_segments,
        capture_slots_used=planner._capture_slots_used[:],
        manual_actions=manual_actions,
        debug_metrics=planner.debug_metrics(capture_detected=capture, planning_ms=planning_ms),
    )


def generate_checkers_p2_sequence(
    state_before: CheckersState,
    start_id: str,
    end_id: str,
    capture_inventory: CaptureInventory | None = None,
) -> GeneratedSequence:
    planning_started = time.perf_counter()
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
    _settle_pending_manual_captures(planner, capture_inventory)

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
    planning_ms = (time.perf_counter() - planning_started) * 1000.0

    return GeneratedSequence(
        lines=planner.lines,
        capture_detected=capture,
        temporary_relocations=temp_count,
        fallback_direct_segments=planner._fallback_direct_segments,
        capture_slots_used=planner._capture_slots_used[:],
        manual_actions=manual_actions,
        debug_metrics=planner.debug_metrics(capture_detected=capture, planning_ms=planning_ms),
    )


def generate_parcheesi_p2_sequence(
    state_before: ParcheesiState,
    start_id: str,
    end_id: str,
    capture_inventory: CaptureInventory | None = None,
) -> GeneratedSequence:
    planning_started = time.perf_counter()
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
        debug_metrics={
            "segments": len(lines),
            "xy_distance_pct": round(
                _pct_distance(start_pct, end_pct) + (_pct_distance(end_pct, slot) if capture else 0.0),
                3,
            ),
            "estimated_piece_moves": 1 + int(capture),
            "planning_time_ms": round((time.perf_counter() - planning_started) * 1000.0, 3),
        },
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
