#!/usr/bin/env python3
"""Player 2 UI: choose game, then play as P2 via drag-drop."""

from __future__ import annotations

import base64
from dataclasses import dataclass, field
from datetime import datetime
import io
import json
from pathlib import Path
import subprocess
import sys
import threading
import time

import pygame

import config
from ai_player import choose_p2_move, close_engine
from board_ui import BoardUI # Keep for other games if needed
from checkers_state import CheckersState, Piece as CheckersPiece
from chess_state import ChessPiece, ChessState
from chess_ui import ChessUI # Keep for other games if needed
from coords import parse_square
from parcheesi_state import ParcheesiState, Piece as ParcheesiPiece
from parcheesi_ui import ParcheesiUI
from ipc.mock_transport import MockTransport
from ipc.protocol import P1MoveMessage, P2MoveMessage, decode_line
from ipc.transport_tcp import TcpClientTransport
from motor_sequence import (
    CaptureInventory,
    GeneratedSequence,
    PercentEndpoint,
    generate_p2_sequence,
    write_sequence_file,
)

ROOT = Path(__file__).resolve().parent


class _TeeStream:
    def __init__(self, primary: object, mirror: object) -> None:
        self._primary = primary
        self._mirror = mirror

    def write(self, data: str) -> int:
        self._primary.write(data)
        self._mirror.write(data)
        return len(data)

    def flush(self) -> None:
        self._primary.flush()
        self._mirror.flush()

    def isatty(self) -> bool:
        return bool(getattr(self._primary, "isatty", lambda: False)())


def _install_runtime_log() -> Path:
    logs_dir = ROOT / "debug_output" / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = logs_dir / f"gui_runtime_{timestamp}.log"
    handle = log_path.open("a", encoding="utf-8")
    sys.stdout = _TeeStream(sys.stdout, handle)  # type: ignore[assignment]
    sys.stderr = _TeeStream(sys.stderr, handle)  # type: ignore[assignment]
    return log_path


def _piece_name(piece: object) -> str:
    return str(getattr(piece, "name", piece))


def _stdin_p1_injector(mock: MockTransport) -> None:
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            decoded = decode_line(line.encode("utf-8"))
        except (UnicodeDecodeError, ValueError, OSError, json.JSONDecodeError) as e:
            print("stdin: skip line:", e, file=sys.stderr)
            continue
        if isinstance(decoded, P1MoveMessage):
            mock.inject_p1_move(
                decoded.frm,
                decoded.to,
                manual_green_captures=decoded.manual_green_captures,
            )
            print("stdin: queued P1 move", decoded.frm, "->", decoded.to, file=sys.stderr)


class _Button:
    def __init__(self, rect: pygame.Rect, label: str) -> None:
        self.rect = rect
        self.label = label


@dataclass
class _ManualFixDrag:
    source_kind: str
    source_ref: str | None
    piece: object
    label: str
    mouse_pos: tuple[int, int]
    origin_pos: tuple[int, int]
    moved: bool = False


@dataclass
class _ManualFixSession:
    kind: str
    backup_state: object
    opened_at: float = field(default_factory=time.perf_counter)
    palette_rects: list[tuple[pygame.Rect, str, object]] = field(default_factory=list)
    done_rect: pygame.Rect = field(default_factory=lambda: pygame.Rect(0, 0, 0, 0))
    cancel_rect: pygame.Rect = field(default_factory=lambda: pygame.Rect(0, 0, 0, 0))
    dragging: _ManualFixDrag | None = None


class _GeometryPreview:
    def __init__(self, msg: dict[str, object]) -> None:
        image_b64 = str(msg.get("image_png_b64", ""))
        if not image_b64:
            raise ValueError("geometry preview missing image_png_b64")
        data = base64.b64decode(image_b64)
        self.image = pygame.image.load(io.BytesIO(data), "geometry_preview.png").convert()
        self.title = str(msg.get("title", "Confirm detected grid"))
        self.summary = str(msg.get("summary", "Confirm the green/yellow grid before continuing."))
        self.source_path = str(msg.get("source_path", ""))
        self.confirm_rect = pygame.Rect(0, 0, 0, 0)
        self.cancel_rect = pygame.Rect(0, 0, 0, 0)


class _GeometryCalibration:
    def __init__(self, msg: dict[str, object]) -> None:
        image_b64 = str(msg.get("image_png_b64", ""))
        if not image_b64:
            raise ValueError("geometry calibration missing image_png_b64")
        data = base64.b64decode(image_b64)
        self.image = pygame.image.load(io.BytesIO(data), "geometry_calibration.png").convert()
        self.title = str(msg.get("title", "Manual Board Geometry"))
        self.summary = str(msg.get("summary", "Click 4 green outer corners, then 4 yellow inner corners."))
        self.source_path = str(msg.get("source_path", ""))
        self.outer_points: list[tuple[float, float]] = []
        self.inner_points: list[tuple[float, float]] = []
        self.image_rect = pygame.Rect(0, 0, 0, 0)
        self.save_rect = pygame.Rect(0, 0, 0, 0)
        self.undo_rect = pygame.Rect(0, 0, 0, 0)
        self.reset_rect = pygame.Rect(0, 0, 0, 0)
        self.cancel_rect = pygame.Rect(0, 0, 0, 0)

    def is_complete(self) -> bool:
        return len(self.outer_points) == 4 and len(self.inner_points) == 4

    def next_instruction(self) -> str:
        order = ["top-left", "top-right", "bottom-right", "bottom-left"]
        if len(self.outer_points) < 4:
            return f"Outer green grid: click {order[len(self.outer_points)]} corner."
        if len(self.inner_points) < 4:
            return f"Inner yellow grid: click {order[len(self.inner_points)]} corner."
        return "Geometry complete. Click Save to use this grid for the game."

    def add_click(self, pos: tuple[int, int]) -> None:
        if not self.image_rect.collidepoint(*pos):
            return
        iw, ih = self.image.get_size()
        if self.image_rect.width <= 0 or self.image_rect.height <= 0:
            return
        x = (pos[0] - self.image_rect.x) * iw / self.image_rect.width
        y = (pos[1] - self.image_rect.y) * ih / self.image_rect.height
        x = max(0.0, min(float(iw - 1), float(x)))
        y = max(0.0, min(float(ih - 1), float(y)))
        if len(self.outer_points) < 4:
            self.outer_points.append((x, y))
        elif len(self.inner_points) < 4:
            self.inner_points.append((x, y))

    def undo(self) -> None:
        if self.inner_points:
            self.inner_points.pop()
        elif self.outer_points:
            self.outer_points.pop()

    def reset(self) -> None:
        self.outer_points.clear()
        self.inner_points.clear()


@dataclass
class _StatusNotice:
    level: str
    title: str
    message: str
    details: list[str] = field(default_factory=list)
    sticky: bool = False
    expires_at_ms: int = 0


@dataclass
class _QueuedCheckersTurn:
    steps: list[tuple[str, str]] = field(default_factory=list)
    stm_lines: list[str] = field(default_factory=list)
    manual_actions: list[str] = field(default_factory=list)
    capture_slots_used: list[str] = field(default_factory=list)
    temporary_relocations: int = 0
    fallback_direct_segments: int = 0

    def append(self, start_id: str, end_id: str, generated: GeneratedSequence) -> None:
        self.steps.append((start_id, end_id))
        self.stm_lines.extend(generated.lines)
        self.manual_actions.extend(generated.manual_actions)
        self.capture_slots_used.extend(generated.capture_slots_used)
        self.temporary_relocations += int(generated.temporary_relocations)
        self.fallback_direct_segments += int(generated.fallback_direct_segments)

    def clear(self) -> None:
        self.steps.clear()
        self.stm_lines.clear()
        self.manual_actions.clear()
        self.capture_slots_used.clear()
        self.temporary_relocations = 0
        self.fallback_direct_segments = 0


def _back_button_rect(ui: object) -> pygame.Rect:
    margin = int(getattr(ui, "margin", 16))
    return pygame.Rect(margin // 2, 14, 96, 36)


def _draw_back_button(ui: object) -> None:
    rect = _back_button_rect(ui)
    mx, my = pygame.mouse.get_pos()
    hover = rect.collidepoint(mx, my)
    bg = (74, 64, 54) if not hover else (92, 78, 62)
    border = (170, 145, 105)

    pygame.draw.rect(ui.screen, bg, rect, border_radius=10)
    pygame.draw.rect(ui.screen, border, rect, width=1, border_radius=10)

    label = ui.font.render("Back", True, (246, 238, 220))
    ui.screen.blit(label, label.get_rect(center=rect.center))


def _mode_button_rect(ui: object) -> pygame.Rect:
    screen = ui.screen
    w = 180
    h = 36
    margin = max(12, int(getattr(ui, "margin", 16)) // 2)
    return pygame.Rect(screen.get_width() - w - margin, margin, w, h)


def _manual_fix_button_rect(ui: object) -> pygame.Rect:
    margin = int(getattr(ui, "margin", 16))
    return pygame.Rect(margin // 2, 58, 126, 36)


def _draw_mode_button(ui: object, ai_mode: bool, game_kind: str) -> None:
    rect = _mode_button_rect(ui)
    mx, my = pygame.mouse.get_pos()
    hover = rect.collidepoint(mx, my)

    if game_kind == "chess":
        ai_label = "AI (Stockfish)"
    else:
        ai_label = "AI (Heuristic)"

    if ai_mode:
        bg = (35, 92, 55) if not hover else (45, 112, 65)
        border = (110, 200, 140)
        text = f"Mode: {ai_label}"
    else:
        bg = (72, 72, 86) if not hover else (84, 84, 98)
        border = (140, 140, 158)
        text = "Mode: Manual"

    pygame.draw.rect(ui.screen, bg, rect, border_radius=10)
    pygame.draw.rect(ui.screen, border, rect, width=1, border_radius=10)

    label = ui.font.render(text, True, (244, 244, 248))
    ui.screen.blit(label, label.get_rect(center=rect.center))


def _draw_manual_fix_button(ui: object) -> None:
    rect = _manual_fix_button_rect(ui)
    mx, my = pygame.mouse.get_pos()
    hover = rect.collidepoint(mx, my)
    bg = (96, 72, 34) if not hover else (116, 86, 40)
    border = (236, 178, 70)

    pygame.draw.rect(ui.screen, bg, rect, border_radius=10)
    pygame.draw.rect(ui.screen, border, rect, width=1, border_radius=10)

    label = ui.font.render("Fix Manual", True, (248, 245, 238))
    ui.screen.blit(label, label.get_rect(center=rect.center))


def _manual_fix_palette_items(kind: str, state: object) -> list[tuple[str, object]]:
    if kind == "chess":
        return [
            ("WP", ChessPiece.P1_PAWN),
            ("WN", ChessPiece.P1_KNIGHT),
            ("WB", ChessPiece.P1_BISHOP),
            ("WR", ChessPiece.P1_ROOK),
            ("WQ", ChessPiece.P1_QUEEN),
            ("WK", ChessPiece.P1_KING),
            ("BP", ChessPiece.P2_PAWN),
            ("BN", ChessPiece.P2_KNIGHT),
            ("BB", ChessPiece.P2_BISHOP),
            ("BR", ChessPiece.P2_ROOK),
            ("BQ", ChessPiece.P2_QUEEN),
            ("BK", ChessPiece.P2_KING),
        ]
    if kind == "checkers":
        return [
            ("P1M", CheckersPiece.P1_MAN),
            ("P1K", CheckersPiece.P1_KING),
            ("P2M", CheckersPiece.P2_MAN),
            ("P2K", CheckersPiece.P2_KING),
        ]
    if kind == "parcheesi" and isinstance(state, ParcheesiState):
        items: list[tuple[str, object]] = []
        for player in range(1, ParcheesiState.NUM_PLAYERS + 1):
            for token in range(1, ParcheesiState.TOKENS_PER_PLAYER + 1):
                piece = ParcheesiPiece[f"P{player}_TOKEN_{token}"]
                loc = state.piece_locations.get(piece)
                if loc is None:
                    continue
                if loc.kind in {"nest", "homearea"}:
                    items.append((f"{player}.{token}", piece))
        return items
    raise ValueError(f"Unsupported manual fix kind: {kind}")


def _manual_fix_piece_label(kind: str, piece: object) -> str:
    if kind == "chess" and isinstance(piece, ChessPiece):
        labels = {
            ChessPiece.P1_PAWN: "WP",
            ChessPiece.P1_KNIGHT: "WN",
            ChessPiece.P1_BISHOP: "WB",
            ChessPiece.P1_ROOK: "WR",
            ChessPiece.P1_QUEEN: "WQ",
            ChessPiece.P1_KING: "WK",
            ChessPiece.P2_PAWN: "BP",
            ChessPiece.P2_KNIGHT: "BN",
            ChessPiece.P2_BISHOP: "BB",
            ChessPiece.P2_ROOK: "BR",
            ChessPiece.P2_QUEEN: "BQ",
            ChessPiece.P2_KING: "BK",
        }
        return labels.get(piece, piece.name)
    if kind == "checkers" and isinstance(piece, CheckersPiece):
        labels = {
            CheckersPiece.P1_MAN: "P1M",
            CheckersPiece.P1_KING: "P1K",
            CheckersPiece.P2_MAN: "P2M",
            CheckersPiece.P2_KING: "P2K",
        }
        return labels.get(piece, piece.name)
    if kind == "parcheesi" and isinstance(piece, ParcheesiPiece):
        return f"{piece.to_player_num()}.{piece.to_token_num()}"
    return _piece_name(piece)


def _draw_manual_fix_palette_entry(
    ui: object,
    screen: pygame.Surface,
    font: pygame.font.Font,
    kind: str,
    rect: pygame.Rect,
    label: str,
    piece: object,
    hover: bool,
) -> None:
    bg = (62, 72, 88)
    border = (145, 150, 170)
    if hover:
        bg = tuple(min(255, c + 18) for c in bg)
    pygame.draw.rect(screen, bg, rect, border_radius=9)
    pygame.draw.rect(screen, border, rect, width=1, border_radius=9)

    if kind == "chess" and isinstance(piece, ChessPiece) and hasattr(ui, "_sprites") and ui._sprites.has(piece):
        src = ui._sprites.board[piece]
        max_w = max(1, rect.width - 10)
        max_h = max(1, rect.height - 8)
        scale = min(max_w / src.get_width(), max_h / src.get_height())
        scaled = pygame.transform.smoothscale(
            src,
            (
                max(1, int(src.get_width() * scale)),
                max(1, int(src.get_height() * scale)),
            ),
        )
        screen.blit(scaled, scaled.get_rect(center=rect.center))
        return

    surf = font.render(label, True, (248, 248, 250))
    screen.blit(surf, surf.get_rect(center=rect.center))


def _draw_manual_fix_drag_ghost(
    ui: object,
    screen: pygame.Surface,
    font: pygame.font.Font,
    kind: str,
    drag: _ManualFixDrag,
) -> None:
    if kind == "chess" and isinstance(drag.piece, ChessPiece) and hasattr(ui, "_sprites") and ui._sprites.has(drag.piece):
        img = ui._sprites.board[drag.piece]
        screen.blit(img, img.get_rect(center=drag.mouse_pos))
        return

    ghost_rect = pygame.Rect(0, 0, 74, 40)
    ghost_rect.center = drag.mouse_pos
    pygame.draw.rect(screen, (32, 38, 48), ghost_rect, border_radius=10)
    pygame.draw.rect(screen, (120, 220, 150), ghost_rect, width=1, border_radius=10)
    ghost = font.render(drag.label, True, (248, 248, 250))
    screen.blit(ghost, ghost.get_rect(center=ghost_rect.center))


def _begin_manual_fix(kind: str, state: object) -> _ManualFixSession:
    return _ManualFixSession(
        kind=kind,
        backup_state=state.copy(),
    )


def _manual_fix_palette_layout(
    ui: object,
    session: _ManualFixSession,
    state: object,
) -> list[tuple[pygame.Rect, str, object]]:
    panel = pygame.Rect(
        max(16, getattr(ui, "sidebar_x", getattr(ui, "screen").get_width() - 240) - 20),
        getattr(ui, "header_height", 128) + 44,
        228,
        getattr(ui, "screen").get_height() - getattr(ui, "header_height", 128) - 96,
    )
    top = panel.y + 56
    items = _manual_fix_palette_items(session.kind, state)
    cols = 2 if session.kind == "parcheesi" else 2
    button_w = 100 if session.kind == "chess" else 92
    button_h = 54 if session.kind == "chess" else 38
    gap_x = 10
    gap_y = 10
    row_w = cols * button_w + (cols - 1) * gap_x
    start_x = panel.x + max(0, (panel.width - row_w) // 2)
    rects: list[tuple[pygame.Rect, str, object]] = []
    for index, (label, value) in enumerate(items):
        row = index // cols
        col = index % cols
        rect = pygame.Rect(
            start_x + col * (button_w + gap_x),
            top + row * (button_h + gap_y),
            button_w,
            button_h,
        )
        rects.append((rect, label, value))
    session.palette_rects = rects
    return rects


def _manual_fix_pick_from_board(
    kind: str,
    state: object,
    ui: object,
    pos: tuple[int, int],
) -> tuple[str, object, tuple[int, int]] | None:
    if kind == "chess" and isinstance(state, ChessState):
        square = ui.square_at_pixel(*pos)
        if square is None:
            return None
        piece = state.get(square)
        if piece == ChessPiece.EMPTY:
            return None
        return square.to_id(), piece, ui.cell_rect(square).center

    if kind == "checkers" and isinstance(state, CheckersState):
        square = ui.square_at_pixel(*pos)
        if square is None:
            return None
        piece = state.get(square)
        if piece == CheckersPiece.EMPTY:
            return None
        return square.to_id(), piece, ui.cell_rect(square).center

    if kind == "parcheesi" and isinstance(state, ParcheesiState):
        location_id = ui.location_id_at_pixel(pos)
        if location_id is None:
            return None
        piece = state.piece_at_id(location_id)
        if piece is ParcheesiPiece.EMPTY:
            return None
        gx, gy = ParcheesiState.location_id_to_grid(location_id)
        center = ui._grid_point_to_screen(gx, gy)
        return location_id, piece, center

    return None


def _manual_fix_drop_drag(
    session: _ManualFixSession,
    state: object,
    ui: object,
    drop_pos: tuple[int, int],
) -> str | None:
    drag = session.dragging
    if drag is None:
        return None

    if session.kind == "chess" and isinstance(state, ChessState):
        turn_side = state.turn_side()
        square = ui.square_at_pixel(*drop_pos)
        if square is None:
            if drag.source_kind == "board" and drag.source_ref is not None:
                state.set(parse_square(drag.source_ref), ChessPiece.EMPTY)
                state.manual_resync_engine(turn_side=turn_side)
                return f"deleted {drag.label}"
            return None
        if drag.source_kind == "board" and drag.source_ref is not None:
            state.set(parse_square(drag.source_ref), ChessPiece.EMPTY)
        state.set(square, drag.piece)
        state.manual_resync_engine(turn_side=turn_side)
        return f"{drag.label} -> {square.to_id()}"

    if session.kind == "checkers" and isinstance(state, CheckersState):
        square = ui.square_at_pixel(*drop_pos)
        if square is None:
            if drag.source_kind == "board" and drag.source_ref is not None:
                state.set(parse_square(drag.source_ref), CheckersPiece.EMPTY)
                state.clear_forced_continuations()
                return f"deleted {drag.label}"
            return None
        if drag.source_kind == "board" and drag.source_ref is not None:
            state.set(parse_square(drag.source_ref), CheckersPiece.EMPTY)
        state.set(square, drag.piece)
        state.clear_forced_continuations()
        return f"{drag.label} -> {square.to_id()}"

    if session.kind == "parcheesi" and isinstance(state, ParcheesiState):
        location_id = ui.location_id_at_pixel(drop_pos)
        if location_id is None:
            state.manual_move_piece(drag.piece, ParcheesiState.location_id("nest", drag.piece.to_player_num(), drag.piece.to_token_num()))
            return f"removed {drag.label} to nest"
        dest = state.parse_location_id(location_id)
        if dest.kind in {"home", "homearea"}:
            err = state.manual_clear_location(location_id)
            if err is not None:
                return f"error: {err}"
        err = state.manual_move_piece(drag.piece, location_id)
        if err is not None:
            return f"error: {err}"
        return f"{drag.label} -> {location_id}"

    return None


def _draw_capture_inventory_overlay(ui: object, inventory_lines: list[str], manual_actions: list[str]) -> None:
    if not config.P2_SHOW_CAPTURE_INVENTORY_OVERLAY:
        return
    if not inventory_lines and not manual_actions and not config.P2_SHOW_EMPTY_CAPTURE_INVENTORY_OVERLAY:
        return

    screen = ui.screen
    sidebar_x = int(getattr(ui, "sidebar_x", screen.get_width() - 260))
    sidebar_w = int(getattr(ui, "sidebar_w", 240))
    w = max(180, min(sidebar_w + 12, screen.get_width() - 24))
    max_lines = max(1, config.P2_CAPTURE_INVENTORY_OVERLAY_MAX_LINES)
    shown_inventory = inventory_lines[:max_lines]
    if len(inventory_lines) > max_lines:
        shown_inventory.append(f"... ({len(inventory_lines) - max_lines} more)")

    lines = [f"Capture Inventory ({len(inventory_lines)})", *shown_inventory]
    if manual_actions:
        lines.append("MANUAL ACTION REQUIRED:")
        lines.extend(manual_actions[:3])

    line_h = 18
    h = 10 + line_h * len(lines) + 8
    x = max(12, min(sidebar_x - 6, screen.get_width() - w - 12))
    y = max(56, screen.get_height() - h - 12)
    panel = pygame.Surface((w, h), pygame.SRCALPHA)
    panel.fill((15, 18, 24, 205))
    screen.blit(panel, (x, y))
    pygame.draw.rect(screen, (120, 150, 180), pygame.Rect(x, y, w, h), width=1)

    for i, text in enumerate(lines):
        color = (230, 235, 245)
        if i == 0:
            color = (255, 230, 130)
        if text.startswith("MANUAL ACTION REQUIRED"):
            color = (255, 120, 120)
        surf = ui.font.render(text, True, color)
        max_text_w = max(1, w - 16)
        if surf.get_width() > max_text_w:
            clipped = text
            while len(clipped) > 4 and ui.font.render(clipped + "...", True, color).get_width() > max_text_w:
                clipped = clipped[:-1]
            surf = ui.font.render(clipped + "...", True, color)
        screen.blit(surf, (x + 8, y + 6 + i * line_h))


def _make_status_notice(
    *,
    level: str,
    title: str,
    message: str,
    details: list[str] | None = None,
    sticky: bool = False,
    duration_ms: int = 0,
) -> _StatusNotice:
    now_ms = pygame.time.get_ticks()
    expires_at_ms = 0 if sticky or duration_ms <= 0 else now_ms + duration_ms
    return _StatusNotice(
        level=level,
        title=title,
        message=message,
        details=list(details or []),
        sticky=sticky,
        expires_at_ms=expires_at_ms,
    )


def _status_notice_from_control_message(msg: dict[str, object]) -> _StatusNotice:
    raw_details = msg.get("details")
    details: list[str] = []
    if isinstance(raw_details, list):
        details = [str(item) for item in raw_details if str(item).strip()]
    return _make_status_notice(
        level=str(msg.get("level", "info")),
        title=str(msg.get("title", "Status")),
        message=str(msg.get("message", "")),
        details=details,
        sticky=bool(msg.get("sticky", False)),
        duration_ms=int(msg.get("duration_ms", 0) or 0),
    )


def _poll_geometry_controls(
    transport: object,
    preview: _GeometryPreview | None,
    calibration: _GeometryCalibration | None,
    status_notice: _StatusNotice | None,
) -> tuple[_GeometryPreview | None, _GeometryCalibration | None, _StatusNotice | None]:
    poll = getattr(transport, "poll_control_message", None)
    if not callable(poll):
        return preview, calibration, status_notice
    while True:
        msg = poll()
        if msg is None:
            return preview, calibration, status_notice
        msg_type = msg.get("type")
        if msg_type == "geometry_preview":
            try:
                preview = _GeometryPreview(msg)
                calibration = None
            except Exception as exc:  # noqa: BLE001
                print(f"Could not display geometry preview: {exc}", file=sys.stderr)
                sender = getattr(transport, "send_control_message", None)
                if callable(sender):
                    sender({"type": "geometry_confirm", "accepted": False, "error": str(exc)})
                preview = None
            continue
        if msg_type == "status":
            try:
                status_notice = _status_notice_from_control_message(msg)
            except Exception as exc:  # noqa: BLE001
                print(f"Could not display status message: {exc}", file=sys.stderr)
            continue
        if msg_type == "geometry_calibration_request":
            preview = None
            calibration = None
            _run_external_geometry_selector(msg, transport)
            continue
        else:
            print(f"Unhandled control message: {msg}", file=sys.stderr)
            continue


def _send_geometry_confirmation(transport: object, accepted: bool) -> None:
    sender = getattr(transport, "send_control_message", None)
    if callable(sender):
        sender({"type": "geometry_confirm", "accepted": bool(accepted)})


def _send_geometry_calibration_result(
    transport: object,
    calibration: _GeometryCalibration,
    accepted: bool,
) -> None:
    sender = getattr(transport, "send_control_message", None)
    if not callable(sender):
        return
    payload: dict[str, object] = {
        "type": "geometry_calibration_result",
        "accepted": bool(accepted),
        "source_path": calibration.source_path,
    }
    if accepted:
        payload["outer_corners_px"] = [[x, y] for x, y in calibration.outer_points]
        payload["chessboard_corners_px"] = [[x, y] for x, y in calibration.inner_points]
    sender(payload)


def _send_geometry_calibration_payload(
    transport: object,
    payload: dict[str, object],
) -> None:
    sender = getattr(transport, "send_control_message", None)
    if callable(sender):
        sender(payload)


def _send_runtime_control_message(transport: object, payload: dict[str, object]) -> bool:
    sender = getattr(transport, "send_control_message", None)
    if not callable(sender):
        return False
    try:
        sender(payload)
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"Failed to send runtime control message: {exc}", file=sys.stderr)
        return False


def _current_status_notice(
    kind: str,
    state: object,
    waiting_for_p1: bool,
    ai_mode: bool,
    notice: _StatusNotice | None,
) -> _StatusNotice:
    now_ms = pygame.time.get_ticks()
    if notice is not None:
        if notice.sticky or notice.expires_at_ms <= 0 or now_ms < notice.expires_at_ms:
            return notice

    if waiting_for_p1:
        if kind == "checkers" and hasattr(state, "p1_must_continue_jump") and state.p1_must_continue_jump():
            forced = state.p1_forced_square_id() if hasattr(state, "p1_forced_square_id") else None
            suffix = f" Continue with {forced}." if forced else ""
            return _make_status_notice(
                level="info",
                title="Waiting For Player 1",
                message=f"Physical multi-jump is still in progress.{suffix}",
                duration_ms=0,
            )
        return _make_status_notice(
            level="info",
            title="Waiting For Player 1",
            message="Make the physical move on the board, then let the Pi capture and validate it.",
            duration_ms=0,
        )

    if kind == "checkers" and hasattr(state, "p2_must_continue_jump") and state.p2_must_continue_jump():
        forced = state.p2_forced_square_id() if hasattr(state, "p2_forced_square_id") else None
        return _make_status_notice(
            level="warning",
            title="Continue Jump",
            message=(
                f"Player 2 must continue the jump with {forced} before the turn can end."
                if forced
                else "Player 2 must continue the jump before the turn can end."
            ),
            duration_ms=0,
        )

    if ai_mode:
        return _make_status_notice(
            level="success",
            title="Player 2 Turn",
            message="AI is selecting and dispatching the next move.",
            duration_ms=0,
        )

    return _make_status_notice(
        level="success",
        title="Player 2 Turn",
        message="Drag and drop the next software move.",
        duration_ms=0,
    )


def _draw_status_notice(
    screen: pygame.Surface,
    font: pygame.font.Font,
    notice: _StatusNotice,
    ui: object | None = None,
) -> None:
    palette = {
        "info": ((18, 28, 44, 220), (108, 163, 230), (235, 242, 250), (180, 205, 230)),
        "success": ((20, 42, 28, 220), (93, 196, 132), (236, 248, 239), (177, 221, 190)),
        "warning": ((55, 38, 14, 220), (236, 178, 70), (250, 242, 225), (235, 205, 150)),
        "error": ((58, 19, 20, 220), (224, 102, 102), (250, 235, 235), (235, 180, 180)),
    }
    bg_rgba, border, title_color, body_color = palette.get(notice.level, palette["info"])

    lines = [notice.message, *notice.details]
    header_margin = 12
    header_left = header_margin
    header_right = screen.get_width() - header_margin
    header_top = 52
    if ui is not None:
        back_rect = _back_button_rect(ui)
        mode_rect = _mode_button_rect(ui)
        header_left = back_rect.right + 18
        header_right = mode_rect.left - 18
        header_top = back_rect.top
    available_w = max(280, header_right - header_left)
    panel_w = min(available_w, 720)
    line_h = 18
    panel_h = 18 + 22 + (line_h * len(lines)) + 12
    x = header_left + max(0, (available_w - panel_w) // 2)
    y = header_top

    panel = pygame.Surface((panel_w, panel_h), pygame.SRCALPHA)
    panel.fill(bg_rgba)
    screen.blit(panel, (x, y))
    pygame.draw.rect(screen, border, pygame.Rect(x, y, panel_w, panel_h), width=2, border_radius=12)

    title = font.render(notice.title, True, title_color)
    screen.blit(title, (x + 12, y + 10))

    for idx, text in enumerate(lines):
        surf = font.render(text, True, body_color)
        max_text_w = panel_w - 24
        if surf.get_width() > max_text_w:
            clipped = text
            while len(clipped) > 4 and font.render(clipped + "...", True, body_color).get_width() > max_text_w:
                clipped = clipped[:-1]
            surf = font.render(clipped + "...", True, body_color)
        screen.blit(surf, (x + 12, y + 36 + idx * line_h))


def _run_external_geometry_selector(msg: dict[str, object], transport: object) -> None:
    image_b64 = str(msg.get("image_png_b64", ""))
    if not image_b64:
        _send_geometry_calibration_payload(
            transport,
            {
                "type": "geometry_calibration_result",
                "accepted": False,
                "error": "geometry_calibration_request missing image_png_b64",
            },
        )
        return

    request_dir = ROOT / "debug_output" / "manual_geometry"
    request_dir.mkdir(parents=True, exist_ok=True)
    image_path = request_dir / "startup_geometry_request.png"
    image_path.write_bytes(base64.b64decode(image_b64))

    selector_script = ROOT / "tools" / "manual_geometry_selector.py"
    cmd = [
        sys.executable,
        str(selector_script),
        "--image",
        str(image_path),
        "--source-path",
        str(msg.get("source_path", "")),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
    except Exception as exc:  # noqa: BLE001
        _send_geometry_calibration_payload(
            transport,
            {"type": "geometry_calibration_result", "accepted": False, "error": str(exc)},
        )
        return

    if result.returncode != 0:
        _send_geometry_calibration_payload(
            transport,
            {
                "type": "geometry_calibration_result",
                "accepted": False,
                "error": result.stderr.strip() or f"selector exited {result.returncode}",
            },
        )
        return

    try:
        payload = json.loads(result.stdout.strip())
    except json.JSONDecodeError as exc:
        _send_geometry_calibration_payload(
            transport,
            {
                "type": "geometry_calibration_result",
                "accepted": False,
                "error": f"selector returned invalid JSON: {exc}",
            },
        )
        return

    if not isinstance(payload, dict):
        payload = {"type": "geometry_calibration_result", "accepted": False, "error": "selector returned non-object"}
    payload["type"] = "geometry_calibration_result"
    payload["source_path"] = str(msg.get("source_path", ""))
    _send_geometry_calibration_payload(transport, payload)


def _wrap_preview_text(font: pygame.font.Font, text: str, max_width: int) -> list[str]:
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = word if not current else f"{current} {word}"
        if font.render(candidate, True, (230, 230, 235)).get_width() <= max_width:
            current = candidate
            continue
        if current:
            lines.append(current)
        current = word
    if current:
        lines.append(current)
    return lines[:3]


def _draw_geometry_preview(
    screen: pygame.Surface,
    font: pygame.font.Font,
    preview: _GeometryPreview,
) -> None:
    overlay = pygame.Surface(screen.get_size(), pygame.SRCALPHA)
    overlay.fill((0, 0, 0, 190))
    screen.blit(overlay, (0, 0))

    margin = 28
    panel_w = screen.get_width() - margin * 2
    panel_h = screen.get_height() - margin * 2
    panel = pygame.Rect(margin, margin, panel_w, panel_h)
    pygame.draw.rect(screen, (24, 28, 34), panel, border_radius=14)
    pygame.draw.rect(screen, (150, 170, 190), panel, width=1, border_radius=14)

    title_font = pygame.font.SysFont("sf pro display, helvetica, arial", 26, bold=True)
    title = title_font.render(preview.title, True, (248, 248, 250))
    screen.blit(title, (panel.x + 20, panel.y + 16))

    text_y = panel.y + 52
    for line in _wrap_preview_text(font, preview.summary, panel_w - 40):
        surf = font.render(line, True, (218, 224, 232))
        screen.blit(surf, (panel.x + 20, text_y))
        text_y += 23

    if preview.source_path:
        path_text = font.render(preview.source_path, True, (145, 156, 170))
        screen.blit(path_text, (panel.x + 20, text_y + 2))

    buttons_y = panel.bottom - 56
    preview.confirm_rect = pygame.Rect(panel.right - 272, buttons_y, 120, 38)
    preview.cancel_rect = pygame.Rect(panel.right - 142, buttons_y, 110, 38)

    image_top = panel.y + 104
    image_bottom = buttons_y - 16
    max_w = panel_w - 40
    max_h = max(80, image_bottom - image_top)
    iw, ih = preview.image.get_size()
    scale = min(max_w / max(iw, 1), max_h / max(ih, 1), 1.0)
    scaled_size = (max(1, int(iw * scale)), max(1, int(ih * scale)))
    shown = pygame.transform.smoothscale(preview.image, scaled_size)
    image_rect = shown.get_rect(center=(screen.get_width() // 2, image_top + max_h // 2))
    screen.blit(shown, image_rect)
    pygame.draw.rect(screen, (85, 95, 110), image_rect, width=1)

    mx, my = pygame.mouse.get_pos()
    for rect, label, good in [
        (preview.confirm_rect, "Confirm", True),
        (preview.cancel_rect, "Cancel", False),
    ]:
        hover = rect.collidepoint(mx, my)
        if good:
            bg = (35, 105, 62) if not hover else (47, 130, 78)
            border = (120, 220, 150)
        else:
            bg = (92, 54, 54) if not hover else (118, 66, 66)
            border = (215, 130, 130)
        pygame.draw.rect(screen, bg, rect, border_radius=10)
        pygame.draw.rect(screen, border, rect, width=1, border_radius=10)
        lab = font.render(label, True, (248, 248, 250))
        screen.blit(lab, lab.get_rect(center=rect.center))


def _handle_geometry_preview_event(
    preview: _GeometryPreview | None,
    event: pygame.event.Event,
    transport: object,
) -> tuple[_GeometryPreview | None, bool]:
    if preview is None:
        return preview, False
    if event.type == pygame.KEYDOWN:
        if event.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
            _send_geometry_confirmation(transport, True)
            return None, True
        if event.key == pygame.K_ESCAPE:
            _send_geometry_confirmation(transport, False)
            return None, True
    if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
        if preview.confirm_rect.collidepoint(*event.pos):
            _send_geometry_confirmation(transport, True)
            return None, True
        if preview.cancel_rect.collidepoint(*event.pos):
            _send_geometry_confirmation(transport, False)
            return None, True
    return preview, True


def _draw_geometry_calibration(
    screen: pygame.Surface,
    font: pygame.font.Font,
    calibration: _GeometryCalibration,
) -> None:
    overlay = pygame.Surface(screen.get_size(), pygame.SRCALPHA)
    overlay.fill((0, 0, 0, 205))
    screen.blit(overlay, (0, 0))

    margin = 22
    panel = pygame.Rect(margin, margin, screen.get_width() - 2 * margin, screen.get_height() - 2 * margin)
    pygame.draw.rect(screen, (23, 27, 34), panel, border_radius=14)
    pygame.draw.rect(screen, (150, 170, 190), panel, width=1, border_radius=14)

    title_font = pygame.font.SysFont("sf pro display, helvetica, arial", 25, bold=True)
    title = title_font.render(calibration.title, True, (248, 248, 250))
    screen.blit(title, (panel.x + 18, panel.y + 14))

    instruction = calibration.next_instruction()
    progress = f"Outer: {len(calibration.outer_points)}/4    Inner: {len(calibration.inner_points)}/4"
    text_lines = [calibration.summary, instruction, progress]
    text_y = panel.y + 48
    for line in text_lines:
        color = (238, 224, 140) if line == instruction else (218, 224, 232)
        surf = font.render(line, True, color)
        screen.blit(surf, (panel.x + 18, text_y))
        text_y += 22

    if calibration.source_path:
        source = font.render(calibration.source_path, True, (145, 156, 170))
        screen.blit(source, (panel.x + 18, text_y))

    buttons_y = panel.bottom - 54
    calibration.save_rect = pygame.Rect(panel.right - 382, buttons_y, 92, 36)
    calibration.undo_rect = pygame.Rect(panel.right - 282, buttons_y, 82, 36)
    calibration.reset_rect = pygame.Rect(panel.right - 192, buttons_y, 82, 36)
    calibration.cancel_rect = pygame.Rect(panel.right - 102, buttons_y, 82, 36)

    image_top = panel.y + 126
    image_bottom = buttons_y - 14
    max_w = panel.width - 36
    max_h = max(80, image_bottom - image_top)
    iw, ih = calibration.image.get_size()
    scale = min(max_w / max(iw, 1), max_h / max(ih, 1), 1.0)
    scaled_size = (max(1, int(iw * scale)), max(1, int(ih * scale)))
    shown = pygame.transform.smoothscale(calibration.image, scaled_size)
    calibration.image_rect = shown.get_rect(center=(screen.get_width() // 2, image_top + max_h // 2))
    screen.blit(shown, calibration.image_rect)
    pygame.draw.rect(screen, (85, 95, 110), calibration.image_rect, width=1)

    def image_to_screen(point: tuple[float, float]) -> tuple[int, int]:
        return (
            int(calibration.image_rect.x + point[0] * calibration.image_rect.width / max(iw, 1)),
            int(calibration.image_rect.y + point[1] * calibration.image_rect.height / max(ih, 1)),
        )

    def draw_points(points: list[tuple[float, float]], color: tuple[int, int, int], label_prefix: str) -> None:
        screen_pts = [image_to_screen(p) for p in points]
        if len(screen_pts) >= 2:
            pygame.draw.lines(screen, color, len(screen_pts) == 4, screen_pts, width=3)
        for i, pt in enumerate(screen_pts, start=1):
            pygame.draw.circle(screen, color, pt, 6)
            pygame.draw.circle(screen, (10, 10, 10), pt, 6, width=1)
            label = font.render(f"{label_prefix}{i}", True, color)
            screen.blit(label, (pt[0] + 7, pt[1] - 8))

    draw_points(calibration.outer_points, (40, 230, 95), "G")
    draw_points(calibration.inner_points, (245, 215, 45), "Y")

    mx, my = pygame.mouse.get_pos()
    button_specs = [
        (calibration.save_rect, "Save", calibration.is_complete(), (35, 105, 62), (120, 220, 150)),
        (calibration.undo_rect, "Undo", bool(calibration.outer_points or calibration.inner_points), (70, 72, 86), (145, 150, 170)),
        (calibration.reset_rect, "Reset", bool(calibration.outer_points or calibration.inner_points), (70, 72, 86), (145, 150, 170)),
        (calibration.cancel_rect, "Cancel", True, (92, 54, 54), (215, 130, 130)),
    ]
    for rect, label, enabled, bg_base, border in button_specs:
        hover = rect.collidepoint(mx, my)
        bg = tuple(min(255, c + 22) for c in bg_base) if hover and enabled else bg_base
        if not enabled:
            bg = (48, 50, 58)
            border = (85, 90, 105)
        pygame.draw.rect(screen, bg, rect, border_radius=9)
        pygame.draw.rect(screen, border, rect, width=1, border_radius=9)
        lab = font.render(label, True, (248, 248, 250) if enabled else (150, 154, 164))
        screen.blit(lab, lab.get_rect(center=rect.center))


def _handle_geometry_calibration_event(
    calibration: _GeometryCalibration | None,
    event: pygame.event.Event,
    transport: object,
) -> tuple[_GeometryCalibration | None, bool]:
    if calibration is None:
        return calibration, False
    if event.type == pygame.KEYDOWN:
        if event.key in (pygame.K_RETURN, pygame.K_KP_ENTER) and calibration.is_complete():
            _send_geometry_calibration_result(transport, calibration, True)
            return None, True
        if event.key == pygame.K_ESCAPE:
            _send_geometry_calibration_result(transport, calibration, False)
            return None, True
        if event.key in (pygame.K_BACKSPACE, pygame.K_DELETE, pygame.K_u):
            calibration.undo()
            return calibration, True
        if event.key == pygame.K_r:
            calibration.reset()
            return calibration, True
    if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
        if calibration.save_rect.collidepoint(*event.pos):
            if calibration.is_complete():
                _send_geometry_calibration_result(transport, calibration, True)
                return None, True
            return calibration, True
        if calibration.undo_rect.collidepoint(*event.pos):
            calibration.undo()
            return calibration, True
        if calibration.reset_rect.collidepoint(*event.pos):
            calibration.reset()
            return calibration, True
        if calibration.cancel_rect.collidepoint(*event.pos):
            _send_geometry_calibration_result(transport, calibration, False)
            return None, True
        calibration.add_click(event.pos)
        return calibration, True
    return calibration, True


def _draw_manual_fix_modal(
    ui: object,
    state: object,
    session: _ManualFixSession,
) -> None:
    screen = ui.screen
    font = ui.font
    panel = pygame.Rect(
        max(16, getattr(ui, "sidebar_x", screen.get_width() - 240) - 20),
        getattr(ui, "header_height", 128) + 44,
        228,
        screen.get_height() - getattr(ui, "header_height", 128) - 96,
    )
    pygame.draw.rect(screen, (24, 28, 34), panel, border_radius=14)
    pygame.draw.rect(screen, (150, 170, 190), panel, width=1, border_radius=14)

    title_font = pygame.font.SysFont("sf pro display, helvetica, arial", 25, bold=True)
    title = title_font.render("Manual Fix", True, (248, 248, 250))
    screen.blit(title, (panel.x + 18, panel.y + 14))
    text_y = panel.y + 56

    mx, my = pygame.mouse.get_pos()
    for rect, label, value in _manual_fix_palette_layout(ui, session, state):
        hover = rect.collidepoint(mx, my)
        _draw_manual_fix_palette_entry(ui, screen, font, session.kind, rect, label, value, hover)

    session.done_rect = pygame.Rect(panel.right - 214, panel.bottom - 52, 90, 36)
    session.cancel_rect = pygame.Rect(panel.right - 114, panel.bottom - 52, 90, 36)
    for rect, label, good in [
        (session.done_rect, "Done", True),
        (session.cancel_rect, "Cancel", False),
    ]:
        hover = rect.collidepoint(mx, my)
        if good:
            bg = (35, 105, 62) if not hover else (47, 130, 78)
            border = (120, 220, 150)
        else:
            bg = (92, 54, 54) if not hover else (118, 66, 66)
            border = (215, 130, 130)
        pygame.draw.rect(screen, bg, rect, border_radius=10)
        pygame.draw.rect(screen, border, rect, width=1, border_radius=10)
        surf = font.render(label, True, (248, 248, 250))
        screen.blit(surf, surf.get_rect(center=rect.center))

    if session.dragging is not None:
        _draw_manual_fix_drag_ghost(ui, screen, font, session.kind, session.dragging)


def _draw_menu(screen: pygame.Surface, font: pygame.font.Font, buttons: list[_Button]) -> None:
    screen.fill((32, 32, 38))
    title_font = pygame.font.SysFont("sf pro display, helvetica, arial", 34, bold=True)
    title = title_font.render("Choose a game", True, (240, 240, 245))
    screen.blit(title, title.get_rect(center=(screen.get_width() // 2, 120)))

    for b in buttons:
        mx, my = pygame.mouse.get_pos()
        hover = b.rect.collidepoint(mx, my)
        bg = (70, 70, 86) if hover else (55, 55, 68)
        pygame.draw.rect(screen, bg, b.rect, border_radius=14)
        pygame.draw.rect(screen, (95, 95, 115), b.rect, width=1, border_radius=14)
        lab = font.render(b.label, True, (245, 245, 250))
        screen.blit(lab, lab.get_rect(center=b.rect.center))

    hint = font.render("Esc: back to menu   R: reset board", True, (170, 170, 180))
    screen.blit(hint, hint.get_rect(center=(screen.get_width() // 2, screen.get_height() - 60)))

    pygame.display.flip()


def _make_transport() -> object:
    if config.TRANSPORT == "tcp":
        transport: object = TcpClientTransport(config.TCP_HOST, config.TCP_PORT)
        print(
            f"Connecting to Pi bridge at {config.TCP_HOST}:{config.TCP_PORT} "
            f"for up to {config.TCP_CONNECT_RETRIES * config.TCP_CONNECT_RETRY_DELAY_SEC:.0f}s...",
            file=sys.stderr,
        )
        transport.connect(
            retries=config.TCP_CONNECT_RETRIES,
            retry_delay_sec=config.TCP_CONNECT_RETRY_DELAY_SEC,
        )
        print("Connected to Pi bridge.", file=sys.stderr)
        return transport

    mock = MockTransport()
    mock.connect()
    threading.Thread(target=_stdin_p1_injector, args=(mock,), daemon=True).start()
    print(
        "Mock mode: paste JSON lines to stdin, e.g.",
        '{"type":"p1_move","from":"c3","to":"d4"}',
        file=sys.stderr,
    )
    return mock


def _run_game_loop(kind: str, transport: object) -> None:
    _send_runtime_control_message(
        transport,
        {"type": "set_game", "game": kind, "reason": "ui_game_selected"},
    )
    waiting_for_p1 = True
    ai_mode = config.CONTROL_MODE == "ai"
    ai_pending = False
    last_move: tuple[str, str] | None = None
    capture_inventory = CaptureInventory(kind)
    latest_manual_actions: list[str] = []

    if kind == "checkers":
        state: object = CheckersState()
        ui: object = BoardUI(config.WINDOW_WIDTH, config.WINDOW_HEIGHT)
    elif kind == "chess":
        state = ChessState()
        ui = ChessUI(config.WINDOW_WIDTH, config.WINDOW_HEIGHT)
    elif kind == "parcheesi":
        state = ParcheesiState()
        ui = ParcheesiUI(config.WINDOW_WIDTH, config.WINDOW_HEIGHT)
        pygame.display.set_caption("Player 2 — Parcheesi")
        if config.TRANSPORT == "mock":
            state.current_player = 2
            waiting_for_p1 = False
    else:
        raise ValueError(f"Unsupported game kind: {kind}")

    geometry_preview: _GeometryPreview | None = None
    geometry_calibration: _GeometryCalibration | None = None
    manual_fix_session: _ManualFixSession | None = None
    status_notice: _StatusNotice | None = None
    pending_checkers_turn = _QueuedCheckersTurn()

    def _new_captured_by_p1(state_before: object, state_after: object) -> list[object]:
        before = getattr(state_before, "captured_by_p1", None)
        after = getattr(state_after, "captured_by_p1", None)
        if not isinstance(before, list) or not isinstance(after, list):
            return []
        if len(after) < len(before):
            return []
        return after[len(before):]

    def _record_manual_green_captures(
        state_before: object,
        state_after: object,
        msg: P1MoveMessage,
    ) -> tuple[int, int]:
        new_captured = _new_captured_by_p1(state_before, state_after)
        expected_records: list[tuple[str, object]] = [("p2", piece) for piece in new_captured]

        if kind == "chess" and isinstance(state_before, ChessState):
            try:
                start_sq = parse_square(msg.frm)
                end_sq = parse_square(msg.to)
                legal_move = state_before._choose_legal_move(start_sq, end_sq)
            except Exception:  # noqa: BLE001
                legal_move = None
            if legal_move is not None and legal_move.promotion is not None:
                promoted_from_piece = state_before.get(start_sq)
                if promoted_from_piece != ChessPiece.EMPTY:
                    expected_records.append(("p1", promoted_from_piece))
                promoted_piece = state_after.get(end_sq) if isinstance(state_after, ChessState) else ChessPiece.EMPTY
                if promoted_piece != ChessPiece.EMPTY:
                    pulled = capture_inventory.take_piece("p1", _piece_name(promoted_piece))
                    if pulled is not None:
                        print(
                            "[PROMOTION] Reused "
                            f"{pulled.captured_side}[{pulled.slot_index}]="
                            f"{pulled.slot.x_pct:.2f}%,{pulled.slot.y_pct:.2f}%:{pulled.piece_name}",
                            file=sys.stderr,
                        )
                    else:
                        print(
                            f"[PROMOTION] No reusable {_piece_name(promoted_piece)} found in capture inventory for P1 promotion.",
                            file=sys.stderr,
                        )

        if not expected_records:
            return (0, 0)

        raw_captures = msg.manual_green_captures or []
        if not raw_captures and kind != "chess":
            return (0, 0)
        tracked = 0
        for (captured_side, piece), raw in zip(expected_records, raw_captures):
            if not isinstance(raw, dict):
                continue
            try:
                x_pct = float(raw["x_pct"])
                y_pct = float(raw["y_pct"])
            except (KeyError, TypeError, ValueError):
                continue
            capture_inventory.add_manual_capture(
                captured_side,
                _piece_name(piece),
                PercentEndpoint(x_pct=x_pct, y_pct=y_pct),
            )
            tracked += 1
        return (tracked, len(expected_records))

    # Add a "Roll Dice" button for Parcheesi.
    roll_dice_button_rect = pygame.Rect(
        ui.screen.get_width() - 202,
        ui.screen.get_height() - 60,
        180,
        40,
    )

    def _dispatch_p2_turn(
        msg: P2MoveMessage,
        *,
        capture_slots_used: list[str],
        temporary_relocations: int,
        fallback_direct_segments: int,
        manual_actions: list[str],
    ) -> None:
        nonlocal waiting_for_p1, ai_pending, last_move, latest_manual_actions, status_notice

        latest_manual_actions = manual_actions[:]
        transport.send_p2_move(msg)
        if config.TRANSPORT == "mock":
            print(json.dumps(msg.to_obj(), separators=(",", ":")), flush=True, file=sys.stderr)

        if config.WRITE_STM_SEQUENCE:
            write_started_at = time.perf_counter()
            out_path = write_sequence_file(
                config.STM_SEQUENCE_FILE,
                msg.stm_sequence or [],
                game=kind,
                start_id=msg.frm,
                end_id=msg.to,
                capture_slots_used=capture_slots_used,
                capture_inventory_summary=capture_inventory.to_summary_strings(),
                manual_actions=manual_actions,
            )
            print(
                "STM sequence updated:"
                f" {out_path} (capture={bool(capture_slots_used)},"
                f" temp_relocations={temporary_relocations},"
                f" fallback_direct={fallback_direct_segments},"
                f" steps={len(msg.stm_sequence or [])},"
                f" capture_slots={len(capture_slots_used)})"
                f" write_time={time.perf_counter() - write_started_at:.3f}s",
                file=sys.stderr,
            )
            if manual_actions:
                for act in manual_actions:
                    print(f"[MANUAL] {act}", file=sys.stderr)
            print(
                "[INVENTORY] " + ", ".join(capture_inventory.to_summary_strings()) if capture_inventory.to_summary_strings() else "[INVENTORY] <empty>",
                file=sys.stderr,
            )

        if kind == "checkers" and msg.turn_steps and len(msg.turn_steps) > 1:
            chain = " | ".join(f"{step['from']}->{step['to']}" for step in msg.turn_steps)
            print(f"P2 checkers turn: {chain}", file=sys.stderr)

        if kind == "chess" and hasattr(state, "is_checkmate"):
            if state.is_checkmate():
                print(f"Game over: checkmate. Winner: {'P2' if state.turn_side() == 'p1' else 'P1'}", file=sys.stderr)
                status_notice = _make_status_notice(
                    level="error",
                    title="Checkmate",
                    message=f"Game over. Winner: {'P2' if state.turn_side() == 'p1' else 'P1'}. Press R to play again.",
                    sticky=True,
                )
                waiting_for_p1 = True
                ai_pending = False
                return
            elif state.is_stalemate():
                print("Game over: stalemate.", file=sys.stderr)
                status_notice = _make_status_notice(
                    level="warning",
                    title="Stalemate",
                    message="Game over. Stalemate. Press R to play again.",
                    sticky=True,
                )
                waiting_for_p1 = True
                ai_pending = False
                return
            elif state.is_check():
                print(f"Check on {state.turn_side().upper()}.", file=sys.stderr)
                status_notice = _make_status_notice(
                    level="warning",
                    title="Check",
                    message=f"Check on {state.turn_side().upper()}.",
                    duration_ms=1800,
                )

        if kind == "parcheesi":
            if state.check_win_condition(2):
                print("Game over: Player 2 wins.", file=sys.stderr)
                waiting_for_p1 = True
                ai_pending = False
            elif state.rolls_remaining and state.get_possible_moves(2):
                waiting_for_p1 = False
                ai_pending = ai_mode
            else:
                state.current_player = 1
                state.clear_dice()
                waiting_for_p1 = True
                ai_pending = False
            return

        waiting_for_p1 = True
        ai_pending = False
        status_notice = _make_status_notice(
            level="success",
            title="Player 2 Move Sent",
            message="Move dispatched to the Pi bridge and STM planner.",
            duration_ms=1800,
        )

    def _apply_and_dispatch_p2_move(start_id: str, end_id: str) -> None:
        nonlocal waiting_for_p1, ai_pending, last_move, latest_manual_actions, status_notice

        state_before = state.copy()
        err = state.try_apply_p2_move(start_id, end_id)
        if err is not None:
            print("Illegal P2 move:", err, file=sys.stderr)
            return

        ui.clear_drag()
        last_move = (start_id, end_id)
        print(f"P2 move: {start_id} -> {end_id}", file=sys.stderr)

        planner_started_at = time.perf_counter()
        generated = generate_p2_sequence(
            kind,
            state_before,
            start_id,
            end_id,
            capture_inventory=capture_inventory,
        )
        print(
            "Motion planner completed:"
            f" {time.perf_counter() - planner_started_at:.3f}s"
            f" (capture={generated.capture_detected},"
            f" temp_relocations={generated.temporary_relocations},"
            f" fallback_direct={generated.fallback_direct_segments},"
            f" stm_steps={len(generated.lines)})",
            file=sys.stderr,
        )
        if kind == "checkers":
            pending_checkers_turn.append(start_id, end_id, generated)
            latest_manual_actions = pending_checkers_turn.manual_actions[:]
        else:
            latest_manual_actions = generated.manual_actions[:]

        if kind == "checkers" and hasattr(state, "p2_must_continue_jump") and state.p2_must_continue_jump():
            forced = state.p2_forced_square_id() if hasattr(state, "p2_forced_square_id") else None
            if forced:
                print(f"Checkers: must continue jump with {forced}", file=sys.stderr)
                status_notice = _make_status_notice(
                    level="warning",
                    title="Continue Jump",
                    message=f"Player 2 must continue the jump with {forced}.",
                    duration_ms=2200,
                )
            waiting_for_p1 = False
            ai_pending = ai_mode
            return

        if kind == "checkers" and pending_checkers_turn.steps:
            turn_steps = [
                {"from": frm, "to": to}
                for frm, to in pending_checkers_turn.steps
            ]
            msg = P2MoveMessage(
                frm=pending_checkers_turn.steps[0][0],
                to=pending_checkers_turn.steps[-1][1],
                game=kind,
                stm_sequence=pending_checkers_turn.stm_lines[:],
                manual_actions=(pending_checkers_turn.manual_actions[:] if pending_checkers_turn.manual_actions else None),
                turn_steps=turn_steps,
            )
            _dispatch_p2_turn(
                msg,
                capture_slots_used=pending_checkers_turn.capture_slots_used[:],
                temporary_relocations=pending_checkers_turn.temporary_relocations,
                fallback_direct_segments=pending_checkers_turn.fallback_direct_segments,
                manual_actions=pending_checkers_turn.manual_actions[:],
            )
            pending_checkers_turn.clear()
            return

        msg = P2MoveMessage(
            frm=start_id,
            to=end_id,
            game=kind,
            stm_sequence=generated.lines,
            manual_actions=(generated.manual_actions[:] if generated.manual_actions else None),
        )
        _dispatch_p2_turn(
            msg,
            capture_slots_used=generated.capture_slots_used,
            temporary_relocations=generated.temporary_relocations,
            fallback_direct_segments=generated.fallback_direct_segments,
            manual_actions=generated.manual_actions,
        )

    while True:
        geometry_preview, geometry_calibration, status_notice = _poll_geometry_controls(
            transport, geometry_preview, geometry_calibration, status_notice
        )

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                raise SystemExit(0)

            geometry_calibration, consumed_by_calibration = _handle_geometry_calibration_event(
                geometry_calibration, event, transport
            )
            if consumed_by_calibration:
                continue

            geometry_preview, consumed_by_preview = _handle_geometry_preview_event(
                geometry_preview, event, transport
            )
            if consumed_by_preview:
                continue

            if manual_fix_session is not None:
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    state = manual_fix_session.backup_state.copy()
                    ui.clear_drag()
                    duration_s = time.perf_counter() - manual_fix_session.opened_at
                    print(f"Manual fix canceled after {duration_s:.3f}s", file=sys.stderr)
                    status_notice = _make_status_notice(
                        level="info",
                        title="Manual Fix Canceled",
                        message="Restored the software state to the snapshot from when manual fix opened.",
                        duration_ms=2200,
                    )
                    manual_fix_session = None
                    continue

                if event.type == pygame.MOUSEMOTION and manual_fix_session.dragging is not None:
                    manual_fix_session.dragging.mouse_pos = event.pos
                    dx = event.pos[0] - manual_fix_session.dragging.origin_pos[0]
                    dy = event.pos[1] - manual_fix_session.dragging.origin_pos[1]
                    if (dx * dx + dy * dy) >= 25:
                        manual_fix_session.dragging.moved = True
                    continue

                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    if manual_fix_session.done_rect.collidepoint(*event.pos):
                        duration_s = time.perf_counter() - manual_fix_session.opened_at
                        refresh_requested = _send_runtime_control_message(
                            transport,
                            {"type": "refresh_reference", "reason": "manual_fix_commit"},
                        )
                        print(f"Manual fix committed after {duration_s:.3f}s", file=sys.stderr)
                        status_notice = _make_status_notice(
                            level="success",
                            title="Manual Fix Applied",
                            message=(
                                "Kept the manual software-state corrections and requested a fresh reference capture from the Pi."
                                if refresh_requested
                                else "Kept the manual software-state corrections."
                            ),
                            duration_ms=2200,
                        )
                        manual_fix_session = None
                        continue
                    if manual_fix_session.cancel_rect.collidepoint(*event.pos):
                        state = manual_fix_session.backup_state.copy()
                        ui.clear_drag()
                        duration_s = time.perf_counter() - manual_fix_session.opened_at
                        print(f"Manual fix canceled after {duration_s:.3f}s", file=sys.stderr)
                        status_notice = _make_status_notice(
                            level="info",
                            title="Manual Fix Canceled",
                            message="Restored the software state to the snapshot from when manual fix opened.",
                            duration_ms=2200,
                        )
                        manual_fix_session = None
                        continue

                    for rect, label, value in _manual_fix_palette_layout(ui, manual_fix_session, state):
                        if rect.collidepoint(*event.pos):
                            manual_fix_session.dragging = _ManualFixDrag(
                                source_kind="palette",
                                source_ref=None,
                                piece=value,
                                label=label,
                                mouse_pos=event.pos,
                                origin_pos=event.pos,
                            )
                            break
                    else:
                        picked = _manual_fix_pick_from_board(kind, state, ui, event.pos)
                        if picked is not None:
                            source_ref, piece, origin = picked
                            manual_fix_session.dragging = _ManualFixDrag(
                                source_kind="board",
                                source_ref=source_ref,
                                piece=piece,
                                label=_manual_fix_piece_label(kind, piece),
                                mouse_pos=event.pos,
                                origin_pos=origin,
                            )
                    continue

                if event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                    if manual_fix_session.dragging is not None:
                        description = _manual_fix_drop_drag(manual_fix_session, state, ui, event.pos)
                        manual_fix_session.dragging = None
                        if description is not None:
                            ui.clear_drag()
                            print(f"[MANUAL_FIX] {description}", file=sys.stderr)
                            level = "warning" if description.startswith("error:") else "success"
                            title = "Manual Fix Error" if level == "warning" else "Manual Fix Applied"
                            message = description[7:] if description.startswith("error: ") else description
                            status_notice = _make_status_notice(
                                level=level,
                                title=title,
                                message=message,
                                duration_ms=2400,
                            )
                    continue

                continue

            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if _back_button_rect(ui).collidepoint(*event.pos):
                    if hasattr(ui, "clear_drag"):
                        ui.clear_drag()
                    return

                if _mode_button_rect(ui).collidepoint(*event.pos):
                    ai_mode = not ai_mode
                    if ai_mode and not waiting_for_p1:
                        ai_pending = True
                    mode_name = "AI (Stockfish)" if ai_mode else "Manual"
                    print(f"Control mode set to: {mode_name}", file=sys.stderr)
                    continue

                if _manual_fix_button_rect(ui).collidepoint(*event.pos):
                    ui.clear_drag()
                    manual_fix_session = _begin_manual_fix(kind, state)
                    print(f"Manual fix opened for {kind}", file=sys.stderr)
                    status_notice = _make_status_notice(
                        level="warning",
                        title="Manual Fix",
                        message="Editing the software state directly. Click Done to keep or Cancel to restore.",
                        sticky=True,
                    )
                    continue
                
                if kind == "parcheesi" and roll_dice_button_rect.collidepoint(*event.pos):
                    if state.current_player != 2:
                        print(f"Parcheesi: waiting for Player {state.current_player}, not P2.", file=sys.stderr)
                        continue
                    if state.rolls_remaining:
                        print(f"Parcheesi: finish remaining dice first: {state.rolls_remaining}", file=sys.stderr)
                        continue
                    state.roll_dice()
                    print(f"Player {state.current_player} rolled: {state.dice_rolls}", file=sys.stderr)
                    if not state.get_possible_moves(2):
                        print("Parcheesi: no legal P2 moves for this roll.", file=sys.stderr)
                        state.current_player = 1
                        state.clear_dice()
                        waiting_for_p1 = True
                    else:
                        waiting_for_p1 = False
                        ai_pending = ai_mode

            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                return

            if event.type == pygame.KEYDOWN and event.key == pygame.K_r and not getattr(
                event, "repeat", False
            ):
                if hasattr(transport, "drain_p1_moves"):
                    transport.drain_p1_moves()
                state.reset()
                capture_inventory.clear()
                pending_checkers_turn.clear()
                latest_manual_actions = []
                ui.clear_drag()
                waiting_for_p1 = True
                ai_pending = False
                last_move = None
                reset_requested = _send_runtime_control_message(
                    transport,
                    {"type": "reset_game", "reason": "board_reset", "game": kind},
                )
                status_notice = _make_status_notice(
                    level="info",
                    title="Board Reset",
                    message=(
                        "Returned to the opening position, cleared queued moves, and requested a Pi-side game reset."
                        if reset_requested
                        else "Returned to the opening position and cleared queued moves."
                    ),
                    duration_ms=1800,
                )
                print("Board reset to starting position.", file=sys.stderr)

            if not waiting_for_p1:
                drop = ui.handle_event(event, state, p2_turn=True)
                if drop:
                    start_id, end_id = drop
                    _apply_and_dispatch_p2_move(start_id, end_id)

        geometry_modal_active = geometry_preview is not None or geometry_calibration is not None
        manual_fix_active = manual_fix_session is not None

        if waiting_for_p1 and not geometry_modal_active and not manual_fix_active:
            msg = transport.poll_p1_move()
            if msg is not None:
                state_before = state.copy()
                state.apply_move_trusted(msg.frm, msg.to)
                tracked_manual, expected_manual = _record_manual_green_captures(
                    state_before,
                    state,
                    msg,
                )
                ui.clear_drag()
                
                # Capture inventory logic needs to be adapted for Parcheesi
                # if kind == "chess":
                #     new_captured = state.captured_by_p1[len(state_before.captured_by_p1):]
                #     for piece in new_captured:
                #         capture_inventory.add_captured_piece("p2", piece.name)
                # elif kind == "checkers":
                #     new_captured = state.captured_by_p1[len(state_before.captured_by_p1):]
                #     for piece in new_captured:
                #         capture_inventory.add_captured_piece("p2", piece.name)
                # elif kind == "parcheesi":
                #     new_captured = state.captured_by_p1[len(state_before.captured_by_p1):]
                #     for piece in new_captured:
                #         capture_inventory.add_captured_piece("p2", piece.name)
                latest_manual_actions = []
                last_move = (msg.frm, msg.to)
                if expected_manual > 0 and tracked_manual < expected_manual:
                    status_notice = _make_status_notice(
                        level="warning",
                        title="Player 1 Move Accepted",
                        message=(
                            f"Applied {msg.frm} -> {msg.to}, but only tracked "
                            f"{tracked_manual}/{expected_manual} manual captured piece(s) in the green area."
                        ),
                        duration_ms=2800,
                    )
                elif tracked_manual > 0:
                    status_notice = _make_status_notice(
                        level="success",
                        title="Player 1 Move Accepted",
                        message=(
                            f"Applied {msg.frm} -> {msg.to}. Tracked {tracked_manual} "
                            "manual captured piece(s) in the green area."
                        ),
                        duration_ms=2200,
                    )
                else:
                    status_notice = _make_status_notice(
                        level="success",
                        title="Player 1 Move Accepted",
                        message=f"Applied {msg.frm} -> {msg.to}.",
                        duration_ms=1800,
                    )
                print(f"P1 move: {msg.frm} -> {msg.to}", file=sys.stderr)
                if expected_manual > 0:
                    print(
                        f"[MANUAL_CAPTURE] tracked={tracked_manual}/{expected_manual}",
                        file=sys.stderr,
                    )
                print(
                    "[INVENTORY] " + ", ".join(capture_inventory.to_summary_strings()) if capture_inventory.to_summary_strings() else "[INVENTORY] <empty>",
                    file=sys.stderr,
                )
                if kind == "chess" and hasattr(state, "is_checkmate"):
                    if state.is_checkmate():
                        print(
                            f"Game over: checkmate. Winner: {'P2' if state.turn_side() == 'p1' else 'P1'}",
                            file=sys.stderr,
                        )
                        status_notice = _make_status_notice(
                            level="error",
                            title="Checkmate",
                            message=f"Game over. Winner: {'P2' if state.turn_side() == 'p1' else 'P1'}. Press R to play again.",
                            sticky=True,
                        )
                        waiting_for_p1 = True
                        ai_pending = False
                    elif state.is_stalemate():
                        print("Game over: stalemate.", file=sys.stderr)
                        status_notice = _make_status_notice(
                            level="warning",
                            title="Stalemate",
                            message="Game over. Stalemate. Press R to play again.",
                            sticky=True,
                        )
                        waiting_for_p1 = True
                        ai_pending = False
                    elif state.is_check():
                        print(f"Check on {state.turn_side().upper()}.", file=sys.stderr)
                        status_notice = _make_status_notice(
                            level="warning",
                            title="Check",
                            message=f"Check on {state.turn_side().upper()}.",
                            duration_ms=1800,
                        )
                if kind == "checkers" and hasattr(state, "p1_must_continue_jump") and state.p1_must_continue_jump():
                    forced = state.p1_forced_square_id() if hasattr(state, "p1_forced_square_id") else None
                    if forced:
                        print(f"Checkers: P1 must continue jump with {forced}", file=sys.stderr)
                    waiting_for_p1 = True
                elif kind == "parcheesi":
                    waiting_for_p1 = False
                    ai_pending = ai_mode
                elif kind == "chess" and hasattr(state, "is_checkmate") and (state.is_checkmate() or state.is_stalemate()):
                    waiting_for_p1 = True
                    ai_pending = False
                else:
                    waiting_for_p1 = False
                    ai_pending = ai_mode

        if not geometry_modal_active and not manual_fix_active and (not waiting_for_p1) and ai_mode and ai_pending:
            chosen = choose_p2_move(kind, state)
            ai_pending = False
            if chosen is None:
                print("AI: no legal P2 move available", file=sys.stderr)
                waiting_for_p1 = True
            else:
                _apply_and_dispatch_p2_move(chosen.start_id, chosen.end_id)

        ui.draw(state, p2_turn=not waiting_for_p1, last_move=last_move)
        _draw_back_button(ui)
        # _draw_capture_inventory_overlay(ui, capture_inventory.to_summary_strings(), latest_manual_actions)
        _draw_manual_fix_button(ui)
        _draw_mode_button(ui, ai_mode, kind)
        _draw_status_notice(
            ui.screen,
            ui.font,
            _current_status_notice(kind, state, waiting_for_p1, ai_mode, status_notice),
            ui,
        )

        # Draw "Roll Dice" button for Parcheesi
        if kind == "parcheesi":
            pygame.draw.rect(ui.screen, (50, 150, 50), roll_dice_button_rect, border_radius=10)
            roll_text = ui.font.render("Roll Dice", True, (255, 255, 255))
            ui.screen.blit(roll_text, roll_text.get_rect(center=roll_dice_button_rect.center))


        if geometry_preview is not None:
            _draw_geometry_preview(ui.screen, ui.font, geometry_preview)
        if geometry_calibration is not None:
            _draw_geometry_calibration(ui.screen, ui.font, geometry_calibration)
        if manual_fix_session is not None:
            _draw_manual_fix_modal(ui, state, manual_fix_session)
        pygame.display.flip()
        ui.tick(60)


def main() -> None:
    runtime_log_path = _install_runtime_log()
    print(f"GUI runtime log: {runtime_log_path}", file=sys.stderr)
    pygame.init()

    # Transport stays open while you switch between games.
    try:
        transport = _make_transport()
    except OSError as e:
        print("Transport connect failed:", e, file=sys.stderr)
        sys.exit(1)

    screen = pygame.display.set_mode((config.WINDOW_WIDTH, config.WINDOW_HEIGHT))
    pygame.display.set_caption("Player 2 — Game Select")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("sf pro display, helvetica, arial", 22)

    if config.START_GAME:
        if config.START_GAME not in {"checkers", "chess", "parcheesi"}:
            print(f"Unknown P2_START_GAME={config.START_GAME!r}; showing menu.", file=sys.stderr)
        else:
            _run_game_loop(config.START_GAME, transport)

    bw = 280
    bh = 70
    gap = 18
    cx = config.WINDOW_WIDTH // 2
    top = 220
    buttons = [
        _Button(pygame.Rect(cx - bw // 2, top, bw, bh), "Checkers"),
        _Button(pygame.Rect(cx - bw // 2, top + bh + gap, bw, bh), "Chess"),
        _Button(pygame.Rect(cx - bw // 2, top + (2 * (bh + gap)), bw, bh), "Parcheesi"),
    ]
    geometry_preview: _GeometryPreview | None = None
    geometry_calibration: _GeometryCalibration | None = None
    status_notice: _StatusNotice | None = None

    try:
        while True:
            geometry_preview, geometry_calibration, status_notice = _poll_geometry_controls(
                transport, geometry_preview, geometry_calibration, status_notice
            )

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return
                geometry_calibration, consumed_by_calibration = _handle_geometry_calibration_event(
                    geometry_calibration, event, transport
                )
                if consumed_by_calibration:
                    continue
                geometry_preview, consumed_by_preview = _handle_geometry_preview_event(
                    geometry_preview, event, transport
                )
                if consumed_by_preview:
                    continue
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    for b in buttons:
                        if b.rect.collidepoint(*event.pos):
                            label = b.label.lower()
                            if label.startswith("check"):
                                kind = "checkers"
                            elif label.startswith("chess"):
                                kind = "chess"
                            else:
                                kind = "parcheesi"
                            _run_game_loop(kind, transport)

            _draw_menu(screen, font, buttons)
            if geometry_preview is None and geometry_calibration is None and status_notice is not None:
                now_ms = pygame.time.get_ticks()
                if status_notice.sticky or status_notice.expires_at_ms <= 0 or now_ms < status_notice.expires_at_ms:
                    _draw_status_notice(screen, font, status_notice)
            if geometry_preview is not None:
                _draw_geometry_preview(screen, font, geometry_preview)
                pygame.display.flip()
            if geometry_calibration is not None:
                _draw_geometry_calibration(screen, font, geometry_calibration)
                pygame.display.flip()
            clock.tick(60)
    finally:
        close_engine()
        if hasattr(transport, "close"):
            transport.close()


if __name__ == "__main__":
    main()
