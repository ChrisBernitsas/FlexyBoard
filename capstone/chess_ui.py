"""Pygame chess board with labels and drag-and-drop for P2 pieces."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import pygame

from chess_sprites import ChessSpriteSet
from chess_state import ChessPiece, ChessState
from coords import FILES, RANKS, Square


@dataclass
class DragState:
    start_square: Square


class ChessUI:
    LIGHT = (240, 217, 181)
    DARK = (181, 136, 99)
    BG = (48, 48, 52)

    P1_COLOR = (235, 235, 240)  # white-ish
    P2_COLOR = (35, 35, 40)  # black-ish
    PANEL_BG = (40, 40, 46)
    PANEL_STROKE = (70, 70, 78)

    def __init__(self, width: int, height: int) -> None:
        pygame.init()
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Player 2 — Chess")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("sf pro display, helvetica, arial", 18)
        self.piece_font = pygame.font.SysFont("sf pro display, helvetica, arial", 26, bold=True)

        self.margin = 36
        self.sidebar_gap = 16
        content_w = max(1, width - 2 * self.margin)
        self.sidebar_w = min(240, max(140, width // 4))
        board_zone_w = max(1, content_w - self.sidebar_w - self.sidebar_gap)
        inner_h = height - 2 * self.margin
        self.cell = max(1, min(board_zone_w, inner_h) // 8)
        self.board_px = self.cell * 8
        self.origin_y = self.margin + (inner_h - self.board_px) // 2
        self.origin_x = self.margin + max(0, (board_zone_w - self.board_px) // 2)
        self.sidebar_x = self.margin + board_zone_w + self.sidebar_gap

        self.drag: Optional[DragState] = None
        self.drag_mouse: Tuple[int, int] = (0, 0)

        self._sprites = ChessSpriteSet.load(self.cell, tray_px=22)

    def clear_drag(self) -> None:
        self.drag = None

    def close(self) -> None:
        pygame.quit()

    def tick(self, fps: int = 60) -> None:
        self.clock.tick(fps)

    def square_at_pixel(self, x: int, y: int) -> Optional[Square]:
        if (
            x < self.origin_x
            or y < self.origin_y
            or x >= self.origin_x + self.board_px
            or y >= self.origin_y + self.board_px
        ):
            return None
        col = (x - self.origin_x) // self.cell
        row_screen = (y - self.origin_y) // self.cell
        if not 0 <= col <= 7 or not 0 <= row_screen <= 7:
            return None
        rank_index = 7 - row_screen
        return Square(col, rank_index)

    def cell_rect(self, sq: Square) -> pygame.Rect:
        row_screen = 7 - sq.rank_index
        x = self.origin_x + sq.file_index * self.cell
        y = self.origin_y + row_screen * self.cell
        return pygame.Rect(x, y, self.cell, self.cell)

    def _draw_labels(self) -> None:
        for i in range(8):
            ch = FILES[i]
            x = self.origin_x + i * self.cell + self.cell // 2
            y_bottom = self.origin_y + self.board_px + 8
            surf = self.font.render(ch, True, (220, 220, 220))
            self.screen.blit(surf, surf.get_rect(center=(x, y_bottom)))

        for i in range(8):
            rank_char = RANKS[7 - i]
            x_left = self.origin_x - 8
            y = self.origin_y + i * self.cell + self.cell // 2
            surf = self.font.render(rank_char, True, (220, 220, 220))
            self.screen.blit(surf, surf.get_rect(midright=(x_left, y)))

    def _piece_label(self, p: ChessPiece) -> str:
        return {
            ChessPiece.P1_PAWN: "P",
            ChessPiece.P1_KNIGHT: "N",
            ChessPiece.P1_BISHOP: "B",
            ChessPiece.P1_ROOK: "R",
            ChessPiece.P1_QUEEN: "Q",
            ChessPiece.P1_KING: "K",
            ChessPiece.P2_PAWN: "P",
            ChessPiece.P2_KNIGHT: "N",
            ChessPiece.P2_BISHOP: "B",
            ChessPiece.P2_ROOK: "R",
            ChessPiece.P2_QUEEN: "Q",
            ChessPiece.P2_KING: "K",
        }.get(p, "")

    def _piece_color(self, p: ChessPiece) -> Tuple[int, int, int]:
        if p == ChessPiece.EMPTY:
            return (0, 0, 0)
        return self.P1_COLOR if p <= ChessPiece.P1_KING else self.P2_COLOR

    def _draw_piece_in_cell(self, sq: Square, p: ChessPiece) -> None:
        rect = self.cell_rect(sq)
        if self._sprites.has(p):
            img = self._sprites.board[p]
            self.screen.blit(img, img.get_rect(center=rect.center))
            return
        label = self._piece_label(p)
        if not label:
            return
        surf = self.piece_font.render(label, True, self._piece_color(p))
        self.screen.blit(surf, surf.get_rect(center=rect.center))

    def _draw_capture_sidebar(self, state: ChessState) -> None:
        top = self.margin
        h = self.screen.get_height() - 2 * self.margin
        panel = pygame.Rect(self.sidebar_x - 6, top - 6, self.sidebar_w + 12, h + 12)
        pygame.draw.rect(self.screen, self.PANEL_BG, panel)
        pygame.draw.rect(self.screen, self.PANEL_STROKE, panel, width=1)

        x = self.sidebar_x
        y = top + 4
        title = self.font.render("Captured", True, (235, 235, 240))
        self.screen.blit(title, (x, y))
        y += title.get_height() + 14

        y = self._draw_capture_section("P1 took (P2)", state.captured_by_p1, x, y, self.sidebar_w)
        self._draw_capture_section("P2 took (P1)", state.captured_by_p2, x, y, self.sidebar_w)

    def _draw_capture_section(
        self, label: str, pieces: List[ChessPiece], x: int, y: int, panel_w: int
    ) -> int:
        sub = self.font.render(label, True, (180, 180, 188))
        self.screen.blit(sub, (x, y))
        y += sub.get_height() + 8
        step = 22
        cols = max(1, (panel_w - 4) // step)
        for i, pe in enumerate(pieces):
            row, col = divmod(i, cols)
            px = x + col * step
            py = y + row * step
            if self._sprites.has(pe):
                img = self._sprites.small[pe]
                cx = px + step // 2
                cy = py + step // 2
                self.screen.blit(img, img.get_rect(center=(cx, cy)))
            else:
                surf = self.font.render(self._piece_label(pe), True, self._piece_color(pe))
                self.screen.blit(surf, (px, py))
        rows = (len(pieces) + cols - 1) // cols if pieces else 0
        y += rows * step + (4 if rows else 0) + 18
        return y

    def draw(self, state: ChessState, p2_turn: bool) -> None:
        self.screen.fill(self.BG)
        self._draw_capture_sidebar(state)
        self._draw_labels()

        for r in range(8):
            for f in range(8):
                sq = Square(f, r)
                rect = self.cell_rect(sq)
                color = self.DARK if (f + r) % 2 == 0 else self.LIGHT
                pygame.draw.rect(self.screen, color, rect)

        for r in range(8):
            for f in range(8):
                sq = Square(f, r)
                p = state.get(sq)
                if p == ChessPiece.EMPTY:
                    continue
                if self.drag and p2_turn and self.drag.start_square == sq:
                    continue
                self._draw_piece_in_cell(sq, p)

        if self.drag and p2_turn:
            moving = state.get(self.drag.start_square)
            if self._sprites.has(moving):
                img = self._sprites.board[moving]
                self.screen.blit(img, img.get_rect(center=self.drag_mouse))
            else:
                label = self._piece_label(moving)
                if label:
                    surf = self.piece_font.render(label, True, self._piece_color(moving))
                    self.screen.blit(surf, surf.get_rect(center=self.drag_mouse))

        pygame.display.flip()

    def handle_event(
        self, event: pygame.event.Event, state: ChessState, p2_turn: bool
    ) -> Optional[Tuple[str, str]]:
        if not p2_turn:
            if event.type == pygame.MOUSEBUTTONUP:
                self.drag = None
            return None

        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            sq = self.square_at_pixel(*event.pos)
            if sq is None:
                return None
            p = state.get(sq)
            if p < ChessPiece.P2_PAWN:
                return None
            self.drag = DragState(start_square=sq)
            self.drag_mouse = event.pos
            return None

        if event.type == pygame.MOUSEMOTION and self.drag:
            self.drag_mouse = event.pos
            return None

        if event.type == pygame.MOUSEBUTTONUP and event.button == 1 and self.drag:
            end_sq = self.square_at_pixel(*event.pos)
            start_sq = self.drag.start_square
            self.drag = None
            if end_sq is None:
                return None
            start_id = start_sq.to_id()
            end_id = end_sq.to_id()
            if start_id == end_id:
                return None
            return start_id, end_id

        return None

