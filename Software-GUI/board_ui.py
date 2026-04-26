"""Pygame board: labels a–h / 1–8, rank 1 at bottom, drag-and-drop for P2 pieces."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import pygame

from checkers_state import CheckersState, Piece
from coords import FILES, RANKS, Square, is_dark_square, parse_square


@dataclass
class DragState:
    start_square: Square
    grab_offset: Tuple[int, int]


class BoardUI:
    LIGHT = (240, 217, 181)
    DARK = (181, 136, 99)
    LABEL_BG = (48, 48, 52)
    P1_COLOR = (220, 60, 60)
    P1_KING = (255, 120, 120)
    P2_COLOR = (40, 40, 40)
    P2_KING = (100, 100, 110)
    HIGHLIGHT = (255, 255, 100, 80)
    SOURCE_HL = (70, 150, 255, 110)
    DEST_HL = (255, 215, 60, 110)

    def __init__(self, width: int, height: int) -> None:
        pygame.init()
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Player 2 — Checkers")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("sf pro display, helvetica, arial", 18)
        self._small_label_font = pygame.font.Font(None, 13)

        self.margin = 40
        self.header_height = 128
        self.footer_height = 44
        self.sidebar_gap = 16
        content_w = max(1, width - 2 * self.margin)
        self.sidebar_w = min(220, max(130, width // 4))
        board_zone_w = max(1, content_w - self.sidebar_w - self.sidebar_gap)
        inner_h = max(1, height - self.header_height - self.footer_height - self.margin)
        self.cell = max(1, min(board_zone_w, inner_h) // 8)
        self.board_px = self.cell * 8
        self.origin_y = self.header_height + max(0, (inner_h - self.board_px) // 2)
        self.origin_x = self.margin + max(0, (board_zone_w - self.board_px) // 2)
        self.sidebar_x = self.margin + board_zone_w + self.sidebar_gap

        self.drag: Optional[DragState] = None
        self.drag_mouse: Tuple[int, int] = (0, 0)

    def clear_drag(self) -> None:
        self.drag = None

    def close(self) -> None:
        pygame.quit()

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

    def center_of_cell(self, sq: Square) -> Tuple[int, int]:
        r = self.cell_rect(sq)
        return r.centerx, r.centery

    def _draw_highlight_square(self, sq: Square, rgba: tuple[int, int, int, int]) -> None:
        rect = self.cell_rect(sq)
        overlay = pygame.Surface((rect.width, rect.height), pygame.SRCALPHA)
        overlay.fill(rgba)
        self.screen.blit(overlay, rect.topleft)

    def draw(self, state: CheckersState, p2_turn: bool, last_move: Optional[Tuple[str, str]] = None) -> None:
        self.screen.fill(self.LABEL_BG)
        self._draw_capture_sidebar(state)
        self._draw_labels()
        for r in range(8):
            for f in range(8):
                sq = Square(f, r)
                rect = self.cell_rect(sq)
                color = self.DARK if is_dark_square(f, r) else self.LIGHT
                pygame.draw.rect(self.screen, color, rect)

        if last_move is not None:
            src_id, dst_id = last_move
            try:
                self._draw_highlight_square(parse_square(src_id), self.SOURCE_HL)
                self._draw_highlight_square(parse_square(dst_id), self.DEST_HL)
            except ValueError:
                pass

        for r in range(8):
            for f in range(8):
                sq = Square(f, r)
                piece = state.get(sq)
                if piece == Piece.EMPTY:
                    continue
                if self.drag and self.drag.start_square == sq:
                    continue
                self._draw_piece(sq, piece)

        if self.drag and p2_turn:
            piece = state.get(self.drag.start_square)
            cx, cy = self.drag_mouse
            self._draw_piece_at(piece, cx, cy)

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

    def _draw_piece(self, sq: Square, piece: Piece) -> None:
        cx, cy = self.center_of_cell(sq)
        self._draw_piece_at(piece, cx, cy)

    def _capture_piece_radius(self) -> int:
        return max(12, int(self.cell * 0.22))

    def _draw_capture_sidebar(self, state: CheckersState) -> None:
        top = self.header_height
        h = self.screen.get_height() - self.header_height - self.margin
        panel = pygame.Rect(self.sidebar_x - 6, top - 6, self.sidebar_w + 12, h + 12)
        pygame.draw.rect(self.screen, (40, 40, 46), panel)
        pygame.draw.rect(self.screen, (70, 70, 78), panel, width=1)

        x = self.sidebar_x
        y = top + 4
        title = self.font.render("Captured", True, (235, 235, 240))
        self.screen.blit(title, (x, y))
        y += title.get_height() + 14

        y = self._draw_capture_section(
            "P1 took (P2 pieces)", state.captured_by_p1, x, y, self.sidebar_w
        )
        self._draw_capture_section(
            "P2 took (P1 pieces)", state.captured_by_p2, x, y, self.sidebar_w
        )

    def _draw_capture_section(
        self, label: str, pieces: List[Piece], x: int, y: int, panel_w: int
    ) -> int:
        sub = self.font.render(label, True, (180, 180, 188))
        self.screen.blit(sub, (x, y))
        y += sub.get_height() + 8
        r = self._capture_piece_radius()
        step = 2 * r + 6
        cols = max(1, (panel_w - 4) // step)
        for i, pe in enumerate(pieces):
            row, col = divmod(i, cols)
            cx = x + r + col * step
            cy = y + r + row * step
            self._draw_piece_at(pe, cx, cy, radius=r)
        rows = (len(pieces) + cols - 1) // cols if pieces else 0
        y += rows * step + (4 if rows else 0) + 18
        return y

    def _draw_piece_at(self, piece: Piece, cx: int, cy: int, radius: Optional[int] = None) -> None:
        r = int(self.cell * 0.38) if radius is None else radius
        if piece in (Piece.P1_MAN, Piece.P1_KING):
            fill = self.P1_KING if piece == Piece.P1_KING else self.P1_COLOR
        else:
            fill = self.P2_KING if piece == Piece.P2_KING else self.P2_COLOR
        pygame.draw.circle(self.screen, fill, (cx, cy), r)
        if piece in (Piece.P1_KING, Piece.P2_KING):
            label_font = self._small_label_font if r < self.cell * 0.34 else self.font
            k = label_font.render("K", True, (255, 255, 255))
            self.screen.blit(k, k.get_rect(center=(cx, cy)))

    def handle_event(
        self,
        event: pygame.event.Event,
        state: CheckersState,
        p2_turn: bool,
    ) -> Optional[Tuple[str, str]]:
        """
        If P2 completes a legal drop, return (start_id, end_id); else None.
        """
        if not p2_turn:
            if event.type == pygame.MOUSEBUTTONUP:
                self.drag = None
            return None

        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            sq = self.square_at_pixel(*event.pos)
            if sq is None:
                return None
            p = state.get(sq)
            if p not in (Piece.P2_MAN, Piece.P2_KING):
                return None
            cx, cy = self.center_of_cell(sq)
            self.drag = DragState(start_square=sq, grab_offset=(event.pos[0] - cx, event.pos[1] - cy))
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

    def tick(self, fps: int = 60) -> None:
        self.clock.tick(fps)
