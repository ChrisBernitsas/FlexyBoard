"""Pygame Parcheesi board UI."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import pygame

from parcheesi_state import ParcheesiState, Piece


class ParcheesiUI:
    LINE_ONLY_BOARD = False
    REFERENCE_LINE_ART = "parcheesi_reference_black_parts_only.png"

    BG = (36, 55, 83)
    PANEL = (34, 37, 46)
    BOARD = (239, 229, 188)
    BOARD_EDGE = (80, 72, 50)
    LINE = (28, 48, 41)
    TRACK = (245, 236, 199)
    SAFE = (196, 222, 213)
    CENTER = (238, 218, 171)
    CENTER_ORANGE = (226, 103, 51)
    HOME_BLUE = (196, 226, 221)
    HOME_GREEN = (145, 186, 48)
    HOME_YELLOW = (224, 218, 27)
    HOME_ORANGE = (230, 180, 178)
    HOME_GRAY = (202, 214, 218)
    HOME_LANE = (210, 80, 88)
    TEXT = (242, 241, 236)
    MUTED = (170, 174, 184)
    HIGHLIGHT = (255, 255, 255, 105)
    DEST_HL = (90, 210, 120, 115)
    LAST_SRC = (70, 150, 255, 120)
    LAST_DST = (255, 215, 60, 120)

    PLAYER_COLORS = {
        1: (218, 60, 62),
        2: (55, 105, 224),
        3: (236, 190, 55),
        4: (54, 174, 92),
    }
    PLAYER_DARK = {
        1: (126, 35, 38),
        2: (32, 62, 135),
        3: (134, 102, 29),
        4: (31, 106, 58),
    }

    def __init__(self, width: int, height: int) -> None:
        pygame.init()
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Player 2 - Parcheesi")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("sf pro display, helvetica, arial", 18)
        self.big_font = pygame.font.SysFont("sf pro display, helvetica, arial", 26, bold=True)
        self.small_font = pygame.font.Font(None, 15)

        self.margin = 42
        self.header_height = 128
        self.footer_height = 44
        self.sidebar_w = min(250, max(190, width // 4))
        self.sidebar_gap = 20
        board_zone_w = width - self.sidebar_w - self.sidebar_gap - 2 * self.margin
        board_zone_h = max(360, height - self.header_height - self.footer_height - self.margin)
        self.board_px = min(board_zone_w, board_zone_h)
        self.board_px = max(360, self.board_px)
        self.origin_x = self.margin
        self.origin_y = self.header_height + max(0, (board_zone_h - self.board_px) // 2)
        self.sidebar_x = self.origin_x + self.board_px + self.sidebar_gap
        self.cell = self.board_px / 19.0
        self.reference_line_art = self._load_reference_line_art()

        self.hit_rects: Dict[str, pygame.Rect] = {}
        self.dragging_piece: Piece | None = None
        self.drag_start_id: str | None = None
        self.drag_current: tuple[int, int] | None = None
        self.drag_offset: tuple[int, int] = (0, 0)

    def tick(self, fps: int = 60) -> None:
        self.clock.tick(fps)

    def clear_drag(self) -> None:
        self.dragging_piece = None
        self.drag_start_id = None
        self.drag_current = None
        self.drag_offset = (0, 0)

    def _grid_to_px(self, gx: float, gy: float) -> tuple[int, int]:
        return (
            int(self.origin_x + (gx + 0.5) * self.cell),
            int(self.origin_y + (gy + 0.5) * self.cell),
        )

    def _grid_rect(self, gx: float, gy: float, gw: float, gh: float) -> pygame.Rect:
        return pygame.Rect(
            int(self.origin_x + gx * self.cell),
            int(self.origin_y + gy * self.cell),
            int(gw * self.cell),
            int(gh * self.cell),
        )

    def _grid_point(self, gx: float, gy: float) -> tuple[int, int]:
        return (
            int(self.origin_x + gx * self.cell),
            int(self.origin_y + gy * self.cell),
        )

    def _load_reference_line_art(self) -> pygame.Surface | None:
        if not self.LINE_ONLY_BOARD:
            return None
        repo_root = Path(__file__).resolve().parents[1]
        candidates = [
            Path(__file__).resolve().parent / "assets" / self.REFERENCE_LINE_ART,
            repo_root / "FlexyBoard-Camera" / "debug_output" / self.REFERENCE_LINE_ART,
            Path.cwd() / "FlexyBoard-Camera" / "debug_output" / self.REFERENCE_LINE_ART,
            Path.cwd().parent / "FlexyBoard-Camera" / "debug_output" / self.REFERENCE_LINE_ART,
        ]
        for path in candidates:
            if path.exists():
                return pygame.image.load(str(path)).convert()
        return None

    def _draw_reference_line_art(self) -> None:
        if self.reference_line_art is None:
            return
        board_rect = pygame.Rect(self.origin_x, self.origin_y, self.board_px, self.board_px)
        scaled = pygame.transform.smoothscale(self.reference_line_art, (self.board_px, self.board_px))
        self.screen.blit(scaled, board_rect.topleft)

    def _grid_cell_rect(self, gx: float, gy: float) -> pygame.Rect:
        """Return the printed-path rectangle for one logical board cell."""
        # In the reference board, the three path lanes fill the complete band
        # between adjacent circular home bases. These constants define that band
        # in the 19x19 virtual board space.
        outer_min = 0.45
        home_min = 7.45
        home_max = 11.55
        outer_max = 18.55
        band_min = 6.45
        band_max = 12.55
        band = band_max - band_min
        lane = band / 3.0
        step = (home_min - outer_min) / 8.0

        ix = int(round(gx))
        iy = int(round(gy))

        # Top and bottom vertical arms: 3 columns x 8 rows.
        lane_for_x = {8: 0, 9: 1, 10: 2}.get(ix)
        if lane_for_x is not None and 0 <= iy <= 7:
            x0 = self.origin_x + (band_min + lane_for_x * lane) * self.cell
            y0 = self.origin_y + (outer_min + iy * step) * self.cell
            return pygame.Rect(round(x0), round(y0), round(lane * self.cell), round(step * self.cell))
        if lane_for_x is not None and 11 <= iy <= 18:
            x0 = self.origin_x + (band_min + lane_for_x * lane) * self.cell
            y0 = self.origin_y + (home_max + (iy - 11) * step) * self.cell
            return pygame.Rect(round(x0), round(y0), round(lane * self.cell), round(step * self.cell))

        # Left and right horizontal arms: 8 columns x 3 rows.
        lane_for_y = {8: 0, 9: 1, 10: 2}.get(iy)
        if lane_for_y is not None and 0 <= ix <= 7:
            x0 = self.origin_x + (outer_min + ix * step) * self.cell
            y0 = self.origin_y + (band_min + lane_for_y * lane) * self.cell
            return pygame.Rect(round(x0), round(y0), round(step * self.cell), round(lane * self.cell))
        if lane_for_y is not None and 11 <= ix <= 18:
            x0 = self.origin_x + (home_max + (ix - 11) * step) * self.cell
            y0 = self.origin_y + (band_min + lane_for_y * lane) * self.cell
            return pygame.Rect(round(x0), round(y0), round(step * self.cell), round(lane * self.cell))

        x0 = self.origin_x + gx * self.cell
        x1 = self.origin_x + (gx + 1.0) * self.cell
        y0 = self.origin_y + gy * self.cell
        y1 = self.origin_y + (gy + 1.0) * self.cell
        return pygame.Rect(round(x0), round(y0), round(x1 - x0), round(y1 - y0))

    def _location_rect(self, location_id: str) -> pygame.Rect:
        gx, gy = ParcheesiState.location_id_to_grid(location_id)
        loc = ParcheesiState.parse_location_id(location_id)
        if loc.kind in {"main", "home"}:
            return self._grid_cell_rect(gx, gy)
        cx, cy = self._grid_to_px(gx, gy)
        size = max(18, int(self.cell * 0.76))
        return pygame.Rect(cx - size // 2, cy - size // 2, size, size)

    def _is_inside_center_home(self, location_id: str) -> bool:
        gx, gy = ParcheesiState.location_id_to_grid(location_id)
        return 8.0 <= gx <= 10.0 and 8.0 <= gy <= 10.0

    def _piece_radius(self) -> int:
        return max(8, int(self.cell * 0.28))

    def _draw_text_center(self, text: str, center: tuple[int, int], color: tuple[int, int, int], font: pygame.font.Font | None = None) -> None:
        f = font or self.font
        surf = f.render(text, True, color)
        self.screen.blit(surf, surf.get_rect(center=center))

    def _draw_curve(
        self,
        p0: tuple[int, int],
        control: tuple[int, int],
        p1: tuple[int, int],
        *,
        width: int = 2,
    ) -> None:
        points: list[tuple[int, int]] = []
        for idx in range(25):
            t = idx / 24.0
            inv = 1.0 - t
            x = inv * inv * p0[0] + 2 * inv * t * control[0] + t * t * p1[0]
            y = inv * inv * p0[1] + 2 * inv * t * control[1] + t * t * p1[1]
            points.append((round(x), round(y)))
        pygame.draw.lines(self.screen, self.LINE, False, points, width=width)

    def _draw_location(self, location_id: str, state: ParcheesiState) -> None:
        rect = self._location_rect(location_id)
        self.hit_rects[location_id] = rect
        loc = ParcheesiState.parse_location_id(location_id)

        if self.LINE_ONLY_BOARD:
            return

        if loc.kind in {"main", "home"} and self._is_inside_center_home(location_id):
            return

        if loc.kind == "main":
            gx, gy = ParcheesiState.location_id_to_grid(location_id)
            if (round(gx), round(gy)) in {(7, 7), (11, 7), (11, 11), (7, 11)}:
                return
            if loc.pos in ParcheesiState.SAFE_SQUARES:
                color = self.SAFE
            else:
                color = self.TRACK
            pygame.draw.rect(self.screen, color, rect)
            pygame.draw.rect(self.screen, self.LINE, rect, width=2)
            if loc.pos in ParcheesiState.SAFE_SQUARES:
                pygame.draw.circle(self.screen, self.LINE, rect.center, max(4, min(rect.width, rect.height) // 3), width=2)
            return

        if loc.kind == "home":
            color = self.HOME_LANE
            pygame.draw.rect(self.screen, color, rect)
            pygame.draw.rect(self.screen, self.LINE, rect, width=2)
            return

        if loc.kind == "nest":
            color = self.HOME_BLUE
            pygame.draw.circle(self.screen, color, rect.center, rect.width // 2)
            pygame.draw.circle(self.screen, self.LINE, rect.center, rect.width // 2, width=2)
            return

        if loc.kind == "homearea":
            # Home-region target is intentionally invisible; finished pieces
            # still draw here, but the printed center region has no token squares.
            return

    def _draw_corner_home(self, player: int, rect: pygame.Rect) -> None:
        """Draw the large color-petal home area used on classic Parcheesi boards."""
        pygame.draw.rect(self.screen, self.BOARD, rect)

        ellipse = rect.inflate(-int(self.cell * 0.45), -int(self.cell * 0.45))
        if not self.LINE_ONLY_BOARD:
            pygame.draw.ellipse(self.screen, self.HOME_BLUE, ellipse)

            # Petals intentionally overlap; the center oval is redrawn afterward.
            pygame.draw.ellipse(self.screen, self.HOME_GREEN, pygame.Rect(ellipse.left, ellipse.centery - ellipse.height // 4, ellipse.width // 3, ellipse.height // 2))
            pygame.draw.ellipse(self.screen, self.HOME_GRAY, pygame.Rect(ellipse.right - ellipse.width // 3, ellipse.centery - ellipse.height // 4, ellipse.width // 3, ellipse.height // 2))
            pygame.draw.ellipse(self.screen, self.HOME_ORANGE, pygame.Rect(ellipse.centerx - ellipse.width // 4, ellipse.top, ellipse.width // 2, ellipse.height // 3))
            pygame.draw.ellipse(self.screen, self.HOME_YELLOW, pygame.Rect(ellipse.centerx - ellipse.width // 4, ellipse.bottom - ellipse.height // 3, ellipse.width // 2, ellipse.height // 3))
            pygame.draw.ellipse(self.screen, self.HOME_BLUE, ellipse.inflate(-int(self.cell * 1.7), -int(self.cell * 1.25)))

            pygame.draw.ellipse(self.screen, self.LINE, ellipse, width=3)
            pygame.draw.line(self.screen, self.LINE, ellipse.midleft, (ellipse.centerx - int(self.cell * 1.8), ellipse.top + int(self.cell * 0.8)), width=2)
            pygame.draw.line(self.screen, self.LINE, ellipse.midleft, (ellipse.centerx - int(self.cell * 1.8), ellipse.bottom - int(self.cell * 0.8)), width=2)
            pygame.draw.line(self.screen, self.LINE, ellipse.midright, (ellipse.centerx + int(self.cell * 1.8), ellipse.top + int(self.cell * 0.8)), width=2)
            pygame.draw.line(self.screen, self.LINE, ellipse.midright, (ellipse.centerx + int(self.cell * 1.8), ellipse.bottom - int(self.cell * 0.8)), width=2)
            return

        pygame.draw.ellipse(self.screen, self.LINE, ellipse, width=3)

        w = ellipse.width
        h = ellipse.height
        top_left = (ellipse.left + round(0.19 * w), ellipse.top + round(0.21 * h))
        bottom_left = (ellipse.left + round(0.19 * w), ellipse.bottom - round(0.21 * h))
        top_right = (ellipse.right - round(0.19 * w), ellipse.top + round(0.21 * h))
        bottom_right = (ellipse.right - round(0.19 * w), ellipse.bottom - round(0.21 * h))

        self._draw_curve(top_left, (ellipse.left + round(0.02 * w), ellipse.centery), bottom_left, width=3)
        self._draw_curve(top_right, (ellipse.right - round(0.02 * w), ellipse.centery), bottom_right, width=3)
        self._draw_curve(top_left, (ellipse.centerx, ellipse.top + round(0.02 * h)), top_right, width=3)
        self._draw_curve(bottom_left, (ellipse.centerx, ellipse.bottom - round(0.02 * h)), bottom_right, width=3)

    def _draw_center_home(self) -> None:
        if self.LINE_ONLY_BOARD and self.reference_line_art is not None:
            return
        center = self._grid_rect(7.7, 7.7, 3.6, 3.6)
        pygame.draw.rect(self.screen, self.BOARD, center, border_radius=3)
        pygame.draw.rect(self.screen, self.LINE, center, width=3)

        # Lightweight decoration so the center reads like the printed home panel in the reference.
        for inset in (0.25, 0.55):
            deco = self._grid_rect(7.7 + inset, 7.7 + inset, 3.6 - inset * 2, 3.6 - inset * 2)
            pygame.draw.rect(self.screen, self.LINE, deco, width=1, border_radius=5)
        self._draw_text_center("HOME", center.center, self.LINE, self.big_font)

    def _draw_center_apron(self) -> None:
        """Draw the angled connector region around HOME from the reference board."""
        if self.LINE_ONLY_BOARD and self.reference_line_art is not None:
            return
        apron = [
            self._grid_point(7.45, 6.45),
            self._grid_point(11.55, 6.45),
            self._grid_point(12.55, 7.45),
            self._grid_point(12.55, 11.55),
            self._grid_point(11.55, 12.55),
            self._grid_point(7.45, 12.55),
            self._grid_point(6.45, 11.55),
            self._grid_point(6.45, 7.45),
        ]
        pygame.draw.polygon(self.screen, self.BOARD, apron)
        pygame.draw.lines(self.screen, self.LINE, True, apron, width=3)

        # Small lane arrows at the HOME edge, matching the printed board's
        # transition from rectangular lanes into the angled center region.
        arrows = [
            (self.HOME_LANE, [self._grid_point(9.0, 7.25), self._grid_point(8.7, 6.45), self._grid_point(9.3, 6.45)]),
            (self.HOME_LANE, [self._grid_point(11.75, 9.0), self._grid_point(12.55, 8.7), self._grid_point(12.55, 9.3)]),
            (self.HOME_LANE, [self._grid_point(9.0, 11.75), self._grid_point(8.7, 12.55), self._grid_point(9.3, 12.55)]),
            (self.HOME_LANE, [self._grid_point(7.25, 9.0), self._grid_point(6.45, 8.7), self._grid_point(6.45, 9.3)]),
        ]
        for color, points in arrows:
            if not self.LINE_ONLY_BOARD:
                pygame.draw.polygon(self.screen, color, points)
            pygame.draw.lines(self.screen, self.LINE, True, points, width=2)

    def _draw_loop_guides(self) -> None:
        if self.LINE_ONLY_BOARD and self.reference_line_art is not None:
            return
        # Cream cross roads and colored home lanes behind the actual hit-test squares.
        road_cells: set[tuple[int, int]] = set()
        # Each arm is three lanes wide and seven squares long, matching the
        # physical board: two outer travel lanes plus one colored home lane.
        for x in range(8, 11):
            for y in list(range(0, 8)) + list(range(11, 19)):
                road_cells.add((x, y))
        for y in range(8, 11):
            for x in list(range(0, 8)) + list(range(11, 19)):
                road_cells.add((x, y))

        center_cells = {(x, y) for x in range(8, 11) for y in range(8, 11)}
        colored_cells: dict[tuple[int, int], tuple[int, int, int]] = {}
        for y in range(0, 8):
            colored_cells[(9, y)] = self.HOME_LANE
        for x in range(11, 19):
            colored_cells[(x, 9)] = self.HOME_LANE
        for y in range(11, 19):
            colored_cells[(9, y)] = self.HOME_LANE
        for x in range(0, 8):
            colored_cells[(x, 9)] = self.HOME_LANE

        for gx, gy in sorted(road_cells):
            if (gx, gy) in center_cells:
                continue
            color = self.BOARD if self.LINE_ONLY_BOARD else colored_cells.get((gx, gy), self.TRACK)
            rect = self._grid_cell_rect(gx, gy)
            pygame.draw.rect(self.screen, color, rect)
            pygame.draw.rect(self.screen, self.LINE, rect, width=2)

        # Light-blue safety markers: third from the center HOME side on both
        # outer travel lanes. The red home lane stays red.
        for gx, gy in (
            (8, 4), (10, 4),
            (14, 8), (14, 10),
            (8, 14), (10, 14),
            (4, 8), (4, 10),
        ):
            rect = self._grid_cell_rect(gx, gy)
            if not self.LINE_ONLY_BOARD:
                pygame.draw.rect(self.screen, self.SAFE, rect)
            pygame.draw.rect(self.screen, self.LINE, rect, width=2)
            pygame.draw.circle(self.screen, self.LINE, rect.center, max(4, min(rect.width, rect.height) // 3), width=2)

    def _draw_board_base(self, state: ParcheesiState) -> None:
        self.hit_rects.clear()
        board_rect = pygame.Rect(self.origin_x - 14, self.origin_y - 14, self.board_px + 28, self.board_px + 28)
        pygame.draw.rect(self.screen, self.BOARD_EDGE, board_rect, border_radius=10)
        pygame.draw.rect(self.screen, self.BOARD, pygame.Rect(self.origin_x, self.origin_y, self.board_px, self.board_px))
        self._draw_reference_line_art()

        self._draw_loop_guides()

        corner_homes = {
            1: self._grid_rect(0.4, 0.4, 6.1, 6.1),
            2: self._grid_rect(12.5, 0.4, 6.1, 6.1),
            3: self._grid_rect(12.5, 12.5, 6.1, 6.1),
            4: self._grid_rect(0.4, 12.5, 6.1, 6.1),
        }
        if not (self.LINE_ONLY_BOARD and self.reference_line_art is not None):
            for player, rect in corner_homes.items():
                self._draw_corner_home(player, rect)

        for location_id in ParcheesiState.iter_draw_locations():
            self._draw_location(location_id, state)

        self._draw_center_apron()
        self._draw_center_home()

        pygame.draw.rect(self.screen, self.LINE, pygame.Rect(self.origin_x, self.origin_y, self.board_px, self.board_px), width=4)

    def _draw_highlight(self, location_id: str, rgba: tuple[int, int, int, int]) -> None:
        rect = self._location_rect(location_id)
        overlay = pygame.Surface((rect.width, rect.height), pygame.SRCALPHA)
        overlay.fill(rgba)
        self.screen.blit(overlay, rect.topleft)

    def _draw_piece_at(self, piece: Piece, center: tuple[int, int], radius: int | None = None) -> None:
        player = piece.to_player_num()
        if player <= 0:
            return
        r = radius if radius is not None else self._piece_radius()
        pygame.draw.circle(self.screen, (28, 28, 32), (center[0] + 2, center[1] + 2), r)
        pygame.draw.circle(self.screen, self.PLAYER_COLORS[player], center, r)
        pygame.draw.circle(self.screen, (255, 255, 255), center, r, width=2)
        self._draw_text_center(str(piece.to_token_num()), center, (255, 255, 255), self.small_font)

    def _piece_center(self, state: ParcheesiState, piece: Piece) -> tuple[int, int]:
        loc_id = state.piece_location_id(piece)
        base = self._location_rect(loc_id).center
        pieces_here = state.pieces_at_id(loc_id)
        if len(pieces_here) <= 1:
            return base
        try:
            idx = pieces_here.index(piece)
        except ValueError:
            idx = 0
        offsets = [(-5, -5), (5, -5), (-5, 5), (5, 5)]
        ox, oy = offsets[idx % len(offsets)]
        return base[0] + ox, base[1] + oy

    def _draw_pieces(self, state: ParcheesiState) -> None:
        for player in range(1, ParcheesiState.NUM_PLAYERS + 1):
            for piece in ParcheesiState.all_pieces_for_player(player):
                if self.dragging_piece == piece:
                    continue
                self._draw_piece_at(piece, self._piece_center(state, piece))
        if self.dragging_piece is not None and self.drag_current is not None:
            self._draw_piece_at(self.dragging_piece, self.drag_current)

    def _draw_sidebar(self, state: ParcheesiState, p2_turn: bool) -> None:
        panel = pygame.Rect(self.sidebar_x - 8, self.margin, self.sidebar_w + 16, self.height - 2 * self.margin)
        pygame.draw.rect(self.screen, self.PANEL, panel, border_radius=16)
        pygame.draw.rect(self.screen, (83, 88, 102), panel, width=1, border_radius=16)

        x = self.sidebar_x + 8
        y = self.margin + 16
        title = self.big_font.render("Parcheesi", True, self.TEXT)
        self.screen.blit(title, (x, y))
        y += 44

        lines = [
            f"Turn: Player {state.current_player}",
            f"Dice: {state.dice_rolls[0]}, {state.dice_rolls[1]}",
            "Remaining: " + (", ".join(map(str, state.rolls_remaining)) if state.rolls_remaining else "none"),
            "Drag current player's token.",
        ]
        if not p2_turn:
            lines.append("Waiting for Player 1 / next turn.")
        if state.last_error:
            lines.append(f"Error: {state.last_error}")

        for line in lines:
            color = (255, 212, 120) if line.startswith("Error:") else self.MUTED
            surf = self.font.render(line, True, color)
            self.screen.blit(surf, (x, y))
            y += 24

        y += 8
        for player in range(1, ParcheesiState.NUM_PLAYERS + 1):
            done = len(state.home_areas[player])
            nest = len(state.nests[player])
            color = self.PLAYER_COLORS[player]
            pygame.draw.circle(self.screen, color, (x + 8, y + 10), 8)
            surf = self.font.render(f"P{player}: nest {nest}, home {done}/4", True, self.TEXT)
            self.screen.blit(surf, (x + 24, y))
            y += 24

    def draw(
        self,
        state: ParcheesiState,
        p2_turn: bool = True,
        last_move: Optional[Tuple[str, str]] = None,
    ) -> None:
        self.screen.fill(self.BG)
        self._draw_board_base(state)

        if last_move is not None:
            self._draw_highlight(last_move[0], self.LAST_SRC)
            self._draw_highlight(last_move[1], self.LAST_DST)

        if self.dragging_piece is not None:
            possible = [
                end_id
                for piece, start_id, end_id in state.get_possible_moves(state.current_player)
                if piece == self.dragging_piece and start_id == self.drag_start_id
            ]
            for end_id in possible:
                self._draw_highlight(end_id, self.DEST_HL)

        self._draw_pieces(state)
        self._draw_sidebar(state, p2_turn)

    def _location_at(self, pos: tuple[int, int]) -> str | None:
        for location_id, rect in self.hit_rects.items():
            if rect.collidepoint(*pos):
                return location_id
        return None

    def _piece_at(self, state: ParcheesiState, pos: tuple[int, int]) -> Piece | None:
        for player in range(1, ParcheesiState.NUM_PLAYERS + 1):
            for piece in reversed(ParcheesiState.all_pieces_for_player(player)):
                cx, cy = self._piece_center(state, piece)
                if (pos[0] - cx) ** 2 + (pos[1] - cy) ** 2 <= self._piece_radius() ** 2:
                    return piece
        return None

    def handle_event(
        self,
        event: pygame.event.Event,
        state: ParcheesiState,
        p2_turn: bool,
    ) -> Optional[Tuple[str, str]]:
        if event.type == pygame.MOUSEBUTTONUP and not p2_turn:
            self.clear_drag()
            return None
        if not p2_turn:
            return None

        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            piece = self._piece_at(state, event.pos)
            if piece is None or piece.to_player_num() != state.current_player:
                return None
            self.dragging_piece = piece
            self.drag_start_id = state.piece_location_id(piece)
            cx, cy = self._piece_center(state, piece)
            self.drag_offset = (cx - event.pos[0], cy - event.pos[1])
            self.drag_current = (event.pos[0] + self.drag_offset[0], event.pos[1] + self.drag_offset[1])
            return None

        if event.type == pygame.MOUSEMOTION and self.dragging_piece is not None:
            self.drag_current = (event.pos[0] + self.drag_offset[0], event.pos[1] + self.drag_offset[1])
            return None

        if event.type == pygame.MOUSEBUTTONUP and event.button == 1 and self.dragging_piece is not None:
            end_id = self._location_at(event.pos)
            start_id = self.drag_start_id
            self.clear_drag()
            if start_id is None or end_id is None or start_id == end_id:
                return None
            return start_id, end_id

        return None
