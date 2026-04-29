"""Pygame Parcheesi board renderer."""

from __future__ import annotations

import math
from statistics import median
from typing import Any

import pygame

from parcheesi_state import ParcheesiState, Piece


class ParcheesiUI:
    BG = (48, 48, 52)
    BOARD_BG = (247, 241, 229)
    LINE = (0, 0, 0)
    GUIDE = (160, 160, 160)
    GOLD = (201, 162, 39)
    LABEL = (50, 50, 50)
    HOME_BG = (251, 242, 233)
    HOME_BORDER = (211, 156, 144)
    HOME_ACCENT = (230, 188, 176)
    HOME_TEXT = (178, 118, 108)
    CIRCLE_CELL_FILL = (232, 241, 249)
    GOAL_LANE_FILL = (239, 154, 145)
    TRIANGLE_TOP_FILL = (232, 241, 249)
    TRIANGLE_RIGHT_FILL = (247, 182, 173)
    TRIANGLE_BOTTOM_FILL = (170, 215, 150)
    TRIANGLE_LEFT_FILL = (222, 200, 90)
    CIRCLE_TOP_FILL = (186, 212, 235)
    CIRCLE_RIGHT_FILL = (247, 182, 173)
    CIRCLE_BOTTOM_FILL = (170, 215, 150)
    CIRCLE_LEFT_FILL = (222, 200, 90)
    PLAYER_PIECE_COLORS = {
        1: (220, 78, 72),
        2: (63, 112, 214),
        3: (242, 195, 57),
        4: (69, 184, 109),
    }
    PIECE_LABEL_LIGHT = (255, 255, 255)
    PIECE_SHADOW = (0, 0, 0, 35)
    START_HL = (70, 150, 255, 85)
    DEST_HL = (255, 215, 60, 95)
    SELECT_RING = (45, 110, 220)
    DICE_BG = (40, 40, 46)
    DICE_FACE = (250, 250, 250)
    DICE_PIP = (32, 32, 38)
    DICE_STROKE = (70, 70, 78)
    DICE_TEXT = (235, 235, 240)
    CORNER_ARC_INSET_RATIO = 0.02
    CORNER_ARC_ENDPOINT_X_RATIO = 0.68
    CORNER_ARC_MID_AXIS_RATIO = 0.56
    ARM_END_EXTENSION_PX = 10
    BLUE_TRACK_POSITIONS = (1, 6, 13, 18, 23, 30, 35, 40, 47, 52, 57, 64)
    ARM_TEMPLATE_SHAPES: list[dict[str, Any]] = [
        {"type": "line", "p1": [0.688622754491018, 0.3473053892215569], "p2": [0.9550898203592815, 0.3473053892215569], "width": 3},
        {"type": "line", "p1": [0.688622754491018, 0.3502994011976048], "p2": [0.688622754491018, 0.6467065868263473], "width": 3},
        {"type": "line", "p1": [0.688622754491018, 0.6467065868263473], "p2": [0.9550898203592815, 0.6511976047904192], "width": 3},
        {"type": "line", "p1": [0.9565868263473054, 0.5479041916167665], "p2": [0.6482035928143712, 0.5464071856287425], "width": 3},
        {"type": "line", "p1": [0.6482035928143712, 0.5464071856287425], "p2": [0.6497005988023952, 0.44610778443113774], "width": 3},
        {"type": "line", "p1": [0.6497005988023952, 0.44610778443113774], "p2": [0.9535928143712575, 0.4491017964071856], "width": 3},
        {"type": "line", "p1": [0.9176646706586826, 0.3473053892215569], "p2": [0.9176646706586826, 0.6511976047904192], "width": 3},
        {"type": "line", "p1": [0.8772455089820359, 0.3458083832335329], "p2": [0.8772455089820359, 0.6497005988023952], "width": 3},
        {"type": "line", "p1": [0.8398203592814372, 0.3473053892215569], "p2": [0.8398203592814372, 0.6497005988023952], "width": 3},
        {"type": "line", "p1": [0.8038922155688623, 0.3502994011976048], "p2": [0.8038922155688623, 0.6452095808383234], "width": 3},
        {"type": "line", "p1": [0.7634730538922155, 0.6467065868263473], "p2": [0.7649700598802395, 0.344311377245509], "width": 3},
        {"type": "line", "p1": [0.7260479041916168, 0.3502994011976048], "p2": [0.7245508982035929, 0.6497005988023952], "width": 3},
        {"type": "circle", "center": [0.7844311377245509, 0.39820359281437123], "radius": 0.018026339189808827, "width": 3},
        {"type": "circle", "center": [0.937125748502994, 0.49850299401197606], "radius": 0.019461077844311378, "width": 3},
        {"type": "circle", "center": [0.7829341317365269, 0.5988023952095808], "radius": 0.019461077844311378, "width": 3},
        {"type": "line", "p1": [0.688622754491018, 0.4820359281437126], "p2": [0.6586826347305389, 0.49850299401197606], "width": 3},
        {"type": "line", "p1": [0.6586826347305389, 0.49850299401197606], "p2": [0.6901197604790419, 0.5149700598802395], "width": 3},
        {"type": "line", "p1": [0.688622754491018, 0.6467065868263473], "p2": [0.6422155688622755, 0.6916167664670658], "width": 3},
        {"type": "line", "p1": [0.6676646706586826, 0.6706586826347305], "p2": [0.6437125748502994, 0.6467065868263473], "width": 3},
    ]
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
        self.label_font = pygame.font.Font(None, 15)
        self.piece_font = pygame.font.Font(None, 16)
        self.dice_font = pygame.font.Font(None, 18)
        self.margin = 42
        self.header_height = 128
        self.footer_height = 44
        self.board_rect = self._build_board_rect()
        self.selected_start_id: str | None = None
        self.drag_start_id: str | None = None
        self.drag_piece: Piece | None = None
        self.drag_mouse: tuple[int, int] = self.board_rect.center
        self.drag_origin: tuple[int, int] | None = None
        self.drag_moved = False

    def _build_board_rect(self) -> pygame.Rect:
        usable_top = self.header_height
        usable_bottom = self.height - self.footer_height
        usable_height = max(300, usable_bottom - usable_top)
        size = min(self.width - 2 * self.margin, usable_height - 20)
        size = max(320, size)
        left = (self.width - size) // 2
        top = usable_top + max(0, (usable_height - size) // 2)
        return pygame.Rect(left, top, size, size)

    def _to_normalized(self, pos: tuple[int, int]) -> tuple[float, float] | None:
        if not self.board_rect.collidepoint(pos):
            return None
        x = (pos[0] - self.board_rect.left) / self.board_rect.width
        y = (pos[1] - self.board_rect.top) / self.board_rect.height
        return (max(0.0, min(1.0, x)), max(0.0, min(1.0, y)))

    def _to_screen(self, point: tuple[float, float]) -> tuple[int, int]:
        return (
            round(self.board_rect.left + point[0] * self.board_rect.width),
            round(self.board_rect.top + point[1] * self.board_rect.height),
        )

    def _shape_width(self, shape: dict[str, Any]) -> int:
        return max(1, int(shape.get("width", 3)))

    def _rotate_point_about(
        self,
        point: tuple[float, float],
        pivot: tuple[float, float],
        quarter_turns: int,
    ) -> tuple[float, float]:
        x, y = point
        px, py = pivot
        turns = quarter_turns % 4
        for _ in range(turns):
            dx = x - px
            dy = y - py
            x = px - dy
            y = py + dx
        return (x, y)


    def _translate_point(self, point: tuple[float, float], dx: float, dy: float) -> tuple[float, float]:
        return (point[0] + dx, point[1] + dy)

    def _reflect_point_x(self, point: tuple[float, float], pivot_x: float = 0.5) -> tuple[float, float]:
        return (2.0 * pivot_x - point[0], point[1])

    def _translate_shape(self, shape: dict[str, Any], dx: float, dy: float) -> dict[str, Any]:
        translated = dict(shape)
        shape_type = shape.get("type")
        if shape_type in {"line", "rect"}:
            translated["p1"] = list(self._translate_point(tuple(shape["p1"]), dx, dy))
            translated["p2"] = list(self._translate_point(tuple(shape["p2"]), dx, dy))
        elif shape_type == "circle":
            translated["center"] = list(self._translate_point(tuple(shape["center"]), dx, dy))
        elif shape_type == "arc":
            translated["center"] = list(self._translate_point(tuple(shape["center"]), dx, dy))
        elif shape_type == "arc3":
            translated["p1"] = list(self._translate_point(tuple(shape["p1"]), dx, dy))
            translated["p2"] = list(self._translate_point(tuple(shape["p2"]), dx, dy))
            translated["p3"] = list(self._translate_point(tuple(shape["p3"]), dx, dy))
        return translated

    def _reflect_shape_x(self, shape: dict[str, Any], pivot_x: float = 0.5) -> dict[str, Any]:
        reflected = dict(shape)
        shape_type = shape.get("type")
        if shape_type in {"line", "rect"}:
            reflected["p1"] = list(self._reflect_point_x(tuple(shape["p1"]), pivot_x))
            reflected["p2"] = list(self._reflect_point_x(tuple(shape["p2"]), pivot_x))
        elif shape_type == "circle":
            reflected["center"] = list(self._reflect_point_x(tuple(shape["center"]), pivot_x))
        elif shape_type == "arc":
            reflected["center"] = list(self._reflect_point_x(tuple(shape["center"]), pivot_x))
            reflected["start_angle"] = math.pi - float(shape["end_angle"])
            reflected["end_angle"] = math.pi - float(shape["start_angle"])
        elif shape_type == "arc3":
            reflected["p1"] = list(self._reflect_point_x(tuple(shape["p1"]), pivot_x))
            reflected["p2"] = list(self._reflect_point_x(tuple(shape["p2"]), pivot_x))
            reflected["p3"] = list(self._reflect_point_x(tuple(shape["p3"]), pivot_x))
        return reflected


    def _shape_bounds(self, shape: dict[str, Any]) -> tuple[float, float, float, float]:
        shape_type = shape.get("type")
        if shape_type in {"line", "rect"}:
            x1, y1 = shape["p1"]
            x2, y2 = shape["p2"]
            return (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))
        if shape_type == "circle":
            cx, cy = shape["center"]
            radius = float(shape["radius"])
            return (cx - radius, cy - radius, cx + radius, cy + radius)
        if shape_type == "arc":
            cx, cy = shape["center"]
            radius = float(shape["radius"])
            return (cx - radius, cy - radius, cx + radius, cy + radius)
        if shape_type == "arc3":
            xs = [shape["p1"][0], shape["p2"][0], shape["p3"][0]]
            ys = [shape["p1"][1], shape["p2"][1], shape["p3"][1]]
            return (min(xs), min(ys), max(xs), max(ys))
        return (0.0, 0.0, 0.0, 0.0)

    def _shapes_bounds(self, shapes: list[dict[str, Any]]) -> tuple[float, float, float, float]:
        bounds = [self._shape_bounds(shape) for shape in shapes]
        return (
            min(bound[0] for bound in bounds),
            min(bound[1] for bound in bounds),
            max(bound[2] for bound in bounds),
            max(bound[3] for bound in bounds),
        )


    def _build_left_arm_basis(self) -> list[dict[str, Any]]:
        cleaned_template = self._build_clean_black_template_shapes()
        if not cleaned_template:
            return []
        left_arm = [self._reflect_shape_x(shape) for shape in cleaned_template]
        self._symmetrize_left_arm(left_arm)
        self._canonicalize_left_arm_grid(left_arm)
        self._extend_left_arm_outer_tip(left_arm)
        self._canonicalize_left_arm_connectors(left_arm)
        return left_arm

    def _build_black_arm_shapes(self) -> list[dict[str, Any]]:
        left_arm = self._build_left_arm_basis()
        if not left_arm:
            return []
        pivot = (0.5, 0.5)
        black_shapes: list[dict[str, Any]] = []
        for quarter_turns in range(4):
            black_shapes.extend(
                self._rotate_shape_about(shape, pivot, quarter_turns)
                for shape in left_arm
            )
        black_shapes.extend(self._build_corner_circle_shapes(left_arm))
        black_shapes.extend(self._build_corner_circle_arc_shapes(left_arm))
        return black_shapes

    def _snap_value(self, value: float, guides: list[float], tol: float = 0.012) -> float:
        if not guides:
            return value
        nearest = min(guides, key=lambda guide: abs(guide - value))
        return nearest if abs(nearest - value) <= tol else value

    def _build_clean_black_template_shapes(self) -> list[dict[str, Any]]:
        template_shapes = self.ARM_TEMPLATE_SHAPES

        horizontal_ys: list[float] = []
        vertical_xs: list[float] = []
        for shape in template_shapes:
            if shape.get("type") != "line":
                continue
            x1, y1 = shape["p1"]
            x2, y2 = shape["p2"]
            dx = abs(float(x2) - float(x1))
            dy = abs(float(y2) - float(y1))
            if dy <= max(0.01, dx * 0.12):
                horizontal_ys.append((float(y1) + float(y2)) / 2.0)
            elif dx <= max(0.01, dy * 0.12):
                vertical_xs.append((float(x1) + float(x2)) / 2.0)

        horizontal_guides = self._cluster_sorted(horizontal_ys, tol=0.012)
        vertical_guides = self._cluster_sorted(vertical_xs, tol=0.012)

        cleaned: list[dict[str, Any]] = []
        for shape in template_shapes:
            shape_type = shape.get("type")
            if shape_type == "line":
                x1, y1 = map(float, shape["p1"])
                x2, y2 = map(float, shape["p2"])
                dx = abs(x2 - x1)
                dy = abs(y2 - y1)
                snapped_x1 = self._snap_value(x1, vertical_guides)
                snapped_x2 = self._snap_value(x2, vertical_guides)
                snapped_y1 = self._snap_value(y1, horizontal_guides)
                snapped_y2 = self._snap_value(y2, horizontal_guides)
                cleaned_line = dict(shape)
                if dy <= max(0.01, dx * 0.12):
                    y = self._snap_value((y1 + y2) / 2.0, horizontal_guides)
                    cleaned_line["p1"] = [snapped_x1, y]
                    cleaned_line["p2"] = [snapped_x2, y]
                elif dx <= max(0.01, dy * 0.12):
                    x = self._snap_value((x1 + x2) / 2.0, vertical_guides)
                    cleaned_line["p1"] = [x, snapped_y1]
                    cleaned_line["p2"] = [x, snapped_y2]
                else:
                    cleaned_line["p1"] = [snapped_x1, snapped_y1]
                    cleaned_line["p2"] = [snapped_x2, snapped_y2]
                cleaned.append(cleaned_line)
            elif shape_type == "circle":
                cleaned_circle = dict(shape)
                cx, cy = map(float, shape["center"])
                cleaned_circle["center"] = [
                    self._snap_value(cx, vertical_guides),
                    self._snap_value(cy, horizontal_guides),
                ]
                cleaned.append(cleaned_circle)
            elif shape_type == "arc3":
                cleaned_arc = dict(shape)
                points = []
                for key in ("p1", "p2", "p3"):
                    px, py = map(float, shape[key])
                    points.append(
                        [
                            self._snap_value(px, vertical_guides),
                            self._snap_value(py, horizontal_guides),
                        ]
                    )
                cleaned_arc["p1"], cleaned_arc["p2"], cleaned_arc["p3"] = points
                cleaned.append(cleaned_arc)
            elif shape_type == "arc":
                cleaned_arc = dict(shape)
                cx, cy = map(float, shape["center"])
                cleaned_arc["center"] = [
                    self._snap_value(cx, vertical_guides),
                    self._snap_value(cy, horizontal_guides),
                ]
                cleaned.append(cleaned_arc)
            else:
                cleaned.append(dict(shape))
        self._canonicalize_connector_diagonals(cleaned)
        return cleaned

    def _rotate_shape_about(
        self,
        shape: dict[str, Any],
        pivot: tuple[float, float],
        quarter_turns: int,
    ) -> dict[str, Any]:
        if quarter_turns % 4 == 0:
            return dict(shape)
        rotated = dict(shape)
        shape_type = shape.get("type")
        if shape_type in {"line", "rect"}:
            rotated["p1"] = list(self._rotate_point_about(tuple(shape["p1"]), pivot, quarter_turns))
            rotated["p2"] = list(self._rotate_point_about(tuple(shape["p2"]), pivot, quarter_turns))
        elif shape_type == "circle":
            rotated["center"] = list(self._rotate_point_about(tuple(shape["center"]), pivot, quarter_turns))
        elif shape_type == "arc":
            angle_offset = quarter_turns * (math.pi / 2.0)
            rotated["center"] = list(self._rotate_point_about(tuple(shape["center"]), pivot, quarter_turns))
            rotated["start_angle"] = float(shape["start_angle"]) + angle_offset
            rotated["end_angle"] = float(shape["end_angle"]) + angle_offset
        elif shape_type == "arc3":
            rotated["p1"] = list(self._rotate_point_about(tuple(shape["p1"]), pivot, quarter_turns))
            rotated["p2"] = list(self._rotate_point_about(tuple(shape["p2"]), pivot, quarter_turns))
            rotated["p3"] = list(self._rotate_point_about(tuple(shape["p3"]), pivot, quarter_turns))
        return rotated

    def _unique_points(self, points: list[tuple[float, float]], tol: float = 0.006) -> list[tuple[float, float]]:
        unique: list[tuple[float, float]] = []
        for point in points:
            if not any(abs(point[0] - other[0]) <= tol and abs(point[1] - other[1]) <= tol for other in unique):
                unique.append(point)
        return unique

    def _unwrap_arc_angles(self, start: float, through: float, end: float) -> tuple[float, float, float]:
        while through < start:
            through += 2 * math.pi
        while end < start:
            end += 2 * math.pi
        while end < through:
            end += 2 * math.pi
        return start, through, end

    def _normalize_angle(self, angle: float) -> float:
        angle = math.fmod(angle, 2.0 * math.pi)
        if angle < 0.0:
            angle += 2.0 * math.pi
        return angle

    def _ccw_delta(self, start: float, end: float) -> float:
        return (end - start) % (2.0 * math.pi)

    def _choose_arc_span(self, start: float, through: float, end: float) -> tuple[float, float]:
        start = self._normalize_angle(start)
        through = self._normalize_angle(through)
        end = self._normalize_angle(end)

        options: list[tuple[float, float, float]] = []

        ccw_sweep = self._ccw_delta(start, end)
        ccw_through = self._ccw_delta(start, through)
        if ccw_through <= ccw_sweep + 1e-9:
            options.append((ccw_sweep, start, start + ccw_sweep))

        cw_sweep = self._ccw_delta(end, start)
        cw_through = self._ccw_delta(end, through)
        if cw_through <= cw_sweep + 1e-9:
            options.append((cw_sweep, end, end + cw_sweep))

        if not options:
            return start, start + ccw_sweep

        _sweep, arc_start, arc_end = min(options, key=lambda item: item[0])
        return arc_start, arc_end

    def _circle_from_three_points(
        self,
        p1: tuple[float, float],
        p2: tuple[float, float],
        p3: tuple[float, float],
    ) -> tuple[tuple[float, float], float, float, float] | None:
        ax, ay = self._to_screen(p1)
        bx, by = self._to_screen(p2)
        cx, cy = self._to_screen(p3)
        det = 2.0 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
        if abs(det) < 1e-6:
            return None
        ux = ((ax * ax + ay * ay) * (by - cy) + (bx * bx + by * by) * (cy - ay) + (cx * cx + cy * cy) * (ay - by)) / det
        uy = ((ax * ax + ay * ay) * (cx - bx) + (bx * bx + by * by) * (ax - cx) + (cx * cx + cy * cy) * (bx - ax)) / det
        radius = math.hypot(ax - ux, ay - uy)
        start = math.atan2(ay - uy, ax - ux)
        through = math.atan2(by - uy, bx - ux)
        end = math.atan2(cy - uy, cx - ux)
        arc_start, arc_end = self._choose_arc_span(start, through, end)
        return ((ux, uy), radius, arc_start, arc_end)

    def _sample_arc_points(
        self,
        center_px: tuple[float, float],
        radius_px: float,
        start_angle: float,
        end_angle: float,
    ) -> list[tuple[int, int]]:
        sweep = max(0.001, end_angle - start_angle)
        steps = max(24, round(radius_px * sweep / 8.0))
        points: list[tuple[int, int]] = []
        for idx in range(steps + 1):
            t = idx / steps
            angle = start_angle + sweep * t
            x = center_px[0] + radius_px * math.cos(angle)
            y = center_px[1] + radius_px * math.sin(angle)
            points.append((round(x), round(y)))
        return points

    def _sample_quadratic_points(
        self,
        p1: tuple[int, int],
        p2: tuple[int, int],
        p3: tuple[int, int],
    ) -> list[tuple[int, int]]:
        chord = math.hypot(p3[0] - p1[0], p3[1] - p1[1])
        control_span = max(
            math.hypot(p2[0] - p1[0], p2[1] - p1[1]),
            math.hypot(p3[0] - p2[0], p3[1] - p2[1]),
        )
        steps = max(24, round(max(chord, control_span) / 6.0))
        points: list[tuple[int, int]] = []
        for idx in range(steps + 1):
            t = idx / steps
            omt = 1.0 - t
            x = (omt * omt * p1[0]) + (2.0 * omt * t * p2[0]) + (t * t * p3[0])
            y = (omt * omt * p1[1]) + (2.0 * omt * t * p2[1]) + (t * t * p3[1])
            points.append((round(x), round(y)))
        if points:
            points[0] = p1
            points[-1] = p3
        return points

    def _sample_circle_arc_screen(
        self,
        center: tuple[float, float],
        radius: float,
        start_point: tuple[float, float],
        end_point: tuple[float, float],
    ) -> list[tuple[int, int]]:
        cx, cy = self._to_screen(center)
        radius_px = max(1.0, radius * self.board_rect.width)
        start_screen = self._to_screen(start_point)
        end_screen = self._to_screen(end_point)
        start_angle = self._normalize_angle(math.atan2(start_screen[1] - cy, start_screen[0] - cx))
        end_angle = self._normalize_angle(math.atan2(end_screen[1] - cy, end_screen[0] - cx))
        if end_angle < start_angle:
            end_angle += 2.0 * math.pi
        points = self._sample_arc_points((cx, cy), radius_px, start_angle, end_angle)
        if points:
            points[0] = start_screen
            points[-1] = end_screen
        return points

    def _arc3_screen_points(self, shape: dict[str, Any]) -> list[tuple[int, int]]:
        p1 = self._to_screen(tuple(shape["p1"]))
        p2 = self._to_screen(tuple(shape["p2"]))
        p3 = self._to_screen(tuple(shape["p3"]))
        return self._sample_quadratic_points(p1, p2, p3)

    def _build_arm_bounds_rect(self, shapes: list[dict[str, Any]]) -> pygame.Rect | None:
        line_like = [shape for shape in shapes if shape.get("type") in {"line", "rect"}]
        if not line_like:
            return None
        min_x, min_y, max_x, max_y = self._shapes_bounds(line_like)
        left = round(self.board_rect.left + min_x * self.board_rect.width)
        top = round(self.board_rect.top + min_y * self.board_rect.height)
        right = round(self.board_rect.left + max_x * self.board_rect.width)
        bottom = round(self.board_rect.top + max_y * self.board_rect.height)
        return pygame.Rect(left, top, max(1, right - left), max(1, bottom - top))

    def _build_home_region_rect(self, left_arm: list[dict[str, Any]]) -> pygame.Rect | None:
        guides = self._extract_arm_guides(left_arm)
        if guides is None:
            return None
        top = float(guides["outer_top_y"])
        bottom = float(guides["outer_bottom_y"])
        left = 1.0 - bottom
        right = 1.0 - top
        screen_left = round(self.board_rect.left + left * self.board_rect.width)
        screen_top = round(self.board_rect.top + top * self.board_rect.height)
        screen_right = round(self.board_rect.left + right * self.board_rect.width)
        screen_bottom = round(self.board_rect.top + bottom * self.board_rect.height)
        return pygame.Rect(
            screen_left,
            screen_top,
            max(1, screen_right - screen_left),
            max(1, screen_bottom - screen_top),
        )

    def _build_circle_cell_shapes(self, left_arm: list[dict[str, Any]]) -> list[dict[str, Any]]:
        bounds = self._build_rule_grid_boundaries(left_arm)
        if bounds is None:
            return []
        x_bounds, y_bounds = bounds

        def cell_rect(gx: int, gy: int) -> dict[str, Any]:
            return {
                "type": "rect",
                "p1": [x_bounds[gx], y_bounds[gy]],
                "p2": [x_bounds[gx + 1], y_bounds[gy + 1]],
            }

        rects: list[dict[str, Any]] = []
        for pos in self.BLUE_TRACK_POSITIONS:
            gx, gy = ParcheesiState.track_grid_position(pos)
            rects.append(cell_rect(int(gx), int(gy)))
        return rects

    def _build_goal_lane_shapes(self, left_arm: list[dict[str, Any]]) -> list[dict[str, Any]]:
        guides = self._extract_arm_guides(left_arm)
        if guides is None:
            return []

        outer_top_y = float(guides["outer_top_y"])
        outer_bottom_y = float(guides["outer_bottom_y"])
        row_height = (outer_bottom_y - outer_top_y) / 3.0
        middle_top_y = outer_top_y + row_height
        middle_bottom_y = outer_top_y + (2.0 * row_height)

        middle_row_tip_x = float(guides["outer_right_x"])
        tol = 0.01
        for shape in left_arm:
            if shape.get("type") != "line":
                continue
            x1, y1 = map(float, shape["p1"])
            x2, y2 = map(float, shape["p2"])
            if abs(y2 - y1) > tol:
                continue
            y = (y1 + y2) / 2.0
            if abs(y - middle_top_y) <= 0.02 or abs(y - middle_bottom_y) <= 0.02:
                middle_row_tip_x = max(middle_row_tip_x, x1, x2)

        x_cuts = [
            float(guides["outer_left_x"]),
            *[float(x) for x in sorted(guides["stripe_xs"])],
            float(guides["outer_right_x"]),
            middle_row_tip_x,
        ]

        rects: list[dict[str, Any]] = []
        for index in range(len(x_cuts) - 1):
            rects.append(
                {
                    "type": "rect",
                    "p1": [x_cuts[index], middle_top_y],
                    "p2": [x_cuts[index + 1], middle_bottom_y],
                }
            )
        return rects

    def _screen_rect_from_shape_rect(self, shape: dict[str, Any]) -> pygame.Rect:
        p1 = self._to_screen(tuple(shape["p1"]))
        p2 = self._to_screen(tuple(shape["p2"]))
        return pygame.Rect(
            min(p1[0], p2[0]),
            min(p1[1], p2[1]),
            abs(p2[0] - p1[0]),
            abs(p2[1] - p1[1]),
        )

    def _build_rule_grid_boundaries(self, left_arm: list[dict[str, Any]]) -> tuple[list[float], list[float]] | None:
        guides = self._extract_arm_guides(left_arm)
        if guides is None:
            return None

        line_like = [shape for shape in left_arm if shape.get("type") in {"line", "rect"}]
        if not line_like:
            return None
        tip_outer = self._shapes_bounds(line_like)[0]

        middle_row_tip = float(guides["outer_right_x"])
        tol = 0.01
        row_height = (float(guides["outer_bottom_y"]) - float(guides["outer_top_y"])) / 3.0
        middle_top_y = float(guides["outer_top_y"]) + row_height
        middle_bottom_y = float(guides["outer_top_y"]) + (2.0 * row_height)
        for shape in left_arm:
            if shape.get("type") != "line":
                continue
            x1, y1 = map(float, shape["p1"])
            x2, y2 = map(float, shape["p2"])
            if abs(y2 - y1) > tol:
                continue
            y = (y1 + y2) / 2.0
            if abs(y - middle_top_y) <= 0.02 or abs(y - middle_bottom_y) <= 0.02:
                middle_row_tip = max(middle_row_tip, x1, x2)

        left_bounds = [
            tip_outer,
            float(guides["outer_left_x"]),
            *[float(x) for x in sorted(guides["stripe_xs"])],
            float(guides["outer_right_x"]),
            middle_row_tip,
        ]
        center_left = left_bounds[-1]
        center_right = 1.0 - center_left
        center_step = (center_right - center_left) / 3.0
        center_inner_1 = center_left + center_step
        center_inner_2 = center_left + 2.0 * center_step
        right_bounds = [1.0 - value for value in reversed(left_bounds)]
        x_bounds = left_bounds + [center_inner_1, center_inner_2] + right_bounds
        y_bounds = list(x_bounds)
        return x_bounds, y_bounds

    def _build_location_label_specs(self, left_arm: list[dict[str, Any]]) -> list[dict[str, Any]]:
        bounds = self._build_rule_grid_boundaries(left_arm)
        if bounds is None:
            return []
        x_bounds, y_bounds = bounds

        def rect_for_grid(gx: int, gy: int) -> pygame.Rect:
            left = round(self.board_rect.left + x_bounds[gx] * self.board_rect.width)
            right = round(self.board_rect.left + x_bounds[gx + 1] * self.board_rect.width)
            top = round(self.board_rect.top + y_bounds[gy] * self.board_rect.height)
            bottom = round(self.board_rect.top + y_bounds[gy + 1] * self.board_rect.height)
            return pygame.Rect(left, top, max(1, right - left), max(1, bottom - top))

        specs: list[dict[str, Any]] = []
        for pos in range(ParcheesiState.MAIN_TRACK_LENGTH):
            gx, gy = ParcheesiState.track_grid_position(pos)
            rect = rect_for_grid(int(gx), int(gy))
            specs.append(
                {
                    "location_id": ParcheesiState.location_id("main", pos=pos),
                    "display_number": str(pos),
                    "rect": rect,
                }
            )

        next_number = ParcheesiState.MAIN_TRACK_LENGTH
        for player in range(1, ParcheesiState.NUM_PLAYERS + 1):
            for pos in range(ParcheesiState.HOME_PATH_LENGTH):
                gx, gy = ParcheesiState.home_grid_position(player, pos)
                rect = rect_for_grid(int(gx), int(gy))
                specs.append(
                    {
                        "location_id": ParcheesiState.location_id("home", player, pos),
                        "display_number": str(next_number),
                        "rect": rect,
                    }
                )
                next_number += 1
        return specs

    def _draw_location_labels(self, specs: list[dict[str, Any]]) -> None:
        for spec in specs:
            rect = spec["rect"]
            label = str(spec["display_number"])
            surf = self.label_font.render(label, True, self.LABEL)
            x = rect.left + 4
            y = rect.top + 3
            self.screen.blit(surf, (x, y))

    def _grid_point_to_screen(self, gx: float, gy: float) -> tuple[int, int]:
        normalized = (
            gx / ParcheesiState.GRID_MAX,
            gy / ParcheesiState.GRID_MAX,
        )
        return self._to_screen(normalized)

    def _location_rect_map(self, left_arm: list[dict[str, Any]]) -> dict[str, pygame.Rect]:
        specs = self._build_location_label_specs(left_arm)
        return {str(spec["location_id"]): spec["rect"] for spec in specs}

    def location_id_at_pixel(self, pos: tuple[int, int]) -> str | None:
        left_arm = self._build_left_arm_basis()
        rect_map = self._location_rect_map(left_arm) if left_arm else {}
        radius = self._piece_radius()
        for location_id in ParcheesiState.iter_draw_locations():
            rect = self._interactive_rect_for_location(location_id, rect_map, radius)
            if rect.collidepoint(pos):
                return location_id
        return None

    def _piece_radius(self) -> int:
        return max(10, round(self.board_rect.width * 0.013))

    def _stack_offsets(self, count: int, spread: int) -> list[tuple[int, int]]:
        if count <= 1:
            return [(0, 0)]
        if count == 2:
            return [(-spread, -spread), (spread, spread)]
        if count == 3:
            return [(-spread, -spread), (spread, -spread), (0, spread)]
        return [
            (-spread, -spread),
            (spread, -spread),
            (-spread, spread),
            (spread, spread),
        ][:count]

    def _interactive_rect_for_location(
        self,
        location_id: str,
        rect_map: dict[str, pygame.Rect],
        radius: int,
    ) -> pygame.Rect:
        rect = rect_map.get(location_id)
        if rect is not None:
            return rect
        gx, gy = ParcheesiState.location_id_to_grid(location_id)
        cx, cy = self._grid_point_to_screen(gx, gy)
        size = radius * 2 + 8
        return pygame.Rect(cx - size // 2, cy - size // 2, size, size)

    def _piece_centers_for_location(
        self,
        location_id: str,
        count: int,
        rect_map: dict[str, pygame.Rect],
        radius: int,
    ) -> list[tuple[int, int]]:
        rect = rect_map.get(location_id)
        if rect is not None:
            base_center = rect.center
            spread = max(6, min(rect.width, rect.height) // 5)
        else:
            gx, gy = ParcheesiState.location_id_to_grid(location_id)
            base_center = self._grid_point_to_screen(gx, gy)
            spread = max(6, radius + 2)
        offsets = self._stack_offsets(count, spread)
        return [(base_center[0] + dx, base_center[1] + dy) for dx, dy in offsets]

    def _piece_label_color(self, player: int) -> tuple[int, int, int]:
        return self.PIECE_LABEL_LIGHT if player in {1, 2} else self.LINE

    def _p2_move_map(self, state: ParcheesiState) -> dict[str, list[str]]:
        move_map: dict[str, list[str]] = {}
        if state.current_player != 2:
            return move_map
        for _piece, start_id, end_id in state.get_possible_moves(2):
            move_map.setdefault(start_id, []).append(end_id)
        return move_map

    def _draw_location_overlay(self, rect: pygame.Rect, rgba: tuple[int, int, int, int]) -> None:
        overlay = pygame.Surface((rect.width, rect.height), pygame.SRCALPHA)
        overlay.fill(rgba)
        self.screen.blit(overlay, rect.topleft)

    def _draw_parcheesi_pieces(
        self,
        state: ParcheesiState,
        rect_map: dict[str, pygame.Rect],
    ) -> None:
        radius = self._piece_radius()
        groups: dict[str, list[Piece]] = {}
        for piece, loc in state.piece_locations.items():
            if piece is Piece.EMPTY:
                continue
            groups.setdefault(state.piece_location_id(piece), []).append(piece)

        for location_id, pieces in groups.items():
            visible_pieces = sorted(pieces, key=lambda p: p.to_token_num())
            if self.drag_piece is not None and self.drag_start_id == location_id and self.drag_moved:
                visible_pieces = [piece for piece in visible_pieces if piece is not self.drag_piece]
            centers = self._piece_centers_for_location(location_id, len(visible_pieces), rect_map, radius)
            for piece, center in zip(visible_pieces, centers):
                player = piece.to_player_num()
                shadow = pygame.Surface((radius * 3, radius * 3), pygame.SRCALPHA)
                shadow_center = (shadow.get_width() // 2, shadow.get_height() // 2)
                pygame.draw.circle(shadow, self.PIECE_SHADOW, shadow_center, radius + 1)
                self.screen.blit(shadow, shadow.get_rect(center=(center[0] + 1, center[1] + 2)))
                pygame.draw.circle(self.screen, self.PLAYER_PIECE_COLORS[player], center, radius)
                pygame.draw.circle(self.screen, self.LINE, center, radius, width=1)
                label = self.piece_font.render(str(piece.to_token_num()), True, self._piece_label_color(player))
                self.screen.blit(label, label.get_rect(center=center))

        if self.drag_piece is not None and self.drag_moved:
            player = self.drag_piece.to_player_num()
            center = self.drag_mouse
            shadow = pygame.Surface((radius * 3, radius * 3), pygame.SRCALPHA)
            shadow_center = (shadow.get_width() // 2, shadow.get_height() // 2)
            pygame.draw.circle(shadow, self.PIECE_SHADOW, shadow_center, radius + 1)
            self.screen.blit(shadow, shadow.get_rect(center=(center[0] + 1, center[1] + 2)))
            pygame.draw.circle(self.screen, self.PLAYER_PIECE_COLORS[player], center, radius)
            pygame.draw.circle(self.screen, self.LINE, center, radius, width=1)
            label = self.piece_font.render(str(self.drag_piece.to_token_num()), True, self._piece_label_color(player))
            self.screen.blit(label, label.get_rect(center=center))

    def _draw_move_hints(
        self,
        state: ParcheesiState,
        rect_map: dict[str, pygame.Rect],
        p2_turn: bool,
    ) -> None:
        if not p2_turn:
            self.selected_start_id = None
            return
        move_map = self._p2_move_map(state)
        if self.selected_start_id not in move_map:
            self.selected_start_id = None
        radius = self._piece_radius()
        for start_id in move_map:
            rect = self._interactive_rect_for_location(start_id, rect_map, radius)
            self._draw_location_overlay(rect, self.START_HL)
        if self.selected_start_id is not None:
            start_rect = self._interactive_rect_for_location(self.selected_start_id, rect_map, radius)
            pygame.draw.rect(self.screen, self.SELECT_RING, start_rect, width=3, border_radius=8)
            for dest_id in move_map.get(self.selected_start_id, []):
                dest_rect = self._interactive_rect_for_location(dest_id, rect_map, radius)
                self._draw_location_overlay(dest_rect, self.DEST_HL)
                pygame.draw.rect(self.screen, self.GOLD, dest_rect, width=2, border_radius=8)

    def _draw_dice_panel(self, state: ParcheesiState) -> None:
        panel_w = 180
        panel_h = 72
        panel = pygame.Rect(self.screen.get_width() - 202, self.screen.get_height() - 142, panel_w, panel_h)
        pygame.draw.rect(self.screen, self.DICE_BG, panel, border_radius=10)
        pygame.draw.rect(self.screen, self.DICE_STROKE, panel, width=1, border_radius=10)

        title = self.small_font.render("Dice", True, self.DICE_TEXT)
        self.screen.blit(title, (panel.left + 10, panel.top + 8))

        die_values = state.dice_rolls if state.dice_rolls != (0, 0) else (0, 0)
        die_size = 34
        die_y = panel.top + 26
        die_gap = 12
        die_left = panel.left + 10
        for idx, value in enumerate(die_values):
            rect = pygame.Rect(die_left + idx * (die_size + die_gap), die_y, die_size, die_size)
            pygame.draw.rect(self.screen, self.DICE_FACE, rect, border_radius=6)
            pygame.draw.rect(self.screen, self.DICE_STROKE, rect, width=1, border_radius=6)
            if value > 0:
                self._draw_die_pips(rect, value)

        remaining = ",".join(str(v) for v in state.rolls_remaining) if state.rolls_remaining else "none"
        rem_text = self.small_font.render(f"Left: {remaining}", True, self.DICE_TEXT)
        self.screen.blit(rem_text, (panel.left + 96, panel.top + 34))

    def _draw_die_pips(self, rect: pygame.Rect, value: int) -> None:
        cx, cy = rect.center
        dx = rect.width // 4
        dy = rect.height // 4
        spots = {
            1: [(cx, cy)],
            2: [(cx - dx, cy - dy), (cx + dx, cy + dy)],
            3: [(cx - dx, cy - dy), (cx, cy), (cx + dx, cy + dy)],
            4: [(cx - dx, cy - dy), (cx + dx, cy - dy), (cx - dx, cy + dy), (cx + dx, cy + dy)],
            5: [(cx - dx, cy - dy), (cx + dx, cy - dy), (cx, cy), (cx - dx, cy + dy), (cx + dx, cy + dy)],
            6: [(cx - dx, cy - dy), (cx + dx, cy - dy), (cx - dx, cy), (cx + dx, cy), (cx - dx, cy + dy), (cx + dx, cy + dy)],
        }
        for px, py in spots.get(value, []):
            pygame.draw.circle(self.screen, self.DICE_PIP, (px, py), 3)

    def _build_triangle_fill_shapes(self, left_arm: list[dict[str, Any]]) -> list[dict[str, Any]]:
        guides = self._extract_arm_guides(left_arm)
        if guides is None:
            return []

        tol = 0.01
        outer_right_x = float(guides["outer_right_x"])
        diagonal_lines = [
            shape
            for shape in left_arm
            if shape.get("type") == "line"
            and abs(float(shape["p1"][0]) - float(shape["p2"][0])) > tol
            and abs(float(shape["p1"][1]) - float(shape["p2"][1])) > tol
        ]
        triangle_lines = [
            shape
            for shape in diagonal_lines
            if (
                abs(float(shape["p1"][0]) - outer_right_x) <= tol
                or abs(float(shape["p2"][0]) - outer_right_x) <= tol
            )
            and max(float(shape["p1"][1]), float(shape["p2"][1])) < float(guides["outer_bottom_y"]) - tol
        ]
        if len(triangle_lines) != 2:
            return []

        endpoints: list[tuple[float, float]] = []
        for shape in triangle_lines:
            endpoints.append((float(shape["p1"][0]), float(shape["p1"][1])))
            endpoints.append((float(shape["p2"][0]), float(shape["p2"][1])))

        tip = max(endpoints, key=lambda point: point[0])
        base_points = sorted(
            [point for point in endpoints if point != tip],
            key=lambda point: point[1],
        )
        if len(base_points) < 2:
            return []

        base_triangle = {
            "points": [base_points[0], tip, base_points[-1]],
            "color": self.TRIANGLE_LEFT_FILL,
        }
        triangle_colors = [
            self.TRIANGLE_LEFT_FILL,
            self.TRIANGLE_TOP_FILL,
            self.TRIANGLE_RIGHT_FILL,
            self.TRIANGLE_BOTTOM_FILL,
        ]
        pivot = (0.5, 0.5)
        fills: list[dict[str, Any]] = []
        for quarter_turns, color in enumerate(triangle_colors):
            fills.append(
                {
                    "points": [
                        self._rotate_point_about(point, pivot, quarter_turns)
                        for point in base_triangle["points"]
                    ],
                    "color": color,
                }
            )
        return fills

    def _build_special_circle_fill_shapes(self, left_arm: list[dict[str, Any]]) -> list[dict[str, Any]]:
        circles = [shape for shape in left_arm if shape.get("type") == "circle"]
        if len(circles) < 3:
            return []

        basis_circle = max(circles, key=lambda shape: float(shape["center"][1]))
        basis_center = tuple(float(value) for value in basis_circle["center"])
        radius = float(basis_circle["radius"])
        circle_colors = [
            self.CIRCLE_LEFT_FILL,
            self.CIRCLE_TOP_FILL,
            self.CIRCLE_RIGHT_FILL,
            self.CIRCLE_BOTTOM_FILL,
        ]
        pivot = (0.5, 0.5)
        fills: list[dict[str, Any]] = []
        for quarter_turns, color in enumerate(circle_colors):
            fills.append(
                {
                    "center": self._rotate_point_about(basis_center, pivot, quarter_turns),
                    "radius": radius,
                    "color": color,
                }
            )
        return fills

    def _draw_home_region(self, rect: pygame.Rect) -> None:
        pygame.draw.rect(self.screen, self.HOME_BG, rect)
        pygame.draw.rect(self.screen, self.HOME_BORDER, rect, width=3)

        inset1 = rect.inflate(-14, -14)
        inset2 = rect.inflate(-28, -28)
        inset3 = rect.inflate(-44, -44)
        pygame.draw.rect(self.screen, self.HOME_ACCENT, inset1, width=2)
        pygame.draw.rect(self.screen, self.HOME_BORDER, inset2, width=1)
        pygame.draw.rect(self.screen, self.HOME_ACCENT, inset3, width=1)

        self._draw_home_corner_details(inset1)
        self._draw_home_corner_details(inset2)

        center = rect.center
        lobe_radius = max(18, rect.width // 7)
        clover_rect = pygame.Rect(0, 0, lobe_radius * 2, lobe_radius * 2)
        lobe_centers = [
            (center[0], center[1] - lobe_radius),
            (center[0] + lobe_radius, center[1]),
            (center[0], center[1] + lobe_radius),
            (center[0] - lobe_radius, center[1]),
        ]
        for cx, cy in lobe_centers:
            clover_rect.center = (cx, cy)
            pygame.draw.ellipse(self.screen, self.HOME_BORDER, clover_rect, width=2)

        center_box = pygame.Rect(0, 0, max(76, rect.width // 2), max(36, rect.height // 6))
        center_box.center = center
        pygame.draw.rect(self.screen, self.HOME_BG, center_box, border_radius=center_box.height // 2)
        pygame.draw.rect(self.screen, self.HOME_BORDER, center_box, width=2, border_radius=center_box.height // 2)
        inner_box = center_box.inflate(-12, -12)
        pygame.draw.rect(self.screen, self.HOME_ACCENT, inner_box, width=1, border_radius=max(8, inner_box.height // 2))

        home_font_size = max(28, rect.width // 8)
        home_font = pygame.font.SysFont("georgia, baskerville, times new roman", home_font_size, bold=True)
        home_surf = home_font.render("HOME", True, self.HOME_TEXT)
        home_rect = home_surf.get_rect(center=center_box.center)
        self.screen.blit(home_surf, home_rect)

    def _draw_home_corner_details(self, rect: pygame.Rect) -> None:
        offset = max(8, rect.width // 18)
        span = max(10, rect.width // 12)
        points = [
            ((rect.left + offset, rect.top + offset), (rect.left + offset + span, rect.top + offset)),
            ((rect.left + offset, rect.top + offset), (rect.left + offset, rect.top + offset + span)),
            ((rect.right - offset, rect.top + offset), (rect.right - offset - span, rect.top + offset)),
            ((rect.right - offset, rect.top + offset), (rect.right - offset, rect.top + offset + span)),
            ((rect.left + offset, rect.bottom - offset), (rect.left + offset + span, rect.bottom - offset)),
            ((rect.left + offset, rect.bottom - offset), (rect.left + offset, rect.bottom - offset - span)),
            ((rect.right - offset, rect.bottom - offset), (rect.right - offset - span, rect.bottom - offset)),
            ((rect.right - offset, rect.bottom - offset), (rect.right - offset, rect.bottom - offset - span)),
        ]
        for start, end in points:
            pygame.draw.line(self.screen, self.HOME_ACCENT, start, end, width=1)

    def _draw_line(self, shape: dict[str, Any], color: tuple[int, int, int]) -> None:
        pygame.draw.line(
            self.screen,
            color,
            self._to_screen(tuple(shape["p1"])),
            self._to_screen(tuple(shape["p2"])),
            width=self._shape_width(shape),
        )

    def _draw_rect(self, shape: dict[str, Any], color: tuple[int, int, int]) -> None:
        p1 = self._to_screen(tuple(shape["p1"]))
        p2 = self._to_screen(tuple(shape["p2"]))
        rect = pygame.Rect(min(p1[0], p2[0]), min(p1[1], p2[1]), abs(p2[0] - p1[0]), abs(p2[1] - p1[1]))
        pygame.draw.rect(self.screen, color, rect, width=self._shape_width(shape))

    def _draw_circle(self, shape: dict[str, Any], color: tuple[int, int, int]) -> None:
        center = self._to_screen(tuple(shape["center"]))
        radius = max(1, round(float(shape["radius"]) * self.board_rect.width))
        pygame.draw.circle(self.screen, color, center, radius, width=self._shape_width(shape))

    def _draw_arc(self, shape: dict[str, Any], color: tuple[int, int, int]) -> None:
        center = self._to_screen(tuple(shape["center"]))
        radius = max(1.0, float(shape["radius"]) * self.board_rect.width)
        start_angle = float(shape["start_angle"])
        end_angle = float(shape["end_angle"])
        while end_angle <= start_angle:
            end_angle += 2 * math.pi
        points = self._sample_arc_points(center, radius, start_angle, end_angle)
        pygame.draw.lines(self.screen, color, False, points, width=self._shape_width(shape))

    def _draw_arc3(self, shape: dict[str, Any], color: tuple[int, int, int]) -> None:
        p1 = self._to_screen(tuple(shape["p1"]))
        p2 = self._to_screen(tuple(shape["p2"]))
        p3 = self._to_screen(tuple(shape["p3"]))
        points = self._sample_quadratic_points(p1, p2, p3)
        pygame.draw.lines(self.screen, color, False, points, width=self._shape_width(shape))

    def _draw_shape(self, shape: dict[str, Any], color: tuple[int, int, int]) -> None:
        shape_type = shape.get("type")
        if shape_type == "line":
            self._draw_line(shape, color)
        elif shape_type == "rect":
            self._draw_rect(shape, color)
        elif shape_type == "circle":
            self._draw_circle(shape, color)
        elif shape_type == "arc":
            self._draw_arc(shape, color)
        elif shape_type == "arc3":
            self._draw_arc3(shape, color)

    def _cluster_sorted(self, values: list[float], tol: float = 0.01) -> list[float]:
        if not values:
            return []
        values = sorted(values)
        groups: list[list[float]] = [[values[0]]]
        for value in values[1:]:
            if abs(value - groups[-1][-1]) <= tol:
                groups[-1].append(value)
            else:
                groups.append([value])
        return [sum(group) / len(group) for group in groups]

    def _extract_arm_guides(self, shapes: list[dict[str, Any]]) -> dict[str, Any] | None:
        tol = 0.01
        line_shapes = [shape for shape in shapes if shape.get("type") == "line"]
        horizontals: list[dict[str, Any]] = []
        verticals: list[dict[str, Any]] = []
        for shape in line_shapes:
            x1, y1 = map(float, shape["p1"])
            x2, y2 = map(float, shape["p2"])
            if abs(y2 - y1) <= tol:
                horizontals.append(shape)
            elif abs(x2 - x1) <= tol:
                verticals.append(shape)
        if not horizontals or not verticals:
            return None

        outer_top_y = min(float(shape["p1"][1]) for shape in horizontals)
        outer_bottom_y = max(float(shape["p1"][1]) for shape in horizontals)
        full_height_verticals = [
            shape
            for shape in verticals
            if min(float(shape["p1"][1]), float(shape["p2"][1])) <= outer_top_y + tol
            and max(float(shape["p1"][1]), float(shape["p2"][1])) >= outer_bottom_y - tol
        ]
        if len(full_height_verticals) < 2:
            return None

        xs = sorted(float(shape["p1"][0]) for shape in full_height_verticals)
        outer_left_x = xs[0]
        outer_right_x = xs[-1]
        stripe_xs = [x for x in xs if outer_left_x + tol < x < outer_right_x - tol]
        if not stripe_xs:
            return None
        return {
            "outer_top_y": outer_top_y,
            "outer_bottom_y": outer_bottom_y,
            "outer_left_x": outer_left_x,
            "outer_right_x": outer_right_x,
            "stripe_xs": stripe_xs,
        }

    def _symmetrize_left_arm(self, shapes: list[dict[str, Any]]) -> None:
        guides = self._extract_arm_guides(shapes)
        if guides is None:
            return
        tol = 0.01
        outer_bottom_y = float(guides["outer_bottom_y"])
        outer_right_x = float(guides["outer_right_x"])

        connector_shapes: set[int] = set()
        for index, shape in enumerate(shapes):
            if shape.get("type") != "line":
                continue
            x1, y1 = map(float, shape["p1"])
            x2, y2 = map(float, shape["p2"])
            if abs(x2 - x1) <= tol or abs(y2 - y1) <= tol:
                continue
            if max(x1, x2) >= outer_right_x - tol and min(y1, y2) >= outer_bottom_y - tol:
                connector_shapes.add(index)

        key_values: list[float] = []
        for index, shape in enumerate(shapes):
            if index in connector_shapes:
                continue
            shape_type = shape.get("type")
            if shape_type in {"line", "rect"}:
                key_values.extend([float(shape["p1"][1]), float(shape["p2"][1])])
            elif shape_type == "circle":
                key_values.append(float(shape["center"][1]))
            elif shape_type == "arc":
                key_values.append(float(shape["center"][1]))
            elif shape_type == "arc3":
                key_values.extend([float(shape["p1"][1]), float(shape["p2"][1]), float(shape["p3"][1])])

        if not key_values:
            return

        y_guides = self._cluster_sorted(key_values, tol=0.012)
        if len(y_guides) < 3:
            return

        source_center = y_guides[len(y_guides) // 2]
        mapping: dict[float, float] = {}
        for index, top_value in enumerate(y_guides[: len(y_guides) // 2]):
            bottom_value = y_guides[-(index + 1)]
            half_span = (bottom_value - top_value) / 2.0
            mapping[top_value] = 0.5 - half_span
            mapping[bottom_value] = 0.5 + half_span
        if len(y_guides) % 2 == 1:
            mapping[source_center] = 0.5

        def snap_y(value: float) -> float:
            nearest = min(y_guides, key=lambda guide: abs(guide - value))
            if abs(nearest - value) <= 0.02:
                return mapping.get(nearest, value)
            delta = value - source_center
            return 0.5 + delta

        for index, shape in enumerate(shapes):
            if index in connector_shapes:
                continue
            shape_type = shape.get("type")
            if shape_type in {"line", "rect"}:
                shape["p1"][1] = snap_y(float(shape["p1"][1]))
                shape["p2"][1] = snap_y(float(shape["p2"][1]))
            elif shape_type == "circle":
                shape["center"][1] = snap_y(float(shape["center"][1]))
            elif shape_type == "arc":
                shape["center"][1] = snap_y(float(shape["center"][1]))
            elif shape_type == "arc3":
                shape["p1"][1] = snap_y(float(shape["p1"][1]))
                shape["p2"][1] = snap_y(float(shape["p2"][1]))
                shape["p3"][1] = snap_y(float(shape["p3"][1]))

    def _canonicalize_left_arm_grid(self, shapes: list[dict[str, Any]]) -> None:
        guides = self._extract_arm_guides(shapes)
        if guides is None:
            return

        tol = 0.012
        outer_top_y = float(guides["outer_top_y"])
        outer_bottom_y = float(guides["outer_bottom_y"])
        outer_left_x = float(guides["outer_left_x"])
        outer_right_x = float(guides["outer_right_x"])
        old_stripes = [float(x) for x in sorted(guides["stripe_xs"])]

        row_height = (outer_bottom_y - outer_top_y) / 3.0
        inner_top_y = outer_top_y + row_height
        inner_bottom_y = outer_top_y + (2.0 * row_height)

        line_like = [shape for shape in shapes if shape.get("type") in {"line", "rect"}]
        if not line_like:
            return
        tip_outer_x = self._shapes_bounds(line_like)[0]
        old_tip_outer_x = tip_outer_x
        old_extra_x = 1.0 - outer_top_y

        inner_column_width = (outer_right_x - outer_left_x) / 6.0
        new_stripes = [outer_left_x + inner_column_width * index for index in range(1, 6)]
        new_extra_x = 1.0 - outer_top_y

        old_cuts = [old_tip_outer_x, outer_left_x, *old_stripes, outer_right_x, old_extra_x]
        new_cuts = [tip_outer_x, outer_left_x, *new_stripes, outer_right_x, new_extra_x]

        def snap_cut_x(value: float) -> float:
            nearest_index = min(range(len(old_cuts)), key=lambda index: abs(old_cuts[index] - value))
            if abs(old_cuts[nearest_index] - value) <= 0.02:
                return new_cuts[nearest_index]
            return value

        column_intervals = list(zip(new_cuts[:-2], new_cuts[1:-1]))
        old_column_intervals = list(zip(old_cuts[:-2], old_cuts[1:-1]))

        def snap_circle_x(value: float) -> float:
            nearest_index = min(
                range(len(old_column_intervals)),
                key=lambda index: abs(((old_column_intervals[index][0] + old_column_intervals[index][1]) / 2.0) - value),
            )
            left_x, right_x = column_intervals[nearest_index]
            return (left_x + right_x) / 2.0

        for shape in shapes:
            shape_type = shape.get("type")
            if shape_type in {"line", "rect"}:
                x1, y1 = map(float, shape["p1"])
                x2, y2 = map(float, shape["p2"])
                if abs(x2 - x1) <= tol:
                    x = snap_cut_x((x1 + x2) / 2.0)
                    shape["p1"][0] = x
                    shape["p2"][0] = x
                    if abs(y1 - outer_top_y) <= 0.03 and abs(y2 - outer_bottom_y) <= 0.03:
                        shape["p1"][1] = outer_top_y
                        shape["p2"][1] = outer_bottom_y
                    elif abs(y2 - outer_top_y) <= 0.03 and abs(y1 - outer_bottom_y) <= 0.03:
                        shape["p1"][1] = outer_bottom_y
                        shape["p2"][1] = outer_top_y
                    elif abs(min(y1, y2) - inner_top_y) <= 0.03 and abs(max(y1, y2) - inner_bottom_y) <= 0.03:
                        if y1 <= y2:
                            shape["p1"][1] = inner_top_y
                            shape["p2"][1] = inner_bottom_y
                        else:
                            shape["p1"][1] = inner_bottom_y
                            shape["p2"][1] = inner_top_y
                elif abs(y2 - y1) <= tol:
                    y = (y1 + y2) / 2.0
                    if abs(y - outer_top_y) <= 0.03:
                        y = outer_top_y
                    elif abs(y - inner_top_y) <= 0.03:
                        y = inner_top_y
                    elif abs(y - 0.5) <= 0.03:
                        y = 0.5
                    elif abs(y - inner_bottom_y) <= 0.03:
                        y = inner_bottom_y
                    elif abs(y - outer_bottom_y) <= 0.03:
                        y = outer_bottom_y
                    shape["p1"][1] = y
                    shape["p2"][1] = y
                    shape["p1"][0] = snap_cut_x(x1)
                    shape["p2"][0] = snap_cut_x(x2)
            elif shape_type == "circle":
                shape["center"][0] = snap_circle_x(float(shape["center"][0]))

    def _extend_left_arm_outer_tip(self, shapes: list[dict[str, Any]]) -> None:
        guides = self._extract_arm_guides(shapes)
        if guides is None:
            return
        if self.board_rect.width <= 0:
            return

        extension = self.ARM_END_EXTENSION_PX / float(self.board_rect.width)
        if extension <= 0:
            return

        tol = 0.01
        inner_tip_boundary_x = float(guides["outer_left_x"])
        line_like = [shape for shape in shapes if shape.get("type") in {"line", "rect"}]
        if not line_like:
            return
        tip_x = self._shapes_bounds(line_like)[0]

        for shape in shapes:
            shape_type = shape.get("type")
            if shape_type in {"line", "rect"}:
                for key in ("p1", "p2"):
                    if abs(float(shape[key][0]) - tip_x) <= tol:
                        shape[key][0] = tip_x - extension
            elif shape_type == "circle":
                center_x = float(shape["center"][0])
                radius = float(shape["radius"])
                if abs((center_x - radius) - tip_x) <= tol:
                    shape["center"][0] = inner_tip_boundary_x - radius

    def _build_corner_circle_basis(self, left_arm: list[dict[str, Any]]) -> dict[str, Any] | None:
        guides = self._extract_arm_guides(left_arm)
        if guides is None:
            return None
        lane_boundaries_from_center = sorted(guides["stripe_xs"], reverse=True)
        if len(lane_boundaries_from_center) < 3:
            return None
        center_y = lane_boundaries_from_center[2]
        radius = guides["outer_top_y"] - center_y
        if radius <= 0:
            return None
        right_tangent_x = 1.0 - guides["outer_bottom_y"]
        center_x = right_tangent_x - radius
        width = int(round(median([self._shape_width(shape) for shape in left_arm])))
        return {
            "type": "circle",
            "center": [center_x, center_y],
            "radius": radius,
            "width": width,
        }

    def _build_corner_circle_shapes(self, left_arm: list[dict[str, Any]]) -> list[dict[str, Any]]:
        top_left_circle = self._build_corner_circle_basis(left_arm)
        if top_left_circle is None:
            return []
        pivot = (0.5, 0.5)
        return [
            self._rotate_shape_about(top_left_circle, pivot, quarter_turns)
            for quarter_turns in range(4)
        ]

    def _build_corner_circle_arc_shapes(self, left_arm: list[dict[str, Any]]) -> list[dict[str, Any]]:
        arc_shapes: list[dict[str, Any]] = []
        for circle in self._build_corner_circle_shapes(left_arm):
            arc_shapes.extend(self._build_single_corner_circle_arc_shapes(circle))
        return arc_shapes

    def _build_single_corner_circle_arc_shapes(self, circle_shape: dict[str, Any]) -> list[dict[str, Any]]:
        cx, cy = map(float, circle_shape["center"])
        radius = float(circle_shape["radius"])
        width = int(circle_shape["width"])
        arc_radius = radius * (1.0 - self.CORNER_ARC_INSET_RATIO)

        endpoint_x = arc_radius * self.CORNER_ARC_ENDPOINT_X_RATIO
        endpoint_y = arc_radius * math.sqrt(max(0.0, 1.0 - self.CORNER_ARC_ENDPOINT_X_RATIO ** 2))
        mid_axis = arc_radius * self.CORNER_ARC_MID_AXIS_RATIO

        top_left = (cx - endpoint_x, cy - endpoint_y)
        top_right = (cx + endpoint_x, cy - endpoint_y)
        bottom_left = (cx - endpoint_x, cy + endpoint_y)
        bottom_right = (cx + endpoint_x, cy + endpoint_y)

        return [
            {"type": "arc3", "p1": list(top_left), "p2": [cx, cy - mid_axis], "p3": list(top_right), "width": width},
            {"type": "arc3", "p1": list(top_right), "p2": [cx + mid_axis, cy], "p3": list(bottom_right), "width": width},
            {"type": "arc3", "p1": list(bottom_left), "p2": [cx, cy + mid_axis], "p3": list(bottom_right), "width": width},
            {"type": "arc3", "p1": list(top_left), "p2": [cx - mid_axis, cy], "p3": list(bottom_left), "width": width},
        ]

    def _build_corner_circle_fill_specs(self, left_arm: list[dict[str, Any]]) -> list[dict[str, Any]]:
        fills: list[dict[str, Any]] = []
        for circle in self._build_corner_circle_shapes(left_arm):
            center = tuple(float(value) for value in circle["center"])
            radius = float(circle["radius"])
            cx, cy = center
            endpoint_x = radius * self.CORNER_ARC_ENDPOINT_X_RATIO
            endpoint_y = radius * math.sqrt(max(0.0, 1.0 - self.CORNER_ARC_ENDPOINT_X_RATIO ** 2))

            outer_top_left = (cx - endpoint_x, cy - endpoint_y)
            outer_top_right = (cx + endpoint_x, cy - endpoint_y)
            outer_bottom_left = (cx - endpoint_x, cy + endpoint_y)
            outer_bottom_right = (cx + endpoint_x, cy + endpoint_y)

            top_arc, right_arc, bottom_arc, left_arc = self._build_single_corner_circle_arc_shapes(circle)
            top_curve = self._arc3_screen_points(top_arc)
            right_curve = self._arc3_screen_points(right_arc)
            bottom_curve = self._arc3_screen_points(bottom_arc)
            left_curve = self._arc3_screen_points(left_arc)

            top_outer = self._sample_circle_arc_screen(center, radius, outer_top_left, outer_top_right)
            right_outer = self._sample_circle_arc_screen(center, radius, outer_top_right, outer_bottom_right)
            bottom_outer = self._sample_circle_arc_screen(center, radius, outer_bottom_right, outer_bottom_left)
            left_outer = self._sample_circle_arc_screen(center, radius, outer_bottom_left, outer_top_left)

            fills.extend(
                [
                    {"color": self.TRIANGLE_LEFT_FILL, "points": top_outer + list(reversed(top_curve))},
                    {"color": self.CIRCLE_TOP_FILL, "points": right_outer + list(reversed(right_curve))},
                    {"color": self.CIRCLE_RIGHT_FILL, "points": bottom_outer + bottom_curve},
                    {"color": self.CIRCLE_BOTTOM_FILL, "points": left_outer + left_curve},
                    {
                        "color": self.CIRCLE_CELL_FILL,
                        "points": top_curve + right_curve[1:] + list(reversed(bottom_curve[:-1])) + list(reversed(left_curve[1:-1])),
                    },
                ]
            )
        return fills

    def _line_length(self, p1: tuple[float, float], p2: tuple[float, float]) -> float:
        return math.hypot(p2[0] - p1[0], p2[1] - p1[1])

    def _project_t_onto_segment(
        self,
        point: tuple[float, float],
        start: tuple[float, float],
        end: tuple[float, float],
    ) -> float:
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        denom = dx * dx + dy * dy
        if denom <= 1e-9:
            return 0.0
        t = ((point[0] - start[0]) * dx + (point[1] - start[1]) * dy) / denom
        return max(0.0, min(1.0, t))

    def _point_line_distance(
        self,
        point: tuple[float, float],
        start: tuple[float, float],
        end: tuple[float, float],
    ) -> float:
        t = self._project_t_onto_segment(point, start, end)
        projection = (
            start[0] + (end[0] - start[0]) * t,
            start[1] + (end[1] - start[1]) * t,
        )
        return self._line_length(point, projection)

    def _canonicalize_connector_diagonals(self, shapes: list[dict[str, Any]]) -> None:
        tol = 0.01
        line_shapes = [shape for shape in shapes if shape.get("type") == "line"]
        horizontals: list[dict[str, Any]] = []
        verticals: list[dict[str, Any]] = []
        diagonals: list[dict[str, Any]] = []
        for shape in line_shapes:
            x1, y1 = map(float, shape["p1"])
            x2, y2 = map(float, shape["p2"])
            if abs(y2 - y1) <= tol:
                horizontals.append(shape)
            elif abs(x2 - x1) <= tol:
                verticals.append(shape)
            else:
                diagonals.append(shape)

        if not horizontals or not verticals or len(diagonals) < 2:
            return

        outer_top_y = min(float(shape["p1"][1]) for shape in horizontals)
        outer_bottom_y = max(float(shape["p1"][1]) for shape in horizontals)
        outer_verticals = [
            shape
            for shape in verticals
            if min(float(shape["p1"][1]), float(shape["p2"][1])) <= outer_top_y + tol
            and max(float(shape["p1"][1]), float(shape["p2"][1])) >= outer_bottom_y - tol
        ]
        if not outer_verticals:
            return
        outer_left_x = min(float(shape["p1"][0]) for shape in outer_verticals)

        connector_candidates = [
            shape
            for shape in diagonals
            if min(float(shape["p1"][1]), float(shape["p2"][1])) >= outer_bottom_y - tol
        ]
        if len(connector_candidates) != 2:
            return

        long_line = max(
            connector_candidates,
            key=lambda shape: self._line_length(tuple(shape["p1"]), tuple(shape["p2"])),
        )
        short_line = min(
            connector_candidates,
            key=lambda shape: self._line_length(tuple(shape["p1"]), tuple(shape["p2"])),
        )

        arm_corner = (outer_left_x, outer_bottom_y)
        raw_long_p1 = tuple(long_line["p1"])
        raw_long_p2 = tuple(long_line["p2"])
        raw_near = min((raw_long_p1, raw_long_p2), key=lambda point: self._line_length(point, arm_corner))
        raw_far = raw_long_p2 if raw_near == raw_long_p1 else raw_long_p1

        leg = (abs(raw_far[0] - raw_near[0]) + abs(raw_far[1] - raw_near[1])) / 2.0
        sign_x = -1.0 if raw_far[0] < raw_near[0] else 1.0
        sign_y = -1.0 if raw_far[1] < raw_near[1] else 1.0

        canon_near = arm_corner
        canon_far = (canon_near[0] + sign_x * leg, canon_near[1] + sign_y * leg)
        long_line["p1"] = [canon_near[0], canon_near[1]]
        long_line["p2"] = [canon_far[0], canon_far[1]]

        short_p1 = tuple(short_line["p1"])
        short_p2 = tuple(short_line["p2"])
        dist1 = self._point_line_distance(short_p1, raw_near, raw_far)
        dist2 = self._point_line_distance(short_p2, raw_near, raw_far)
        raw_junction, raw_outer = (short_p1, short_p2) if dist1 <= dist2 else (short_p2, short_p1)
        junction_t = self._project_t_onto_segment(raw_junction, raw_near, raw_far)
        canon_junction = (
            canon_near[0] + (canon_far[0] - canon_near[0]) * junction_t,
            canon_near[1] + (canon_far[1] - canon_near[1]) * junction_t,
        )

        perp_candidates = ((sign_y, -sign_x), (-sign_y, sign_x))
        target_perp_y = -1.0 if outer_bottom_y < canon_junction[1] else 1.0
        perp_x, _perp_y = next(
            (px, py) for px, py in perp_candidates if py == target_perp_y
        )
        delta = abs(canon_junction[1] - outer_bottom_y)
        if delta <= 1e-9:
            delta = (abs(raw_outer[0] - raw_junction[0]) + abs(raw_outer[1] - raw_junction[1])) / 2.0
        canon_outer = (canon_junction[0] + perp_x * delta, outer_bottom_y)
        short_line["p1"] = [canon_junction[0], canon_junction[1]]
        short_line["p2"] = [canon_outer[0], canon_outer[1]]

    def _canonicalize_left_arm_connectors(self, shapes: list[dict[str, Any]]) -> None:
        tol = 0.01
        line_shapes = [shape for shape in shapes if shape.get("type") == "line"]
        horizontals: list[dict[str, Any]] = []
        verticals: list[dict[str, Any]] = []
        diagonals: list[dict[str, Any]] = []
        for shape in line_shapes:
            x1, y1 = map(float, shape["p1"])
            x2, y2 = map(float, shape["p2"])
            if abs(y2 - y1) <= tol:
                horizontals.append(shape)
            elif abs(x2 - x1) <= tol:
                verticals.append(shape)
            else:
                diagonals.append(shape)

        if not horizontals or not verticals or len(diagonals) < 2:
            return

        outer_top_y = min(float(shape["p1"][1]) for shape in horizontals)
        outer_bottom_y = max(float(shape["p1"][1]) for shape in horizontals)
        outer_verticals = [
            shape
            for shape in verticals
            if min(float(shape["p1"][1]), float(shape["p2"][1])) <= outer_top_y + tol
            and max(float(shape["p1"][1]), float(shape["p2"][1])) >= outer_bottom_y - tol
        ]
        if not outer_verticals:
            return
        outer_right_x = max(float(shape["p1"][0]) for shape in outer_verticals)

        connector_candidates = [
            shape
            for shape in diagonals
            if max(float(shape["p1"][0]), float(shape["p2"][0])) >= outer_right_x - tol
            and min(float(shape["p1"][1]), float(shape["p2"][1])) >= outer_bottom_y - tol
        ]
        if len(connector_candidates) != 2:
            return

        long_line = max(
            connector_candidates,
            key=lambda shape: self._line_length(tuple(shape["p1"]), tuple(shape["p2"])),
        )
        short_line = min(
            connector_candidates,
            key=lambda shape: self._line_length(tuple(shape["p1"]), tuple(shape["p2"])),
        )

        old_near = (outer_right_x, outer_bottom_y)
        old_long_p1 = tuple(long_line["p1"])
        old_long_p2 = tuple(long_line["p2"])
        if self._line_length(old_long_p1, old_near) <= self._line_length(old_long_p2, old_near):
            raw_near, raw_far = old_long_p1, old_long_p2
        else:
            raw_near, raw_far = old_long_p2, old_long_p1

        exact_near = old_near
        exact_far = (outer_top_y, 1.0 - outer_right_x)
        long_line["p1"] = [exact_near[0], exact_near[1]]
        long_line["p2"] = [exact_far[0], exact_far[1]]

        exact_outer = (1.0 - outer_bottom_y, outer_bottom_y)
        junction_t = self._project_t_onto_segment(exact_outer, exact_near, exact_far)
        exact_junction = (
            exact_near[0] + (exact_far[0] - exact_near[0]) * junction_t,
            exact_near[1] + (exact_far[1] - exact_near[1]) * junction_t,
        )
        short_line["p1"] = [exact_junction[0], exact_junction[1]]
        short_line["p2"] = [exact_outer[0], exact_outer[1]]

    def tick(self, fps: int = 60) -> None:
        self.clock.tick(fps)

    def clear_drag(self) -> None:
        self.selected_start_id = None
        self.drag_start_id = None
        self.drag_piece = None
        self.drag_origin = None
        self.drag_moved = False

    def handle_event(
        self,
        event: pygame.event.Event,
        state: ParcheesiState,
        p2_turn: bool,
    ) -> tuple[str, str] | None:
        if event.type == pygame.MOUSEMOTION and self.drag_start_id is not None:
            self.drag_mouse = event.pos
            if self.drag_origin is not None:
                dx = event.pos[0] - self.drag_origin[0]
                dy = event.pos[1] - self.drag_origin[1]
                if (dx * dx + dy * dy) >= 25:
                    self.drag_moved = True
            return None

        if event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            if self.drag_start_id is None:
                return None
            left_arm = self._build_left_arm_basis()
            rect_map = self._location_rect_map(left_arm) if left_arm else {}
            move_map = self._p2_move_map(state)
            start_id = self.drag_start_id
            for dest_id in move_map.get(start_id, []):
                if self._interactive_rect_for_location(dest_id, rect_map, self._piece_radius()).collidepoint(event.pos):
                    self.clear_drag()
                    return (start_id, dest_id)
            if not self.drag_moved:
                self.selected_start_id = start_id
            self.drag_start_id = None
            self.drag_piece = None
            self.drag_origin = None
            self.drag_moved = False
            return None

        if event.type != pygame.MOUSEBUTTONDOWN or event.button != 1:
            return None

        if not p2_turn or state.current_player != 2:
            self.clear_drag()
            return None

        self.drag_mouse = event.pos
        left_arm = self._build_left_arm_basis()
        rect_map = self._location_rect_map(left_arm) if left_arm else {}
        move_map = self._p2_move_map(state)
        if self.selected_start_id not in move_map:
            self.selected_start_id = None
        if not move_map:
            return None

        radius = self._piece_radius()
        clicked_start = next(
            (
                start_id
                for start_id in move_map
                if self._interactive_rect_for_location(start_id, rect_map, radius).collidepoint(event.pos)
            ),
            None,
        )

        if self.selected_start_id is not None:
            for dest_id in move_map.get(self.selected_start_id, []):
                if self._interactive_rect_for_location(dest_id, rect_map, radius).collidepoint(event.pos):
                    start_id = self.selected_start_id
                    self.clear_drag()
                    return (start_id, dest_id)

        if clicked_start is not None:
            self.selected_start_id = None if clicked_start == self.selected_start_id else clicked_start
            self.drag_start_id = self.selected_start_id
            self.drag_origin = event.pos
            self.drag_moved = False
            self.drag_piece = next(
                (piece for piece in state.pieces_at_id(clicked_start) if piece.to_player_num() == 2),
                None,
            )
            return None

        self.clear_drag()
        return None

    def draw(
        self,
        state: ParcheesiState,
        p2_turn: bool = True,
        last_move: object | None = None,
    ) -> None:
        del last_move
        self.screen.fill(self.BG)
        pygame.draw.rect(self.screen, self.BOARD_BG, self.board_rect)
        left_arm = self._build_left_arm_basis()
        black_shapes = []
        circle_cell_shapes = []
        goal_lane_shapes = []
        triangle_fill_shapes = []
        special_circle_fills = []
        corner_circle_fill_specs = []
        location_label_specs: list[dict[str, Any]] = []
        rect_map: dict[str, pygame.Rect] = {}
        if left_arm:
            pivot = (0.5, 0.5)
            basis_circle_cells = self._build_circle_cell_shapes(left_arm)
            circle_cell_shapes = list(basis_circle_cells)
            basis_goal_lane = self._build_goal_lane_shapes(left_arm)
            triangle_fill_shapes = self._build_triangle_fill_shapes(left_arm)
            special_circle_fills = self._build_special_circle_fill_shapes(left_arm)
            corner_circle_fill_specs = self._build_corner_circle_fill_specs(left_arm)
            location_label_specs = self._build_location_label_specs(left_arm)
            rect_map = {str(spec["location_id"]): spec["rect"] for spec in location_label_specs}
            for quarter_turns in range(4):
                black_shapes.extend(
                    self._rotate_shape_about(shape, pivot, quarter_turns)
                    for shape in left_arm
                )
                goal_lane_shapes.extend(
                    self._rotate_shape_about(shape, pivot, quarter_turns)
                    for shape in basis_goal_lane
                )
            black_shapes.extend(self._build_corner_circle_shapes(left_arm))
            black_shapes.extend(self._build_corner_circle_arc_shapes(left_arm))
        pygame.draw.rect(self.screen, self.LINE, self.board_rect, width=2)
        gold_rect = self._build_arm_bounds_rect(black_shapes)
        home_rect = self._build_home_region_rect(left_arm) if left_arm else None
        if home_rect is not None:
            self._draw_home_region(home_rect)
        for shape in circle_cell_shapes:
            pygame.draw.rect(self.screen, self.CIRCLE_CELL_FILL, self._screen_rect_from_shape_rect(shape))
        for shape in goal_lane_shapes:
            pygame.draw.rect(self.screen, self.GOAL_LANE_FILL, self._screen_rect_from_shape_rect(shape))
        for fill in triangle_fill_shapes:
            pygame.draw.polygon(self.screen, fill["color"], [self._to_screen(point) for point in fill["points"]])
        for fill in special_circle_fills:
            pygame.draw.circle(
                self.screen,
                fill["color"],
                self._to_screen(fill["center"]),
                max(1, round(float(fill["radius"]) * self.board_rect.width)),
            )
        for fill in corner_circle_fill_specs:
            pygame.draw.polygon(self.screen, fill["color"], fill["points"])
        if gold_rect is not None:
            pygame.draw.rect(self.screen, self.GOLD, gold_rect, width=3)
        self._draw_move_hints(state, rect_map, p2_turn)
        for shape in black_shapes:
            self._draw_shape(shape, self.LINE)
        self._draw_parcheesi_pieces(state, rect_map)
        self._draw_location_labels(location_label_specs)
        self._draw_dice_panel(state)
