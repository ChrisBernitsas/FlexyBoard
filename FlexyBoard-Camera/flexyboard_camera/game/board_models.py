from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class BoardCoord:
    x: int
    y: int

    def in_bounds(self, width: int, height: int) -> bool:
        return 0 <= self.x < width and 0 <= self.y < height

    def to_algebraic(self) -> str:
        return f"{chr(ord('a') + self.x)}{self.y + 1}"


@dataclass(frozen=True, slots=True)
class BoardSpec:
    game: str
    width: int
    height: int
    square_size_mm: float
    origin_offset_mm_x: float = 0.0
    origin_offset_mm_y: float = 0.0

    def center_mm(self, coord: BoardCoord) -> tuple[float, float]:
        x_mm = self.origin_offset_mm_x + (coord.x + 0.5) * self.square_size_mm
        y_mm = self.origin_offset_mm_y + (coord.y + 0.5) * self.square_size_mm
        return x_mm, y_mm

    def validate(self, coord: BoardCoord) -> None:
        if not coord.in_bounds(self.width, self.height):
            raise ValueError(f"Coordinate {coord} out of bounds for {self.width}x{self.height} board")
