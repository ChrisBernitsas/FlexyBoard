"""Algebraic square IDs (a1–h8) with rank 1 at the bottom row."""

from __future__ import annotations

from dataclasses import dataclass

FILES = "abcdefgh"
RANKS = "12345678"


@dataclass(frozen=True)
class Square:
    """file_index 0=a, rank_index 0=rank 1 (bottom row on screen)."""

    file_index: int
    rank_index: int

    def __post_init__(self) -> None:
        if not 0 <= self.file_index <= 7:
            raise ValueError("file_index must be 0..7")
        if not 0 <= self.rank_index <= 7:
            raise ValueError("rank_index must be 0..7")

    def to_id(self) -> str:
        return format_square(self.file_index, self.rank_index)


def parse_square(square_id: str) -> Square:
    s = square_id.strip().lower()
    if len(s) != 2:
        raise ValueError(f"expected two-character square id, got {square_id!r}")
    f, r = s[0], s[1]
    if f not in FILES or r not in RANKS:
        raise ValueError(f"invalid square id: {square_id!r}")
    return Square(FILES.index(f), RANKS.index(r))


def format_square(file_index: int, rank_index: int) -> str:
    if not 0 <= file_index <= 7 or not 0 <= rank_index <= 7:
        raise ValueError("indices must be 0..7")
    return FILES[file_index] + RANKS[rank_index]


def is_dark_square(file_index: int, rank_index: int) -> bool:
    """a1 is dark (US-style checker coloring)."""
    return (file_index + rank_index) % 2 == 0
