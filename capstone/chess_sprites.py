"""Load chess piece images from a folder (flexible filenames)."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import pygame

from chess_state import ChessPiece

_ROOT = Path(__file__).resolve().parent

_LETTER_FOR_PIECE: Dict[ChessPiece, str] = {
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
}

_NAME_FOR_LETTER = {
    "P": "pawn",
    "N": "knight",
    "B": "bishop",
    "R": "rook",
    "Q": "queen",
    "K": "king",
}


def chess_piece_dirs() -> Tuple[Path, ...]:
    """Search order: env override, ./chessPieces, ./assets/chess."""
    env = os.environ.get("P2_CHESS_PIECES_DIR", "").strip()
    dirs: list[Path] = []
    if env:
        dirs.append(Path(env).expanduser())
    dirs.append(_ROOT / "chessPieces")
    dirs.append(_ROOT / "assets" / "chess")
    return tuple(dirs)


def _color_prefix(piece: ChessPiece) -> str:
    return "w" if piece <= ChessPiece.P1_KING else "b"


def _filename_stems(color: str, letter: str) -> Iterable[str]:
    side = "white" if color == "w" else "black"
    side_cap = "White" if color == "w" else "Black"
    name = _NAME_FOR_LETTER[letter]
    yield f"{color}{letter}"
    yield f"{color.upper()}{letter.upper()}"
    yield f"{color}{letter.lower()}"
    yield f"{side}_{name}"
    yield f"{side}-{name}"
    yield f"{side_cap}-{name}"
    yield f"{name}_{side}"
    yield f"{name}-{side}"
    yield f"{side_cap}_{name}"
    yield f"{letter}_{side}"
    yield f"{letter}-{side}"
    yield f"{letter}{color}"
    yield f"{letter}{color.upper()}"


def _find_image_file(folder: Path, color: str, letter: str) -> Optional[Path]:
    for stem in _filename_stems(color, letter):
        for ext in (".png", ".webp", ".jpg", ".jpeg", ".gif"):
            path = folder / f"{stem}{ext}"
            if path.is_file():
                return path
    return None


def load_raw_surfaces() -> Tuple[Dict[ChessPiece, pygame.Surface], Path]:
    """
    Load one surface per piece type (P1=white, P2=black).
    Returns (mapping, folder_used). Mapping may be partial if files missing.
    """
    best_count = 0
    best_raw: Dict[ChessPiece, pygame.Surface] = {}
    best_folder = _ROOT / "chessPieces"
    for folder in chess_piece_dirs():
        if not folder.is_dir():
            continue
        trial: Dict[ChessPiece, pygame.Surface] = {}
        for piece in _LETTER_FOR_PIECE:
            letter = _LETTER_FOR_PIECE[piece]
            color = _color_prefix(piece)
            path = _find_image_file(folder, color, letter)
            if path is None:
                continue
            try:
                img = pygame.image.load(str(path)).convert_alpha()
            except pygame.error:
                continue
            trial[piece] = img
        if len(trial) > best_count:
            best_count = len(trial)
            best_raw = trial
            best_folder = folder
    if best_count == 0:
        print(
            "Chess sprites: no images found; using letters. "
            f"Add PNGs under {_ROOT / 'chessPieces'} (or set P2_CHESS_PIECES_DIR).",
            file=sys.stderr,
        )
    else:
        print(
            f"Chess sprites: loaded {best_count}/12 from {best_folder}",
            file=sys.stderr,
        )
    return best_raw, best_folder


def scale_for_cell(surface: pygame.Surface, cell_px: int) -> pygame.Surface:
    max_side = max(1, int(cell_px * 0.88))
    w, h = surface.get_size()
    scale = min(max_side / w, max_side / h, 1.0)
    nw = max(1, int(w * scale))
    nh = max(1, int(h * scale))
    return pygame.transform.smoothscale(surface, (nw, nh))


class ChessSpriteSet:
    """Scaled sprites for board + smaller ones for the captured tray."""

    def __init__(
        self,
        board: Dict[ChessPiece, pygame.Surface],
        small: Dict[ChessPiece, pygame.Surface],
        source_dir: Path,
    ) -> None:
        self.board = board
        self.small = small
        self.source_dir = source_dir

    @classmethod
    def load(cls, cell_px: int, tray_px: int = 20) -> "ChessSpriteSet":
        raw, folder = load_raw_surfaces()
        board: Dict[ChessPiece, pygame.Surface] = {}
        small: Dict[ChessPiece, pygame.Surface] = {}
        for piece, surf in raw.items():
            board[piece] = scale_for_cell(surf, cell_px)
            small[piece] = scale_for_cell(surf, tray_px)
        return cls(board, small, folder)

    def has(self, piece: ChessPiece) -> bool:
        return piece in self.board
