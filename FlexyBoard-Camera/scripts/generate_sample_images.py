from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def make_board(square_px: int = 100) -> np.ndarray:
    board = np.zeros((8 * square_px, 8 * square_px, 3), dtype=np.uint8)
    for y in range(8):
        for x in range(8):
            c = 200 if (x + y) % 2 == 0 else 110
            board[y * square_px : (y + 1) * square_px, x * square_px : (x + 1) * square_px] = (c, c, c)
    return board


def draw_piece(img: np.ndarray, coord: tuple[int, int], square_px: int = 100) -> None:
    x, y = coord
    cx = x * square_px + square_px // 2
    cy = y * square_px + square_px // 2
    cv2.circle(img, (cx, cy), square_px // 3, (35, 35, 35), -1)


def main() -> int:
    out_dir = Path("sample_data")
    out_dir.mkdir(parents=True, exist_ok=True)

    before = make_board()
    after = make_board()
    draw_piece(before, (1, 1))
    draw_piece(after, (1, 3))

    cv2.imwrite(str(out_dir / "before_demo.png"), before)
    cv2.imwrite(str(out_dir / "after_demo.png"), after)
    print("Wrote sample_data/before_demo.png and sample_data/after_demo.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
