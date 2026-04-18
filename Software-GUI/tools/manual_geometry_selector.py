#!/usr/bin/env python3
"""Manual green/yellow grid selector used as a separate GUI window."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import pygame


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select outer and inner board geometry from an image.")
    parser.add_argument("--image", required=True, help="Image to annotate")
    parser.add_argument("--source-path", default="", help="Original source path to show in the selector")
    return parser.parse_args()


class SelectorState:
    def __init__(self, image: pygame.Surface, source_path: str) -> None:
        self.image = image
        self.source_path = source_path
        self.outer: list[tuple[float, float]] = []
        self.inner: list[tuple[float, float]] = []
        self.image_rect = pygame.Rect(0, 0, 0, 0)
        self.save_rect = pygame.Rect(0, 0, 0, 0)
        self.undo_rect = pygame.Rect(0, 0, 0, 0)
        self.reset_rect = pygame.Rect(0, 0, 0, 0)
        self.cancel_rect = pygame.Rect(0, 0, 0, 0)

    def complete(self) -> bool:
        return len(self.outer) == 4 and len(self.inner) == 4

    def instruction(self) -> str:
        order = ["top-left", "top-right", "bottom-right", "bottom-left"]
        if len(self.outer) < 4:
            return f"Green outer grid: click {order[len(self.outer)]} corner."
        if len(self.inner) < 4:
            return f"Yellow inner grid: click {order[len(self.inner)]} corner."
        return "All corners selected. Click Save or press Enter."

    def add_click(self, pos: tuple[int, int]) -> None:
        if not self.image_rect.collidepoint(*pos):
            return
        iw, ih = self.image.get_size()
        x = (pos[0] - self.image_rect.x) * iw / max(self.image_rect.width, 1)
        y = (pos[1] - self.image_rect.y) * ih / max(self.image_rect.height, 1)
        x = max(0.0, min(float(iw - 1), float(x)))
        y = max(0.0, min(float(ih - 1), float(y)))
        if len(self.outer) < 4:
            self.outer.append((x, y))
        elif len(self.inner) < 4:
            self.inner.append((x, y))

    def undo(self) -> None:
        if self.inner:
            self.inner.pop()
        elif self.outer:
            self.outer.pop()

    def reset(self) -> None:
        self.outer.clear()
        self.inner.clear()

    def result(self, accepted: bool) -> dict[str, object]:
        payload: dict[str, object] = {"accepted": bool(accepted)}
        if accepted:
            payload["outer_corners_px"] = [[x, y] for x, y in self.outer]
            payload["chessboard_corners_px"] = [[x, y] for x, y in self.inner]
        return payload


def _draw_button(
    screen: pygame.Surface,
    font: pygame.font.Font,
    rect: pygame.Rect,
    label: str,
    *,
    enabled: bool,
    bg: tuple[int, int, int],
    border: tuple[int, int, int],
) -> None:
    mx, my = pygame.mouse.get_pos()
    hover = rect.collidepoint(mx, my)
    if enabled and hover:
        fill = tuple(min(255, c + 24) for c in bg)
    elif enabled:
        fill = bg
    else:
        fill = (48, 50, 58)
        border = (85, 90, 105)
    pygame.draw.rect(screen, fill, rect, border_radius=9)
    pygame.draw.rect(screen, border, rect, width=1, border_radius=9)
    text = font.render(label, True, (248, 248, 250) if enabled else (150, 154, 164))
    screen.blit(text, text.get_rect(center=rect.center))


def _draw_points(
    screen: pygame.Surface,
    font: pygame.font.Font,
    state: SelectorState,
    points: list[tuple[float, float]],
    color: tuple[int, int, int],
    prefix: str,
) -> None:
    iw, ih = state.image.get_size()

    def to_screen(point: tuple[float, float]) -> tuple[int, int]:
        return (
            int(state.image_rect.x + point[0] * state.image_rect.width / max(iw, 1)),
            int(state.image_rect.y + point[1] * state.image_rect.height / max(ih, 1)),
        )

    screen_points = [to_screen(p) for p in points]
    if len(screen_points) >= 2:
        pygame.draw.lines(screen, color, len(screen_points) == 4, screen_points, width=3)
    for idx, point in enumerate(screen_points, start=1):
        pygame.draw.circle(screen, color, point, 7)
        pygame.draw.circle(screen, (8, 8, 8), point, 7, width=1)
        label = font.render(f"{prefix}{idx}", True, color)
        screen.blit(label, (point[0] + 8, point[1] - 8))


def _draw(screen: pygame.Surface, font: pygame.font.Font, title_font: pygame.font.Font, state: SelectorState) -> None:
    screen.fill((20, 23, 30))
    width, height = screen.get_size()

    title = title_font.render("Manual Board Geometry", True, (248, 248, 250))
    screen.blit(title, (22, 16))

    lines = [
        "Click 4 green outer corners, then 4 yellow inner corners.",
        "Order for each grid: top-left, top-right, bottom-right, bottom-left.",
        state.instruction(),
        f"Green outer: {len(state.outer)}/4    Yellow inner: {len(state.inner)}/4",
    ]
    y = 52
    for line in lines:
        color = (242, 220, 120) if line == state.instruction() else (218, 224, 232)
        text = font.render(line, True, color)
        screen.blit(text, (22, y))
        y += 22
    if state.source_path:
        source = font.render(state.source_path, True, (145, 156, 170))
        screen.blit(source, (22, y))

    buttons_y = height - 52
    state.save_rect = pygame.Rect(width - 394, buttons_y, 92, 36)
    state.undo_rect = pygame.Rect(width - 294, buttons_y, 84, 36)
    state.reset_rect = pygame.Rect(width - 202, buttons_y, 84, 36)
    state.cancel_rect = pygame.Rect(width - 110, buttons_y, 88, 36)

    image_top = 154
    image_bottom = buttons_y - 14
    max_w = width - 44
    max_h = max(80, image_bottom - image_top)
    iw, ih = state.image.get_size()
    scale = min(max_w / max(iw, 1), max_h / max(ih, 1), 1.0)
    scaled_size = (max(1, int(iw * scale)), max(1, int(ih * scale)))
    shown = pygame.transform.smoothscale(state.image, scaled_size)
    state.image_rect = shown.get_rect(center=(width // 2, image_top + max_h // 2))
    screen.blit(shown, state.image_rect)
    pygame.draw.rect(screen, (90, 100, 116), state.image_rect, width=1)

    _draw_points(screen, font, state, state.outer, (40, 230, 95), "G")
    _draw_points(screen, font, state, state.inner, (245, 215, 45), "Y")

    _draw_button(
        screen,
        font,
        state.save_rect,
        "Save",
        enabled=state.complete(),
        bg=(35, 105, 62),
        border=(120, 220, 150),
    )
    _draw_button(
        screen,
        font,
        state.undo_rect,
        "Undo",
        enabled=bool(state.outer or state.inner),
        bg=(70, 72, 86),
        border=(145, 150, 170),
    )
    _draw_button(
        screen,
        font,
        state.reset_rect,
        "Reset",
        enabled=bool(state.outer or state.inner),
        bg=(70, 72, 86),
        border=(145, 150, 170),
    )
    _draw_button(
        screen,
        font,
        state.cancel_rect,
        "Cancel",
        enabled=True,
        bg=(92, 54, 54),
        border=(215, 130, 130),
    )
    pygame.display.flip()


def main() -> int:
    args = _parse_args()
    image_path = Path(args.image)
    if not image_path.exists():
        print(json.dumps({"accepted": False, "error": f"missing image: {image_path}"}))
        return 2

    pygame.init()
    pygame.display.set_caption("FlexyBoard Manual Grid Selector")
    info = pygame.display.Info()
    window_w = min(max(960, info.current_w - 180), 1400)
    window_h = min(max(720, info.current_h - 180), 950)
    screen = pygame.display.set_mode((window_w, window_h))
    font = pygame.font.SysFont("sf pro display, helvetica, arial", 20)
    title_font = pygame.font.SysFont("sf pro display, helvetica, arial", 28, bold=True)
    image = pygame.image.load(str(image_path)).convert()
    state = SelectorState(image, args.source_path)
    clock = pygame.time.Clock()

    while True:
        _draw(screen, font, title_font, state)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                print(json.dumps(state.result(False), separators=(",", ":")))
                return 0
            if event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_RETURN, pygame.K_KP_ENTER) and state.complete():
                    print(json.dumps(state.result(True), separators=(",", ":")))
                    return 0
                if event.key == pygame.K_ESCAPE:
                    print(json.dumps(state.result(False), separators=(",", ":")))
                    return 0
                if event.key in (pygame.K_BACKSPACE, pygame.K_DELETE, pygame.K_u):
                    state.undo()
                if event.key == pygame.K_r:
                    state.reset()
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if state.save_rect.collidepoint(*event.pos):
                    if state.complete():
                        print(json.dumps(state.result(True), separators=(",", ":")))
                        return 0
                    continue
                if state.undo_rect.collidepoint(*event.pos):
                    state.undo()
                    continue
                if state.reset_rect.collidepoint(*event.pos):
                    state.reset()
                    continue
                if state.cancel_rect.collidepoint(*event.pos):
                    print(json.dumps(state.result(False), separators=(",", ":")))
                    return 0
                state.add_click(event.pos)
        clock.tick(60)


if __name__ == "__main__":
    raise SystemExit(main())
