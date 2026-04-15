#!/usr/bin/env python3
"""Player 2 UI: choose game, then play as P2 via drag-drop."""

from __future__ import annotations

import json
import sys
import threading

import pygame

import config
from board_ui import BoardUI
from checkers_state import CheckersState
from chess_state import ChessState
from chess_ui import ChessUI
from ipc.mock_transport import MockTransport
from ipc.protocol import P1MoveMessage, P2MoveMessage, decode_line
from ipc.transport_tcp import TcpClientTransport


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
            mock.inject_p1_move(decoded.frm, decoded.to)
            print("stdin: queued P1 move", decoded.frm, "->", decoded.to, file=sys.stderr)


class _Button:
    def __init__(self, rect: pygame.Rect, label: str) -> None:
        self.rect = rect
        self.label = label


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
        transport.connect()
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
    waiting_for_p1 = True

    if kind == "checkers":
        state: object = CheckersState()
        ui: object = BoardUI(config.WINDOW_WIDTH, config.WINDOW_HEIGHT)
    else:
        state = ChessState()
        ui = ChessUI(config.WINDOW_WIDTH, config.WINDOW_HEIGHT)

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                raise SystemExit(0)

            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                if hasattr(ui, "close"):
                    ui.close()
                return

            if event.type == pygame.KEYDOWN and event.key == pygame.K_r and not getattr(
                event, "repeat", False
            ):
                if hasattr(transport, "drain_p1_moves"):
                    transport.drain_p1_moves()
                state.reset()
                ui.clear_drag()
                waiting_for_p1 = True
                print("Board reset to starting position.", file=sys.stderr)

            if not waiting_for_p1:
                drop = ui.handle_event(event, state, p2_turn=True)
                if drop:
                    start_id, end_id = drop
                    err = state.try_apply_p2_move(start_id, end_id)
                    if err is None:
                        msg = P2MoveMessage(frm=start_id, to=end_id)
                        transport.send_p2_move(msg)
                        if config.TRANSPORT == "mock":
                            print(
                                json.dumps(msg.to_obj(), separators=(",", ":")),
                                flush=True,
                                file=sys.stderr,
                            )
                        waiting_for_p1 = True
                    else:
                        print("Illegal P2 move:", err, file=sys.stderr)

        if waiting_for_p1:
            msg = transport.poll_p1_move()
            if msg is not None:
                state.apply_move_trusted(msg.frm, msg.to)
                waiting_for_p1 = False

        ui.draw(state, p2_turn=not waiting_for_p1)
        ui.tick(60)


def main() -> None:
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

    bw = 280
    bh = 70
    gap = 18
    cx = config.WINDOW_WIDTH // 2
    top = 220
    buttons = [
        _Button(pygame.Rect(cx - bw // 2, top, bw, bh), "Checkers"),
        _Button(pygame.Rect(cx - bw // 2, top + bh + gap, bw, bh), "Chess"),
    ]

    try:
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    for b in buttons:
                        if b.rect.collidepoint(*event.pos):
                            kind = "checkers" if b.label.lower().startswith("check") else "chess"
                            _run_game_loop(kind, transport)

            _draw_menu(screen, font, buttons)
            clock.tick(60)
    finally:
        if hasattr(transport, "close"):
            transport.close()


if __name__ == "__main__":
    main()
