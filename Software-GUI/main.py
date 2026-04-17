#!/usr/bin/env python3
"""Player 2 UI: choose game, then play as P2 via drag-drop."""

from __future__ import annotations

import json
import sys
import threading

import pygame

import config
from ai_player import choose_p2_move, close_engine
from board_ui import BoardUI
from checkers_state import CheckersState
from chess_state import ChessState
from chess_ui import ChessUI
from parcheesi_state import ParcheesiState
from ipc.mock_transport import MockTransport
from ipc.protocol import P1MoveMessage, P2MoveMessage, decode_line
from ipc.transport_tcp import TcpClientTransport
from motor_sequence import CaptureInventory, generate_p2_sequence, write_sequence_file


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


def _mode_button_rect(ui: object) -> pygame.Rect:
    screen = ui.screen
    w = 180
    h = 36
    margin = 12
    return pygame.Rect(screen.get_width() - w - margin, margin, w, h)


def _draw_mode_button(ui: object, ai_mode: bool, game_kind: str) -> None:
    rect = _mode_button_rect(ui)
    mx, my = pygame.mouse.get_pos()
    hover = rect.collidepoint(mx, my)

    if game_kind == "chess":
        ai_label = "AI (Stockfish)"
    else:
        ai_label = "AI (Heuristic)"

    if ai_mode:
        bg = (35, 92, 55) if not hover else (45, 112, 65)
        border = (110, 200, 140)
        text = f"Mode: {ai_label}"
    else:
        bg = (72, 72, 86) if not hover else (84, 84, 98)
        border = (140, 140, 158)
        text = "Mode: Manual"

    pygame.draw.rect(ui.screen, bg, rect, border_radius=10)
    pygame.draw.rect(ui.screen, border, rect, width=1, border_radius=10)

    label = ui.font.render(text, True, (244, 244, 248))
    ui.screen.blit(label, label.get_rect(center=rect.center))
    pygame.display.update(rect)


def _draw_capture_inventory_overlay(ui: object, inventory_lines: list[str], manual_actions: list[str]) -> None:
    if not config.P2_SHOW_CAPTURE_INVENTORY_OVERLAY:
        return

    x = 12
    y = 56
    w = 380
    max_lines = max(1, config.P2_CAPTURE_INVENTORY_OVERLAY_MAX_LINES)
    shown_inventory = inventory_lines[:max_lines]
    if len(inventory_lines) > max_lines:
        shown_inventory.append(f"... ({len(inventory_lines) - max_lines} more)")

    lines = [f"Capture Inventory ({len(inventory_lines)})", *shown_inventory]
    if manual_actions:
        lines.append("MANUAL ACTION REQUIRED:")
        lines.extend(manual_actions[:3])

    line_h = 18
    h = 10 + line_h * len(lines) + 8
    panel = pygame.Surface((w, h), pygame.SRCALPHA)
    panel.fill((15, 18, 24, 205))
    ui.screen.blit(panel, (x, y))
    pygame.draw.rect(ui.screen, (120, 150, 180), pygame.Rect(x, y, w, h), width=1)

    for i, text in enumerate(lines):
        color = (230, 235, 245)
        if i == 0:
            color = (255, 230, 130)
        if text.startswith("MANUAL ACTION REQUIRED"):
            color = (255, 120, 120)
        surf = ui.font.render(text, True, color)
        ui.screen.blit(surf, (x + 8, y + 6 + i * line_h))


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
    ai_mode = config.CONTROL_MODE == "ai"
    ai_pending = False
    last_move: tuple[str, str] | None = None
    capture_inventory = CaptureInventory(kind)
    latest_manual_actions: list[str] = []

    if kind == "checkers":
        state: object = CheckersState()
        ui: object = BoardUI(config.WINDOW_WIDTH, config.WINDOW_HEIGHT)
    elif kind == "chess":
        state = ChessState()
        ui = ChessUI(config.WINDOW_WIDTH, config.WINDOW_HEIGHT)
    elif kind == "parcheesi":
        state = ParcheesiState()
        ui = BoardUI(config.WINDOW_WIDTH, config.WINDOW_HEIGHT)
        pygame.display.set_caption("Player 2 — Parcheesi")
    else:
        raise ValueError(f"Unsupported game kind: {kind}")

    def _apply_and_dispatch_p2_move(start_id: str, end_id: str) -> None:
        nonlocal waiting_for_p1, ai_pending, last_move, latest_manual_actions

        state_before = state.copy()
        err = state.try_apply_p2_move(start_id, end_id)
        if err is not None:
            print("Illegal P2 move:", err, file=sys.stderr)
            return

        last_move = (start_id, end_id)
        print(f"P2 move: {start_id} -> {end_id}", file=sys.stderr)

        generated = generate_p2_sequence(
            kind,
            state_before,
            start_id,
            end_id,
            capture_inventory=capture_inventory,
        )
        latest_manual_actions = generated.manual_actions[:]
        msg = P2MoveMessage(
            frm=start_id,
            to=end_id,
            game=kind,
            stm_sequence=generated.lines,
            manual_actions=(generated.manual_actions[:] if generated.manual_actions else None),
        )
        transport.send_p2_move(msg)
        if config.TRANSPORT == "mock":
            print(json.dumps(msg.to_obj(), separators=(",", ":")), flush=True, file=sys.stderr)

        if config.WRITE_STM_SEQUENCE:
            out_path = write_sequence_file(
                config.STM_SEQUENCE_FILE,
                generated.lines,
                game=kind,
                start_id=start_id,
                end_id=end_id,
                capture_slots_used=generated.capture_slots_used,
                capture_inventory_summary=capture_inventory.to_summary_strings(),
                manual_actions=generated.manual_actions,
            )
            print(
                "STM sequence updated:"
                f" {out_path} (capture={generated.capture_detected},"
                f" temp_relocations={generated.temporary_relocations},"
                f" fallback_direct={generated.fallback_direct_segments},"
                f" steps={len(generated.lines)},"
                f" capture_slots={len(generated.capture_slots_used)})",
                file=sys.stderr,
            )
            if generated.manual_actions:
                for act in generated.manual_actions:
                    print(f"[MANUAL] {act}", file=sys.stderr)
            print(
                "[INVENTORY] " + ", ".join(capture_inventory.to_summary_strings()) if capture_inventory.to_summary_strings() else "[INVENTORY] <empty>",
                file=sys.stderr,
            )

        if kind == "chess" and hasattr(state, "is_checkmate"):
            if state.is_checkmate():
                print(f"Game over: checkmate. Winner: {'P2' if state.turn_side() == 'p1' else 'P1'}", file=sys.stderr)
            elif state.is_stalemate():
                print("Game over: stalemate.", file=sys.stderr)
            elif state.is_check():
                print(f"Check on {state.turn_side().upper()}.", file=sys.stderr)

        if kind == "checkers" and hasattr(state, "p2_must_continue_jump") and state.p2_must_continue_jump():
            forced = state.p2_forced_square_id() if hasattr(state, "p2_forced_square_id") else None
            if forced:
                print(f"Checkers: must continue jump with {forced}", file=sys.stderr)
            waiting_for_p1 = False
            ai_pending = ai_mode
        else:
            waiting_for_p1 = True

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                raise SystemExit(0)

            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if _mode_button_rect(ui).collidepoint(*event.pos):
                    ai_mode = not ai_mode
                    if ai_mode and not waiting_for_p1:
                        ai_pending = True
                    mode_name = "AI (Stockfish)" if ai_mode else "Manual"
                    print(f"Control mode set to: {mode_name}", file=sys.stderr)
                    continue

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
                capture_inventory.clear()
                latest_manual_actions = []
                ui.clear_drag()
                waiting_for_p1 = True
                ai_pending = False
                last_move = None
                print("Board reset to starting position.", file=sys.stderr)

            if not waiting_for_p1:
                drop = ui.handle_event(event, state, p2_turn=True)
                if drop:
                    start_id, end_id = drop
                    _apply_and_dispatch_p2_move(start_id, end_id)

        if waiting_for_p1:
            msg = transport.poll_p1_move()
            if msg is not None:
                state_before = state.copy()
                state.apply_move_trusted(msg.frm, msg.to)
                if kind == "chess":
                    new_captured = state.captured_by_p1[len(state_before.captured_by_p1):]
                    for piece in new_captured:
                        capture_inventory.add_captured_piece("p2", piece.name)
                elif kind == "checkers":
                    new_captured = state.captured_by_p1[len(state_before.captured_by_p1):]
                    for piece in new_captured:
                        capture_inventory.add_captured_piece("p2", piece.name)
                elif kind == "parcheesi":
                    new_captured = state.captured_by_p1[len(state_before.captured_by_p1):]
                    for piece in new_captured:
                        capture_inventory.add_captured_piece("p2", piece.name)
                latest_manual_actions = []
                last_move = (msg.frm, msg.to)
                print(f"P1 move: {msg.frm} -> {msg.to}", file=sys.stderr)
                print(
                    "[INVENTORY] " + ", ".join(capture_inventory.to_summary_strings()) if capture_inventory.to_summary_strings() else "[INVENTORY] <empty>",
                    file=sys.stderr,
                )
                if kind == "chess" and hasattr(state, "is_checkmate"):
                    if state.is_checkmate():
                        print(
                            f"Game over: checkmate. Winner: {'P2' if state.turn_side() == 'p1' else 'P1'}",
                            file=sys.stderr,
                        )
                    elif state.is_stalemate():
                        print("Game over: stalemate.", file=sys.stderr)
                    elif state.is_check():
                        print(f"Check on {state.turn_side().upper()}.", file=sys.stderr)
                if kind == "checkers" and hasattr(state, "p1_must_continue_jump") and state.p1_must_continue_jump():
                    forced = state.p1_forced_square_id() if hasattr(state, "p1_forced_square_id") else None
                    if forced:
                        print(f"Checkers: P1 must continue jump with {forced}", file=sys.stderr)
                    waiting_for_p1 = True
                else:
                    waiting_for_p1 = False
                    ai_pending = ai_mode

        if (not waiting_for_p1) and ai_mode and ai_pending:
            chosen = choose_p2_move(kind, state)
            ai_pending = False
            if chosen is None:
                print("AI: no legal P2 move available", file=sys.stderr)
                waiting_for_p1 = True
            else:
                _apply_and_dispatch_p2_move(chosen.start_id, chosen.end_id)

        ui.draw(state, p2_turn=not waiting_for_p1, last_move=last_move)
        _draw_capture_inventory_overlay(ui, capture_inventory.to_summary_strings(), latest_manual_actions)
        _draw_mode_button(ui, ai_mode, kind)
        pygame.display.flip()
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
        _Button(pygame.Rect(cx - bw // 2, top + (2 * (bh + gap)), bw, bh), "Parcheesi"),
    ]

    try:
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    for b in buttons:
                        if b.rect.collidepoint(*event.pos):
                            label = b.label.lower()
                            if label.startswith("check"):
                                kind = "checkers"
                            elif label.startswith("chess"):
                                kind = "chess"
                            else:
                                kind = "parcheesi"
                            _run_game_loop(kind, transport)

            _draw_menu(screen, font, buttons)
            clock.tick(60)
    finally:
        close_engine()
        if hasattr(transport, "close"):
            transport.close()


if __name__ == "__main__":
    main()
