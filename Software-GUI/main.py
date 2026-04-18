#!/usr/bin/env python3
"""Player 2 UI: choose game, then play as P2 via drag-drop."""

from __future__ import annotations

import base64
import io
import json
from pathlib import Path
import subprocess
import sys
import threading

import pygame

import config
from ai_player import choose_p2_move, close_engine
from board_ui import BoardUI # Keep for other games if needed
from checkers_state import CheckersState
from chess_state import ChessState
from chess_ui import ChessUI # Keep for other games if needed
from parcheesi_state import ParcheesiState
from parcheesi_ui import ParcheesiUI
from ipc.mock_transport import MockTransport
from ipc.protocol import P1MoveMessage, P2MoveMessage, decode_line
from ipc.transport_tcp import TcpClientTransport
from motor_sequence import CaptureInventory, generate_p2_sequence, write_sequence_file

ROOT = Path(__file__).resolve().parent


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


class _GeometryPreview:
    def __init__(self, msg: dict[str, object]) -> None:
        image_b64 = str(msg.get("image_png_b64", ""))
        if not image_b64:
            raise ValueError("geometry preview missing image_png_b64")
        data = base64.b64decode(image_b64)
        self.image = pygame.image.load(io.BytesIO(data), "geometry_preview.png").convert()
        self.title = str(msg.get("title", "Confirm detected grid"))
        self.summary = str(msg.get("summary", "Confirm the green/yellow grid before continuing."))
        self.source_path = str(msg.get("source_path", ""))
        self.confirm_rect = pygame.Rect(0, 0, 0, 0)
        self.cancel_rect = pygame.Rect(0, 0, 0, 0)


class _GeometryCalibration:
    def __init__(self, msg: dict[str, object]) -> None:
        image_b64 = str(msg.get("image_png_b64", ""))
        if not image_b64:
            raise ValueError("geometry calibration missing image_png_b64")
        data = base64.b64decode(image_b64)
        self.image = pygame.image.load(io.BytesIO(data), "geometry_calibration.png").convert()
        self.title = str(msg.get("title", "Manual Board Geometry"))
        self.summary = str(msg.get("summary", "Click 4 green outer corners, then 4 yellow inner corners."))
        self.source_path = str(msg.get("source_path", ""))
        self.outer_points: list[tuple[float, float]] = []
        self.inner_points: list[tuple[float, float]] = []
        self.image_rect = pygame.Rect(0, 0, 0, 0)
        self.save_rect = pygame.Rect(0, 0, 0, 0)
        self.undo_rect = pygame.Rect(0, 0, 0, 0)
        self.reset_rect = pygame.Rect(0, 0, 0, 0)
        self.cancel_rect = pygame.Rect(0, 0, 0, 0)

    def is_complete(self) -> bool:
        return len(self.outer_points) == 4 and len(self.inner_points) == 4

    def next_instruction(self) -> str:
        order = ["top-left", "top-right", "bottom-right", "bottom-left"]
        if len(self.outer_points) < 4:
            return f"Outer green grid: click {order[len(self.outer_points)]} corner."
        if len(self.inner_points) < 4:
            return f"Inner yellow grid: click {order[len(self.inner_points)]} corner."
        return "Geometry complete. Click Save to use this grid for the game."

    def add_click(self, pos: tuple[int, int]) -> None:
        if not self.image_rect.collidepoint(*pos):
            return
        iw, ih = self.image.get_size()
        if self.image_rect.width <= 0 or self.image_rect.height <= 0:
            return
        x = (pos[0] - self.image_rect.x) * iw / self.image_rect.width
        y = (pos[1] - self.image_rect.y) * ih / self.image_rect.height
        x = max(0.0, min(float(iw - 1), float(x)))
        y = max(0.0, min(float(ih - 1), float(y)))
        if len(self.outer_points) < 4:
            self.outer_points.append((x, y))
        elif len(self.inner_points) < 4:
            self.inner_points.append((x, y))

    def undo(self) -> None:
        if self.inner_points:
            self.inner_points.pop()
        elif self.outer_points:
            self.outer_points.pop()

    def reset(self) -> None:
        self.outer_points.clear()
        self.inner_points.clear()


def _back_button_rect(ui: object) -> pygame.Rect:
    return pygame.Rect(12, 12, 96, 36)


def _draw_back_button(ui: object) -> None:
    rect = _back_button_rect(ui)
    mx, my = pygame.mouse.get_pos()
    hover = rect.collidepoint(mx, my)
    bg = (74, 64, 54) if not hover else (92, 78, 62)
    border = (170, 145, 105)

    pygame.draw.rect(ui.screen, bg, rect, border_radius=10)
    pygame.draw.rect(ui.screen, border, rect, width=1, border_radius=10)

    label = ui.font.render("Back", True, (246, 238, 220))
    ui.screen.blit(label, label.get_rect(center=rect.center))


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
    if not inventory_lines and not manual_actions and not config.P2_SHOW_EMPTY_CAPTURE_INVENTORY_OVERLAY:
        return

    screen = ui.screen
    sidebar_x = int(getattr(ui, "sidebar_x", screen.get_width() - 260))
    sidebar_w = int(getattr(ui, "sidebar_w", 240))
    w = max(180, min(sidebar_w + 12, screen.get_width() - 24))
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
    x = max(12, min(sidebar_x - 6, screen.get_width() - w - 12))
    y = max(56, screen.get_height() - h - 12)
    panel = pygame.Surface((w, h), pygame.SRCALPHA)
    panel.fill((15, 18, 24, 205))
    screen.blit(panel, (x, y))
    pygame.draw.rect(screen, (120, 150, 180), pygame.Rect(x, y, w, h), width=1)

    for i, text in enumerate(lines):
        color = (230, 235, 245)
        if i == 0:
            color = (255, 230, 130)
        if text.startswith("MANUAL ACTION REQUIRED"):
            color = (255, 120, 120)
        surf = ui.font.render(text, True, color)
        max_text_w = max(1, w - 16)
        if surf.get_width() > max_text_w:
            clipped = text
            while len(clipped) > 4 and ui.font.render(clipped + "...", True, color).get_width() > max_text_w:
                clipped = clipped[:-1]
            surf = ui.font.render(clipped + "...", True, color)
        screen.blit(surf, (x + 8, y + 6 + i * line_h))


def _poll_geometry_controls(
    transport: object,
    preview: _GeometryPreview | None,
    calibration: _GeometryCalibration | None,
) -> tuple[_GeometryPreview | None, _GeometryCalibration | None]:
    poll = getattr(transport, "poll_control_message", None)
    if not callable(poll):
        return preview, calibration
    while True:
        msg = poll()
        if msg is None:
            return preview, calibration
        msg_type = msg.get("type")
        if msg_type == "geometry_preview":
            try:
                preview = _GeometryPreview(msg)
                calibration = None
            except Exception as exc:  # noqa: BLE001
                print(f"Could not display geometry preview: {exc}", file=sys.stderr)
                sender = getattr(transport, "send_control_message", None)
                if callable(sender):
                    sender({"type": "geometry_confirm", "accepted": False, "error": str(exc)})
                preview = None
            continue
        if msg_type == "geometry_calibration_request":
            preview = None
            calibration = None
            _run_external_geometry_selector(msg, transport)
            continue
        else:
            print(f"Unhandled control message: {msg}", file=sys.stderr)
            continue


def _send_geometry_confirmation(transport: object, accepted: bool) -> None:
    sender = getattr(transport, "send_control_message", None)
    if callable(sender):
        sender({"type": "geometry_confirm", "accepted": bool(accepted)})


def _send_geometry_calibration_result(
    transport: object,
    calibration: _GeometryCalibration,
    accepted: bool,
) -> None:
    sender = getattr(transport, "send_control_message", None)
    if not callable(sender):
        return
    payload: dict[str, object] = {
        "type": "geometry_calibration_result",
        "accepted": bool(accepted),
        "source_path": calibration.source_path,
    }
    if accepted:
        payload["outer_corners_px"] = [[x, y] for x, y in calibration.outer_points]
        payload["chessboard_corners_px"] = [[x, y] for x, y in calibration.inner_points]
    sender(payload)


def _send_geometry_calibration_payload(
    transport: object,
    payload: dict[str, object],
) -> None:
    sender = getattr(transport, "send_control_message", None)
    if callable(sender):
        sender(payload)


def _run_external_geometry_selector(msg: dict[str, object], transport: object) -> None:
    image_b64 = str(msg.get("image_png_b64", ""))
    if not image_b64:
        _send_geometry_calibration_payload(
            transport,
            {
                "type": "geometry_calibration_result",
                "accepted": False,
                "error": "geometry_calibration_request missing image_png_b64",
            },
        )
        return

    request_dir = ROOT / "debug_output" / "manual_geometry"
    request_dir.mkdir(parents=True, exist_ok=True)
    image_path = request_dir / "startup_geometry_request.png"
    image_path.write_bytes(base64.b64decode(image_b64))

    selector_script = ROOT / "tools" / "manual_geometry_selector.py"
    cmd = [
        sys.executable,
        str(selector_script),
        "--image",
        str(image_path),
        "--source-path",
        str(msg.get("source_path", "")),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
    except Exception as exc:  # noqa: BLE001
        _send_geometry_calibration_payload(
            transport,
            {"type": "geometry_calibration_result", "accepted": False, "error": str(exc)},
        )
        return

    if result.returncode != 0:
        _send_geometry_calibration_payload(
            transport,
            {
                "type": "geometry_calibration_result",
                "accepted": False,
                "error": result.stderr.strip() or f"selector exited {result.returncode}",
            },
        )
        return

    try:
        payload = json.loads(result.stdout.strip())
    except json.JSONDecodeError as exc:
        _send_geometry_calibration_payload(
            transport,
            {
                "type": "geometry_calibration_result",
                "accepted": False,
                "error": f"selector returned invalid JSON: {exc}",
            },
        )
        return

    if not isinstance(payload, dict):
        payload = {"type": "geometry_calibration_result", "accepted": False, "error": "selector returned non-object"}
    payload["type"] = "geometry_calibration_result"
    payload["source_path"] = str(msg.get("source_path", ""))
    _send_geometry_calibration_payload(transport, payload)


def _wrap_preview_text(font: pygame.font.Font, text: str, max_width: int) -> list[str]:
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = word if not current else f"{current} {word}"
        if font.render(candidate, True, (230, 230, 235)).get_width() <= max_width:
            current = candidate
            continue
        if current:
            lines.append(current)
        current = word
    if current:
        lines.append(current)
    return lines[:3]


def _draw_geometry_preview(
    screen: pygame.Surface,
    font: pygame.font.Font,
    preview: _GeometryPreview,
) -> None:
    overlay = pygame.Surface(screen.get_size(), pygame.SRCALPHA)
    overlay.fill((0, 0, 0, 190))
    screen.blit(overlay, (0, 0))

    margin = 28
    panel_w = screen.get_width() - margin * 2
    panel_h = screen.get_height() - margin * 2
    panel = pygame.Rect(margin, margin, panel_w, panel_h)
    pygame.draw.rect(screen, (24, 28, 34), panel, border_radius=14)
    pygame.draw.rect(screen, (150, 170, 190), panel, width=1, border_radius=14)

    title_font = pygame.font.SysFont("sf pro display, helvetica, arial", 26, bold=True)
    title = title_font.render(preview.title, True, (248, 248, 250))
    screen.blit(title, (panel.x + 20, panel.y + 16))

    text_y = panel.y + 52
    for line in _wrap_preview_text(font, preview.summary, panel_w - 40):
        surf = font.render(line, True, (218, 224, 232))
        screen.blit(surf, (panel.x + 20, text_y))
        text_y += 23

    if preview.source_path:
        path_text = font.render(preview.source_path, True, (145, 156, 170))
        screen.blit(path_text, (panel.x + 20, text_y + 2))

    buttons_y = panel.bottom - 56
    preview.confirm_rect = pygame.Rect(panel.right - 272, buttons_y, 120, 38)
    preview.cancel_rect = pygame.Rect(panel.right - 142, buttons_y, 110, 38)

    image_top = panel.y + 104
    image_bottom = buttons_y - 16
    max_w = panel_w - 40
    max_h = max(80, image_bottom - image_top)
    iw, ih = preview.image.get_size()
    scale = min(max_w / max(iw, 1), max_h / max(ih, 1), 1.0)
    scaled_size = (max(1, int(iw * scale)), max(1, int(ih * scale)))
    shown = pygame.transform.smoothscale(preview.image, scaled_size)
    image_rect = shown.get_rect(center=(screen.get_width() // 2, image_top + max_h // 2))
    screen.blit(shown, image_rect)
    pygame.draw.rect(screen, (85, 95, 110), image_rect, width=1)

    mx, my = pygame.mouse.get_pos()
    for rect, label, good in [
        (preview.confirm_rect, "Confirm", True),
        (preview.cancel_rect, "Cancel", False),
    ]:
        hover = rect.collidepoint(mx, my)
        if good:
            bg = (35, 105, 62) if not hover else (47, 130, 78)
            border = (120, 220, 150)
        else:
            bg = (92, 54, 54) if not hover else (118, 66, 66)
            border = (215, 130, 130)
        pygame.draw.rect(screen, bg, rect, border_radius=10)
        pygame.draw.rect(screen, border, rect, width=1, border_radius=10)
        lab = font.render(label, True, (248, 248, 250))
        screen.blit(lab, lab.get_rect(center=rect.center))


def _handle_geometry_preview_event(
    preview: _GeometryPreview | None,
    event: pygame.event.Event,
    transport: object,
) -> tuple[_GeometryPreview | None, bool]:
    if preview is None:
        return preview, False
    if event.type == pygame.KEYDOWN:
        if event.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
            _send_geometry_confirmation(transport, True)
            return None, True
        if event.key == pygame.K_ESCAPE:
            _send_geometry_confirmation(transport, False)
            return None, True
    if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
        if preview.confirm_rect.collidepoint(*event.pos):
            _send_geometry_confirmation(transport, True)
            return None, True
        if preview.cancel_rect.collidepoint(*event.pos):
            _send_geometry_confirmation(transport, False)
            return None, True
    return preview, True


def _draw_geometry_calibration(
    screen: pygame.Surface,
    font: pygame.font.Font,
    calibration: _GeometryCalibration,
) -> None:
    overlay = pygame.Surface(screen.get_size(), pygame.SRCALPHA)
    overlay.fill((0, 0, 0, 205))
    screen.blit(overlay, (0, 0))

    margin = 22
    panel = pygame.Rect(margin, margin, screen.get_width() - 2 * margin, screen.get_height() - 2 * margin)
    pygame.draw.rect(screen, (23, 27, 34), panel, border_radius=14)
    pygame.draw.rect(screen, (150, 170, 190), panel, width=1, border_radius=14)

    title_font = pygame.font.SysFont("sf pro display, helvetica, arial", 25, bold=True)
    title = title_font.render(calibration.title, True, (248, 248, 250))
    screen.blit(title, (panel.x + 18, panel.y + 14))

    instruction = calibration.next_instruction()
    progress = f"Outer: {len(calibration.outer_points)}/4    Inner: {len(calibration.inner_points)}/4"
    text_lines = [calibration.summary, instruction, progress]
    text_y = panel.y + 48
    for line in text_lines:
        color = (238, 224, 140) if line == instruction else (218, 224, 232)
        surf = font.render(line, True, color)
        screen.blit(surf, (panel.x + 18, text_y))
        text_y += 22

    if calibration.source_path:
        source = font.render(calibration.source_path, True, (145, 156, 170))
        screen.blit(source, (panel.x + 18, text_y))

    buttons_y = panel.bottom - 54
    calibration.save_rect = pygame.Rect(panel.right - 382, buttons_y, 92, 36)
    calibration.undo_rect = pygame.Rect(panel.right - 282, buttons_y, 82, 36)
    calibration.reset_rect = pygame.Rect(panel.right - 192, buttons_y, 82, 36)
    calibration.cancel_rect = pygame.Rect(panel.right - 102, buttons_y, 82, 36)

    image_top = panel.y + 126
    image_bottom = buttons_y - 14
    max_w = panel.width - 36
    max_h = max(80, image_bottom - image_top)
    iw, ih = calibration.image.get_size()
    scale = min(max_w / max(iw, 1), max_h / max(ih, 1), 1.0)
    scaled_size = (max(1, int(iw * scale)), max(1, int(ih * scale)))
    shown = pygame.transform.smoothscale(calibration.image, scaled_size)
    calibration.image_rect = shown.get_rect(center=(screen.get_width() // 2, image_top + max_h // 2))
    screen.blit(shown, calibration.image_rect)
    pygame.draw.rect(screen, (85, 95, 110), calibration.image_rect, width=1)

    def image_to_screen(point: tuple[float, float]) -> tuple[int, int]:
        return (
            int(calibration.image_rect.x + point[0] * calibration.image_rect.width / max(iw, 1)),
            int(calibration.image_rect.y + point[1] * calibration.image_rect.height / max(ih, 1)),
        )

    def draw_points(points: list[tuple[float, float]], color: tuple[int, int, int], label_prefix: str) -> None:
        screen_pts = [image_to_screen(p) for p in points]
        if len(screen_pts) >= 2:
            pygame.draw.lines(screen, color, len(screen_pts) == 4, screen_pts, width=3)
        for i, pt in enumerate(screen_pts, start=1):
            pygame.draw.circle(screen, color, pt, 6)
            pygame.draw.circle(screen, (10, 10, 10), pt, 6, width=1)
            label = font.render(f"{label_prefix}{i}", True, color)
            screen.blit(label, (pt[0] + 7, pt[1] - 8))

    draw_points(calibration.outer_points, (40, 230, 95), "G")
    draw_points(calibration.inner_points, (245, 215, 45), "Y")

    mx, my = pygame.mouse.get_pos()
    button_specs = [
        (calibration.save_rect, "Save", calibration.is_complete(), (35, 105, 62), (120, 220, 150)),
        (calibration.undo_rect, "Undo", bool(calibration.outer_points or calibration.inner_points), (70, 72, 86), (145, 150, 170)),
        (calibration.reset_rect, "Reset", bool(calibration.outer_points or calibration.inner_points), (70, 72, 86), (145, 150, 170)),
        (calibration.cancel_rect, "Cancel", True, (92, 54, 54), (215, 130, 130)),
    ]
    for rect, label, enabled, bg_base, border in button_specs:
        hover = rect.collidepoint(mx, my)
        bg = tuple(min(255, c + 22) for c in bg_base) if hover and enabled else bg_base
        if not enabled:
            bg = (48, 50, 58)
            border = (85, 90, 105)
        pygame.draw.rect(screen, bg, rect, border_radius=9)
        pygame.draw.rect(screen, border, rect, width=1, border_radius=9)
        lab = font.render(label, True, (248, 248, 250) if enabled else (150, 154, 164))
        screen.blit(lab, lab.get_rect(center=rect.center))


def _handle_geometry_calibration_event(
    calibration: _GeometryCalibration | None,
    event: pygame.event.Event,
    transport: object,
) -> tuple[_GeometryCalibration | None, bool]:
    if calibration is None:
        return calibration, False
    if event.type == pygame.KEYDOWN:
        if event.key in (pygame.K_RETURN, pygame.K_KP_ENTER) and calibration.is_complete():
            _send_geometry_calibration_result(transport, calibration, True)
            return None, True
        if event.key == pygame.K_ESCAPE:
            _send_geometry_calibration_result(transport, calibration, False)
            return None, True
        if event.key in (pygame.K_BACKSPACE, pygame.K_DELETE, pygame.K_u):
            calibration.undo()
            return calibration, True
        if event.key == pygame.K_r:
            calibration.reset()
            return calibration, True
    if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
        if calibration.save_rect.collidepoint(*event.pos):
            if calibration.is_complete():
                _send_geometry_calibration_result(transport, calibration, True)
                return None, True
            return calibration, True
        if calibration.undo_rect.collidepoint(*event.pos):
            calibration.undo()
            return calibration, True
        if calibration.reset_rect.collidepoint(*event.pos):
            calibration.reset()
            return calibration, True
        if calibration.cancel_rect.collidepoint(*event.pos):
            _send_geometry_calibration_result(transport, calibration, False)
            return None, True
        calibration.add_click(event.pos)
        return calibration, True
    return calibration, True


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
        print(
            f"Connecting to Pi bridge at {config.TCP_HOST}:{config.TCP_PORT} "
            f"for up to {config.TCP_CONNECT_RETRIES * config.TCP_CONNECT_RETRY_DELAY_SEC:.0f}s...",
            file=sys.stderr,
        )
        transport.connect(
            retries=config.TCP_CONNECT_RETRIES,
            retry_delay_sec=config.TCP_CONNECT_RETRY_DELAY_SEC,
        )
        print("Connected to Pi bridge.", file=sys.stderr)
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
        ui = ParcheesiUI(config.WINDOW_WIDTH, config.WINDOW_HEIGHT)
        pygame.display.set_caption("Player 2 — Parcheesi")
        if config.TRANSPORT == "mock":
            state.current_player = 2
            waiting_for_p1 = False
    else:
        raise ValueError(f"Unsupported game kind: {kind}")

    geometry_preview: _GeometryPreview | None = None
    geometry_calibration: _GeometryCalibration | None = None

    # Add a "Roll Dice" button for Parcheesi.
    roll_dice_button_rect = pygame.Rect(
        ui.screen.get_width() - 202,
        ui.screen.get_height() - 60,
        180,
        40,
    )

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
        elif kind == "parcheesi":
            if state.check_win_condition(2):
                print("Game over: Player 2 wins.", file=sys.stderr)
                waiting_for_p1 = True
            elif state.rolls_remaining and state.get_possible_moves(2):
                waiting_for_p1 = False
                ai_pending = ai_mode
            else:
                state.current_player = 1
                state.clear_dice()
                waiting_for_p1 = True
        else:
            waiting_for_p1 = True

    while True:
        geometry_preview, geometry_calibration = _poll_geometry_controls(
            transport, geometry_preview, geometry_calibration
        )

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                raise SystemExit(0)

            geometry_calibration, consumed_by_calibration = _handle_geometry_calibration_event(
                geometry_calibration, event, transport
            )
            if consumed_by_calibration:
                continue

            geometry_preview, consumed_by_preview = _handle_geometry_preview_event(
                geometry_preview, event, transport
            )
            if consumed_by_preview:
                continue

            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if _back_button_rect(ui).collidepoint(*event.pos):
                    if hasattr(ui, "clear_drag"):
                        ui.clear_drag()
                    return

                if _mode_button_rect(ui).collidepoint(*event.pos):
                    ai_mode = not ai_mode
                    if ai_mode and not waiting_for_p1:
                        ai_pending = True
                    mode_name = "AI (Stockfish)" if ai_mode else "Manual"
                    print(f"Control mode set to: {mode_name}", file=sys.stderr)
                    continue
                
                if kind == "parcheesi" and roll_dice_button_rect.collidepoint(*event.pos):
                    if state.current_player != 2:
                        print(f"Parcheesi: waiting for Player {state.current_player}, not P2.", file=sys.stderr)
                        continue
                    if state.rolls_remaining:
                        print(f"Parcheesi: finish remaining dice first: {state.rolls_remaining}", file=sys.stderr)
                        continue
                    state.roll_dice()
                    print(f"Player {state.current_player} rolled: {state.dice_rolls}", file=sys.stderr)
                    if not state.get_possible_moves(2):
                        print("Parcheesi: no legal P2 moves for this roll.", file=sys.stderr)
                        state.current_player = 1
                        state.clear_dice()
                        waiting_for_p1 = True
                    else:
                        waiting_for_p1 = False
                        ai_pending = ai_mode

            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
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

        geometry_modal_active = geometry_preview is not None or geometry_calibration is not None

        if waiting_for_p1 and not geometry_modal_active:
            msg = transport.poll_p1_move()
            if msg is not None:
                state_before = state.copy()
                state.apply_move_trusted(msg.frm, msg.to)
                
                # Capture inventory logic needs to be adapted for Parcheesi
                # if kind == "chess":
                #     new_captured = state.captured_by_p1[len(state_before.captured_by_p1):]
                #     for piece in new_captured:
                #         capture_inventory.add_captured_piece("p2", piece.name)
                # elif kind == "checkers":
                #     new_captured = state.captured_by_p1[len(state_before.captured_by_p1):]
                #     for piece in new_captured:
                #         capture_inventory.add_captured_piece("p2", piece.name)
                # elif kind == "parcheesi":
                #     new_captured = state.captured_by_p1[len(state_before.captured_by_p1):]
                #     for piece in new_captured:
                #         capture_inventory.add_captured_piece("p2", piece.name)
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
                elif kind == "parcheesi":
                    waiting_for_p1 = False
                    ai_pending = ai_mode
                else:
                    waiting_for_p1 = False
                    ai_pending = ai_mode

        if not geometry_modal_active and (not waiting_for_p1) and ai_mode and ai_pending:
            chosen = choose_p2_move(kind, state)
            ai_pending = False
            if chosen is None:
                print("AI: no legal P2 move available", file=sys.stderr)
                waiting_for_p1 = True
            else:
                _apply_and_dispatch_p2_move(chosen.start_id, chosen.end_id)

        ui.draw(state, p2_turn=not waiting_for_p1, last_move=last_move)
        _draw_back_button(ui)
        # _draw_capture_inventory_overlay(ui, capture_inventory.to_summary_strings(), latest_manual_actions)
        _draw_mode_button(ui, ai_mode, kind)

        # Draw "Roll Dice" button for Parcheesi
        if kind == "parcheesi":
            pygame.draw.rect(ui.screen, (50, 150, 50), roll_dice_button_rect, border_radius=10)
            roll_text = ui.font.render("Roll Dice", True, (255, 255, 255))
            ui.screen.blit(roll_text, roll_text.get_rect(center=roll_dice_button_rect.center))


        if geometry_preview is not None:
            _draw_geometry_preview(ui.screen, ui.font, geometry_preview)
        if geometry_calibration is not None:
            _draw_geometry_calibration(ui.screen, ui.font, geometry_calibration)
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

    if config.START_GAME:
        if config.START_GAME not in {"checkers", "chess", "parcheesi"}:
            print(f"Unknown P2_START_GAME={config.START_GAME!r}; showing menu.", file=sys.stderr)
        else:
            _run_game_loop(config.START_GAME, transport)

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
    geometry_preview: _GeometryPreview | None = None
    geometry_calibration: _GeometryCalibration | None = None

    try:
        while True:
            geometry_preview, geometry_calibration = _poll_geometry_controls(
                transport, geometry_preview, geometry_calibration
            )

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return
                geometry_calibration, consumed_by_calibration = _handle_geometry_calibration_event(
                    geometry_calibration, event, transport
                )
                if consumed_by_calibration:
                    continue
                geometry_preview, consumed_by_preview = _handle_geometry_preview_event(
                    geometry_preview, event, transport
                )
                if consumed_by_preview:
                    continue
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
            if geometry_preview is not None:
                _draw_geometry_preview(screen, font, geometry_preview)
                pygame.display.flip()
            if geometry_calibration is not None:
                _draw_geometry_calibration(screen, font, geometry_calibration)
                pygame.display.flip()
            clock.tick(60)
    finally:
        close_engine()
        if hasattr(transport, "close"):
            transport.close()


if __name__ == "__main__":
    main()
