#!/usr/bin/env python3
from __future__ import annotations

import argparse
from http import server
import os
from pathlib import Path
import socket
import sys
import threading
import time

import cv2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from flexyboard_camera.utils.config import load_config


def _parse_roi(raw: str) -> tuple[int, int, int, int]:
    parts = [int(item.strip()) for item in raw.split(",")]
    if len(parts) != 4:
        raise ValueError("ROI must be x,y,w,h")
    return parts[0], parts[1], parts[2], parts[3]


def _build_display_frame(
    frame,
    *,
    cap: cv2.VideoCapture,
    index: int,
    roi: tuple[int, int, int, int],
    show_roi: bool,
):
    display = frame.copy()
    if show_roi:
        x, y, w, h = roi
        cv2.rectangle(display, (x, y), (x + w, y + h), (0, 200, 255), 2)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cv2.putText(
        display,
        f"{actual_w}x{actual_h} idx={index}",
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return display


class _PreviewState:
    def __init__(self) -> None:
        self.latest_jpeg: bytes | None = None
        self.lock = threading.Lock()
        self.stop_event = threading.Event()


def _run_http_preview(
    *,
    cap: cv2.VideoCapture,
    index: int,
    roi: tuple[int, int, int, int],
    show_roi: bool,
    host: str,
    port: int,
) -> int:
    state = _PreviewState()

    class PreviewHandler(server.BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            if self.path in {"/", "/index.html"}:
                body = f"""<!doctype html>
<html>
  <head>
    <meta charset="utf-8">
    <title>FlexyBoard Camera Preview</title>
    <style>
      body {{
        margin: 0;
        background: #202124;
        color: #f1f3f4;
        font: 14px -apple-system, BlinkMacSystemFont, sans-serif;
      }}
      .wrap {{
        padding: 16px;
      }}
      img {{
        display: block;
        max-width: 100%;
        height: auto;
        border: 1px solid #5f6368;
      }}
    </style>
  </head>
  <body>
    <div class="wrap">
      <div>FlexyBoard Camera Preview</div>
      <img src="/stream.mjpg" alt="preview stream" />
    </div>
  </body>
</html>
""".encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return

            if self.path == "/snapshot.jpg":
                with state.lock:
                    payload = state.latest_jpeg
                if payload is None:
                    self.send_error(503, "No frame available yet")
                    return
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
                return

            if self.path == "/stream.mjpg":
                self.send_response(200)
                self.send_header("Age", "0")
                self.send_header("Cache-Control", "no-cache, private")
                self.send_header("Pragma", "no-cache")
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.end_headers()
                try:
                    while not state.stop_event.is_set():
                        with state.lock:
                            payload = state.latest_jpeg
                        if payload is None:
                            time.sleep(0.05)
                            continue
                        self.wfile.write(b"--frame\r\n")
                        self.wfile.write(b"Content-Type: image/jpeg\r\n")
                        self.wfile.write(f"Content-Length: {len(payload)}\r\n\r\n".encode("ascii"))
                        self.wfile.write(payload)
                        self.wfile.write(b"\r\n")
                        time.sleep(0.03)
                except (BrokenPipeError, ConnectionResetError):
                    return
                return

            self.send_error(404)

        def log_message(self, format: str, *args) -> None:  # noqa: A003
            return

    def capture_loop() -> None:
        while not state.stop_event.is_set():
            ok = cap.grab()
            if not ok:
                time.sleep(0.02)
                continue
            ok, frame = cap.retrieve()
            if not ok or frame is None:
                time.sleep(0.02)
                continue
            display = _build_display_frame(frame, cap=cap, index=index, roi=roi, show_roi=show_roi)
            ok, encoded = cv2.imencode(".jpg", display, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            if not ok:
                continue
            with state.lock:
                state.latest_jpeg = encoded.tobytes()

    httpd = server.ThreadingHTTPServer((host, port), PreviewHandler)
    worker = threading.Thread(target=capture_loop, daemon=True)
    worker.start()

    nice_host = host if host not in {"0.0.0.0", ""} else socket.gethostname()
    print(f"Preview server: http://{nice_host}:{port}/", file=sys.stderr)
    print(f"Pi hostname URL: http://flexyboard-pi.local:{port}/", file=sys.stderr)
    print("Press Ctrl+C to stop.", file=sys.stderr)

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        state.stop_event.set()
        httpd.shutdown()
        httpd.server_close()
        worker.join(timeout=1.0)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Show a live preview of the configured camera feed.")
    parser.add_argument("--config", default="configs/default.yaml", help="Config file to read camera defaults from")
    parser.add_argument("--index", type=int, default=None, help="Override camera index")
    parser.add_argument("--width", type=int, default=None, help="Override camera width")
    parser.add_argument("--height", type=int, default=None, help="Override camera height")
    parser.add_argument("--roi", default=None, help="Optional ROI override as x,y,w,h")
    parser.add_argument("--hide-roi", action="store_true", help="Do not draw the configured ROI rectangle")
    parser.add_argument("--http-port", type=int, default=0, help="Serve preview over HTTP instead of opening a window")
    parser.add_argument("--http-host", default="0.0.0.0", help="HTTP bind host when --http-port is used")
    parser.add_argument(
        "--snapshot-out",
        default="debug_output/camera_preview_snapshot.png",
        help="Path to write a snapshot when you press 's'",
    )
    parser.add_argument("--window-name", default="FlexyBoard Camera Preview")
    args = parser.parse_args()

    config = load_config(ROOT / args.config)
    index = config.camera.index if args.index is None else args.index
    width = config.camera.width if args.width is None else args.width
    height = config.camera.height if args.height is None else args.height
    roi = config.vision.roi if args.roi is None else _parse_roi(args.roi)

    cap = cv2.VideoCapture(index)
    if not cap or not cap.isOpened():
        print(f"Unable to open camera index {index}", file=sys.stderr)
        return 1

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(width))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(height))
    if hasattr(cv2, "CAP_PROP_BUFFERSIZE"):
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1.0)

    snapshot_out = ROOT / args.snapshot_out
    show_roi = not args.hide_roi

    print(
        "Camera preview controls: "
        "'q' or ESC quit, 's' save snapshot, 'r' toggle ROI overlay",
        file=sys.stderr,
    )
    print(
        f"index={index} requested={width}x{height} roi={tuple(int(v) for v in roi)}",
        file=sys.stderr,
    )

    if args.http_port:
        try:
            return _run_http_preview(
                cap=cap,
                index=index,
                roi=roi,
                show_roi=show_roi,
                host=args.http_host,
                port=args.http_port,
            )
        finally:
            cap.release()

    if not (hasattr(sys, "ps1") or ("DISPLAY" in os.environ)):
        cap.release()
        print(
            "No DISPLAY found. Use --http-port for headless preview, "
            "for example: python3 scripts/camera_preview.py --http-port 8766",
            file=sys.stderr,
        )
        return 2

    try:
        while True:
            ok = cap.grab()
            if not ok:
                time.sleep(0.02)
                continue
            ok, frame = cap.retrieve()
            if not ok or frame is None:
                time.sleep(0.02)
                continue

            display = _build_display_frame(frame, cap=cap, index=index, roi=roi, show_roi=show_roi)

            cv2.imshow(args.window_name, display)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            if key == ord("r"):
                show_roi = not show_roi
            if key == ord("s"):
                snapshot_out.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(snapshot_out), frame)
                print(f"Saved snapshot to {snapshot_out}", file=sys.stderr)
    finally:
        cap.release()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
