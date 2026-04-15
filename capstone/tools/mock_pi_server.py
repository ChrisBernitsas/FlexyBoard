#!/usr/bin/env python3
"""
Minimal TCP server for local testing: mimics the Pi speaking the same JSON line protocol.

The P2 client connects as a TCP client; run this first, then start main.py with P2_TRANSPORT=tcp.

- Anything you type on stdin (one JSON object per line) is sent to the client as a p1_move.
- Lines sent by the client (p2_move) are printed to stdout.
"""

from __future__ import annotations

import socket
import sys
import threading


HOST = "0.0.0.0"
PORT = 8765


def _reader(conn: socket.socket) -> None:
    buf = bytearray()
    try:
        while True:
            chunk = conn.recv(4096)
            if not chunk:
                break
            buf.extend(chunk)
            while b"\n" in buf:
                line, _, rest = buf.partition(b"\n")
                buf = bytearray(rest)
                print("p2:", line.decode("utf-8", errors="replace"), flush=True)
    except OSError:
        pass


def main() -> None:
    port = int(sys.argv[1]) if len(sys.argv) > 1 else PORT
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((HOST, port))
    srv.listen(1)
    print(f"listening on {HOST}:{port} — connect P2 client, then type JSON lines for p1_move", flush=True)
    conn, addr = srv.accept()
    srv.close()
    print("client from", addr, flush=True)
    threading.Thread(target=_reader, args=(conn,), daemon=True).start()
    try:
        for line in sys.stdin:
            s = line.strip()
            if not s:
                continue
            conn.sendall((s + "\n").encode("utf-8"))
    except BrokenPipeError:
        pass
    finally:
        conn.close()


if __name__ == "__main__":
    main()
