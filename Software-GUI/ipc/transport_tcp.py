"""TCP client: connect to Pi server, read p1_move lines, write p2_move lines."""

from __future__ import annotations

import socket
import threading
from queue import Queue, Empty
from typing import Optional

from ipc.protocol import P1MoveMessage, P2MoveMessage, decode_line, encode_line


class TcpClientTransport:
    """
    Pi runs a TCP server. This client connects and:
    - background thread reads newline-delimited JSON and pushes P1MoveMessage to queue
    - send_p2_move writes one JSON line
    """

    def __init__(self, host: str, port: int) -> None:
        self._host = host
        self._port = port
        self._sock: Optional[socket.socket] = None
        self._buf = bytearray()
        self._incoming: Queue[P1MoveMessage] = Queue()
        self._reader_thread: Optional[threading.Thread] = None
        self._stop = threading.Event()

    def connect(self) -> None:
        s = socket.create_connection((self._host, self._port), timeout=10.0)
        s.settimeout(0.5)
        self._sock = s
        self._stop.clear()
        self._reader_thread = threading.Thread(target=self._read_loop, daemon=True)
        self._reader_thread.start()

    def close(self) -> None:
        self._stop.set()
        if self._sock:
            try:
                self._sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            try:
                self._sock.close()
            except OSError:
                pass
            self._sock = None
        if self._reader_thread and self._reader_thread.is_alive():
            self._reader_thread.join(timeout=2.0)
        self._reader_thread = None

    def poll_p1_move(self, block: bool = False, timeout: float = 0.0) -> Optional[P1MoveMessage]:
        try:
            return self._incoming.get(block=block, timeout=timeout if block else 0)
        except Empty:
            return None

    def drain_p1_moves(self) -> None:
        while True:
            try:
                self._incoming.get_nowait()
            except Empty:
                break

    def send_p2_move(self, msg: P2MoveMessage) -> None:
        if not self._sock:
            raise RuntimeError("not connected")
        data = encode_line(msg.to_obj())
        self._sock.sendall(data)

    def _read_loop(self) -> None:
        assert self._sock is not None
        while not self._stop.is_set():
            try:
                chunk = self._sock.recv(4096)
            except socket.timeout:
                continue
            except OSError:
                break
            if not chunk:
                break
            self._buf.extend(chunk)
            while b"\n" in self._buf:
                line, _, rest = self._buf.partition(b"\n")
                self._buf = bytearray(rest)
                decoded = decode_line(bytes(line))
                if isinstance(decoded, P1MoveMessage):
                    self._incoming.put(decoded)
