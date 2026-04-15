"""Runtime options for P2 checkers client."""

import os

# "mock" or "tcp"
TRANSPORT = os.environ.get("P2_TRANSPORT", "mock").lower()

TCP_HOST = os.environ.get("P2_TCP_HOST", "127.0.0.1")
TCP_PORT = int(os.environ.get("P2_TCP_PORT", "8765"))

WINDOW_WIDTH = int(os.environ.get("P2_WINDOW_W", "900"))
WINDOW_HEIGHT = int(os.environ.get("P2_WINDOW_H", "720"))
