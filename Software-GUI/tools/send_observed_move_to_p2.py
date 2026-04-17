#!/usr/bin/env python3
"""Send player1 observed move JSON (from CV output) to capstone TCP client as p1_move."""

from __future__ import annotations

import argparse
import json
import socket
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from coords import format_square


def _extract_observed_move(payload: dict[str, Any]) -> dict[str, Any]:
    for key in ("player1_observed_move", "inferred_move", "move_event"):
        value = payload.get(key)
        if isinstance(value, dict):
            return value
    return payload


def _coord_to_square_id(coord: dict[str, Any]) -> str:
    x = int(coord["x"])
    y = int(coord["y"])
    if not (0 <= x <= 7 and 0 <= y <= 7):
        raise ValueError(f"board coord out of range: ({x},{y})")
    return format_square(x, y)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("analysis_json", type=Path, help="Path to analysis/result JSON containing observed move")
    parser.add_argument("--host", default="127.0.0.1", help="capstone TCP host")
    parser.add_argument("--port", type=int, default=8765, help="capstone TCP port")
    args = parser.parse_args()

    data = json.loads(args.analysis_json.read_text(encoding="utf-8"))
    observed = _extract_observed_move(data)

    src = observed.get("source")
    dst = observed.get("destination")
    if not isinstance(src, dict) or not isinstance(dst, dict):
        raise SystemExit("Observed move missing source/destination coordinates")

    line_obj = {
        "type": "p1_move",
        "from": _coord_to_square_id(src),
        "to": _coord_to_square_id(dst),
    }
    line = (json.dumps(line_obj, separators=(",", ":")) + "\n").encode("utf-8")

    with socket.create_connection((args.host, args.port), timeout=10.0) as s:
        s.sendall(line)

    print(json.dumps({"status": "ok", "sent": line_obj, "host": args.host, "port": args.port}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
