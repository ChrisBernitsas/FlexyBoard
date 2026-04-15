from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from flexyboard_camera.comms.stm32_client import ClientSettings, STM32Client


def print_block(label: str, payload: object) -> None:
    print(f"\n=== {label} ===")
    print(json.dumps(payload, indent=2))


def run(port: str, timeout_sec: float) -> int:
    client = STM32Client(ClientSettings(port=port, baudrate=115200, timeout_sec=timeout_sec, retries=2))

    ping = [pkt.to_dict() for pkt in client.ping()]
    print_block("PING", ping)

    home = [pkt.to_dict() for pkt in client.home()]
    print_block("HOME", home)

    status = [pkt.to_dict() for pkt in client.get_status()]
    print_block("STATUS", status)

    valid_move = {
        "game": "checkers",
        "source": {"x": 3, "y": 1},
        "dest": {"x": 3, "y": 3},
        "path_mode": "direct",
        "capture": False,
        "waypoints": [],
    }
    execute_ok = [pkt.to_dict() for pkt in client.execute_move(valid_move)]
    print_block("EXECUTE_MOVE (valid)", execute_ok)

    bad_move = {
        "game": "checkers",
        "source": {"x": -1, "y": 1},
        "dest": {"x": 99, "y": 3},
        "path_mode": "direct",
        "capture": False,
        "waypoints": [],
    }
    execute_bad = [pkt.to_dict() for pkt in client.execute_move(bad_move)]
    print_block("EXECUTE_MOVE (out_of_bounds)", execute_bad)

    # Malformed packet test (for mock ports only).
    if port.startswith("mock://"):
        transport = client._ensure_connected()  # noqa: SLF001
        malformed = '{"type":"PING","seq":1,"ts_ms":1,"payload":{},"checksum":"DEADBEEF"}'
        transport.write_line(malformed)
        malformed_resp = transport.read_line(timeout_sec=0.2)
        print_block("MALFORMED PACKET", malformed_resp)

    client.close()

    # Timeout path test.
    timeout_client = STM32Client(
        ClientSettings(port="mock://silent", baudrate=115200, timeout_sec=0.05, retries=1)
    )
    timeout_result = "unexpected_success"
    try:
        timeout_client.ping()
    except RuntimeError as exc:
        timeout_result = str(exc)
    finally:
        timeout_client.close()

    print_block("TIMEOUT CASE", {"result": timeout_result})
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="FlexyBoard integration harness for STM32 serial commands")
    parser.add_argument("--port", default="mock://stm32", help="Serial port or mock://stm32")
    parser.add_argument("--timeout", type=float, default=0.3)
    args = parser.parse_args()
    return run(port=args.port, timeout_sec=args.timeout)


if __name__ == "__main__":
    raise SystemExit(main())
