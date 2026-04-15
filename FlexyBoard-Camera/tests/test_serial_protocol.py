from __future__ import annotations

import json

import pytest

from flexyboard_camera.comms.serial_protocol import ProtocolError, build_packet, parse_packet


def test_build_parse_round_trip() -> None:
    raw = build_packet("EXECUTE_MOVE", seq=12, payload={"source": {"x": 1, "y": 1}, "dest": {"x": 1, "y": 3}})
    packet = parse_packet(raw)

    assert packet.type == "EXECUTE_MOVE"
    assert packet.seq == 12
    assert packet.payload["source"]["x"] == 1


def test_checksum_validation_failure() -> None:
    raw = build_packet("PING", seq=1, payload={})
    obj = json.loads(raw)
    obj["checksum"] = "00000000"

    with pytest.raises(ProtocolError):
        parse_packet(json.dumps(obj, separators=(",", ":")))
