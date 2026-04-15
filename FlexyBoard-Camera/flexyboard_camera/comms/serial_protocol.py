from __future__ import annotations

import json
import time
import zlib
from dataclasses import dataclass, field
from typing import Any

MESSAGE_TYPES = {
    "HOME",
    "STOP",
    "EXECUTE_MOVE",
    "GET_STATUS",
    "PING",
    "ACK",
    "RECEIVED",
    "BUSY",
    "DONE",
    "ERROR",
    "STATUS",
}


class ProtocolError(ValueError):
    pass


@dataclass(slots=True)
class PacketEnvelope:
    type: str
    seq: int
    ts_ms: int
    payload: dict[str, Any]
    checksum: str
    extensions: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        output = {
            "type": self.type,
            "seq": self.seq,
            "ts_ms": self.ts_ms,
            "payload": self.payload,
            "checksum": self.checksum,
        }
        output.update(self.extensions)
        return output


def _canonical_json(type_name: str, seq: int, ts_ms: int, payload: dict[str, Any]) -> str:
    canonical = {
        "type": type_name,
        "seq": seq,
        "ts_ms": ts_ms,
        "payload": payload,
    }
    return json.dumps(canonical, separators=(",", ":"), sort_keys=False)


def compute_checksum(type_name: str, seq: int, ts_ms: int, payload: dict[str, Any]) -> str:
    canonical = _canonical_json(type_name=type_name, seq=seq, ts_ms=ts_ms, payload=payload)
    crc = zlib.crc32(canonical.encode("utf-8")) & 0xFFFFFFFF
    return f"{crc:08X}"


def build_packet(type_name: str, seq: int, payload: dict[str, Any], ts_ms: int | None = None, **extensions: Any) -> str:
    if type_name not in MESSAGE_TYPES:
        raise ProtocolError(f"Unknown message type: {type_name}")

    ts_ms = int(time.time() * 1000) if ts_ms is None else ts_ms
    checksum = compute_checksum(type_name=type_name, seq=seq, ts_ms=ts_ms, payload=payload)

    packet: dict[str, Any] = {
        "type": type_name,
        "seq": int(seq),
        "ts_ms": int(ts_ms),
        "payload": payload,
        "checksum": checksum,
    }
    packet.update(extensions)
    return json.dumps(packet, separators=(",", ":"), sort_keys=False)


def parse_packet(raw: str) -> PacketEnvelope:
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ProtocolError(f"Invalid JSON packet: {exc}") from exc

    for key in ("type", "seq", "ts_ms", "payload", "checksum"):
        if key not in parsed:
            raise ProtocolError(f"Missing required field: {key}")

    type_name = str(parsed["type"])
    seq = int(parsed["seq"])
    ts_ms = int(parsed["ts_ms"])
    payload = parsed["payload"]
    checksum = str(parsed["checksum"]).upper()

    if type_name not in MESSAGE_TYPES:
        raise ProtocolError(f"Unknown message type: {type_name}")
    if not isinstance(payload, dict):
        raise ProtocolError("Payload must be an object")

    expected = compute_checksum(type_name=type_name, seq=seq, ts_ms=ts_ms, payload=payload)
    if checksum != expected:
        raise ProtocolError(f"Checksum mismatch; expected={expected} got={checksum}")

    extensions = {k: v for k, v in parsed.items() if k not in {"type", "seq", "ts_ms", "payload", "checksum"}}
    return PacketEnvelope(type=type_name, seq=seq, ts_ms=ts_ms, payload=payload, checksum=checksum, extensions=extensions)
