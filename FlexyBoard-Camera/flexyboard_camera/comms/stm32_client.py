from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Protocol

from flexyboard_camera.comms.serial_protocol import PacketEnvelope, ProtocolError, build_packet, parse_packet

logger = logging.getLogger(__name__)

try:
    import serial  # type: ignore
except Exception:  # pragma: no cover - handled by runtime fallback
    serial = None


class Transport(Protocol):
    def write_line(self, line: str) -> None:
        ...

    def read_line(self, timeout_sec: float) -> str | None:
        ...

    def close(self) -> None:
        ...


class SerialTransport:
    def __init__(self, port: str, baudrate: int, timeout_sec: float):
        if serial is None:
            raise RuntimeError("pyserial is required for hardware serial mode")
        self._serial = serial.Serial(port=port, baudrate=baudrate, timeout=timeout_sec)

    def write_line(self, line: str) -> None:
        self._serial.write((line + "\n").encode("utf-8"))
        self._serial.flush()

    def read_line(self, timeout_sec: float) -> str | None:
        old_timeout = self._serial.timeout
        self._serial.timeout = timeout_sec
        try:
            raw = self._serial.readline()
        finally:
            self._serial.timeout = old_timeout

        if not raw:
            return None
        return raw.decode("utf-8", errors="replace").strip()

    def close(self) -> None:
        self._serial.close()


class MockSTM32Device:
    def __init__(self) -> None:
        self.state = "INIT"
        self.homed = False
        self.busy = False
        self.seq_reply = 1000

    def _reply(self, msg_type: str, payload: dict) -> str:
        self.seq_reply += 1
        return build_packet(msg_type, self.seq_reply, payload)

    @staticmethod
    def _validate_move(payload: dict) -> bool:
        try:
            sx = int(payload["source"]["x"])
            sy = int(payload["source"]["y"])
            dx = int(payload["dest"]["x"])
            dy = int(payload["dest"]["y"])
        except Exception:
            return False

        return all(0 <= v <= 7 for v in (sx, sy, dx, dy))

    def on_packet(self, raw: str) -> list[str]:
        try:
            pkt = parse_packet(raw)
        except ProtocolError as exc:
            return [self._reply("ERROR", {"code": "INVALID_PACKET", "detail": str(exc)})]

        msg_type = pkt.type
        payload = pkt.payload
        responses: list[str] = [self._reply("RECEIVED", {"ack_seq": pkt.seq, "for": msg_type})]

        if msg_type == "PING":
            responses.append(self._reply("ACK", {"ack_seq": pkt.seq, "pong": True}))
            return responses

        if msg_type == "GET_STATUS":
            responses.append(self._reply("STATUS", {"state": self.state, "homed": self.homed, "busy": self.busy}))
            return responses

        if msg_type == "HOME":
            if self.busy:
                responses.append(self._reply("BUSY", {"state": self.state}))
                return responses
            self.state = "HOMING"
            self.busy = True
            self.homed = True
            self.state = "READY"
            self.busy = False
            responses.append(self._reply("DONE", {"op": "HOME", "ok": True}))
            return responses

        if msg_type == "STOP":
            self.busy = False
            self.state = "ESTOP"
            responses.append(self._reply("DONE", {"op": "STOP", "state": self.state}))
            return responses

        if msg_type == "EXECUTE_MOVE":
            if not self.homed:
                responses.append(self._reply("ERROR", {"code": "NOT_HOMED"}))
                return responses
            if self.busy:
                responses.append(self._reply("BUSY", {"state": self.state}))
                return responses
            if not self._validate_move(payload):
                responses.append(self._reply("ERROR", {"code": "OUT_OF_BOUNDS"}))
                return responses

            self.busy = True
            self.state = "EXECUTING_MOVE"
            # Simulated execution delay
            time.sleep(0.02)
            self.busy = False
            self.state = "READY"
            responses.append(self._reply("DONE", {"op": "EXECUTE_MOVE", "ok": True}))
            return responses

        responses.append(self._reply("ERROR", {"code": "UNSUPPORTED_TYPE", "type": msg_type}))
        return responses


class MockTransport:
    def __init__(self) -> None:
        self.device = MockSTM32Device()
        self._queue: list[str] = []

    def write_line(self, line: str) -> None:
        self._queue.extend(self.device.on_packet(line))

    def read_line(self, timeout_sec: float) -> str | None:
        if self._queue:
            return self._queue.pop(0)

        time.sleep(min(timeout_sec, 0.01))
        return None

    def close(self) -> None:
        self._queue.clear()


class SilentTransport:
    def write_line(self, line: str) -> None:
        _ = line

    def read_line(self, timeout_sec: float) -> str | None:
        time.sleep(min(timeout_sec, 0.01))
        return None

    def close(self) -> None:
        return None


@dataclass(slots=True)
class ClientSettings:
    port: str
    baudrate: int
    timeout_sec: float
    retries: int


class STM32Client:
    def __init__(self, settings: ClientSettings):
        self.settings = settings
        self._transport: Transport | None = None
        self._seq = 1

    def connect(self) -> None:
        if self.settings.port.startswith("mock://silent"):
            self._transport = SilentTransport()
            logger.info("Connected to silent mock transport")
            return

        if self.settings.port.startswith("mock://"):
            self._transport = MockTransport()
            logger.info("Connected to mock STM32 transport")
            return

        self._transport = SerialTransport(
            port=self.settings.port,
            baudrate=self.settings.baudrate,
            timeout_sec=self.settings.timeout_sec,
        )
        logger.info("Connected to serial port %s", self.settings.port)

    def close(self) -> None:
        if self._transport is not None:
            self._transport.close()
            self._transport = None

    def _ensure_connected(self) -> Transport:
        if self._transport is None:
            self.connect()
        assert self._transport is not None
        return self._transport

    def send_command(
        self,
        msg_type: str,
        payload: dict,
        terminal_types: set[str] | None = None,
    ) -> list[PacketEnvelope]:
        terminal_types = terminal_types or {"DONE", "ERROR", "STATUS", "ACK", "BUSY"}
        transport = self._ensure_connected()

        last_error: RuntimeError | None = None
        for attempt in range(1, self.settings.retries + 1):
            seq = self._seq
            self._seq += 1
            packet = build_packet(msg_type, seq=seq, payload=payload)
            logger.info("Sending packet seq=%d type=%s attempt=%d", seq, msg_type, attempt)
            transport.write_line(packet)

            responses: list[PacketEnvelope] = []
            deadline = time.monotonic() + self.settings.timeout_sec
            while time.monotonic() < deadline:
                raw = transport.read_line(timeout_sec=0.05)
                if raw is None:
                    continue

                try:
                    parsed = parse_packet(raw)
                except ProtocolError as exc:
                    logger.warning("Dropped invalid packet: %s", exc)
                    continue

                responses.append(parsed)
                if parsed.type in terminal_types:
                    return responses

            last_error = RuntimeError(f"Timeout waiting for terminal response to {msg_type}")
            logger.warning(str(last_error))

        assert last_error is not None
        raise last_error

    def ping(self) -> list[PacketEnvelope]:
        return self.send_command("PING", payload={})

    def get_status(self) -> list[PacketEnvelope]:
        return self.send_command("GET_STATUS", payload={}, terminal_types={"STATUS", "ERROR", "BUSY"})

    def home(self) -> list[PacketEnvelope]:
        return self.send_command("HOME", payload={})

    def execute_move(self, payload: dict) -> list[PacketEnvelope]:
        return self.send_command("EXECUTE_MOVE", payload=payload)
