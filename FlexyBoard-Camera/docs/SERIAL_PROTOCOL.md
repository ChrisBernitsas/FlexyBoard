# FlexyBoard Serial Protocol (v0.1 JSON Bring-Up)

This document defines the shared serial interface used by:
- Raspberry Pi camera/CV stack (`FlexyBoard-Camera`)
- STM32 motor-control firmware (`FlexyBoard-Motor-Control`)

## 1) Transport
- Physical layer: UART (`115200 8N1` default)
- Framing: one JSON object per line (`\n` terminated)
- Character set: UTF-8

## 2) Packet Envelope
All packets use the same envelope fields:

```json
{
  "type": "EXECUTE_MOVE",
  "seq": 12,
  "ts_ms": 1711200000,
  "payload": {"...": "..."},
  "checksum": "89ABCDEF"
}
```

Required fields:
- `type`: message type string
- `seq`: sender sequence number (uint32)
- `ts_ms`: sender timestamp (ms)
- `payload`: object payload
- `checksum`: uppercase hex CRC32 (`8` chars)

## 3) Checksum Algorithm
Checksum is CRC32 (polynomial `0xEDB88320`) over canonical JSON bytes:

1. Build canonical object with keys in this exact order:
   - `type`, `seq`, `ts_ms`, `payload`
2. Serialize compact JSON (no whitespace)
3. Compute CRC32 over UTF-8 bytes
4. Add `checksum` as uppercase 8-char hex string

Canonical example string:

```json
{"type":"HOME","seq":1,"ts_ms":1711200000,"payload":{}}
```

## 4) Message Types

Pi -> STM32:
- `PING`
- `GET_STATUS`
- `HOME`
- `STOP`
- `RESET_FAULT`
- `EXECUTE_MOVE`

STM32 -> Pi:
- `RECEIVED`
- `ACK`
- `STATUS`
- `DONE`
- `BUSY`
- `ERROR`

## 5) Standard Command Payloads

### `EXECUTE_MOVE`
```json
{
  "game": "chess",
  "source": {"x": 3, "y": 1},
  "dest": {"x": 3, "y": 3},
  "path_mode": "direct",
  "capture": false,
  "waypoints": []
}
```

### `GET_STATUS` / `HOME` / `PING` / `STOP`
```json
{}
```

## 6) Standard Response Payloads

### `RECEIVED`
```json
{"ack_seq": 12, "for": "EXECUTE_MOVE"}
```

### `DONE`
```json
{"op": "EXECUTE_MOVE", "ok": true}
```

### `STATUS`
```json
{"state": "READY", "homed": true, "busy": false, "fault": false, "fault_code": "NONE"}
```

### `ERROR`
```json
{"code": "OUT_OF_BOUNDS", "detail": "source_out_of_bounds"}
```

## 7) ACK/Execution Order
For accepted commands, STM32 sends:
1. `RECEIVED`
2. Terminal response: `DONE`, `ERROR`, `BUSY`, or `STATUS`

## 8) Extension Policy
- Keep envelope keys fixed (`type`, `seq`, `ts_ms`, `payload`, `checksum`)
- Add future fields inside `payload`
- Unknown payload fields must be ignored if not needed by receiver

## 9) Typical Error Codes
- `INVALID_PACKET`
- `NOT_HOMED`
- `OUT_OF_BOUNDS`
- `HOMING_FAILED`
- `MOVE_FAILED`
- `FAULT_LOCKED`
- `UNSUPPORTED_TYPE`
- `DISPATCH_ERROR`

## 10) Binary Migration Path
Future binary protocol can keep semantics while replacing JSON framing with:
- fixed header (type, seq, payload_len)
- binary payload
- CRC16/CRC32 trailer
