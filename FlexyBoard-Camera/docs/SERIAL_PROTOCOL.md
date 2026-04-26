# FlexyBoard Serial Protocol (v0.1 JSON Bring-Up)

This document defines the shared serial interface used by:
- Raspberry Pi camera/CV stack (`FlexyBoard-Camera`)
- STM32 motor-control firmware (`FlexyBoard-Motor-Control`)

## Current Bring-Up Commands (ASCII, line-based)

The current Pi<->STM bring-up path uses simple newline-terminated ASCII commands over `/dev/ttyACM0` (USB CDC serial on NUCLEO), not JSON framing yet.

Supported STM commands:
- `PING` -> `PONG`
- `ZERO` -> `OK ZERO`
- `STATUS` -> `STATUS cur_x=<steps> cur_y=<steps>`
- `RETURN_START` -> `OK RETURN_START`
- `GOTO gx gy` (board coords 0..7)
- `GOTO_STEPS x y` (absolute workspace steps)
- `GOTOPCT px py` (0..100% of green-grid workspace)
- `MOVE sx sy dx dy` (board coords 0..7)
- `MOVE_STEPS sx sy dx dy` (absolute workspace steps)
- `PICKUP_STEPS sx sy` (move to source and engage Z)
- `MOVEHELD_STEPS dx dy` (move while keeping magnet/piece engaged)
- `RELEASE_STEPS dx dy` (move to destination and disengage Z)
- `MOVEPCT spx spy dpx dpy` (percent endpoints)

Pi move sender behavior (`scripts/send_moves_from_file.py`):
- File board endpoints are game coordinates (`a1=(0,0)`, `h8=(7,7)`) written as `x,y`.
- File percent endpoints are physical green-grid/workspace percentages written as `x%,y%`.
- `configs/default.yaml` `motor.board_orientation` maps game board coordinates to motor board coordinates before step interpolation.
- Current default: `game_a8_at_motor_00`, meaning game `a8` maps to motor board `(0,0)` because the motor home/rest side is at the top-right camera corner.
- The script resolves endpoints to absolute step coordinates and sends `MOVE_STEPS ...`.
- For consecutive file lines where `previous_dest == next_source`, the sender now groups them into one chained physical move:
  - `PICKUP_STEPS ...`
  - zero or more `MOVEHELD_STEPS ...`
  - `RELEASE_STEPS ...`
- If the connected STM firmware does not support the chained commands yet, the sender falls back to legacy per-line `MOVE_STEPS ...`.

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
