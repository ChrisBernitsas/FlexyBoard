Testing:
Terminal 1 (start first), write p1 moves here
   cd "/Users/christopher/Desktop/Design Code/Software-GUI"
   python3 tools/mock_pi_server.py 8765

terminal 2 (make p2 moves on screen)
   cd "/Users/christopher/Desktop/Design Code/Software-GUI"
   source .venv/bin/activate
   export P2_TRANSPORT=tcp
   export P2_TCP_HOST=127.0.0.1
   export P2_TCP_PORT=8765
   python3 main.py


# Player 2 — Checkers + Chess + Parcheesi (Python)

Computer-side UI for the second player: an 8×8 board labeled **columns a–h** and **rows 1–8**, with **rank 1 at the bottom** (row labels on the left run 8 down to 1 top-to-bottom).

On launch you’ll see a **game select** screen to choose **Checkers**, **Chess**, or **Parcheesi**.

## Coordinates

- Square IDs are two-character strings: `a1` … `h8` (file + rank).
- **a1** is the bottom-left corner on screen; **h8** is the top-right.
- Dark squares use `(file_index + rank_index) % 2 == 0`; pieces start on dark squares only (American checkers layout: P1 on ranks 1–3, P2 on ranks 6–8).

## Wire format (Pi ↔ this app)

Newline-delimited JSON (UTF-8), one object per line.

**Pi → P2 (after P1 moves):**

```json
{"type":"p1_move","from":"c3","to":"d4"}
```

**P2 → Pi (after P2 drag-drops a legal move):**

```json
{"type":"p2_move","from":"f6","to":"e5"}
```

Current bridge mode also includes planner output in the same message:

```json
{
  "type":"p2_move",
  "from":"f6",
  "to":"e5",
  "game":"chess",
  "stm_sequence":["5,5 -> 4,4","4,4 -> 92.00%,12.00%"]
}
```

## New integration behavior (CV/physical-board pipeline)

When P2 makes a move (human drag-drop or AI mode), the app can now auto-generate an STM move sequence file for motors:

- Default output file:
  - `../FlexyBoard-Camera/sample_data/stm32_move_sequence.txt`
- Format:
  - one move per line: `source -> dest`
  - board endpoints as `x,y` (0..7)
  - off-board staging endpoints as percentages `x%,y%` (green-grid percent space)

Capture handling:
- Chess capture: moves captured piece from destination square to off-board staging slot, then moves P2 piece.
- Checkers jump-capture: moves jumped midpoint piece to staging, then moves P2 piece.
- Parcheesi mode: emits percent-based source/destination segments for token motion, plus capture staging if destination is occupied by P1 token.
- Captured-piece inventory is tracked by slot (`side + piece type + slot coordinate`) during runtime.
- P1 captures received from Pi are also recorded into this slot inventory so promotion can reuse them.
- If a route is blocked, planner can temporarily relocate blockers to side slots, complete the move, then restore blockers.
- UI highlights the latest move with colored source/destination squares.

## Run — mock transport (no Pi)

Default: `P2_TRANSPORT=mock`. Type or paste JSON lines **to the same terminal’s stdin** to inject P1 moves (a background thread queues them).

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python main.py
```

Example line to paste:

```json
{"type":"p1_move","from":"c3","to":"d4"}
```

Then drag a **black** (P2) piece on screen; the client sends a `p2_move` JSON line to **stderr** (and stores it in `MockTransport.sent_p2`). In TCP mode the same payload is written to the socket.

### AI mode (optional)

Set:

```bash
export P2_CONTROL_MODE=ai
```

Flow in AI mode:
- app receives `p1_move`
- app applies it
- app chooses a legal P2 reply
  - chess: Stockfish when available, fallback otherwise
  - checkers/parcheesi: simple legal heuristic
- app sends `p2_move`
- app writes STM sequence file

You can toggle mode live using the top-right button:
- `Mode: Manual` -> drag/drop P2 move yourself
- `Mode: AI (...)` -> auto-play P2 move for the selected game

## Run — TCP (Raspberry Pi or test server)

1. Start a server that listens and speaks the same protocol. Included test server:

   ```
   cd "/Users/christopher/Desktop/Design Code/Software-GUI"
   python tools/mock_pi_server.py 8765
   ```

2. In another terminal:

   ```bash
   cd "/Users/christopher/Desktop/Design Code/Software-GUI"
   source .venv/bin/activate
   export P2_TRANSPORT=tcp
   export P2_TCP_HOST=127.0.0.1
   export P2_TCP_PORT=8765
   python main.py
 ```

3. Type `p1_move` JSON lines into the **mock_pi_server** terminal; watch `p2_move` lines print when you complete a move in the UI.

## Run with real Pi bridge

One-command launcher from your Mac:

```bash
cd "/Users/christopher/Desktop/Design Code/FlexyBoard-Camera"
./scripts/run_full_system_mac.sh
```

This starts the Pi bridge and the local GUI in separate Terminal windows. The bridge uses rolling capture by default: one initial reference image, then one new image after each Player 1 turn.

The launcher syncs the Pi-side `FlexyBoard-Camera` code before starting unless you run it with `SYNC=0`.

1. On Pi:

```bash
cd ~/FlexyBoard-Camera
source .venv/bin/activate
python3 scripts/run_pi_software_bridge.py
```

2. On your Mac:

```bash
cd "/Users/christopher/Desktop/Design Code/Software-GUI"
source .venv/bin/activate
python3 main.py
```

Defaults already match (`tcp`, host `flexyboard-pi.local`, port `8765`), so no exports are required for the normal hardware path.

### Feed observed move from CV output

If you already have `analysis.json` (or final result JSON) from `FlexyBoard-Camera`, you can send the observed P1 move into this app over TCP:

```bash
cd "/Users/christopher/Desktop/Design Code/Software-GUI"
python3 tools/send_observed_move_to_p2.py /path/to/analysis.json --host 127.0.0.1 --port 8765
```

The window includes a **Captured** panel on the right: **P1 took** lists black (P2) pieces removed by player 1’s jumps; **P2 took** lists red (P1) pieces removed by player 2’s jumps.

## Reset in the UI

With the game window focused, press **R** to restore the opening position, clear any in-progress drag, discard queued incoming `p1_move` messages (so a stale line in mock/TCP does not apply right after reset), and return to **waiting for P1**.

Press **Esc** to return to the **game select** menu.

## Game flow

1. App starts in **waiting for P1**; P2 cannot move until a `p1_move` is applied.
2. P1 moves from the Pi are applied **without** full rule validation (`apply_move_trusted`), including a single two-square jump (captured middle square cleared).
3. P2 uses the mouse: pick up a P2 piece, drop on a square; the move is validated locally; on success a `p2_move` is sent and the app waits for P1 again.

## Limitations (v1)

- One step or one jump per message (no multi-hop chain in one tuple).
- Checkers now enforces:
  - mandatory capture when available
  - forced same-piece continuation for multi-jump sequences (P1 and P2)
  - man/king movement rules and kinging

### Chess limitations (v1)

- Full legal move enforcement is enabled via `python-chess`:
  - check/checkmate/stalemate legality
  - castling
  - en passant
- Promotion UI is not implemented yet:
  - when only `from,to` is provided, promotion defaults to queen.
- When physical promotion replacement is enabled, planner first tries to pull the promoted piece
  from captured-piece inventory slots of the same side/type.
  If not available, it falls back to configured reserve coordinates.

### Parcheesi mode notes (v1)

- Parcheesi now uses explicit location IDs for the full board:
  - `nest_2_1`
  - `main_17`
  - `home_2_3`
  - `homearea_2_1`
- The GUI renders four nests, the 68-space main track, safe/start spaces, home paths, and home area.
- Rules enforced in software include dice entry on 5, remaining dice, safe squares, blockades, captures, and exact home entry.
- The motor planner converts Parcheesi location IDs into percent-based green-grid coordinates for STM movement.

Run the GUI without a Pi connection:

```bash
cd "/Users/christopher/Desktop/Design Code/Software-GUI"
source .venv/bin/activate
P2_TRANSPORT=mock python3 main.py
```

Directly open Parcheesi:

```bash
cd "/Users/christopher/Desktop/Design Code/Software-GUI"
source .venv/bin/activate
P2_TRANSPORT=mock P2_START_GAME=parcheesi python3 main.py
```

### Chess piece images (optional)

Put **12** piece images (PNG/WebP/JPEG) in a folder next to the project:

- Default folder: [`chessPieces/`](chessPieces/) (create it beside `main.py`).

**Naming (any one pattern per piece is enough):** e.g. `wP.png` … `wK.png` and `bP.png` … `bK.png` (white = P1 bottom, black = P2 top). The loader also matches **`white-pawn.png`**, **`black-king.png`**, `white_pawn.png`, etc.

Override the folder with:

```bash
export P2_CHESS_PIECES_DIR=/path/to/your/chessPieces
```

If no images load, the chess board falls back to letter labels.

## Env vars for motor-sequence integration

- `P2_CONTROL_MODE`:
  - initial mode at startup (`human` default)
- `P2_WRITE_STM_SEQUENCE`:
  - `1` (default) to write sequence file, `0` to disable
- `P2_STM_SEQUENCE_FILE`:
  - output path override for generated sequence file
- `P2_CAPTURE_STAGING_MODE`:
  - default `bottom_lane` (stores captures in bottom strip space)
  - set `side_lane` for legacy right-side vertical stack behavior
- `P2_CAPTURE_BOTTOM_COLUMNS`:
  - default `8`
- `P2_CAPTURE_BOTTOM_X_START_PCT`:
  - default `19.7`
- `P2_CAPTURE_BOTTOM_X_STEP_PCT`:
  - default `8.2`
- `P2_CAPTURE_BOTTOM_P1_BASE_Y_PCT`:
  - default `78`
- `P2_CAPTURE_BOTTOM_P2_BASE_Y_PCT`:
  - default `88`
- `P2_CAPTURE_BOTTOM_ROW_STEP_Y_PCT`:
  - default `7.5`
- `P2_CAPTURE_BOTTOM_P2_REVERSE_X`:
  - default `1` (P2 captured slots fill from right-to-left)
- Legacy (`side_lane`) capture vars:
  - `P2_CAPTURE_STAGING_X_PCT`, `P2_CAPTURE_STAGING_P1_BASE_Y_PCT`,
    `P2_CAPTURE_STAGING_P2_BASE_Y_PCT`, `P2_CAPTURE_STAGING_STEP_Y_PCT`
- `P2_TEMP_RELOCATE_X_PCT`:
  - default `8`
- `P2_TEMP_RELOCATE_BASE_Y_PCT`:
  - default `18`
- `P2_TEMP_RELOCATE_STEP_Y_PCT`:
  - default `4`
- `P2_MAX_TEMP_RELOCATIONS`:
  - default `12`
- `P2_RESTORE_TEMP_RELOCATIONS`:
  - default `1` (restore blockers to original squares after main move)
- `P2_PROMOTION_REPLACE_PHYSICAL`:
  - default `0` (off). If `1`, emit physical promotion swap flow.
- `P2_PROMOTION_REQUIRE_MANUAL_IF_MISSING`:
  - default `1`. If a matching promoted piece is not found in captured-slot inventory,
    emit a manual action instead of auto-falling back to reserve coordinates.
- `P2_PROMOTION_STAGING_X_PCT`, `P2_PROMOTION_STAGING_Y_PCT`:
  - where the promoted pawn is moved before replacement (when enabled)
  - defaults now target bottom strip staging area
- `P2_PROMOTION_RESERVE_X_PCT`, `P2_PROMOTION_RESERVE_Y_PCT`:
  - where the replacement promoted piece is picked from (when enabled)
  - defaults now target bottom strip reserve area
- `P2_SHOW_CAPTURE_INVENTORY_OVERLAY`:
  - default `1` to show live capture-slot inventory overlay in GUI
- `P2_CAPTURE_INVENTORY_OVERLAY_MAX_LINES`:
  - default `8` (max lines shown in overlay)
- `P2_PARCHEESI_MIN_X_PCT`, `P2_PARCHEESI_MAX_X_PCT`:
  - percent X bounds of Parcheesi square projection region
- `P2_PARCHEESI_MIN_Y_PCT`, `P2_PARCHEESI_MAX_Y_PCT`:
  - percent Y bounds of Parcheesi square projection region
- `P2_PARCHEESI_INVERT_X`, `P2_PARCHEESI_INVERT_Y`:
  - optional axis flips for Parcheesi percent mapping (default `0`)

## Env vars for Stockfish chess AI

- `P2_STOCKFISH_PATH`:
  - explicit UCI engine path (optional)
- `P2_STOCKFISH_MOVE_TIME_SEC`:
  - per-move think time in seconds (default `0.35`)
- `P2_STOCKFISH_FALLBACK_TO_SIMPLE`:
  - `1` (default): if engine unavailable/invalid move, fallback to simple legal-move AI

## Default connection profile

- `P2_TRANSPORT` default: `tcp`
- `P2_TCP_HOST` default: `flexyboard-pi.local`
- `P2_TCP_PORT` default: `8765`

## Motion planning notes

See [`docs/MOTION_PLANNING_NOTES.md`](docs/MOTION_PLANNING_NOTES.md) for the collision-aware strategy for:
- knight/diagonal routing
- blocker relocation
- capture/staging slot assignment

Detailed implementation:
- [`docs/MOTION_PLANNER_ALGORITHM.md`](docs/MOTION_PLANNER_ALGORITHM.md)
