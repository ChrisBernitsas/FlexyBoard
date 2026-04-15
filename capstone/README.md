Testing:
Terminal 1 (start first), write p1 moves here
   cd ~/Desktop/capstone
   python3 tools/mock_pi_server.py 8765

terminal 2 (make p2 moves on screen)
   cd /Users/ninini/Desktop/capstone
   source .venv/bin/activate
   export P2_TRANSPORT=tcp
   export P2_TCP_HOST=127.0.0.1
   export P2_TCP_PORT=8765
   python3 main.py


# Player 2 — Checkers + Chess (Python)

Computer-side UI for the second player: an 8×8 board labeled **columns a–h** and **rows 1–8**, with **rank 1 at the bottom** (row labels on the left run 8 down to 1 top-to-bottom).

On launch you’ll see a **game select** screen to choose **Checkers** or **Chess**.

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

## Run — TCP (Raspberry Pi or test server)

1. Start a server that listens and speaks the same protocol. Included test server:

   ```
   cd ~/Desktop/capstone
   python tools/mock_pi_server.py 8765
   ```

2. In another terminal:

   ```bash
   cd /Users/ninini/Desktop/capstone
   source .venv/bin/activate
   export P2_TRANSPORT=tcp
   export P2_TCP_HOST=127.0.0.1
   export P2_TCP_PORT=8765
   python main.py
 ```

3. Type `p1_move` JSON lines into the **mock_pi_server** terminal; watch `p2_move` lines print when you complete a move in the UI.

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
- P2 validation is basic (diagonal step/jump, capture on jump, men forward-only).

### Chess limitations (v1)

- Basic move legality for P2 pieces (pawn/knight/bishop/rook/queen/king) is enforced.
- **Not implemented yet**: check/checkmate, castling, en passant, pawn promotion UI.

### Chess piece images (optional)

Put **12** piece images (PNG/WebP/JPEG) in a folder next to the project:

- Default folder: [`chessPieces/`](chessPieces/) (create it beside `main.py`).

**Naming (any one pattern per piece is enough):** e.g. `wP.png` … `wK.png` and `bP.png` … `bK.png` (white = P1 bottom, black = P2 top). The loader also matches **`white-pawn.png`**, **`black-king.png`**, `white_pawn.png`, etc.

Override the folder with:

```bash
export P2_CHESS_PIECES_DIR=/path/to/your/chessPieces
```

If no images load, the chess board falls back to letter labels.
