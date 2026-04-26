# Motion Planner Algorithm (Software-GUI)

This is the planner that generates `stm32_move_sequence.txt` from a chosen Player-2 move.

It is designed for the current STM firmware behavior:
- one sequence line = one pick-and-place action (`source -> dest`)
- STM itself does `move_to_source -> Z pickup -> move_to_dest -> Z release`

Because of that, all collision-avoidance logic must happen on the software side.

Game support in this module:
- chess: full routing + special handling (castling/en passant/promotion flow hooks)
- checkers: routing + jump-capture midpoint handling
- parcheesi: percent-based source/destination segments (integration mode)

## Inputs

- game state **before** P2 move (`ChessState` or `CheckersState`)
- requested move (`start_id`, `end_id`)
- capture staging configuration (off-board percent coordinates)
- temporary relocation staging configuration (off-board percent coordinates)

## Output

- ordered move list, each line:
  - board endpoint: `x,y`
  - off-board endpoint: `x%,y%`
- file format consumed by `FlexyBoard-Camera/scripts/send_moves_from_file.py`

## High-level Pipeline

1. Build occupied-board map from state_before.
2. Detect capture square(s).
   - Chess: destination square if occupied by P1 piece.
   - Checkers: jumped midpoint square on 2-step diagonal jump.
3. If capture exists:
   - move captured piece to next capture slot off-board.
4. Plan movement for P2 moving piece from source to destination.
5. If route is blocked:
   - move blocking piece(s) to temporary off-board slots.
   - retry routing.
6. After primary move is complete:
   - restore temporary blockers back to their original squares (reverse order).

## Board Routing Core

### Graph model

- Nodes: board squares `(x,y)` on 8x8 grid.
- Traversal: 8-neighbor physical waypoints are considered for board-to-board moves.
- Straight steps collide with the destination square.
- Diagonal steps collide with the destination square plus both orthogonal side squares swept by the piece.
  - Example: moving `g8 -> f7` must have `f7`, `g7`, and `f8` clear or those pieces must be relocated first.
- Occupancy comes from the current software board state before the planned move.

### Search

- Weighted A* over `(position, unique_blockers_seen)`.
- Primary objective: minimize the number of unique physical blockers that must be moved.
- Secondary objective: minimize route length.
- If the best route crosses occupied collision cells, those exact blockers are relocated first, then the route is recomputed.
- If path found, path nodes are compressed into straight runs to reduce command count.
  - Example path nodes: `(2,2)->(2,3)->(2,4)->(3,4)->(4,4)`
  - Compressed segments:
    - `2,2 -> 2,4`
    - `2,4 -> 4,4`

## Off-board Routing

For moves to/from `%` endpoints:

- planner chooses board-side entry/exit by side:
  - `x% >= 50` -> right edge (file 7)
  - `x% < 50` -> left edge (file 0)
- planner picks the best edge rank candidate by shortest reachable A* path.
- then adds one bridge segment between board edge and `%` endpoint.

## Blocker Relocation Logic

For board-to-board moves:

1. Compute the lowest-blocker physical route from source to destination.
2. Identify only the pieces in that route's collision cells.
3. Move each blocker to a temporary parking location.
4. Prefer temporary parking on an empty board square outside the intended route because it is faster and avoids off-board travel.
   - The blocker may pass through empty future route cells while relocating; it just cannot park there.
5. Use off-board percent temporary slots only when no safe board-square parking route exists.
6. Recompute the route after every relocation.
7. Move the original piece to its destination.
8. Restore temporary blockers back to their original squares in reverse order.

This means a knight move such as `g8 -> f6` can clear only the `g7` pawn:

```text
g7 -> g5
g8 -> g6
g6 -> f6
g5 -> g7
```

The current sequence format still emits this as four pick/place commands. A future STM command that supports holding a piece through multiple waypoints could compress the knight's two movement lines into one physical pickup.

Protected squares are never selected as blockers:
- moving piece source and destination
- any additional protected context (example: capture source square while removing captured piece)

## Restore Phase

Temporary blockers are restored in reverse relocation order:
- `temp_slot -> original_square`
- reverse order helps avoid restore-path deadlocks

Can be disabled with `P2_RESTORE_TEMP_RELOCATIONS=0`.

## Staging Slot Allocation

### Capture slots

Used for real captures (not temporary blockers).

Default mode is `bottom_lane`:
- slots are piece-sized positions across the bottom strip
- x starts at `P2_CAPTURE_BOTTOM_X_START_PCT` and advances by `P2_CAPTURE_BOTTOM_X_STEP_PCT`
- `P2_CAPTURE_BOTTOM_COLUMNS` controls how many slots per row
- P1 and P2 captures use separate base rows:
  - `P2_CAPTURE_BOTTOM_P1_BASE_Y_PCT`
  - `P2_CAPTURE_BOTTOM_P2_BASE_Y_PCT`
- additional rows use `P2_CAPTURE_BOTTOM_ROW_STEP_Y_PCT`
- planner/runtime tracks inventory per slot:
  - captured side (`p1`/`p2`)
  - piece type (`P1_PAWN`, `P2_KING`, etc.)
  - slot coordinate (`x%,y%`)
- P1 captures (reported by Pi move stream) are added to this same inventory map.

Legacy mode `side_lane` is still supported:
- x = `P2_CAPTURE_STAGING_X_PCT`
- P1 captured pieces stack downward from `P2_CAPTURE_STAGING_P1_BASE_Y_PCT`
- P2 captured pieces stack upward from `P2_CAPTURE_STAGING_P2_BASE_Y_PCT`

### Temporary relocation slots

Used only for short-term blocker parking, defaults:
- x = 8%
- y = `P2_TEMP_RELOCATE_BASE_Y_PCT + n * P2_TEMP_RELOCATE_STEP_Y_PCT`
- max count controlled by `P2_MAX_TEMP_RELOCATIONS`

## Current Tradeoffs

- Robustness > speed: routes are grid-safe and may use multiple pick/place segments.
- Planner uses board occupancy, not geometric piece meshes.
- Diagonal collision uses a conservative square-sweep model, not exact piece radius geometry.
- Physical routes do not need to match the chess piece's legal movement shape. The game state validates legality; the motor planner finds a safe mechanical route.
- For dense positions, many temporary relocations can increase move time.
- If no grid route can be found after relocation attempts, planner emits a direct fallback segment (tracked in `fallback_direct_segments`).
- Checkers multi-jump physical sequencing is bundled into one GUI/bridge turn using `turn_steps`, while still emitting the full STM path list for the complete sequence.
- Promotion physical replacement is optional and disabled by default unless configured.

## Chess Special Cases (Physical Planning)

- Castling:
  - planner detects castling via python-chess legal move metadata
  - emits explicit direct king/rook segments (no blocker-relocation search for castling itself)
- En passant:
  - planner detects en passant capture square (not destination)
  - emits capture-removal from the correct captured-pawn square to capture staging
  - then emits moving pawn source->destination
- Promotion:
  - logic detection works via legal move metadata
  - default behavior keeps promoted pawn at destination (no physical swap)
  - if `P2_PROMOTION_REPLACE_PHYSICAL=1`, planner emits:
    - destination -> promotion staging bin
    - promotion source -> destination
  - promotion source selection:
    - first try inventory slot with matching side/type piece (example: `P2_QUEEN`)
    - if none exists:
      - default: emit `PROMOTION_MANUAL_REQUIRED` action (no auto reserve pull)
      - optional fallback when `P2_PROMOTION_REQUIRE_MANUAL_IF_MISSING=0`

## Runtime Tuning

Environment variables:

- `P2_CAPTURE_STAGING_MODE`
- `P2_CAPTURE_BOTTOM_COLUMNS`
- `P2_CAPTURE_BOTTOM_X_START_PCT`
- `P2_CAPTURE_BOTTOM_X_STEP_PCT`
- `P2_CAPTURE_BOTTOM_P1_BASE_Y_PCT`
- `P2_CAPTURE_BOTTOM_P2_BASE_Y_PCT`
- `P2_CAPTURE_BOTTOM_ROW_STEP_Y_PCT`
- `P2_CAPTURE_BOTTOM_P2_REVERSE_X`
- (legacy `side_lane`) `P2_CAPTURE_STAGING_X_PCT`, `P2_CAPTURE_STAGING_P1_BASE_Y_PCT`,
  `P2_CAPTURE_STAGING_P2_BASE_Y_PCT`, `P2_CAPTURE_STAGING_STEP_Y_PCT`
- `P2_TEMP_RELOCATE_X_PCT`
- `P2_TEMP_RELOCATE_BASE_Y_PCT`
- `P2_TEMP_RELOCATE_STEP_Y_PCT`
- `P2_MAX_TEMP_RELOCATIONS`
- `P2_RESTORE_TEMP_RELOCATIONS`
- `P2_PROMOTION_REPLACE_PHYSICAL`
- `P2_PROMOTION_REQUIRE_MANUAL_IF_MISSING`
- `P2_PROMOTION_STAGING_X_PCT`
- `P2_PROMOTION_STAGING_Y_PCT`
- `P2_PROMOTION_RESERVE_X_PCT`
- `P2_PROMOTION_RESERVE_Y_PCT`
- `P2_SHOW_CAPTURE_INVENTORY_OVERLAY`
- `P2_CAPTURE_INVENTORY_OVERLAY_MAX_LINES`
