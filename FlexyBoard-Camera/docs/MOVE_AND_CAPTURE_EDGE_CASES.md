# Move + Capture Edge Cases (PI -> STM32 Planning)

This file defines special cases and edge cases the vision/planning stack must handle before sending motion sequences to STM32.

## 1. Core Assumptions
- Board is stationary between BEFORE/AFTER captures.
- Camera is fixed.
- Piece movement is inferred from square-level changes.
- STM32 receives explicit motion steps (`source -> dest` segments), not game logic.

## 2. Generic CV Edge Cases
- Lighting shifts between captures cause false changed squares.
- Hand/arm/phone occlusion during AFTER capture.
- Motion blur when capture happens before piece fully settles.
- Shadows crossing board that look like occupancy changes.
- Partial board detection failure in either BEFORE or AFTER.
- Board warp mismatch between BEFORE/AFTER causing diff drift.

## 3. Capture Detection Edge Cases
- Standard capture with only 2 changed squares (source emptied, destination changed) vs noisy 3+ squares.
- False capture from noise on a third square.
- Destination already occupied in BEFORE but subtle intensity difference (low contrast piece).
- Source square not clearly emptied due to lingering shadow/reflection.
- Multiple nearby moved pieces in one turn (illegal for most games, but possible due to user error).

## 4. Chess-Specific Cases
- Normal move (non-capture).
- Capture.
- En passant (capture not on destination square).
- Castling (king + rook move in one turn).
- Promotion:
  - pawn reaches last rank;
  - promoted piece type required (Q/R/B/N).
- Check/checkmate/stalemate state validation.
- Illegal move from CV should be rejected and re-capture requested.

## 5. Checkers-Specific Cases
- Simple diagonal move.
- Single jump capture.
- Multi-jump capture chain in one turn.
- Forced capture rule (if enforced in your mode).
- Kinging (piece reaches back rank).
- Capture removal timing when multiple jumps occur.

## 6. Parcheesi-Specific Cases (planned)
- Piece/token ID tracking over repeated positions.
- Entry from home, blockades, captures/sends-home.
- Multi-step turn from dice values.
- Safe squares where capture is disallowed.

## 7. Motion Planning / STM32 Edge Cases
- Move requires path detour around blockers (e.g., chess knight jump context: magnet gantry cannot pass through tall pieces).
- Temporary relocation of blocking pieces:
  - move blocker to staging square;
  - execute primary move;
  - restore blocker(s).
- Collision risk along straight line path.
- Out-of-bounds/stall recovery.
- Missed steps / drift accumulation and periodic re-zeroing.
- Command timeout / partial execution / retry behavior.

## 8. Required Outputs Per Turn (from Pi planner)
- Observed move:
  - `source`
  - `destination`
  - `capture` flag
  - confidence + diagnostics
- Planned execution sequence for STM32:
  - ordered list of `source -> dest` segments
  - optional staging moves (for obstacle handling)
  - final return-to-home policy (`RETURN_START` / fixed home square).

## 9. Off-Board Capture Placement Requirement
- Captured pieces must be moved to a side "capture tray" area outside the 8x8 board.
- This requires motion targets outside board-index coordinates (`x,y` in `0..7`).
- Pi planner must support mixed routes:
  - on-board pickup/drop using board squares;
  - off-board staging/drop using workspace coordinates.

Recommended coordinate model:
- `coord_space = board`:
  - integer board squares (`x,y`) for game logic.
- `coord_space = workspace_pct`:
  - normalized workspace coordinates (`x_pct`,`y_pct`) in `[0,1]`,
  - where `(0,0)` is machine origin and `(1,1)` is max reachable area.
- STM32 converts workspace percentages to steps using calibrated axis spans.

Why this is useful:
- Board geometry can vary by game layout, but workspace mapping stays stable.
- Off-board capture tray positions become game-agnostic.
- Same protocol can support chess/checkers/parcheesi with different board footprints.

Implementation note (current status):
- Current payload path assumes board-style `source/dest` only.
- Add a planner/protocol upgrade to support per-step coordinate space and conversion.

## 10. Immediate Next Implementations
- Per-game special move handlers:
  - Chess: castling, en passant, promotion.
  - Checkers: multi-jump + kinging.
- Stronger capture validation using:
  - destination occupancy in BEFORE;
  - source occupancy drop in AFTER;
  - legal-move consistency with board state.
- Explicit path planner that emits detour waypoints for STM32.
- Coordinate-space upgrade for off-board capture placement:
  - add workspace staging zones;
  - allow mixed board/workspace step sequences;
  - keep STM32 as execution layer (Pi owns high-level route planning).
