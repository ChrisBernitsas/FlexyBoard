# Motion Planning Notes (STM Move Sequences)

## Goal
Generate Pi-side move sequences that avoid collisions while moving physical pieces on a crowded board.

## Recommended Model
- Keep STM32 low-level: it executes ordered source/dest moves.
- Keep planning on Pi/software:
  - occupancy map of board + staging area
  - collision-aware path checks
  - temporary relocation of blockers when needed

## Core Rules
1. Every move is validated against a swept path, not just source/destination cells.
2. Knight-style transfers are not treated as a geometric “jump”; planner must route around obstacles.
3. Captured pieces go to unique staging slots (never reused while occupied).

## Path Strategy
1. Try direct segment source->dest.
2. Try L-path bends (X then Y, Y then X).
3. If blocked, run grid search (A*) over allowed waypoints (board + safe perimeter lanes).
4. If still blocked, relocate minimal blockers:
   - choose nearest empty safe slots
   - move blocker(s) out
   - execute main move
   - optionally restore blockers if game logic requires

## Collision Check
- Discretize path into substeps.
- For each substep, compute swept footprint (piece radius + safety margin).
- Reject path if footprint intersects occupied square footprint.

## Staging Area
- Maintain a slot table in percent coordinates of green grid.
- Reserve:
  - capture slots
  - temporary blocker slots
- Mark slot occupied/free in planner state.

## Output to STM
- Emit flattened sequence of `source -> dest` lines:
  - board squares: `x,y`
  - staging/perimeter points: `x%,y%`

