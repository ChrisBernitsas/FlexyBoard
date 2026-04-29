# FlexyBoard-Camera

Raspberry Pi camera + CV + command orchestration stack for FlexyBoard.

This module is responsible for:
- Camera capture (`before` / `after` turn images)
- Calibration and board ROI handling
- Image preprocessing + square-level diff detection
- Move inference (`source`, `destination`, confidence)
- High-level response move generation
- Serial command generation and STM32 communication
- Debug artifact logging and JSON event records

Shared serial protocol spec:
- `docs/SERIAL_PROTOCOL.md` (local copy)
- `../FlexyBoard-Shared/SERIAL_PROTOCOL.md`

## Quick Start

```bash
cd FlexyBoard-Camera
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Generate sample images (dry-run assets):

```bash
python scripts/generate_sample_images.py
```

Probe camera indices and save a live snapshot (recommended first on Pi):

```bash
python scripts/camera_probe.py --max-index 5 --width 1920 --height 1080
```

Detect chessboard corners from a captured image (for board-to-coordinate mapping sanity check):

```bash
python scripts/detect_board_corners.py --image debug_output/before_latest.png
```

Detect both outer sheet and inner chessboard regions:

```bash
python scripts/detect_board_regions.py --image debug_output/before_latest.png
```

Run full board analysis + square numbering + before/after diff:

```bash
python scripts/analyze_board_and_diff.py \
  --before debug_output/before_20260329_165100.png \
  --after debug_output/after_20260329_165100.png \
  --label-mode index
```

By default, this script uses `configs/corners_info.json` (if present) to lock green/yellow geometry to a stable board layout.
Disable that behavior with `--disable-geometry-reference`.

Manually draw outer/chessboard grids on a specific turn pair (BEFORE + AFTER), while viewing `raw` / `hsv_mask` / `dark_mask`:

```bash
python scripts/annotate_turn_geometry.py \
  --before debug_output/before_latest.png \
  --after debug_output/after_latest.png
```

Controls in the annotation window:
- left-click: add point
- right-click or `u`: undo
- `1` / `2` / `3`: raw / hsv_mask / dark_mask view
- `o` / `i`: switch active target (outer vs chessboard)
- `s` (or Enter): save image annotation when both quads have 4 points
- `c`: clear points
- `q` or Esc: abort

Use the saved JSON directly as analyzer geometry lock:

```bash
python scripts/analyze_board_and_diff.py \
  --before debug_output/before_latest.png \
  --after debug_output/after_latest.png \
  --geometry-reference debug_output/manual_turn_geometry_YYYYMMDD_HHMMSS.json
```

This generates:
- outer board/sheet corners (for your taped perimeter)
- inner chessboard corners
- 64 square polygons with indices/labels
- changed-square list from before/after diff
- overlays and `analysis.json` under `debug_output/board_analysis_<timestamp>/`
- `official/` subfolder: locked/reference geometry outputs, diff outputs, `analysis_summary.txt`
- `algorithm_live/` subfolder: fresh per-image detection overlays and detector debug masks
- locked overlays (reference geometry): `official/before_grid_overlay.png`, `official/after_grid_overlay.png`
- live overlays (fresh per-image detection): `algorithm_live/before_grid_overlay_live.png`, `algorithm_live/after_grid_overlay_live.png`
- detector debug artifacts for green/yellow region finding, including masks and candidate overlays:
  - `before_detector_hsv_mask_raw.png`, `before_detector_hsv_mask.png`, `before_detector_dark_mask_raw.png`
  - `before_detector_candidate_overlay.png`, `before_detector_late_tape_mask.png`, `before_detector_late_tape_projection_overlay.png`
  - matching `after_detector_*` files
  - separate overlays for review: `*_outer_only_overlay.png` and `*_chess_only_overlay.png`

One-command capture + wait + analyze + STM32 payload generation:

```bash
python scripts/run_turn_capture_analyze.py --wait-mode enter
```

### Live Bridge: CV -> Software-GUI -> STM32

One-command Mac launcher for the normal hardware path:

```bash
cd "/Users/christopher/Desktop/Design Code/FlexyBoard-Camera"
./scripts/run_full_system_mac.sh
```

This opens two Terminal windows:
- Raspberry Pi bridge: camera/CV + STM32 dispatch
- local `Software-GUI`: board UI + Player 2 move planning

By default the launcher syncs `FlexyBoard-Camera` and `FlexyBoard-Motor-Control` to the Pi first. It excludes virtualenv/build/debug folders so generated files do not overwrite the Pi runtime.

If the STM32 command protocol changed, flash the STM32 before launching the bridge:

```bash
cd "/Users/christopher/Desktop/Design Code/FlexyBoard-Camera"
FLASH_STM32=1 ./scripts/run_full_system_mac.sh
```

Useful launcher flags:
- `SYNC=0 ./scripts/run_full_system_mac.sh`: skip syncing both repos
- `SYNC_MOTOR=0 ./scripts/run_full_system_mac.sh`: sync camera only
- `BRIDGE_ARGS="--once --no-stm-send" ./scripts/run_full_system_mac.sh`: one CV/GUI test turn without moving motors

Run this on the Pi to host the TCP bridge used by `Software-GUI`:

```bash
cd ~/FlexyBoard-Camera
source .venv/bin/activate
python3 scripts/run_pi_software_bridge.py
```

Default bridge mode is rolling capture, similar to Chess-Tracker:
1. capture one initial reference image when the system starts
2. wait for Player 1 to move and press Enter
3. capture one current image
4. diff previous reference -> current image
5. resolve Player 1 move against legal game state on Pi
6. send resolved Player 1 move step(s) to Software-GUI as `{"type":"p1_move","from":"...","to":"..."}`
7. receive Software-GUI Player 2 reply (`p2_move`) with `stm_sequence`
8. write `sample_data/stm32_move_sequence.txt`
9. call `scripts/send_moves_from_file.py` to execute on STM32
10. capture a fresh reference image after STM32 movement completes

Per-turn legal-resolution artifact:
- `player1_resolved_move.json` in the analysis folder

Optional:
- run once only: `--once`
- skip STM execution (debug bridge only): `--no-stm-send`
- GPIO trigger instead of Enter: `--wait-mode gpio --gpio-pin 17`
- old two-image-per-turn behavior: `--capture-mode two-shot`

Flow for this command:
1. captures `before`
2. waits for Enter (or GPIO trigger)
3. captures `after`
4. runs board+square diff analysis
5. records Player 1 observed move in `player1_observed_move.json`
6. computes Player 2 response move sequence (internal planner by default)
7. prints generated STM32 move payload/sequence without sending

This flow also uses the geometry reference file by default:
- `configs/corners_info.json`
- if missing/invalid, it falls back to live detection

It also writes a turn summary text file in the analysis folder:
- `turn_decision_summary.txt` (changed squares + inferred move + STM32 payload to send)
- `player1_observed_move.json` (what CV inferred for Player 1)
- `player2_software_move_sequence.json` (what Pi will send to STM32)

### External Player 2 Software Integration

If your Player 2 software runs separately and outputs a move sequence JSON, pass it directly:

```bash
python scripts/run_turn_capture_analyze.py \
  --wait-mode enter \
  --player2-response-json /path/to/player2_output.json

# example file in this repo:
python scripts/run_turn_capture_analyze.py \
  --wait-mode enter \
  --player2-response-json sample_data/player2_response_sequence.example.json
```

Accepted JSON shapes:
- list of steps: `[{"source":{"x":2,"y":2},"dest":{"x":3,"y":3}}, ...]`
- object with sequence key:
  - `player2_software_move_sequence`
  - `move_sequence`
  - `stm32_move_sequence`
  - `steps`
  - `moves`
- single object with optional `waypoints`:
  - `{"source": {...}, "dest": {...}, "waypoints": [...]}`

When `--send` is added, this Player 2 sequence is what goes to STM32.

STM32 send model:
- Pi now builds a sequence of move steps and sends them one-by-one as `EXECUTE_MOVE`.
- `stm32_move_sequence` in the result JSON is the exact ordered list sent when `--send` is used.
- For each step, STM32 receives only movement coordinates:
  - `source: {x, y}`
  - `dest: {x, y}`

Tape projection notes:
- Tape projection is now disabled by default.
- Enable only for experiments with `--enable-tape-projection`.

To actually send, add `--send` and configure `comms.port` to a real serial device.

Example for USB-connected STM32 (`/dev/ttyACM0` on Pi is common):

```bash
python scripts/run_turn_capture_analyze.py \
  --wait-mode enter \
  --send \
  --serial-port /dev/ttyACM0 \
  --serial-baudrate 115200
```

Run calibration (ROI-corner based starter calibration):

```bash
python -m flexyboard_camera.app.main --config configs/default.yaml calibrate
```

When `vision.auto_detect_board: true`, calibration and inference will try to detect:
- the outer brown sheet
- the inner chessboard corners
and will store detection overlays under `debug_output/` and per-cycle folders.

Infer move from saved test images:

```bash
python -m flexyboard_camera.app.main --config configs/default.yaml \
  infer_move --before sample_data/before_demo.png --after sample_data/after_demo.png
```

Run full end-turn cycle in mock mode (no GPIO, no real STM32):

```bash
python -m flexyboard_camera.app.main --config configs/default.yaml \
  run_end_turn_cycle --before sample_data/before_demo.png --after sample_data/after_demo.png --force
```

Run serial integration harness:

```bash
python scripts/integration_harness.py --port mock://stm32
```

Send move sequence from a text file to STM32:

```bash
python scripts/send_moves_from_file.py
```

Edit this file before running:
- `sample_data/stm32_move_sequence.txt`

Default serial port used by script:
- `/dev/ttyACM0`

Text file format (`sample_data/stm32_move_sequence.txt`):
- one move per line
- accepted examples:
  - `2,2 -> 5,5`
  - `2 2 5 5`
- `#` comment lines are ignored

Default behavior of this script:
1. `PING`
2. `ZERO` (resets STM32 internal position tracker to machine `(0,0)`)
3. execute each file move as `MOVE sx sy dx dy`
4. append return-to-zero move from last destination to `(0,0)`
5. `STATUS`

This script intentionally supports no command-line flags.

If you get timeout/no response, ensure STM32 is flashed with the UART `PING`/`ZERO`/`MOVE` parser firmware.

### Capture On Pi + Auto Copy To Mac

From your Mac (local repo), run:

```bash
./scripts/capture_on_pi_and_pull.sh before
./scripts/capture_on_pi_and_pull.sh after
```

Or run both in one flow:

```bash
./scripts/capture_on_pi_and_pull.sh both
```

Run full turn-cycle on Pi and auto-pull/open analysis locally:

```bash
./scripts/run_turn_on_pi_and_pull.sh
```

This command:
1. starts `run_turn_capture_analyze.py` on Pi
2. captures BEFORE, waits for Enter, captures AFTER
3. runs board/square/diff analysis on Pi
4. pulls latest `board_analysis_*` folder to local `debug_output/turn_run_<timestamp>/`
5. opens key overlays and `analysis.json` on macOS

Run full turn-cycle on Pi and save locally without opening any windows:

```bash
./scripts/run_turn_on_pi_save_only.sh
```

This script:
- runs capture on the Raspberry Pi over SSH
- copies images back to local `debug_output/`
- uses timestamped filenames (for example `before_20260329_010530.png`)

Optional environment overrides:
- `PI_HOST` (default `flexyboard-pi.local`)
- `PI_USER` (default `pi`)
- `PI_REPO` (default `/home/pi/FlexyBoard-Camera`)
- `CONFIG` (default `configs/default.yaml`)
- `PI_PASSWORD` (optional; requires `sshpass` for non-interactive password entry)

Password automation example:

```bash
PI_PASSWORD=flexyboard ./scripts/capture_on_pi_and_pull.sh both
```

## Directory Layout

```text
FlexyBoard-Camera/
  configs/
    default.yaml
    calibration.example.json
  flexyboard_camera/
    app/
      main.py
      cli.py
      end_turn_controller.py
    camera/
      camera_manager.py
      capture.py
    vision/
      calibration.py
      preprocess.py
      diff_detector.py
      move_inference.py
      piece_classifier.py
    game/
      board_models.py
      move_models.py
      game_rules.py
    comms/
      serial_protocol.py
      stm32_client.py
    utils/
      config.py
      logging_utils.py
      paths.py
  scripts/
    generate_sample_images.py
    integration_harness.py
    camera_probe.py
    detect_board_corners.py
    detect_board_regions.py
    annotate_turn_geometry.py
    analyze_board_and_diff.py
    run_turn_capture_analyze.py
  tests/
    conftest.py
    test_calibration.py
    test_diff_and_inference.py
    test_serial_protocol.py
    test_end_turn_pipeline.py
  debug_output/
  logs/
  sample_data/
  requirements.txt
  pyproject.toml
```

## CLI Commands

- `calibrate`
- `capture_before [--image PATH]`
- `capture_after [--image PATH]`
- `infer_move --before PATH --after PATH`
- `send_move --sx N --sy N --dx N --dy N [--game chess|checkers|sorry]`
- `run_end_turn_cycle [--before PATH --after PATH --force]`
- `ping`
- `status`

## Behavior and Safety

- Confidence gate: if inferred move confidence is below config threshold, motion is blocked unless `--force` is set.
- Configurable retries for camera capture and serial command attempts.
- All inference attempts write debug artifacts:
  - raw images
  - processed images
  - diff and threshold images
  - changed-square JSON
  - inferred move JSON

## Config Notes

Primary runtime config: `configs/default.yaml`

Key settings:
- `app.game`
- `analysis.diff_threshold` / `analysis.min_changed_ratio` (main full-bridge changed-square sensitivity)
- `analysis.geometry_reference`
- `analysis.board_lock_source`
- `analysis.disable_tape_projection`
- `vision.roi`
- `vision.auto_detect_board` (if true, detects outer sheet + chessboard and diffs on warped chessboard only)
- `vision.warp_square_px`
- `vision.outer_sheet_hsv_lower` / `vision.outer_sheet_hsv_upper`
- `vision.outer_sheet_max_area_to_chessboard_ratio`
- `vision.fallback_outer_margins_squares`
- `vision.diff_threshold` / `vision.changed_square_pixel_ratio` (older controller/calibration CLI path)
- `app.confidence_threshold`
- `comms.port` (`mock://stm32` for dry run)
- `safety.auto_home_before_move`

Build/update `corners_info.json` from BEFORE captures:

```bash
python scripts/build_before_geometry_reference.py
```

### YOLO Piece Classifier (Optional)

Install optional YOLO dependency:

```bash
pip install -r requirements-yolo.txt
```

Set in `configs/default.yaml`:
- `classifier.enabled: true`
- `classifier.backend: yolo`
- `classifier.model_path: path/to/your_model.pt`

Important:
- A generic YOLO model will not reliably identify chess/checkers pieces from top-down board images.
- For robust results you need a model trained on your top-down board-camera dataset.

## Tests

If `pytest` is installed:

```bash
pytest
```

If `pytest` is not installed, basic syntax check:

```bash
python3 -m compileall flexyboard_camera scripts tests
```

## GPIO / Button Triggering

The MVP flow supports software triggers from CLI and saved-image replay. Hardware GPIO button integration can be added in `app/end_turn_controller.py` as a thin trigger source without changing inference/comms modules.

## Future Extension Hooks

- Replace `vision/piece_classifier.py` with YOLO/ML model inference
- Add game-state engine in `game/game_rules.py`
- Add binary serial mode alongside JSON in `comms/serial_protocol.py`
- Add RPi GPIO end-turn interrupt handler
