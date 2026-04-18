"""Runtime options for P2 checkers client."""

import os
import shutil

# "mock" or "tcp"
# Default to real Pi connection for normal hardware workflow.
TRANSPORT = os.environ.get("P2_TRANSPORT", "tcp").lower()

TCP_HOST = os.environ.get("P2_TCP_HOST", "flexyboard-pi.local")
TCP_PORT = int(os.environ.get("P2_TCP_PORT", "8765"))
TCP_CONNECT_RETRIES = int(os.environ.get("P2_TCP_CONNECT_RETRIES", "180"))
TCP_CONNECT_RETRY_DELAY_SEC = float(os.environ.get("P2_TCP_CONNECT_RETRY_DELAY_SEC", "1.0"))

WINDOW_WIDTH = int(os.environ.get("P2_WINDOW_W", "1040"))
WINDOW_HEIGHT = int(os.environ.get("P2_WINDOW_H", "760"))

# "human" (drag-drop in UI) or "ai" (auto-pick legal P2 move)
# Runtime can be toggled in the GUI via mode button.
CONTROL_MODE = os.environ.get("P2_CONTROL_MODE", "human").lower()
START_GAME = os.environ.get("P2_START_GAME", "").strip().lower()

# Write the motor move sequence for the latest P2 move.
WRITE_STM_SEQUENCE = os.environ.get("P2_WRITE_STM_SEQUENCE", "1").strip() not in {"0", "false", "False"}
STM_SEQUENCE_FILE = os.environ.get(
    "P2_STM_SEQUENCE_FILE",
    os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "FlexyBoard-Camera",
        "sample_data",
        "stm32_move_sequence.txt",
    ),
)

# Off-board capture staging area (percent of green grid).
# Mode "bottom_lane" stores captured pieces in the bottom strip between green/yellow regions.
# Mode "side_lane" preserves the older right-side vertical stack behavior.
CAPTURE_STAGING_MODE = os.environ.get("P2_CAPTURE_STAGING_MODE", "bottom_lane").strip().lower()

# Legacy side-lane settings (used when CAPTURE_STAGING_MODE=side_lane).
CAPTURE_STAGING_X_PCT = float(os.environ.get("P2_CAPTURE_STAGING_X_PCT", "92"))
CAPTURE_STAGING_P1_BASE_Y_PCT = float(os.environ.get("P2_CAPTURE_STAGING_P1_BASE_Y_PCT", "12"))
CAPTURE_STAGING_P2_BASE_Y_PCT = float(os.environ.get("P2_CAPTURE_STAGING_P2_BASE_Y_PCT", "78"))
CAPTURE_STAGING_STEP_Y_PCT = float(os.environ.get("P2_CAPTURE_STAGING_STEP_Y_PCT", "4"))

# Bottom-lane settings (used when CAPTURE_STAGING_MODE=bottom_lane).
# Defaults are aligned to your current calibrated green/yellow geometry:
# x ~ 19.7..77.1 in 8 slots, y rows in the bottom strip.
CAPTURE_BOTTOM_COLUMNS = int(os.environ.get("P2_CAPTURE_BOTTOM_COLUMNS", "8"))
CAPTURE_BOTTOM_X_START_PCT = float(os.environ.get("P2_CAPTURE_BOTTOM_X_START_PCT", "19.7"))
CAPTURE_BOTTOM_X_STEP_PCT = float(os.environ.get("P2_CAPTURE_BOTTOM_X_STEP_PCT", "8.2"))
CAPTURE_BOTTOM_P1_BASE_Y_PCT = float(os.environ.get("P2_CAPTURE_BOTTOM_P1_BASE_Y_PCT", "78"))
CAPTURE_BOTTOM_P2_BASE_Y_PCT = float(os.environ.get("P2_CAPTURE_BOTTOM_P2_BASE_Y_PCT", "88"))
CAPTURE_BOTTOM_ROW_STEP_Y_PCT = float(os.environ.get("P2_CAPTURE_BOTTOM_ROW_STEP_Y_PCT", "7.5"))
CAPTURE_BOTTOM_P2_REVERSE_X = (
    os.environ.get("P2_CAPTURE_BOTTOM_P2_REVERSE_X", "1").strip() not in {"0", "false", "False"}
)

# Temporary relocation staging area (used to clear blocker pieces, then restore them).
TEMP_RELOCATE_X_PCT = float(os.environ.get("P2_TEMP_RELOCATE_X_PCT", "8"))
TEMP_RELOCATE_BASE_Y_PCT = float(os.environ.get("P2_TEMP_RELOCATE_BASE_Y_PCT", "18"))
TEMP_RELOCATE_STEP_Y_PCT = float(os.environ.get("P2_TEMP_RELOCATE_STEP_Y_PCT", "4"))
MAX_TEMP_RELOCATIONS = int(os.environ.get("P2_MAX_TEMP_RELOCATIONS", "12"))
RESTORE_TEMP_RELOCATIONS = (
    os.environ.get("P2_RESTORE_TEMP_RELOCATIONS", "1").strip() not in {"0", "false", "False"}
)

# Chess promotion physical replacement flow (optional).
# When enabled, planner emits:
#   1) promotion destination square -> promotion staging bin
#   2) reserve promoted piece slot -> promotion destination square
# Keep disabled unless you have a real reserve-piece lane configured.
P2_PROMOTION_REPLACE_PHYSICAL = (
    os.environ.get("P2_PROMOTION_REPLACE_PHYSICAL", "0").strip() not in {"0", "false", "False"}
)
P2_PROMOTION_STAGING_X_PCT = float(os.environ.get("P2_PROMOTION_STAGING_X_PCT", "50"))
P2_PROMOTION_STAGING_Y_PCT = float(os.environ.get("P2_PROMOTION_STAGING_Y_PCT", "96"))
P2_PROMOTION_RESERVE_X_PCT = float(os.environ.get("P2_PROMOTION_RESERVE_X_PCT", "90"))
P2_PROMOTION_RESERVE_Y_PCT = float(os.environ.get("P2_PROMOTION_RESERVE_Y_PCT", "96"))
# If promotion piece is not found in captured-slot inventory, require manual replacement
# instead of automatically using reserve fallback coordinates.
P2_PROMOTION_REQUIRE_MANUAL_IF_MISSING = (
    os.environ.get("P2_PROMOTION_REQUIRE_MANUAL_IF_MISSING", "1").strip() not in {"0", "false", "False"}
)

# Debug overlay in GUI for captured-slot inventory visibility.
P2_SHOW_CAPTURE_INVENTORY_OVERLAY = (
    os.environ.get("P2_SHOW_CAPTURE_INVENTORY_OVERLAY", "1").strip() not in {"0", "false", "False"}
)
P2_SHOW_EMPTY_CAPTURE_INVENTORY_OVERLAY = (
    os.environ.get("P2_SHOW_EMPTY_CAPTURE_INVENTORY_OVERLAY", "0").strip() not in {"0", "false", "False"}
)
P2_CAPTURE_INVENTORY_OVERLAY_MAX_LINES = int(os.environ.get("P2_CAPTURE_INVENTORY_OVERLAY_MAX_LINES", "8"))

# Parcheesi percent-grid mapping (relative to green-grid workspace).
# Square IDs (a1..h8) are projected into this percent box.
P2_PARCHEESI_MIN_X_PCT = float(os.environ.get("P2_PARCHEESI_MIN_X_PCT", "18"))
P2_PARCHEESI_MAX_X_PCT = float(os.environ.get("P2_PARCHEESI_MAX_X_PCT", "78"))
P2_PARCHEESI_MIN_Y_PCT = float(os.environ.get("P2_PARCHEESI_MIN_Y_PCT", "18"))
P2_PARCHEESI_MAX_Y_PCT = float(os.environ.get("P2_PARCHEESI_MAX_Y_PCT", "78"))
P2_PARCHEESI_INVERT_X = (
    os.environ.get("P2_PARCHEESI_INVERT_X", "0").strip() not in {"0", "false", "False"}
)
P2_PARCHEESI_INVERT_Y = (
    os.environ.get("P2_PARCHEESI_INVERT_Y", "0").strip() not in {"0", "false", "False"}
)

# Chess AI configuration (Stockfish).
# If empty, the app attempts auto-discovery (PATH `stockfish`, then bundled Chess-Tracker exe).
STOCKFISH_PATH = os.environ.get("P2_STOCKFISH_PATH", "").strip()
STOCKFISH_MOVE_TIME_SEC = float(os.environ.get("P2_STOCKFISH_MOVE_TIME_SEC", "0.35"))
STOCKFISH_FALLBACK_TO_SIMPLE = (
    os.environ.get("P2_STOCKFISH_FALLBACK_TO_SIMPLE", "1").strip() not in {"0", "false", "False"}
)


def discover_stockfish_path() -> str | None:
    if STOCKFISH_PATH:
        return STOCKFISH_PATH

    which = shutil.which("stockfish")
    if which:
        return which

    candidate = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "Chess-Tracker",
        "content",
        "code",
        "Chess Tracker",
        "stockfish-windows-x86-64-avx2.exe",
    )
    if os.path.exists(candidate):
        return candidate

    return None
