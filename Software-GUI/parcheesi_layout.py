from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
STORED_DATA_DIR = ROOT / "FlexyBoard-Camera" / "configs" / "all_manual" / "parcheesi" / "stored_data"
MAPPING_PATH = STORED_DATA_DIR / "parcheesi_location_mapping.json"

_TOKEN_OFFSETS = {
    1: (-0.020, -0.020),
    2: (0.020, -0.020),
    3: (-0.020, 0.020),
    4: (0.020, 0.020),
}


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


@lru_cache(maxsize=1)
def _load_mapping() -> dict[str, Any]:
    return json.loads(MAPPING_PATH.read_text(encoding="utf-8"))


def _parse_token_suffix(location_id: str) -> tuple[str, int | None]:
    parts = location_id.strip().lower().split("_")
    if len(parts) == 3 and parts[0] in {"nest", "homearea"}:
        try:
            token = int(parts[2])
        except ValueError:
            return location_id, None
        return f"{parts[0]}_{parts[1]}", token
    return location_id, None


def normalized_point_for_location(location_id: str) -> tuple[float, float]:
    mapping = _load_mapping()["location_to_region"]
    base_location, token = _parse_token_suffix(location_id)
    if base_location.startswith("homearea_"):
        entry = mapping["home_center"]
    else:
        entry = mapping[base_location]
    x_norm, y_norm = (float(entry["centroid_normalized"][0]), float(entry["centroid_normalized"][1]))
    if token is not None:
        dx, dy = _TOKEN_OFFSETS.get(token, (0.0, 0.0))
        x_norm = _clamp01(x_norm + dx)
        y_norm = _clamp01(y_norm + dy)
    return (x_norm, y_norm)


def location_to_pct(location_id: str) -> tuple[float, float]:
    x_norm, y_norm = normalized_point_for_location(location_id)
    # STM percent space uses S1/top-right as the origin: x=0 on the right edge, y=0 on the top edge.
    return ((1.0 - x_norm) * 100.0, y_norm * 100.0)
