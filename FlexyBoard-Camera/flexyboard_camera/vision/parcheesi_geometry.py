from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
STORED_DATA_DIR = ROOT / "configs" / "all_manual" / "parcheesi" / "stored_data"
DEFAULT_TEMPLATE_PATH = STORED_DATA_DIR / "parcheesi_full_101_corner_distances_corrected.json"
DEFAULT_MAPPING_PATH = STORED_DATA_DIR / "parcheesi_location_mapping.json"
OUTER_NORM = np.float32(
    [
        [0.0, 0.0],
        [1.0, 0.0],
        [1.0, 1.0],
        [0.0, 1.0],
    ]
)


def _order_quad(points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32).reshape(4, 2)
    sums = pts.sum(axis=1)
    diffs = np.diff(pts, axis=1).reshape(-1)
    return np.array(
        [
            pts[np.argmin(sums)],
            pts[np.argmin(diffs)],
            pts[np.argmax(sums)],
            pts[np.argmax(diffs)],
        ],
        dtype=np.float32,
    )


@lru_cache(maxsize=1)
def _load_template() -> dict[str, Any]:
    return json.loads(DEFAULT_TEMPLATE_PATH.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def _load_mapping() -> dict[str, Any]:
    return json.loads(DEFAULT_MAPPING_PATH.read_text(encoding="utf-8"))


def parcheesi_layout_payload() -> dict[str, Any]:
    mapping = _load_mapping()
    template = _load_template()
    return {
        "template_file": DEFAULT_TEMPLATE_PATH.name,
        "mapping_file": DEFAULT_MAPPING_PATH.name,
        "region_count": len(template.get("regions", [])),
        "location_count": len(mapping.get("location_to_region", {})),
        "board_frame_definition": template.get("board_frame_definition"),
        "origin_corner": "S1_top_right",
    }


def _project_points(outer_corners_px: np.ndarray, points_normalized: np.ndarray) -> np.ndarray:
    transform = cv2.getPerspectiveTransform(OUTER_NORM, _order_quad(outer_corners_px))
    return cv2.perspectiveTransform(points_normalized.reshape(-1, 1, 2), transform).reshape(-1, 2)


def project_parcheesi_regions(outer_corners_px: np.ndarray) -> list[dict[str, Any]]:
    outer_quad = _order_quad(np.asarray(outer_corners_px, dtype=np.float32))
    mapping = _load_mapping()["location_to_region"]
    projected: list[dict[str, Any]] = []
    for location_id, entry in mapping.items():
        centroid_norm = np.asarray(entry["centroid_normalized"], dtype=np.float32).reshape(1, 2)
        polygon_norm = np.asarray(entry["polygon_board_normalized"], dtype=np.float32).reshape(-1, 2)
        centroid_px = _project_points(outer_quad, centroid_norm)[0]
        polygon_px = _project_points(outer_quad, polygon_norm)
        projected.append(
            {
                "location_id": location_id,
                "kind": str(entry["kind"]),
                "logical_player": int(entry.get("logical_player", 0)),
                "logical_pos": int(entry.get("logical_pos", 0)),
                "region_id": int(entry["region_id"]),
                "region_label": str(entry["region_label"]),
                "region_name": str(entry["region_name"]),
                "region_group": str(entry["region_group"]),
                "region_type": str(entry["region_type"]),
                "centroid_normalized": [float(entry["centroid_normalized"][0]), float(entry["centroid_normalized"][1])],
                "centroid_px": [float(centroid_px[0]), float(centroid_px[1])],
                "polygon_board_normalized": [[float(x), float(y)] for x, y in polygon_norm.tolist()],
                "polygon_px": [[float(x), float(y)] for x, y in polygon_px.tolist()],
            }
        )
    projected.sort(key=lambda item: int(item["region_id"]))
    return projected


def draw_parcheesi_overlay(
    image_bgr: np.ndarray,
    *,
    outer_corners_px: np.ndarray,
    projected_regions: list[dict[str, Any]],
    show_labels: bool = True,
    outer_thickness: int = 5,
    region_thickness: int = 2,
) -> np.ndarray:
    overlay = image_bgr.copy()
    outer_i = np.round(_order_quad(outer_corners_px)).astype(np.int32)
    cv2.polylines(overlay, [outer_i], True, (0, 255, 0), outer_thickness, cv2.LINE_AA)
    for idx, pt in enumerate(outer_i):
        cv2.circle(overlay, tuple(pt), 7, (0, 255, 0), -1, cv2.LINE_AA)
        cv2.putText(
            overlay,
            f"S{idx}",
            tuple(pt + np.array([8, -8])),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

    for item in projected_regions:
        poly = np.round(np.asarray(item["polygon_px"], dtype=np.float32)).astype(np.int32)
        cv2.polylines(overlay, [poly], True, (255, 0, 0), region_thickness, cv2.LINE_AA)
        if not show_labels:
            continue
        cx, cy = item["centroid_px"]
        cv2.putText(
            overlay,
            str(item["region_label"]),
            (int(round(cx)) - 10, int(round(cy)) + 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 0, 0),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            overlay,
            str(item["region_label"]),
            (int(round(cx)) - 10, int(round(cy)) + 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    label = "OUTER = GREEN | PARCHEESI REGIONS = BLUE"
    cv2.putText(overlay, label, (30, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.82, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(overlay, label, (30, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.82, (255, 255, 255), 2, cv2.LINE_AA)
    return overlay
