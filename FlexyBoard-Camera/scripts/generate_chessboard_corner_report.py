#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sorted_move_dirs(game_dir: Path) -> list[Path]:
    def key(path: Path) -> tuple[int, str]:
        suffix = path.name.removeprefix("Move")
        try:
            return (int(suffix), path.name)
        except ValueError:
            return (10**9, path.name)

    return sorted(
        [path for path in game_dir.iterdir() if path.is_dir() and path.name.startswith("Move")],
        key=key,
    )


def _quad_array(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    try:
        quad = np.array(value, dtype=np.float32).reshape(4, 2)
    except Exception:  # noqa: BLE001
        return None
    return quad


def _crop_shift(quad: np.ndarray | None, crop_meta: dict[str, Any], *, to_raw: bool) -> np.ndarray | None:
    if quad is None:
        return None
    x0 = float(crop_meta.get("x0_px", 0.0))
    y0 = float(crop_meta.get("y0_px", 0.0))
    shifted = quad.astype(np.float32).copy()
    if to_raw:
        shifted[:, 0] += x0
        shifted[:, 1] += y0
    else:
        shifted[:, 0] -= x0
        shifted[:, 1] -= y0
    return shifted


def _corner_errors(a: np.ndarray | None, b: np.ndarray | None) -> dict[str, Any] | None:
    if a is None or b is None:
        return None
    dists = np.linalg.norm(a.astype(np.float32) - b.astype(np.float32), axis=1)
    return {
        "per_corner_px": [float(x) for x in dists.tolist()],
        "mean_px": float(np.mean(dists)),
        "max_px": float(np.max(dists)),
        "rmse_px": float(math.sqrt(float(np.mean(np.square(dists))))),
    }


def _draw_quad(
    image: np.ndarray,
    quad: np.ndarray | None,
    *,
    line_color: tuple[int, int, int],
    point_color: tuple[int, int, int],
    label_prefix: str,
) -> None:
    if quad is None:
        return
    poly = quad.astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(image, [poly], True, line_color, 3, cv2.LINE_AA)
    for idx, pt in enumerate(quad):
        x = int(round(float(pt[0])))
        y = int(round(float(pt[1])))
        cv2.circle(image, (x, y), 7, point_color, -1, cv2.LINE_AA)
        cv2.putText(
            image,
            f"{label_prefix}{idx}",
            (x + 8, y - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            line_color,
            2,
            cv2.LINE_AA,
        )


def _draw_legend(image: np.ndarray) -> None:
    cv2.rectangle(image, (16, 16), (350, 90), (245, 245, 245), -1, cv2.LINE_AA)
    cv2.rectangle(image, (16, 16), (350, 90), (40, 40, 40), 2, cv2.LINE_AA)
    cv2.line(image, (32, 40), (92, 40), (0, 220, 0), 3, cv2.LINE_AA)
    cv2.circle(image, (62, 40), 5, (0, 255, 255), -1, cv2.LINE_AA)
    cv2.putText(
        image,
        "Manual chessboard corners",
        (108, 46),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (30, 30, 30),
        2,
        cv2.LINE_AA,
    )
    cv2.line(image, (32, 68), (92, 68), (0, 255, 255), 3, cv2.LINE_AA)
    cv2.circle(image, (62, 68), 5, (0, 0, 255), -1, cv2.LINE_AA)
    cv2.putText(
        image,
        "Auto/detected chessboard corners",
        (108, 74),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (30, 30, 30),
        2,
        cv2.LINE_AA,
    )


def _make_overlay(
    *,
    raw_image_path: Path,
    manual_raw_quad: np.ndarray | None,
    auto_raw_quad: np.ndarray | None,
    out_path: Path,
) -> None:
    frame = cv2.imread(str(raw_image_path))
    if frame is None:
        return
    vis = frame.copy()
    _draw_quad(
        vis,
        manual_raw_quad,
        line_color=(0, 220, 0),
        point_color=(0, 255, 255),
        label_prefix="M",
    )
    _draw_quad(
        vis,
        auto_raw_quad,
        line_color=(0, 255, 255),
        point_color=(0, 0, 255),
        label_prefix="A",
    )
    _draw_legend(vis)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), vis)


def generate_report(game_dir: Path) -> Path:
    manual_payload = _load_json(game_dir / "corners_info.json")
    manual_raw_quad = _quad_array(manual_payload.get("chessboard_corners_px"))
    if manual_raw_quad is None:
        raise SystemExit(f"Missing chessboard_corners_px in {game_dir / 'corners_info.json'}")

    report_entries: list[dict[str, Any]] = []
    before_means: list[float] = []
    after_means: list[float] = []

    for move_dir in _sorted_move_dirs(game_dir):
        analysis_path = move_dir / "analysis.json"
        if not analysis_path.exists():
            continue
        analysis = _load_json(analysis_path)
        crop_info = analysis.get("pre_detection_crop") or {}
        before_crop = crop_info.get("before") or {}
        after_crop = crop_info.get("after") or {}

        manual_before_cropped = _crop_shift(manual_raw_quad, before_crop, to_raw=False)
        manual_after_cropped = _crop_shift(manual_raw_quad, after_crop, to_raw=False)

        before_auto_cropped = _quad_array(
            analysis.get("algorithm_live_before_detected_chessboard_corners_px")
            or analysis.get("before_chessboard_corners_px")
        )
        after_auto_cropped = _quad_array(
            analysis.get("algorithm_live_after_detected_chessboard_corners_px")
            or analysis.get("after_chessboard_corners_px")
        )

        before_errors = _corner_errors(before_auto_cropped, manual_before_cropped)
        after_errors = _corner_errors(after_auto_cropped, manual_after_cropped)
        if before_errors is not None:
            before_means.append(float(before_errors["mean_px"]))
        if after_errors is not None:
            after_means.append(float(after_errors["mean_px"]))

        before_auto_raw = _crop_shift(before_auto_cropped, before_crop, to_raw=True)
        after_auto_raw = _crop_shift(after_auto_cropped, after_crop, to_raw=True)

        algorithm_live_dir = move_dir / "algorithm_live"
        _make_overlay(
            raw_image_path=move_dir / "before.png",
            manual_raw_quad=manual_raw_quad,
            auto_raw_quad=before_auto_raw,
            out_path=algorithm_live_dir / "before_chessboard_compare_manual_vs_detected.png",
        )
        _make_overlay(
            raw_image_path=move_dir / "after.png",
            manual_raw_quad=manual_raw_quad,
            auto_raw_quad=after_auto_raw,
            out_path=algorithm_live_dir / "after_chessboard_compare_manual_vs_detected.png",
        )

        report_entries.append(
            {
                "move": move_dir.name,
                "compared_against": "manual_startup_chessboard_corners_px",
                "manual_raw_chessboard_corners_px": manual_raw_quad.tolist(),
                "manual_before_cropped_chessboard_corners_px": (
                    manual_before_cropped.tolist() if manual_before_cropped is not None else None
                ),
                "manual_after_cropped_chessboard_corners_px": (
                    manual_after_cropped.tolist() if manual_after_cropped is not None else None
                ),
                "auto_before_detected_chessboard_corners_px": (
                    before_auto_cropped.tolist() if before_auto_cropped is not None else None
                ),
                "auto_after_detected_chessboard_corners_px": (
                    after_auto_cropped.tolist() if after_auto_cropped is not None else None
                ),
                "before_error_vs_manual_cropped": before_errors,
                "after_error_vs_manual_cropped": after_errors,
                "outputs": {
                    "before_compare_overlay": str(
                        algorithm_live_dir / "before_chessboard_compare_manual_vs_detected.png"
                    ),
                    "after_compare_overlay": str(
                        algorithm_live_dir / "after_chessboard_compare_manual_vs_detected.png"
                    ),
                },
            }
        )

    summary = {
        "game": game_dir.name,
        "manual_reference_path": str(game_dir / "corners_info.json"),
        "moves_analyzed": len(report_entries),
        "mean_before_error_px_across_moves": (
            float(np.mean(before_means)) if before_means else None
        ),
        "mean_after_error_px_across_moves": (
            float(np.mean(after_means)) if after_means else None
        ),
        "max_before_error_px_across_moves": (
            float(np.max(before_means)) if before_means else None
        ),
        "max_after_error_px_across_moves": (
            float(np.max(after_means)) if after_means else None
        ),
    }

    report = {
        "summary": summary,
        "moves": report_entries,
    }
    out_path = game_dir / "chessboard_corner_alignment_report.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate chessboard-corner alignment report and overlays.")
    parser.add_argument(
        "--game-dir",
        type=Path,
        required=True,
        help="Path to a debug_output/Games/GameN directory.",
    )
    args = parser.parse_args()
    out = generate_report(args.game_dir.resolve())
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
