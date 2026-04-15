#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2


def probe_index(index: int, width: int, height: int) -> dict:
    cap = cv2.VideoCapture(index)
    if not cap or not cap.isOpened():
        return {"index": index, "opened": False}

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(width))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(height))

    frame = None
    for _ in range(4):
        ok, candidate = cap.read()
        if ok and candidate is not None:
            frame = candidate

    info = {
        "index": index,
        "opened": frame is not None,
        "actual_width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "actual_height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }
    cap.release()
    return info


def capture_snapshot(index: int, width: int, height: int, out_path: Path) -> bool:
    cap = cv2.VideoCapture(index)
    if not cap or not cap.isOpened():
        return False

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(width))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(height))

    frame = None
    for _ in range(6):
        ok, candidate = cap.read()
        if ok and candidate is not None:
            frame = candidate

    cap.release()
    if frame is None:
        return False

    out_path.parent.mkdir(parents=True, exist_ok=True)
    return bool(cv2.imwrite(str(out_path), frame))


def main() -> int:
    parser = argparse.ArgumentParser(description="Probe USB camera indices and optionally save a snapshot.")
    parser.add_argument("--max-index", type=int, default=5, help="Probe indices 0..max-index")
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument(
        "--snapshot-index",
        type=int,
        default=None,
        help="Capture a snapshot from this specific index (defaults to first opened index).",
    )
    parser.add_argument(
        "--snapshot-out",
        default="debug_output/camera_probe_snapshot.png",
        help="Path for saved snapshot",
    )
    args = parser.parse_args()

    results = [probe_index(i, args.width, args.height) for i in range(args.max_index + 1)]
    opened = [r for r in results if r.get("opened")]

    payload = {"probe": results, "opened_indices": [r["index"] for r in opened]}
    print(json.dumps(payload, indent=2))

    if not opened:
        return 1

    snapshot_index = args.snapshot_index if args.snapshot_index is not None else opened[0]["index"]
    snapshot_ok = capture_snapshot(snapshot_index, args.width, args.height, Path(args.snapshot_out))
    print(
        json.dumps(
            {
                "snapshot_index": snapshot_index,
                "snapshot_out": args.snapshot_out,
                "snapshot_saved": snapshot_ok,
            },
            indent=2,
        )
    )
    return 0 if snapshot_ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
