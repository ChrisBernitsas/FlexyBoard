from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from flexyboard_camera.game.board_models import BoardCoord


@dataclass(slots=True)
class SquareChange:
    coord: BoardCoord
    pixel_ratio: float
    signed_intensity_delta: float
    detection_sources: tuple[str, ...] = ("square_ratio",)
    contour_area: float = 0.0
    contour_rank: int | None = None


@dataclass(slots=True)
class ContourSquareCandidate:
    coord: BoardCoord
    pixel_ratio: float
    signed_intensity_delta: float
    contour_area: float
    contour_rank: int


@dataclass(slots=True)
class DiffResult:
    diff_image: np.ndarray
    threshold_image: np.ndarray
    changes: list[SquareChange]
    contour_candidates: list[ContourSquareCandidate]


def detect_square_changes(
    before_img: np.ndarray,
    after_img: np.ndarray,
    board_size: tuple[int, int],
    diff_threshold: int,
    min_changed_ratio: float,
) -> DiffResult:
    diff = cv2.absdiff(before_img, after_img)
    diff_blurred = cv2.GaussianBlur(diff, (3, 3), 0)
    diff_boosted = cv2.convertScaleAbs(diff_blurred, alpha=1.3, beta=0)
    _, thresh = cv2.threshold(diff_boosted, diff_threshold, 255, cv2.THRESH_BINARY)
    square_thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    width, height = board_size
    img_h, img_w = square_thresh.shape
    square_w = img_w // width
    square_h = img_h // height
    square_area = max(1, square_w * square_h)

    changes_by_coord: dict[tuple[int, int], SquareChange] = {}
    signed = after_img.astype(np.int16) - before_img.astype(np.int16)

    def square_signed_mean(x: int, y: int) -> float:
        x0 = x * square_w
        y0 = y * square_h
        x1 = (x + 1) * square_w
        y1 = (y + 1) * square_h
        square_signed = signed[y0:y1, x0:x1]
        return float(np.mean(square_signed))

    def add_or_update_change(
        x: int,
        y: int,
        pixel_ratio: float,
        *,
        source: str = "square_ratio",
        contour_area: float = 0.0,
        contour_rank: int | None = None,
    ) -> None:
        if x < 0 or x >= width or y < 0 or y >= height:
            return
        signed_mean = square_signed_mean(x, y)
        key = (x, y)
        existing = changes_by_coord.get(key)
        if existing is None or pixel_ratio > existing.pixel_ratio:
            sources = (source,) if existing is None else tuple(sorted({*existing.detection_sources, source}))
            changes_by_coord[key] = SquareChange(
                coord=BoardCoord(x=x, y=y),
                pixel_ratio=pixel_ratio,
                signed_intensity_delta=signed_mean,
                detection_sources=sources,
                contour_area=max(contour_area, existing.contour_area if existing is not None else 0.0),
                contour_rank=contour_rank if contour_rank is not None else (existing.contour_rank if existing is not None else None),
            )
        elif source not in existing.detection_sources:
            existing.detection_sources = tuple(sorted({*existing.detection_sources, source}))
            existing.contour_area = max(existing.contour_area, contour_area)
            if existing.contour_rank is None or (contour_rank is not None and contour_rank < existing.contour_rank):
                existing.contour_rank = contour_rank

    for y in range(height):
        for x in range(width):
            x0 = x * square_w
            y0 = y * square_h
            x1 = (x + 1) * square_w
            y1 = (y + 1) * square_h

            square_mask = square_thresh[y0:y1, x0:x1]
            active = float(np.count_nonzero(square_mask))
            total = float(square_mask.size)
            ratio = active / total if total else 0.0
            if ratio < min_changed_ratio:
                continue

            add_or_update_change(x, y, ratio)

    # Chess-Tracker-style assist: find changed blobs and map each blob back to the
    # square it occupies. This catches weak destination changes, such as a light
    # piece moving onto a light square, that may not cover enough of the whole square.
    contour_mask = cv2.dilate(thresh, None, iterations=4)
    contour_mask = cv2.erode(contour_mask, None, iterations=2)
    kernel = np.ones((3, 3), np.uint8)
    contour_mask = cv2.morphologyEx(contour_mask, cv2.MORPH_OPEN, kernel)
    contour_mask = cv2.morphologyEx(contour_mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(contour_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_contour_area = max(60.0, square_area * 0.02)
    contour_by_coord: dict[tuple[int, int], tuple[float, float, float]] = {}
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < min_contour_area:
            continue

        x, y, w, h = cv2.boundingRect(contour)
        moments = cv2.moments(contour)
        if moments["m00"]:
            cx = int(moments["m10"] / moments["m00"])
            cy = int(moments["m01"] / moments["m00"])
        else:
            cx = x + w // 2
            cy = y + h // 2

        probes = [
            (cx, cy),
            (cx, int(y + 0.20 * h)),
            (cx, int(y + 0.30 * h)),
            (cx, int(y + 0.40 * h)),
            (int(x + 0.35 * w), cy),
            (int(x + 0.65 * w), cy),
        ]
        votes: dict[tuple[int, int], int] = {}
        for px, py in probes:
            sx = int(px // square_w)
            sy = int(py // square_h)
            if 0 <= sx < width and 0 <= sy < height:
                votes[(sx, sy)] = votes.get((sx, sy), 0) + 1
        if not votes:
            continue

        best_x, best_y = max(votes.items(), key=lambda item: item[1])[0]
        x0 = best_x * square_w
        y0 = best_y * square_h
        x1 = (best_x + 1) * square_w
        y1 = (best_y + 1) * square_h
        local_ratio = float(np.count_nonzero(square_thresh[y0:y1, x0:x1])) / float(square_area)
        if local_ratio >= max(0.015, min_changed_ratio * 0.35):
            key = (best_x, best_y)
            signed_mean = square_signed_mean(best_x, best_y)
            existing = contour_by_coord.get(key)
            if existing is None or area > existing[0]:
                contour_by_coord[key] = (area, local_ratio, signed_mean)

    contour_candidates: list[ContourSquareCandidate] = []
    for rank, ((x, y), (area, local_ratio, signed_mean)) in enumerate(
        sorted(contour_by_coord.items(), key=lambda item: item[1][0], reverse=True),
        start=1,
    ):
        candidate = ContourSquareCandidate(
            coord=BoardCoord(x=x, y=y),
            pixel_ratio=local_ratio,
            signed_intensity_delta=signed_mean,
            contour_area=area,
            contour_rank=rank,
        )
        contour_candidates.append(candidate)
        if rank <= 2:
            # Use only the two strongest coherent blobs as Chess-Tracker-style
            # movement candidates. Lower-ranked contours stay in debug output
            # but do not pollute the default changed-square set.
            add_or_update_change(
                x,
                y,
                local_ratio,
                source="contour_top2",
                contour_area=area,
                contour_rank=rank,
            )

    changes = list(changes_by_coord.values())
    changes.sort(key=lambda item: item.pixel_ratio, reverse=True)
    return DiffResult(
        diff_image=diff_boosted,
        threshold_image=cv2.bitwise_or(square_thresh, contour_mask),
        changes=changes,
        contour_candidates=contour_candidates,
    )
