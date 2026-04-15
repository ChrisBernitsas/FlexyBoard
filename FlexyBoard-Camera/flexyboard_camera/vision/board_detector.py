from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(slots=True)
class BoardDetection:
    outer_sheet_corners: np.ndarray | None  # shape: [4, 2], TL/TR/BR/BL
    chessboard_corners: np.ndarray | None  # shape: [4, 2], TL/TR/BR/BL


@dataclass(frozen=True, slots=True)
class BoardSquareGeometry:
    index: int
    x: int
    y: int
    label: str
    corners_px: np.ndarray  # shape: [4, 2], TL/TR/BR/BL
    center_px: np.ndarray  # shape: [2]

    def to_dict(self) -> dict[str, object]:
        return {
            "index": self.index,
            "x": self.x,
            "y": self.y,
            "label": self.label,
            "corners_px": [[float(x), float(y)] for x, y in self.corners_px.tolist()],
            "center_px": [float(self.center_px[0]), float(self.center_px[1])],
        }


def _order_quad(points: np.ndarray) -> np.ndarray:
    pts = points.astype(np.float32).reshape(-1, 2)
    if pts.shape[0] != 4:
        raise ValueError("Quadrilateral must have exactly 4 points")

    sums = pts.sum(axis=1)
    diffs = np.diff(pts, axis=1).reshape(-1)

    ordered = np.zeros((4, 2), dtype=np.float32)
    ordered[0] = pts[np.argmin(sums)]  # TL
    ordered[2] = pts[np.argmax(sums)]  # BR
    ordered[1] = pts[np.argmin(diffs)]  # TR
    ordered[3] = pts[np.argmax(diffs)]  # BL
    return ordered


def _quad_area(quad: np.ndarray) -> float:
    pts = quad.astype(np.float32).reshape(-1, 2)
    return abs(float(cv2.contourArea(pts.reshape(-1, 1, 2))))


def _quad_aspect_ratio(quad: np.ndarray) -> float:
    pts = quad.astype(np.float32).reshape(-1, 2)
    top = float(np.linalg.norm(pts[1] - pts[0]))
    right = float(np.linalg.norm(pts[2] - pts[1]))
    bottom = float(np.linalg.norm(pts[2] - pts[3]))
    left = float(np.linalg.norm(pts[3] - pts[0]))
    width = (top + bottom) * 0.5
    height = (left + right) * 0.5
    if height <= 1e-6 or width <= 1e-6:
        return 1.0
    ratio = width / height
    if ratio < 1.0:
        ratio = 1.0 / ratio
    return ratio


def _quad_encloses_points(quad: np.ndarray, points: np.ndarray) -> bool:
    contour = quad.astype(np.float32).reshape(-1, 1, 2)
    for p in points.astype(np.float32).reshape(-1, 2):
        inside = cv2.pointPolygonTest(contour, (float(p[0]), float(p[1])), False)
        if inside < 0:
            return False
    return True


def _collect_outer_candidates_from_mask(
    *,
    mask: np.ndarray,
    frame_shape: tuple[int, int, int],
    min_area_ratio: float,
    border_margin_px: int,
    must_enclose_quad: np.ndarray | None,
    enclosed_area: float | None,
    max_area_to_enclosed_quad_ratio: float,
) -> list[tuple[float, np.ndarray]]:
    h, w = frame_shape[:2]
    frame_area = float(h * w)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates: list[tuple[float, np.ndarray]] = []
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if frame_area <= 0.0 or area / frame_area < min_area_ratio:
            continue

        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        if len(approx) == 4:
            quad = _order_quad(approx.reshape(4, 2))
        else:
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            quad = _order_quad(box)

        # Guard against highly inflated min-area rectangles for irregular blobs:
        # fallback to axis-aligned bounds before rejecting.
        quad_area = _quad_area(quad)
        if area > 1e-6 and (quad_area / area) > 3.2:
            bx, by, bw2, bh2 = cv2.boundingRect(contour)
            quad = _order_quad(
                np.array(
                    [
                        [float(bx), float(by)],
                        [float(bx + bw2), float(by)],
                        [float(bx + bw2), float(by + bh2)],
                        [float(bx), float(by + bh2)],
                    ],
                    dtype=np.float32,
                )
            )
            quad_area = _quad_area(quad)
            if area > 1e-6 and (quad_area / area) > 6.0:
                continue

        x, y, bw, bh = cv2.boundingRect(quad.astype(np.float32).reshape(-1, 1, 2))
        touches = 0
        if x <= border_margin_px:
            touches += 1
        if y <= border_margin_px:
            touches += 1
        if (x + bw) >= (w - border_margin_px):
            touches += 1
        if (y + bh) >= (h - border_margin_px):
            touches += 1
        # Relax border-touch rejection:
        # - allow candidates touching up to 3 borders (framing can be tight / partially cropped),
        # - only reject pathological full-frame boxes touching all borders.
        if touches >= 4:
            continue
        if touches >= 3 and bw >= int(0.98 * w) and bh >= int(0.98 * h):
            continue

        aspect = _quad_aspect_ratio(quad)
        if aspect > 1.8:
            continue

        if must_enclose_quad is not None:
            if not _quad_encloses_points(quad, must_enclose_quad):
                continue
            if enclosed_area is not None and enclosed_area > 0.0:
                if area / enclosed_area > max_area_to_enclosed_quad_ratio:
                    continue

        candidates.append((area, quad))

    return candidates


def _collect_inner_hole_candidates_from_ring_mask(
    *,
    mask: np.ndarray,
    frame_shape: tuple[int, int, int],
    min_area_ratio: float,
    border_margin_px: int,
    must_enclose_quad: np.ndarray | None,
    enclosed_area: float | None,
    max_area_to_enclosed_quad_ratio: float,
) -> list[tuple[float, np.ndarray]]:
    """
    Detect inner-hole quads from a ring-like mask (white ring on black background).

    For dark tape, the desired "green" board boundary is often the *inside* edge of that
    white ring. This helper extracts those inner holes as candidate quads.
    """
    h, w = frame_shape[:2]
    frame_area = float(max(1, h * w))
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None or len(contours) == 0:
        return []

    # OpenCV returns shape [1, N, 4] where entries are [next, prev, first_child, parent].
    hinfo = hierarchy[0]
    candidates: list[tuple[float, np.ndarray]] = []
    for idx, contour in enumerate(contours):
        parent_idx = int(hinfo[idx][3])
        # Hole contours are children of white regions.
        if parent_idx < 0:
            continue

        area = abs(float(cv2.contourArea(contour)))
        if area / frame_area < min_area_ratio:
            continue

        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        if len(approx) == 4:
            quad = _order_quad(approx.reshape(4, 2))
        else:
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            quad = _order_quad(box)

        # Guard against highly inflated min-area rectangles for irregular hole shapes:
        # fallback to axis-aligned bounds before rejecting.
        quad_area = _quad_area(quad)
        if area > 1e-6 and (quad_area / area) > 3.2:
            bx, by, bw2, bh2 = cv2.boundingRect(contour)
            quad = _order_quad(
                np.array(
                    [
                        [float(bx), float(by)],
                        [float(bx + bw2), float(by)],
                        [float(bx + bw2), float(by + bh2)],
                        [float(bx), float(by + bh2)],
                    ],
                    dtype=np.float32,
                )
            )
            quad_area = _quad_area(quad)
            if area > 1e-6 and (quad_area / area) > 6.0:
                continue

        x, y, bw, bh = cv2.boundingRect(quad.astype(np.float32).reshape(-1, 1, 2))
        touches = 0
        if x <= border_margin_px:
            touches += 1
        if y <= border_margin_px:
            touches += 1
        if (x + bw) >= (w - border_margin_px):
            touches += 1
        if (y + bh) >= (h - border_margin_px):
            touches += 1
        # Inner-hole candidate should not cling to image border.
        if touches > 0:
            continue

        aspect = _quad_aspect_ratio(quad)
        if aspect > 1.8:
            continue

        if must_enclose_quad is not None:
            if not _quad_encloses_points(quad, must_enclose_quad):
                continue
            if enclosed_area is not None and enclosed_area > 0.0:
                area_ratio = area / enclosed_area
                if area_ratio > max_area_to_enclosed_quad_ratio:
                    continue
                # Inner ring boundary should usually be at least as large as the chessboard.
                if area_ratio < 0.95:
                    continue

        candidates.append((area, quad))

    return candidates


def _draw_outer_candidate_overlay(
    frame_bgr: np.ndarray,
    candidates: list[tuple[float, np.ndarray]],
    selected: np.ndarray | None,
) -> np.ndarray:
    vis = frame_bgr.copy()
    for idx, (area, quad) in enumerate(candidates):
        q = quad.astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(vis, [q], True, (255, 128, 0), 2, cv2.LINE_AA)
        anchor = quad.astype(np.int32).reshape(-1, 2)[0]
        cv2.putText(
            vis,
            f"C{idx} A={int(area)}",
            (int(anchor[0]) + 6, int(anchor[1]) - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 128, 0),
            1,
            cv2.LINE_AA,
        )

    if selected is not None:
        s = selected.astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(vis, [s], True, (0, 255, 0), 3, cv2.LINE_AA)
        for i, pt in enumerate(selected.astype(np.int32).reshape(-1, 2)):
            cv2.putText(
                vis,
                f"S{i}",
                (int(pt[0]) + 6, int(pt[1]) - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
    return vis


def _line_support_ratio(mask: np.ndarray, p0: np.ndarray, p1: np.ndarray, *, band_px: int) -> float:
    h, w = mask.shape[:2]
    line_mask = np.zeros((h, w), dtype=np.uint8)
    a = (int(round(float(p0[0]))), int(round(float(p0[1]))))
    b = (int(round(float(p1[0]))), int(round(float(p1[1]))))
    thickness = max(3, int(band_px))
    cv2.line(line_mask, a, b, 255, thickness=thickness, lineType=cv2.LINE_AA)
    covered = cv2.countNonZero(line_mask)
    if covered <= 0:
        return 0.0
    overlap = cv2.countNonZero(cv2.bitwise_and(mask, line_mask))
    return float(overlap) / float(covered)


def _quad_side_support(mask: np.ndarray, quad: np.ndarray, *, band_px: int) -> list[float]:
    q = quad.astype(np.float32).reshape(4, 2)
    ratios: list[float] = []
    for i in range(4):
        p0 = q[i]
        p1 = q[(i + 1) % 4]
        ratios.append(_line_support_ratio(mask, p0, p1, band_px=band_px))
    return ratios


def detect_outer_sheet(
    frame_bgr: np.ndarray,
    hsv_lower: tuple[int, int, int],
    hsv_upper: tuple[int, int, int],
    min_area_ratio: float = 0.1,
    must_enclose_quad: np.ndarray | None = None,
    max_area_to_enclosed_quad_ratio: float = 5.0,
    expected_area_to_enclosed_ratio: float | None = None,
    outer_candidate_mode: str = "auto",
    debug: dict[str, object] | None = None,
) -> np.ndarray | None:
    # Kept in signature for API compatibility; detection is dark-mask-only.
    _ = hsv_lower
    _ = hsv_upper

    h, w = frame_bgr.shape[:2]
    frame_area = float(max(1, h * w))
    border_margin_px = max(8, int(min(h, w) * 0.01))
    enclosed_area = _quad_area(must_enclose_quad) if must_enclose_quad is not None else None

    kernel = np.ones((5, 5), dtype=np.uint8)
    # Dark-mask primary path for outer boundary detection.
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, dark_mask_raw = cv2.threshold(blur, 95, 255, cv2.THRESH_BINARY_INV)
    dark_mask = cv2.morphologyEx(dark_mask_raw, cv2.MORPH_CLOSE, kernel, iterations=2)
    dark_mask = cv2.dilate(dark_mask, np.ones((3, 3), dtype=np.uint8), iterations=1)
    dark_mask_on_ratio = float(cv2.countNonZero(dark_mask)) / frame_area

    # Ring emphasis from dark mask only.
    tape_ring_mask = dark_mask.copy()
    tape_ring_mask = cv2.morphologyEx(tape_ring_mask, cv2.MORPH_CLOSE, np.ones((7, 7), dtype=np.uint8), iterations=2)
    tape_ring_mask = cv2.morphologyEx(tape_ring_mask, cv2.MORPH_OPEN, np.ones((3, 3), dtype=np.uint8), iterations=1)
    tape_ring_candidates = _collect_outer_candidates_from_mask(
        mask=tape_ring_mask,
        frame_shape=frame_bgr.shape,
        min_area_ratio=max(0.04, min_area_ratio * 0.45),
        border_margin_px=border_margin_px,
        must_enclose_quad=must_enclose_quad,
        enclosed_area=enclosed_area,
        max_area_to_enclosed_quad_ratio=max_area_to_enclosed_quad_ratio,
    )
    # Preferred path: use the *inside edge* (hole) of the dark-derived tape ring.
    tape_ring_inner_candidates = _collect_inner_hole_candidates_from_ring_mask(
        mask=tape_ring_mask,
        frame_shape=frame_bgr.shape,
        min_area_ratio=max(0.04, min_area_ratio * 0.45),
        border_margin_px=border_margin_px,
        must_enclose_quad=must_enclose_quad,
        enclosed_area=enclosed_area,
        max_area_to_enclosed_quad_ratio=max_area_to_enclosed_quad_ratio,
    )

    # For outer-ring detection, suppress interior chessboard region when available.
    # This biases dark-mask candidates toward the duct-tape boundary instead of inner squares.
    dark_mask_candidates = dark_mask.copy()
    if must_enclose_quad is not None:
        inner_mask = np.zeros_like(dark_mask_candidates)
        cv2.fillPoly(
            inner_mask,
            [must_enclose_quad.astype(np.int32).reshape(-1, 1, 2)],
            255,
        )
        inner_mask = cv2.dilate(inner_mask, np.ones((15, 15), dtype=np.uint8), iterations=1)
        dark_mask_candidates[inner_mask > 0] = 0

    dark_candidates = _collect_outer_candidates_from_mask(
        mask=dark_mask_candidates,
        frame_shape=frame_bgr.shape,
        min_area_ratio=min_area_ratio,
        border_margin_px=border_margin_px,
        must_enclose_quad=must_enclose_quad,
        enclosed_area=enclosed_area,
        max_area_to_enclosed_quad_ratio=max_area_to_enclosed_quad_ratio,
    )

    mode = outer_candidate_mode.strip().lower()
    if mode not in {"auto", "hsv_only", "dark_only"}:
        mode = "auto"
    # Legacy compatibility: hsv_only now aliases dark-only behavior.
    mode_effective = "dark_only" if mode == "hsv_only" else mode

    selected_candidate_source = "none"
    if mode_effective == "dark_only":
        if tape_ring_inner_candidates:
            candidates = tape_ring_inner_candidates
            selected_candidate_source = "dark_tape_inner_ring"
        elif tape_ring_candidates:
            candidates = tape_ring_candidates
            selected_candidate_source = "dark_hsv_tape_ring"
        else:
            candidates = dark_candidates
            selected_candidate_source = "dark"
    else:
        if tape_ring_inner_candidates:
            candidates = tape_ring_inner_candidates
            selected_candidate_source = "dark_tape_inner_ring"
        elif tape_ring_candidates:
            candidates = tape_ring_candidates
            selected_candidate_source = "dark_hsv_tape_ring"
        else:
            candidates = dark_candidates
            selected_candidate_source = "dark"

    if not candidates and mode_effective == "dark_only":
        h, w = frame_bgr.shape[:2]
        frame_area = float(max(1, h * w))
        contours, _ = cv2.findContours(dark_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        largest_contour = None
        largest_area = 0.0
        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area / frame_area < min_area_ratio:
                continue
            if area > largest_area:
                largest_area = area
                largest_contour = contour
        if largest_contour is not None:
            rect = cv2.minAreaRect(largest_contour)
            quad = _order_quad(cv2.boxPoints(rect))
            candidates = [(largest_area, quad)]
            selected_candidate_source = "dark_fallback_largest_contour"

    if not candidates:
        if debug is not None:
            debug["dark_mask_raw"] = dark_mask_raw
            debug["dark_mask"] = dark_mask
            debug["tape_ring_mask"] = tape_ring_mask
            debug["dark_mask_candidates"] = dark_mask_candidates
            debug["dark_mask_on_ratio"] = dark_mask_on_ratio
            debug["dark_candidate_count"] = len(dark_candidates)
            debug["tape_ring_candidate_count"] = len(tape_ring_candidates)
            debug["tape_ring_inner_candidate_count"] = len(tape_ring_inner_candidates)
            debug["outer_candidate_mode"] = mode
            debug["outer_candidate_mode_effective"] = mode_effective
            debug["outer_candidate_source"] = "none"
            debug["candidate_overlay"] = frame_bgr.copy()
        return None

    if debug is not None:
        debug["dark_mask_raw"] = dark_mask_raw
        debug["dark_mask"] = dark_mask
        debug["tape_ring_mask"] = tape_ring_mask
        debug["dark_mask_candidates"] = dark_mask_candidates
        debug["dark_mask_on_ratio"] = dark_mask_on_ratio
        debug["dark_candidate_count"] = len(dark_candidates)
        debug["tape_ring_candidate_count"] = len(tape_ring_candidates)
        debug["tape_ring_inner_candidate_count"] = len(tape_ring_inner_candidates)
        debug["outer_candidate_mode"] = mode
        debug["outer_candidate_mode_effective"] = mode_effective
        debug["outer_candidate_source"] = selected_candidate_source

    selected: np.ndarray
    if must_enclose_quad is not None:
        if enclosed_area is not None and enclosed_area > 0.0 and expected_area_to_enclosed_ratio is not None:
            # Prefer enclosure closest to expected area ratio.
            candidates.sort(
                key=lambda pair: (
                    abs((pair[0] / enclosed_area) - expected_area_to_enclosed_ratio),
                    pair[0],
                )
            )
        else:
            # Otherwise prefer the smallest valid enclosure around the board.
            candidates.sort(key=lambda pair: pair[0])
    else:
        # No chessboard enclosure available:
        # rank candidates by a likelihood score that still favors large regions,
        # but penalizes border-hugging / full-span / skewed boxes.
        def _candidate_likelihood(pair: tuple[float, np.ndarray]) -> tuple[float, float]:
            area, quad = pair
            x, y, bw, bh = cv2.boundingRect(quad.astype(np.float32).reshape(-1, 1, 2))
            touches = 0
            if x <= border_margin_px:
                touches += 1
            if y <= border_margin_px:
                touches += 1
            if (x + bw) >= (w - border_margin_px):
                touches += 1
            if (y + bh) >= (h - border_margin_px):
                touches += 1

            aspect = _quad_aspect_ratio(quad)
            touch_penalty = 0.16 * float(touches)
            span_penalty = 0.0
            if bw >= int(0.90 * w):
                span_penalty += 0.30
            if bh >= int(0.90 * h):
                span_penalty += 0.30
            aspect_penalty = min(abs(aspect - 1.0), 1.0) * 0.20
            multiplier = max(0.05, 1.0 - touch_penalty - span_penalty - aspect_penalty)
            return (area * multiplier, area)

        candidates.sort(key=_candidate_likelihood, reverse=True)
    selected = candidates[0][1]

    if debug is not None:
        candidate_overlay = _draw_outer_candidate_overlay(frame_bgr, candidates, selected)
        debug["candidate_overlay"] = candidate_overlay
        debug["selected_candidate_corners_px"] = [[float(x), float(y)] for x, y in selected.reshape(4, 2).tolist()]
        debug["selected_candidate_area_px2"] = float(_quad_area(selected))

    return selected


def _find_boundary_near_board(
    profile: np.ndarray,
    start: int,
    end: int,
    *,
    prefer_near_high: bool,
    support_threshold: float,
) -> int | None:
    if end <= start:
        return None

    section = profile[start:end]
    if section.size == 0:
        return None

    indices = np.where(section >= support_threshold)[0]
    if indices.size == 0:
        return None

    if prefer_near_high:
        return int(start + int(indices.max()))
    return int(start + int(indices.min()))


def detect_outer_sheet_from_tape_projection(
    frame_bgr: np.ndarray,
    chessboard_corners: np.ndarray,
    board_size: tuple[int, int],
    search_margin_squares: float = 4.5,
    debug: dict[str, object] | None = None,
) -> np.ndarray | None:
    cols, rows = board_size
    if cols <= 0 or rows <= 0:
        return None

    board_h = board_space_to_image_homography(board_corners=chessboard_corners, board_size=board_size)

    margin = float(search_margin_squares)
    search_src_board = np.array(
        [
            [-margin, -margin],
            [float(cols) + margin, -margin],
            [float(cols) + margin, float(rows) + margin],
            [-margin, float(rows) + margin],
        ],
        dtype=np.float32,
    ).reshape(-1, 1, 2)
    search_src_img = cv2.perspectiveTransform(search_src_board, board_h).reshape(-1, 2).astype(np.float32)

    square_px = 100
    out_w = int(round((float(cols) + (2.0 * margin)) * square_px))
    out_h = int(round((float(rows) + (2.0 * margin)) * square_px))
    out_w = max(out_w, 400)
    out_h = max(out_h, 400)

    search_dst = np.array(
        [
            [0.0, 0.0],
            [float(out_w - 1), 0.0],
            [float(out_w - 1), float(out_h - 1)],
            [0.0, float(out_h - 1)],
        ],
        dtype=np.float32,
    )
    h_img_to_search = cv2.getPerspectiveTransform(search_src_img, search_dst)
    h_search_to_img = cv2.getPerspectiveTransform(search_dst, search_src_img)
    warped = cv2.warpPerspective(frame_bgr, h_img_to_search, (out_w, out_h))

    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    _, otsu_inv = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    tape_raw = otsu_inv.copy()
    tape_raw_unmasked = tape_raw.copy()

    board_x0 = int(round(margin * square_px))
    board_y0 = int(round(margin * square_px))
    board_x1 = int(round((margin + float(cols)) * square_px))
    board_y1 = int(round((margin + float(rows)) * square_px))

    board_mask = np.zeros_like(tape_raw)
    cv2.rectangle(board_mask, (board_x0, board_y0), (board_x1, board_y1), 255, thickness=-1)
    board_mask = cv2.dilate(board_mask, np.ones((11, 11), dtype=np.uint8), iterations=1)
    tape_raw[board_mask > 0] = 0

    tape_mask = cv2.morphologyEx(tape_raw, cv2.MORPH_CLOSE, np.ones((11, 11), dtype=np.uint8), iterations=2)
    tape_mask = cv2.morphologyEx(tape_mask, cv2.MORPH_OPEN, np.ones((5, 5), dtype=np.uint8), iterations=1)
    tape_mask = cv2.dilate(tape_mask, np.ones((3, 3), dtype=np.uint8), iterations=1)

    col_profile = tape_mask.sum(axis=0).astype(np.float32) / 255.0
    row_profile = tape_mask.sum(axis=1).astype(np.float32) / 255.0
    # Smooth profiles to reduce spurious peaks.
    smooth_kernel = np.ones((31,), dtype=np.float32) / 31.0
    col_profile = np.convolve(col_profile, smooth_kernel, mode="same")
    row_profile = np.convolve(row_profile, smooth_kernel, mode="same")
    if debug is not None:
        debug["tape_warped"] = warped
        debug["tape_mask_raw_unmasked"] = tape_raw_unmasked
        debug["tape_mask_raw"] = tape_raw
        debug["tape_mask"] = tape_mask

    min_gap_px = max(16, int(0.2 * square_px))
    support_col = float(out_h) * 0.02
    support_row = float(out_w) * 0.02
    left = _find_boundary_near_board(
        col_profile,
        0,
        max(0, board_x0 - min_gap_px),
        prefer_near_high=True,
        support_threshold=support_col,
    )
    right = _find_boundary_near_board(
        col_profile,
        min(out_w, board_x1 + min_gap_px),
        out_w,
        prefer_near_high=False,
        support_threshold=support_col,
    )
    top = _find_boundary_near_board(
        row_profile,
        0,
        max(0, board_y0 - min_gap_px),
        prefer_near_high=True,
        support_threshold=support_row,
    )
    bottom = _find_boundary_near_board(
        row_profile,
        min(out_h, board_y1 + min_gap_px),
        out_h,
        prefer_near_high=False,
        support_threshold=support_row,
    )
    if debug is not None:
        proj_vis = cv2.cvtColor(tape_mask, cv2.COLOR_GRAY2BGR)
        cv2.rectangle(proj_vis, (board_x0, board_y0), (board_x1, board_y1), (0, 255, 255), 2)
        if left is not None:
            cv2.line(proj_vis, (left, 0), (left, out_h - 1), (255, 0, 0), 1)
        if right is not None:
            cv2.line(proj_vis, (right, 0), (right, out_h - 1), (255, 0, 0), 1)
        if top is not None:
            cv2.line(proj_vis, (0, top), (out_w - 1, top), (255, 0, 0), 1)
        if bottom is not None:
            cv2.line(proj_vis, (0, bottom), (out_w - 1, bottom), (255, 0, 0), 1)
        debug["tape_projection_overlay"] = proj_vis

    if left is None or right is None or top is None or bottom is None:
        return None
    if right <= left + 20 or bottom <= top + 20:
        return None

    outer_search = np.array(
        [
            [float(left), float(top)],
            [float(right), float(top)],
            [float(right), float(bottom)],
            [float(left), float(bottom)],
        ],
        dtype=np.float32,
    ).reshape(-1, 1, 2)
    outer_img = cv2.perspectiveTransform(outer_search, h_search_to_img).reshape(-1, 2)
    outer_quad = _order_quad(outer_img)

    if not _quad_encloses_points(outer_quad, chessboard_corners):
        return None

    if debug is not None:
        proj_vis = debug.get("tape_projection_overlay")
        if isinstance(proj_vis, np.ndarray):
            cv2.rectangle(proj_vis, (left, top), (right, bottom), (0, 255, 0), 2)
            debug["tape_projection_overlay"] = proj_vis

    return outer_quad


def _detect_inner_corners(gray: np.ndarray, pattern_size: tuple[int, int]) -> tuple[bool, np.ndarray | None]:
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
    return cv2.findChessboardCorners(gray, pattern_size, flags)


def detect_chessboard_corners(
    frame_bgr: np.ndarray,
    board_size: tuple[int, int],
) -> np.ndarray | None:
    cols, rows = board_size
    if cols < 2 or rows < 2:
        return None
    pattern_size = (cols - 1, rows - 1)

    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    found, corners = _detect_inner_corners(gray, pattern_size)
    if not found or corners is None:
        return None

    if corners.dtype != np.float32:
        corners = corners.astype(np.float32)
    if corners.shape[-1] != 2:
        corners = corners.reshape(-1, 1, 2)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

    grid = refined.reshape(pattern_size[1], pattern_size[0], 2)
    tl = grid[0, 0]
    tr = grid[0, -1]
    br = grid[-1, -1]
    bl = grid[-1, 0]

    dx = (tr - tl) / max(pattern_size[0] - 1, 1)
    dy = (bl - tl) / max(pattern_size[1] - 1, 1)

    outer = np.array(
        [
            tl - dx - dy,
            tr + dx - dy,
            br + dx + dy,
            bl - dx + dy,
        ],
        dtype=np.float32,
    )
    return _order_quad(outer)


def detect_board_regions(
    frame_bgr: np.ndarray,
    board_size: tuple[int, int],
    outer_sheet_hsv_lower: tuple[int, int, int],
    outer_sheet_hsv_upper: tuple[int, int, int],
    min_outer_area_ratio: float = 0.1,
    max_outer_area_to_chessboard_ratio: float = 3.5,
    fallback_outer_margins_squares: tuple[float, float, float, float] = (3.2, 3.2, 1.4, 2.4),
    outer_candidate_mode: str = "auto",
    enable_tape_projection: bool = True,
    debug: dict[str, object] | None = None,
) -> BoardDetection:
    cols, rows = board_size
    left, right, top, bottom = fallback_outer_margins_squares
    expected_outer_to_chess_ratio = (
        (float(cols) + left + right) * (float(rows) + top + bottom)
    ) / (float(cols) * float(rows))

    chessboard = detect_chessboard_corners(frame_bgr=frame_bgr, board_size=board_size)
    outer_source = "none"
    outer_debug: dict[str, object] = {}
    mode = outer_candidate_mode.strip().lower()
    # If chessboard is found, require outer candidates to enclose it for both auto/hsv/dark modes.
    require_enclose = chessboard is not None

    outer_sheet = detect_outer_sheet(
        frame_bgr=frame_bgr,
        hsv_lower=outer_sheet_hsv_lower,
        hsv_upper=outer_sheet_hsv_upper,
        min_area_ratio=min_outer_area_ratio,
        must_enclose_quad=chessboard if require_enclose else None,
        max_area_to_enclosed_quad_ratio=max_outer_area_to_chessboard_ratio,
        expected_area_to_enclosed_ratio=expected_outer_to_chess_ratio if chessboard is not None else None,
        outer_candidate_mode=outer_candidate_mode,
        debug=outer_debug,
    )
    outer_candidate_source = str(outer_debug.get("outer_candidate_source", "")).strip()
    if outer_sheet is not None:
        outer_source = outer_candidate_source if outer_candidate_source else "hsv_or_dark"
    if debug is not None:
        debug.update(outer_debug)

    # Dark-mask-guided fallback: if one side of the tape ring is weak/missing (commonly top),
    # but at least 3 expected sides are supported, infer the outer quad from the chessboard.
    if chessboard is not None:
        dark_mask_dbg = outer_debug.get("dark_mask")
        if isinstance(dark_mask_dbg, np.ndarray):
            expected_outer = estimate_outer_sheet_from_chessboard(
                chessboard_corners=chessboard,
                board_size=board_size,
                margins_squares=fallback_outer_margins_squares,
            )
            band_px = max(8, int(min(frame_bgr.shape[:2]) * 0.012))
            expected_support = _quad_side_support(dark_mask_dbg, expected_outer, band_px=band_px)
            support_threshold = 0.08
            expected_strong = sum(1 for v in expected_support if v >= support_threshold)

            apply_inferred_outer = False
            if expected_strong >= 3:
                if outer_sheet is None:
                    apply_inferred_outer = True
                else:
                    current_support = _quad_side_support(dark_mask_dbg, outer_sheet, band_px=band_px)
                    current_mean = float(np.mean(current_support))
                    expected_mean = float(np.mean(expected_support))
                    expected_area = _quad_area(expected_outer)
                    current_area = _quad_area(outer_sheet)
                    area_small_vs_expected = current_area < (0.82 * expected_area)
                    support_much_better = expected_mean > (current_mean + 0.06)
                    missing_one_side = min(expected_support) < 0.02
                    if area_small_vs_expected or support_much_better or missing_one_side:
                        apply_inferred_outer = True

            if apply_inferred_outer:
                outer_sheet = expected_outer
                outer_source = "dark_mask_three_side_infer"
                if debug is not None:
                    debug["outer_candidate_source"] = "dark_mask_three_side_infer"
                    debug["selected_candidate_corners_px"] = [
                        [float(x), float(y)] for x, y in expected_outer.reshape(4, 2).tolist()
                    ]
                    debug["selected_candidate_area_px2"] = float(_quad_area(expected_outer))
                    candidate_overlay = debug.get("candidate_overlay")
                    if isinstance(candidate_overlay, np.ndarray):
                        vis = candidate_overlay.copy()
                        poly = expected_outer.astype(np.int32).reshape(-1, 1, 2)
                        cv2.polylines(vis, [poly], True, (0, 255, 0), 3, cv2.LINE_AA)
                        cv2.putText(
                            vis,
                            "INFER_3SIDE",
                            (16, 28),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0, 255, 0),
                            2,
                            cv2.LINE_AA,
                        )
                        debug["candidate_overlay"] = vis

            if debug is not None:
                debug["expected_outer_side_support"] = [float(v) for v in expected_support]
                debug["expected_outer_side_support_strong_count"] = int(expected_strong)
                debug["expected_outer_side_support_threshold"] = float(support_threshold)

    tape_debug: dict[str, object] = {}
    tape_outer = None
    if enable_tape_projection and chessboard is not None:
        tape_outer = detect_outer_sheet_from_tape_projection(
            frame_bgr=frame_bgr,
            chessboard_corners=chessboard,
            board_size=board_size,
            debug=tape_debug,
        )
    if debug is not None:
        debug.update(tape_debug)

    if enable_tape_projection and chessboard is not None and tape_outer is not None:
        chess_area = _quad_area(chessboard)
        if chess_area > 1.0:
            tape_ratio = _quad_area(tape_outer) / chess_area
            if outer_sheet is None:
                outer_sheet = tape_outer
                outer_source = "tape_projection"
            else:
                outer_ratio = _quad_area(outer_sheet) / chess_area
                tape_err = abs(tape_ratio - expected_outer_to_chess_ratio)
                outer_err = abs(outer_ratio - expected_outer_to_chess_ratio)
                if tape_err + 0.15 < outer_err:
                    outer_sheet = tape_outer
                    outer_source = "tape_projection"

    if chessboard is None and outer_sheet is not None:
        expected_inner_to_outer_area = (float(cols) * float(rows)) / (
            (float(cols) + left + right) * (float(rows) + top + bottom)
        )
        chessboard = detect_chessboard_from_outer_sheet(
            frame_bgr=frame_bgr,
            outer_sheet_corners=outer_sheet,
            expected_inner_to_outer_area_ratio=expected_inner_to_outer_area,
        )
    if chessboard is None and outer_sheet is not None:
        chessboard = estimate_chessboard_from_outer_sheet(
            outer_sheet_corners=outer_sheet,
            board_size=board_size,
            margins_squares=fallback_outer_margins_squares,
        )
    if chessboard is not None:
        refined = refine_chessboard_from_dark_squares(
            frame_bgr=frame_bgr,
            coarse_chessboard_corners=chessboard,
        )
        if refined is not None:
            chessboard = refined

    # Late 3-side infer pass: catches cases where chessboard is only available after
    # outer-to-inner fallback estimation.
    if chessboard is not None:
        dark_mask_dbg = outer_debug.get("dark_mask")
        if isinstance(dark_mask_dbg, np.ndarray):
            expected_outer = estimate_outer_sheet_from_chessboard(
                chessboard_corners=chessboard,
                board_size=board_size,
                margins_squares=fallback_outer_margins_squares,
            )
            band_px = max(8, int(min(frame_bgr.shape[:2]) * 0.012))
            expected_support = _quad_side_support(dark_mask_dbg, expected_outer, band_px=band_px)
            support_threshold = 0.08
            expected_strong = sum(1 for v in expected_support if v >= support_threshold)

            apply_inferred_outer = False
            if expected_strong >= 3:
                if outer_sheet is None:
                    apply_inferred_outer = True
                else:
                    current_support = _quad_side_support(dark_mask_dbg, outer_sheet, band_px=band_px)
                    current_mean = float(np.mean(current_support))
                    expected_mean = float(np.mean(expected_support))
                    expected_area = _quad_area(expected_outer)
                    current_area = _quad_area(outer_sheet)
                    area_small_vs_expected = current_area < (0.82 * expected_area)
                    support_much_better = expected_mean > (current_mean + 0.06)
                    missing_one_side = min(expected_support) < 0.02
                    if area_small_vs_expected or support_much_better or missing_one_side:
                        apply_inferred_outer = True

            if apply_inferred_outer:
                outer_sheet = expected_outer
                outer_source = "dark_mask_three_side_infer"
                if debug is not None:
                    debug["outer_candidate_source"] = "dark_mask_three_side_infer"
                    debug["selected_candidate_corners_px"] = [
                        [float(x), float(y)] for x, y in expected_outer.reshape(4, 2).tolist()
                    ]
                    debug["selected_candidate_area_px2"] = float(_quad_area(expected_outer))
                    candidate_overlay = debug.get("candidate_overlay")
                    if isinstance(candidate_overlay, np.ndarray):
                        vis = candidate_overlay.copy()
                        poly = expected_outer.astype(np.int32).reshape(-1, 1, 2)
                        cv2.polylines(vis, [poly], True, (0, 255, 0), 3, cv2.LINE_AA)
                        cv2.putText(
                            vis,
                            "INFER_3SIDE",
                            (16, 28),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0, 255, 0),
                            2,
                            cv2.LINE_AA,
                        )
                        debug["candidate_overlay"] = vis

            if debug is not None:
                debug["expected_outer_side_support"] = [float(v) for v in expected_support]
                debug["expected_outer_side_support_strong_count"] = int(expected_strong)
                debug["expected_outer_side_support_threshold"] = float(support_threshold)

    ratio_prune_allowed = mode != "dark_only" and outer_source not in {
        "dark_tape_inner_ring",
        "dark_hsv_tape_ring",
        "dark_enclosing_hsv",
        "dark_guided_hsv_overlap",
    }
    if ratio_prune_allowed and chessboard is not None and outer_sheet is not None:
        chess_area = _quad_area(chessboard)
        outer_area = _quad_area(outer_sheet)
        if chess_area > 1.0:
            ratio = outer_area / chess_area
            # Replace unstable outer detections with chessboard-based estimate
            # when area ratio is implausible for the configured board margins.
            min_ratio = expected_outer_to_chess_ratio * 0.72
            max_ratio = expected_outer_to_chess_ratio * 1.22
            if ratio < min_ratio or ratio > max_ratio:
                outer_sheet = None

    # Retry tape-based outer detection once chessboard is available, even if
    # initial chessboard detection failed earlier in the pipeline.
    if enable_tape_projection and chessboard is not None:
        tape_debug_late: dict[str, object] = {}
        tape_outer_late = detect_outer_sheet_from_tape_projection(
            frame_bgr=frame_bgr,
            chessboard_corners=chessboard,
            board_size=board_size,
            debug=tape_debug_late,
        )
        if debug is not None:
            debug.update({f"late_{k}": v for k, v in tape_debug_late.items()})
        if tape_outer_late is not None:
            chess_area = _quad_area(chessboard)
            if chess_area > 1.0:
                tape_ratio = _quad_area(tape_outer_late) / chess_area
                tape_err = abs(tape_ratio - expected_outer_to_chess_ratio)
                if outer_sheet is None:
                    outer_sheet = tape_outer_late
                    outer_source = "tape_projection"
                else:
                    outer_ratio = _quad_area(outer_sheet) / chess_area
                    outer_err = abs(outer_ratio - expected_outer_to_chess_ratio)
                    if tape_err + 0.10 < outer_err:
                        outer_sheet = tape_outer_late
                        outer_source = "tape_projection"

    if outer_sheet is None and chessboard is not None:
        outer_sheet = estimate_outer_sheet_from_chessboard(
            chessboard_corners=chessboard,
            board_size=board_size,
            margins_squares=fallback_outer_margins_squares,
        )
        outer_source = "estimated_from_chessboard"
    if debug is not None:
        debug["outer_source"] = outer_source
        debug["has_chessboard"] = chessboard is not None
        debug["has_outer_sheet"] = outer_sheet is not None
    return BoardDetection(outer_sheet_corners=outer_sheet, chessboard_corners=chessboard)


def estimate_outer_sheet_from_chessboard(
    chessboard_corners: np.ndarray,
    board_size: tuple[int, int],
    margins_squares: tuple[float, float, float, float],
) -> np.ndarray:
    left, right, top, bottom = margins_squares
    cols, rows = board_size

    src = np.array(
        [
            [0.0, 0.0],
            [float(cols), 0.0],
            [float(cols), float(rows)],
            [0.0, float(rows)],
        ],
        dtype=np.float32,
    )
    dst = chessboard_corners.astype(np.float32)
    h_matrix = cv2.getPerspectiveTransform(src, dst)

    outer_src = np.array(
        [
            [-left, -top],
            [float(cols) + right, -top],
            [float(cols) + right, float(rows) + bottom],
            [-left, float(rows) + bottom],
        ],
        dtype=np.float32,
    ).reshape(-1, 1, 2)
    outer_dst = cv2.perspectiveTransform(outer_src, h_matrix).reshape(-1, 2)
    return _order_quad(outer_dst)


def estimate_chessboard_from_outer_sheet(
    outer_sheet_corners: np.ndarray,
    board_size: tuple[int, int],
    margins_squares: tuple[float, float, float, float],
) -> np.ndarray:
    left, right, top, bottom = margins_squares
    cols, rows = board_size

    outer_src = np.array(
        [
            [-left, -top],
            [float(cols) + right, -top],
            [float(cols) + right, float(rows) + bottom],
            [-left, float(rows) + bottom],
        ],
        dtype=np.float32,
    )
    h_matrix = cv2.getPerspectiveTransform(outer_src, outer_sheet_corners.astype(np.float32))

    inner_src = np.array(
        [
            [0.0, 0.0],
            [float(cols), 0.0],
            [float(cols), float(rows)],
            [0.0, float(rows)],
        ],
        dtype=np.float32,
    ).reshape(-1, 1, 2)
    inner_dst = cv2.perspectiveTransform(inner_src, h_matrix).reshape(-1, 2)
    return _order_quad(inner_dst)


def detect_chessboard_from_outer_sheet(
    frame_bgr: np.ndarray,
    outer_sheet_corners: np.ndarray,
    expected_inner_to_outer_area_ratio: float = 0.38,
) -> np.ndarray | None:
    outer = outer_sheet_corners.astype(np.float32)
    outer_area = _quad_area(outer)
    if outer_area <= 1.0:
        return None

    edge_candidate = _detect_chessboard_from_outer_sheet_edges(
        frame_bgr=frame_bgr,
        outer_sheet_corners=outer,
        expected_inner_to_outer_area_ratio=expected_inner_to_outer_area_ratio,
    )
    if edge_candidate is not None:
        return edge_candidate

    # Fallback to dark-region contours if edge-based method fails.
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    # Dark content inside the sheet: black grid lines + dark squares.
    _, dark_mask = cv2.threshold(blur, 90, 255, cv2.THRESH_BINARY_INV)
    dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_CLOSE, np.ones((5, 5), dtype=np.uint8), iterations=2)

    region_mask = np.zeros_like(dark_mask)
    cv2.fillPoly(region_mask, [outer.astype(np.int32).reshape(-1, 1, 2)], 255)
    region_mask = cv2.erode(region_mask, np.ones((9, 9), dtype=np.uint8), iterations=1)
    masked = cv2.bitwise_and(dark_mask, region_mask)

    contours, _ = cv2.findContours(masked, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    candidates: list[tuple[float, float, np.ndarray]] = []
    for contour in contours:
        area = float(cv2.contourArea(contour))
        ratio = area / outer_area
        if ratio < 0.08 or ratio > 0.85:
            continue

        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        if len(approx) == 4:
            quad = _order_quad(approx.reshape(4, 2))
        else:
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            quad = _order_quad(box)

        if not _quad_encloses_points(outer, quad):
            continue

        aspect = _quad_aspect_ratio(quad)
        if aspect > 1.35:
            continue

        score = abs(ratio - expected_inner_to_outer_area_ratio)
        candidates.append((score, -area, quad))

    if not candidates:
        return None

    candidates.sort(key=lambda item: (item[0], item[1]))
    return candidates[0][2]


def _edge_support_along_quad(edge_img: np.ndarray, quad: np.ndarray, samples_per_edge: int = 80) -> float:
    hits = 0
    total = 0
    pts = quad.astype(np.float32).reshape(-1, 2)
    for i in range(4):
        p1 = pts[i]
        p2 = pts[(i + 1) % 4]
        for t in np.linspace(0.0, 1.0, samples_per_edge):
            x = int(round(p1[0] * (1.0 - t) + p2[0] * t))
            y = int(round(p1[1] * (1.0 - t) + p2[1] * t))
            if x < 0 or y < 0 or y >= edge_img.shape[0] or x >= edge_img.shape[1]:
                continue
            total += 1
            if edge_img[y, x] > 0:
                hits += 1
    if total <= 0:
        return 0.0
    return float(hits) / float(total)


def _detect_chessboard_from_outer_sheet_edges(
    frame_bgr: np.ndarray,
    outer_sheet_corners: np.ndarray,
    expected_inner_to_outer_area_ratio: float,
) -> np.ndarray | None:
    outer = outer_sheet_corners.astype(np.float32)
    outer_area = _quad_area(outer)
    if outer_area <= 1.0:
        return None

    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 140)

    region_mask = np.zeros_like(edges)
    cv2.fillPoly(region_mask, [outer.astype(np.int32).reshape(-1, 1, 2)], 255)
    region_mask = cv2.erode(region_mask, np.ones((5, 5), dtype=np.uint8), iterations=1)
    edges = cv2.bitwise_and(edges, region_mask)
    edges = cv2.dilate(edges, np.ones((3, 3), dtype=np.uint8), iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    best_quad: np.ndarray | None = None
    best_score = -1e9

    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < 5000.0:
            continue

        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        if len(approx) == 4:
            quad = _order_quad(approx.reshape(4, 2))
        else:
            rect = cv2.minAreaRect(contour)
            quad = _order_quad(cv2.boxPoints(rect))

        quad_area = _quad_area(quad)
        quad_ratio = quad_area / outer_area
        if quad_ratio < 0.15 or quad_ratio > 0.75:
            continue

        aspect = _quad_aspect_ratio(quad)
        if aspect > 1.5:
            continue

        if not _quad_encloses_points(outer, quad):
            continue

        support = _edge_support_along_quad(edges, quad, samples_per_edge=80)
        ratio_error = abs(quad_ratio - expected_inner_to_outer_area_ratio)

        # High edge support and expected area ratio both matter.
        score = (2.0 * support) - ratio_error
        if score > best_score:
            best_score = score
            best_quad = quad

    if best_quad is None:
        return None
    return best_quad


def refine_chessboard_from_dark_squares(
    frame_bgr: np.ndarray,
    coarse_chessboard_corners: np.ndarray,
) -> np.ndarray | None:
    src = coarse_chessboard_corners.astype(np.float32)
    out_w = 960
    out_h = 960
    dst = np.array(
        [
            [0.0, 0.0],
            [float(out_w - 1), 0.0],
            [float(out_w - 1), float(out_h - 1)],
            [0.0, float(out_h - 1)],
        ],
        dtype=np.float32,
    )
    h_forward = cv2.getPerspectiveTransform(src, dst)
    h_inverse = cv2.getPerspectiveTransform(dst, src)

    warped = cv2.warpPerspective(frame_bgr, h_forward, (out_w, out_h))
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, dark = cv2.threshold(blur, 95, 255, cv2.THRESH_BINARY_INV)
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((3, 3), dtype=np.uint8), iterations=1)

    contours, _ = cv2.findContours(dark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    img_area = float(out_w * out_h)
    boxes: list[tuple[int, int, int, int]] = []
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < img_area * 0.0012 or area > img_area * 0.03:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        if x <= 3 or y <= 3 or (x + w) >= (out_w - 3) or (y + h) >= (out_h - 3):
            continue
        if w <= 3 or h <= 3:
            continue
        ratio = float(w) / float(h)
        if ratio < 0.6 or ratio > 1.6:
            continue
        boxes.append((x, y, w, h))

    # Need enough square-like blobs to trust refinement.
    if len(boxes) < 10:
        return None

    x0 = min(x for x, _, _, _ in boxes)
    y0 = min(y for _, y, _, _ in boxes)
    x1 = max(x + w for x, _, w, _ in boxes)
    y1 = max(y + h for _, y, _, h in boxes)

    # Expand a little so the full board border is included.
    pad_x = int(0.03 * out_w)
    pad_y = int(0.03 * out_h)
    x0 = max(0, x0 - pad_x)
    y0 = max(0, y0 - pad_y)
    x1 = min(out_w - 1, x1 + pad_x)
    y1 = min(out_h - 1, y1 + pad_y)

    refined_warp = np.array(
        [
            [float(x0), float(y0)],
            [float(x1), float(y0)],
            [float(x1), float(y1)],
            [float(x0), float(y1)],
        ],
        dtype=np.float32,
    ).reshape(-1, 1, 2)
    refined_img = cv2.perspectiveTransform(refined_warp, h_inverse).reshape(-1, 2)
    return _order_quad(refined_img)


def warp_to_board(
    frame_bgr: np.ndarray,
    board_corners: np.ndarray,
    board_size: tuple[int, int],
    square_px: int = 96,
) -> tuple[np.ndarray, np.ndarray]:
    cols, rows = board_size
    out_w = max(cols * square_px, cols)
    out_h = max(rows * square_px, rows)

    dst = np.array(
        [
            [0.0, 0.0],
            [float(out_w - 1), 0.0],
            [float(out_w - 1), float(out_h - 1)],
            [0.0, float(out_h - 1)],
        ],
        dtype=np.float32,
    )
    h_matrix = cv2.getPerspectiveTransform(board_corners.astype(np.float32), dst)
    warped = cv2.warpPerspective(frame_bgr, h_matrix, (out_w, out_h))
    return warped, h_matrix


def draw_detection_overlay(frame_bgr: np.ndarray, detection: BoardDetection) -> np.ndarray:
    vis = frame_bgr.copy()
    if detection.outer_sheet_corners is not None:
        outer = detection.outer_sheet_corners.astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(vis, [outer], True, (0, 255, 0), 3)
        for idx, pt in enumerate(outer.reshape(-1, 2)):
            cv2.putText(
                vis,
                f"S{idx}",
                (int(pt[0]) + 6, int(pt[1]) - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

    if detection.chessboard_corners is not None:
        board = detection.chessboard_corners.astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(vis, [board], True, (0, 255, 255), 3)
        for idx, pt in enumerate(board.reshape(-1, 2)):
            cv2.circle(vis, (int(pt[0]), int(pt[1])), 6, (0, 0, 255), -1)
            cv2.putText(
                vis,
                f"B{idx}",
                (int(pt[0]) + 6, int(pt[1]) - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )
    return vis


def _square_label(x: int, y: int) -> str:
    if 0 <= x < 26:
        return f"{chr(ord('a') + x)}{y + 1}"
    return f"f{x}_{y + 1}"


def board_space_to_image_homography(board_corners: np.ndarray, board_size: tuple[int, int]) -> np.ndarray:
    cols, rows = board_size
    src = np.array(
        [
            [0.0, 0.0],
            [float(cols), 0.0],
            [float(cols), float(rows)],
            [0.0, float(rows)],
        ],
        dtype=np.float32,
    )
    dst = board_corners.astype(np.float32)
    return cv2.getPerspectiveTransform(src, dst)


def generate_square_geometry(board_corners: np.ndarray, board_size: tuple[int, int]) -> list[BoardSquareGeometry]:
    cols, rows = board_size
    h_matrix = board_space_to_image_homography(board_corners=board_corners, board_size=board_size)

    squares: list[BoardSquareGeometry] = []
    for y in range(rows):
        for x in range(cols):
            index = y * cols + x
            poly_board = np.array(
                [
                    [float(x), float(y)],
                    [float(x + 1), float(y)],
                    [float(x + 1), float(y + 1)],
                    [float(x), float(y + 1)],
                ],
                dtype=np.float32,
            ).reshape(-1, 1, 2)
            center_board = np.array([[[float(x) + 0.5, float(y) + 0.5]]], dtype=np.float32)

            poly_img = cv2.perspectiveTransform(poly_board, h_matrix).reshape(-1, 2)
            center_img = cv2.perspectiveTransform(center_board, h_matrix).reshape(2)

            squares.append(
                BoardSquareGeometry(
                    index=index,
                    x=x,
                    y=y,
                    label=_square_label(x, y),
                    corners_px=poly_img.astype(np.float32),
                    center_px=center_img.astype(np.float32),
                )
            )
    return squares


def draw_square_grid_overlay(
    frame_bgr: np.ndarray,
    squares: list[BoardSquareGeometry],
    label_mode: str = "index",
    line_thickness: int = 1,
) -> np.ndarray:
    vis = frame_bgr.copy()
    label_mode_norm = label_mode.lower().strip()

    for square in squares:
        poly_i = square.corners_px.astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(vis, [poly_i], True, (255, 128, 0), line_thickness, cv2.LINE_AA)

        if label_mode_norm == "none":
            continue
        if label_mode_norm == "coord":
            label = square.label
        else:
            label = str(square.index)

        center = square.center_px.astype(np.int32)
        cv2.putText(
            vis,
            label,
            (int(center[0]) - 10, int(center[1]) + 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.32,
            (0, 0, 255),
            1,
            cv2.LINE_AA,
        )

    return vis
