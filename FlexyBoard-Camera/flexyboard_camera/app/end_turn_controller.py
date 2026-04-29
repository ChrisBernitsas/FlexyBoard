from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from flexyboard_camera.camera.camera_manager import CameraManager, CameraSettings
from flexyboard_camera.camera.capture import save_frame
from flexyboard_camera.comms.stm32_client import ClientSettings, STM32Client
from flexyboard_camera.game.board_models import BoardSpec
from flexyboard_camera.game.game_rules import ResponsePlanner
from flexyboard_camera.game.move_models import EndTurnResult, MoveEvent
from flexyboard_camera.utils.config import Config
from flexyboard_camera.utils.paths import ensure_dir, timestamp_slug
from flexyboard_camera.vision.board_detector import (
    detect_board_regions,
    draw_detection_overlay,
    warp_to_board,
)
from flexyboard_camera.vision.calibration import CalibrationData, default_corners_from_roi
from flexyboard_camera.vision.diff_detector import detect_square_changes
from flexyboard_camera.vision.move_inference import InferenceInputs, infer_move
from flexyboard_camera.vision.piece_classifier import build_piece_classifier
from flexyboard_camera.vision.preprocess import preprocess_frame

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class CycleArtifacts:
    run_dir: Path
    before_raw: Path
    after_raw: Path
    before_processed: Path
    after_processed: Path
    board_detection_overlay: Path
    after_board_detection_overlay: Path
    before_aligned: Path
    after_aligned: Path
    diff_image: Path
    diff_threshold: Path
    move_json: Path
    change_json: Path


class EndTurnController:
    def __init__(self, config: Config):
        self.config = config
        self._camera = CameraManager(
            CameraSettings(
                index=config.camera.index,
                width=config.camera.width,
                height=config.camera.height,
                retries=config.camera.retries,
                retry_delay_sec=config.camera.retry_delay_sec,
                pre_capture_flush_frames=config.camera.pre_capture_flush_frames,
                pre_capture_flush_delay_sec=config.camera.pre_capture_flush_delay_sec,
            )
        )
        self._stm32 = STM32Client(
            ClientSettings(
                port=config.comms.port,
                baudrate=config.comms.baudrate,
                timeout_sec=config.comms.timeout_sec,
                retries=config.comms.retries,
            )
        )
        self._classifier = build_piece_classifier(
            enabled=config.classifier.enabled,
            backend=config.classifier.backend,
            model_path=config.classifier.model_path,
            confidence_threshold=config.classifier.confidence_threshold,
            input_size=config.classifier.input_size,
            device=config.classifier.device,
        )
        self._board_spec = BoardSpec(
            game=config.app.game,
            width=config.vision.board_size[0],
            height=config.vision.board_size[1],
            square_size_mm=config.board.square_size_mm,
            origin_offset_mm_x=config.board.origin_offset_mm[0],
            origin_offset_mm_y=config.board.origin_offset_mm[1],
        )
        self._planner = ResponsePlanner(self._board_spec)
        self._before_frame: np.ndarray | None = None

    def _debug_root(self) -> Path:
        return ensure_dir(self.config.paths.debug_dir)

    def calibrate(self, output_path: str | Path | None = None) -> CalibrationData:
        image_points: list[tuple[float, float]]
        board_points: list[tuple[float, float]]
        roi = self.config.vision.roi

        if self.config.vision.auto_detect_board:
            frame = self._camera.capture_frame()
            detection = detect_board_regions(
                frame_bgr=frame,
                board_size=self.config.vision.board_size,
                outer_sheet_hsv_lower=self.config.vision.outer_sheet_hsv_lower,
                outer_sheet_hsv_upper=self.config.vision.outer_sheet_hsv_upper,
                min_outer_area_ratio=self.config.vision.outer_sheet_min_area_ratio,
                max_outer_area_to_chessboard_ratio=self.config.vision.outer_sheet_max_area_to_chessboard_ratio,
                fallback_outer_margins_squares=self.config.vision.fallback_outer_margins_squares,
            )
            save_frame(draw_detection_overlay(frame, detection), self._debug_root() / "calibration_detection_overlay.png")
            if detection.chessboard_corners is not None:
                corners = detection.chessboard_corners.astype(float).tolist()
                image_points = [(float(x), float(y)) for x, y in corners]
                board_points = [
                    (0.0, 0.0),
                    (float(self.config.vision.board_size[0]), 0.0),
                    (float(self.config.vision.board_size[0]), float(self.config.vision.board_size[1])),
                    (0.0, float(self.config.vision.board_size[1])),
                ]
                xs = [p[0] for p in image_points]
                ys = [p[1] for p in image_points]
                x0 = int(max(0.0, min(xs)))
                y0 = int(max(0.0, min(ys)))
                x1 = int(max(xs))
                y1 = int(max(ys))
                roi = (x0, y0, max(1, x1 - x0), max(1, y1 - y0))
            else:
                logger.warning(
                    "Auto board detection enabled for calibration but chessboard not found; using ROI corners fallback."
                )
                image_points, board_points = default_corners_from_roi(
                    self.config.vision.roi, self.config.vision.board_size
                )
        else:
            image_points, board_points = default_corners_from_roi(self.config.vision.roi, self.config.vision.board_size)

        calibration = CalibrationData.compute(
            board_size=self.config.vision.board_size,
            roi=roi,
            image_points=image_points,
            board_points=board_points,
        )
        output = output_path or self.config.paths.calibration_file
        calibration.save(output)
        logger.info("Calibration saved to %s", output)
        return calibration

    def capture_before(
        self,
        image_path: str | None = None,
        *,
        reopen_stream: bool = True,
        output_path: str | Path | None = None,
    ) -> Path:
        if image_path:
            frame = self._camera.load_frame(image_path)
        else:
            # Reopen stream to avoid stale buffered frames persisting across long idle periods.
            if reopen_stream:
                self._camera.close()
            frame = self._camera.capture_frame()
        self._before_frame = frame
        out = Path(output_path) if output_path is not None else self._debug_root() / "before_latest.png"
        save_frame(frame, out)
        logger.info("Captured before-frame at %s", out)
        return out

    def capture_after(
        self,
        image_path: str | None = None,
        *,
        reopen_stream: bool = True,
        output_path: str | Path | None = None,
    ) -> Path:
        if image_path:
            frame = self._camera.load_frame(image_path)
        else:
            # Force a fresh stream reopen so AFTER capture is guaranteed post-trigger.
            if reopen_stream:
                self._camera.close()
            frame = self._camera.capture_frame()
        out = Path(output_path) if output_path is not None else self._debug_root() / "after_latest.png"
        save_frame(frame, out)
        logger.info("Captured after-frame at %s", out)
        return out

    @staticmethod
    def _write_json(path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)

    def _create_artifact_paths(self) -> CycleArtifacts:
        run_dir = self._debug_root() / f"cycle_{timestamp_slug()}"
        run_dir.mkdir(parents=True, exist_ok=True)
        return CycleArtifacts(
            run_dir=run_dir,
            before_raw=run_dir / "before_raw.png",
            after_raw=run_dir / "after_raw.png",
            before_processed=run_dir / "before_processed.png",
            after_processed=run_dir / "after_processed.png",
            board_detection_overlay=run_dir / "board_detection_overlay.png",
            after_board_detection_overlay=run_dir / "after_board_detection_overlay.png",
            before_aligned=run_dir / "before_aligned.png",
            after_aligned=run_dir / "after_aligned.png",
            diff_image=run_dir / "diff.png",
            diff_threshold=run_dir / "diff_threshold.png",
            move_json=run_dir / "inferred_move.json",
            change_json=run_dir / "changed_squares.json",
        )

    def infer_move(self, before_path: str | None = None, after_path: str | None = None) -> tuple[MoveEvent, CycleArtifacts]:
        artifacts = self._create_artifact_paths()

        before_raw = self._camera.load_frame(before_path) if before_path else self._before_frame
        if before_raw is None:
            before_raw = self._camera.capture_frame()

        after_raw = self._camera.load_frame(after_path) if after_path else self._camera.capture_frame()

        save_frame(before_raw, artifacts.before_raw)
        save_frame(after_raw, artifacts.after_raw)

        effective_before = before_raw
        effective_after = after_raw
        effective_roi = self.config.vision.roi

        if self.config.vision.auto_detect_board:
            before_detection = detect_board_regions(
                frame_bgr=before_raw,
                board_size=self.config.vision.board_size,
                outer_sheet_hsv_lower=self.config.vision.outer_sheet_hsv_lower,
                outer_sheet_hsv_upper=self.config.vision.outer_sheet_hsv_upper,
                min_outer_area_ratio=self.config.vision.outer_sheet_min_area_ratio,
                max_outer_area_to_chessboard_ratio=self.config.vision.outer_sheet_max_area_to_chessboard_ratio,
                fallback_outer_margins_squares=self.config.vision.fallback_outer_margins_squares,
            )
            after_detection = detect_board_regions(
                frame_bgr=after_raw,
                board_size=self.config.vision.board_size,
                outer_sheet_hsv_lower=self.config.vision.outer_sheet_hsv_lower,
                outer_sheet_hsv_upper=self.config.vision.outer_sheet_hsv_upper,
                min_outer_area_ratio=self.config.vision.outer_sheet_min_area_ratio,
                max_outer_area_to_chessboard_ratio=self.config.vision.outer_sheet_max_area_to_chessboard_ratio,
                fallback_outer_margins_squares=self.config.vision.fallback_outer_margins_squares,
            )
            save_frame(draw_detection_overlay(before_raw, before_detection), artifacts.board_detection_overlay)
            save_frame(draw_detection_overlay(after_raw, after_detection), artifacts.after_board_detection_overlay)

            before_corners = before_detection.chessboard_corners
            after_corners = after_detection.chessboard_corners

            if before_corners is not None:
                if after_corners is None:
                    after_corners = before_corners
                warped_before, _ = warp_to_board(
                    frame_bgr=before_raw,
                    board_corners=before_corners,
                    board_size=self.config.vision.board_size,
                    square_px=self.config.vision.warp_square_px,
                )
                warped_after, _ = warp_to_board(
                    frame_bgr=after_raw,
                    board_corners=after_corners,
                    board_size=self.config.vision.board_size,
                    square_px=self.config.vision.warp_square_px,
                )
                effective_before = warped_before
                effective_after = warped_after
                save_frame(effective_before, artifacts.before_aligned)
                save_frame(effective_after, artifacts.after_aligned)
                h, w = effective_before.shape[:2]
                effective_roi = (0, 0, w, h)
                logger.info(
                    "Auto board detection applied: diff restricted to chessboard warp (%dx%d).",
                    w,
                    h,
                )
            else:
                logger.warning(
                    "Auto board detection enabled but chessboard was not found; falling back to configured ROI."
                )

        before_pre = preprocess_frame(effective_before, roi=effective_roi, blur_kernel=self.config.vision.blur_kernel)
        after_pre = preprocess_frame(effective_after, roi=effective_roi, blur_kernel=self.config.vision.blur_kernel)

        save_frame(before_pre.enhanced, artifacts.before_processed)
        save_frame(after_pre.enhanced, artifacts.after_processed)

        diff = detect_square_changes(
            before_img=before_pre.enhanced,
            after_img=after_pre.enhanced,
            board_size=self.config.vision.board_size,
            diff_threshold=self.config.vision.diff_threshold,
            min_changed_ratio=self.config.vision.changed_square_pixel_ratio,
        )
        save_frame(diff.diff_image, artifacts.diff_image)
        save_frame(diff.threshold_image, artifacts.diff_threshold)

        move = infer_move(
            InferenceInputs(
                game=self.config.app.game,
                board_size=self.config.vision.board_size,
                before_img=before_pre.enhanced,
                after_img=after_pre.enhanced,
                changes=diff.changes,
            ),
            classifier=self._classifier,
        )

        self._write_json(
            artifacts.change_json,
            {
                "changed_squares": [
                    {
                        "x": c.coord.x,
                        "y": c.coord.y,
                        "pixel_ratio": c.pixel_ratio,
                        "signed_intensity_delta": c.signed_intensity_delta,
                    }
                    for c in diff.changes
                ]
            },
        )
        self._write_json(artifacts.move_json, move.to_dict())

        return move, artifacts

    def send_move(self, move: MoveEvent) -> list[dict[str, Any]]:
        if move.source is None or move.destination is None:
            raise ValueError("Cannot send move with missing source/destination")

        plan = self._planner.propose_response_plan(move) or []
        if not plan:
            raise ValueError("Unable to build response command for move")

        if self.config.safety.auto_home_before_move:
            self._stm32.home()

        terminal_statuses: list[dict[str, Any]] = []
        for command in plan:
            for payload in command.to_step_payloads(minimal_for_stm32=True):
                responses = self._stm32.execute_move(payload)
                if responses:
                    terminal_statuses.append(responses[-1].to_dict())
        return terminal_statuses

    def run_end_turn_cycle(
        self,
        before_path: str | None = None,
        after_path: str | None = None,
        force_low_confidence: bool = False,
    ) -> EndTurnResult:
        start = time.monotonic()
        move, _ = self.infer_move(before_path=before_path, after_path=after_path)

        blocked_reason: str | None = None
        send_allowed = move.confidence >= self.config.app.confidence_threshold or force_low_confidence
        if not send_allowed and self.config.safety.fail_on_low_confidence:
            blocked_reason = (
                f"Low confidence {move.confidence:.3f} < threshold {self.config.app.confidence_threshold:.3f}"
            )
            logger.warning(blocked_reason)
            return EndTurnResult(
                move_event=move,
                response_command=None,
                sent_to_stm32=False,
                stm32_status=None,
                blocked_reason=blocked_reason,
            )

        response_plan = self._planner.propose_response_plan(move) or []
        if not response_plan:
            blocked_reason = "No valid response command available for inferred move"
            logger.warning(blocked_reason)
            return EndTurnResult(
                move_event=move,
                response_command=None,
                sent_to_stm32=False,
                stm32_status=None,
                blocked_reason=blocked_reason,
            )

        if self.config.safety.auto_home_before_move:
            self._stm32.home()

        responses = []
        for command in response_plan:
            for payload in command.to_step_payloads(minimal_for_stm32=True):
                responses = self._stm32.execute_move(payload)
        elapsed_ms = (time.monotonic() - start) * 1000.0
        logger.info("End-turn cycle completed in %.1f ms", elapsed_ms)
        return EndTurnResult(
            move_event=move,
            response_command=response_plan[0],
            sent_to_stm32=True,
            stm32_status=responses[-1].to_dict() if responses else None,
        )
