from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True)
class AppConfig:
    game: str
    confidence_threshold: float
    allow_low_confidence_override: bool


@dataclass(slots=True)
class CameraConfig:
    index: int
    width: int
    height: int
    retries: int
    retry_delay_sec: float
    pre_capture_flush_frames: int = 8
    pre_capture_flush_delay_sec: float = 0.02


@dataclass(slots=True)
class VisionConfig:
    board_size: tuple[int, int]
    roi: tuple[int, int, int, int]
    blur_kernel: int
    diff_threshold: int
    min_changed_squares: int
    changed_square_pixel_ratio: float
    auto_detect_board: bool
    warp_square_px: int
    outer_sheet_hsv_lower: tuple[int, int, int]
    outer_sheet_hsv_upper: tuple[int, int, int]
    outer_sheet_min_area_ratio: float
    outer_sheet_max_area_to_chessboard_ratio: float
    fallback_outer_margins_squares: tuple[float, float, float, float]


@dataclass(slots=True)
class AnalysisConfig:
    label_mode: str
    inner_shrink: float
    diff_threshold: int
    min_changed_ratio: float
    outer_candidate_mode: str
    disable_tape_projection: bool
    board_lock_source: str
    geometry_reference: str
    disable_geometry_reference: bool
    camera_square_orientation: str


@dataclass(slots=True)
class BoardConfig:
    square_size_mm: float
    origin_offset_mm: tuple[float, float]


@dataclass(slots=True)
class MotorConfig:
    board_orientation: str


@dataclass(slots=True)
class CommsConfig:
    port: str
    baudrate: int
    timeout_sec: float
    retries: int


@dataclass(slots=True)
class PathsConfig:
    calibration_file: str
    logs_dir: str
    debug_dir: str


@dataclass(slots=True)
class SafetyConfig:
    auto_home_before_move: bool
    fail_on_low_confidence: bool


@dataclass(slots=True)
class ClassifierConfig:
    enabled: bool
    backend: str
    model_path: str | None
    confidence_threshold: float
    input_size: int
    device: str | None


@dataclass(slots=True)
class Config:
    app: AppConfig
    camera: CameraConfig
    vision: VisionConfig
    analysis: AnalysisConfig
    board: BoardConfig
    comms: CommsConfig
    paths: PathsConfig
    safety: SafetyConfig
    classifier: ClassifierConfig
    motor: MotorConfig

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Config":
        camera_raw = data["camera"]
        classifier_raw = data.get("classifier", {})
        analysis_raw = data.get("analysis", {})
        motor_raw = data.get("motor", {})
        return cls(
            app=AppConfig(**data["app"]),
            camera=CameraConfig(
                index=int(camera_raw["index"]),
                width=int(camera_raw["width"]),
                height=int(camera_raw["height"]),
                retries=int(camera_raw["retries"]),
                retry_delay_sec=float(camera_raw["retry_delay_sec"]),
                pre_capture_flush_frames=int(camera_raw.get("pre_capture_flush_frames", 8)),
                pre_capture_flush_delay_sec=float(camera_raw.get("pre_capture_flush_delay_sec", 0.02)),
            ),
            vision=VisionConfig(
                board_size=tuple(data["vision"]["board_size"]),
                roi=tuple(data["vision"]["roi"]),
                blur_kernel=int(data["vision"]["blur_kernel"]),
                diff_threshold=int(data["vision"]["diff_threshold"]),
                min_changed_squares=int(data["vision"]["min_changed_squares"]),
                changed_square_pixel_ratio=float(data["vision"]["changed_square_pixel_ratio"]),
                auto_detect_board=bool(data["vision"].get("auto_detect_board", False)),
                warp_square_px=int(data["vision"].get("warp_square_px", 96)),
                outer_sheet_hsv_lower=tuple(data["vision"].get("outer_sheet_hsv_lower", [8, 20, 50])),
                outer_sheet_hsv_upper=tuple(data["vision"].get("outer_sheet_hsv_upper", [35, 255, 255])),
                outer_sheet_min_area_ratio=float(data["vision"].get("outer_sheet_min_area_ratio", 0.1)),
                outer_sheet_max_area_to_chessboard_ratio=float(
                    data["vision"].get("outer_sheet_max_area_to_chessboard_ratio", 3.5)
                ),
                fallback_outer_margins_squares=tuple(
                    data["vision"].get("fallback_outer_margins_squares", [3.2, 3.2, 1.4, 2.4])
                ),
            ),
            analysis=AnalysisConfig(
                label_mode=str(analysis_raw.get("label_mode", "index")),
                inner_shrink=float(analysis_raw.get("inner_shrink", 0.02)),
                diff_threshold=int(analysis_raw.get("diff_threshold", 45)),
                min_changed_ratio=float(analysis_raw.get("min_changed_ratio", 0.28)),
                outer_candidate_mode=str(analysis_raw.get("outer_candidate_mode", "auto")),
                disable_tape_projection=bool(analysis_raw.get("disable_tape_projection", True)),
                board_lock_source=str(analysis_raw.get("board_lock_source", "before")),
                geometry_reference=str(
                    analysis_raw.get("geometry_reference", "configs/corners_info.json")
                ),
                disable_geometry_reference=bool(analysis_raw.get("disable_geometry_reference", False)),
                camera_square_orientation=str(analysis_raw.get("camera_square_orientation", "identity")),
            ),
            board=BoardConfig(
                square_size_mm=float(data["board"]["square_size_mm"]),
                origin_offset_mm=tuple(data["board"]["origin_offset_mm"]),
            ),
            motor=MotorConfig(
                board_orientation=str(motor_raw.get("board_orientation", "identity")),
            ),
            comms=CommsConfig(**data["comms"]),
            paths=PathsConfig(**data["paths"]),
            safety=SafetyConfig(**data["safety"]),
            classifier=ClassifierConfig(
                enabled=bool(classifier_raw.get("enabled", False)),
                backend=str(classifier_raw.get("backend", "stub")),
                model_path=classifier_raw.get("model_path"),
                confidence_threshold=float(classifier_raw.get("confidence_threshold", 0.35)),
                input_size=int(classifier_raw.get("input_size", 640)),
                device=classifier_raw.get("device"),
            ),
        )


def load_config(config_path: str | Path) -> Config:
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    return Config.from_dict(data)
