from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class YoloDetectionInput:
    class_name: str
    confidence: float
    bbox: tuple[float, float, float, float] | list[float]
    position: str | None = None


@dataclass(frozen=True)
class YoloResultInput:
    image_width: int
    image_height: int
    detections: list[YoloDetectionInput]
    frame_id: str | int | None = None
    timestamp_ms: float | None = None


@dataclass(frozen=True)
class MotionResultInput:
    available: bool
    direction: str
    speed: float | str | None = None
    confidence: float | None = None
    frame_id: str | int | None = None
    timestamp_ms: float | None = None


@dataclass(frozen=True)
class ImageQualityInput:
    is_blurry: bool
    blur_score: float | None = None
