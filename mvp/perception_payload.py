#!/usr/bin/env python3
"""YOLO/Optical Flow 결과를 Perception 연동 계약 payload로 변환한다.

`docs/perception_integration_contract.md`의 `yolo_payload`와 `flow_payload`
형식을 만든다. 이 모듈은 카메라, CUDA, YOLO 모델, VLM에 의존하지 않는 순수
변환 계층이므로 하드웨어 없이 단위 테스트할 수 있다.
"""

from __future__ import annotations

from typing import Any, Iterable, Optional


# escalator_mvp의 내부 방향 표기 → 계약 direction
FLOW_DIRECTION_MAP = {
    "UP": "up",
    "DOWN": "down",
    "STATIONARY": "stopped",
}

# 방향을 판정하지 못한 상태. `stopped`로 추측하지 않고 unknown으로 둔다.
UNDECIDED_DIRECTIONS = frozenset(
    {"UNCERTAIN", "ANALYZING", "NO_ESCALATOR", "UNKNOWN", ""}
)

SUPPORTED_DIRECTIONS = frozenset({"up", "down", "stopped", "unknown"})


def normalize_direction(direction: Any) -> str:
    """내부 방향 문자열을 계약 direction으로 바꾼다. 모르면 unknown이다."""

    if not isinstance(direction, str):
        return "unknown"
    token = direction.strip().upper()
    if token in UNDECIDED_DIRECTIONS:
        return "unknown"
    normalized = FLOW_DIRECTION_MAP.get(token)
    if normalized is not None:
        return normalized
    lowered = direction.strip().lower()
    return lowered if lowered in SUPPORTED_DIRECTIONS else "unknown"


def _float(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number or number in (float("inf"), float("-inf")):
        return None
    return number


def _clock_from_center_offset(center_offset: float) -> int:
    """화면 좌우 위치를 대략적인 시계 방향으로 표현한다(보조 정보)."""

    if center_offset < -0.6:
        return 10
    if center_offset < -0.2:
        return 11
    if center_offset <= 0.2:
        return 12
    if center_offset <= 0.6:
        return 1
    return 2


def _clamped_bbox(
    bbox: Iterable[Any],
    image_width: int,
    image_height: int,
) -> Optional[tuple[float, float, float, float]]:
    """bbox를 원본 프레임 범위로 자르고, 면적이 없으면 버린다."""

    values = [_float(value) for value in bbox]
    if len(values) != 4 or any(value is None for value in values):
        return None
    x1, y1, x2, y2 = values
    x1 = min(max(x1, 0.0), float(image_width))
    x2 = min(max(x2, 0.0), float(image_width))
    y1 = min(max(y1, 0.0), float(image_height))
    y2 = min(max(y2, 0.0), float(image_height))
    if x1 >= x2 or y1 >= y2:
        return None
    return x1, y1, x2, y2


def build_detection_payload(
    detection: dict[str, Any],
    *,
    image_width: int,
    image_height: int,
) -> Optional[dict[str, Any]]:
    """escalator_mvp detection 하나를 계약 detection으로 바꾼다."""

    bbox = _clamped_bbox(
        detection.get("bbox", ()), image_width, image_height
    )
    if bbox is None:
        return None
    x1, y1, x2, y2 = bbox

    confidence = _float(detection.get("confidence"))
    if confidence is None:
        return None
    confidence = min(max(confidence, 0.0), 1.0)

    center_x = (x1 + x2) / 2
    center_offset = (center_x - image_width / 2) / (image_width / 2)
    area_ratio = ((x2 - x1) * (y2 - y1)) / float(image_width * image_height)
    class_id = detection.get("class_id")

    return {
        "cls_id": int(class_id) if isinstance(class_id, (int, float)) else None,
        "cls_name": str(detection.get("class_name", "")),
        "conf": confidence,
        "x1": x1,
        "y1": y1,
        "x2": x2,
        "y2": y2,
        "track_id": detection.get("track_id"),
        "clock": _clock_from_center_offset(center_offset),
        "distance_m": None,
        "area_ratio": area_ratio,
        "center_offset": center_offset,
    }


def build_yolo_payload(
    *,
    frame_id: int,
    timestamp: float,
    image_width: int,
    image_height: int,
    detections: Iterable[dict[str, Any]] = (),
    quality: Optional[dict[str, Any]] = None,
    latency_ms: float = 0.0,
    ok: bool = True,
    error: Any = None,
) -> dict[str, Any]:
    """계약 §4의 yolo_payload를 만든다."""

    converted: list[dict[str, Any]] = []
    if ok:
        for detection in detections:
            payload = build_detection_payload(
                detection,
                image_width=image_width,
                image_height=image_height,
            )
            if payload is not None:
                converted.append(payload)

    return {
        "frame_id": int(frame_id),
        "timestamp": float(timestamp),
        "image_width": int(image_width),
        "image_height": int(image_height),
        "detections": converted,
        "quality": dict(quality) if quality else {},
        "latency_ms": float(latency_ms),
        "ok": bool(ok),
        "error": error,
    }


def build_flow_payload(
    *,
    frame_id: int,
    image_width: int,
    image_height: int,
    direction: Any,
    analysis: Optional[dict[str, Any]] = None,
    stable_frames: int = 0,
    ok: bool = True,
    error: Any = None,
) -> dict[str, Any]:
    """계약 §9의 flow_payload를 만든다.

    `direction`은 다수결로 안정화된 방향이어야 한다. 판정하지 못한 상태는
    `unknown`이며 이때 `available=false`가 된다.
    """

    normalized = normalize_direction(direction) if ok else "unknown"
    available = bool(ok) and normalized != "unknown" and analysis is not None

    source = analysis or {}
    mean_dx = _float(source.get("dx")) or 0.0
    mean_dy = _float(source.get("dy")) or 0.0
    magnitude = _float(source.get("magnitude")) or 0.0
    confidence = _float(source.get("confidence")) or 0.0
    valid_ratio = _float(source.get("valid_ratio")) or 0.0
    roi = _clamped_bbox(
        source.get("roi", ()), image_width, image_height
    ) if source.get("roi") else None

    return {
        "frame_id": int(frame_id),
        "available": available,
        "direction": normalized,
        "confidence": min(max(confidence, 0.0), 1.0),
        "speed": magnitude,
        "mean_dx": mean_dx,
        "mean_dy": mean_dy,
        "magnitude": magnitude,
        "valid_ratio": min(max(valid_ratio, 0.0), 1.0),
        "roi": list(roi) if roi is not None else None,
        "roi_source": "detection" if roi is not None else None,
        "stable_frames": int(stable_frames),
        "image_width": int(image_width),
        "image_height": int(image_height),
        "ok": bool(ok),
        "error": error,
    }


def build_failed_payloads(
    *,
    frame_id: int,
    timestamp: float,
    image_width: int,
    image_height: int,
    error: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """처리 실패 시 같은 오류 정보를 가진 두 payload를 만든다."""

    return (
        build_yolo_payload(
            frame_id=frame_id,
            timestamp=timestamp,
            image_width=image_width,
            image_height=image_height,
            ok=False,
            error=error,
        ),
        build_flow_payload(
            frame_id=frame_id,
            image_width=image_width,
            image_height=image_height,
            direction="unknown",
            ok=False,
            error=error,
        ),
    )
