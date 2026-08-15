from __future__ import annotations

import math
from dataclasses import asdict, is_dataclass
from typing import Any

from src.exceptions import (
    FrameSynchronizationError,
    IntegrationError,
    MotionResultValidationError,
    PerceptionUnavailableError,
    YoloResultValidationError,
)
from src.integration_schema import (
    ImageQualityInput,
    MotionResultInput,
    YoloDetectionInput,
    YoloResultInput,
)
from src.metadata_schema import SCHEMA_VERSION, validate_metadata


DEFAULT_USER_QUERY = "주변 상황을 알려줘."

# Perception `process_split()`의 detection 필드와 기존 내부 필드를 모두 받는다.
# 두 표기가 함께 들어오고 값이 다르면 추측하지 않고 validation error를 낸다.
DETECTION_CLASS_KEYS = ("class_name", "cls_name")
DETECTION_CONFIDENCE_KEYS = ("confidence", "conf")
BBOX_CORNER_KEYS = ("x1", "y1", "x2", "y2")

CLASS_ALIASES = {
    "elevator_button": "elevator_button",
    "elevator-button": "elevator_button",
    "lift_button": "elevator_button",
    "lift-button": "elevator_button",
    "elevator_call_button": "elevator_button",
    "elevator-call-button": "elevator_button",
    "escalator": "escalator",
    "moving_stairs": "escalator",
    "moving-stairs": "escalator",
}

POSITION_ALIASES = {
    "left": "left",
    "좌측": "left",
    "왼쪽": "left",
    "front": "front",
    "center": "front",
    "centre": "front",
    "middle": "front",
    "정면": "front",
    "중앙": "front",
    "right": "right",
    "우측": "right",
    "오른쪽": "right",
}

DIRECTION_ALIASES = {
    "up": "up",
    "upward": "up",
    "ascending": "up",
    "상행": "up",
    "위": "up",
    "위쪽": "up",
    "down": "down",
    "downward": "down",
    "descending": "down",
    "하행": "down",
    "아래": "down",
    "아래쪽": "down",
    "left": "left",
    "좌": "left",
    "왼쪽": "left",
    "right": "right",
    "우": "right",
    "오른쪽": "right",
    "opening": "opening",
    "open": "opening",
    "열림": "opening",
    "열리는중": "opening",
    "closing": "closing",
    "close": "closing",
    "닫힘": "closing",
    "닫히는중": "closing",
    "stopped": "stopped",
    "stop": "stopped",
    "stationary": "stopped",
    "정지": "stopped",
    "멈춤": "stopped",
    "unknown": "unknown",
    "none": "unknown",
    "unavailable": "unknown",
    "알수없음": "unknown",
}


def _mapping(value: Any, field: str, error_type: type[IntegrationError]) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if is_dataclass(value):
        return asdict(value)
    raise error_type(f"{field}는 dict 또는 지원 dataclass여야 합니다.")


def _finite_number(value: Any, field: str, error_type: type[IntegrationError]) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise error_type(f"{field}는 유한한 숫자여야 합니다.")
    number = float(value)
    if not math.isfinite(number):
        raise error_type(f"{field}는 유한한 숫자여야 합니다.")
    return number


def _positive_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise YoloResultValidationError(f"{field}는 양의 정수여야 합니다.")
    return value


def _validate_optional_context(
    value: dict[str, Any], error_type: type[IntegrationError]
) -> None:
    frame_id = value.get("frame_id")
    if frame_id is not None and (
        isinstance(frame_id, bool) or not isinstance(frame_id, (str, int))
    ):
        raise error_type("frame_id는 문자열, 정수 또는 null이어야 합니다.")
    if value.get("timestamp_ms") is not None:
        _finite_number(value["timestamp_ms"], "timestamp_ms", error_type)


def _comparable(value: Any) -> Any:
    """alias 충돌 비교용 값. 표기 차이는 허용하고 실제 값 차이만 잡는다."""

    if isinstance(value, str):
        return value.strip().lower()
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return value


def _alias_value(
    source: dict[str, Any],
    keys: tuple[str, ...],
    field: str,
) -> Any:
    """canonical 키를 우선하되 서로 다른 alias 값이 함께 오면 거부한다."""

    present = [(key, source[key]) for key in keys if source.get(key) is not None]
    if not present:
        return None
    first = _comparable(present[0][1])
    if any(_comparable(value) != first for _, value in present[1:]):
        raise YoloResultValidationError(
            f"{field}에 서로 다른 값이 동시에 들어왔습니다: "
            f"{', '.join(key for key, _ in present)}"
        )
    return present[0][1]


def _detection_bbox(detection: dict[str, Any], field: str) -> list[Any]:
    """`bbox`와 `x1,y1,x2,y2` 두 표기를 하나의 bbox list로 만든다."""

    corners = {
        key: detection[key]
        for key in BBOX_CORNER_KEYS
        if detection.get(key) is not None
    }
    if corners and len(corners) != len(BBOX_CORNER_KEYS):
        missing = [key for key in BBOX_CORNER_KEYS if key not in corners]
        raise YoloResultValidationError(
            f"{field}: bbox 좌표가 일부만 있습니다. 누락된 키: {', '.join(missing)}"
        )
    corner_bbox = (
        [corners[key] for key in BBOX_CORNER_KEYS] if corners else None
    )

    bbox_value = detection.get("bbox")
    if bbox_value is None:
        if corner_bbox is None:
            raise YoloResultValidationError(
                f"{field}: bbox 또는 x1, y1, x2, y2가 필요합니다."
            )
        return corner_bbox

    if not isinstance(bbox_value, (list, tuple)) or len(bbox_value) != 4:
        raise YoloResultValidationError(f"{field}.bbox는 숫자 4개여야 합니다.")

    if corner_bbox is not None and [
        _comparable(value) for value in bbox_value
    ] != [_comparable(value) for value in corner_bbox]:
        raise YoloResultValidationError(
            f"{field}: bbox와 x1, y1, x2, y2 값이 서로 다릅니다."
        )
    return list(bbox_value)


def _require_payload_ok(source: dict[str, Any], field: str) -> None:
    """Perception payload의 `ok` 플래그를 확인한다. 없으면 성공으로 본다."""

    if "ok" not in source:
        return
    ok = source["ok"]
    if not isinstance(ok, bool):
        raise YoloResultValidationError(f"{field}.ok는 boolean이어야 합니다.")
    if not ok:
        raise PerceptionUnavailableError(
            f"{field}가 ok=false 상태이므로 사용할 수 없습니다."
        )


def _quality_from_yolo_payload(source: dict[str, Any]) -> dict[str, Any] | None:
    """YOLO payload의 `quality`를 내부 image_quality 입력으로 옮긴다."""

    value = source.get("quality")
    if value is None:
        return None
    if not isinstance(value, dict):
        raise IntegrationError("yolo_result.quality는 dict여야 합니다.")
    if not value:
        return None
    quality: dict[str, Any] = {"is_blurry": value.get("is_blurry", False)}
    if value.get("blur_score") is not None:
        quality["blur_score"] = value["blur_score"]
    return quality


def _normalized_position(position: Any) -> str:
    if not isinstance(position, str) or not position.strip():
        raise YoloResultValidationError("position은 지원 alias 문자열이어야 합니다.")
    normalized = POSITION_ALIASES.get(position.strip().lower())
    if normalized is None:
        raise YoloResultValidationError(f"지원하지 않는 position입니다: {position}")
    return normalized


def _position_from_bbox(bbox: list[float], image_width: int) -> str:
    center_x = (bbox[0] + bbox[2]) / 2
    if center_x < image_width / 3:
        return "left"
    if center_x > image_width * 2 / 3:
        return "right"
    return "front"


def _normalize_yolo(
    value: YoloResultInput | dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    source = _mapping(value, "yolo_result", YoloResultValidationError)
    _require_payload_ok(source, "yolo_result")
    width = _positive_int(source.get("image_width"), "image_width")
    height = _positive_int(source.get("image_height"), "image_height")
    detections = source.get("detections")
    if not isinstance(detections, list):
        raise YoloResultValidationError("detections는 list여야 합니다.")
    _validate_optional_context(source, YoloResultValidationError)

    normalized: list[dict[str, Any]] = []
    for index, detection_value in enumerate(detections):
        detection = _mapping(
            detection_value,
            f"detections[{index}]",
            YoloResultValidationError,
        )
        class_name = _alias_value(
            detection,
            DETECTION_CLASS_KEYS,
            f"detections[{index}].class_name",
        )
        if not isinstance(class_name, str) or not class_name.strip():
            raise YoloResultValidationError(
                f"detections[{index}].class_name은 비어 있지 않은 문자열이어야 합니다."
            )
        confidence = _finite_number(
            _alias_value(
                detection,
                DETECTION_CONFIDENCE_KEYS,
                f"detections[{index}].confidence",
            ),
            f"detections[{index}].confidence",
            YoloResultValidationError,
        )
        if not 0.0 <= confidence <= 1.0:
            raise YoloResultValidationError(
                f"detections[{index}].confidence는 0 이상 1 이하여야 합니다."
            )
        bbox_value = _detection_bbox(detection, f"detections[{index}]")
        bbox = [
            _finite_number(
                coordinate,
                f"detections[{index}].bbox[{coordinate_index}]",
                YoloResultValidationError,
            )
            for coordinate_index, coordinate in enumerate(bbox_value)
        ]
        x1, y1, x2, y2 = bbox
        if x1 >= x2 or y1 >= y2:
            raise YoloResultValidationError(
                f"detections[{index}].bbox는 x1 < x2, y1 < y2여야 합니다."
            )
        if x1 < 0 or y1 < 0 or x2 > width or y2 > height:
            raise YoloResultValidationError(
                f"detections[{index}].bbox가 이미지 범위를 벗어났습니다."
            )
        position_value = detection.get("position")
        position = (
            _normalized_position(position_value)
            if position_value is not None
            else _position_from_bbox(bbox, width)
        )
        normalized_class = CLASS_ALIASES.get(class_name.strip().lower())
        if normalized_class is not None:
            normalized.append(
                {
                    "class_name": normalized_class,
                    "confidence": confidence,
                    "bbox": bbox,
                    "position": position,
                }
            )

    normalized.sort(key=lambda item: item["confidence"], reverse=True)
    return source, normalized


def _normalize_motion(
    value: MotionResultInput | dict[str, Any] | None,
    *,
    target: str,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    if value is None:
        return None, {
            "available": False,
            "target": None,
            "direction": "unknown",
            "speed": "unknown",
            "confidence": None,
        }
    source = _mapping(value, "motion_result", MotionResultValidationError)
    ok = source.get("ok")
    if ok is not None and not isinstance(ok, bool):
        raise MotionResultValidationError("motion_result.ok는 boolean이어야 합니다.")
    if ok is False:
        # Flow 실패는 전체 실패가 아니라 방향 정보만 사용하지 않는 상태다.
        return source, {
            "available": False,
            "target": None,
            "direction": "unknown",
            "speed": "unknown",
            "confidence": None,
        }
    available = source.get("available")
    if not isinstance(available, bool):
        raise MotionResultValidationError("available은 boolean이어야 합니다.")
    direction_value = source.get("direction")
    if not isinstance(direction_value, str):
        raise MotionResultValidationError("direction은 문자열이어야 합니다.")
    direction = DIRECTION_ALIASES.get(direction_value.strip().lower())
    if direction is None:
        raise MotionResultValidationError(
            f"지원하지 않는 direction입니다: {direction_value}"
        )
    _validate_optional_context(source, MotionResultValidationError)

    speed_value = source.get("speed")
    if speed_value is None:
        speed = "unknown"
    elif isinstance(speed_value, bool):
        raise MotionResultValidationError("speed는 숫자, unknown 또는 null이어야 합니다.")
    elif isinstance(speed_value, (int, float)):
        speed = str(_finite_number(speed_value, "speed", MotionResultValidationError))
    elif isinstance(speed_value, str) and speed_value.strip().lower() == "unknown":
        speed = "unknown"
    else:
        raise MotionResultValidationError("speed는 숫자, unknown 또는 null이어야 합니다.")

    confidence_value = source.get("confidence")
    if confidence_value is None:
        confidence = None
    else:
        confidence = _finite_number(
            confidence_value,
            "motion_result.confidence",
            MotionResultValidationError,
        )
        if not 0.0 <= confidence <= 1.0:
            raise MotionResultValidationError(
                "motion_result.confidence는 0 이상 1 이하여야 합니다."
            )

    if not available:
        direction = "unknown"
    return source, {
        "available": available,
        "target": target if available else None,
        "direction": direction,
        "speed": speed,
        "confidence": confidence if available else None,
    }


def _normalize_quality(
    value: ImageQualityInput | dict[str, Any] | None,
) -> dict[str, Any]:
    if value is None:
        return {"is_blurry": False, "blur_score": 0.0}
    source = _mapping(value, "image_quality", IntegrationError)
    is_blurry = source.get("is_blurry")
    if not isinstance(is_blurry, bool):
        raise IntegrationError("image_quality.is_blurry는 boolean이어야 합니다.")
    blur_score_value = source.get("blur_score")
    blur_score = (
        0.0
        if blur_score_value is None
        else _finite_number(blur_score_value, "blur_score", IntegrationError)
    )
    if blur_score < 0:
        raise IntegrationError("blur_score는 0 이상이어야 합니다.")
    return {"is_blurry": is_blurry, "blur_score": blur_score}


def build_vlm_metadata(
    *,
    yolo_result: YoloResultInput | dict[str, Any],
    motion_result: MotionResultInput | dict[str, Any] | None = None,
    image_quality: ImageQualityInput | dict[str, Any] | None = None,
    user_query: str = DEFAULT_USER_QUERY,
    strict_frame_sync: bool = False,
) -> dict[str, Any]:
    if not isinstance(user_query, str):
        raise IntegrationError("user_query는 문자열이어야 합니다.")
    normalized_query = user_query.strip() or DEFAULT_USER_QUERY

    yolo_source, detections = _normalize_yolo(yolo_result)
    motion_target = detections[0]["class_name"] if detections else "scene"
    motion_source, motion = _normalize_motion(
        motion_result,
        target=motion_target,
    )
    if motion_source is not None:
        yolo_frame = yolo_source.get("frame_id")
        motion_frame = motion_source.get("frame_id")
        if yolo_frame is not None and motion_frame is not None and yolo_frame != motion_frame:
            if strict_frame_sync:
                raise FrameSynchronizationError(
                    "YOLO와 motion frame_id가 일치하지 않습니다."
                )
            motion = {
                "available": False,
                "target": None,
                "direction": "unknown",
                "speed": "unknown",
                "confidence": None,
            }

    quality_input = (
        image_quality
        if image_quality is not None
        else _quality_from_yolo_payload(yolo_source)
    )

    metadata = {
        "schema_version": SCHEMA_VERSION,
        "image_path": "integration_input",
        "user_query": normalized_query,
        "image_quality": _normalize_quality(quality_input),
        "detections": detections,
        "motion": motion,
        "image_width": yolo_source["image_width"],
        "image_height": yolo_source["image_height"],
    }
    return validate_metadata(metadata)
