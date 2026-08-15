from __future__ import annotations

import math
from typing import Any


SCHEMA_VERSION = "1.0"
MAX_USER_QUERY_LENGTH = 500
ALLOWED_POSITIONS = frozenset(
    {"left", "front", "right", "unknown"}
)
ALLOWED_DIRECTIONS = frozenset(
    {
        "up",
        "down",
        "left",
        "right",
        "opening",
        "closing",
        "stopped",
        "unknown",
    }
)


class MetadataValidationError(ValueError):
    """입력 metadata가 schema 1.0을 만족하지 않을 때 발생한다."""


def _fail(field: str, message: str) -> None:
    raise MetadataValidationError(f"{field}: {message}")


def _require_dict(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        _fail(field, "객체(dict)여야 합니다.")
    return value


def _require_finite_number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(field, "유한한 숫자여야 합니다.")

    number = float(value)
    if not math.isfinite(number):
        _fail(field, "NaN 또는 무한대가 아닌 유한한 숫자여야 합니다.")
    return number


def _validate_confidence(value: Any, field: str) -> None:
    number = _require_finite_number(value, field)
    if not 0.0 <= number <= 1.0:
        _fail(field, "0 이상 1 이하이어야 합니다.")


def _validate_optional_string(value: Any, field: str) -> None:
    if value is not None and not isinstance(value, str):
        _fail(field, "문자열 또는 null이어야 합니다.")


def validate_metadata(metadata: Any) -> dict[str, Any]:
    """metadata schema 1.0을 검증하고 원본 dict를 반환한다."""

    data = _require_dict(metadata, "metadata")

    if data.get("schema_version") != SCHEMA_VERSION:
        _fail("schema_version", f'"{SCHEMA_VERSION}"이어야 합니다.')

    image_path = data.get("image_path")
    if not isinstance(image_path, str) or not image_path.strip():
        _fail("image_path", "비어 있지 않은 문자열이어야 합니다.")

    user_query = data.get("user_query")
    if not isinstance(user_query, str):
        _fail("user_query", "문자열이어야 합니다.")
    query_length = len(user_query.strip())
    if query_length == 0:
        _fail("user_query", "비어 있으면 안 됩니다.")
    if query_length > MAX_USER_QUERY_LENGTH:
        _fail(
            "user_query",
            f"공백 제거 후 {MAX_USER_QUERY_LENGTH}자 이하여야 합니다.",
        )

    image_width = data.get("image_width")
    image_height = data.get("image_height")
    if (
        isinstance(image_width, bool)
        or not isinstance(image_width, int)
        or image_width <= 0
    ):
        _fail("image_width", "양의 정수여야 합니다.")
    if (
        isinstance(image_height, bool)
        or not isinstance(image_height, int)
        or image_height <= 0
    ):
        _fail("image_height", "양의 정수여야 합니다.")

    image_quality = _require_dict(
        data.get("image_quality"), "image_quality"
    )
    if not isinstance(image_quality.get("is_blurry"), bool):
        _fail("image_quality.is_blurry", "boolean이어야 합니다.")
    blur_score = _require_finite_number(
        image_quality.get("blur_score"), "image_quality.blur_score"
    )
    if blur_score < 0:
        _fail("image_quality.blur_score", "0 이상이어야 합니다.")

    detections = data.get("detections")
    if not isinstance(detections, list):
        _fail("detections", "list여야 합니다.")

    for index, detection_value in enumerate(detections):
        field = f"detections[{index}]"
        detection = _require_dict(detection_value, field)

        class_name = detection.get("class_name")
        if not isinstance(class_name, str) or not class_name.strip():
            _fail(f"{field}.class_name", "비어 있지 않은 문자열이어야 합니다.")

        _validate_confidence(
            detection.get("confidence"), f"{field}.confidence"
        )

        bbox = detection.get("bbox")
        if not isinstance(bbox, list) or len(bbox) != 4:
            _fail(f"{field}.bbox", "숫자 4개의 list여야 합니다.")
        coordinates = [
            _require_finite_number(value, f"{field}.bbox[{bbox_index}]")
            for bbox_index, value in enumerate(bbox)
        ]
        x1, y1, x2, y2 = coordinates
        if x1 >= x2 or y1 >= y2:
            _fail(f"{field}.bbox", "x1 < x2, y1 < y2여야 합니다.")
        if x1 < 0 or y1 < 0 or x2 > image_width or y2 > image_height:
            _fail(
                f"{field}.bbox",
                "image_width와 image_height 범위 안에 있어야 합니다.",
            )

        position = detection.get("position")
        if position not in ALLOWED_POSITIONS:
            _fail(
                f"{field}.position",
                f"허용 값은 {sorted(ALLOWED_POSITIONS)}입니다.",
            )

    motion = _require_dict(data.get("motion"), "motion")
    available = motion.get("available")
    if not isinstance(available, bool):
        _fail("motion.available", "boolean이어야 합니다.")

    direction = motion.get("direction")
    if direction not in ALLOWED_DIRECTIONS:
        _fail(
            "motion.direction",
            f"허용 값은 {sorted(ALLOWED_DIRECTIONS)}입니다.",
        )
    if not available and direction != "unknown":
        _fail(
            "motion.direction",
            'motion.available이 false이면 "unknown"이어야 합니다.',
        )

    target = motion.get("target")
    if available and (
        not isinstance(target, str) or not target.strip()
    ):
        _fail(
            "motion.target",
            "motion.available이 true이면 비어 있지 않은 문자열이어야 합니다.",
        )
    if not available and target is not None:
        _fail("motion.target", "motion.available이 false이면 null이어야 합니다.")

    _validate_optional_string(motion.get("speed"), "motion.speed")
    motion_confidence = motion.get("confidence")
    if motion_confidence is not None:
        _validate_confidence(motion_confidence, "motion.confidence")

    return data
