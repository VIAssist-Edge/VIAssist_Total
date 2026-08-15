"""Perception `process_split()` payload를 기존 VLM metadata 계약으로 옮긴다.

이 계층의 규칙은 `docs/perception_integration_contract.md`를 따른다. 내부
metadata schema, VLMService, Safety Validator는 변경하지 않고 입력만 정규화한다.
"""

from __future__ import annotations

from typing import Any

from src.exceptions import (
    MotionResultValidationError,
    PerceptionUnavailableError,
)
from src.metadata_adapter import DEFAULT_USER_QUERY, build_vlm_metadata

# Perception Optical Flow가 사용하는 direction 집합. 내부 metadata schema의
# direction 집합(문 열림/닫힘 포함)보다 좁다.
PERCEPTION_DIRECTIONS = frozenset({"up", "down", "stopped", "unknown"})

UNAVAILABLE_MOTION: dict[str, Any] = {
    "available": False,
    "direction": "unknown",
    "speed": "unknown",
}

YOLO_UNAVAILABLE_MESSAGE = (
    "주변 객체를 확인하기 어렵습니다. 카메라를 잠시 고정해 주세요."
)
PERCEPTION_UNAVAILABLE_MESSAGE = (
    "주변 상황을 확인하기 어렵습니다. 카메라를 잠시 고정해 주세요."
)


def is_flow_usable(flow_payload: Any) -> bool:
    """방향 안내에 사용할 수 있는 Flow payload인지 보수적으로 판정한다."""

    if not isinstance(flow_payload, dict):
        return False
    if not flow_payload.get("ok", False):
        return False
    if flow_payload.get("available") is not True:
        return False
    direction = flow_payload.get("direction")
    if not isinstance(direction, str):
        return False
    return direction.strip().lower() in (PERCEPTION_DIRECTIONS - {"unknown"})


def normalize_flow_payload(flow_payload: Any) -> dict[str, Any]:
    """Flow payload를 기존 Adapter가 받는 motion 입력으로 정규화한다.

    `available=false`, `ok=false`, 지원하지 않는 direction, `unknown`은 모두
    방향을 사용할 수 없는 상태로 내린다. 방향을 추측하지 않는다.
    """

    if flow_payload is None:
        return dict(UNAVAILABLE_MOTION)
    if not isinstance(flow_payload, dict):
        raise MotionResultValidationError(
            "flow_payload는 dict 또는 null이어야 합니다."
        )

    available = flow_payload.get("available")
    if available is not None and not isinstance(available, bool):
        # 안전 판단의 최우선 필드이므로 문자열 "true" 같은 값을 추측하지 않는다.
        raise MotionResultValidationError(
            "flow_payload.available은 boolean이어야 합니다."
        )

    motion: dict[str, Any] = dict(UNAVAILABLE_MOTION)
    frame_id = flow_payload.get("frame_id")
    if frame_id is not None:
        motion["frame_id"] = frame_id

    if not is_flow_usable(flow_payload):
        return motion

    motion["available"] = True
    motion["direction"] = str(flow_payload["direction"]).strip().lower()
    speed = flow_payload.get("speed")
    motion["speed"] = speed if speed is not None else "unknown"
    confidence = flow_payload.get("confidence")
    if confidence is not None:
        motion["confidence"] = confidence
    return motion


def build_perception_metadata(
    *,
    yolo_payload: dict[str, Any],
    flow_payload: dict[str, Any] | None = None,
    image_quality: dict[str, Any] | None = None,
    user_query: str = DEFAULT_USER_QUERY,
    strict_frame_sync: bool = False,
) -> dict[str, Any]:
    """Perception payload 쌍을 검증된 VLM metadata로 만든다."""

    return build_vlm_metadata(
        yolo_result=yolo_payload,
        motion_result=normalize_flow_payload(flow_payload),
        image_quality=image_quality,
        user_query=user_query,
        strict_frame_sync=strict_frame_sync,
    )


def build_perception_unavailable_result(
    *,
    reason: str,
    message: str,
    detail: str,
    model_id: str = "unknown",
) -> dict[str, Any]:
    """Perception을 사용할 수 없을 때의 결정적 결과 JSON을 만든다.

    VLM을 호출하지 않으므로 객체나 방향을 추측하지 않는다. 최종 JSON의 키는
    기존 결과 계약과 동일하게 유지한다.
    """

    return {
        "message": message,
        "target": None,
        "position": "unknown",
        "status": "uncertain",
        "confidence": "low",
        "detection_confidence": None,
        "latency_ms": 0.0,
        "peak_gpu_memory_mb": 0.0,
        "model_id": model_id,
        "safety_validated": True,
        "used_fallback": True,
        "validation_reasons": [reason],
        "message_source": "fallback",
        "raw_vlm_message": "",
        "service_status": "degraded",
        "error": {
            "type": "perception_error",
            "code": PerceptionUnavailableError.default_error_code,
            "message": detail,
            "retryable": True,
        },
        "fallback_reason": reason,
    }
