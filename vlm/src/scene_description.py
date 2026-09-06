from __future__ import annotations

from typing import Any

from src.quality_checker import (
    ValidationResult,
    _has_markdown_structure,
    _has_valid_guidance_endings,
    _is_korean_guidance,
    _is_question_form,
    _looks_like_json,
    _normalize_message,
    _sentence_count,
)
from src.safety_rules import (
    DISTANCE_PATTERNS,
    EXCESSIVE_ACTION_TERMS,
    MAX_MESSAGE_LENGTH,
    MAX_SENTENCES,
    SAFETY_JUDGMENT_TERMS,
    UNNATURAL_KOREAN_TERMS,
    UNNECESSARY_OUTPUT_TERMS,
    UNSUPPORTED_STATE_TERMS,
)

import re


SCENE_FALLBACK_MESSAGE = "주변 상황을 정확히 설명하기 어렵습니다."
DEFAULT_SCENE_QUERY = "주변 상황을 설명해줘."


def validate_scene_message(message: str) -> ValidationResult:
    """일반 장면 설명 문장의 형식과 과잉 주장만 검사한다.

    escalator/elevator_button처럼 detection과 대조하는 검증(위치·방향 일치,
    미지원 객체 차단)은 하지 않는다. 이 모드는 YOLO SUPPORTED_TARGETS 밖의
    객체(사람, 차량 등)를 설명하는 것이 목적이므로, 어떤 객체를 언급했는지는
    막지 않고 문장의 형식과 안전 관련 과잉 주장만 막는다.
    """

    reasons: list[str] = []
    normalized = _normalize_message(message) if isinstance(message, str) else ""

    if not normalized:
        reasons.append("empty_message")
    else:
        if _is_question_form(normalized):
            reasons.append("question_form_output")
        if any(term in normalized for term in UNNATURAL_KOREAN_TERMS):
            reasons.append("unnatural_korean_style")
        if not _has_valid_guidance_endings(normalized):
            reasons.append("invalid_guidance_ending")
        if _looks_like_json(message):
            reasons.append("json_output")
        if _has_markdown_structure(message):
            reasons.append("markdown_or_list_output")
        if len(normalized) > MAX_MESSAGE_LENGTH:
            reasons.append("message_too_long")
        if _sentence_count(normalized) > MAX_SENTENCES:
            reasons.append("too_many_sentences")
        if not _is_korean_guidance(normalized):
            reasons.append("not_korean_guidance")
        if any(
            re.search(pattern, normalized, re.IGNORECASE)
            for pattern in DISTANCE_PATTERNS
        ):
            reasons.append("unsupported_distance")
        if any(term in normalized for term in SAFETY_JUDGMENT_TERMS):
            reasons.append("unsupported_safety_judgment")
        if any(term in normalized for term in UNSUPPORTED_STATE_TERMS):
            reasons.append("unsupported_object_state")
        if any(term in normalized for term in EXCESSIVE_ACTION_TERMS):
            reasons.append("excessive_action_instruction")
        if any(
            term.lower() in normalized.lower() for term in UNNECESSARY_OUTPUT_TERMS
        ):
            reasons.append("unnecessary_output")

    unique_reasons = list(dict.fromkeys(reasons))
    is_valid = not unique_reasons
    return ValidationResult(
        is_valid=is_valid,
        reasons=unique_reasons,
        used_fallback=not is_valid,
        normalized_message=normalized,
        fallback_reason=unique_reasons[0] if unique_reasons else None,
        message_source="vlm" if is_valid else "fallback",
    )


def build_scene_result(
    message: str,
    latency_ms: float,
    peak_gpu_memory_mb: float,
    model_id: str,
) -> dict[str, Any]:
    validation = validate_scene_message(message)
    final_message = (
        validation.normalized_message if validation.is_valid else SCENE_FALLBACK_MESSAGE
    )

    return {
        "mode": "scene_description",
        "message": final_message,
        "target": None,
        "position": "unknown",
        "status": "described" if validation.is_valid else "uncertain",
        "latency_ms": latency_ms,
        "peak_gpu_memory_mb": peak_gpu_memory_mb,
        "model_id": model_id,
        "safety_validated": True,
        "used_fallback": validation.used_fallback,
        "validation_reasons": validation.reasons,
        "message_source": validation.message_source,
        "raw_vlm_message": message,
    }
