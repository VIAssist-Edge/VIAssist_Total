from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from src.metadata_schema import MetadataValidationError, validate_metadata
from src.safety_rules import (
    CONFIDENCE_THRESHOLD,
    DIRECTION_SYNONYMS,
    DISTANCE_PATTERNS,
    EXCESSIVE_ACTION_TERMS,
    GUIDANCE_ENDINGS,
    MAX_MESSAGE_LENGTH,
    MAX_SENTENCES,
    MOVEMENT_TERMS,
    OTHER_OBJECT_TERMS,
    POSITION_KO,
    POSITION_SYNONYMS,
    QUESTION_ENDINGS,
    SAFETY_JUDGMENT_TERMS,
    TARGET_MESSAGE_SYNONYMS,
    UNNECESSARY_OUTPUT_TERMS,
    UNNATURAL_KOREAN_TERMS,
    UNSUPPORTED_STATE_TERMS,
)


@dataclass(frozen=True)
class ValidationResult:
    is_valid: bool
    reasons: list[str]
    used_fallback: bool
    normalized_message: str
    fallback_reason: str | None
    message_source: str


def _normalize_message(message: str) -> str:
    return re.sub(r"\s+", " ", message).strip()


def _normalize_for_query_comparison(message: str) -> str:
    """질문 반복 비교용으로 공백, 문장부호, 대소문자 차이를 제거한다."""

    return re.sub(r"[^0-9a-z가-힣]", "", message.lower())


def _is_question_form(message: str) -> bool:
    stripped = message.rstrip().rstrip(".!… ")
    return "?" in message or any(
        stripped.endswith(ending) for ending in QUESTION_ENDINGS
    )


def _has_valid_guidance_endings(message: str) -> bool:
    sentences = [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s*", message)
        if sentence.strip()
    ]
    return bool(sentences) and all(
        sentence.endswith(GUIDANCE_ENDINGS) for sentence in sentences
    )


def _looks_like_json(message: str) -> bool:
    stripped = message.strip()
    if stripped.startswith(("{", "[")):
        return True
    try:
        return isinstance(json.loads(stripped), (dict, list))
    except (json.JSONDecodeError, TypeError):
        return False


def _has_markdown_structure(message: str) -> bool:
    if "```" in message:
        return True
    return bool(
        re.search(
            r"(?m)^\s*(?:#{1,6}\s+|[-*+]\s+|\d+[.)]\s+)",
            message,
        )
        or re.search(
            r"^\s*[가-힣A-Za-z ]{1,20}:\s*(?:\n|$)",
            message,
        )
    )


def _sentence_count(message: str) -> int:
    endings = re.findall(r"[.!?]+|(?:다|요)(?=\s|$)", message)
    return max(1, len(endings)) if message.strip() else 0


def _is_korean_guidance(message: str) -> bool:
    hangul_count = len(re.findall(r"[가-힣]", message))
    letter_count = len(re.findall(r"[A-Za-z가-힣]", message))
    return (
        hangul_count >= 5
        and letter_count > 0
        and hangul_count / letter_count >= 0.5
    )


def _matched_categories(
    message: str,
    synonyms: dict[str, tuple[str, ...]],
) -> set[str]:
    return {
        category
        for category, terms in synonyms.items()
        if any(term in message for term in terms)
    }


def validate_vlm_message(
    message: str,
    metadata: dict[str, Any],
    selected_detection: dict[str, Any] | None,
    confidence_threshold: float = CONFIDENCE_THRESHOLD,
) -> ValidationResult:
    """VLM 문장이 검증된 vision 사실만 포함하는지 보수적으로 검사한다."""

    reasons: list[str] = []

    metadata_is_valid = True
    try:
        validate_metadata(metadata)
    except (MetadataValidationError, TypeError, AttributeError):
        metadata_is_valid = False
        reasons.append("invalid_metadata")

    normalized = _normalize_message(message) if isinstance(message, str) else ""
    safe_metadata = metadata if isinstance(metadata, dict) else {}
    image_quality_value = safe_metadata.get("image_quality", {})
    image_quality = (
        image_quality_value if isinstance(image_quality_value, dict) else {}
    )
    detections_value = safe_metadata.get("detections", [])
    detections = detections_value if isinstance(detections_value, list) else []
    motion_value = safe_metadata.get("motion", {})
    motion = motion_value if isinstance(motion_value, dict) else {}

    if image_quality.get("is_blurry") is True:
        reasons.append("blurry_image")
    if not detections:
        reasons.append("no_detection")
    elif selected_detection is None:
        reasons.append("no_supported_detection")
    elif (
        metadata_is_valid
        and float(selected_detection.get("confidence", 0.0))
        < confidence_threshold
    ):
        reasons.append("low_detection_confidence")

    if not normalized:
        reasons.append("empty_message")
    else:
        if _is_question_form(normalized):
            reasons.append("question_form_output")

        normalized_query = _normalize_for_query_comparison(
            str(safe_metadata.get("user_query", ""))
        )
        normalized_output = _normalize_for_query_comparison(normalized)
        if (
            normalized_query
            and normalized_query == normalized_output
        ):
            reasons.append("repeated_user_query")
        elif (
            len(normalized_query) >= 6
            and normalized_query in normalized_output
        ):
            reasons.append("repeated_user_query")

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

        if any(re.search(pattern, normalized, re.IGNORECASE) for pattern in DISTANCE_PATTERNS):
            reasons.append("unsupported_distance")
        if any(term in normalized for term in SAFETY_JUDGMENT_TERMS):
            reasons.append("unsupported_safety_judgment")
        if any(term in normalized for term in UNSUPPORTED_STATE_TERMS):
            reasons.append("unsupported_object_state")
        if any(term in normalized for term in EXCESSIVE_ACTION_TERMS):
            reasons.append("excessive_action_instruction")
        if any(term.lower() in normalized.lower() for term in UNNECESSARY_OUTPUT_TERMS):
            reasons.append("unnecessary_output")

        if selected_detection is not None:
            target = str(selected_detection.get("class_name"))
            target_terms = TARGET_MESSAGE_SYNONYMS.get(target, ())
            if not any(term in normalized for term in target_terms):
                reasons.append("selected_object_not_mentioned")
            if any(term in normalized for term in OTHER_OBJECT_TERMS):
                reasons.append("object_not_in_metadata")

            claimed_positions = _matched_categories(
                normalized, POSITION_SYNONYMS
            )
            expected_position = selected_detection.get("position")
            if any(
                position != expected_position
                for position in claimed_positions
            ):
                reasons.append("position_mismatch")
            if expected_position == "unknown" and claimed_positions:
                reasons.append("unsupported_position")

        claimed_directions = _matched_categories(
            normalized, DIRECTION_SYNONYMS
        )
        claims_movement = claimed_directions or any(
            term in normalized for term in MOVEMENT_TERMS
        )
        if motion.get("available") is not True and claims_movement:
            reasons.append("motion_unavailable_but_claimed")
        elif motion.get("available") is True:
            expected_direction = motion.get("direction")
            if any(
                direction != expected_direction
                for direction in claimed_directions
            ):
                reasons.append("direction_mismatch")
            if (
                expected_direction == "unknown"
                and claimed_directions
            ):
                reasons.append("unsupported_direction")
            if (
                selected_detection is not None
                and selected_detection.get("class_name") == "escalator"
                and expected_direction in {"up", "down"}
                and expected_direction not in claimed_directions
            ):
                reasons.append("required_motion_direction_missing")

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


def build_safe_fallback(
    metadata: dict[str, Any],
    selected_detection: dict[str, Any] | None,
    reason: str,
) -> str:
    """모델 호출 없이 구조화 metadata만으로 결정적 안내를 만든다."""

    if reason == "invalid_metadata":
        return "입력 정보를 확인하기 어렵습니다. 다시 시도해 주세요."

    image_quality = metadata.get("image_quality", {})
    detections = metadata.get("detections", [])
    if image_quality.get("is_blurry") is True:
        return "화면이 흐려 확인하기 어렵습니다. 카메라를 잠시 고정해 주세요."
    if not detections:
        return "목표 객체를 확인하기 어렵습니다. 카메라를 천천히 좌우로 움직여 주세요."
    if selected_detection is None:
        return "대상이 보이지만 정확한 안내를 제공하기 어렵습니다."
    if float(selected_detection.get("confidence", 0.0)) < CONFIDENCE_THRESHOLD:
        return "대상으로 보이는 물체가 있지만 정확히 확인하기 어렵습니다. 가까이에서 다시 촬영해 주세요."

    target = selected_detection.get("class_name")
    position = selected_detection.get("position")
    position_ko = POSITION_KO.get(position)

    if target == "elevator_button":
        if position_ko is None:
            return "엘리베이터 버튼이 보이지만 위치를 정확히 확인하기 어렵습니다."
        return f"{position_ko}에 엘리베이터 버튼이 있습니다."

    if target == "escalator":
        if position_ko is None:
            return "에스컬레이터가 보이지만 위치를 정확히 확인하기 어렵습니다."
        motion = metadata.get("motion", {})
        if motion.get("available") is True and motion.get("direction") == "up":
            return f"{position_ko}에 위쪽으로 운행하는 에스컬레이터가 있습니다."
        if motion.get("available") is True and motion.get("direction") == "down":
            return f"{position_ko}에 아래쪽으로 운행하는 에스컬레이터가 있습니다."
        return f"{position_ko}에 에스컬레이터가 있습니다. 운행 방향은 확인하기 어렵습니다."

    return "대상이 보이지만 정확한 안내를 제공하기 어렵습니다."
