#!/usr/bin/env python3
"""검증 실패·폴백 사유를 사용자가 들을 수 있는 한국어 안내로 바꾼다.

Safety Validator가 VLM 답변을 버리면 사용자에게는 일반적인 폴백 문장만
들린다. 왜 제대로 안내하지 못했는지 모르면 사용자가 대응할 수 없으므로
(카메라를 고정한다든지, 가까이 간다든지) 사유를 짧게 덧붙인다.

원칙:
- 사용자가 행동으로 바꿀 수 있는 사유를 먼저 고른다.
- 내부 품질 문제는 한 문장으로 뭉뚱그린다. 시각장애인에게 오류 코드를
  읽어 주는 것은 도움이 되지 않는다.
- 사유가 여러 개면 우선순위가 가장 높은 하나만 말한다.
"""

from __future__ import annotations

from typing import Any, Optional


# 사용자가 대응할 수 있는 사유. 위에서부터 우선한다.
ACTIONABLE_NOTICES: tuple[tuple[str, str], ...] = (
    ("blurry_image", "화면이 흐립니다. 카메라를 잠시 고정해 주세요."),
    ("low_detection_confidence", "물체가 흐릿하게 보입니다. 조금 더 가까이 가주세요."),
    ("no_supported_detection", "주변에서 확인되는 물체가 없습니다."),
    ("no_detection", "주변에서 확인되는 물체가 없습니다."),
)

# 서비스 쪽 사유(fallback_reason). 역시 사용자가 알아야 하는 것들이다.
SERVICE_NOTICES: dict[str, str] = {
    "vlm_timeout": "안내 만드는 데 시간이 오래 걸립니다. 다시 요청해 주세요.",
    "service_unavailable": "안내 기능을 지금 사용할 수 없습니다.",
    "vlm_inference_error": "안내를 만들지 못했습니다. 다시 요청해 주세요.",
    "vlm_image_error": "화면을 읽지 못했습니다. 카메라를 확인해 주세요.",
    "metadata_invalid": "입력 정보를 확인하기 어렵습니다.",
}

# 문장 형식이 규칙에 맞지 않아 버린 경우.
FORMAT_REASONS = frozenset({
    "empty_message",
    "question_form_output",
    "repeated_user_query",
    "invalid_guidance_ending",
    "json_output",
    "markdown_or_list_output",
    "message_too_long",
    "too_many_sentences",
    "not_korean_guidance",
    "unnatural_korean_style",
    "unnecessary_output",
})
FORMAT_NOTICE = "안내 문장을 확인하지 못해 기본 안내로 대체했습니다."

# 확인되지 않은 내용을 주장해서 버린 경우. 안전 측면에서 중요한 차단이다.
HALLUCINATION_REASONS = frozenset({
    "selected_object_not_mentioned",
    "object_not_in_metadata",
    "position_mismatch",
    "unsupported_position",
    "motion_unavailable_but_claimed",
    "direction_mismatch",
    "unsupported_direction",
    "required_motion_direction_missing",
    "unsupported_distance",
    "unsupported_safety_judgment",
    "unsupported_object_state",
    "excessive_action_instruction",
})
HALLUCINATION_NOTICE = "확인되지 않은 내용이 있어 안내를 제한했습니다."


def build_failure_notice(vlm_result: Optional[dict[str, Any]]) -> Optional[str]:
    """폴백이 일어난 이유를 한 문장으로 돌려준다. 정상이면 None."""

    if not isinstance(vlm_result, dict):
        return None
    if not vlm_result.get("used_fallback"):
        return None

    reasons = vlm_result.get("validation_reasons") or []
    reason_set = {str(r) for r in reasons}

    # 1순위: 사용자가 행동으로 해결할 수 있는 것
    for key, notice in ACTIONABLE_NOTICES:
        if key in reason_set:
            return notice

    # 2순위: 서비스 상태
    fallback_reason = vlm_result.get("fallback_reason")
    if fallback_reason in SERVICE_NOTICES:
        return SERVICE_NOTICES[fallback_reason]

    # 3순위: 안전 차단 (형식 문제보다 사용자에게 의미가 크다)
    if reason_set & HALLUCINATION_REASONS:
        return HALLUCINATION_NOTICE

    # 4순위: 형식 문제
    if reason_set & FORMAT_REASONS:
        return FORMAT_NOTICE

    return None
