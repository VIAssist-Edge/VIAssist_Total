from __future__ import annotations

from typing import Any


CONFIDENCE_THRESHOLD = 0.5
MAX_MESSAGE_LENGTH = 120
MAX_SENTENCES = 2

SUPPORTED_TARGETS = frozenset({"elevator_button", "escalator"})

TARGET_QUERY_KEYWORDS = {
    "elevator_button": ("엘리베이터 버튼", "승강기 버튼", "버튼"),
    "escalator": ("에스컬레이터",),
}

TARGET_MESSAGE_SYNONYMS = {
    "elevator_button": (
        "엘리베이터 버튼",
        "엘리베이터 호출 버튼",
        "승강기 버튼",
        "승강기 호출 버튼",
    ),
    "escalator": ("에스컬레이터",),
}

POSITION_SYNONYMS = {
    "left": ("왼쪽", "좌측"),
    "front": ("정면", "앞쪽", "앞"),
    "right": ("오른쪽", "우측"),
}

POSITION_KO = {
    "left": "왼쪽",
    "front": "정면",
    "right": "오른쪽",
}

DIRECTION_SYNONYMS = {
    "up": ("위쪽", "위로", "상행", "올라가는", "올라가고", "위 방향"),
    "down": ("아래쪽", "아래로", "하행", "내려가는", "내려가고", "아래 방향"),
    "left": ("왼쪽으로 움직", "왼쪽으로 이동", "좌측으로 이동"),
    "right": ("오른쪽으로 움직", "오른쪽으로 이동", "우측으로 이동"),
    "opening": ("열리고", "열리는 중", "개방 중", "열립니다", "열려 있"),
    "closing": ("닫히고", "닫히는 중", "폐쇄 중", "닫힙니다", "닫혀 있"),
    "stopped": ("정지해", "정지 중", "멈춰", "멈춘"),
}

QUESTION_ENDINGS = (
    "있나요",
    "인가요",
    "보이나요",
    "맞나요",
    "어디인가요",
    "어디에 있나요",
    "할까요",
    "되나요",
)

UNNATURAL_KOREAN_TERMS = (
    "있는 것입니다",
    "보이는 것입니다",
    "위치해 있는 것입니다",
    "존재하는 것입니다",
    "있는 것으로 보입니다",
    "것으로 확인됩니다",
)

# 생성문에 허용하는 짧은 TTS 안내 종결. fallback은 요청 문장을 포함할 수 있어
# 아래의 FALLBACK_ONLY_GUIDANCE_ENDINGS를 추가로 사용할 수 있다.
GUIDANCE_ENDINGS = (
    "있습니다.",
    "보입니다.",
    "확인됩니다.",
    "운행합니다.",
    "열려 있습니다.",
    "닫혀 있습니다.",
    "확인하기 어렵습니다.",
)

FALLBACK_ONLY_GUIDANCE_ENDINGS = ("주세요.",)

MOVEMENT_TERMS = tuple(
    dict.fromkeys(
        synonym
        for synonyms in DIRECTION_SYNONYMS.values()
        for synonym in synonyms
    )
) + (
    "운행 중",
    "운행합니다",
    "운행하고",
    "운행하는",
    "움직이고",
    "움직이는",
    "이동 중",
)

# 지원 대상 외 객체를 추가로 단정하는 흔한 표현을 보수적으로 차단한다.
OTHER_OBJECT_TERMS = (
    "계단",
    "자동차",
    "차량",
    "사람",
    "자전거",
    "오토바이",
    "횡단보도",
    "신호등",
    "장애물",
    "문",
    "의자",
    "소화전",
)

DISTANCE_PATTERNS = (
    r"\d+(?:\.\d+)?\s*(?:cm|m|센티미터|미터)",
    r"(?:가까이|가까운|가깝습니다|멀리|먼 곳|멉니다)",
)

SAFETY_JUDGMENT_TERMS = (
    "안전합니다",
    "안전해요",
    "안전하지 않습니다",
    "위험합니다",
    "위험해요",
    "위험하지 않습니다",
    "안전한 상태",
    "위험한 상태",
)

UNSUPPORTED_STATE_TERMS = (
    "고장났",
    "고장입",
    "정상 작동",
    "작동 중",
    "사용 가능",
    "켜져 있",
    "꺼져 있",
)

EXCESSIVE_ACTION_TERMS = (
    "탑승해도 됩니다",
    "지나가도 됩니다",
    "바로 탑승하세요",
    "즉시 탑승하세요",
    "버튼을 누르세요",
    "바로 버튼을",
    "탑승하세요",
    "지나가세요",
    "누르십시오",
)

UNNECESSARY_OUTPUT_TERMS = (
    "저는 AI",
    "저는 인공지능",
    "언어 모델",
    "죄송합니다",
    "미안합니다",
    "이미지를 분석",
    "이미지를 보면",
    "배경에는",
    "배경에",
    "설명드리",
)


def infer_requested_target(user_query: str) -> str | None:
    """간단한 고정 keyword 규칙으로 사용자 요청 target을 찾는다."""

    for target in ("elevator_button", "escalator"):
        if any(
            keyword in user_query
            for keyword in TARGET_QUERY_KEYWORDS[target]
        ):
            return target
    return None


def select_detection(metadata: dict[str, Any]) -> dict[str, Any] | None:
    """지원 대상 중 질의 일치와 confidence 순으로 대표 detection을 고른다."""

    detections = metadata.get("detections", [])
    supported = [
        detection
        for detection in detections
        if detection.get("class_name") in SUPPORTED_TARGETS
    ]
    requested_target = infer_requested_target(
        metadata.get("user_query", "")
    )

    if requested_target is not None:
        supported = [
            detection
            for detection in supported
            if detection.get("class_name") == requested_target
        ]

    if not supported:
        return None

    return max(
        supported,
        key=lambda detection: float(detection["confidence"]),
    )
