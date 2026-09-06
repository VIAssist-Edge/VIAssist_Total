from __future__ import annotations

import json
from typing import Any


POSITION_MAP = {
    "left": "왼쪽",
    "front": "정면",
    "right": "오른쪽",
    "unknown": "알 수 없음",
}

DIRECTION_MAP = {
    "up": "위쪽",
    "down": "아래쪽",
    "left": "왼쪽",
    "right": "오른쪽",
    "opening": "열리는 중",
    "closing": "닫히는 중",
    "stopped": "정지",
    "unknown": "알 수 없음",
}


def build_prompt(metadata: dict[str, Any]) -> str:
    """
    YOLO 및 Optical Flow 결과를 VLM용 프롬프트로 변환한다.
    안전 관련 구조화 정보는 VLM이 임의로 변경하지 않도록 명시한다.
    """

    user_query = str(
        metadata.get("user_query", "주변 상황을 알려줘.")
    )

    detections = metadata.get("detections", [])
    motion = metadata.get("motion", {})
    image_quality = metadata.get("image_quality", {})

    normalized_detections: list[dict[str, Any]] = []

    for detection in detections:
        position = str(detection.get("position", "unknown"))

        entry = {
            "class_name": detection.get(
                "class_name",
                "unknown",
            ),
            "confidence": detection.get("confidence", 0.0),
            "position": position,
            "position_ko": POSITION_MAP.get(
                position,
                "알 수 없음",
            ),
            "bbox": detection.get("bbox", []),
        }
        # 객체별 움직임을 그 객체 옆에 붙여서 보낸다. 전역 motion 하나만
        # 보내던 때는 어느 물체가 움직이는지 VLM이 알 수 없었다.
        detection_motion = detection.get("motion")
        if isinstance(detection_motion, dict) and detection_motion.get("available"):
            motion_direction = str(detection_motion.get("direction", "unknown"))
            entry["motion"] = {
                "direction": motion_direction,
                "direction_ko": DIRECTION_MAP.get(motion_direction, "알 수 없음"),
            }
        normalized_detections.append(entry)

    direction = str(motion.get("direction", "unknown"))

    structured_data = {
        "user_query": user_query,
        "image_quality": image_quality,
        "detections": normalized_detections,
        "motion": {
            "available": motion.get("available", False),
            "direction": direction,
            "direction_ko": DIRECTION_MAP.get(
                direction,
                "알 수 없음",
            ),
            "speed": motion.get("speed", "unknown"),
        },
    }

    metadata_text = json.dumps(
        structured_data,
        ensure_ascii=False,
        indent=2,
    )

    return f"""
You are an assistant for safe walking guidance for a visually impaired user.

Analyze the image together with the structured vision analysis below.

Structured vision analysis:
{metadata_text}

Follow these rules strictly:

1. Answer in Korean.
2. Use only one or two short sentences.
3. Include only information needed for walking.
4. Use 왼쪽, 정면, 오른쪽 for object position.
5. The object class and position from detections take priority over your visual guess.
6. Escalator direction or door movement must only come from motion data.
7. Do not invent distance, movement, safety, or object state.
8. If information is uncertain, say "확인하기 어렵습니다."
9. Do not describe irrelevant background details.
10. Return only the guidance sentence, not JSON and not an explanation.
11. 사용자에게 질문하지 마세요.
12. 사용자 질문을 그대로 반복하지 마세요.
13. 반드시 평서형 안내 문장으로 답하세요.
14. "~있는 것입니다" 같은 번역체를 사용하지 마세요.
15. "~있습니다" 형태의 자연스러운 안내 문장을 사용하세요.
16. Optical Flow 방향이 제공되면 에스컬레이터 안내에 반드시 포함하세요.
17. detections[].motion이 있으면 그 물체가 움직이는 방향이라는 뜻입니다.
    보행에 영향을 주는 움직임만 골라 안내에 반영하세요.

User question:
{user_query}
""".strip()


def build_scene_prompt(user_query: str) -> str:
    """YOLO SUPPORTED_TARGETS와 무관하게 화면 전체를 설명하는 프롬프트.

    detection·motion 구조화 데이터를 대조하지 않으므로, escalator_button 등
    특정 시설이 아니라 사람·차량 등 화면에 보이는 일반적인 상황을 설명할 때
    쓴다. 검증되지 않은 사실(거리, 안전 여부, 상태)은 여전히 금지한다.
    """

    return f"""
You are an assistant for safe walking guidance for a visually impaired user.

Describe the general scene in the image so the user understands what is
currently around them.

Follow these rules strictly:

1. Answer in Korean.
2. Use only one or two short sentences.
3. Mention only people, vehicles, or objects you can clearly see in the image.
4. Use 왼쪽, 정면, 오른쪽 if you need to describe where something is.
5. Do not invent distance, speed, or physical measurements.
6. Do not state whether a situation or object is safe or dangerous.
7. Do not give strong action commands such as "건너세요" or "타세요".
8. If nothing meaningful is visible or the image is unclear, say "확인하기 어렵습니다."
9. Do not describe irrelevant background details.
10. Return only the guidance sentence, not JSON and not an explanation.
11. 사용자에게 질문하지 마세요.
12. 사용자 질문을 그대로 반복하지 마세요.
13. 반드시 평서형 안내 문장으로 답하세요.
14. "~있는 것입니다" 같은 번역체를 사용하지 마세요.
15. "~있습니다" 형태의 자연스러운 안내 문장을 사용하세요.

User question:
{user_query}
""".strip()
