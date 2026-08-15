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

        normalized_detections.append(
            {
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
        )

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

User question:
{user_query}
""".strip()
