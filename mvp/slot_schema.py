#!/usr/bin/env python3
"""슬롯 출력 방식: VLM에게 문장이 아니라 값 몇 개만 뽑게 한다.

왜 이렇게 하나:
- 지연의 대부분이 자기회귀 디코딩이다. 이 젯슨 실측으로 SmolVLM-500M이
  10.5 tok/s이므로, 60토큰 문장은 5.6초지만 15토큰 슬롯은 1.4초다.
- 한국어 문장을 VLM이 만들 필요가 없어진다. 템플릿이 만든다. 그래서
  한국어를 못 하는 작은 모델도 쓸 수 있다.
- 값이 열거형이라 검증이 단순하다. 문장 검증(33종 거부 사유)보다
  훨씬 확실하게 환각을 막는다.
"""

from __future__ import annotations

import json
import re
from typing import Any, Optional

from class_rules import (
    CLASS_RULES,
    DIRECTION_KO,
    MOVER_DIRECTION_KO,
    POSITION_KO,
    josa,
    rule_for,
)


ALLOWED_POSITIONS = ("left", "front", "right")
ALLOWED_MOTIONS = ("approaching", "receding", "crossing_left", "crossing_right",
                   "up", "down", "still", "unknown")

# 슬롯 모션 → 우리 방향 표기
MOTION_TO_DIRECTION = {
    "approaching": "down",
    "receding": "up",
    "crossing_left": "left",
    "crossing_right": "right",
    "up": "up",
    "down": "down",
}


def build_slot_prompt(class_names: list[str]) -> str:
    """슬롯만 뽑게 하는 짧은 영어 프롬프트.

    한국어를 요구하지 않으므로 작은 모델도 따라온다. 클래스 목록을 주어
    임의의 단어를 지어내지 못하게 한다.
    """

    allowed = ", ".join(sorted(class_names))
    return (
        "Look at the image and answer with ONLY one JSON object. No other text.\n"
        '{"object": "<one of the list>", "position": "left|front|right", '
        '"motion": "approaching|receding|crossing_left|crossing_right|still|unknown"}\n'
        f"Allowed objects: {allowed}\n"
        'If nothing from the list is visible, answer {"object": "none", '
        '"position": "front", "motion": "unknown"}'
    )


JSON_PATTERN = re.compile(r"\{[^{}]*\}", re.DOTALL)


def parse_slots(raw: str) -> Optional[dict[str, str]]:
    """모델 출력에서 슬롯을 뽑는다. 값이 허용 목록 밖이면 버린다.

    작은 모델은 JSON 앞뒤에 잡소리를 붙이는 일이 잦으므로 첫 JSON 덩어리만
    찾아 쓴다. 파싱 실패는 예외가 아니라 None으로 돌려 상위가 폴백하게 한다.
    """

    if not raw:
        return None
    match = JSON_PATTERN.search(raw)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None

    obj = str(data.get("object", "")).strip().lower().replace(" ", "_")
    position = str(data.get("position", "")).strip().lower()
    motion = str(data.get("motion", "unknown")).strip().lower()

    if obj in ("", "none", "null"):
        return {"object": "none", "position": "front", "motion": "unknown"}
    if obj not in CLASS_RULES:
        # 목록 밖의 물체를 지어냈다. 신뢰할 수 없으므로 버린다.
        return None
    if position not in ALLOWED_POSITIONS:
        return None
    if motion not in ALLOWED_MOTIONS:
        motion = "unknown"

    return {"object": obj, "position": position, "motion": motion}


def render_korean(slots: dict[str, str]) -> str:
    """슬롯을 한국어 안내 문장으로 만든다. 모델은 여기 관여하지 않는다."""

    obj = slots.get("object", "none")
    if obj == "none":
        return "주변에서 확인되는 물체가 없습니다."

    korean, kind, _history, _use_motion = rule_for(obj)
    position_ko = POSITION_KO.get(slots.get("position", "front"), "정면")
    motion = slots.get("motion", "unknown")
    direction = MOTION_TO_DIRECTION.get(motion)

    if kind == "mover" and direction in MOVER_DIRECTION_KO:
        return (f"{position_ko}에서 {josa(korean, '이', '가')} "
                f"{MOVER_DIRECTION_KO[direction]}.")
    if kind == "transit" and direction in DIRECTION_KO:
        return (f"{position_ko}에 {DIRECTION_KO[direction]} 운행하는 "
                f"{josa(korean, '이', '가')} 있습니다.")
    return f"{position_ko}에 {josa(korean, '이', '가')} 있습니다."


def slots_agree_with_detections(
    slots: dict[str, str],
    detections: list[dict[str, Any]],
) -> bool:
    """슬롯이 YOLO 탐지와 어긋나지 않는지 본다.

    YOLO가 본 것이 진실에 가깝다. VLM이 탐지에 없는 물체를 말하면
    환각으로 보고 거부한다. 탐지가 아예 없을 때는 검증할 근거가 없으므로
    통과시킨다(그때는 VLM만이 유일한 정보원이다).
    """

    if slots.get("object") == "none":
        return True
    if not detections:
        return True
    seen = {str(d.get("class_name", "")).lower() for d in detections}
    return slots["object"] in seen
