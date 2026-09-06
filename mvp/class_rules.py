#!/usr/bin/env python3
"""클래스별 안내 규칙 표.

`best.pt`의 33개 클래스 각각에 대해 (1) 한국어 표기, (2) 안내 문장 형태,
(3) 방향 안정화에 쓸 히스토리 길이, (4) 움직임을 안내에 반영할지를 정한다.

히스토리 길이를 클래스마다 다르게 두는 이유:
에스컬레이터는 느리고 일정한 방향으로 움직이므로 길게 잡아 오판을 줄이는 게
낫다. 반대로 사람·차량은 빠르게 방향이 바뀌므로 12프레임(450ms)이면 이미
지나간 정보가 된다. 26.7fps 기준 프레임당 37.5ms다.

문장 형태(kind)의 의미:
  fixture   고정 시설물. 위치만 안내한다. 움직임은 의미 없다.
  transit   운행 방향이 안내의 핵심. 방향을 못 구하면 그렇게 말한다.
  mover     스스로 움직이는 대상. 접근 여부가 중요하다.
  obstacle  통행을 막는 장애물. 존재와 위치가 핵심.
  signal    상태를 읽어야 의미가 있는 것. 상태는 VLM 몫으로 남긴다.
"""

from __future__ import annotations

from typing import Any, Optional


# 기본값. 표에 없는 클래스가 들어와도 안전하게 동작한다.
DEFAULT_HISTORY = 8
DEFAULT_KIND = "obstacle"

# (한국어명, 문장형태, 히스토리 길이, 움직임 반영 여부)
CLASS_RULES: dict[str, tuple[str, str, int, bool]] = {
    # ── 운행 시설: 방향이 안내의 핵심. 느리고 일정하므로 길게 본다 ──
    "escalator":               ("에스컬레이터", "transit", 12, True),
    "elevator_button":         ("엘리베이터 버튼", "fixture", 12, False),
    "disp_up":                 ("상행 표시", "signal", 12, False),
    "disp_down":               ("하행 표시", "signal", 12, False),

    # ── 문 상태: 열림/닫힘이 바뀌는 순간이 중요하므로 중간 길이 ──
    "open":                    ("열린 문", "signal", 6, True),
    "close":                   ("닫힌 문", "signal", 6, True),
    "middle":                  ("반쯤 열린 문", "signal", 6, True),

    # ── 개찰구·키오스크: 고정 시설 ──
    "subway_ticket_gate_all":  ("개찰구", "fixture", 10, False),
    "subway_ticket_gate_each": ("개찰구", "fixture", 10, False),
    "s_button":                ("버튼", "fixture", 10, False),
    "s_display":               ("표시등", "signal", 10, False),
    "kiosk":                   ("키오스크", "fixture", 10, False),

    # ── 움직이는 것: 빠르게 반응해야 하므로 짧게 ──
    "person":                  ("사람", "mover", 5, True),
    "bicycle":                 ("자전거", "mover", 4, True),
    "motorcycle":              ("오토바이", "mover", 4, True),
    "car":                     ("승용차", "mover", 4, True),
    "bus":                     ("버스", "mover", 4, True),
    "truck":                   ("트럭", "mover", 4, True),
    "stroller":                ("유모차", "mover", 5, True),
    "carrier":                 ("캐리어", "mover", 6, True),

    # ── 고정 장애물: 움직이지 않으므로 방향 판정이 불필요 ──
    "bollard":                 ("볼라드", "obstacle", 10, False),
    "pole":                    ("기둥", "obstacle", 10, False),
    "tree_trunk":              ("나무", "obstacle", 10, False),
    "fire_hydrant":            ("소화전", "obstacle", 10, False),
    "barricade":               ("바리케이드", "obstacle", 10, False),
    "movable_signage":         ("입간판", "obstacle", 8, False),
    "potted_plant":            ("화분", "obstacle", 10, False),
    "bench":                   ("벤치", "obstacle", 10, False),
    "chair":                   ("의자", "obstacle", 10, False),
    "table":                   ("탁자", "obstacle", 10, False),

    # ── 교통 표지: 상태 판독은 VLM 몫 ──
    "traffic_light":           ("신호등", "signal", 10, False),
    "traffic_sign":            ("교통 표지판", "fixture", 10, False),
    "stop":                    ("정지 표지", "fixture", 10, False),
}

# 방향 → 한국어. mover는 접근/이동 표현을 따로 쓴다.
DIRECTION_KO = {
    "up": "위쪽으로",
    "down": "아래쪽으로",
    "left": "왼쪽으로",
    "right": "오른쪽으로",
}
MOVER_DIRECTION_KO = {
    "up": "멀어지고 있습니다",
    "down": "다가오고 있습니다",
    "left": "왼쪽으로 지나가고 있습니다",
    "right": "오른쪽으로 지나가고 있습니다",
}
POSITION_KO = {"left": "왼쪽", "front": "정면", "right": "오른쪽"}

# 보행 안전에 더 급한 것이 앞에 오도록. 같은 프레임에 여러 개가 잡히면
# 이 순서로 무엇을 먼저 말할지 정한다.
KIND_PRIORITY = {"mover": 0, "transit": 1, "obstacle": 2, "signal": 3, "fixture": 4}


def has_batchim(word: str) -> bool:
    """마지막 글자에 받침이 있는지. 한글 음절은 (코드-0xAC00)%28로 종성을 안다."""

    if not word:
        return False
    last = word[-1]
    if not ("가" <= last <= "힣"):
        return False
    return (ord(last) - 0xAC00) % 28 != 0


def josa(word: str, with_batchim: str, without_batchim: str) -> str:
    """받침에 맞는 조사를 붙여 돌려준다. '승용차이' 같은 오류를 막는다."""

    return word + (with_batchim if has_batchim(word) else without_batchim)


def rule_for(class_name: str) -> tuple[str, str, int, bool]:
    class_name = normalize_class(class_name)
    return CLASS_RULES.get(
        class_name, (class_name, DEFAULT_KIND, DEFAULT_HISTORY, False)
    )


# YOLO 원문 클래스명 → CLASS_RULES 키. 그 외는 소문자·밑줄로만 정규화한다(vlm/src/metadata_adapter와 같은 규칙).
CLASS_ALIASES = {
    "escalator_falling": "escalator",
    "moving_stairs": "escalator",
    "elevator-button": "elevator_button",
    "lift_button": "elevator_button",
}
# center_offset(-1 왼쪽 끝 … +1 오른쪽 끝) → 위치. 화면을 셋으로 나눈다.
POSITION_SPLIT = 0.33


def normalize_class(raw: Any) -> str:
    name = str(raw or "").strip().lower().replace(" ", "_").replace("-", "_")
    return CLASS_ALIASES.get(name, name)


def normalize_detection(detection: dict[str, Any]) -> dict[str, Any]:
    """두 표기의 detection을 하나로 맞춘다.

    - 내부 표기: class_name / confidence / position
    - 스냅샷 계약(perception_payload): cls_name / conf / center_offset·clock
    둘 다 받아 class_name(정규화)·confidence·position·motion을 채운 새 dict를 돌려준다.
    """

    raw_name = detection.get("class_name", detection.get("cls_name", ""))
    confidence = detection.get("confidence", detection.get("conf", 0.0))
    try:
        confidence = float(confidence or 0.0)
    except (TypeError, ValueError):
        confidence = 0.0

    position = detection.get("position")
    if position not in POSITION_KO:
        offset = detection.get("center_offset")
        if isinstance(offset, (int, float)):
            position = "left" if offset <= -POSITION_SPLIT else "right" if offset >= POSITION_SPLIT else "front"
        else:
            position = "unknown"

    normalized = dict(detection)
    normalized["class_name"] = normalize_class(raw_name)
    normalized["confidence"] = confidence
    normalized["position"] = position
    return normalized


def history_for(class_name: str) -> int:
    """이 클래스의 방향 안정화 히스토리 길이."""

    return rule_for(class_name)[2]


def korean_name(class_name: str) -> str:
    return rule_for(class_name)[0]


def urgency(detection: dict[str, Any]) -> tuple[int, float]:
    """정렬용 키. 낮을수록 먼저 말한다."""

    detection = normalize_detection(detection)
    _ko, kind, _hist, _use_motion = rule_for(detection.get("class_name", ""))
    return (KIND_PRIORITY.get(kind, 9), -float(detection.get("confidence", 0.0)))


def describe_detection(detection: dict[str, Any]) -> Optional[str]:
    """detection 하나를 한국어 한 문장으로 만든다. 모델 호출 없음."""

    detection = normalize_detection(detection)
    class_name = str(detection.get("class_name", ""))
    korean, kind, _history, use_motion = rule_for(class_name)

    position = str(detection.get("position", "unknown"))
    position_ko = POSITION_KO.get(position)

    motion = detection.get("motion") or {}
    direction = str(motion.get("direction", "unknown"))
    has_motion = bool(motion.get("available")) and direction in DIRECTION_KO

    if position_ko is None:
        return f"{josa(korean, '이', '가')} 보이지만 위치를 확인하기 어렵습니다."

    if kind == "transit":
        if use_motion and has_motion:
            return (f"{position_ko}에 {DIRECTION_KO[direction]} 운행하는 "
                    f"{josa(korean, '이', '가')} 있습니다.")
        return (f"{position_ko}에 {josa(korean, '이', '가')} 있습니다. "
                "운행 방향은 확인하기 어렵습니다.")

    if kind == "mover":
        if use_motion and has_motion:
            return (f"{position_ko}에서 {josa(korean, '이', '가')} "
                    f"{MOVER_DIRECTION_KO[direction]}.")
        return f"{position_ko}에 {josa(korean, '이', '가')} 있습니다."

    if kind == "signal":
        if use_motion and has_motion:
            return f"{position_ko}의 {josa(korean, '이', '가')} 움직이고 있습니다."
        return f"{position_ko}에 {josa(korean, '이', '가')} 있습니다."

    # fixture, obstacle
    return f"{position_ko}에 {josa(korean, '이', '가')} 있습니다."


def build_guidance(
    detections: list[dict[str, Any]],
    *,
    max_items: int = 2,
    min_confidence: float = 0.5,
) -> str:
    """탐지 목록 전체를 한두 문장의 안내로 만든다.

    급한 것부터 최대 `max_items`개만 말한다. 시각장애인에게 화면의 모든
    물체를 나열하는 것은 도움이 되지 않는다.
    """

    usable = [
        d for d in (normalize_detection(x) for x in detections)
        if d["confidence"] >= min_confidence
    ]
    if not usable:
        return "주변에서 확인되는 물체가 없습니다."

    usable.sort(key=urgency)
    sentences = []
    for detection in usable[:max_items]:
        sentence = describe_detection(detection)
        if sentence and sentence not in sentences:
            sentences.append(sentence)
    return " ".join(sentences)
