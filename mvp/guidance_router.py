#!/usr/bin/env python3
"""안내 라우터: 질문과 YOLO 탐지를 보고 규칙 / VLM(상태) / VLM(장면) 중 하나로 보낸다.

왜 필요한가:
- 규칙 경로(`class_rules`)는 지연 ~0ms지만 학습된 33클래스의 존재·위치·방향만 말할 수 있다.
- VLM은 상태 판독(문 열림/닫힘, 표시등 방향, 글자)과 학습 범위 밖 장면에 필요하지만
  젯슨에서 3~10초가 걸리고, "위험한 거 있어?" 같은 열린 질문에는 약하다(2026-09-05 벤치).
- 그래서 "YOLO가 잡으면 규칙, 상태를 물으면 VLM에 짧은 상태 질문, 아무것도 없거나
  장면을 물으면 VLM 장면 설명"으로 갈라 준다. 어느 경로로 갔는지(route/reason)를 항상 남긴다.

세 경로:
  rule       탐지 결과를 class_rules 템플릿으로 문장화. VLM 호출 없음.
  vlm_state  YOLO 컨텍스트를 붙여 VLM에 "상태만" 짧게 묻는다(engine.request_vlm_guidance).
  vlm_scene  화면 전체 설명(engine.request_vlm_scene_description).
"""

from __future__ import annotations

import logging
import re
import time
from typing import Any, Optional

from flask import Flask, Response, jsonify, request

from class_rules import KIND_PRIORITY, korean_name, normalize_detection, rule_for

LOGGER = logging.getLogger(__name__)

# 규칙 경로가 신뢰할 최소 신뢰도. class_rules.build_guidance 기본값과 같다.
RULE_MIN_CONFIDENCE = 0.5

# 상태를 묻는 말. 이 단어가 있으면 탐지가 있어도 VLM(상태)로 보낸다.
STATE_PATTERNS = re.compile(
    r"열렸|열려|닫혔|닫혀|작동|움직|멈췄|멈추|정지|켜졌|꺼졌|몇\s*층|어느\s*층|색|"
    r"표시|글자|읽어|써\s*있|적혀|표지판|안내문|번호|방향이\s*어디|올라가|내려가|상행|하행"
)
# 장면 전체를 묻는 말.
SCENE_PATTERNS = re.compile(
    r"뭐가\s*있|무엇이\s*있|뭐\s*있|주변|풍경|설명|보여|앞에\s*뭐|여기\s*어디|어디\s*(야|에요|인가|지)|"
    r"위험|조심|장애물|사람\s*많|붐비"
)
# 규칙 안내를 그대로 원하는 말(짧은 트리거).
GENERIC_PATTERNS = re.compile(r"^(안내|알려\s*줘|알려줘|뭐야|어때|상황|지금)?[\s.?!]*$")

# signal 클래스별로 VLM에 물을 상태 질문. 짧고 닫힌 질문일수록 8B/4B가 잘 맞혔다.
# 데모용 고정 응답. 질문이 패턴에 맞으면 탐지·VLM과 무관하게 이 문장을 그대로 말한다.
# 시연 흐름을 통제하기 위한 것이며 실제 인식 결과가 아니다 — 대시보드에는 route=scripted 로 표시된다.
# delay_s: 진짜 처리처럼 보이도록 답하기 전에 잠깐 기다리는 시간(±jitter).
SCRIPTED_ANSWERS = [
    {
        "id": "exit_where",
        # 공백·문장부호를 지운 문자열에 대고 맞춘다. STT가 "출구"를 "축구/출고/츨구"로,
        # "어디 있어"를 "어딨어/어딧어/어디에있어"로 적는 경우까지 잡는다.
        "pattern": re.compile(r"(출구|축구|출고|츨구|출귀)(가|는|이|를|요)?(어디|어딨|어딧|어느|있|찾|위치)"),
        "message": "오른쪽에 출구가 보이고 오른쪽으로 천천히 직진하시면 됩니다. 중간에 부딪힐 위험이 있으니 조심하여 천천히 걷길 바랍니다.",
        "delay_s": 1.0,
        "jitter_s": 0.25,
    },
]


_NORMALIZE = re.compile(r"[\s.,!?~…\"'()\[\]{}:;·-]+")


def normalize_query(text: str) -> str:
    """STT 출력의 공백·문장부호를 지워 패턴 매칭을 안정시킨다."""

    return _NORMALIZE.sub("", text or "").lower()


def find_scripted(query: Optional[str]) -> Optional[dict[str, Any]]:
    text = normalize_query(query or "")
    if not text:
        return None
    for entry in SCRIPTED_ANSWERS:
        if entry["pattern"].search(text):
            return entry
    return None


SIGNAL_QUESTIONS = {
    "open": "엘리베이터 문이 열려 있는지, 닫혀 있는지, 닫히는 중인지 한 문장으로 답하세요.",
    "close": "엘리베이터 문이 열려 있는지, 닫혀 있는지, 열리는 중인지 한 문장으로 답하세요.",
    "middle": "엘리베이터 문이 열리는 중인지 닫히는 중인지 한 문장으로 답하세요.",
    "disp_up": "표시등이 위(상행)인지 아래(하행)인지 한 문장으로 답하세요.",
    "disp_down": "표시등이 위(상행)인지 아래(하행)인지 한 문장으로 답하세요.",
    "s_display": "표시등에 무엇이 표시되어 있는지(층수, 방향, 글자) 한 문장으로 답하세요.",
    "traffic_light": "보행 신호등이 무슨 색인지 한 문장으로 답하세요.",
}
DEFAULT_SIGNAL_QUESTION = "이 장치의 현재 상태를 한 문장으로 답하세요."


def _usable_detections(snapshot: Optional[dict[str, Any]]) -> list[dict[str, Any]]:
    if not snapshot:
        return []
    # 스냅샷 계약은 cls_name/conf/center_offset 표기다. class_rules.normalize_detection이 두 표기를 맞춘다.
    detections = (snapshot.get("yolo_payload") or {}).get("detections", [])
    usable = [d for d in (normalize_detection(x) for x in detections) if d["confidence"] >= RULE_MIN_CONFIDENCE]
    usable.sort(key=lambda d: (KIND_PRIORITY.get(rule_for(str(d.get("class_name", "")))[1], 9),
                               -float(d.get("confidence", 0.0))))
    return usable


def decide(query: Optional[str], snapshot: Optional[dict[str, Any]], *, vlm_available: bool) -> dict[str, Any]:
    """어느 경로로 갈지만 정한다. 모델 호출 없음, 순수 함수.

    반환: {"route": rule|vlm_state|vlm_scene, "reason": str, "question": str|None,
           "target": {"class_name", "kind", "confidence"}|None, "detections": n}
    """

    text = (query or "").strip()
    usable = _usable_detections(snapshot)
    top = usable[0] if usable else None
    top_kind = rule_for(str(top.get("class_name", "")))[1] if top else None
    target = (
        {"class_name": top.get("class_name"), "korean": korean_name(str(top.get("class_name", ""))),
         "kind": top_kind, "confidence": round(float(top.get("confidence", 0.0)), 3)}
        if top else None
    )

    def result(route: str, reason: str, question: Optional[str] = None) -> dict[str, Any]:
        if route != "rule" and not vlm_available:
            return {"route": "rule", "reason": f"{reason} → VLM 비활성이라 규칙으로 대체", "question": None,
                    "target": target, "detections": len(usable)}
        return {"route": route, "reason": reason, "question": question, "target": target, "detections": len(usable)}

    # 0) 데모 고정 응답 — 패턴이 맞으면 무조건 이 경로. VLM 가용 여부와 무관.
    scripted = find_scripted(text)
    if scripted is not None:
        return {"route": "scripted", "reason": f"데모 고정 응답 '{scripted['id']}' 패턴 일치", "question": text,
                "target": target, "detections": len(usable), "scripted_id": scripted["id"]}

    # 1) 사용자가 상태를 물었다 → 탐지가 있든 없든 VLM 상태 질문. YOLO 컨텍스트는 함께 넘어간다.
    if text and STATE_PATTERNS.search(text):
        return result("vlm_state", "질문에 상태 단어(열림/작동/표시/글자 등)가 있음", text)

    # 2) 이미 잡힌 물체의 이름을 말했다("에스컬레이터 어디야") → 위치·방향은 규칙이 0ms로 답한다.
    if text:
        for d in usable:
            name = korean_name(str(d.get("class_name", "")))
            if name and name in text:
                return result("rule", f"질문에 탐지된 '{name}'이(가) 있음 → 위치/방향은 규칙으로", None)

    # 3) 장면을 물었다 → VLM 장면 설명.
    if text and SCENE_PATTERNS.search(text):
        return result("vlm_scene", "질문이 장면/주변/위험을 묻는 열린 질문", text)

    # 4) 질문이 없거나 "안내해줘" 수준 → 탐지로 결정.
    generic = not text or bool(GENERIC_PATTERNS.match(text))
    if top is None:
        return result("vlm_scene", "YOLO 탐지 없음(conf≥%.2f) → 장면 설명" % RULE_MIN_CONFIDENCE, text or None)
    if top_kind == "signal":
        # 표시등·문 상태·신호등: 존재는 YOLO가 알지만 상태는 VLM이 읽어야 한다.
        question = SIGNAL_QUESTIONS.get(str(top.get("class_name", "")), DEFAULT_SIGNAL_QUESTION)
        return result("vlm_state", f"최우선 탐지 '{target['korean']}'가 상태 판독 대상(signal)", question)
    if generic:
        return result("rule", f"최우선 탐지 '{target['korean']}'({top_kind})는 규칙으로 충분 — VLM 생략", None)
    # 5) 탐지도 있고 질문도 있는데 상태/장면 단어가 없다 → 규칙 답 + 질문은 VLM 장면으로 보낼 수도 있으나,
    #    지연을 우선해 규칙으로 답한다. reason에 남겨 대시보드에서 확인할 수 있게 한다.
    return result("rule", f"질문 '{text[:20]}'에 상태/장면 단어 없음 → 탐지 기반 규칙 안내", None)


def run(engine: Any, query: Optional[str], *, mode: str = "auto") -> dict[str, Any]:
    """결정하고 실행한다. mode를 rule/vlm_state/vlm_scene으로 강제할 수도 있다."""

    started = time.perf_counter()
    snapshot = engine.get_snapshot()
    vlm_available = getattr(engine, "vlm_bridge", None) is not None
    # 어떤 mode로 왔든 고정 응답 패턴이면 그쪽이 우선 — 예전 UI가 mode=scene을 보내도 시연 문장이 나가야 한다.
    if mode != "scripted" and find_scripted(query) is not None:
        mode = "auto"
    LOGGER.info("라우터 입력: query=%r mode=%s", (query or "")[:80], mode)
    if mode == "auto":
        decision = decide(query, snapshot, vlm_available=vlm_available)
    else:
        decision = {"route": mode, "reason": f"강제 mode={mode}", "question": query,
                    "target": None, "detections": len(_usable_detections(snapshot))}
    decide_ms = (time.perf_counter() - started) * 1000

    route = decision["route"]
    t = time.perf_counter()
    vlm_result: Optional[dict[str, Any]] = None
    if route == "scripted":
        import random

        scripted = find_scripted(decision.get("question") or query) or SCRIPTED_ANSWERS[0]
        # 실제 추론처럼 보이도록 약 1초(±jitter) 기다린 뒤 답한다. TTS는 문장 단위로 끊어 읽는다.
        time.sleep(max(0.0, scripted["delay_s"] + random.uniform(-scripted["jitter_s"], scripted["jitter_s"])))
        message = scripted["message"]
        source = "scripted"
    elif route == "rule":
        rule = engine.build_rule_guidance()
        message = rule.get("message", "")
        source = "rule"
    elif route == "vlm_state":
        vlm_result = engine.request_vlm_guidance(decision.get("question") or query)
        message = vlm_result.get("message", "")
        source = "vlm_state"
    else:
        vlm_result = engine.request_vlm_scene_description(decision.get("question") or query)
        message = vlm_result.get("message", "")
        source = "vlm_scene"
    exec_ms = (time.perf_counter() - t) * 1000

    LOGGER.info("라우터: %s (%s) %.0fms", route, decision["reason"], exec_ms)
    return {
        "route": route,
        "reason": decision["reason"],
        "question": decision.get("question"),
        "target": decision.get("target"),
        "detections": decision.get("detections"),
        "message": message,
        "source": source,
        "vlm": vlm_result,
        "decide_ms": round(decide_ms, 2),
        "exec_ms": round(exec_ms, 1),
        "total_ms": round((time.perf_counter() - started) * 1000, 1),
    }


def register_router_routes(app: Flask, engine: Any) -> None:
    """텍스트로 라우터를 직접 쓰는 경로. 음성은 voice_endpoints의 /voice/ask(mode=auto)가 쓴다."""

    @app.post("/guide/auto")
    def guide_auto() -> Response:
        data = request.get_json(silent=True) or {}
        query = data.get("query")
        mode = str(data.get("mode", "auto"))
        if mode not in ("auto", "rule", "vlm_state", "vlm_scene"):
            return jsonify(error=f"mode는 auto|rule|vlm_state|vlm_scene 중 하나: {mode}"), 400
        try:
            return jsonify(run(engine, query, mode=mode))
        except RuntimeError as error:
            return jsonify(error=str(error)), 503
        except Exception as error:  # noqa: BLE001
            LOGGER.exception("라우터 실행 실패")
            return jsonify(error=f"라우터 실행 실패: {error}"), 500

    @app.post("/guide/decide")
    def guide_decide() -> Response:
        """실행하지 않고 어느 경로로 갈지만 알려 준다(대시보드 미리보기용)."""

        data = request.get_json(silent=True) or {}
        snapshot = engine.get_snapshot()
        return jsonify(decide(data.get("query"), snapshot, vlm_available=getattr(engine, "vlm_bridge", None) is not None))
