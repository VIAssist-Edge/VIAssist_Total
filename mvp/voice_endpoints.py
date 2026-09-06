#!/usr/bin/env python3
"""음성 입출력(STT/TTS)을 웹 API로 노출하는 얇은 경계.

`escalator_mvp.py`를 크게 건드리지 않으려고 라우트 등록을 이 모듈로 분리했다.
모델은 프로세스 시작 시 한 번만 로드하고, 추론은 사용자의 명시적 요청이
있을 때만 실행한다. VLM 경계(`vlm_bridge`)와 같은 원칙을 따른다.

경로:
  POST /voice/listen  마이크로 한 번 듣고 텍스트만 돌려준다(발화 종료 후 일괄).
  GET  /voice/stream  말하는 도중 부분 인식 결과를 SSE로 흘려보낸다.
  POST /voice/say     {"text": "..."} 를 음성으로 재생한다.
  POST /voice/ask     듣기 -> VLM 안내 -> 말하기를 한 번에 수행한다.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Optional

from flask import Flask, Response, jsonify, request, stream_with_context


LOGGER = logging.getLogger(__name__)


class VoiceServices:
    """STT/TTS 인스턴스를 함께 들고 있는 묶음."""

    def __init__(self, stt: Any, tts: Any) -> None:
        self.stt = stt
        self.tts = tts

    @classmethod
    def from_args(cls, args: Any) -> "VoiceServices":
        """설정에 따라 STT/TTS를 한 번씩 만든다. 호출 비용이 크다."""

        from stt_service import STTService
        from tts_service import TTSService

        stt = STTService(
            model_size=getattr(args, "stt_model", "base"),
            device=getattr(args, "stt_device", "cpu"),
            compute_type=getattr(args, "stt_compute_type", "int8"),
            language=getattr(args, "stt_language", "ko"),
            noise_multiplier=getattr(args, "stt_noise_multiplier", 2.5),
        )
        tts = TTSService(
            language=getattr(args, "tts_language", "ko"),
            alsa_device=getattr(args, "tts_alsa_device", None),
            engine=getattr(args, "tts_engine", "melo"),
            melo_device=getattr(args, "tts_melo_device", "cuda"),
            speed=getattr(args, "tts_speed", 1.0),
        )
        return cls(stt, tts)


# 말을 못 알아들었을 때 사용자에게 돌려주는 문장. VLM을 부르지 않는다.
NOT_HEARD_MESSAGE = "말씀을 알아듣지 못했습니다. 다시 말씀해 주세요."


def _compose_spoken_text(message: str, notice: Optional[str]) -> str:
    """안내 문장 뒤에 실패 사유를 덧붙인다.

    사유를 아예 말하지 않으면 사용자는 왜 안내가 부실한지 알 수 없고,
    카메라를 고정한다든지 하는 대응을 할 수 없다.
    """

    if not notice:
        return message
    if not message:
        return notice
    return f"{message} {notice}"


def register_voice_routes(app: Flask, engine: Any) -> None:
    """`engine.voice`가 있을 때만 음성 경로를 등록한다."""

    voice: Optional[VoiceServices] = getattr(engine, "voice", None)
    if voice is None:
        return

    def _unavailable() -> Response:
        return jsonify(error="음성 기능이 비활성화되어 있습니다."), 503

    @app.post("/voice/listen")
    def voice_listen() -> Response:
        from stt_service import STTBusyError

        data = request.get_json(silent=True) or {}
        try:
            result = voice.stt.listen(
                max_seconds=float(data.get("max_seconds", 8.0)),
                start_timeout_s=float(data.get("start_timeout_s", 4.0)),
            )
        except STTBusyError as error:
            return jsonify(error=str(error)), 429
        except Exception as error:  # noqa: BLE001 - 요청 단위로 격리한다
            LOGGER.exception("음성 인식 실패")
            return jsonify(error=f"음성 인식에 실패했습니다: {error}"), 500
        return jsonify(result)

    @app.get("/voice/stream")
    def voice_stream() -> Response:
        """부분 인식 결과를 Server-Sent Events로 내보낸다.

        발화가 끝날 때까지 기다리지 않고 중간 결과를 먼저 보여 주므로
        체감 지연이 줄어든다. 마지막에 type=final 이벤트가 한 번 온다.
        """

        from stt_service import STTBusyError

        max_seconds = float(request.args.get("max_seconds", 8.0))
        start_timeout_s = float(request.args.get("start_timeout_s", 4.0))

        def events():
            try:
                for event in voice.stt.listen_streaming(
                    max_seconds=max_seconds,
                    start_timeout_s=start_timeout_s,
                ):
                    yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
            except STTBusyError as error:
                payload = {"type": "error", "error": str(error)}
                yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
            except Exception as error:  # noqa: BLE001
                LOGGER.exception("스트리밍 인식 실패")
                payload = {"type": "error", "error": str(error)}
                yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

        return Response(
            stream_with_context(events()),
            mimetype="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.post("/voice/say")
    def voice_say() -> Response:
        data = request.get_json(silent=True) or {}
        text = str(data.get("text", "")).strip()
        if not text:
            return jsonify(error="재생할 문장이 없습니다."), 400
        try:
            result = voice.tts.speak(text)
        except Exception as error:  # noqa: BLE001
            LOGGER.exception("음성 재생 실패")
            return jsonify(error=f"음성 재생에 실패했습니다: {error}"), 500
        return jsonify(result)

    @app.post("/voice/guide")
    def voice_guide() -> Response:
        """규칙 기반 안내를 만들어 바로 읽어 준다. VLM 호출 없음."""

        import time as _time

        data = request.get_json(silent=True) or {}
        started = _time.perf_counter()
        result = engine.build_rule_guidance(
            max_items=int(data.get("max_items", 2))
        )
        guide_ms = (_time.perf_counter() - started) * 1000

        message = result.get("message", "")
        spoken = voice.tts.speak(message) if message else None
        total_ms = (_time.perf_counter() - started) * 1000

        return jsonify(
            guide=result,
            spoken=spoken,
            message=message,
            guide_ms=round(guide_ms, 3),
            total_ms=round(total_ms, 1),
        )

    @app.post("/voice/ask")
    def voice_ask() -> Response:
        """마이크로 질문을 받아 VLM 안내를 만들고 그대로 읽어 준다."""

        from stt_service import STTBusyError
        from vlm_bridge import VLMBusyError

        if engine.vlm_bridge is None:
            return jsonify(error="VLM이 비활성화되어 있습니다."), 503

        data = request.get_json(silent=True) or {}
        # auto(기본): guidance_router가 규칙/VLM(상태)/VLM(장면) 중 하나로 보낸다.
        # rule/vlm_state/vlm_scene: 경로 강제. scene/guidance: 예전 값(VLM 직접 호출) 유지.
        mode = str(data.get("mode", "auto"))

        try:
            # 스트리밍으로 들으면 말하는 동안 인식이 병행되므로 발화 종료
            # 시점에 남는 처리량이 적다. 부분 결과는 여기서 쓰지 않고
            # 마지막 final 이벤트만 취한다.
            heard = {}
            for event in voice.stt.listen_streaming(
                max_seconds=float(data.get("max_seconds", 8.0)),
                start_timeout_s=float(data.get("start_timeout_s", 4.0)),
            ):
                if event.get("type") == "final":
                    heard = event
        except STTBusyError as error:
            return jsonify(error=str(error)), 429
        except Exception as error:  # noqa: BLE001
            LOGGER.exception("음성 인식 실패")
            return jsonify(error=f"음성 인식에 실패했습니다: {error}"), 500

        query = heard.get("text", "").strip()
        if not query:
            spoken = voice.tts.speak(NOT_HEARD_MESSAGE)
            return jsonify(
                heard=heard,
                vlm=None,
                spoken=spoken,
                message=NOT_HEARD_MESSAGE,
            )

        routed = None
        LOGGER.info("음성 질문 인식: %r (mode=%s, %.2fs)", query, mode, heard.get("audio_seconds", 0.0))
        try:
            import guidance_router

            # 예전 UI가 mode=scene/guidance를 보내도 시연용 고정 응답은 우선 적용한다.
            if guidance_router.find_scripted(query) is not None:
                mode = "auto"
            if mode in ("auto", "rule", "vlm_state", "vlm_scene", "scripted"):

                routed = guidance_router.run(engine, query, mode=mode)
                # 규칙 경로면 vlm_result가 없다. 아래 failure_notice가 안전하게 넘어가도록 빈 dict.
                vlm_result = routed.get("vlm") or {"message": routed.get("message", ""), "status": "rule"}
            elif mode == "guidance":
                vlm_result = engine.request_vlm_guidance(query)
            else:
                vlm_result = engine.request_vlm_scene_description(query)
        except VLMBusyError as error:
            return jsonify(error=str(error)), 429
        except ValueError as error:
            return jsonify(error=str(error)), 400
        except RuntimeError as error:
            return jsonify(error=str(error)), 503
        except Exception as error:  # noqa: BLE001
            LOGGER.exception("VLM 요청 실패")
            return jsonify(error=f"VLM 요청에 실패했습니다: {error}"), 500

        message = vlm_result.get("message", "")

        # 가드레일에 걸려 폴백이 나갔으면 그 사유도 함께 읽어 준다.
        notice = None
        if bool(data.get("announce_failures", True)):
            from failure_notice import build_failure_notice

            notice = build_failure_notice(vlm_result)

        spoken_text = _compose_spoken_text(message, notice)
        spoken = voice.tts.speak(spoken_text) if spoken_text else None

        return jsonify(
            heard=heard,
            vlm=vlm_result,
            spoken=spoken,
            message=message,
            failure_notice=notice,
            spoken_text=spoken_text,
            route=(routed or {}).get("route", mode),
            route_reason=(routed or {}).get("reason"),
            route_question=(routed or {}).get("question"),
        )
