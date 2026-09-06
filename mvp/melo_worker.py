#!/usr/bin/env python3
"""MeloTTS 한국어 합성 워커. `~/melo_env` 안에서 실행된다.

MeloTTS는 transformers 4.27.4를 요구하는데 메인 환경은 4.49.0을 쓴다(로컬
SmolVLM 경로가 AutoModelForImageTextToText를 필요로 함). 그래서 같은
프로세스에서 import할 수 없고, 이 워커를 별도 프로세스로 띄워 파이프로 쓴다.

모델 로드가 5초대라 요청마다 새로 띄우면 안 된다. 한 번 띄워 두고
stdin으로 JSON 한 줄씩 받아 처리한다.

    입력  {"text": "정면에 사람이 있습니다.", "path": "/tmp/a.wav", "speed": 1.0}
    출력  {"ok": true, "path": "...", "synthesize_ms": 712.3, "audio_seconds": 3.2}
"""

from __future__ import annotations

import json
import os
import sys
import time
import wave


def main() -> None:
    device = os.environ.get("MELO_DEVICE", "cuda")

    try:
        from melo.api import TTS
    except Exception as error:  # noqa: BLE001
        print(json.dumps({"ok": False, "ready": False, "error": str(error)}),
              flush=True)
        return

    started = time.perf_counter()
    try:
        tts = TTS(language="KR", device=device)
        speaker_ids = tts.hps.data.spk2id
        speaker_name = next(iter(getattr(speaker_ids, "__dict__", {"KR": 0})))
        speaker_id = getattr(speaker_ids, speaker_name)
    except Exception as error:  # noqa: BLE001
        print(json.dumps({"ok": False, "ready": False, "error": str(error)}),
              flush=True)
        return

    load_ms = (time.perf_counter() - started) * 1000

    # 첫 합성은 커널 초기화 때문에 느리다. 준비 신호를 보내기 전에 한 번 돌린다.
    try:
        tts.tts_to_file("준비", speaker_id, "/tmp/_melo_warmup.wav",
                        speed=1.0, quiet=True)
    except Exception:  # noqa: BLE001
        pass

    print(json.dumps({
        "ok": True, "ready": True, "device": device,
        "speaker": speaker_name, "load_ms": round(load_ms, 1),
    }), flush=True)

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
        except json.JSONDecodeError as error:
            print(json.dumps({"ok": False, "error": f"bad json: {error}"}),
                  flush=True)
            continue

        text = str(request.get("text", "")).strip()
        path = str(request.get("path", "")).strip()
        speed = float(request.get("speed", 1.0))
        if not text or not path:
            print(json.dumps({"ok": False, "error": "text/path 필요"}), flush=True)
            continue

        try:
            started = time.perf_counter()
            tts.tts_to_file(text, speaker_id, path, speed=speed, quiet=True)
            synth_ms = (time.perf_counter() - started) * 1000
            with wave.open(path) as handle:
                audio_s = handle.getnframes() / handle.getframerate()
            print(json.dumps({
                "ok": True, "path": path,
                "synthesize_ms": round(synth_ms, 1),
                "audio_seconds": round(audio_s, 2),
            }), flush=True)
        except Exception as error:  # noqa: BLE001 - 요청 단위로 격리한다
            print(json.dumps({"ok": False, "error": str(error)}), flush=True)


if __name__ == "__main__":
    main()
