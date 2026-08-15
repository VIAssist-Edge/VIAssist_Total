#!/usr/bin/env python3
"""Webcam-to-Gemini web test for the viassist Jetson MVP.

개발 실험 전용 도구다. 외부 API를 사용하고 온디바이스 안전 경로
(`VLMService.infer_safe()` + Safety Validator)를 거치지 않으므로, production
안내 경로로 사용하면 안 된다. 실제 안내 경로는 `vlm_webcam_test.py`의 guide
모드와 `escalator_mvp.py --enable-vlm`이다.

This script is a standalone alternative to vlm_webcam_test.py.
It uses a free-tier Gemini model (default: gemini-2.5-flash) via the
Gemini REST API and keeps the same webcam + Flask flow.

Run:
    export GEMINI_API_KEY=your_api_key
    python3 vlm_webcam_test_gemini.py
    # Open http://<jetson-ip>:5002 from another device.
"""

from __future__ import annotations

import argparse
import atexit
import base64
import json
import logging
import os
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import cv2
import numpy as np
from flask import Flask, Response, jsonify, render_template_string, request


LOGGER = logging.getLogger("vlm_webcam_test_gemini")


class LatestFrameCamera:
    """Read continuously so Gemini inference always captures a recent frame."""

    def __init__(self, index: int, width: int, height: int, fps: int) -> None:
        self.index = index
        self.width = width
        self.height = height
        self.fps = fps
        self._frame: Optional[np.ndarray] = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._capture: Optional[cv2.VideoCapture] = None

    def start(self) -> None:
        capture = cv2.VideoCapture(self.index)
        if not capture.isOpened():
            raise RuntimeError(f"카메라를 열 수 없습니다: index={self.index}")
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        capture.set(cv2.CAP_PROP_FPS, self.fps)
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self._capture = capture
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()

    def _read_loop(self) -> None:
        while not self._stop.is_set():
            if self._capture is None:
                break
            ok, frame = self._capture.read()
            if ok and frame is not None:
                with self._lock:
                    self._frame = frame
            else:
                LOGGER.warning("카메라 프레임 읽기 실패")
                time.sleep(0.05)

    def latest(self) -> Optional[np.ndarray]:
        with self._lock:
            return None if self._frame is None else self._frame.copy()

    def jpeg(self, quality: int = 80) -> Optional[bytes]:
        frame = self.latest()
        if frame is None:
            return None
        ok, encoded = cv2.imencode(
            ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality]
        )
        return encoded.tobytes() if ok else None

    def stop(self) -> None:
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        if self._capture is not None:
            self._capture.release()
            self._capture = None


class GeminiWebcamVLM:
    def __init__(self, camera: LatestFrameCamera, model: str, api_key: str) -> None:
        self.camera = camera
        self.model = model
        self.api_key = api_key
        self.inference_lock = threading.Lock()

    def infer(self, query: str) -> dict[str, object]:
        query = query.strip()
        if not query:
            raise ValueError("질문을 입력해 주세요.")
        if len(query) > 500:
            raise ValueError("질문은 500자 이하로 입력해 주세요.")

        frame = self.camera.latest()
        if frame is None:
            raise RuntimeError("아직 카메라 프레임이 준비되지 않았습니다.")

        ok, encoded = cv2.imencode(".jpg", frame)
        if not ok:
            raise RuntimeError("캡처 이미지를 변환하지 못했습니다.")

        prompt = (
            "당신은 시각장애인의 주변 확인을 돕는 비전 어시스턴트입니다. "
            "제공된 한 장의 이미지만 근거로 한국어로 짧고 명확하게 답하세요. "
            "보이지 않거나 확실하지 않은 내용은 추측하지 말고 확인하기 어렵다고 답하세요. "
            f"사용자 요청: {query}"
        )

        started = time.perf_counter()
        with self.inference_lock:
            result_text = self._query_gemini(prompt, encoded.tobytes())
        latency_ms = round((time.perf_counter() - started) * 1000, 2)

        return {
            "query": query,
            "result": result_text,
            "latency_ms": latency_ms,
            "model_id": self.model,
            "captured_image": "data:image/jpeg;base64,"
            + base64.b64encode(encoded.tobytes()).decode("ascii"),
        }

    def _query_gemini(self, prompt: str, image_bytes: bytes) -> str:
        base64_image = base64.b64encode(image_bytes).decode("ascii")
        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": prompt},
                        {
                            "inlineData": {
                                "mimeType": "image/jpeg",
                                "data": base64_image,
                            }
                        },
                    ]
                }
            ],
            "generationConfig": {
                "temperature": 0.2,
                "maxOutputTokens": 1000,
            },
        }

        models = [self.model]
        if self.model != "gemini-2.5-flash":
            models.append("gemini-2.5-flash")
        if self.model != "gemini-2.0-flash":
            models.append("gemini-2.0-flash")

        last_error: Optional[Exception] = None
        for model_id in models:
            try:
                response_text = self._call_model(model_id, payload)
                return response_text
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                LOGGER.warning("Gemini 모델 %s 실패: %s", model_id, exc)

        raise RuntimeError(f"Gemini 호출 실패: {last_error}") if last_error else RuntimeError("Gemini 호출 실패")

    def _call_model(self, model_id: str, payload: dict[str, object]) -> str:
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/{model_id}:generateContent"
            f"?key={self.api_key}"
        )
        data = json.dumps(payload).encode("utf-8")
        request = Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")

        try:
            with urlopen(request, timeout=90) as response:
                body = response.read().decode("utf-8")
        except HTTPError as error:
            detail = error.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"HTTP {error.code}: {detail}") from error
        except URLError as error:
            raise RuntimeError(f"네트워크 오류: {error.reason}") from error

        try:
            parsed = json.loads(body)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Gemini 응답 파싱 실패: {body}") from exc

        candidates = parsed.get("candidates") or []
        if not candidates:
            raise RuntimeError(f"Gemini 응답이 비어 있습니다: {body}")

        parts = candidates[0].get("content", {}).get("parts", [])
        if not parts:
            raise RuntimeError(f"Gemini 응답 텍스트가 없습니다: {body}")

        text_chunks = []
        for part in parts:
            if isinstance(part, dict) and part.get("text"):
                text_chunks.append(part["text"])
        if not text_chunks:
            raise RuntimeError(f"Gemini 응답 텍스트가 없습니다: {body}")
        return "".join(text_chunks).strip()


HTML_PAGE = """
<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>ViAssist Webcam Gemini Test</title>
  <style>
    body { margin: 0; background: #101114; color: #f3f4f6; font-family: sans-serif; }
    main { width: min(1050px, 94vw); margin: 24px auto; }
    .panel { background: #1b1d22; padding: 18px; border-radius: 12px; margin-top: 16px; }
    .camera { width: 100%; border-radius: 8px; background: #000; }
    form { display: flex; gap: 8px; }
    input { flex: 1; padding: 13px; font-size: 16px; border-radius: 8px; border: 1px solid #555; }
    button { padding: 0 20px; border: 0; border-radius: 8px; background: #36a269; color: white; font-weight: bold; }
    button:disabled { opacity: .5; }
    #capture { display: none; width: min(480px, 100%); margin-top: 12px; border-radius: 8px; }
    #answer { white-space: pre-wrap; line-height: 1.6; }
    #meta { color: #aeb4bf; font-size: 14px; }
  </style>
</head>
<body>
  <main>
    <h1>Webcam Gemini Test</h1>
    <div class="panel"><img class="camera" src="/video_feed" alt="webcam live stream"></div>
    <div class="panel">
      <form id="form">
        <input id="query" maxlength="500" value="지금 정면에 무엇이 보이는지 알려줘." required>
        <button id="submit" type="submit">캡처 후 실행</button>
      </form>
      <img id="capture" alt="Gemini에 전달된 캡처">
      <h2>Gemini 결과</h2>
      <div id="answer">질문을 입력하고 버튼을 누르세요.</div>
      <p id="meta"></p>
    </div>
  </main>
  <script>
    const form = document.getElementById('form');
    const button = document.getElementById('submit');
    form.addEventListener('submit', async (event) => {
      event.preventDefault();
      button.disabled = true;
      document.getElementById('answer').textContent = '캡처 및 추론 중...';
      document.getElementById('meta').textContent = '';
      try {
        const response = await fetch('/infer', {
          method: 'POST', headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({query: document.getElementById('query').value})
        });
        const data = await response.json();
        if (!response.ok) throw new Error(data.error || '추론 실패');
        document.getElementById('answer').textContent = data.result;
        document.getElementById('meta').textContent = `${data.model_id} | ${data.latency_ms} ms`;
        const capture = document.getElementById('capture');
        capture.src = data.captured_image;
        capture.style.display = 'block';
      } catch (error) {
        document.getElementById('answer').textContent = `오류: ${error.message}`;
      } finally { button.disabled = false; }
    });
  </script>
</body>
</html>
"""


def create_app(engine: GeminiWebcamVLM) -> Flask:
    app = Flask(__name__)

    @app.get("/")
    def index() -> str:
        return render_template_string(HTML_PAGE)

    @app.get("/video_feed")
    def video_feed() -> Response:
        def frames():
            while True:
                jpeg = engine.camera.jpeg()
                if jpeg is not None:
                    yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
                time.sleep(0.04)

        return Response(frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

    @app.post("/infer")
    def infer() -> tuple[Response, int] | Response:
        data = request.get_json(silent=True) or {}
        try:
            return jsonify(engine.infer(str(data.get("query", ""))))
        except ValueError as error:
            return jsonify(error=str(error)), 400
        except Exception as error:
            LOGGER.exception("Gemini 추론 실패")
            return jsonify(error=str(error)), 500

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Jetson webcam Gemini web test")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--camera-fps", type=int, default=30)
    parser.add_argument("--port", type=int, default=5002)
    parser.add_argument(
        "--model",
        default="gemini-2.5-flash",
        help="Gemini 모델 이름 (예: gemini-2.5-flash, gemini-2.0-flash)",
    )
    parser.add_argument(
        "--api-key-env",
        default="GEMINI_API_KEY",
        help="API 키가 저장된 환경변수 이름",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    args = parse_args()

    api_key = os.environ.get(args.api_key_env) or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise SystemExit(
            f"API 키가 없습니다. {args.api_key_env} 또는 GOOGLE_API_KEY 환경변수를 설정하세요."
        )

    camera = LatestFrameCamera(args.camera, args.width, args.height, args.camera_fps)
    camera.start()
    atexit.register(camera.stop)
    try:
        engine = GeminiWebcamVLM(camera, args.model, api_key)
        LOGGER.info("웹 페이지: http://0.0.0.0:%d", args.port)
        create_app(engine).run(host="0.0.0.0", port=args.port, threaded=True, use_reloader=False)
    finally:
        camera.stop()


if __name__ == "__main__":
    main()
