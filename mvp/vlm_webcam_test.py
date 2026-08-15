#!/usr/bin/env python3
"""Webcam-to-VLM web test for the viassist Jetson MVP.

두 가지 모드를 명확히 구분한다.

1. guide (기본)
   Perception payload → Metadata Adapter → VLMService.infer_safe() →
   Safety Validator → 최종 JSON. 실제 제품 경로와 같다. 사용자에게 보여주는
   문장은 검증이 끝난 `message`뿐이다. YOLO payload가 없으면 detection이
   없는 상태로 처리하며, 보이는 객체를 추측하지 않는다.

2. dev_describe (`--allow-dev-describe`로만 활성화)
   안전 정책이 아직 지원하지 않는 자유 형식 장면 설명을 모델에서 직접 받는
   개발 전용 모드다. 결과는 `dev_raw_text`로만 반환하며 사용자 안내나 TTS에
   사용하면 안 된다.

Run:
    python3 vlm_webcam_test.py
    # Open http://<jetson-ip>:5001 from another device.
"""

from __future__ import annotations

import argparse
import base64
import logging
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
from flask import Flask, Response, jsonify, render_template_string, request

from perception_payload import build_flow_payload, build_yolo_payload
from vlm_bridge import DEFAULT_CONFIG_PATH, VLMBridge, VLMBusyError


LOGGER = logging.getLogger("vlm_webcam_test")

DEV_MODE_WARNING = (
    "개발 전용 출력입니다. 안전 검증을 거치지 않았으므로 사용자 안내나 TTS에 "
    "사용하면 안 됩니다."
)


class LatestFrameCamera:
    """Read continuously so VLM inference always captures a recent frame."""

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
        """여러 번 호출해도 안전하다."""

        if self._stop.is_set() and self._capture is None:
            return
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        self._thread = None
        if self._capture is not None:
            self._capture.release()
            self._capture = None


class WebcamVLM:
    def __init__(
        self,
        camera: LatestFrameCamera,
        config_path: Path,
        *,
        allow_dev_describe: bool = False,
        timeout_seconds: float = 10.0,
    ) -> None:
        self.camera = camera
        self.allow_dev_describe = allow_dev_describe
        LOGGER.info("VLM 로딩 중: %s", config_path)
        self.bridge = VLMBridge.from_config(
            config_path,
            timeout_seconds=timeout_seconds,
        )
        LOGGER.info("VLM 로딩 완료: %s", self.bridge.model_id)
        self._frame_counter = 0
        self._counter_lock = threading.Lock()

    def _next_frame_id(self) -> int:
        with self._counter_lock:
            self._frame_counter += 1
            return self._frame_counter

    def _capture(self) -> tuple[np.ndarray, bytes]:
        frame = self.camera.latest()
        if frame is None:
            raise RuntimeError("아직 카메라 프레임이 준비되지 않았습니다.")
        ok, encoded = cv2.imencode(".jpg", frame)
        if not ok:
            raise RuntimeError("캡처 이미지를 변환하지 못했습니다.")
        return frame, encoded.tobytes()

    @staticmethod
    def _data_url(image_bytes: bytes) -> str:
        return "data:image/jpeg;base64," + base64.b64encode(image_bytes).decode(
            "ascii"
        )

    @staticmethod
    def _optional_payload(value: Any, field: str) -> Optional[dict[str, Any]]:
        if value is None:
            return None
        if not isinstance(value, dict):
            raise ValueError(f"{field}는 JSON 객체여야 합니다.")
        return value

    def guide(
        self,
        query: Optional[str],
        *,
        yolo_payload: Any = None,
        flow_payload: Any = None,
    ) -> dict[str, Any]:
        """제품과 동일한 안전 경로로 안내 문장을 만든다."""

        frame, image_bytes = self._capture()
        frame_h, frame_w = frame.shape[:2]
        frame_id = self._next_frame_id()

        yolo = self._optional_payload(yolo_payload, "yolo_payload")
        flow = self._optional_payload(flow_payload, "flow_payload")
        if yolo is None:
            # 이 도구에는 YOLO가 없다. detection을 지어내지 않고 빈 목록으로 둔다.
            yolo = build_yolo_payload(
                frame_id=frame_id,
                timestamp=time.time(),
                image_width=frame_w,
                image_height=frame_h,
                detections=(),
            )
            flow = flow or build_flow_payload(
                frame_id=frame_id,
                image_width=frame_w,
                image_height=frame_h,
                direction="unknown",
            )

        result = self.bridge.describe(
            frame=frame,
            yolo_payload=yolo,
            flow_payload=flow,
            user_query=query,
        )
        result["mode"] = "guide"
        result["captured_image"] = self._data_url(image_bytes)
        return result

    def dev_describe(self, query: Optional[str]) -> dict[str, Any]:
        """개발 전용 자유 형식 설명. 안전 검증을 거치지 않는다."""

        if not self.allow_dev_describe:
            raise PermissionError(
                "개발 모드가 꺼져 있습니다. --allow-dev-describe로 실행하세요."
            )
        normalized_query = VLMBridge.normalize_query(query)
        frame, image_bytes = self._capture()

        prompt = (
            "당신은 시각장애인의 주변 확인을 돕는 비전 어시스턴트입니다. "
            "제공된 한 장의 이미지만 근거로 한국어로 짧고 명확하게 답하세요. "
            "보이지 않거나 확실하지 않은 내용은 추측하지 말고 확인하기 어렵다고 답하세요. "
            f"사용자 요청: {normalized_query}"
        )

        started = time.perf_counter()
        engine = self.bridge.pipeline.service.engine
        with self.bridge.inference_lock, tempfile.TemporaryDirectory(
            prefix="viassist_vlm_dev_"
        ) as temp_dir:
            image_path = Path(temp_dir) / "capture.jpg"
            image_path.write_bytes(image_bytes)
            raw_text = engine.generate(
                image_path=image_path,
                prompt=prompt,
                metadata={"user_query": normalized_query, "source": "dev_describe"},
            )
        latency_ms = round((time.perf_counter() - started) * 1000, 2)

        return {
            "mode": "dev_describe",
            "user_query": normalized_query,
            "dev_raw_text": raw_text,
            "safety_validated": False,
            "message_source": "raw_vlm_dev_mode",
            "warning": DEV_MODE_WARNING,
            "request_latency_ms": latency_ms,
            "model_id": self.bridge.model_id,
            "captured_image": self._data_url(image_bytes),
        }


HTML_PAGE = """
<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>ViAssist Webcam VLM Test</title>
  <style>
    body { margin: 0; background: #101114; color: #f3f4f6; font-family: sans-serif; }
    main { width: min(1050px, 94vw); margin: 24px auto; }
    .panel { background: #1b1d22; padding: 18px; border-radius: 12px; margin-top: 16px; }
    .camera { width: 100%; border-radius: 8px; background: #000; }
    form { display: flex; gap: 8px; flex-wrap: wrap; }
    input { flex: 1; min-width: 240px; padding: 13px; font-size: 16px;
            border-radius: 8px; border: 1px solid #555; }
    button { padding: 0 20px; border: 0; border-radius: 8px; background: #36a269;
             color: white; font-weight: bold; }
    button.dev { background: #a2543a; }
    button:disabled { opacity: .5; }
    #capture { display: none; width: min(480px, 100%); margin-top: 12px; border-radius: 8px; }
    #answer { white-space: pre-wrap; line-height: 1.6; font-size: 20px; }
    #meta { color: #aeb4bf; font-size: 14px; }
    .dev-panel { border: 1px solid #a2543a; }
    .warn { color: #f0b27a; font-size: 14px; }
    details { margin-top: 10px; color: #aeb4bf; }
    pre { background: #14161a; padding: 12px; border-radius: 8px; overflow-x: auto; }
  </style>
</head>
<body>
  <main>
    <h1>Webcam VLM Test</h1>
    <div class="panel"><img class="camera" src="/video_feed" alt="webcam live stream"></div>

    <div class="panel">
      <h2>안전 안내 모드 (guide)</h2>
      <p class="warn">
        Metadata Schema와 Safety Validator를 통과한 문장만 표시합니다.
        이 도구에는 YOLO가 없으므로 detection이 없는 상태로 처리됩니다.
      </p>
      <form id="form">
        <input id="query" maxlength="500" value="주변 상황을 알려줘." required>
        <button id="submit" type="submit">캡처 후 안전 안내</button>
      </form>
      <img id="capture" alt="VLM에 전달된 캡처">
      <h3>사용자 안내 문장</h3>
      <div id="answer">질문을 입력하고 버튼을 누르세요.</div>
      <p id="meta"></p>
      <details>
        <summary>디버깅 정보 (raw_vlm_message 포함, 사용자 안내에 사용 금지)</summary>
        <pre id="debug">-</pre>
      </details>
    </div>

    <div class="panel dev-panel" id="dev-panel" hidden>
      <h2>개발 전용 자유 설명 모드 (dev_describe)</h2>
      <p class="warn">
        안전 검증을 거치지 않은 모델 원문입니다. 사용자 안내나 TTS에 사용하지 마세요.
      </p>
      <button id="dev-submit" class="dev" type="button">캡처 후 개발 모드 실행</button>
      <pre id="dev-answer">-</pre>
    </div>
  </main>
  <script>
    const form = document.getElementById('form');
    const button = document.getElementById('submit');
    const queryInput = document.getElementById('query');

    function showCapture(dataUrl) {
      const capture = document.getElementById('capture');
      capture.src = dataUrl;
      capture.style.display = 'block';
    }

    form.addEventListener('submit', async (event) => {
      event.preventDefault();
      button.disabled = true;
      document.getElementById('answer').textContent = '캡처 및 추론 중...';
      document.getElementById('meta').textContent = '';
      try {
        const response = await fetch('/guide', {
          method: 'POST', headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({query: queryInput.value})
        });
        const data = await response.json();
        if (!response.ok) throw new Error(data.error || '추론 실패');
        // 검증된 message만 사용자에게 노출한다.
        document.getElementById('answer').textContent = data.message;
        document.getElementById('meta').textContent =
          `${data.model_id} | ${data.status} | ${data.service_status} | ` +
          `${data.request_latency_ms} ms | source=${data.message_source}`;
        document.getElementById('debug').textContent = JSON.stringify(data, null, 2);
        showCapture(data.captured_image);
      } catch (error) {
        document.getElementById('answer').textContent = `오류: ${error.message}`;
      } finally { button.disabled = false; }
    });

    const devButton = document.getElementById('dev-submit');
    fetch('/modes').then((response) => response.json()).then((modes) => {
      if (modes.dev_describe) document.getElementById('dev-panel').hidden = false;
    });
    devButton.addEventListener('click', async () => {
      devButton.disabled = true;
      document.getElementById('dev-answer').textContent = '개발 모드 실행 중...';
      try {
        const response = await fetch('/dev_describe', {
          method: 'POST', headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({query: queryInput.value})
        });
        const data = await response.json();
        if (!response.ok) throw new Error(data.error || '추론 실패');
        document.getElementById('dev-answer').textContent =
          `${data.warning}\\n\\n${data.dev_raw_text}\\n\\n${data.request_latency_ms} ms`;
        showCapture(data.captured_image);
      } catch (error) {
        document.getElementById('dev-answer').textContent = `오류: ${error.message}`;
      } finally { devButton.disabled = false; }
    });
  </script>
</body>
</html>
"""


def create_app(engine: WebcamVLM) -> Flask:
    app = Flask(__name__)

    @app.get("/")
    def index() -> str:
        return render_template_string(HTML_PAGE)

    @app.get("/modes")
    def modes() -> Response:
        return jsonify(guide=True, dev_describe=engine.allow_dev_describe)

    @app.get("/video_feed")
    def video_feed() -> Response:
        def frames():
            while True:
                jpeg = engine.camera.jpeg()
                if jpeg is not None:
                    yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
                time.sleep(0.04)

        return Response(frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

    @app.post("/guide")
    def guide() -> tuple[Response, int] | Response:
        data = request.get_json(silent=True) or {}
        try:
            return jsonify(
                engine.guide(
                    data.get("query"),
                    yolo_payload=data.get("yolo_payload"),
                    flow_payload=data.get("flow_payload"),
                )
            )
        except VLMBusyError as error:
            return jsonify(error=str(error)), 429
        except ValueError as error:
            return jsonify(error=str(error)), 400
        except RuntimeError as error:
            return jsonify(error=str(error)), 503
        except Exception as error:  # noqa: BLE001 - 요청 단위로 격리한다
            LOGGER.exception("안전 안내 요청 실패")
            return jsonify(error=str(error)), 500

    @app.post("/dev_describe")
    def dev_describe() -> tuple[Response, int] | Response:
        data = request.get_json(silent=True) or {}
        try:
            return jsonify(engine.dev_describe(data.get("query")))
        except PermissionError as error:
            return jsonify(error=str(error)), 403
        except ValueError as error:
            return jsonify(error=str(error)), 400
        except RuntimeError as error:
            return jsonify(error=str(error)), 503
        except Exception as error:  # noqa: BLE001 - 요청 단위로 격리한다
            LOGGER.exception("개발 모드 요청 실패")
            return jsonify(error=str(error)), 500

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Jetson webcam VLM web test")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--camera-fps", type=int, default=30)
    parser.add_argument("--port", type=int, default=5001)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--vlm-timeout", type=float, default=10.0)
    parser.add_argument(
        "--allow-dev-describe",
        action="store_true",
        help="안전 검증을 거치지 않는 개발 전용 장면 설명 모드를 활성화한다.",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    args = parse_args()
    camera = LatestFrameCamera(args.camera, args.width, args.height, args.camera_fps)
    camera.start()
    # 종료 정리는 finally 한 곳에서만 수행한다(stop()은 재호출해도 안전하다).
    try:
        engine = WebcamVLM(
            camera,
            args.config,
            allow_dev_describe=args.allow_dev_describe,
            timeout_seconds=args.vlm_timeout,
        )
        if args.allow_dev_describe:
            LOGGER.warning("개발 전용 dev_describe 모드가 켜졌습니다. %s", DEV_MODE_WARNING)
        LOGGER.info("웹 페이지: http://0.0.0.0:%d", args.port)
        create_app(engine).run(host="0.0.0.0", port=args.port, threaded=True, use_reloader=False)
    finally:
        camera.stop()


if __name__ == "__main__":
    main()
