#!/usr/bin/env python3
"""모듈별 점검 대시보드(/test).

STT / TTS / VLM / YOLO / Optical Flow 다섯 모듈을 **각각 따로** 눌러 보고
결과를 눈·귀로 바로 확인하기 위한 경계. `escalator_mvp.py`의 `create_app()`
에서 `register_test_routes(app, engine)` 한 번만 부른다.

원칙:
- 모델을 새로 로드하지 않는다. 이미 떠 있는 `engine`의 YOLO/VLM/STT/TTS
  인스턴스를 그대로 쓴다(젯슨 메모리가 빡빡하다).
- 무거운 작업(flow 시각화 스트림 등)은 브라우저가 해당 탭을 열고 있을 때만
  돈다. 나머지는 요청 단위다.
- 여기서 무엇이 실패해도 본 서비스(/, /status, /video_feed …)는 영향이 없다.
"""

from __future__ import annotations

import base64
import functools
import io
import json
import logging
import shutil
import subprocess
import sys
import threading
import time
import traceback
import uuid
import wave
from collections import deque
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Optional

import cv2
import numpy as np
from flask import Flask, Response, jsonify, request, send_from_directory
from werkzeug.exceptions import HTTPException


LOGGER = logging.getLogger("module_test")
HTML_PATH = Path(__file__).resolve().parent / "module_test.html"

# flow 시각화 스트림 상한 FPS. 본 처리 루프에 CPU를 양보한다.
FLOW_STREAM_FPS = 6.0
# engine이 만든 flow 필드가 이보다 오래됐으면(탐지 없음) 직접 계산한다.
FLOW_FIELD_MAX_AGE_S = 0.5


# ---------------------------------------------------------------------------
# 로그 링버퍼 — 대시보드에서 최근 오류를 바로 볼 수 있게 한다.
# ---------------------------------------------------------------------------
class RingBufferHandler(logging.Handler):
    def __init__(self, capacity: int = 400) -> None:
        super().__init__()
        self.records: deque[dict[str, Any]] = deque(maxlen=capacity)
        self.setFormatter(logging.Formatter("%(message)s"))

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self.records.append(
                {
                    "ts": record.created,
                    "level": record.levelname,
                    "logger": record.name,
                    "msg": self.format(record)[:600],
                }
            )
        except Exception:  # noqa: BLE001 - 로깅 실패는 조용히 무시
            pass


# ---------------------------------------------------------------------------
# 오류 기록 — 나중에 "언제·어디서·무슨 입력으로" 났는지 파일에서 확인하기 위한 것.
#   reports/module_test_errors.jsonl 에 한 줄씩 append 한다(재시작해도 남음).
#   출처 세 가지: (1) /test/* 라우트 예외  (2) 루트 로거 ERROR 이상(기존 코드의
#   LOGGER.exception 포함)  (3) 스레드 미처리 예외(처리 루프·카메라 스레드).
# ---------------------------------------------------------------------------
ERRORS_PATH = Path(__file__).resolve().parent / "reports" / "module_test_errors.jsonl"
ERRORS_KEEP_IN_MEMORY = 200
# 파일이 무한정 커지지 않도록 이 크기를 넘으면 앞부분을 잘라낸다.
ERRORS_MAX_BYTES = 5 * 2**20


class ErrorRecorder:
    def __init__(self, path: Path = ERRORS_PATH) -> None:
        self.path = path
        self.lock = threading.Lock()
        self.recent: deque[dict[str, Any]] = deque(maxlen=ERRORS_KEEP_IN_MEMORY)
        self.counts: dict[str, int] = {}
        self.total = 0
        self.started_at = time.time()
        self._load_recent()

    def _load_recent(self) -> None:
        try:
            lines = self.path.read_text(encoding="utf-8").splitlines()
        except Exception:  # noqa: BLE001
            return
        for line in lines[-ERRORS_KEEP_IN_MEMORY:]:
            try:
                self.recent.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    def record(
        self,
        *,
        module: str,
        where: str,
        error: Optional[BaseException] = None,
        message: Optional[str] = None,
        source: str = "route",
        extra: Optional[dict[str, Any]] = None,
        tb: Optional[str] = None,
    ) -> dict[str, Any]:
        entry = {
            "id": uuid.uuid4().hex[:10],
            "ts": time.time(),
            "iso": time.strftime("%Y-%m-%d %H:%M:%S"),
            "module": module,
            "where": where,
            "source": source,
            "type": type(error).__name__ if error is not None else None,
            "message": (message if message is not None else (str(error) if error else ""))[:1000],
            "traceback": (
                tb if tb is not None
                else ("".join(traceback.format_exception(type(error), error, error.__traceback__))[-6000:]
                      if error is not None else None)
            ),
            "extra": extra or {},
            "thread": threading.current_thread().name,
        }
        with self.lock:
            self.recent.append(entry)
            self.counts[module] = self.counts.get(module, 0) + 1
            self.total += 1
            try:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                if self.path.exists() and self.path.stat().st_size > ERRORS_MAX_BYTES:
                    data = self.path.read_bytes()[-ERRORS_MAX_BYTES // 2:]
                    self.path.write_bytes(data[data.find(b"\n") + 1:])
                with self.path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
            except Exception:  # noqa: BLE001 - 기록 실패가 또 오류를 만들면 안 된다
                pass
        return entry

    def summary(self) -> dict[str, Any]:
        with self.lock:
            last = self.recent[-1] if self.recent else None
            return {
                "total_since_start": self.total,
                "by_module": dict(self.counts),
                "stored": len(self.recent),
                "last": {k: last[k] for k in ("iso", "module", "where", "type", "message")} if last else None,
                "file": str(self.path),
            }

    def clear(self) -> None:
        with self.lock:
            self.recent.clear()
            self.counts.clear()
            self.total = 0
            try:
                self.path.write_text("", encoding="utf-8")
            except Exception:  # noqa: BLE001
                pass


class ErrorForwardHandler(logging.Handler):
    """루트 로거의 ERROR 이상을 ErrorRecorder로 넘긴다(기존 코드의 LOGGER.exception 포함)."""

    def __init__(self, recorder: ErrorRecorder) -> None:
        super().__init__(level=logging.ERROR)
        self.recorder = recorder

    def emit(self, record: logging.LogRecord) -> None:
        if record.name in ("werkzeug", "module_test.guard"):
            return  # 접근 로그·이미 기록한 가드 예외는 중복이다
        try:
            error = record.exc_info[1] if record.exc_info else None
            tb = "".join(traceback.format_exception(*record.exc_info)) if record.exc_info else None
            self.recorder.record(
                module=_module_from_logger(record.name, record.funcName),
                where=f"{record.name}:{record.funcName}",
                error=error,
                message=record.getMessage(),
                source="log",
                tb=tb,
            )
        except Exception:  # noqa: BLE001
            pass


def _module_from_logger(name: str, func: str = "") -> str:
    """로거 이름과 함수 이름으로 어느 모듈의 오류인지 대략 분류한다."""

    lowered = f"{name}:{func}".lower()
    for key, module in (
        ("stt", "stt"), ("voice", "stt"), ("audio", "stt"),
        ("tts", "tts"), ("melo", "tts"),
        ("vlm", "vlm"), ("flow", "flow"), ("yolo", "yolo"),
        ("pipeline", "pipeline"),
    ):
        if key in lowered:
            return module
    if "escalator" in lowered or "camera" in lowered or "processing" in lowered:
        return "yolo"
    return "system"


def install_thread_excepthook(recorder: ErrorRecorder) -> None:
    """스레드 안에서 잡히지 않은 예외(처리 루프·카메라 리더)를 기록한다."""

    previous = getattr(threading, "excepthook", None)

    def hook(args: Any) -> None:
        try:
            tb = "".join(traceback.format_exception(args.exc_type, args.exc_value, args.exc_traceback))
            recorder.record(
                module="system",
                where=f"thread:{getattr(args.thread, 'name', '?')}",
                error=args.exc_value,
                source="thread",
                tb=tb,
            )
        finally:
            if previous is not None:
                previous(args)

    if hasattr(threading, "excepthook"):
        threading.excepthook = hook


# ---------------------------------------------------------------------------
# 영상 파일 입력 — 실제 카메라 대신 녹화 영상을 처리 루프에 흘려 넣는다.
#   LatestFrameCamera와 같은 인터페이스(start/get_latest/stop/_thread)라
#   engine.camera 자리에 그대로 바꿔 끼울 수 있다. 루프는 매 프레임
#   self.camera.get_latest()를 부르므로 교체 즉시 반영된다.
# ---------------------------------------------------------------------------
VIDEOS_DIR = Path(__file__).resolve().parent / "reports" / "videos"
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}


class VideoFileSource:
    def __init__(self, path: Path, loop: bool = True, speed: float = 1.0) -> None:
        self.path = path
        self.loop = loop
        self.speed = max(0.1, min(float(speed), 4.0))
        self._frame: Optional[np.ndarray] = None
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._pause_event = threading.Event()  # set = 일시정지
        self._restart_flag = False
        self._seek_to: Optional[int] = None
        self._thread: Optional[threading.Thread] = None
        self._capture: Optional[cv2.VideoCapture] = None
        self.fps = 0.0
        self.frame_count = 0
        self.width = 0
        self.height = 0
        self.position = 0
        self.loops_done = 0
        self.finished = False
        self.error: Optional[str] = None
        self.started_at = 0.0

    def start(self) -> None:
        capture = cv2.VideoCapture(str(self.path))
        if not capture.isOpened():
            raise RuntimeError(f"영상을 열 수 없습니다: {self.path.name}")
        self._capture = capture
        self.fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0) or 24.0
        self.frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self.width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        self.height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        self.started_at = time.time()
        self._thread = threading.Thread(target=self._reader_loop, name="video-source", daemon=True)
        self._thread.start()

    def _reader_loop(self) -> None:
        interval = 1.0 / (self.fps * self.speed)
        next_tick = time.perf_counter()
        while not self._stop_event.is_set():
            if self._pause_event.is_set() and not self._restart_flag and self._seek_to is None:
                time.sleep(0.05)
                next_tick = time.perf_counter()
                continue
            if self._capture is None:
                break
            if self._restart_flag:
                self._capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                self.position = 0
                self.finished = False
                self._restart_flag = False
            if self._seek_to is not None:
                target = max(0, min(self._seek_to, max(self.frame_count - 1, 0)))
                self._capture.set(cv2.CAP_PROP_POS_FRAMES, target)
                self.position = target
                self.finished = False
                self._seek_to = None
            ok, frame = self._capture.read()
            if not ok or frame is None:
                if self.loop:
                    self._capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    self.position = 0
                    self.loops_done += 1
                    continue
                self.finished = True
                self._pause_event.set()
                continue
            self.position += 1
            with self._lock:
                self._frame = frame
            # 원본 FPS(×배속)로 페이싱한다. 실카메라와 같은 시간 흐름을 흉내낸다.
            next_tick += interval
            delay = next_tick - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
            else:
                next_tick = time.perf_counter()

    def get_latest(self) -> Optional[np.ndarray]:
        with self._lock:
            return None if self._frame is None else self._frame.copy()

    def pause(self) -> None:
        self._pause_event.set()

    def play(self) -> None:
        self._pause_event.clear()

    def restart(self) -> None:
        self._restart_flag = True
        self._pause_event.clear()

    def seek(self, frame_index: int) -> None:
        self._seek_to = int(frame_index)

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        if self._capture is not None:
            self._capture.release()
            self._capture = None

    def info(self) -> dict[str, Any]:
        return {
            "name": self.path.name,
            "fps": round(self.fps, 2),
            "speed": self.speed,
            "frame_count": self.frame_count,
            "duration_s": round(self.frame_count / self.fps, 2) if self.fps else None,
            "position": self.position,
            "position_s": round(self.position / self.fps, 2) if self.fps else None,
            "width": self.width,
            "height": self.height,
            "loop": self.loop,
            "loops_done": self.loops_done,
            "paused": self._pause_event.is_set(),
            "finished": self.finished,
            "thread_alive": bool(self._thread and self._thread.is_alive()),
        }


class DirectionTimeline:
    """영상 모드 동안 처리 루프 status를 주기적으로 샘플링해 시간축으로 남긴다."""

    def __init__(self, engine: Any, source: VideoFileSource, hz: float = 10.0) -> None:
        self.engine = engine
        self.source = source
        self.interval = 1.0 / hz
        self.samples: deque[dict[str, Any]] = deque(maxlen=6000)
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._loop, name="direction-timeline", daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        last_frame = -1
        while not self._stop_event.is_set():
            time.sleep(self.interval)
            with self.engine.status_lock:
                status = self.engine.status
            frame_index = getattr(self.engine, "frame_index", 0)
            if frame_index == last_frame:
                continue  # 일시정지 중이면 같은 프레임이 반복된다
            last_frame = frame_index
            self.samples.append({
                "t": round(time.time() - self.source.started_at, 2),
                "pos": self.source.position,
                "loop": self.source.loops_done,
                "stable": status.stable_direction,
                "raw": status.raw_direction,
                "cls": status.class_name,
                "conf": round(float(status.confidence), 3),
                "mag": round(float(status.motion_magnitude), 2),
                "dy": round(float(status.motion_dy), 2),
                "dconf": round(float(status.direction_confidence), 2),
                "src": status.motion_source,
            })

    def stop(self) -> None:
        self._stop_event.set()


# ---------------------------------------------------------------------------
# 시스템 지표 — tegrastats는 1초 블로킹이라 sysfs를 직접 읽는다.
# ---------------------------------------------------------------------------
def _read_first(paths: list[str]) -> Optional[str]:
    for path in paths:
        try:
            return Path(path).read_text().strip()
        except Exception:  # noqa: BLE001
            continue
    return None


def system_metrics() -> dict[str, Any]:
    mem: dict[str, int] = {}
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            key, _, rest = line.partition(":")
            if key in ("MemTotal", "MemAvailable", "SwapTotal", "SwapFree"):
                mem[key] = int(rest.strip().split()[0]) // 1024
    except Exception:  # noqa: BLE001
        pass

    gpu_load = _read_first(
        ["/sys/devices/platform/gpu.0/load", "/sys/devices/gpu.0/load"]
    )

    # Jetson은 CPU/GPU가 DRAM 하나를 나눠 쓴다(별도 VRAM 없음). 이 프로세스가
    # CUDA 텐서로 잡고 있는 양만 따로 잰다 — MeloTTS 워커(별도 프로세스)는 제외.
    cuda: dict[str, Any] = {}
    try:
        import torch

        if torch.cuda.is_available():
            cuda = {
                "allocated_mb": round(torch.cuda.memory_allocated() / 2**20, 1),
                "reserved_mb": round(torch.cuda.memory_reserved() / 2**20, 1),
                "max_allocated_mb": round(torch.cuda.max_memory_allocated() / 2**20, 1),
            }
    except Exception:  # noqa: BLE001
        pass
    temps: dict[str, float] = {}
    try:
        for zone in sorted(Path("/sys/class/thermal").glob("thermal_zone*")):
            name = (zone / "type").read_text().strip()
            if any(k in name for k in ("cpu", "gpu", "soc0")):
                temps[name.replace("-thermal", "")] = (
                    int((zone / "temp").read_text().strip()) / 1000.0
                )
    except Exception:  # noqa: BLE001
        pass

    return {
        "memory_kind": "unified DRAM (CPU+GPU 공유, 별도 VRAM 없음)",
        "cuda_this_process": cuda,
        "mem_total_mb": mem.get("MemTotal"),
        "mem_available_mb": mem.get("MemAvailable"),
        "swap_used_mb": (
            mem["SwapTotal"] - mem["SwapFree"]
            if "SwapTotal" in mem and "SwapFree" in mem
            else None
        ),
        "swap_total_mb": mem.get("SwapTotal"),
        "gpu_load_pct": (int(gpu_load) / 10.0) if gpu_load and gpu_load.isdigit() else None,
        "temps_c": temps,
    }


# ---------------------------------------------------------------------------
# Optical flow 시각화
# ---------------------------------------------------------------------------
def flow_to_hsv_bgr(flow: np.ndarray, max_mag: float = 8.0) -> np.ndarray:
    """flow 벡터장을 색상환(방향=색, 세기=밝기) 이미지로 바꾼다."""

    fx, fy = flow[..., 0], flow[..., 1]
    mag, ang = cv2.cartToPolar(fx, fy, angleInDegrees=True)
    hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)
    hsv[..., 0] = (ang / 2).astype(np.uint8)  # OpenCV hue: 0~180
    hsv[..., 1] = 255
    hsv[..., 2] = np.clip(mag / max_mag * 255.0, 0, 255).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def draw_flow_arrows(
    canvas: np.ndarray,
    flow: np.ndarray,
    *,
    step: int = 16,
    scale: float = 3.0,
    min_mag: float = 0.3,
    color=(0, 255, 255),
) -> int:
    """격자 위치마다 flow 벡터를 화살표로 그린다. 그린 개수를 돌려준다."""

    h, w = flow.shape[:2]
    drawn = 0
    for y in range(step // 2, h, step):
        for x in range(step // 2, w, step):
            vx, vy = flow[y, x]
            if float(np.hypot(vx, vy)) < min_mag:
                continue
            end = (int(x + vx * scale), int(y + vy * scale))
            cv2.arrowedLine(canvas, (x, y), end, color, 1, tipLength=0.35)
            drawn += 1
    return drawn


class FlowVisualizer:
    """engine의 flow 필드를 가로채 보관하고, 없으면 직접 계산한다."""

    def __init__(self, engine: Any) -> None:
        self.engine = engine
        self.lock = threading.Lock()
        self.last_field: Optional[dict[str, Any]] = None
        self.last_field_at = 0.0
        self.own_prev_gray: Optional[np.ndarray] = None
        self.own_prev_at = 0.0
        # engine 쪽 flow 계산(ego 보정 + Farneback) 소요 시간. 본 코드는 이걸
        # 재지 않으므로 여기서 감싸서 잰다.
        self.last_flow_ms = 0.0
        self.flow_ms_ema = 0.0
        self._install_hook()

    def _install_hook(self) -> None:
        original = self.engine._compute_flow_field

        def hooked(*args: Any, **kwargs: Any) -> dict[str, Any]:
            started = time.perf_counter()
            field = original(*args, **kwargs)
            elapsed = (time.perf_counter() - started) * 1000
            with self.lock:
                self.last_field = field
                self.last_field_at = time.monotonic()
                self.last_flow_ms = elapsed
                self.flow_ms_ema = elapsed if not self.flow_ms_ema else 0.9 * self.flow_ms_ema + 0.1 * elapsed
            return field

        self.engine._compute_flow_field = hooked

    def _own_flow(self, frame: np.ndarray) -> Optional[dict[str, Any]]:
        """탐지가 없어 engine flow가 없을 때, 연속 프레임으로 직접 계산한다."""

        gray, _sx, _sy = self.engine._prepare_flow_frame(frame)
        now = time.monotonic()
        prev = self.own_prev_gray
        prev_at = self.own_prev_at
        self.own_prev_gray = gray
        self.own_prev_at = now
        # 너무 오래된 프레임과 비교하면 움직임이 과장된다.
        if prev is None or prev.shape != gray.shape or now - prev_at > 1.0:
            return None
        flow = cv2.calcOpticalFlowFarneback(
            prev, gray, None, 0.5, 3, 21, 3, 5, 1.2, 0
        )
        return {
            "flow": flow,
            "motion_source": "test_fullframe(no-detection)",
            "homography_inliers": 0,
            "flow_width": gray.shape[1],
            "flow_height": gray.shape[0],
        }

    def render(self) -> tuple[Optional[bytes], dict[str, Any]]:
        """(JPEG bytes, meta). 왼쪽=원본+화살표, 오른쪽=색상환 flow."""

        frame = self.engine.camera.get_latest()
        if frame is None:
            return None, {"error": "카메라 프레임 없음"}

        with self.lock:
            field = self.last_field
            age = time.monotonic() - self.last_field_at
        source = "engine"
        if field is None or age > FLOW_FIELD_MAX_AGE_S:
            field = self._own_flow(frame)
            source = "own"
            if field is None:
                # 아직 두 번째 프레임 전이다. 원본만 보여 준다.
                ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                return (buf.tobytes() if ok else None), {
                    "source": "warming", "arrows": 0,
                }

        flow = field["flow"]
        fh, fw = flow.shape[:2]
        H, W = frame.shape[:2]

        left = frame.copy()
        small_canvas = cv2.resize(left, (fw, fh))
        arrows = draw_flow_arrows(small_canvas, flow)
        left = cv2.resize(small_canvas, (W, H), interpolation=cv2.INTER_LINEAR)

        # 탐지된 객체 ROI와 판정 결과를 겹친다(engine 내부 상태, 읽기 전용).
        try:
            for det in list(self.engine.last_detections):
                x1, y1, x2, y2 = det["bbox"]
                motion = det.get("motion") or {}
                cv2.rectangle(left, (x1, y1), (x2, y2), (0, 220, 0), 2)
                label = (
                    f'{det["class_name"]} #{det.get("track_id")} '
                    f'{motion.get("stable_direction", motion.get("direction", "-"))} '
                    f'dy={motion.get("dy", 0):+.2f} c={motion.get("confidence", 0):.2f}'
                )
                cv2.putText(left, label, (x1, max(18, y1 - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 0), 1, cv2.LINE_AA)
        except Exception:  # noqa: BLE001 - 처리 스레드와 경합하면 그냥 건너뛴다
            pass

        right = cv2.resize(flow_to_hsv_bgr(flow), (W, H), interpolation=cv2.INTER_NEAREST)
        cv2.putText(right, f"src={field.get('motion_source')} inliers={field.get('homography_inliers')}",
                    (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(left, f"arrows={arrows} field={source} {fw}x{fh}",
                    (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)

        composite = np.hstack([left, right])
        ok, buf = cv2.imencode(".jpg", composite, [cv2.IMWRITE_JPEG_QUALITY, 70])
        return (buf.tobytes() if ok else None), {
            "source": source,
            "motion_source": field.get("motion_source"),
            "homography_inliers": field.get("homography_inliers"),
            "arrows": arrows,
            "flow_size": [fw, fh],
        }


# ---------------------------------------------------------------------------
# 오디오 유틸
# ---------------------------------------------------------------------------
def pcm_to_wav_bytes(pcm: bytes, rate: int = 16000) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(pcm)
    return buffer.getvalue()


def pcm_envelope(pcm: bytes, points: int = 400) -> list[int]:
    """파형 그리기용. 구간별 절대값 최대치를 points개로 줄인다."""

    if not pcm:
        return []
    samples = np.frombuffer(pcm, dtype=np.int16)
    if samples.size == 0:
        return []
    chunk = max(1, samples.size // points)
    trimmed = samples[: (samples.size // chunk) * chunk]
    if trimmed.size == 0:
        return [int(np.abs(samples).max())]
    return np.abs(trimmed.reshape(-1, chunk)).max(axis=1).astype(int).tolist()


def record_fixed(stt: Any, seconds: float) -> bytes:
    """VAD 없이 고정 길이로 젯슨 마이크를 녹음해 16kHz PCM을 돌려준다.

    마이크 자체가 소리를 받는지(게인, 장치 선택)를 VAD/임계값과 분리해서
    확인하기 위한 경로다.
    """

    import audioop

    import sounddevice as sd

    native_rate = stt.native_rate
    frames: list[bytes] = []

    def callback(indata, _frames, _time, _status) -> None:
        frames.append(bytes(indata))

    with sd.RawInputStream(
        samplerate=native_rate,
        blocksize=stt.native_frame_samples,
        device=stt.input_device,
        dtype="int16",
        channels=1,
        callback=callback,
    ):
        time.sleep(seconds)

    raw = b"".join(frames)
    if native_rate != 16000:
        raw, _ = audioop.ratecv(raw, 2, 1, native_rate, 16000, None)
    return raw


def convert_to_wav16k(src: Path, dst: Path) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg가 없어 업로드 오디오를 변환할 수 없습니다.")
    completed = subprocess.run(
        [ffmpeg, "-y", "-loglevel", "error", "-i", str(src),
         "-ac", "1", "-ar", "16000", "-f", "wav", str(dst)],
        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=60,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "ffmpeg 변환 실패: " + completed.stderr.decode("utf-8", "ignore")[:300]
        )


def decode_upload_image() -> Optional[np.ndarray]:
    """multipart(file) 또는 JSON(image_b64) 둘 다 받는다."""

    file = request.files.get("file")
    data: Optional[bytes] = None
    if file is not None:
        data = file.read()
    else:
        body = request.get_json(silent=True) or {}
        b64 = body.get("image_b64")
        if b64:
            data = base64.b64decode(b64.split(",")[-1])
    if not data:
        return None
    array = np.frombuffer(data, dtype=np.uint8)
    return cv2.imdecode(array, cv2.IMREAD_COLOR)


def jpeg_b64(image: np.ndarray, quality: int = 80) -> str:
    ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode() if ok else ""


# ---------------------------------------------------------------------------
# 라우트 등록
# ---------------------------------------------------------------------------
def register_test_routes(app: Flask, engine: Any) -> None:
    log_buffer = RingBufferHandler()
    log_buffer.setLevel(logging.INFO)
    logging.getLogger().addHandler(log_buffer)

    errors = ErrorRecorder()
    logging.getLogger().addHandler(ErrorForwardHandler(errors))
    install_thread_excepthook(errors)
    guard_logger = logging.getLogger("module_test.guard")

    def guarded(module: str):
        """라우트 예외를 잡아 기록하고, 원인을 담은 JSON으로 응답한다.

        기록 항목: 모듈, 경로, 예외 타입/메시지, traceback, 요청 파라미터 요약.
        응답의 error_id로 파일(reports/module_test_errors.jsonl)에서 찾을 수 있다.
        """

        def decorator(view):
            @functools.wraps(view)
            def wrapper(*args: Any, **kwargs: Any):
                try:
                    return view(*args, **kwargs)
                except HTTPException:
                    raise  # 404 같은 정상적인 HTTP 응답은 오류가 아니다
                except Exception as error:  # noqa: BLE001 - 요청 단위 격리
                    try:
                        body = request.get_json(silent=True)
                        params = {
                            "path": request.path,
                            "method": request.method,
                            "args": dict(request.args),
                            "json": {k: (v if isinstance(v, (int, float, bool)) or (isinstance(v, str) and len(v) < 200) else f"<{type(v).__name__}>")
                                     for k, v in (body or {}).items()} if isinstance(body, dict) else None,
                            "form": {k: v[:200] for k, v in request.form.items()},
                            "files": list(request.files.keys()),
                        }
                    except Exception:  # noqa: BLE001
                        params = {"path": request.path}
                    entry = errors.record(module=module, where=request.path, error=error, extra=params)
                    guard_logger.error("[%s] %s: %s (error_id=%s)", module, request.path, error, entry["id"])
                    return jsonify(
                        error=f"{type(error).__name__}: {error}",
                        error_id=entry["id"],
                        module=module,
                        where=request.path,
                        hint="오류 탭 또는 GET /test/errors 에서 traceback 확인",
                    ), 500
            return wrapper
        return decorator

    flow_vis = FlowVisualizer(engine)
    started_at = time.time()

    # 업로드 이미지 YOLO 테스트용. 본 루프의 tracker 상태를 건드리지 않도록
    # 첫 요청 때 별도 인스턴스를 만든다(.pt 기준 수십 MB).
    upload_model_lock = threading.Lock()
    upload_model: dict[str, Any] = {}

    def _voice():
        return getattr(engine, "voice", None)

    def _stt():
        voice = _voice()
        return getattr(voice, "stt", None) if voice else None

    def _tts():
        voice = _voice()
        return getattr(voice, "tts", None) if voice else None

    def vlm_config_values() -> dict[str, Any]:
        """실행 중인 VLM 설정 파일에서 비교에 의미 있는 값만 뽑는다."""

        try:
            path = Path(str(getattr(engine.args, "vlm_config", "")))
            if not path.is_absolute():
                path = Path(__file__).resolve().parent / path
            data = json.loads(path.read_text(encoding="utf-8"))
            return {k: data.get(k) for k in (
                "model_id", "image_longest_edge", "max_image_size",
                "max_new_tokens", "quantization", "device",
            )}
        except Exception as error:  # noqa: BLE001
            return {"error": str(error)}

    def vlm_config_label() -> str:
        values = vlm_config_values()
        return (
            f"{str(values.get('model_id', '?')).split('/')[-1]} "
            f"edge={values.get('image_longest_edge')} tile={values.get('max_image_size')} "
            f"tok={values.get('max_new_tokens')}"
        )

    def mem_snapshot() -> dict[str, Any]:
        s = system_metrics()
        return {
            "mem_available_mb": s["mem_available_mb"],
            "swap_used_mb": s["swap_used_mb"],
            "cuda_allocated_mb": (s.get("cuda_this_process") or {}).get("allocated_mb"),
        }

    # 벤치 결과는 재시작해도 남도록 파일에 쌓는다(설정 전후 비교용).
    bench_path = Path(__file__).resolve().parent / "reports" / "module_test_bench.json"
    bench_lock = threading.Lock()

    def load_bench() -> list[dict[str, Any]]:
        try:
            return json.loads(bench_path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            return []

    def save_bench(runs: list[dict[str, Any]]) -> None:
        bench_path.parent.mkdir(parents=True, exist_ok=True)
        bench_path.write_text(json.dumps(runs, ensure_ascii=False, indent=1), encoding="utf-8")

    # 이 값 밑으로 내려가면 VLM 호출을 더 하지 않는다. 젯슨 전체가 멈춘 적이 있다.
    MEMORY_GUARD_MB = 350

    # ---- 페이지 / 공통 -------------------------------------------------
    @app.get("/test")
    @guarded("system")
    def test_page() -> Response:
        try:
            html = HTML_PATH.read_text(encoding="utf-8")
        except FileNotFoundError:
            return Response("module_test.html이 없습니다.", status=500)
        return Response(html, mimetype="text/html")

    @app.get("/test/health")
    @guarded("system")
    def test_health() -> Response:
        status = engine.get_status()
        stt = _stt()
        tts = _tts()
        bridge = getattr(engine, "vlm_bridge", None)

        input_device_name = None
        if stt is not None:
            try:
                import sounddevice as sd

                info = sd.query_devices(stt.input_device) if stt.input_device is not None else sd.query_devices(kind="input")
                input_device_name = str(info.get("name"))
            except Exception:  # noqa: BLE001
                input_device_name = None

        melo = getattr(tts, "melo", None) if tts else None
        processing_thread = getattr(engine, "processing_thread", None)
        camera_thread = getattr(getattr(engine, "camera", None), "_thread", None)
        status_age_s = round(time.time() - status.get("timestamp", 0), 2) if status.get("timestamp") else None
        return jsonify(
            uptime_s=round(time.time() - started_at, 1),
            errors=errors.summary(),
            liveness={
                "processing_thread_alive": bool(processing_thread and processing_thread.is_alive()),
                "camera_thread_alive": bool(camera_thread and camera_thread.is_alive()),
                # status가 2초 넘게 안 갱신되면 루프가 멈춘 것이다.
                "status_age_s": status_age_s,
                "loop_stalled": bool(status_age_s is not None and status_age_s > 2.0),
                "loop_error_count": status.get("error_count", 0),
                "loop_last_error": status.get("last_error", ""),
            },
            system=system_metrics(),
            yolo={
                "model": str(engine.args.model),
                "classes": len(engine.class_names),
                "class_names": [str(v) for v in engine.class_names.values()],
                "device": str(engine.args.device),
                "imgsz": engine.args.imgsz,
                "conf": engine.args.conf,
                "yolo_every": engine.args.yolo_every,
                "tracker": engine.args.tracker,
                "camera_ok": status["camera_ok"],
                "processing_fps": round(status["processing_fps"], 1),
                "yolo_inference_ms": round(status["yolo_inference_ms"], 1),
            },
            flow={
                "flow_width": engine.args.flow_width,
                "min_motion": engine.args.min_motion,
                "max_motion": engine.args.max_motion,
                "direction_threshold": engine.args.direction_threshold,
                "direction_history": engine.args.direction_history,
                "direction_majority": engine.args.direction_majority,
                "motion_source": status["motion_source"],
                "stable_direction": status["stable_direction"],
                "flow_ms_last": round(flow_vis.last_flow_ms, 2),
                "flow_ms_ema": round(flow_vis.flow_ms_ema, 2),
            },
            vlm={
                "enabled": bridge is not None,
                "model_id": getattr(bridge, "model_id", None),
                "busy": bool(getattr(bridge, "is_busy", False)),
                # 유휴 언로드 상태. False면 다음 요청 때 ~28 s 재로드가 붙는다.
                "loaded": bool(getattr(bridge, "is_loaded", bridge is not None)),
                "idle_unload_s": getattr(bridge, "idle_unload_seconds", None),
                "idle_s": round(time.monotonic() - getattr(bridge, "last_used_at", time.monotonic()), 1) if bridge else None,
                "load_count": getattr(bridge, "load_count", None),
                "last_load_ms": round(getattr(bridge, "last_load_ms", 0.0) or 0.0),
                "request_count": getattr(bridge, "request_count", 0),
                "timeout_s": getattr(bridge, "timeout_seconds", None),
                "config": str(getattr(engine.args, "vlm_config", "")),
                "config_values": vlm_config_values(),
            },
            stt={
                "enabled": stt is not None,
                "model_size": getattr(stt, "model_size", None),
                "language": getattr(stt, "language", None),
                "input_device": getattr(stt, "input_device", None),
                "input_device_name": input_device_name,
                "native_rate": getattr(stt, "native_rate", None),
                "noise_multiplier": getattr(stt, "noise_multiplier", None),
                "last_noise_floor": getattr(stt, "last_noise_floor", None),
                "last_speech_threshold": getattr(stt, "last_speech_threshold", None),
                "busy": bool(getattr(stt, "is_busy", False)),
            },
            tts={
                "enabled": tts is not None,
                "engine": getattr(tts, "engine", None),
                "engine_error": getattr(tts, "engine_error", None),
                "melo_alive": bool(melo.is_alive) if melo is not None else None,
                "melo_speaker": getattr(melo, "speaker", None),
                "melo_load_ms": getattr(melo, "load_ms", None),
                "player": (tts.player[0] if tts and tts.player else None),
                "alsa_device": getattr(tts, "alsa_device", None),
                "speed": getattr(tts, "speed", None),
                "cache_dir": str(getattr(tts, "cache_dir", "")),
            },
        )

    @app.get("/test/errors")
    @guarded("system")
    def test_errors() -> Response:
        n = int(request.args.get("n", 100))
        module = request.args.get("module")
        with errors.lock:
            rows = list(errors.recent)
        if module:
            rows = [r for r in rows if r.get("module") == module]
        return jsonify(summary=errors.summary(), errors=rows[-n:])

    @app.delete("/test/errors")
    @guarded("system")
    def test_errors_clear() -> Response:
        errors.clear()
        return jsonify(ok=True)

    @app.post("/test/errors/client")
    @guarded("system")
    def test_errors_client() -> Response:
        """브라우저(대시보드 JS)에서 난 오류도 같은 파일에 남긴다."""

        data = request.get_json(silent=True) or {}
        entry = errors.record(
            module="dashboard",
            where=str(data.get("where", "browser"))[:200],
            message=str(data.get("message", ""))[:1000],
            source="browser",
            tb=str(data.get("stack", ""))[:6000] or None,
            extra={"ua": request.user_agent.string[:200], "tab": data.get("tab")},
        )
        return jsonify(ok=True, error_id=entry["id"])

    @app.get("/test/log")
    @guarded("system")
    def test_log() -> Response:
        n = int(request.args.get("n", 120))
        level = request.args.get("level", "INFO").upper()
        threshold = logging.getLevelName(level)
        threshold = threshold if isinstance(threshold, int) else logging.INFO
        rows = [
            r for r in list(log_buffer.records)
            if logging.getLevelName(r["level"]) >= threshold
            # werkzeug 접근 로그는 잡음이라 뺀다.
            and r["logger"] != "werkzeug"
        ]
        return jsonify(records=rows[-n:])

    # ---- YOLO ------------------------------------------------------------
    @app.get("/test/yolo/frame.jpg")
    @guarded("yolo")
    def yolo_raw_frame() -> Response:
        frame = engine.camera.get_latest()
        if frame is None:
            return jsonify(error="카메라 프레임 없음"), 503
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return Response(buf.tobytes(), mimetype="image/jpeg")

    @app.post("/test/yolo/image")
    @guarded("yolo")
    def yolo_image() -> Response:
        image = decode_upload_image()
        if image is None:
            return jsonify(error="이미지를 읽지 못했습니다."), 400
        conf = float(request.form.get("conf", request.args.get("conf", engine.args.conf)))

        with upload_model_lock:
            if "model" not in upload_model:
                from ultralytics import YOLO

                path = str(engine.args.model)
                LOGGER.info("업로드 테스트용 YOLO 인스턴스 로드: %s", path)
                upload_model["model"] = (
                    YOLO(path, task="detect") if path.endswith(".engine") else YOLO(path)
                )
            model = upload_model["model"]
            started = time.perf_counter()
            results = model.predict(
                source=image, conf=conf, imgsz=engine.args.imgsz,
                device=engine.args.device, verbose=False,
            )
            infer_ms = (time.perf_counter() - started) * 1000

        detections = []
        annotated = image.copy()
        if results and results[0].boxes is not None:
            for box in results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].detach().cpu().numpy().tolist())
                cid = int(box.cls[0].item())
                c = float(box.conf[0].item())
                name = str(engine.class_names.get(cid, cid))
                detections.append({"class_id": cid, "class_name": name,
                                   "confidence": round(c, 3), "bbox": [x1, y1, x2, y2]})
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 220, 0), 2)
                cv2.putText(annotated, f"{name} {c:.2f}", (x1, max(18, y1 - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 0), 2, cv2.LINE_AA)
        detections.sort(key=lambda d: -d["confidence"])
        return jsonify(
            inference_ms=round(infer_ms, 1),
            conf=conf,
            image_size=[image.shape[1], image.shape[0]],
            detections=detections,
            annotated=jpeg_b64(annotated),
        )

    # ---- Optical Flow ------------------------------------------------------
    @app.get("/test/flow/frame.jpg")
    @guarded("flow")
    def flow_frame() -> Response:
        jpeg, meta = flow_vis.render()
        if jpeg is None:
            return jsonify(meta), 503
        return Response(jpeg, mimetype="image/jpeg", headers={"X-Flow-Meta": json.dumps(meta)})

    @app.get("/test/flow/stream")
    @guarded("flow")
    def flow_stream() -> Response:
        interval = 1.0 / FLOW_STREAM_FPS

        def generate():
            while not engine.stop_event.is_set():
                tick = time.perf_counter()
                try:
                    jpeg, _meta = flow_vis.render()
                except Exception:  # noqa: BLE001
                    LOGGER.exception("flow 시각화 실패")
                    jpeg = None
                if jpeg is not None:
                    yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n")
                time.sleep(max(0.0, interval - (time.perf_counter() - tick)))

        return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")

    @app.get("/test/flow/stats")
    @guarded("flow")
    def flow_stats() -> Response:
        status = engine.get_status()
        tracks = []
        try:
            for det in list(engine.last_detections):
                motion = det.get("motion") or {}
                tracks.append({
                    "track_id": det.get("track_id"),
                    "class_name": det.get("class_name"),
                    "confidence": round(float(det.get("confidence", 0)), 3),
                    "bbox": list(det.get("bbox", ())),
                    "direction": motion.get("direction"),
                    "stable_direction": motion.get("stable_direction"),
                    "stable_ratio": motion.get("stable_ratio"),
                    "history_size": motion.get("history_size"),
                    "dx": motion.get("dx"),
                    "dy": motion.get("dy"),
                    "magnitude": motion.get("magnitude"),
                    "motion_confidence": motion.get("confidence"),
                    "valid_ratio": motion.get("valid_ratio"),
                    "motion_source": motion.get("motion_source"),
                    "homography_inliers": motion.get("homography_inliers"),
                })
        except Exception:  # noqa: BLE001
            pass
        return jsonify(
            status=status,
            primary_track_id=getattr(engine, "primary_track_id", None),
            stable_frames=getattr(engine, "stable_frames", 0),
            tracks=tracks,
        )

    # ---- VLM ---------------------------------------------------------------
    @app.get("/test/vlm/frame.jpg")
    @guarded("vlm")
    def vlm_frame() -> Response:
        """VLM에 실제로 들어가는 프레임(YOLO가 돈 스냅샷)을 그대로 보여 준다."""

        snapshot = engine.get_snapshot()
        if snapshot is None:
            return jsonify(error="아직 분석된 프레임이 없습니다."), 503
        ok, buf = cv2.imencode(".jpg", snapshot["frame"], [cv2.IMWRITE_JPEG_QUALITY, 85])
        return Response(
            buf.tobytes(), mimetype="image/jpeg",
            headers={"X-Frame-Id": str(snapshot["frame_id"]),
                     "X-Detections": str(len(snapshot["yolo_payload"].get("detections", [])))},
        )

    @app.post("/test/vlm/image")
    @guarded("vlm")
    def vlm_image() -> Response:
        """업로드한 이미지로 VLM 장면 설명을 돌린다(카메라와 무관)."""

        bridge = getattr(engine, "vlm_bridge", None)
        if bridge is None:
            return jsonify(error="VLM이 비활성화되어 있습니다."), 503
        image = decode_upload_image()
        if image is None:
            return jsonify(error="이미지를 읽지 못했습니다."), 400
        query = request.form.get("query") or (request.get_json(silent=True) or {}).get("query")
        try:
            query = bridge.normalize_query(query)
        except ValueError as error:
            return jsonify(error=str(error)), 400

        if not bridge.inference_lock.acquire(blocking=False):
            return jsonify(error="이전 VLM 요청이 아직 처리 중입니다."), 429
        started = time.perf_counter()
        try:
            with TemporaryDirectory(prefix="viassist_vlm_test_") as temp_dir:
                path = Path(temp_dir) / "upload.jpg"
                cv2.imwrite(str(path), image, [cv2.IMWRITE_JPEG_QUALITY, bridge.jpeg_quality])
                result = bridge.pipeline.process_scene_description(
                    image_path=path, user_query=query,
                    timeout_seconds=bridge.timeout_seconds,
                )
        except Exception as error:  # noqa: BLE001
            LOGGER.exception("VLM 업로드 테스트 실패")
            return jsonify(error=str(error)), 500
        finally:
            bridge.inference_lock.release()
        result["request_latency_ms"] = round((time.perf_counter() - started) * 1000, 1)
        result["user_query"] = query
        return jsonify(result)

    # ---- STT ---------------------------------------------------------------
    @app.get("/test/audio/devices")
    @guarded("stt")
    def audio_devices() -> Response:
        try:
            import sounddevice as sd

            devices = []
            for index, dev in enumerate(sd.query_devices()):
                devices.append({
                    "index": index, "name": dev.get("name"),
                    "in": dev.get("max_input_channels"), "out": dev.get("max_output_channels"),
                    "default_samplerate": dev.get("default_samplerate"),
                })
            return jsonify(devices=devices, default=list(sd.default.device))
        except Exception as error:  # noqa: BLE001
            return jsonify(error=str(error)), 500

    @app.get("/test/stt/level")
    @guarded("stt")
    def stt_level() -> Response:
        """0.4초 동안 마이크 RMS를 재서 돌려준다(입력 레벨 미터용)."""

        stt = _stt()
        if stt is None:
            return jsonify(error="STT 비활성"), 503
        if not stt._lock.acquire(blocking=False):
            return jsonify(error="녹음 중"), 429
        try:
            import audioop

            pcm = record_fixed(stt, 0.4)
            rms = audioop.rms(pcm, 2) if pcm else 0
            peak = int(np.abs(np.frombuffer(pcm, dtype=np.int16)).max()) if pcm else 0
        except Exception as error:  # noqa: BLE001
            return jsonify(error=str(error)), 500
        finally:
            stt._lock.release()
        return jsonify(rms=rms, peak=peak, bytes=len(pcm))

    @app.post("/test/stt/listen")
    @guarded("stt")
    def stt_listen() -> Response:
        """젯슨 마이크로 녹음 → 인식. 녹음된 소리도 함께 돌려줘 들어볼 수 있다.

        mode=vad  : 기존과 같은 VAD+RMS 임계값 녹음(말이 끝나면 멈춤)
        mode=fixed: seconds 동안 무조건 녹음(마이크 자체 점검용)
        """

        stt = _stt()
        if stt is None:
            return jsonify(error="STT 비활성"), 503
        data = request.get_json(silent=True) or {}
        mode = str(data.get("mode", "vad"))
        if not stt._lock.acquire(blocking=False):
            return jsonify(error="이전 음성 인식이 아직 처리 중입니다."), 429
        try:
            record_started = time.perf_counter()
            if mode == "fixed":
                pcm = record_fixed(stt, float(data.get("seconds", 4.0)))
                heard = bool(pcm)
            else:
                multiplier = data.get("noise_multiplier")
                pcm = stt.record(
                    max_seconds=float(data.get("max_seconds", 8.0)),
                    start_timeout_s=float(data.get("start_timeout_s", 4.0)),
                    silence_ms=int(data.get("silence_ms", 800)),
                    noise_multiplier=float(multiplier) if multiplier else None,
                )
                heard = bool(pcm)
            record_ms = (time.perf_counter() - record_started) * 1000

            transcribe_started = time.perf_counter()
            text = stt.transcribe_pcm(pcm) if pcm else ""
            transcribe_ms = (time.perf_counter() - transcribe_started) * 1000
        except Exception as error:  # noqa: BLE001
            LOGGER.exception("STT 테스트 실패")
            return jsonify(error=str(error)), 500
        finally:
            stt._lock.release()

        wav = pcm_to_wav_bytes(pcm) if pcm else b""
        return jsonify(
            mode=mode,
            text=text,
            heard_speech=heard,
            audio_seconds=round(len(pcm) / 32000.0, 2),
            record_ms=round(record_ms, 1),
            transcribe_ms=round(transcribe_ms, 1),
            model_size=stt.model_size,
            noise_floor=stt.last_noise_floor,
            speech_threshold=stt.last_speech_threshold,
            envelope=pcm_envelope(pcm),
            audio_b64=("data:audio/wav;base64," + base64.b64encode(wav).decode()) if wav else None,
        )

    @app.post("/test/stt/upload")
    @guarded("stt")
    def stt_upload() -> Response:
        """브라우저(PC 마이크)나 파일에서 온 오디오를 인식한다. 젯슨 마이크와 무관."""

        stt = _stt()
        if stt is None:
            return jsonify(error="STT 비활성"), 503
        file = request.files.get("file")
        if file is None:
            return jsonify(error="오디오 파일이 없습니다."), 400
        if not stt._lock.acquire(blocking=False):
            return jsonify(error="이전 음성 인식이 아직 처리 중입니다."), 429
        try:
            with TemporaryDirectory(prefix="viassist_stt_up_") as temp_dir:
                suffix = Path(file.filename or "audio.webm").suffix or ".webm"
                src = Path(temp_dir) / f"input{suffix}"
                dst = Path(temp_dir) / "input16k.wav"
                file.save(str(src))
                convert_started = time.perf_counter()
                convert_to_wav16k(src, dst)
                convert_ms = (time.perf_counter() - convert_started) * 1000
                with wave.open(str(dst), "rb") as handle:
                    pcm = handle.readframes(handle.getnframes())
                started = time.perf_counter()
                text = stt.transcribe_file(dst)
                transcribe_ms = (time.perf_counter() - started) * 1000
        except Exception as error:  # noqa: BLE001
            LOGGER.exception("STT 업로드 테스트 실패")
            return jsonify(error=str(error)), 500
        finally:
            stt._lock.release()
        return jsonify(
            text=text,
            audio_seconds=round(len(pcm) / 32000.0, 2),
            convert_ms=round(convert_ms, 1),
            transcribe_ms=round(transcribe_ms, 1),
            model_size=stt.model_size,
            envelope=pcm_envelope(pcm),
        )

    # ---- TTS ---------------------------------------------------------------
    @app.post("/test/tts/synth")
    @guarded("tts")
    def tts_synth() -> Response:
        """문장을 합성한다. play=true면 젯슨 스피커로도 재생. 브라우저용 URL을 돌려준다."""

        tts = _tts()
        if tts is None:
            return jsonify(error="TTS 비활성"), 503
        data = request.get_json(silent=True) or {}
        text = str(data.get("text", "")).strip()
        if not text:
            return jsonify(error="문장이 비어 있습니다."), 400
        play = bool(data.get("play", False))
        try:
            result = tts.speak(text, play_audio=play)
        except Exception as error:  # noqa: BLE001
            LOGGER.exception("TTS 테스트 실패")
            return jsonify(error=str(error)), 500
        path = Path(result.get("audio_path", ""))
        result["audio_url"] = f"/test/tts/audio/{path.name}" if path.name else None
        try:
            if path.suffix == ".wav" and path.is_file():
                with wave.open(str(path), "rb") as handle:
                    result["audio_seconds"] = round(handle.getnframes() / handle.getframerate(), 2)
        except Exception:  # noqa: BLE001
            pass
        return jsonify(result)

    @app.get("/test/tts/audio/<name>")
    @guarded("tts")
    def tts_audio(name: str) -> Response:
        tts = _tts()
        if tts is None:
            return jsonify(error="TTS 비활성"), 503
        # 캐시 디렉터리 안의 파일만 내보낸다.
        return send_from_directory(str(tts.cache_dir), name, max_age=0)

    # ---- 입력 소스: 실카메라 ↔ 영상 파일 ------------------------------------
    live_camera = engine.camera
    source_lock = threading.Lock()
    source_state: dict[str, Any] = {"video": None, "timeline": None}

    def _reset_engine_motion_state() -> None:
        # 소스가 바뀌면 이전 프레임/검출/방향 이력이 무의미하다.
        engine.previous_gray_small = None
        engine.last_detections = []
        engine.primary_track_id = None
        stabilizer = getattr(engine, "direction_stabilizer", None)
        if stabilizer is not None and hasattr(stabilizer, "reset"):
            stabilizer.reset()

    def _switch_to_camera() -> None:
        video = source_state["video"]
        timeline = source_state["timeline"]
        # 기동 시 카메라가 없었으면(USB 미연결) 여기서 다시 열어 본다. 리더 루프는 살아 있어
        # _capture가 채워지는 순간부터 프레임을 읽기 시작한다. 재시작 없이 카메라 재연결.
        if getattr(live_camera, "_capture", None) is None and hasattr(live_camera, "_open_camera"):
            try:
                live_camera._open_camera()
                LOGGER.info("실카메라 재연결 성공 (index=%s)", getattr(live_camera, "camera_index", "?"))
            except Exception as error:  # noqa: BLE001
                LOGGER.warning("실카메라 재연결 실패: %s", error)
        engine.camera = live_camera
        _reset_engine_motion_state()
        if timeline is not None:
            timeline.stop()
        if video is not None:
            video.stop()
        source_state["video"] = None
        source_state["timeline"] = None

    def _source_info() -> dict[str, Any]:
        video = source_state["video"]
        return {
            "mode": "video" if video is not None else "camera",
            "video": video.info() if video is not None else None,
            "camera_index": getattr(live_camera, "camera_index", None),
            "camera_connected": getattr(live_camera, "_capture", None) is not None,
        }

    def _list_videos() -> list[dict[str, Any]]:
        VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
        rows = []
        for p in sorted(VIDEOS_DIR.iterdir()):
            if p.suffix.lower() in VIDEO_EXTS:
                rows.append({"name": p.name, "size_mb": round(p.stat().st_size / 2**20, 2), "mtime": p.stat().st_mtime})
        return rows

    @app.get("/test/source")
    @guarded("yolo")
    def source_get() -> Response:
        return jsonify(**_source_info(), videos=_list_videos())

    @app.post("/test/source/upload")
    @guarded("yolo")
    def source_upload() -> Response:
        file = request.files.get("file")
        if file is None or not file.filename:
            return jsonify(error="영상 파일이 없습니다."), 400
        suffix = Path(file.filename).suffix.lower()
        if suffix not in VIDEO_EXTS:
            return jsonify(error=f"지원하지 않는 확장자: {suffix}"), 400
        VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
        safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in Path(file.filename).name)
        target = VIDEOS_DIR / safe
        file.save(str(target))
        return jsonify(ok=True, name=safe, size_mb=round(target.stat().st_size / 2**20, 2), videos=_list_videos())

    @app.post("/test/source/video")
    @guarded("yolo")
    def source_video() -> Response:
        """저장된 영상 파일을 처리 루프 입력으로 바꿔 끼운다."""

        data = request.get_json(silent=True) or {}
        name = Path(str(data.get("name", ""))).name
        path = VIDEOS_DIR / name
        if not name or not path.exists():
            return jsonify(error=f"영상이 없습니다: {name}"), 404
        loop = bool(data.get("loop", True))
        speed = float(data.get("speed", 1.0))
        with source_lock:
            if source_state["video"] is not None:
                _switch_to_camera()
            video = VideoFileSource(path, loop=loop, speed=speed)
            video.start()
            _reset_engine_motion_state()
            engine.camera = video
            source_state["video"] = video
            source_state["timeline"] = DirectionTimeline(engine, video)
        LOGGER.info("입력 소스 → 영상 %s (%dx%d %.1ffps %d프레임, loop=%s)", name, video.width, video.height, video.fps, video.frame_count, loop)
        return jsonify(**_source_info())

    @app.post("/test/source/camera")
    @guarded("yolo")
    def source_camera() -> Response:
        with source_lock:
            _switch_to_camera()
        LOGGER.info("입력 소스 → 실카메라")
        return jsonify(**_source_info())

    @app.post("/test/source/control")
    @guarded("yolo")
    def source_control() -> Response:
        data = request.get_json(silent=True) or {}
        video = source_state["video"]
        if video is None:
            return jsonify(error="영상 모드가 아닙니다."), 409
        action = str(data.get("action", ""))
        if action == "pause":
            video.pause()
        elif action == "play":
            video.play()
        elif action == "restart":
            _reset_engine_motion_state()
            video.restart()
            timeline = source_state["timeline"]
            if timeline is not None:
                timeline.samples.clear()
                video.started_at = time.time()
        elif action == "seek":
            _reset_engine_motion_state()
            video.seek(int(data.get("frame", 0)))
        elif action == "loop":
            video.loop = bool(data.get("loop", True))
        else:
            return jsonify(error=f"알 수 없는 action: {action}"), 400
        return jsonify(**_source_info())

    @app.get("/test/source/timeline")
    @guarded("yolo")
    def source_timeline() -> Response:
        timeline = source_state["timeline"]
        video = source_state["video"]
        samples = list(timeline.samples) if timeline is not None else []
        # 연속 구간으로 묶어 "몇 초~몇 초는 UP" 식으로 읽기 쉽게 한다.
        segments: list[dict[str, Any]] = []
        for s in samples:
            if segments and segments[-1]["stable"] == s["stable"] and segments[-1]["loop"] == s["loop"]:
                segments[-1]["end_pos"] = s["pos"]
                segments[-1]["end_t"] = s["t"]
                segments[-1]["n"] += 1
            else:
                segments.append({"stable": s["stable"], "loop": s["loop"], "start_pos": s["pos"], "end_pos": s["pos"], "start_t": s["t"], "end_t": s["t"], "n": 1})
        counts: dict[str, int] = {}
        for s in samples:
            counts[s["stable"]] = counts.get(s["stable"], 0) + 1
        return jsonify(
            samples=samples[-600:],
            segments=segments,
            counts=counts,
            total=len(samples),
            video=video.info() if video is not None else None,
        )

    # ---- VLM 벤치마크 (설정 전후 비교) ------------------------------------
    @app.post("/test/vlm/bench")
    @guarded("vlm")
    def vlm_bench() -> Response:
        """같은 질문으로 N번 연속 호출해 지연·메모리 변화를 기록한다.

        각 호출 전에 DRAM 여유를 확인해 MEMORY_GUARD_MB 밑이면 중단한다.
        결과는 reports/module_test_bench.json에 설정 라벨과 함께 남는다.
        """

        bridge = getattr(engine, "vlm_bridge", None)
        if bridge is None:
            return jsonify(error="VLM이 비활성화되어 있습니다."), 503
        data = request.get_json(silent=True) or {}
        calls = max(1, min(int(data.get("calls", 3)), 10))
        mode = str(data.get("mode", "scene"))
        query = data.get("query") or None
        note = str(data.get("note", ""))

        if not bench_lock.acquire(blocking=False):
            return jsonify(error="이미 벤치가 진행 중입니다."), 429
        try:
            run: dict[str, Any] = {
                "ts": time.time(),
                "label": vlm_config_label(),
                "config": vlm_config_values(),
                "mode": mode,
                "note": note,
                "mem_before": mem_snapshot(),
                "calls": [],
                "aborted": None,
            }
            for index in range(calls):
                before = mem_snapshot()
                if (before["mem_available_mb"] or 0) < MEMORY_GUARD_MB:
                    run["aborted"] = (
                        f"{index + 1}번째 호출 전 DRAM 여유 {before['mem_available_mb']}MB "
                        f"< 가드 {MEMORY_GUARD_MB}MB — 중단"
                    )
                    break
                started = time.perf_counter()
                try:
                    if mode == "guidance":
                        result = engine.request_vlm_guidance(query)
                    else:
                        result = engine.request_vlm_scene_description(query)
                    error = (result.get("error") or {}).get("code") if isinstance(result.get("error"), dict) else result.get("error")
                    entry = {
                        "latency_ms": round((time.perf_counter() - started) * 1000, 1),
                        "status": result.get("status"),
                        "error": error,
                        "message": result.get("message"),
                        "service_status": result.get("service_status"),
                    }
                except Exception as error:  # noqa: BLE001
                    entry = {
                        "latency_ms": round((time.perf_counter() - started) * 1000, 1),
                        "status": "exception", "error": str(error)[:200], "message": None,
                    }
                after = mem_snapshot()
                entry["mem_before"] = before
                entry["mem_after"] = after
                entry["mem_delta_mb"] = (
                    (after["mem_available_mb"] or 0) - (before["mem_available_mb"] or 0)
                )
                run["calls"].append(entry)
            run["mem_after"] = mem_snapshot()
            lat = [c["latency_ms"] for c in run["calls"]]
            run["summary"] = {
                "calls": len(lat),
                "avg_ms": round(sum(lat) / len(lat), 1) if lat else None,
                "min_ms": min(lat) if lat else None,
                "max_ms": max(lat) if lat else None,
                "timeouts": sum(1 for c in run["calls"] if c.get("error") == "INFERENCE_TIMEOUT"),
                "mem_delta_total_mb": (
                    (run["mem_after"]["mem_available_mb"] or 0)
                    - (run["mem_before"]["mem_available_mb"] or 0)
                ),
                "mem_delta_per_call_mb": (
                    round(sum(c["mem_delta_mb"] for c in run["calls"]) / len(lat), 1) if lat else None
                ),
            }
            runs = load_bench()
            runs.append(run)
            save_bench(runs)
        finally:
            bench_lock.release()
        return jsonify(run)

    @app.get("/test/vlm/bench/history")
    @guarded("vlm")
    def vlm_bench_history() -> Response:
        return jsonify(runs=load_bench(), current_label=vlm_config_label())

    @app.delete("/test/vlm/bench/history")
    @guarded("vlm")
    def vlm_bench_clear() -> Response:
        with bench_lock:
            save_bench([])
        return jsonify(ok=True)

    # ---- 파이프라인 단계별 시간 -----------------------------------------------
    @app.post("/test/pipeline/run")
    @guarded("pipeline")
    def pipeline_run() -> Response:
        """실제 안내 경로를 한 번 돌리며 단계마다 시간을 잰다.

        (마이크 STT) → 스냅샷(YOLO+flow는 이미 돈 값) → VLM 또는 규칙 → TTS 합성 → 재생
        use_mic=false면 text 질문으로 STT를 건너뛴다.
        """

        data = request.get_json(silent=True) or {}
        use_mic = bool(data.get("use_mic", False))
        mode = str(data.get("mode", "scene"))  # scene | guidance | rule
        play = bool(data.get("play", False))
        query = data.get("query") or None
        stt = _stt()
        tts = _tts()
        bridge = getattr(engine, "vlm_bridge", None)

        stages: list[dict[str, Any]] = []
        total_started = time.perf_counter()

        def stage(name: str, ms: float, **extra: Any) -> None:
            stages.append({"stage": name, "ms": round(ms, 1), **extra})

        # 1) STT
        heard_text = None
        if use_mic:
            if stt is None:
                return jsonify(error="STT 비활성"), 503
            if not stt._lock.acquire(blocking=False):
                return jsonify(error="STT가 사용 중입니다."), 429
            try:
                t = time.perf_counter()
                pcm = stt.record(
                    max_seconds=float(data.get("max_seconds", 8.0)),
                    start_timeout_s=float(data.get("start_timeout_s", 4.0)),
                )
                stage("STT 녹음(VAD)", (time.perf_counter() - t) * 1000,
                      audio_seconds=round(len(pcm) / 32000.0, 2), heard_speech=bool(pcm))
                t = time.perf_counter()
                heard_text = stt.transcribe_pcm(pcm) if pcm else ""
                stage("STT 인식(whisper)", (time.perf_counter() - t) * 1000, text=heard_text)
                LOGGER.info("파이프라인 STT 인식: %r (%.2fs 오디오)", heard_text, len(pcm) / 32000.0)
            finally:
                stt._lock.release()
            query = heard_text or query

        # 2) 스냅샷 (YOLO/flow는 처리 루프가 이미 계산해 둔 값 — 그 프레임의 실제 소요 시간을 같이 적는다)
        t = time.perf_counter()
        snapshot = engine.get_snapshot()
        snapshot_ms = (time.perf_counter() - t) * 1000
        if snapshot is None:
            return jsonify(error="아직 분석된 프레임이 없습니다."), 503
        stage("스냅샷 취득", snapshot_ms, frame_id=snapshot["frame_id"],
              frame_age_ms=round((time.time() - snapshot["captured_at"]) * 1000, 1))
        stage("YOLO 추론 (해당 프레임, 루프에서 측정)", snapshot["yolo_payload"].get("latency_ms", 0.0),
              detections=len(snapshot["yolo_payload"].get("detections", [])), async_in_loop=True)
        stage("Optical Flow (ego보정+Farneback, 루프에서 측정)", flow_vis.last_flow_ms,
              direction=snapshot["flow_payload"].get("direction"), async_in_loop=True)

        # 3) 안내 문장 생성
        message = ""
        vlm_meta: dict[str, Any] = {}
        route_info: dict[str, Any] = {}
        if mode == "auto":
            # 라우터가 규칙/VLM(상태)/VLM(장면) 중 하나를 고른다. 어느 경로였는지 단계에 남긴다.
            import guidance_router

            t = time.perf_counter()
            decision = guidance_router.decide(query, snapshot, vlm_available=bridge is not None)
            stage("라우터 판단", (time.perf_counter() - t) * 1000, route=decision["route"],
                  reason=decision["reason"], detections=decision["detections"],
                  target=(decision.get("target") or {}).get("korean"))
            route_info = decision
            if decision["route"] not in ("rule", "scripted"):
                mem_now = mem_snapshot()
                if (mem_now["mem_available_mb"] or 0) < MEMORY_GUARD_MB:
                    return jsonify(error=f"DRAM 여유 {mem_now['mem_available_mb']}MB < 가드 {MEMORY_GUARD_MB}MB — VLM 호출 중단", stages=stages), 507
            t = time.perf_counter()
            try:
                routed = guidance_router.run(engine, query, mode=decision["route"])
            except Exception as error:  # noqa: BLE001
                stage("실행(" + decision["route"] + ")", (time.perf_counter() - t) * 1000, error=str(error)[:200])
                return jsonify(error=str(error), stages=stages), 500
            label = {"rule": "규칙 안내 생성", "vlm_state": "VLM 상태 질문", "vlm_scene": "VLM 장면 설명",
                     "scripted": "고정 응답(데모, 지연 포함)"}[decision["route"]]
            vlm_res = routed.get("vlm") or {}
            err = vlm_res.get("error")
            stage(label, routed["exec_ms"], source=routed["source"],
                  **({"status": vlm_res.get("status"), "error": (err.get("code") if isinstance(err, dict) else err)} if vlm_res else {}),
                  **({"question": routed.get("question")} if routed.get("question") and decision["route"] == "vlm_state" else {}))
            message = routed.get("message", "")
            if vlm_res:
                vlm_meta = {k: vlm_res.get(k) for k in ("status", "service_status", "fallback_reason", "model_id")}
        elif mode == "rule":
            t = time.perf_counter()
            result = engine.build_rule_guidance()
            stage("규칙 안내 생성", (time.perf_counter() - t) * 1000, source="rule")
            message = result.get("message", "")
        else:
            if bridge is None:
                return jsonify(error="VLM 비활성"), 503
            mem_now = mem_snapshot()
            if (mem_now["mem_available_mb"] or 0) < MEMORY_GUARD_MB:
                return jsonify(error=f"DRAM 여유 {mem_now['mem_available_mb']}MB < 가드 {MEMORY_GUARD_MB}MB — VLM 호출 중단"), 507
            t = time.perf_counter()
            try:
                if mode == "guidance":
                    result = engine.request_vlm_guidance(query)
                else:
                    result = engine.request_vlm_scene_description(query)
            except Exception as error:  # noqa: BLE001
                stage("VLM 추론", (time.perf_counter() - t) * 1000, error=str(error)[:200])
                return jsonify(error=str(error), stages=stages), 500
            err = result.get("error")
            stage("VLM 추론 (" + mode + ")", (time.perf_counter() - t) * 1000,
                  status=result.get("status"), error=(err.get("code") if isinstance(err, dict) else err),
                  model_id=result.get("model_id"))
            message = result.get("message", "")
            vlm_meta = {k: result.get(k) for k in ("status", "service_status", "fallback_reason", "model_id")}

        # 4) TTS
        spoken: Optional[dict[str, Any]] = None
        if tts is not None and message:
            t = time.perf_counter()
            try:
                spoken = tts.speak(message, play_audio=play)
                stage("TTS 합성" + (" (캐시)" if spoken.get("cached") else ""), spoken.get("synthesize_ms", 0.0),
                      engine=spoken.get("engine"), cached=spoken.get("cached"))
                if play:
                    stage("TTS 재생(젯슨 스피커)", spoken.get("play_ms", 0.0),
                          spoken=spoken.get("spoken"), play_error=spoken.get("play_error"))
                path = Path(spoken.get("audio_path", ""))
                spoken["audio_url"] = f"/test/tts/audio/{path.name}" if path.name else None
            except Exception as error:  # noqa: BLE001
                stage("TTS 합성", (time.perf_counter() - t) * 1000, error=str(error)[:200])

        total_ms = (time.perf_counter() - total_started) * 1000
        sync_ms = sum(s["ms"] for s in stages if not s.get("async_in_loop"))
        return jsonify(
            stages=stages,
            total_ms=round(total_ms, 1),
            sync_ms=round(sync_ms, 1),
            heard_text=heard_text,
            query=query,
            message=message,
            vlm=vlm_meta,
            route=route_info,
            spoken=spoken,
            mem=mem_snapshot(),
        )

    LOGGER.info("모듈 점검 대시보드: /test")
