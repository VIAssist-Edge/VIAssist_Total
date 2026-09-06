#!/usr/bin/env python3
"""MVP와 VLM 모듈을 잇는 얇은 경계.

- `VLMService`와 모델은 프로세스 시작 시 한 번만 만들고 재사용한다.
- 추론은 사용자의 명시적 요청이 있을 때만 실행한다.
- 모든 실제 추론은 `VIAssistVLMPipeline.process_perception()`을 거치며,
  내부적으로 `VLMService.infer_safe()`와 Safety Validator를 통과한다.
- raw `engine.generate()`는 이 경로에서 호출하지 않는다.
"""

from __future__ import annotations

import gc
import os
import sys
import threading
import time
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Optional

import cv2


VLM_ROOT = Path(__file__).resolve().parents[1] / "vlm"
if str(VLM_ROOT) not in sys.path:
    sys.path.insert(0, str(VLM_ROOT))

from src.config_loader import load_config  # noqa: E402
from src.integration_pipeline import VIAssistVLMPipeline  # noqa: E402
from src.metadata_adapter import DEFAULT_USER_QUERY  # noqa: E402
from src.vlm_service import VLMService  # noqa: E402


DEFAULT_CONFIG_PATH = VLM_ROOT / "config" / "jetson.json"
MAX_USER_QUERY_LENGTH = 500


class VLMBusyError(RuntimeError):
    """이미 다른 추론이 실행 중일 때 발생한다."""


class VLMBridge:
    """요청 기반 VLM 안내를 만드는 단일 인스턴스 경계."""

    def __init__(
        self,
        pipeline: VIAssistVLMPipeline,
        *,
        timeout_seconds: Optional[float] = 10.0,
        jpeg_quality: int = 90,
        config_path: Optional[Path] = None,
        strict_frame_sync: bool = False,
        idle_unload_seconds: Optional[float] = None,
    ) -> None:
        self.pipeline: Optional[VIAssistVLMPipeline] = pipeline
        self.timeout_seconds = timeout_seconds
        self.jpeg_quality = jpeg_quality
        self._lock = threading.Lock()
        self.request_count = 0

        # 유휴 언로드: 통합 메모리 젯슨에서 VLM 1.7 GB를 안 쓸 때 비워 둔다.
        self._config_path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
        self._strict_frame_sync = strict_frame_sync
        self._cached_model_id = self._read_model_id()
        self.last_used_at = time.monotonic()
        self.load_count = 1
        self.unload_count = 0
        self.last_load_ms = 0.0
        env_idle = os.environ.get("VLM_IDLE_UNLOAD_S")
        self.idle_unload_seconds = (
            float(env_idle) if env_idle is not None else
            (idle_unload_seconds if idle_unload_seconds is not None else 300.0)
        )
        self.reload_min_available_mb = float(os.environ.get("VLM_RELOAD_MIN_MB", "1900"))
        if self.idle_unload_seconds and self.idle_unload_seconds > 0:
            threading.Thread(target=self._idle_watcher, name="vlm-idle-unload", daemon=True).start()

    def _read_model_id(self) -> str:
        service = getattr(self.pipeline, "service", None)
        return getattr(getattr(service, "engine", None), "model_id", "unknown")

    @property
    def is_loaded(self) -> bool:
        return self.pipeline is not None

    @staticmethod
    def _available_mb() -> Optional[float]:
        try:
            with open("/proc/meminfo", encoding="utf-8") as handle:
                for line in handle:
                    if line.startswith("MemAvailable:"):
                        return int(line.split()[1]) / 1024.0
        except OSError:
            return None
        return None

    def unload(self) -> bool:
        """모델을 메모리에서 내린다. 추론 중이면 건너뛴다(False)."""

        if not self._lock.acquire(blocking=False):
            return False
        try:
            if self.pipeline is None:
                return False
            self.pipeline = None
            gc.collect()
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:  # noqa: BLE001
                pass
            self.unload_count += 1
            print(f"[VLM] 유휴 {self.idle_unload_seconds:.0f}s → 언로드 (가용 {self._available_mb() or 0:.0f} MB)", flush=True)
            return True
        finally:
            self._lock.release()

    def _ensure_loaded(self) -> None:
        """lock을 잡은 상태에서 호출. 내려가 있으면 다시 올린다."""

        if self.pipeline is not None:
            return
        available = self._available_mb()
        if available is not None and available < self.reload_min_available_mb:
            raise RuntimeError(
                f"메모리 여유 {available:.0f} MB < {self.reload_min_available_mb:.0f} MB — VLM을 다시 올릴 수 없습니다."
            )
        started = time.perf_counter()
        config = load_config(self._config_path)
        service = VLMService.from_config(config)
        self.pipeline = VIAssistVLMPipeline(service, strict_frame_sync=self._strict_frame_sync)
        self._cached_model_id = self._read_model_id()
        self.last_load_ms = (time.perf_counter() - started) * 1000
        self.load_count += 1
        print(f"[VLM] 재로드 {self.last_load_ms:.0f} ms ({self._cached_model_id})", flush=True)

    def _idle_watcher(self) -> None:
        while True:
            time.sleep(15.0)
            try:
                if (
                    self.pipeline is not None
                    and time.monotonic() - self.last_used_at >= self.idle_unload_seconds
                ):
                    self.unload()
            except Exception:  # noqa: BLE001 - 감시 스레드는 죽지 않는다
                pass

    @classmethod
    def from_config(
        cls,
        config_path: Path = DEFAULT_CONFIG_PATH,
        *,
        strict_frame_sync: bool = False,
        timeout_seconds: Optional[float] = 10.0,
        jpeg_quality: int = 90,
    ) -> "VLMBridge":
        """설정을 읽어 모델을 한 번 로드한다. 호출 비용이 크다."""

        config = load_config(Path(config_path))
        service = VLMService.from_config(config)
        pipeline = VIAssistVLMPipeline(
            service,
            strict_frame_sync=strict_frame_sync,
        )
        return cls(
            pipeline,
            timeout_seconds=timeout_seconds,
            jpeg_quality=jpeg_quality,
            config_path=Path(config_path),
            strict_frame_sync=strict_frame_sync,
        )

    @property
    def is_busy(self) -> bool:
        return self._lock.locked()

    @property
    def inference_lock(self) -> threading.Lock:
        """같은 GPU를 쓰는 다른 작업을 동일한 lock으로 직렬화하기 위해 노출한다."""

        return self._lock

    @property
    def model_id(self) -> str:
        if self.pipeline is None:
            return self._cached_model_id
        return self._read_model_id()

    @staticmethod
    def normalize_query(user_query: Optional[str]) -> str:
        if user_query is None:
            return DEFAULT_USER_QUERY
        if not isinstance(user_query, str):
            raise ValueError("질문은 문자열이어야 합니다.")
        query = user_query.strip()
        if not query:
            return DEFAULT_USER_QUERY
        if len(query) > MAX_USER_QUERY_LENGTH:
            raise ValueError(
                f"질문은 {MAX_USER_QUERY_LENGTH}자 이하로 입력해 주세요."
            )
        return query

    def describe(
        self,
        *,
        frame: Any,
        yolo_payload: dict[str, Any],
        flow_payload: Optional[dict[str, Any]] = None,
        user_query: Optional[str] = None,
        wait: bool = False,
    ) -> dict[str, Any]:
        """최신 프레임과 동기화된 payload로 한 번 추론한다.

        임시 이미지 파일은 요청마다 만들고 요청이 끝나면 삭제한다.
        """

        if frame is None:
            raise ValueError("추론할 프레임이 아직 준비되지 않았습니다.")
        query = self.normalize_query(user_query)

        if not self._lock.acquire(blocking=wait):
            raise VLMBusyError("이전 VLM 요청이 아직 처리 중입니다.")

        started = time.perf_counter()
        try:
            with TemporaryDirectory(prefix="viassist_vlm_") as temp_dir:
                image_path = Path(temp_dir) / "frame.jpg"
                encoded_ok, encoded = cv2.imencode(
                    ".jpg",
                    frame,
                    [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality],
                )
                if not encoded_ok:
                    raise RuntimeError("캡처 이미지를 인코딩하지 못했습니다.")
                image_path.write_bytes(encoded.tobytes())

                self._ensure_loaded()
                self.last_used_at = time.monotonic()
                result = self.pipeline.process_perception(
                    image_path=image_path,
                    yolo_payload=yolo_payload,
                    flow_payload=flow_payload,
                    user_query=query,
                    timeout_seconds=self.timeout_seconds,
                )
            self.request_count += 1
            self.last_used_at = time.monotonic()
        finally:
            self._lock.release()

        result["request_latency_ms"] = round(
            (time.perf_counter() - started) * 1000, 2
        )
        result["user_query"] = query
        return result

    def describe_scene(
        self,
        *,
        frame: Any,
        user_query: Optional[str] = None,
        wait: bool = False,
    ) -> dict[str, Any]:
        """YOLO/Optical Flow와 무관하게 현재 프레임 전체를 설명한다.

        `describe()`와 달리 elevator_button/escalator 탐지가 없어도 되며,
        SUPPORTED_TARGETS 밖의 객체(사람, 차량 등)도 안내에 포함될 수 있다.
        문장 형식·과잉 주장 차단은 `describe()`와 동일하게 유지된다.
        """

        if frame is None:
            raise ValueError("추론할 프레임이 아직 준비되지 않았습니다.")
        query = self.normalize_query(user_query)

        if not self._lock.acquire(blocking=wait):
            raise VLMBusyError("이전 VLM 요청이 아직 처리 중입니다.")

        started = time.perf_counter()
        try:
            with TemporaryDirectory(prefix="viassist_vlm_scene_") as temp_dir:
                image_path = Path(temp_dir) / "frame.jpg"
                encoded_ok, encoded = cv2.imencode(
                    ".jpg",
                    frame,
                    [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality],
                )
                if not encoded_ok:
                    raise RuntimeError("캡처 이미지를 인코딩하지 못했습니다.")
                image_path.write_bytes(encoded.tobytes())

                self._ensure_loaded()
                self.last_used_at = time.monotonic()
                result = self.pipeline.process_scene_description(
                    image_path=image_path,
                    user_query=query,
                    timeout_seconds=self.timeout_seconds,
                )
            self.request_count += 1
            self.last_used_at = time.monotonic()
        finally:
            self._lock.release()

        result["request_latency_ms"] = round(
            (time.perf_counter() - started) * 1000, 2
        )
        result["user_query"] = query
        return result
