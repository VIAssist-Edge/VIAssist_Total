#!/usr/bin/env python3
"""MVP와 VLM 모듈을 잇는 얇은 경계.

- `VLMService`와 모델은 프로세스 시작 시 한 번만 만들고 재사용한다.
- 추론은 사용자의 명시적 요청이 있을 때만 실행한다.
- 모든 실제 추론은 `VIAssistVLMPipeline.process_perception()`을 거치며,
  내부적으로 `VLMService.infer_safe()`와 Safety Validator를 통과한다.
- raw `engine.generate()`는 이 경로에서 호출하지 않는다.
"""

from __future__ import annotations

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
    ) -> None:
        self.pipeline = pipeline
        self.timeout_seconds = timeout_seconds
        self.jpeg_quality = jpeg_quality
        self._lock = threading.Lock()
        self.request_count = 0

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
        service = getattr(self.pipeline, "service", None)
        return getattr(getattr(service, "engine", None), "model_id", "unknown")

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

                result = self.pipeline.process_perception(
                    image_path=image_path,
                    yolo_payload=yolo_payload,
                    flow_payload=flow_payload,
                    user_query=query,
                    timeout_seconds=self.timeout_seconds,
                )
            self.request_count += 1
        finally:
            self._lock.release()

        result["request_latency_ms"] = round(
            (time.perf_counter() - started) * 1000, 2
        )
        result["user_query"] = query
        return result
