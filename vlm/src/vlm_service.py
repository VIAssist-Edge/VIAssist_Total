from __future__ import annotations

import time
import math
from pathlib import Path
from typing import Any

from src.exceptions import (
    VLMError,
    VLMImageError,
    VLMInferenceError,
    VLMMetadataError,
    VLMModelLoadError,
    VLMOutOfMemoryError,
    VLMServiceUnavailableError,
    VLMTimeoutError,
)
from src.metadata_schema import MetadataValidationError, validate_metadata
from src.prompt_builder import build_prompt
from src.result_parser import build_result
from src.safety_rules import select_detection
from src.vlm_engine import MockVLMEngine, SmolVLMEngine


class VLMService:
    """한 번 생성한 엔진을 재사용하며 안전 검증된 추론 결과를 만든다."""

    def __init__(self, engine: Any) -> None:
        self.engine = engine
        self.is_available = True
        self.last_error_code: str | None = None

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "VLMService":
        try:
            if config["use_mock_model"]:
                engine = MockVLMEngine()
            else:
                engine = SmolVLMEngine(
                    model_id=config["model_id"],
                    max_new_tokens=config["max_new_tokens"],
                    device=config["device"],
                    image_longest_edge=config["image_longest_edge"],
                    max_image_size=config["max_image_size"],
                )
        except VLMError:
            raise
        except Exception as error:
            raise VLMModelLoadError(
                "VLM 모델을 초기화하지 못했습니다.",
                original_exception=error,
            ) from error
        return cls(engine)

    def infer(
        self,
        image_path: Path | None,
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        validated_metadata = validate_metadata(metadata)
        selected_detection = select_detection(validated_metadata)
        prompt = build_prompt(validated_metadata)

        start_time = time.perf_counter()
        message = self.engine.generate(
            image_path=image_path,
            prompt=prompt,
            metadata=validated_metadata,
        )

        if getattr(self.engine, "device", None) == "cuda":
            torch = getattr(self.engine, "torch", None)
            if torch is not None:
                torch.cuda.synchronize()

        latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
        peak_gpu_memory_mb = 0.0
        if hasattr(self.engine, "get_peak_memory_mb"):
            peak_gpu_memory_mb = self.engine.get_peak_memory_mb()

        return build_result(
            message=message,
            metadata=validated_metadata,
            latency_ms=latency_ms,
            peak_gpu_memory_mb=peak_gpu_memory_mb,
            model_id=getattr(self.engine, "model_id", "mock-rule-engine"),
            selected_detection=selected_detection,
        )

    @staticmethod
    def _validate_timeout(timeout_seconds: float | None) -> None:
        if timeout_seconds is None:
            return
        if (
            isinstance(timeout_seconds, bool)
            or not isinstance(timeout_seconds, (int, float))
            or not math.isfinite(float(timeout_seconds))
            or timeout_seconds <= 0
        ):
            raise ValueError("timeout_seconds는 0보다 큰 숫자여야 합니다.")

    @staticmethod
    def _is_fatal_inference_error(error: VLMError) -> bool:
        if isinstance(error, VLMOutOfMemoryError):
            return True
        original = error.original_exception
        detail = str(original if original is not None else error).lower()
        fatal_markers = (
            "allocator",
            "internal assert",
            "device-side assert",
            "illegal memory access",
            "cuda error",
        )
        return isinstance(error, VLMInferenceError) and any(
            marker in detail for marker in fatal_markers
        )

    def _build_error_result(
        self,
        metadata: dict[str, Any],
        error: VLMError,
        *,
        latency_ms: float,
        fallback_reason: str,
    ) -> dict[str, Any]:
        metadata_is_valid = True
        try:
            validated_metadata = validate_metadata(metadata)
            selected_detection = select_detection(validated_metadata)
        except (MetadataValidationError, TypeError, AttributeError):
            metadata_is_valid = False
            validated_metadata = metadata if isinstance(metadata, dict) else {}
            selected_detection = None

        peak_gpu_memory_mb = 0.0
        if hasattr(self.engine, "get_peak_memory_mb"):
            try:
                peak_gpu_memory_mb = self.engine.get_peak_memory_mb()
            except Exception:
                peak_gpu_memory_mb = 0.0

        result = build_result(
            message="",
            metadata=validated_metadata,
            latency_ms=round(latency_ms, 2),
            peak_gpu_memory_mb=peak_gpu_memory_mb,
            model_id=getattr(self.engine, "model_id", "mock-rule-engine"),
            selected_detection=selected_detection,
        )
        if not metadata_is_valid:
            result["message"] = "입력 정보를 확인하기 어렵습니다. 다시 시도해 주세요."
        result.update(
            {
                "safety_validated": True,
                "used_fallback": True,
                "message_source": "fallback",
                "raw_vlm_message": "",
                "service_status": (
                    "unavailable" if not self.is_available else "degraded"
                ),
                "error": error.public_error(),
                "fallback_reason": fallback_reason,
            }
        )
        return result

    def infer_safe(
        self,
        image_path: Path | None,
        metadata: dict[str, Any],
        *,
        timeout_seconds: float | None = None,
    ) -> dict[str, Any]:
        self._validate_timeout(timeout_seconds)

        try:
            validate_metadata(metadata)
        except (MetadataValidationError, TypeError, AttributeError) as original:
            error = VLMMetadataError(
                "입력 정보를 확인하기 어렵습니다.",
                original_exception=original,
            )
            self.last_error_code = error.error_code
            return self._build_error_result(
                metadata,
                error,
                latency_ms=0.0,
                fallback_reason="metadata_invalid",
            )

        if not self.is_available:
            error = VLMServiceUnavailableError(
                "VLM 서비스를 현재 사용할 수 없습니다."
            )
            self.last_error_code = error.error_code
            return self._build_error_result(
                metadata,
                error,
                latency_ms=0.0,
                fallback_reason="service_unavailable",
            )

        start_time = time.perf_counter()
        try:
            result = self.infer(image_path, metadata)
        except VLMError as error:
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            if self._is_fatal_inference_error(error):
                self.is_available = False
            self.last_error_code = error.error_code
            return self._build_error_result(
                metadata,
                error,
                latency_ms=elapsed_ms,
                fallback_reason=(
                    "vlm_image_error"
                    if isinstance(error, VLMImageError)
                    else "vlm_inference_error"
                ),
            )
        except Exception as original:
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            error = VLMInferenceError(
                "VLM 추론에 실패했습니다.",
                original_exception=original,
            )
            if self._is_fatal_inference_error(error):
                self.is_available = False
            self.last_error_code = error.error_code
            return self._build_error_result(
                metadata,
                error,
                latency_ms=elapsed_ms,
                fallback_reason="vlm_inference_error",
            )

        elapsed_seconds = time.perf_counter() - start_time
        if timeout_seconds is not None and elapsed_seconds > timeout_seconds:
            error = VLMTimeoutError(
                "VLM 추론 제한 시간을 초과했습니다."
            )
            self.last_error_code = error.error_code
            return self._build_error_result(
                metadata,
                error,
                latency_ms=float(result["latency_ms"]),
                fallback_reason="vlm_timeout",
            )

        self.last_error_code = None
        result.update(
            {
                "service_status": "ok",
                "error": None,
                "fallback_reason": None,
            }
        )
        return result
