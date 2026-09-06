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
from src.prompt_builder import build_prompt, build_scene_prompt
from src.result_parser import build_result
from src.safety_rules import select_detection
from src.scene_description import (
    DEFAULT_SCENE_QUERY,
    SCENE_FALLBACK_MESSAGE,
    build_scene_result,
)
from src.config_loader import ENGINE_GEMINI, ENGINE_LOCAL
from src.vlm_engine import MockVLMEngine, SmolVLMEngine


class VLMService:
    """한 번 생성한 엔진을 재사용하며 안전 검증된 추론 결과를 만든다."""

    def __init__(self, engine: Any) -> None:
        self.engine = engine
        self.is_available = True
        self.last_error_code: str | None = None

    @staticmethod
    def _build_engine(config: dict[str, Any]) -> Any:
        """설정의 engine 값에 따라 엔진 하나를 만든다.

        engine 키가 없는 기존 설정은 로컬 SmolVLM 엔진으로 동작한다.
        """

        if config["use_mock_model"]:
            return MockVLMEngine()

        engine_name = config.get("engine", ENGINE_LOCAL)
        if engine_name == ENGINE_GEMINI:
            # API 엔진은 stdlib만 쓰므로 여기서 import해도 비용이 없다.
            from src.gemini_engine import GeminiVLMEngine

            return GeminiVLMEngine(
                model_id=config["model_id"],
                max_new_tokens=config["max_new_tokens"],
                fallback_model_ids=config.get("fallback_model_ids"),
                request_timeout_seconds=config.get(
                    "request_timeout_seconds", 20.0
                ),
                max_image_size=config["max_image_size"],
                jpeg_quality=config.get("jpeg_quality", 85),
                stream=config.get("stream", True),
            )

        if engine_name == "llamacpp":
            # llama-server(OpenAI 호환) 클라이언트. 서버는 엔진이 자식 프로세스로 띄운다.
            from src.llamacpp_engine import LlamaCppVLMEngine

            block = dict(config.get("llamacpp") or {})
            return LlamaCppVLMEngine(
                model_id=config["model_id"],
                max_new_tokens=config["max_new_tokens"],
                image_longest_edge=config["image_longest_edge"],
                **block,
            )

        return SmolVLMEngine(
            model_id=config["model_id"],
            max_new_tokens=config["max_new_tokens"],
            device=config["device"],
            image_longest_edge=config["image_longest_edge"],
            max_image_size=config["max_image_size"],
        )

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "VLMService":
        try:
            engine = cls._build_engine(config)
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

    def _build_scene_error_result(
        self,
        error: VLMError,
        *,
        latency_ms: float = 0.0,
        fallback_reason: str,
    ) -> dict[str, Any]:
        peak_gpu_memory_mb = 0.0
        if hasattr(self.engine, "get_peak_memory_mb"):
            try:
                peak_gpu_memory_mb = self.engine.get_peak_memory_mb()
            except Exception:
                peak_gpu_memory_mb = 0.0

        result = build_scene_result(
            message="",
            latency_ms=round(latency_ms, 2),
            peak_gpu_memory_mb=peak_gpu_memory_mb,
            model_id=getattr(self.engine, "model_id", "mock-rule-engine"),
        )
        result.update(
            {
                "message": SCENE_FALLBACK_MESSAGE,
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

    def describe_scene(
        self,
        image_path: Path | None,
        user_query: str = DEFAULT_SCENE_QUERY,
        *,
        timeout_seconds: float | None = None,
    ) -> dict[str, Any]:
        """SUPPORTED_TARGETS와 무관하게 화면 전체를 설명하는 별도 경로.

        `infer_safe()`와 달리 YOLO/Optical Flow metadata와 대조하지 않으므로
        elevator_button·escalator 밖의 객체(사람, 차량 등)도 언급할 수 있다.
        문장 형식과 과잉 주장(거리·안전 판단·강한 행동 지시 등) 차단은 그대로
        유지한다. 검증 실패나 VLM 오류는 언제나 결정적 fallback으로 대체한다.
        """

        self._validate_timeout(timeout_seconds)

        if not self.is_available:
            error = VLMServiceUnavailableError(
                "VLM 서비스를 현재 사용할 수 없습니다."
            )
            self.last_error_code = error.error_code
            return self._build_scene_error_result(
                error,
                fallback_reason="service_unavailable",
            )

        prompt = build_scene_prompt(user_query)
        start_time = time.perf_counter()
        try:
            message = self.engine.generate(
                image_path=image_path,
                prompt=prompt,
                metadata={"user_query": user_query, "mode": "scene_description"},
            )
            if getattr(self.engine, "device", None) == "cuda":
                torch = getattr(self.engine, "torch", None)
                if torch is not None:
                    torch.cuda.synchronize()
        except VLMError as error:
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            if self._is_fatal_inference_error(error):
                self.is_available = False
            self.last_error_code = error.error_code
            return self._build_scene_error_result(
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
            return self._build_scene_error_result(
                error,
                latency_ms=elapsed_ms,
                fallback_reason="vlm_inference_error",
            )

        latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
        peak_gpu_memory_mb = 0.0
        if hasattr(self.engine, "get_peak_memory_mb"):
            peak_gpu_memory_mb = self.engine.get_peak_memory_mb()

        if (
            timeout_seconds is not None
            and (latency_ms / 1000) > timeout_seconds
        ):
            error = VLMTimeoutError("VLM 추론 제한 시간을 초과했습니다.")
            self.last_error_code = error.error_code
            return self._build_scene_error_result(
                error,
                latency_ms=latency_ms,
                fallback_reason="vlm_timeout",
            )

        self.last_error_code = None
        result = build_scene_result(
            message=message,
            latency_ms=latency_ms,
            peak_gpu_memory_mb=peak_gpu_memory_mb,
            model_id=getattr(self.engine, "model_id", "mock-rule-engine"),
        )
        result.update(
            {
                "service_status": "ok",
                "error": None,
                "fallback_reason": None,
            }
        )
        return result
