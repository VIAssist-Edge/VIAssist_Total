from __future__ import annotations

from typing import Any


class VLMError(Exception):
    default_error_code = "UNKNOWN_ERROR"
    default_retryable = False

    def __init__(
        self,
        message: str,
        *,
        error_code: str | None = None,
        retryable: bool | None = None,
        original_exception: BaseException | None = None,
    ) -> None:
        super().__init__(message)
        self.error_code = error_code or self.default_error_code
        self.message = message
        self.retryable = self.default_retryable if retryable is None else retryable
        self.original_exception = original_exception

    def public_error(self) -> dict[str, Any]:
        """Traceback이나 내부 예외 없이 외부 노출용 정보만 반환한다."""

        return {
            "code": self.error_code,
            "message": self.message,
            "retryable": self.retryable,
        }


class VLMConfigurationError(VLMError):
    default_error_code = "CONFIG_ERROR"


class VLMModelLoadError(VLMError):
    default_error_code = "MODEL_LOAD_ERROR"


class VLMInferenceError(VLMError):
    default_error_code = "INFERENCE_ERROR"
    default_retryable = True


class VLMOutOfMemoryError(VLMInferenceError):
    default_error_code = "CUDA_OUT_OF_MEMORY"


class VLMTimeoutError(VLMInferenceError):
    default_error_code = "INFERENCE_TIMEOUT"


class VLMImageError(VLMError):
    default_error_code = "IMAGE_INVALID"
    default_retryable = True


class VLMMetadataError(VLMError):
    default_error_code = "METADATA_INVALID"
    default_retryable = True


class VLMServiceUnavailableError(VLMError):
    default_error_code = "SERVICE_UNAVAILABLE"
    default_retryable = False


class IntegrationError(VLMError):
    default_error_code = "INTEGRATION_ERROR"


class YoloResultValidationError(IntegrationError):
    default_error_code = "YOLO_RESULT_INVALID"


class MotionResultValidationError(IntegrationError):
    default_error_code = "MOTION_RESULT_INVALID"


class FrameSynchronizationError(IntegrationError):
    default_error_code = "FRAME_MISMATCH"


class PerceptionUnavailableError(IntegrationError):
    """Perception payload가 ok=false이거나 사용할 수 없는 상태일 때 발생한다."""

    default_error_code = "PERCEPTION_UNAVAILABLE"
    default_retryable = True
