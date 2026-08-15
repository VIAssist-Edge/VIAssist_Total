from __future__ import annotations

import json
import time
import unittest
from pathlib import Path

from src.exceptions import (
    VLMImageError,
    VLMInferenceError,
    VLMOutOfMemoryError,
)
from src.vlm_service import VLMService


ROOT = Path(__file__).resolve().parents[1]
EXISTING_RESULT_FIELDS = {
    "message",
    "target",
    "position",
    "status",
    "confidence",
    "detection_confidence",
    "latency_ms",
    "peak_gpu_memory_mb",
    "model_id",
    "safety_validated",
    "used_fallback",
    "validation_reasons",
    "message_source",
    "raw_vlm_message",
}


def valid_metadata() -> dict:
    return json.loads(
        (ROOT / "samples" / "elevator_button.json").read_text(encoding="utf-8")
    )


class FakeEngine:
    def __init__(self, outcome, *, delay: float = 0.0) -> None:
        self.outcome = outcome
        self.delay = delay
        self.generate_calls = 0
        self.model_id = "fake-model"
        self.device = "cpu"

    def generate(self, image_path, prompt, metadata) -> str:
        self.generate_calls += 1
        if self.delay:
            time.sleep(self.delay)
        if isinstance(self.outcome, BaseException):
            raise self.outcome
        return self.outcome

    def get_peak_memory_mb(self) -> float:
        return 12.5


class VLMErrorHandlingTest(unittest.TestCase):
    def test_normal_infer_safe(self) -> None:
        service = VLMService(
            FakeEngine("오른쪽에 엘리베이터 버튼이 있습니다.")
        )
        result = service.infer_safe(None, valid_metadata())
        self.assertEqual(result["service_status"], "ok")
        self.assertIsNone(result["error"])
        self.assertIsNone(result["fallback_reason"])

    def test_image_error_returns_fallback_and_stays_available(self) -> None:
        engine = FakeEngine(VLMImageError("이미지를 처리할 수 없습니다."))
        service = VLMService(engine)
        result = service.infer_safe(None, valid_metadata())
        self.assertEqual(result["service_status"], "degraded")
        self.assertEqual(result["error"]["code"], "IMAGE_INVALID")
        self.assertTrue(result["used_fallback"])
        self.assertTrue(service.is_available)

    def test_general_inference_error_returns_degraded_fallback(self) -> None:
        service = VLMService(
            FakeEngine(VLMInferenceError("VLM 추론에 실패했습니다."))
        )
        result = service.infer_safe(None, valid_metadata())
        self.assertEqual(result["service_status"], "degraded")
        self.assertEqual(result["error"]["code"], "INFERENCE_ERROR")
        self.assertTrue(service.is_available)

    def test_cuda_oom_marks_service_unavailable(self) -> None:
        service = VLMService(
            FakeEngine(VLMOutOfMemoryError("VLM 추론 메모리가 부족합니다."))
        )
        result = service.infer_safe(None, valid_metadata())
        self.assertEqual(result["service_status"], "unavailable")
        self.assertEqual(result["error"]["code"], "CUDA_OUT_OF_MEMORY")
        self.assertFalse(service.is_available)

    def test_unavailable_service_does_not_call_engine_again(self) -> None:
        engine = FakeEngine(
            VLMOutOfMemoryError("VLM 추론 메모리가 부족합니다.")
        )
        service = VLMService(engine)
        service.infer_safe(None, valid_metadata())
        result = service.infer_safe(None, valid_metadata())
        self.assertEqual(engine.generate_calls, 1)
        self.assertEqual(result["error"]["code"], "SERVICE_UNAVAILABLE")
        self.assertEqual(result["service_status"], "unavailable")

    def test_elapsed_timeout_returns_fallback_without_unavailable(self) -> None:
        service = VLMService(
            FakeEngine(
                "오른쪽에 엘리베이터 버튼이 있습니다.",
                delay=0.02,
            )
        )
        result = service.infer_safe(
            None, valid_metadata(), timeout_seconds=0.001
        )
        self.assertEqual(result["error"]["code"], "INFERENCE_TIMEOUT")
        self.assertEqual(result["service_status"], "degraded")
        self.assertTrue(service.is_available)

    def test_invalid_timeout_values(self) -> None:
        service = VLMService(FakeEngine("unused"))
        for value in (0, -1, True, "1"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                service.infer_safe(None, valid_metadata(), timeout_seconds=value)

    def test_error_fallback_preserves_existing_result_contract(self) -> None:
        service = VLMService(
            FakeEngine(VLMInferenceError("VLM 추론에 실패했습니다."))
        )
        result = service.infer_safe(None, valid_metadata())
        self.assertTrue(EXISTING_RESULT_FIELDS.issubset(result))
        self.assertEqual(result["raw_vlm_message"], "")
        self.assertEqual(result["message_source"], "fallback")
        self.assertTrue(result["safety_validated"])

    def test_invalid_metadata_returns_safe_metadata_error(self) -> None:
        engine = FakeEngine("unused")
        service = VLMService(engine)
        result = service.infer_safe(None, {})
        self.assertEqual(result["error"]["code"], "METADATA_INVALID")
        self.assertEqual(
            result["message"],
            "입력 정보를 확인하기 어렵습니다. 다시 시도해 주세요.",
        )
        self.assertEqual(engine.generate_calls, 0)
        self.assertTrue(service.is_available)


if __name__ == "__main__":
    unittest.main()
