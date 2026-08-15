from __future__ import annotations

import json
import unittest
from pathlib import Path
from unittest.mock import patch

from src.config_loader import load_config
from src.vlm_service import VLMService


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_FIELDS = {
    "message",
    "target",
    "position",
    "status",
    "latency_ms",
    "model_id",
    "safety_validated",
    "used_fallback",
    "raw_vlm_message",
}


def load_metadata() -> dict:
    return json.loads(
        (ROOT / "samples" / "elevator_button.json").read_text(encoding="utf-8")
    )


class FakeEngine:
    instances = 0

    def __init__(self, message: str = "오른쪽에 엘리베이터 버튼이 있습니다.") -> None:
        type(self).instances += 1
        self.message = message
        self.generate_calls = 0
        self.model_id = "fake-model"
        self.device = "cpu"

    def generate(self, image_path, prompt, metadata) -> str:
        self.generate_calls += 1
        return self.message

    def get_peak_memory_mb(self) -> float:
        return 0.0


class VLMServiceTest(unittest.TestCase):
    def setUp(self) -> None:
        FakeEngine.instances = 0

    def test_infer_preserves_result_contract(self) -> None:
        service = VLMService(FakeEngine())
        result = service.infer(None, load_metadata())
        self.assertTrue(EXPECTED_FIELDS.issubset(result))
        self.assertEqual(result["model_id"], "fake-model")
        self.assertFalse(result["used_fallback"])

    def test_repeated_infer_reuses_same_engine(self) -> None:
        engine = FakeEngine()
        service = VLMService(engine)
        service.infer(None, load_metadata())
        service.infer(None, load_metadata())
        self.assertEqual(FakeEngine.instances, 1)
        self.assertIs(service.engine, engine)
        self.assertEqual(engine.generate_calls, 2)

    def test_question_output_is_replaced_by_fallback(self) -> None:
        raw_message = "오른쪽에 엘리베이터 버튼이 있나요?"
        service = VLMService(FakeEngine(raw_message))
        result = service.infer(None, load_metadata())
        self.assertTrue(result["used_fallback"])
        self.assertIn("question_form_output", result["validation_reasons"])
        self.assertEqual(result["raw_vlm_message"], raw_message)
        self.assertNotEqual(result["message"], raw_message)

    def test_mock_config_does_not_require_model_or_cuda(self) -> None:
        service = VLMService.from_config(
            {
                "environment": "test",
                "device": "cuda",
                "max_new_tokens": 60,
                "use_mock_model": True,
                "model_id": "not-downloaded/model",
                "image_longest_edge": 1024,
                "max_image_size": 512,
            }
        )
        result = service.infer(None, load_metadata())
        self.assertEqual(result["model_id"], "mock-rule-engine")

    def test_jetson_stabilization_values_are_injected_into_engine(self) -> None:
        config = load_config(ROOT / "config" / "jetson.json")
        sentinel_engine = FakeEngine()
        with patch(
            "src.vlm_service.SmolVLMEngine", return_value=sentinel_engine
        ) as engine_class:
            service = VLMService.from_config(config)

        self.assertIs(service.engine, sentinel_engine)
        engine_class.assert_called_once_with(
            model_id="HuggingFaceTB/SmolVLM-500M-Instruct",
            max_new_tokens=60,
            device="cuda",
            image_longest_edge=1024,
            max_image_size=512,
        )


if __name__ == "__main__":
    unittest.main()
