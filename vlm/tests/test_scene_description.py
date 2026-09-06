from __future__ import annotations

import unittest

from src.scene_description import (
    SCENE_FALLBACK_MESSAGE,
    build_scene_result,
    validate_scene_message,
)
from src.vlm_service import VLMService


class SceneValidationTest(unittest.TestCase):
    def test_valid_scene_description_is_accepted(self) -> None:
        result = validate_scene_message("정면에 사람이 서 있습니다.")
        self.assertTrue(result.is_valid)
        self.assertEqual(result.message_source, "vlm")

    def test_mentioning_person_is_not_blocked(self) -> None:
        """SUPPORTED_TARGETS(escalator/elevator_button) 밖의 객체도 허용한다."""

        result = validate_scene_message("왼쪽에 자전거를 탄 사람이 지나가고 있습니다.")
        self.assertTrue(result.is_valid)

    def test_question_form_is_rejected(self) -> None:
        result = validate_scene_message("앞에 사람이 있나요?")
        self.assertFalse(result.is_valid)
        self.assertIn("question_form_output", result.reasons)

    def test_safety_judgment_is_rejected(self) -> None:
        result = validate_scene_message("안전합니다. 지나가도 됩니다.")
        self.assertFalse(result.is_valid)
        self.assertIn("unsupported_safety_judgment", result.reasons)

    def test_distance_claim_is_rejected(self) -> None:
        result = validate_scene_message("50센티미터 앞에 사람이 있습니다.")
        self.assertFalse(result.is_valid)
        self.assertIn("unsupported_distance", result.reasons)

    def test_excessive_action_instruction_is_rejected(self) -> None:
        result = validate_scene_message("바로 지나가세요.")
        self.assertFalse(result.is_valid)
        self.assertIn("excessive_action_instruction", result.reasons)

    def test_referential_filler_sentence_is_rejected(self) -> None:
        """실제 내용 없이 '아래에 있습니다' 식으로 참조만 하는 문장은 차단한다."""

        result = validate_scene_message("주변 상황은 아래에 있습니다.")
        self.assertFalse(result.is_valid)
        self.assertIn("unnatural_korean_style", result.reasons)

    def test_literal_position_below_is_not_blocked(self) -> None:
        """주격 조사로 실제 위치를 설명하는 문장은 걸리지 않아야 한다."""

        result = validate_scene_message("계단이 아래에 있습니다.")
        self.assertTrue(result.is_valid)

    def test_too_many_sentences_is_rejected(self) -> None:
        result = validate_scene_message(
            "정면에 사람이 있습니다. 오른쪽에 자전거가 있습니다. 왼쪽에 차량이 있습니다."
        )
        self.assertFalse(result.is_valid)
        self.assertIn("too_many_sentences", result.reasons)

    def test_empty_message_falls_back(self) -> None:
        result = build_scene_result(
            message="",
            latency_ms=1.0,
            peak_gpu_memory_mb=0.0,
            model_id="fake-model",
        )
        self.assertEqual(result["message"], SCENE_FALLBACK_MESSAGE)
        self.assertTrue(result["used_fallback"])


class FakeEngine:
    def __init__(self, message: str = "정면에 사람이 있습니다.") -> None:
        self.message = message
        self.generate_calls = 0
        self.model_id = "fake-model"
        self.device = "cpu"

    def generate(self, image_path, prompt, metadata) -> str:
        self.generate_calls += 1
        return self.message

    def get_peak_memory_mb(self) -> float:
        return 0.0


class VLMServiceDescribeSceneTest(unittest.TestCase):
    def test_valid_message_is_used_as_is(self) -> None:
        service = VLMService(FakeEngine("정면에 사람이 서 있습니다."))
        result = service.describe_scene(None, "주변 상황을 설명해줘.")
        self.assertEqual(result["message"], "정면에 사람이 서 있습니다.")
        self.assertFalse(result["used_fallback"])
        self.assertEqual(result["service_status"], "ok")
        self.assertEqual(result["mode"], "scene_description")

    def test_question_output_is_replaced_by_fallback(self) -> None:
        service = VLMService(FakeEngine("앞에 사람이 있나요?"))
        result = service.describe_scene(None, "주변 상황을 설명해줘.")
        self.assertTrue(result["used_fallback"])
        self.assertEqual(result["message"], SCENE_FALLBACK_MESSAGE)
        self.assertIn("question_form_output", result["validation_reasons"])

    def test_service_unavailable_returns_fallback(self) -> None:
        engine = FakeEngine()
        service = VLMService(engine)
        service.is_available = False
        result = service.describe_scene(None, "주변 상황을 설명해줘.")
        self.assertTrue(result["used_fallback"])
        self.assertEqual(result["service_status"], "unavailable")
        self.assertEqual(engine.generate_calls, 0)


if __name__ == "__main__":
    unittest.main()
