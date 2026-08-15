from __future__ import annotations

import copy
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from src.quality_checker import build_safe_fallback, validate_vlm_message
from src.result_parser import build_result
from src.safety_rules import select_detection
from src.vlm_engine import MockVLMEngine


ROOT = Path(__file__).resolve().parents[1]
EXISTING_OUTPUT_FIELDS = {
    "message",
    "target",
    "position",
    "status",
    "confidence",
    "latency_ms",
    "peak_gpu_memory_mb",
    "model_id",
}
SAFETY_OUTPUT_FIELDS = {
    "safety_validated",
    "used_fallback",
    "validation_reasons",
    "message_source",
    "raw_vlm_message",
    "detection_confidence",
}


def load_sample(name: str = "elevator_button") -> dict:
    return json.loads(
        (ROOT / "samples" / f"{name}.json").read_text(encoding="utf-8")
    )


def validate(message: str, metadata: dict):
    return validate_vlm_message(
        message,
        metadata,
        select_detection(metadata),
    )


def result_for(message: str, metadata: dict) -> dict:
    return build_result(
        message=message,
        metadata=metadata,
        latency_ms=1.0,
        peak_gpu_memory_mb=2.0,
        model_id="test-model",
        selected_detection=select_detection(metadata),
    )


class DetectionSelectionTest(unittest.TestCase):
    def test_query_target_has_priority_over_higher_confidence(self) -> None:
        metadata = load_sample()
        metadata["detections"].append(
            {
                "class_name": "escalator",
                "confidence": 0.99,
                "bbox": [0, 0, 100, 100],
                "position": "front",
            }
        )
        self.assertEqual(
            select_detection(metadata)["class_name"],
            "elevator_button",
        )

    def test_highest_supported_target_when_query_has_no_target(self) -> None:
        metadata = load_sample()
        metadata["user_query"] = "주변 상황을 알려 주세요."
        metadata["detections"].append(
            {
                "class_name": "escalator",
                "confidence": 0.99,
                "bbox": [0, 0, 100, 100],
                "position": "front",
            }
        )
        self.assertEqual(select_detection(metadata)["class_name"], "escalator")


class SafetyValidatorPassTest(unittest.TestCase):
    def test_right_elevator_button(self) -> None:
        self.assertTrue(
            validate("오른쪽에 엘리베이터 버튼이 있습니다.", load_sample()).is_valid
        )

    def test_right_elevator_call_button_synonym(self) -> None:
        self.assertTrue(
            validate(
                "우측에 엘리베이터 호출 버튼이 있습니다.", load_sample()
            ).is_valid
        )

    def test_left_synonym(self) -> None:
        metadata = load_sample()
        metadata["detections"][0]["position"] = "left"
        self.assertTrue(validate("좌측에 엘리베이터 버튼이 있습니다.", metadata).is_valid)

    def test_front_synonym(self) -> None:
        metadata = load_sample()
        metadata["detections"][0]["position"] = "front"
        self.assertTrue(validate("앞쪽에 엘리베이터 버튼이 있습니다.", metadata).is_valid)

    def test_escalator_up(self) -> None:
        self.assertTrue(
            validate(
                "정면에 위쪽으로 운행하는 에스컬레이터가 있습니다.",
                load_sample("escalator"),
            ).is_valid
        )

    def test_escalator_down(self) -> None:
        metadata = load_sample("escalator")
        metadata["motion"]["direction"] = "down"
        self.assertTrue(
            validate("앞에 하행 에스컬레이터가 있습니다.", metadata).is_valid
        )

    def test_escalator_up_with_upward_synonym(self) -> None:
        self.assertTrue(
            validate(
                "앞쪽에 상행 에스컬레이터가 있습니다.",
                load_sample("escalator"),
            ).is_valid
        )

    def test_escalator_down_with_downward_phrase(self) -> None:
        metadata = load_sample("escalator")
        metadata["motion"]["direction"] = "down"
        self.assertTrue(
            validate(
                "정면에 아래로 내려가는 에스컬레이터가 있습니다.", metadata
            ).is_valid
        )

    def test_two_sentence_guidance(self) -> None:
        metadata = load_sample("escalator")
        metadata["motion"] = {
            "available": False,
            "target": None,
            "direction": "unknown",
            "speed": None,
            "confidence": None,
        }
        message = "정면에 에스컬레이터가 있습니다. 운행 방향은 확인하기 어렵습니다."
        self.assertTrue(validate(message, metadata).is_valid)


class SafetyValidatorFallbackTest(unittest.TestCase):
    def assert_fallback_reason(
        self, message: str, metadata: dict, reason: str
    ) -> None:
        validation = validate(message, metadata)
        self.assertFalse(validation.is_valid)
        self.assertTrue(validation.used_fallback)
        self.assertIn(reason, validation.reasons)

    def test_wrong_position(self) -> None:
        self.assert_fallback_reason(
            "왼쪽에 엘리베이터 버튼이 있습니다.",
            load_sample(),
            "position_mismatch",
        )

    def test_question_with_question_mark(self) -> None:
        self.assert_fallback_reason(
            "앞에 에스컬레이터가 있나요?",
            load_sample("escalator"),
            "question_form_output",
        )

    def test_question_ending_without_question_mark(self) -> None:
        self.assert_fallback_reason(
            "에스컬레이터가 앞에 있나요",
            load_sample("escalator"),
            "question_form_output",
        )

    def test_output_identical_to_user_query(self) -> None:
        metadata = load_sample()
        self.assert_fallback_reason(
            metadata["user_query"], metadata, "repeated_user_query"
        )

    def test_output_repeats_user_query_with_label(self) -> None:
        metadata = load_sample()
        self.assert_fallback_reason(
            f"질문: {metadata['user_query']}", metadata, "repeated_user_query"
        )

    def test_unnatural_korean_style(self) -> None:
        self.assert_fallback_reason(
            "엘리베이터 버튼이 오른쪽에 있는 것입니다.",
            load_sample(),
            "unnatural_korean_style",
        )

    def test_unnatural_existence_style(self) -> None:
        self.assert_fallback_reason(
            "엘리베이터 버튼이 오른쪽에 존재하는 것입니다.",
            load_sample(),
            "unnatural_korean_style",
        )

    def test_escalator_up_requires_direction(self) -> None:
        self.assert_fallback_reason(
            "정면에 에스컬레이터가 있습니다.",
            load_sample("escalator"),
            "required_motion_direction_missing",
        )

    def test_escalator_down_requires_direction(self) -> None:
        metadata = load_sample("escalator")
        metadata["motion"]["direction"] = "down"
        self.assert_fallback_reason(
            "정면에 에스컬레이터가 있습니다.",
            metadata,
            "required_motion_direction_missing",
        )

    def test_incomplete_noun_ending(self) -> None:
        self.assert_fallback_reason(
            "오른쪽 엘리베이터 버튼",
            load_sample(),
            "invalid_guidance_ending",
        )

    def test_object_not_in_detection(self) -> None:
        self.assert_fallback_reason(
            "오른쪽에 계단과 엘리베이터 버튼이 있습니다.",
            load_sample(),
            "object_not_in_metadata",
        )

    def test_motion_unavailable_but_up_claimed(self) -> None:
        metadata = load_sample("escalator")
        metadata["motion"] = {
            "available": False,
            "target": None,
            "direction": "unknown",
            "speed": None,
            "confidence": None,
        }
        self.assert_fallback_reason(
            "정면에 상행 에스컬레이터가 있습니다.",
            metadata,
            "motion_unavailable_but_claimed",
        )

    def test_motion_up_but_down_claimed(self) -> None:
        self.assert_fallback_reason(
            "정면에 하행 에스컬레이터가 있습니다.",
            load_sample("escalator"),
            "direction_mismatch",
        )

    def test_unsupported_50cm_distance(self) -> None:
        self.assert_fallback_reason(
            "오른쪽 50cm 앞에 엘리베이터 버튼이 있습니다.",
            load_sample(),
            "unsupported_distance",
        )

    def test_unsupported_one_meter_distance(self) -> None:
        self.assert_fallback_reason(
            "오른쪽 1미터 거리에 엘리베이터 버튼이 있습니다.",
            load_sample(),
            "unsupported_distance",
        )

    def test_safety_judgment(self) -> None:
        self.assert_fallback_reason(
            "오른쪽에 엘리베이터 버튼이 있어 안전합니다.",
            load_sample(),
            "unsupported_safety_judgment",
        )

    def test_unsupported_object_state(self) -> None:
        self.assert_fallback_reason(
            "오른쪽에 엘리베이터 버튼이 정상 작동 중입니다.",
            load_sample(),
            "unsupported_object_state",
        )

    def test_boarding_permission(self) -> None:
        self.assert_fallback_reason(
            "정면에 상행 에스컬레이터가 있어 탑승해도 됩니다.",
            load_sample("escalator"),
            "excessive_action_instruction",
        )

    def test_press_button_instruction(self) -> None:
        self.assert_fallback_reason(
            "오른쪽에 엘리베이터 버튼이 있으니 바로 버튼을 누르세요.",
            load_sample(),
            "excessive_action_instruction",
        )

    def test_empty_output(self) -> None:
        self.assert_fallback_reason("   ", load_sample(), "empty_message")

    def test_json_output(self) -> None:
        self.assert_fallback_reason(
            '{"message": "오른쪽에 엘리베이터 버튼이 있습니다."}',
            load_sample(),
            "json_output",
        )

    def test_markdown_code_block(self) -> None:
        self.assert_fallback_reason(
            "```\n오른쪽에 엘리베이터 버튼이 있습니다.\n```",
            load_sample(),
            "markdown_or_list_output",
        )

    def test_markdown_heading(self) -> None:
        self.assert_fallback_reason(
            "# 안내\n오른쪽에 엘리베이터 버튼이 있습니다.",
            load_sample(),
            "markdown_or_list_output",
        )

    def test_markdown_list(self) -> None:
        self.assert_fallback_reason(
            "- 오른쪽에 엘리베이터 버튼이 있습니다.",
            load_sample(),
            "markdown_or_list_output",
        )

    def test_three_sentences(self) -> None:
        self.assert_fallback_reason(
            "오른쪽에 엘리베이터 버튼이 있습니다. 버튼이 보입니다. 확인했습니다.",
            load_sample(),
            "too_many_sentences",
        )

    def test_message_over_120_characters(self) -> None:
        message = "오른쪽에 엘리베이터 버튼이 있습니다. " + "안내" * 50
        self.assert_fallback_reason(
            message, load_sample(), "message_too_long"
        )

    def test_blurry_image(self) -> None:
        metadata = load_sample()
        metadata["image_quality"]["is_blurry"] = True
        result = result_for("오른쪽에 엘리베이터 버튼이 있습니다.", metadata)
        self.assertEqual(result["status"], "uncertain")
        self.assertEqual(
            result["message"],
            "화면이 흐려 확인하기 어렵습니다. 카메라를 잠시 고정해 주세요.",
        )

    def test_no_detection(self) -> None:
        metadata = load_sample()
        metadata["detections"] = []
        result = result_for("오른쪽에 엘리베이터 버튼이 있습니다.", metadata)
        self.assertEqual(result["status"], "not_found")
        self.assertEqual(
            result["message"],
            "목표 객체를 확인하기 어렵습니다. 카메라를 천천히 좌우로 움직여 주세요.",
        )

    def test_low_confidence(self) -> None:
        metadata = load_sample()
        metadata["detections"][0]["confidence"] = 0.49
        result = result_for("오른쪽에 엘리베이터 버튼이 있습니다.", metadata)
        self.assertEqual(result["status"], "uncertain")
        self.assertIn("low_detection_confidence", result["validation_reasons"])

    def test_only_unsupported_object(self) -> None:
        metadata = load_sample()
        metadata["user_query"] = "주변 상황을 알려 주세요."
        metadata["detections"][0]["class_name"] = "door"
        result = result_for("오른쪽에 문이 있습니다.", metadata)
        self.assertEqual(
            result["message"],
            "대상이 보이지만 정확한 안내를 제공하기 어렵습니다.",
        )
        self.assertTrue(result["used_fallback"])

    def test_invalid_metadata_state_uses_fallback(self) -> None:
        metadata = load_sample()
        metadata["schema_version"] = "invalid"
        result = build_result(
            message="오른쪽에 엘리베이터 버튼이 있습니다.",
            metadata=metadata,
            latency_ms=1.0,
            peak_gpu_memory_mb=2.0,
            model_id="test-model",
        )
        self.assertEqual(result["status"], "uncertain")
        self.assertIn("invalid_metadata", result["validation_reasons"])
        self.assertEqual(
            result["message"],
            "입력 정보를 확인하기 어렵습니다. 다시 시도해 주세요.",
        )

    def test_malformed_nested_metadata_does_not_raise(self) -> None:
        metadata = load_sample()
        metadata["motion"] = []
        validation = validate_vlm_message(
            "오른쪽에 엘리베이터 버튼이 있습니다.",
            metadata,
            None,
        )
        self.assertFalse(validation.is_valid)
        self.assertIn("invalid_metadata", validation.reasons)


class SafetyIntegrationTest(unittest.TestCase):
    def test_fallback_is_deterministic(self) -> None:
        metadata = load_sample()
        selected = select_detection(metadata)
        first = build_safe_fallback(metadata, selected, "position_mismatch")
        second = build_safe_fallback(metadata, selected, "position_mismatch")
        self.assertEqual(first, second)

    def test_elevator_fallback_uses_selected_position(self) -> None:
        expected = {
            "left": "왼쪽에 엘리베이터 버튼이 있습니다.",
            "front": "정면에 엘리베이터 버튼이 있습니다.",
            "right": "오른쪽에 엘리베이터 버튼이 있습니다.",
            "unknown": "엘리베이터 버튼이 보이지만 위치를 정확히 확인하기 어렵습니다.",
        }
        for position, message in expected.items():
            with self.subTest(position=position):
                metadata = load_sample()
                metadata["detections"][0]["position"] = position
                self.assertEqual(
                    build_safe_fallback(
                        metadata,
                        select_detection(metadata),
                        "validation_failed",
                    ),
                    message,
                )

    def test_escalator_unknown_motion_fallback(self) -> None:
        metadata = load_sample("escalator")
        metadata["motion"] = {
            "available": False,
            "target": None,
            "direction": "unknown",
            "speed": None,
            "confidence": None,
        }
        self.assertEqual(
            build_safe_fallback(
                metadata,
                select_detection(metadata),
                "validation_failed",
            ),
            "정면에 에스컬레이터가 있습니다. 운행 방향은 확인하기 어렵습니다.",
        )

    def test_mock_and_simulated_vlm_use_same_result_safety_path(self) -> None:
        metadata = load_sample()
        selected = select_detection(metadata)
        mock_message = MockVLMEngine().generate(None, "unused", metadata)
        mock_result = result_for(mock_message, metadata)
        simulated_vlm_result = result_for(
            "왼쪽에 엘리베이터 버튼이 있습니다.", metadata
        )
        self.assertEqual(mock_result["message_source"], "vlm")
        self.assertEqual(simulated_vlm_result["message_source"], "fallback")
        self.assertEqual(
            simulated_vlm_result["message"],
            build_safe_fallback(metadata, selected, "position_mismatch"),
        )

    def test_result_keeps_old_fields_and_adds_safety_fields(self) -> None:
        result = result_for(
            "오른쪽에 엘리베이터 버튼이 있습니다.", load_sample()
        )
        self.assertTrue(EXISTING_OUTPUT_FIELDS <= result.keys())
        self.assertTrue(SAFETY_OUTPUT_FIELDS <= result.keys())
        self.assertEqual(result["detection_confidence"], 0.91)
        self.assertIsInstance(json.dumps(result, ensure_ascii=False), str)

    def test_sample_mock_cli_runs_without_loading_model(self) -> None:
        with tempfile.TemporaryDirectory() as temp_directory:
            for sample in ("elevator_button", "escalator"):
                with self.subTest(sample=sample):
                    output = Path(temp_directory) / f"{sample}.json"
                    completed = subprocess.run(
                        [
                            sys.executable,
                            "main.py",
                            "--metadata",
                            f"samples/{sample}.json",
                            "--image",
                            f"samples/{sample}.png",
                            "--engine",
                            "mock",
                            "--output",
                            str(output),
                        ],
                        cwd=ROOT,
                        capture_output=True,
                        text=True,
                        check=False,
                    )
                    self.assertEqual(completed.returncode, 0, completed.stderr)
                    result = json.loads(output.read_text(encoding="utf-8"))
                    self.assertTrue(EXISTING_OUTPUT_FIELDS <= result.keys())
                    self.assertTrue(SAFETY_OUTPUT_FIELDS <= result.keys())


if __name__ == "__main__":
    unittest.main()
