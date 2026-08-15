"""Perception payload → Adapter → VLMService → Safety Validator → 최종 JSON.

카메라, CUDA, YOLO 모델 없이 mock 엔진으로 전체 경로를 검증한다.
"""

from __future__ import annotations

import unittest

from src.integration_pipeline import VIAssistVLMPipeline
from src.vlm_engine import MockVLMEngine
from src.vlm_service import VLMService

from tests.test_perception_adapter import (
    flow_payload,
    perception_detection,
    yolo_payload,
)


RESULT_KEYS = {
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
    "service_status",
    "error",
    "fallback_reason",
}


class ScriptedEngine:
    """검증 대상 문장을 그대로 돌려주는 mock 엔진."""

    model_id = "scripted-test-engine"

    def __init__(self, message: str) -> None:
        self.message = message
        self.calls: list[dict] = []

    def generate(self, image_path, prompt, metadata) -> str:
        self.calls.append(
            {"image_path": image_path, "prompt": prompt, "metadata": metadata}
        )
        return self.message


def pipeline_with(engine) -> tuple[VIAssistVLMPipeline, object]:
    service = VLMService(engine)
    return VIAssistVLMPipeline(service), engine


class PerceptionEndToEndTest(unittest.TestCase):
    def test_escalator_up_full_flow(self) -> None:
        pipeline, _ = pipeline_with(MockVLMEngine())
        result = pipeline.process_perception(
            image_path=None,
            yolo_payload=yolo_payload(),
            flow_payload=flow_payload(direction="up"),
            user_query="에스컬레이터 방향을 알려줘.",
        )
        self.assertEqual(RESULT_KEYS, set(result))
        self.assertEqual(
            result["message"], "정면에 위쪽으로 운행하는 에스컬레이터가 있습니다."
        )
        self.assertEqual(result["target"], "escalator")
        self.assertEqual(result["position"], "front")
        self.assertEqual(result["status"], "detected")
        self.assertEqual(result["service_status"], "ok")
        self.assertTrue(result["safety_validated"])
        self.assertIsNone(result["error"])

    def test_flow_unavailable_keeps_object_guidance_without_direction(self) -> None:
        pipeline, _ = pipeline_with(MockVLMEngine())
        result = pipeline.process_perception(
            image_path=None,
            yolo_payload=yolo_payload(),
            flow_payload=flow_payload(available=False, direction="up"),
            user_query="에스컬레이터 방향을 알려줘.",
        )
        self.assertIn("에스컬레이터", result["message"])
        self.assertNotIn("위쪽", result["message"])
        self.assertEqual(result["status"], "detected")

    def test_hallucinated_direction_is_replaced_by_fallback(self) -> None:
        pipeline, engine = pipeline_with(
            ScriptedEngine("정면에 아래쪽으로 운행하는 에스컬레이터가 있습니다.")
        )
        result = pipeline.process_perception(
            image_path=None,
            yolo_payload=yolo_payload(),
            flow_payload=flow_payload(direction="up"),
        )
        self.assertEqual(len(engine.calls), 1)
        self.assertTrue(result["used_fallback"])
        self.assertEqual(result["message_source"], "fallback")
        self.assertIn("direction_mismatch", result["validation_reasons"])
        self.assertEqual(
            result["message"], "정면에 위쪽으로 운행하는 에스컬레이터가 있습니다."
        )
        self.assertEqual(
            result["raw_vlm_message"],
            "정면에 아래쪽으로 운행하는 에스컬레이터가 있습니다.",
        )

    def test_movement_claim_without_flow_is_replaced(self) -> None:
        pipeline, _ = pipeline_with(
            ScriptedEngine("정면에 위쪽으로 운행하는 에스컬레이터가 있습니다.")
        )
        result = pipeline.process_perception(
            image_path=None,
            yolo_payload=yolo_payload(),
            flow_payload=flow_payload(available=False, direction="up"),
        )
        self.assertIn("motion_unavailable_but_claimed", result["validation_reasons"])
        self.assertNotIn("위쪽", result["message"])

    def test_object_not_in_detections_is_replaced(self) -> None:
        pipeline, _ = pipeline_with(
            ScriptedEngine("정면에 계단이 있습니다.")
        )
        result = pipeline.process_perception(
            image_path=None,
            yolo_payload=yolo_payload(),
            flow_payload=flow_payload(direction="up"),
        )
        self.assertTrue(result["used_fallback"])
        self.assertNotIn("계단", result["message"])

    def test_yolo_failure_skips_vlm_and_degrades(self) -> None:
        pipeline, engine = pipeline_with(ScriptedEngine("사용되면 안 되는 문장입니다."))
        result = pipeline.process_perception(
            image_path=None,
            yolo_payload=yolo_payload(ok=False, error="camera timeout"),
            flow_payload=flow_payload(ok=False),
        )
        self.assertEqual(engine.calls, [])
        self.assertEqual(RESULT_KEYS, set(result))
        self.assertEqual(
            result["message"], "주변 상황을 확인하기 어렵습니다. 카메라를 잠시 고정해 주세요."
        )
        self.assertEqual(result["fallback_reason"], "perception_unavailable")
        self.assertEqual(result["service_status"], "degraded")
        self.assertIsNone(result["target"])
        self.assertEqual(result["position"], "unknown")
        self.assertEqual(result["status"], "uncertain")
        self.assertEqual(result["raw_vlm_message"], "")
        self.assertEqual(result["error"]["type"], "perception_error")

    def test_yolo_failure_with_usable_flow_reports_yolo_unavailable(self) -> None:
        pipeline, engine = pipeline_with(ScriptedEngine("사용되면 안 되는 문장입니다."))
        result = pipeline.process_perception(
            image_path=None,
            yolo_payload=yolo_payload(ok=False),
            flow_payload=flow_payload(direction="up"),
        )
        self.assertEqual(engine.calls, [])
        self.assertEqual(result["fallback_reason"], "yolo_unavailable")
        self.assertEqual(
            result["message"], "주변 객체를 확인하기 어렵습니다. 카메라를 잠시 고정해 주세요."
        )

    def test_malformed_bbox_degrades_without_calling_vlm(self) -> None:
        pipeline, engine = pipeline_with(ScriptedEngine("사용되면 안 되는 문장입니다."))
        result = pipeline.process_perception(
            image_path=None,
            yolo_payload=yolo_payload(perception_detection(x1=900, x2=400)),
            flow_payload=flow_payload(direction="up"),
        )
        self.assertEqual(engine.calls, [])
        self.assertTrue(result["used_fallback"])
        self.assertEqual(result["service_status"], "degraded")

    def test_no_detection_does_not_invent_object(self) -> None:
        pipeline, _ = pipeline_with(MockVLMEngine())
        result = pipeline.process_perception(
            image_path=None,
            yolo_payload=yolo_payload(detections=[]),
            flow_payload=flow_payload(direction="up"),
        )
        self.assertEqual(result["status"], "not_found")
        self.assertEqual(result["target"], "unknown")
        self.assertNotIn("에스컬레이터", result["message"])

    def test_blurry_quality_uses_conservative_message(self) -> None:
        pipeline, _ = pipeline_with(MockVLMEngine())
        result = pipeline.process_perception(
            image_path=None,
            yolo_payload=yolo_payload(quality={"is_blurry": True, "blur_score": 8.0}),
            flow_payload=flow_payload(direction="up"),
        )
        self.assertEqual(result["status"], "uncertain")
        self.assertIn("흐려", result["message"])

    def test_unsupported_class_is_not_reported_as_detected(self) -> None:
        pipeline, _ = pipeline_with(MockVLMEngine())
        result = pipeline.process_perception(
            image_path=None,
            yolo_payload=yolo_payload(perception_detection(cls_name="gate")),
            flow_payload=flow_payload(direction="up"),
        )
        self.assertNotEqual(result["status"], "detected")
        self.assertEqual(result["target"], "unknown")

    def test_service_and_engine_are_reused_between_requests(self) -> None:
        pipeline, engine = pipeline_with(
            ScriptedEngine("정면에 위쪽으로 운행하는 에스컬레이터가 있습니다.")
        )
        for _ in range(3):
            pipeline.process_perception(
                image_path=None,
                yolo_payload=yolo_payload(),
                flow_payload=flow_payload(direction="up"),
            )
        self.assertEqual(len(engine.calls), 3)
        self.assertIs(pipeline.service.engine, engine)

    def test_elevator_button_position_guidance(self) -> None:
        pipeline, _ = pipeline_with(MockVLMEngine())
        result = pipeline.process_perception(
            image_path=None,
            yolo_payload=yolo_payload(
                perception_detection(
                    cls_name="elevator_button",
                    conf=0.91,
                    x1=900,
                    y1=150,
                    x2=1100,
                    y2=600,
                )
            ),
            user_query="엘리베이터 버튼이 어디 있어?",
        )
        self.assertEqual(result["target"], "elevator_button")
        self.assertEqual(result["position"], "right")
        self.assertEqual(result["message"], "오른쪽에 엘리베이터 버튼이 있습니다.")


if __name__ == "__main__":
    unittest.main()
