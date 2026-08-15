from __future__ import annotations

import unittest
from pathlib import Path

from src.exceptions import YoloResultValidationError
from src.integration_pipeline import VIAssistVLMPipeline


def yolo() -> dict:
    return {
        "image_width": 300,
        "image_height": 200,
        "detections": [
            {
                "class_name": "elevator-button",
                "confidence": 0.9,
                "bbox": [210, 20, 270, 100],
            }
        ],
    }


class FakeService:
    def __init__(self) -> None:
        self.safe_calls = []
        self.strict_calls = []
        self.safe_result = {"mode": "safe"}
        self.strict_result = {"mode": "strict"}

    def infer_safe(self, image_path, metadata, *, timeout_seconds=None):
        self.safe_calls.append((image_path, metadata, timeout_seconds))
        return self.safe_result

    def infer(self, image_path, metadata):
        self.strict_calls.append((image_path, metadata))
        return self.strict_result


class IntegrationPipelineTest(unittest.TestCase):
    def test_pipeline_keeps_and_reuses_service(self) -> None:
        service = FakeService()
        pipeline = VIAssistVLMPipeline(service)
        pipeline.process(image_path=None, yolo_result=yolo())
        pipeline.process(image_path=None, yolo_result=yolo())
        self.assertIs(pipeline.service, service)
        self.assertEqual(len(service.safe_calls), 2)

    def test_safe_true_calls_infer_safe(self) -> None:
        service = FakeService()
        result = VIAssistVLMPipeline(service).process(
            image_path=Path("image.png"),
            yolo_result=yolo(),
            safe=True,
        )
        self.assertIs(result, service.safe_result)
        self.assertEqual(len(service.safe_calls), 1)
        self.assertEqual(service.strict_calls, [])

    def test_safe_false_calls_infer(self) -> None:
        service = FakeService()
        result = VIAssistVLMPipeline(service).process(
            image_path=None,
            yolo_result=yolo(),
            safe=False,
        )
        self.assertIs(result, service.strict_result)
        self.assertEqual(len(service.strict_calls), 1)
        self.assertEqual(service.safe_calls, [])

    def test_timeout_is_forwarded_only_to_safe_inference(self) -> None:
        service = FakeService()
        VIAssistVLMPipeline(service).process(
            image_path=None,
            yolo_result=yolo(),
            timeout_seconds=3.5,
        )
        self.assertEqual(service.safe_calls[0][2], 3.5)

    def test_build_metadata_without_vlm_call(self) -> None:
        service = FakeService()
        metadata = VIAssistVLMPipeline(service).build_metadata(yolo_result=yolo())
        self.assertEqual(metadata["detections"][0]["class_name"], "elevator_button")
        self.assertEqual(service.safe_calls, [])
        self.assertEqual(service.strict_calls, [])

    def test_adapter_error_does_not_call_service(self) -> None:
        service = FakeService()
        invalid = yolo()
        invalid["detections"][0]["position"] = "invalid"
        with self.assertRaises(YoloResultValidationError):
            VIAssistVLMPipeline(service).process(
                image_path=None,
                yolo_result=invalid,
            )
        self.assertEqual(service.safe_calls, [])
        self.assertEqual(service.strict_calls, [])

    def test_result_is_returned_without_pipeline_rewriting(self) -> None:
        service = FakeService()
        expected = {"custom": [1, 2, 3]}
        service.safe_result = expected
        actual = VIAssistVLMPipeline(service).process(
            image_path=None,
            yolo_result=yolo(),
        )
        self.assertIs(actual, expected)


if __name__ == "__main__":
    unittest.main()
