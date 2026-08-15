from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.run_integration import load_json, run_integration
from src.exceptions import FrameSynchronizationError


ROOT = Path(__file__).resolve().parents[1]


def config() -> dict:
    return {
        "environment": "test",
        "device": "cpu",
        "model_id": "fake-model",
        "use_mock_model": True,
    }


class FakeService:
    def __init__(self) -> None:
        self.calls = 0

    def infer_safe(self, image_path, metadata, *, timeout_seconds=None):
        self.calls += 1
        return {"result": "safe", "target": metadata["detections"][0]["class_name"]}

    def infer(self, image_path, metadata):
        self.calls += 1
        return {"result": "strict"}


class RunIntegrationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.output = Path(self.tempdir.name) / "nested" / "result.json"
        self.metadata_output = Path(self.tempdir.name) / "nested" / "metadata.json"
        self.yolo = load_json(
            ROOT / "samples" / "integration" / "yolo_elevator.json"
        )
        self.motion = load_json(
            ROOT / "samples" / "integration" / "motion_unavailable.json"
        )
        self.quality = load_json(
            ROOT / "samples" / "integration" / "quality_clear.json"
        )

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def test_sample_load_output_save_and_factory_once(self) -> None:
        factory_calls = 0
        service = FakeService()

        def factory(input_config):
            nonlocal factory_calls
            factory_calls += 1
            return service

        metadata, result = run_integration(
            config=config(),
            image_path=ROOT / "samples" / "elevator_button.png",
            yolo_result=self.yolo,
            motion_result=self.motion,
            image_quality=self.quality,
            user_query="버튼 위치를 알려줘.",
            output_path=self.output,
            metadata_output_path=self.metadata_output,
            strict_frame_sync=False,
            strict_inference=False,
            timeout_seconds=10,
            service_factory=factory,
        )
        self.assertEqual(factory_calls, 1)
        self.assertEqual(service.calls, 1)
        self.assertEqual(result["target"], "elevator_button")
        self.assertEqual(
            json.loads(self.metadata_output.read_text(encoding="utf-8")), metadata
        )
        self.assertEqual(json.loads(self.output.read_text(encoding="utf-8")), result)

    def test_perception_payloads_reach_the_safe_path(self) -> None:
        service = FakeService()
        metadata, result = run_integration(
            config=config(),
            image_path=None,
            yolo_result=load_json(
                ROOT / "samples" / "integration" / "perception_yolo_escalator.json"
            ),
            motion_result=load_json(
                ROOT
                / "samples"
                / "integration"
                / "perception_flow_escalator_up.json"
            ),
            image_quality=None,
            user_query="에스컬레이터 방향을 알려줘.",
            output_path=self.output,
            metadata_output_path=self.metadata_output,
            strict_frame_sync=False,
            strict_inference=False,
            timeout_seconds=10,
            service_factory=lambda _: service,
            perception=True,
        )
        self.assertEqual(service.calls, 1)
        self.assertEqual(result["target"], "escalator")
        self.assertEqual(metadata["detections"][0]["bbox"], [400.0, 100.0, 900.0, 700.0])
        self.assertTrue(metadata["motion"]["available"])
        self.assertEqual(metadata["motion"]["direction"], "up")
        self.assertFalse(metadata["image_quality"]["is_blurry"])

    def test_perception_failure_degrades_without_calling_service(self) -> None:
        service = FakeService()
        _, result = run_integration(
            config=config(),
            image_path=None,
            yolo_result=load_json(
                ROOT / "samples" / "integration" / "perception_failed.json"
            ),
            motion_result=None,
            image_quality=None,
            user_query="",
            output_path=self.output,
            metadata_output_path=None,
            strict_frame_sync=False,
            strict_inference=False,
            timeout_seconds=None,
            service_factory=lambda _: service,
            perception=True,
        )
        self.assertEqual(service.calls, 0)
        self.assertEqual(result["fallback_reason"], "perception_unavailable")
        self.assertEqual(result["service_status"], "degraded")
        self.assertEqual(
            json.loads(self.output.read_text(encoding="utf-8")), result
        )

    def test_perception_rejects_strict_inference(self) -> None:
        with self.assertRaises(ValueError):
            run_integration(
                config=config(),
                image_path=None,
                yolo_result=self.yolo,
                motion_result=None,
                image_quality=None,
                user_query="",
                output_path=self.output,
                metadata_output_path=None,
                strict_frame_sync=False,
                strict_inference=True,
                timeout_seconds=None,
                service_factory=lambda _: FakeService(),
                perception=True,
            )

    def test_load_json_rejects_invalid_json(self) -> None:
        path = Path(self.tempdir.name) / "invalid.json"
        path.write_text("{invalid", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "JSON"):
            load_json(path)

    def test_strict_frame_mismatch_raises(self) -> None:
        motion = dict(self.motion)
        motion["frame_id"] = 999
        service = FakeService()
        with self.assertRaises(FrameSynchronizationError):
            run_integration(
                config=config(),
                image_path=None,
                yolo_result=self.yolo,
                motion_result=motion,
                image_quality=self.quality,
                user_query="",
                output_path=self.output,
                metadata_output_path=None,
                strict_frame_sync=True,
                strict_inference=False,
                timeout_seconds=None,
                service_factory=lambda _: service,
            )
        self.assertEqual(service.calls, 0)


if __name__ == "__main__":
    unittest.main()
