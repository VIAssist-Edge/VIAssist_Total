"""Perception → Adapter → VLMService → Safety Validator → 최종 JSON mock 통합 테스트.

카메라, CUDA, YOLO 모델 없이 numpy 프레임과 mock 엔진만으로 실행한다.
"""

from __future__ import annotations

import sys
import threading
import unittest
from pathlib import Path

import cv2
import numpy as np

MVP_ROOT = Path(__file__).resolve().parents[1]
if str(MVP_ROOT) not in sys.path:
    sys.path.insert(0, str(MVP_ROOT))

from perception_payload import build_flow_payload, build_yolo_payload  # noqa: E402
from vlm_bridge import VLMBridge, VLMBusyError  # noqa: E402

from src.integration_pipeline import VIAssistVLMPipeline  # noqa: E402
from src.vlm_engine import MockVLMEngine  # noqa: E402
from src.vlm_service import VLMService  # noqa: E402


FRAME_WIDTH = 640
FRAME_HEIGHT = 480


def frame() -> np.ndarray:
    image = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
    image[100:400, 150:500] = 180
    return image


def escalator_detection(**overrides) -> dict:
    value = {
        "class_id": 1,
        "class_name": "escalator",
        "confidence": 0.93,
        "bbox": (150, 100, 500, 400),
        "area": 350 * 300,
    }
    value.update(overrides)
    return value


def payloads(
    *,
    detections=(escalator_detection(),),
    direction: str = "UP",
    frame_id: int = 42,
    yolo_ok: bool = True,
) -> tuple[dict, dict]:
    yolo = build_yolo_payload(
        frame_id=frame_id,
        timestamp=1750000000.0,
        image_width=FRAME_WIDTH,
        image_height=FRAME_HEIGHT,
        detections=detections,
        latency_ms=31.0,
        ok=yolo_ok,
        error=None if yolo_ok else "yolo failed",
    )
    flow = build_flow_payload(
        frame_id=frame_id,
        image_width=FRAME_WIDTH,
        image_height=FRAME_HEIGHT,
        direction=direction,
        analysis={
            "dx": 0.1,
            "dy": -1.6,
            "magnitude": 1.6,
            "confidence": 0.81,
            "valid_ratio": 0.42,
            "roi": (170.0, 120.0, 480.0, 380.0),
        },
        stable_frames=9,
    )
    return yolo, flow


class RecordingEngine:
    """전달된 이미지 경로와 호출 횟수를 기록하는 mock 엔진."""

    model_id = "recording-mock-engine"

    def __init__(self, message: str) -> None:
        self.message = message
        self.calls: list[dict] = []

    def generate(self, image_path, prompt, metadata) -> str:
        self.calls.append(
            {
                "image_path": image_path,
                "exists_during_call": image_path is not None
                and Path(image_path).is_file(),
                "metadata": metadata,
            }
        )
        return self.message


def bridge_with(engine) -> VLMBridge:
    pipeline = VIAssistVLMPipeline(VLMService(engine))
    return VLMBridge(pipeline, timeout_seconds=None)


class MvpVlmIntegrationTest(unittest.TestCase):
    def test_escalator_up_end_to_end(self) -> None:
        bridge = bridge_with(MockVLMEngine())
        yolo, flow = payloads(direction="UP")
        result = bridge.describe(
            frame=frame(),
            yolo_payload=yolo,
            flow_payload=flow,
            user_query="에스컬레이터 방향을 알려줘.",
        )
        self.assertEqual(
            result["message"], "정면에 위쪽으로 운행하는 에스컬레이터가 있습니다."
        )
        self.assertEqual(result["target"], "escalator")
        self.assertEqual(result["position"], "front")
        self.assertEqual(result["status"], "detected")
        self.assertEqual(result["service_status"], "ok")
        self.assertTrue(result["safety_validated"])

    def test_escalator_down_end_to_end(self) -> None:
        bridge = bridge_with(MockVLMEngine())
        yolo, flow = payloads(direction="DOWN")
        result = bridge.describe(
            frame=frame(), yolo_payload=yolo, flow_payload=flow
        )
        self.assertIn("아래쪽", result["message"])

    def test_stationary_does_not_claim_direction(self) -> None:
        bridge = bridge_with(MockVLMEngine())
        yolo, flow = payloads(direction="STATIONARY")
        result = bridge.describe(
            frame=frame(), yolo_payload=yolo, flow_payload=flow
        )
        self.assertNotIn("위쪽", result["message"])
        self.assertNotIn("아래쪽", result["message"])

    def test_uncertain_direction_keeps_object_guidance(self) -> None:
        bridge = bridge_with(MockVLMEngine())
        yolo, flow = payloads(direction="UNCERTAIN")
        result = bridge.describe(
            frame=frame(), yolo_payload=yolo, flow_payload=flow
        )
        self.assertIn("에스컬레이터", result["message"])
        self.assertIn("확인하기 어렵습니다", result["message"])

    def test_frame_id_mismatch_drops_direction(self) -> None:
        bridge = bridge_with(MockVLMEngine())
        yolo, _ = payloads(direction="UP", frame_id=42)
        _, flow = payloads(direction="UP", frame_id=77)
        result = bridge.describe(
            frame=frame(), yolo_payload=yolo, flow_payload=flow
        )
        self.assertNotIn("위쪽", result["message"])

    def test_unsafe_vlm_text_is_replaced_before_reaching_user(self) -> None:
        engine = RecordingEngine("바로 탑승하세요. 50센티미터 앞에 계단이 있습니다.")
        bridge = bridge_with(engine)
        yolo, flow = payloads(direction="UP")
        result = bridge.describe(
            frame=frame(), yolo_payload=yolo, flow_payload=flow
        )
        self.assertEqual(len(engine.calls), 1)
        self.assertTrue(result["used_fallback"])
        self.assertEqual(result["message_source"], "fallback")
        self.assertNotIn("탑승", result["message"])
        self.assertNotIn("센티미터", result["message"])
        self.assertEqual(
            result["raw_vlm_message"], "바로 탑승하세요. 50센티미터 앞에 계단이 있습니다."
        )

    def test_yolo_failure_never_calls_the_model(self) -> None:
        engine = RecordingEngine("사용되면 안 되는 문장입니다.")
        bridge = bridge_with(engine)
        yolo, flow = payloads(yolo_ok=False)
        result = bridge.describe(
            frame=frame(), yolo_payload=yolo, flow_payload=flow
        )
        self.assertEqual(engine.calls, [])
        self.assertEqual(result["fallback_reason"], "yolo_unavailable")
        self.assertEqual(result["service_status"], "degraded")

    def test_temporary_image_exists_during_call_and_is_removed_after(self) -> None:
        engine = RecordingEngine("정면에 위쪽으로 운행하는 에스컬레이터가 있습니다.")
        bridge = bridge_with(engine)
        yolo, flow = payloads(direction="UP")
        bridge.describe(frame=frame(), yolo_payload=yolo, flow_payload=flow)
        call = engine.calls[0]
        self.assertTrue(call["exists_during_call"])
        self.assertFalse(Path(call["image_path"]).exists())
        self.assertFalse(Path(call["image_path"]).parent.exists())

    def test_image_sent_to_vlm_matches_the_analyzed_frame(self) -> None:
        saved: dict = {}

        class CopyingEngine(RecordingEngine):
            def generate(self, image_path, prompt, metadata):
                saved["image"] = cv2.imread(str(image_path))
                return super().generate(image_path, prompt, metadata)

        bridge = bridge_with(CopyingEngine("정면에 위쪽으로 운행하는 에스컬레이터가 있습니다."))
        yolo, flow = payloads(direction="UP")
        source = frame()
        bridge.describe(frame=source, yolo_payload=yolo, flow_payload=flow)
        self.assertEqual(saved["image"].shape, source.shape)
        self.assertLess(float(np.abs(saved["image"].astype(int) - source).mean()), 5.0)

    def test_concurrent_request_is_rejected_instead_of_running_twice(self) -> None:
        bridge = bridge_with(MockVLMEngine())
        yolo, flow = payloads(direction="UP")
        bridge.inference_lock.acquire()
        try:
            with self.assertRaises(VLMBusyError):
                bridge.describe(
                    frame=frame(), yolo_payload=yolo, flow_payload=flow
                )
        finally:
            bridge.inference_lock.release()

    def test_lock_serializes_waiting_requests(self) -> None:
        bridge = bridge_with(MockVLMEngine())
        yolo, flow = payloads(direction="UP")
        results: list[dict] = []
        errors: list[BaseException] = []

        def run() -> None:
            try:
                results.append(
                    bridge.describe(
                        frame=frame(),
                        yolo_payload=yolo,
                        flow_payload=flow,
                        wait=True,
                    )
                )
            except BaseException as error:  # noqa: BLE001 - 테스트 스레드 격리
                errors.append(error)

        threads = [threading.Thread(target=run) for _ in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=30)

        self.assertEqual(errors, [])
        self.assertEqual(len(results), 4)
        self.assertEqual(bridge.request_count, 4)
        self.assertFalse(bridge.is_busy)

    def test_missing_frame_is_rejected(self) -> None:
        bridge = bridge_with(MockVLMEngine())
        yolo, flow = payloads()
        with self.assertRaises(ValueError):
            bridge.describe(frame=None, yolo_payload=yolo, flow_payload=flow)

    def test_too_long_query_is_rejected(self) -> None:
        bridge = bridge_with(MockVLMEngine())
        yolo, flow = payloads()
        with self.assertRaises(ValueError):
            bridge.describe(
                frame=frame(),
                yolo_payload=yolo,
                flow_payload=flow,
                user_query="가" * 501,
            )

    def test_no_detection_does_not_invent_objects(self) -> None:
        bridge = bridge_with(MockVLMEngine())
        yolo, flow = payloads(detections=(), direction="UNCERTAIN")
        result = bridge.describe(
            frame=frame(), yolo_payload=yolo, flow_payload=flow
        )
        self.assertEqual(result["status"], "not_found")
        self.assertNotIn("에스컬레이터", result["message"])

    def test_unsupported_class_is_not_guided_as_detected(self) -> None:
        bridge = bridge_with(MockVLMEngine())
        yolo, flow = payloads(
            detections=(escalator_detection(class_name="gate"),),
            direction="UNCERTAIN",
        )
        result = bridge.describe(
            frame=frame(), yolo_payload=yolo, flow_payload=flow
        )
        self.assertNotEqual(result["status"], "detected")

    def test_describe_scene_does_not_need_yolo_or_flow_payload(self) -> None:
        """`describe()`와 달리 elevator_button/escalator 탐지가 없어도 동작한다."""

        engine = RecordingEngine("정면에 사람이 서 있습니다.")
        bridge = bridge_with(engine)
        result = bridge.describe_scene(
            frame=frame(), user_query="주변에 뭐가 있어?"
        )
        self.assertEqual(result["mode"], "scene_description")
        self.assertEqual(result["message"], "정면에 사람이 서 있습니다.")
        self.assertFalse(result["used_fallback"])
        self.assertEqual(len(engine.calls), 1)
        self.assertEqual(engine.calls[0]["metadata"]["mode"], "scene_description")

    def test_describe_scene_still_blocks_unsafe_output_forms(self) -> None:
        engine = RecordingEngine("안전합니다. 바로 지나가세요.")
        bridge = bridge_with(engine)
        result = bridge.describe_scene(frame=frame(), user_query="주변 설명해줘.")
        self.assertTrue(result["used_fallback"])
        self.assertNotIn("안전합니다", result["message"])
        self.assertNotIn("지나가세요", result["message"])

    def test_describe_scene_missing_frame_is_rejected(self) -> None:
        bridge = bridge_with(MockVLMEngine())
        with self.assertRaises(ValueError):
            bridge.describe_scene(frame=None, user_query="주변 설명해줘.")


if __name__ == "__main__":
    unittest.main()
