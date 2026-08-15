"""escalator_mvp의 VLM 요청 경계 테스트 (카메라·YOLO 가중치 불필요).

`EscalatorMVP.__init__`은 카메라와 YOLO 모델을 만들기 때문에, 여기서는
snapshot/요청 경로에 필요한 속성만 직접 채워 순수 로직을 검증한다.
"""

from __future__ import annotations

import sys
import threading
import unittest
from pathlib import Path

import numpy as np

MVP_ROOT = Path(__file__).resolve().parents[1]
if str(MVP_ROOT) not in sys.path:
    sys.path.insert(0, str(MVP_ROOT))

import escalator_mvp  # noqa: E402


def frame() -> np.ndarray:
    return np.full((480, 640, 3), 120, dtype=np.uint8)


def analysis() -> dict:
    return {
        "direction": "UP",
        "dx": 0.1,
        "dy": -1.6,
        "magnitude": 1.6,
        "confidence": 0.8,
        "valid_ratio": 0.4,
        "roi": (170.0, 120.0, 480.0, 380.0),
    }


class RecordingBridge:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def describe(self, *, frame, yolo_payload, flow_payload, user_query=None) -> dict:
        self.calls.append(
            {
                "frame": frame,
                "yolo_payload": yolo_payload,
                "flow_payload": flow_payload,
                "user_query": user_query,
            }
        )
        return {"message": "정면에 위쪽으로 운행하는 에스컬레이터가 있습니다."}


def bare_engine() -> escalator_mvp.EscalatorMVP:
    engine = escalator_mvp.EscalatorMVP.__new__(escalator_mvp.EscalatorMVP)
    engine.snapshot_lock = threading.Lock()
    engine.latest_snapshot = None
    engine.vlm_bridge = None
    engine.frame_index = 42
    engine.stable_frames = 9
    return engine


class SnapshotTest(unittest.TestCase):
    def publish(self, engine, *, direction="UP", detections=None) -> None:
        engine._publish_snapshot(
            frame=frame(),
            detections=detections
            if detections is not None
            else [
                {
                    "class_id": 1,
                    "class_name": "escalator",
                    "confidence": 0.93,
                    "bbox": (150, 100, 500, 400),
                    "area": 350 * 300,
                }
            ],
            analysis=analysis(),
            stable_direction=direction,
            yolo_ms=31.0,
        )

    def test_payloads_share_the_frame_id(self) -> None:
        engine = bare_engine()
        self.publish(engine)
        snapshot = engine.get_snapshot()
        self.assertEqual(snapshot["yolo_payload"]["frame_id"], 42)
        self.assertEqual(snapshot["flow_payload"]["frame_id"], 42)

    def test_payloads_use_contract_keys(self) -> None:
        engine = bare_engine()
        self.publish(engine)
        detection = engine.get_snapshot()["yolo_payload"]["detections"][0]
        self.assertEqual(detection["cls_name"], "escalator")
        self.assertIn("x1", detection)
        self.assertNotIn("bbox", detection)

    def test_flow_direction_is_normalized(self) -> None:
        engine = bare_engine()
        self.publish(engine, direction="DOWN")
        flow = engine.get_snapshot()["flow_payload"]
        self.assertEqual(flow["direction"], "down")
        self.assertTrue(flow["available"])

    def test_undecided_direction_is_unavailable(self) -> None:
        engine = bare_engine()
        self.publish(engine, direction="ANALYZING")
        flow = engine.get_snapshot()["flow_payload"]
        self.assertFalse(flow["available"])
        self.assertEqual(flow["direction"], "unknown")

    def test_snapshot_frame_is_copied_for_the_caller(self) -> None:
        engine = bare_engine()
        self.publish(engine)
        first = engine.get_snapshot()["frame"]
        first[:] = 0
        second = engine.get_snapshot()["frame"]
        self.assertTrue((second == 120).all())

    def test_no_snapshot_before_first_yolo_frame(self) -> None:
        self.assertIsNone(bare_engine().get_snapshot())


class GuidanceRequestTest(unittest.TestCase):
    def test_disabled_vlm_raises(self) -> None:
        engine = bare_engine()
        with self.assertRaises(RuntimeError):
            engine.request_vlm_guidance("주변 상황을 알려줘.")

    def test_missing_snapshot_raises(self) -> None:
        engine = bare_engine()
        engine.vlm_bridge = RecordingBridge()
        with self.assertRaises(RuntimeError):
            engine.request_vlm_guidance(None)

    def test_request_uses_latest_synchronized_payloads(self) -> None:
        engine = bare_engine()
        bridge = RecordingBridge()
        engine.vlm_bridge = bridge
        SnapshotTest().publish(engine)
        result = engine.request_vlm_guidance("에스컬레이터 방향을 알려줘.")
        self.assertEqual(len(bridge.calls), 1)
        call = bridge.calls[0]
        self.assertEqual(call["user_query"], "에스컬레이터 방향을 알려줘.")
        self.assertEqual(
            call["yolo_payload"]["frame_id"], call["flow_payload"]["frame_id"]
        )
        self.assertEqual(call["frame"].shape, (480, 640, 3))
        self.assertIn("에스컬레이터", result["message"])


if __name__ == "__main__":
    unittest.main()
