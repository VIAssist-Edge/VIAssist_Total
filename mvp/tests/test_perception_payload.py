"""YOLO/Optical Flow 결과 → 계약 payload 변환 테스트 (카메라·모델 불필요)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

MVP_ROOT = Path(__file__).resolve().parents[1]
if str(MVP_ROOT) not in sys.path:
    sys.path.insert(0, str(MVP_ROOT))

from perception_payload import (  # noqa: E402
    build_detection_payload,
    build_failed_payloads,
    build_flow_payload,
    build_yolo_payload,
    normalize_direction,
)


def mvp_detection(**overrides) -> dict:
    value = {
        "class_id": 3,
        "class_name": "escalator",
        "confidence": 0.93,
        "bbox": (100, 50, 500, 470),
        "area": 400 * 420,
    }
    value.update(overrides)
    return value


def analysis(**overrides) -> dict:
    value = {
        "direction": "UP",
        "dx": 0.1,
        "dy": -1.5,
        "magnitude": 1.5,
        "confidence": 0.82,
        "valid_ratio": 0.4,
        "roi": (120.0, 70.0, 480.0, 450.0),
    }
    value.update(overrides)
    return value


class DirectionNormalizationTest(unittest.TestCase):
    def test_supported_directions(self) -> None:
        self.assertEqual(normalize_direction("UP"), "up")
        self.assertEqual(normalize_direction("DOWN"), "down")
        self.assertEqual(normalize_direction("STATIONARY"), "stopped")

    def test_undecided_directions_become_unknown(self) -> None:
        for value in ("UNCERTAIN", "ANALYZING", "NO_ESCALATOR", "", None, 3):
            with self.subTest(value=value):
                self.assertEqual(normalize_direction(value), "unknown")

    def test_stationary_is_not_confused_with_unknown(self) -> None:
        self.assertNotEqual(
            normalize_direction("STATIONARY"), normalize_direction("UNCERTAIN")
        )


class YoloPayloadTest(unittest.TestCase):
    def payload(self, *detections, **overrides) -> dict:
        return build_yolo_payload(
            frame_id=12,
            timestamp=1750000000.0,
            image_width=640,
            image_height=480,
            detections=detections or (mvp_detection(),),
            **overrides,
        )

    def test_contract_fields_exist(self) -> None:
        payload = self.payload()
        self.assertEqual(
            set(payload),
            {
                "frame_id",
                "timestamp",
                "image_width",
                "image_height",
                "detections",
                "quality",
                "latency_ms",
                "ok",
                "error",
            },
        )
        self.assertTrue(payload["ok"])
        self.assertIsNone(payload["error"])

    def test_detection_uses_contract_keys(self) -> None:
        detection = self.payload()["detections"][0]
        self.assertEqual(detection["cls_name"], "escalator")
        self.assertEqual(detection["conf"], 0.93)
        self.assertEqual(
            (detection["x1"], detection["y1"], detection["x2"], detection["y2"]),
            (100.0, 50.0, 500.0, 470.0),
        )
        self.assertIsNone(detection["distance_m"])

    def test_bbox_is_clamped_into_frame(self) -> None:
        detection = self.payload(mvp_detection(bbox=(-20, -5, 900, 700)))[
            "detections"
        ][0]
        self.assertEqual(detection["x1"], 0.0)
        self.assertEqual(detection["y1"], 0.0)
        self.assertEqual(detection["x2"], 640.0)
        self.assertEqual(detection["y2"], 480.0)

    def test_degenerate_bbox_is_dropped(self) -> None:
        self.assertEqual(self.payload(mvp_detection(bbox=(300, 10, 300, 10)))["detections"], [])

    def test_offscreen_bbox_is_dropped(self) -> None:
        self.assertEqual(self.payload(mvp_detection(bbox=(700, 10, 900, 100)))["detections"], [])

    def test_confidence_is_clamped(self) -> None:
        detection = self.payload(mvp_detection(confidence=1.4))["detections"][0]
        self.assertEqual(detection["conf"], 1.0)

    def test_area_ratio_and_center_offset(self) -> None:
        detection = build_detection_payload(
            mvp_detection(bbox=(0, 0, 320, 240)),
            image_width=640,
            image_height=480,
        )
        self.assertAlmostEqual(detection["area_ratio"], 0.25)
        self.assertAlmostEqual(detection["center_offset"], -0.5)

    def test_failed_payload_has_no_detections(self) -> None:
        payload = self.payload(ok=False, error="camera timeout")
        self.assertFalse(payload["ok"])
        self.assertEqual(payload["detections"], [])
        self.assertEqual(payload["error"], "camera timeout")


class FlowPayloadTest(unittest.TestCase):
    def payload(self, **overrides) -> dict:
        kwargs = {
            "frame_id": 12,
            "image_width": 640,
            "image_height": 480,
            "direction": "UP",
            "analysis": analysis(),
            "stable_frames": 8,
        }
        kwargs.update(overrides)
        return build_flow_payload(**kwargs)

    def test_contract_fields_exist(self) -> None:
        payload = self.payload()
        self.assertEqual(
            set(payload),
            {
                "frame_id",
                "available",
                "direction",
                "confidence",
                "speed",
                "mean_dx",
                "mean_dy",
                "magnitude",
                "valid_ratio",
                "roi",
                "roi_source",
                "stable_frames",
                "image_width",
                "image_height",
                "ok",
                "error",
            },
        )

    def test_up_direction_is_available(self) -> None:
        payload = self.payload()
        self.assertTrue(payload["available"])
        self.assertEqual(payload["direction"], "up")
        self.assertEqual(payload["roi_source"], "detection")

    def test_stationary_is_available_as_stopped(self) -> None:
        payload = self.payload(direction="STATIONARY")
        self.assertTrue(payload["available"])
        self.assertEqual(payload["direction"], "stopped")

    def test_uncertain_is_unavailable(self) -> None:
        payload = self.payload(direction="UNCERTAIN")
        self.assertFalse(payload["available"])
        self.assertEqual(payload["direction"], "unknown")

    def test_missing_analysis_is_unavailable(self) -> None:
        payload = self.payload(analysis=None)
        self.assertFalse(payload["available"])
        self.assertIsNone(payload["roi"])

    def test_not_ok_is_unavailable(self) -> None:
        payload = self.payload(ok=False, error="flow failed")
        self.assertFalse(payload["available"])
        self.assertEqual(payload["direction"], "unknown")

    def test_roi_is_clamped(self) -> None:
        payload = self.payload(analysis=analysis(roi=(-10, -10, 900, 900)))
        self.assertEqual(payload["roi"], [0.0, 0.0, 640.0, 480.0])

    def test_failed_pair_shares_error(self) -> None:
        yolo, flow = build_failed_payloads(
            frame_id=5,
            timestamp=1.0,
            image_width=640,
            image_height=480,
            error={"type": "camera_error", "message": "read failed"},
        )
        self.assertFalse(yolo["ok"])
        self.assertFalse(flow["ok"])
        self.assertEqual(yolo["error"], flow["error"])
        self.assertEqual(yolo["frame_id"], flow["frame_id"])


if __name__ == "__main__":
    unittest.main()
