from __future__ import annotations

import copy
import unittest

from src.exceptions import (
    FrameSynchronizationError,
    IntegrationError,
    MotionResultValidationError,
    PerceptionUnavailableError,
    YoloResultValidationError,
)
from src.metadata_schema import validate_metadata
from src.perception_adapter import (
    build_perception_metadata,
    is_flow_usable,
    normalize_flow_payload,
)


def perception_detection(**overrides) -> dict:
    """docs/perception_integration_contract.md의 detection 형식."""

    value = {
        "cls_id": 30,
        "cls_name": "escalator",
        "conf": 0.95,
        "x1": 400,
        "y1": 100,
        "x2": 900,
        "y2": 700,
        "track_id": None,
        "clock": 12,
        "distance_m": None,
        "area_ratio": 0.32,
        "center_offset": 0.01,
    }
    value.update(overrides)
    return value


def yolo_payload(*detections: dict, **overrides) -> dict:
    value = {
        "frame_id": 128,
        "timestamp": 1750000000.123,
        "image_width": 1280,
        "image_height": 720,
        "detections": list(detections) if detections else [perception_detection()],
        "quality": {},
        "latency_ms": 32.6,
        "ok": True,
        "error": None,
    }
    value.update(overrides)
    return value


def flow_payload(**overrides) -> dict:
    value = {
        "frame_id": 128,
        "available": True,
        "direction": "up",
        "confidence": 0.83,
        "speed": 1.84,
        "mean_dx": -0.12,
        "mean_dy": -1.84,
        "magnitude": 1.84,
        "valid_ratio": 0.41,
        "roi": [420, 260, 900, 700],
        "roi_source": "detection",
        "stable_frames": 7,
        "image_width": 1280,
        "image_height": 720,
        "ok": True,
        "error": None,
    }
    value.update(overrides)
    return value


class PerceptionYoloPayloadTest(unittest.TestCase):
    def build(self, yolo=None, flow=None, **kwargs):
        return build_perception_metadata(
            yolo_payload=yolo if yolo is not None else yolo_payload(),
            flow_payload=flow,
            **kwargs,
        )

    def test_perception_style_detection_is_converted(self) -> None:
        metadata = self.build()
        detection = metadata["detections"][0]
        self.assertEqual(detection["class_name"], "escalator")
        self.assertEqual(detection["confidence"], 0.95)
        self.assertEqual(detection["bbox"], [400.0, 100.0, 900.0, 700.0])
        self.assertEqual(detection["position"], "front")
        self.assertIs(validate_metadata(metadata), metadata)

    def test_internal_style_detection_still_works(self) -> None:
        metadata = self.build(
            yolo_payload(
                {
                    "class_name": "moving_stairs",
                    "confidence": 0.9,
                    "bbox": [400, 100, 900, 700],
                }
            )
        )
        self.assertEqual(metadata["detections"][0]["class_name"], "escalator")

    def test_both_styles_with_equal_values_are_accepted(self) -> None:
        metadata = self.build(
            yolo_payload(
                perception_detection(
                    class_name="escalator",
                    confidence=0.95,
                    bbox=[400, 100, 900, 700],
                )
            )
        )
        self.assertEqual(metadata["detections"][0]["class_name"], "escalator")

    def test_conflicting_class_aliases_rejected(self) -> None:
        with self.assertRaises(YoloResultValidationError):
            self.build(
                yolo_payload(
                    perception_detection(
                        cls_name="escalator", class_name="elevator_button"
                    )
                )
            )

    def test_class_alias_case_difference_is_not_a_conflict(self) -> None:
        metadata = self.build(
            yolo_payload(
                perception_detection(cls_name="Escalator", class_name="escalator")
            )
        )
        self.assertEqual(metadata["detections"][0]["class_name"], "escalator")

    def test_conflicting_confidence_aliases_rejected(self) -> None:
        with self.assertRaises(YoloResultValidationError):
            self.build(
                yolo_payload(perception_detection(conf=0.95, confidence=0.42))
            )

    def test_conflicting_bbox_and_corners_rejected(self) -> None:
        with self.assertRaises(YoloResultValidationError):
            self.build(
                yolo_payload(
                    perception_detection(bbox=[10, 20, 30, 40]),
                )
            )

    def test_matching_bbox_and_corners_accepted(self) -> None:
        metadata = self.build(
            yolo_payload(perception_detection(bbox=[400, 100, 900, 700]))
        )
        self.assertEqual(metadata["detections"][0]["bbox"], [400.0, 100.0, 900.0, 700.0])

    def test_partial_corner_keys_rejected(self) -> None:
        detection = perception_detection()
        del detection["y2"]
        with self.assertRaises(YoloResultValidationError):
            self.build(yolo_payload(detection))

    def test_missing_bbox_and_corners_rejected(self) -> None:
        with self.assertRaises(YoloResultValidationError):
            self.build(
                yolo_payload({"cls_name": "escalator", "conf": 0.9})
            )

    def test_malformed_corner_value_rejected(self) -> None:
        with self.assertRaises(YoloResultValidationError):
            self.build(yolo_payload(perception_detection(x2="900")))

    def test_reversed_corners_rejected(self) -> None:
        with self.assertRaises(YoloResultValidationError):
            self.build(yolo_payload(perception_detection(x1=900, x2=400)))

    def test_corners_outside_image_rejected(self) -> None:
        with self.assertRaises(YoloResultValidationError):
            self.build(yolo_payload(perception_detection(x2=1281)))

    def test_ok_false_raises_perception_unavailable(self) -> None:
        with self.assertRaises(PerceptionUnavailableError):
            self.build(yolo_payload(ok=False, error="camera timeout"))

    def test_ok_must_be_boolean(self) -> None:
        with self.assertRaises(YoloResultValidationError):
            self.build(yolo_payload(ok="true"))

    def test_empty_detections_are_allowed(self) -> None:
        metadata = self.build(yolo_payload(detections=[]))
        self.assertEqual(metadata["detections"], [])

    def test_detections_must_be_list(self) -> None:
        with self.assertRaises(YoloResultValidationError):
            self.build(yolo_payload(detections=None))

    def test_unsupported_class_is_filtered_out(self) -> None:
        metadata = self.build(yolo_payload(perception_detection(cls_name="gate")))
        self.assertEqual(metadata["detections"], [])

    def test_position_uses_bbox_center(self) -> None:
        metadata = self.build(
            yolo_payload(perception_detection(x1=0, y1=0, x2=200, y2=300))
        )
        self.assertEqual(metadata["detections"][0]["position"], "left")

    def test_quality_is_mapped_to_image_quality(self) -> None:
        metadata = self.build(
            yolo_payload(quality={"is_blurry": True, "blur_score": 8.0})
        )
        self.assertEqual(
            metadata["image_quality"], {"is_blurry": True, "blur_score": 8.0}
        )

    def test_empty_quality_uses_safe_default(self) -> None:
        metadata = self.build()
        self.assertEqual(
            metadata["image_quality"], {"is_blurry": False, "blur_score": 0.0}
        )

    def test_unknown_quality_fields_are_dropped(self) -> None:
        metadata = self.build(
            yolo_payload(
                quality={"is_blurry": False, "blur_score": 3.0, "brightness": 12}
            )
        )
        self.assertEqual(
            metadata["image_quality"], {"is_blurry": False, "blur_score": 3.0}
        )

    def test_explicit_image_quality_argument_wins(self) -> None:
        metadata = self.build(
            yolo_payload(quality={"is_blurry": True}),
            image_quality={"is_blurry": False, "blur_score": 100.0},
        )
        self.assertFalse(metadata["image_quality"]["is_blurry"])

    def test_invalid_quality_type_rejected(self) -> None:
        with self.assertRaises(IntegrationError):
            self.build(yolo_payload(quality={"is_blurry": "no"}))

    def test_payloads_are_not_mutated(self) -> None:
        yolo = yolo_payload()
        flow = flow_payload()
        before = copy.deepcopy((yolo, flow))
        self.build(yolo, flow)
        self.assertEqual((yolo, flow), before)


class PerceptionFlowPayloadTest(unittest.TestCase):
    def motion(self, **overrides) -> dict:
        metadata = build_perception_metadata(
            yolo_payload=yolo_payload(),
            flow_payload=flow_payload(**overrides),
        )
        return metadata["motion"]

    def test_direction_up(self) -> None:
        motion = self.motion(direction="up")
        self.assertTrue(motion["available"])
        self.assertEqual(motion["direction"], "up")
        self.assertEqual(motion["target"], "escalator")
        self.assertEqual(motion["speed"], "1.84")
        self.assertEqual(motion["confidence"], 0.83)

    def test_direction_down(self) -> None:
        self.assertEqual(self.motion(direction="down")["direction"], "down")

    def test_direction_stopped_is_available(self) -> None:
        motion = self.motion(direction="stopped")
        self.assertTrue(motion["available"])
        self.assertEqual(motion["direction"], "stopped")

    def test_direction_unknown_becomes_unavailable(self) -> None:
        motion = self.motion(direction="unknown")
        self.assertFalse(motion["available"])
        self.assertEqual(motion["direction"], "unknown")
        self.assertIsNone(motion["target"])

    def test_unsupported_direction_becomes_unavailable(self) -> None:
        for direction in ("left", "opening", "sideways", ""):
            with self.subTest(direction=direction):
                motion = self.motion(direction=direction)
                self.assertFalse(motion["available"])
                self.assertEqual(motion["direction"], "unknown")

    def test_available_false_ignores_direction(self) -> None:
        motion = self.motion(available=False, direction="up")
        self.assertFalse(motion["available"])
        self.assertEqual(motion["direction"], "unknown")

    def test_ok_false_ignores_direction(self) -> None:
        motion = self.motion(ok=False, error="flow failed")
        self.assertFalse(motion["available"])
        self.assertEqual(motion["direction"], "unknown")

    def test_missing_ok_is_treated_as_unusable(self) -> None:
        payload = flow_payload()
        del payload["ok"]
        self.assertFalse(is_flow_usable(payload))
        self.assertFalse(normalize_flow_payload(payload)["available"])

    def test_non_boolean_available_rejected(self) -> None:
        with self.assertRaises(MotionResultValidationError):
            self.motion(available="true")

    def test_missing_flow_payload_is_unavailable(self) -> None:
        metadata = build_perception_metadata(yolo_payload=yolo_payload())
        self.assertFalse(metadata["motion"]["available"])
        self.assertEqual(metadata["motion"]["direction"], "unknown")

    def test_frame_mismatch_non_strict_drops_motion(self) -> None:
        motion = self.motion(frame_id=999)
        self.assertFalse(motion["available"])
        self.assertEqual(motion["direction"], "unknown")

    def test_frame_mismatch_strict_raises(self) -> None:
        with self.assertRaises(FrameSynchronizationError):
            build_perception_metadata(
                yolo_payload=yolo_payload(),
                flow_payload=flow_payload(frame_id=999),
                strict_frame_sync=True,
            )

    def test_matching_frame_ids_keep_motion(self) -> None:
        self.assertTrue(self.motion(frame_id=128)["available"])

    def test_speed_is_kept_as_string_not_physical_speed(self) -> None:
        self.assertEqual(self.motion(speed=0.5)["speed"], "0.5")

    def test_missing_speed_becomes_unknown(self) -> None:
        self.assertEqual(self.motion(speed=None)["speed"], "unknown")

    def test_roi_is_not_copied_into_metadata(self) -> None:
        self.assertNotIn("roi", self.motion())


if __name__ == "__main__":
    unittest.main()
