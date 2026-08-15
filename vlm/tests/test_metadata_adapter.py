from __future__ import annotations

import copy
import math
import unittest

from src.exceptions import (
    FrameSynchronizationError,
    IntegrationError,
    MotionResultValidationError,
    YoloResultValidationError,
)
from src.metadata_adapter import DEFAULT_USER_QUERY, build_vlm_metadata
from src.metadata_schema import validate_metadata


def yolo(*detections: dict, frame_id=1, width: int = 300) -> dict:
    return {
        "image_width": width,
        "image_height": 300,
        "frame_id": frame_id,
        "timestamp_ms": 1000.0,
        "detections": list(detections) or [
            {
                "class_name": "elevator_button",
                "confidence": 0.9,
                "bbox": [210, 10, 270, 100],
            }
        ],
    }


def detection(**overrides) -> dict:
    value = {
        "class_name": "elevator_button",
        "confidence": 0.9,
        "bbox": [210, 10, 270, 100],
    }
    value.update(overrides)
    return value


class MetadataAdapterTest(unittest.TestCase):
    def build(self, yolo_value=None, **kwargs):
        return build_vlm_metadata(yolo_result=yolo_value or yolo(), **kwargs)

    def test_elevator_alias(self) -> None:
        result = self.build(yolo(detection(class_name="elevator-button")))
        self.assertEqual(result["detections"][0]["class_name"], "elevator_button")

    def test_escalator_alias(self) -> None:
        result = self.build(yolo(detection(class_name="moving_stairs")))
        self.assertEqual(result["detections"][0]["class_name"], "escalator")

    def test_confidence_descending_stable_sort(self) -> None:
        result = self.build(
            yolo(
                detection(class_name="elevator_button", confidence=0.7),
                detection(class_name="moving_stairs", confidence=0.9),
                detection(class_name="lift_button", confidence=0.9),
            )
        )
        self.assertEqual(
            [item["class_name"] for item in result["detections"]],
            ["escalator", "elevator_button", "elevator_button"],
        )

    def test_unsupported_class_is_filtered(self) -> None:
        result = self.build(yolo(detection(class_name="chair")))
        self.assertEqual(result["detections"], [])

    def test_bbox_position_left(self) -> None:
        result = self.build(yolo(detection(bbox=[0, 0, 60, 50])))
        self.assertEqual(result["detections"][0]["position"], "left")

    def test_bbox_position_front_including_boundary(self) -> None:
        result = self.build(yolo(detection(bbox=[70, 0, 130, 50])))
        self.assertEqual(result["detections"][0]["position"], "front")

    def test_bbox_position_right(self) -> None:
        result = self.build(yolo(detection(bbox=[210, 0, 270, 50])))
        self.assertEqual(result["detections"][0]["position"], "right")

    def test_explicit_position_has_priority(self) -> None:
        result = self.build(
            yolo(detection(bbox=[0, 0, 60, 50], position="right"))
        )
        self.assertEqual(result["detections"][0]["position"], "right")

    def test_position_aliases(self) -> None:
        for alias, expected in (("LEFT", "left"), ("중앙", "front"), ("우측", "right")):
            with self.subTest(alias=alias):
                result = self.build(yolo(detection(position=alias)))
                self.assertEqual(result["detections"][0]["position"], expected)

    def test_invalid_position_rejected(self) -> None:
        with self.assertRaises(YoloResultValidationError):
            self.build(yolo(detection(position="near")))

    def test_confidence_range_rejected(self) -> None:
        with self.assertRaises(YoloResultValidationError):
            self.build(yolo(detection(confidence=1.1)))

    def test_confidence_bool_rejected(self) -> None:
        with self.assertRaises(YoloResultValidationError):
            self.build(yolo(detection(confidence=True)))

    def test_bbox_length_rejected(self) -> None:
        with self.assertRaises(YoloResultValidationError):
            self.build(yolo(detection(bbox=[1, 2, 3])))

    def test_bbox_coordinate_order_rejected(self) -> None:
        with self.assertRaises(YoloResultValidationError):
            self.build(yolo(detection(bbox=[20, 0, 10, 50])))

    def test_bbox_out_of_image_rejected(self) -> None:
        with self.assertRaises(YoloResultValidationError):
            self.build(yolo(detection(bbox=[0, 0, 301, 50])))

    def test_image_width_bool_rejected(self) -> None:
        value = yolo()
        value["image_width"] = True
        with self.assertRaises(YoloResultValidationError):
            self.build(value)

    def test_motion_direction_alias(self) -> None:
        result = self.build(
            motion_result={"available": True, "direction": "ascending", "speed": 0.72}
        )
        self.assertEqual(result["motion"]["direction"], "up")
        self.assertEqual(result["motion"]["speed"], "0.72")

    def test_unavailable_motion_forces_unknown(self) -> None:
        result = self.build(
            motion_result={"available": False, "direction": "down", "speed": None}
        )
        self.assertEqual(result["motion"]["direction"], "unknown")
        self.assertIsNone(result["motion"]["target"])

    def test_missing_motion_uses_default(self) -> None:
        result = self.build()
        self.assertFalse(result["motion"]["available"])
        self.assertEqual(result["motion"]["direction"], "unknown")

    def test_speed_none_becomes_unknown(self) -> None:
        result = self.build(
            motion_result={"available": True, "direction": "up", "speed": None}
        )
        self.assertEqual(result["motion"]["speed"], "unknown")

    def test_motion_nan_and_inf_rejected(self) -> None:
        for value in (math.nan, math.inf):
            with self.subTest(value=value), self.assertRaises(MotionResultValidationError):
                self.build(
                    motion_result={"available": True, "direction": "up", "speed": value}
                )

    def test_matching_frame_ids(self) -> None:
        result = self.build(
            motion_result={"available": True, "direction": "up", "frame_id": 1}
        )
        self.assertTrue(result["motion"]["available"])

    def test_frame_mismatch_non_strict_disables_motion(self) -> None:
        result = self.build(
            motion_result={"available": True, "direction": "up", "frame_id": 2}
        )
        self.assertFalse(result["motion"]["available"])
        self.assertEqual(result["motion"]["direction"], "unknown")

    def test_frame_mismatch_strict_raises(self) -> None:
        with self.assertRaises(FrameSynchronizationError):
            self.build(
                motion_result={"available": True, "direction": "up", "frame_id": 2},
                strict_frame_sync=True,
            )

    def test_default_image_quality(self) -> None:
        result = self.build()
        self.assertEqual(
            result["image_quality"],
            {"is_blurry": False, "blur_score": 0.0},
        )

    def test_blurry_must_be_boolean(self) -> None:
        with self.assertRaises(IntegrationError):
            self.build(image_quality={"is_blurry": 0})

    def test_empty_user_query_uses_default(self) -> None:
        result = self.build(user_query="  ")
        self.assertEqual(result["user_query"], DEFAULT_USER_QUERY)

    def test_final_metadata_passes_existing_validator(self) -> None:
        result = self.build(
            motion_result={"available": True, "direction": "상행", "speed": 0.5},
            image_quality={"is_blurry": False},
        )
        self.assertIs(validate_metadata(result), result)

    def test_input_dicts_are_not_mutated(self) -> None:
        yolo_value = yolo(detection(class_name="elevator-button"))
        motion = {"available": True, "direction": "ascending", "speed": 0.72}
        quality = {"is_blurry": False, "blur_score": 10.0, "source": "camera"}
        before = copy.deepcopy((yolo_value, motion, quality))
        self.build(yolo_value, motion_result=motion, image_quality=quality)
        self.assertEqual((yolo_value, motion, quality), before)


if __name__ == "__main__":
    unittest.main()
