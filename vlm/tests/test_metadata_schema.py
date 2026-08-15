from __future__ import annotations

import copy
import json
import math
import struct
import unittest
from pathlib import Path

from src.metadata_schema import MetadataValidationError, validate_metadata


ROOT = Path(__file__).resolve().parents[1]


def valid_metadata() -> dict:
    return {
        "schema_version": "1.0",
        "image_path": "samples/elevator_button.png",
        "user_query": "엘리베이터 버튼이 어디에 있나요?",
        "image_quality": {"is_blurry": False, "blur_score": 182.4},
        "detections": [
            {
                "class_name": "elevator_button",
                "confidence": 0.93,
                "bbox": [563, 335, 596, 408],
                "position": "right",
            }
        ],
        "motion": {
            "available": False,
            "target": None,
            "direction": "unknown",
            "speed": None,
            "confidence": None,
        },
        "image_width": 692,
        "image_height": 694,
    }


class MetadataSchemaTest(unittest.TestCase):
    def assert_invalid(self, metadata: dict, field: str) -> None:
        with self.assertRaisesRegex(MetadataValidationError, field):
            validate_metadata(metadata)

    def test_valid_metadata(self) -> None:
        metadata = valid_metadata()
        self.assertIs(validate_metadata(metadata), metadata)

    def test_detections_must_be_list(self) -> None:
        metadata = valid_metadata()
        metadata["detections"] = {}
        self.assert_invalid(metadata, "detections")

    def test_detection_item_must_be_dict(self) -> None:
        metadata = valid_metadata()
        metadata["detections"] = ["invalid"]
        self.assert_invalid(metadata, r"detections\[0\]")

    def test_confidence_rejects_string(self) -> None:
        metadata = valid_metadata()
        metadata["detections"][0]["confidence"] = "0.93"
        self.assert_invalid(metadata, "confidence")

    def test_confidence_rejects_nan(self) -> None:
        metadata = valid_metadata()
        metadata["detections"][0]["confidence"] = math.nan
        self.assert_invalid(metadata, "confidence")

    def test_confidence_rejects_value_above_one(self) -> None:
        metadata = valid_metadata()
        metadata["detections"][0]["confidence"] = 1.01
        self.assert_invalid(metadata, "confidence")

    def test_bbox_rejects_wrong_length(self) -> None:
        metadata = valid_metadata()
        metadata["detections"][0]["bbox"] = [1, 2, 3]
        self.assert_invalid(metadata, "bbox")

    def test_bbox_rejects_wrong_coordinate_order(self) -> None:
        metadata = valid_metadata()
        metadata["detections"][0]["bbox"] = [10, 10, 5, 20]
        self.assert_invalid(metadata, "bbox")

    def test_bbox_rejects_out_of_bounds(self) -> None:
        metadata = valid_metadata()
        metadata["detections"][0]["bbox"] = [0, 0, 693, 100]
        self.assert_invalid(metadata, "bbox")

    def test_rejects_invalid_position(self) -> None:
        metadata = valid_metadata()
        metadata["detections"][0]["position"] = "center"
        self.assert_invalid(metadata, "position")

    def test_unavailable_motion_requires_unknown_direction(self) -> None:
        metadata = valid_metadata()
        metadata["motion"]["direction"] = "up"
        self.assert_invalid(metadata, "motion.direction")

    def test_validated_metadata_is_json_serializable(self) -> None:
        serialized = json.dumps(validate_metadata(valid_metadata()))
        self.assertIsInstance(serialized, str)

    def test_all_cleaned_samples_are_valid(self) -> None:
        for json_path in sorted((ROOT / "samples").glob("*.json")):
            with self.subTest(sample=json_path.name):
                metadata = json.loads(json_path.read_text(encoding="utf-8"))
                validate_metadata(metadata)

                image_path = ROOT / metadata["image_path"]
                self.assertTrue(image_path.is_file())
                with image_path.open("rb") as image_file:
                    self.assertEqual(image_file.read(8), b"\x89PNG\r\n\x1a\n")
                    chunk_length = struct.unpack(">I", image_file.read(4))[0]
                    self.assertEqual(image_file.read(4), b"IHDR")
                    self.assertEqual(chunk_length, 13)
                    width, height = struct.unpack(">II", image_file.read(8))
                self.assertEqual(width, metadata["image_width"])
                self.assertEqual(height, metadata["image_height"])

    def test_validator_does_not_mutate_metadata(self) -> None:
        metadata = valid_metadata()
        before = copy.deepcopy(metadata)
        validate_metadata(metadata)
        self.assertEqual(metadata, before)


if __name__ == "__main__":
    unittest.main()
