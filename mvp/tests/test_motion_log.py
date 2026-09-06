from __future__ import annotations

import csv
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

MVP_ROOT = Path(__file__).resolve().parents[1]
if str(MVP_ROOT) not in sys.path:
    sys.path.insert(0, str(MVP_ROOT))

from motion_log import MOTION_LOG_FIELDS, MotionLogWriter, build_motion_log_row  # noqa: E402


def sample_analysis(**overrides) -> dict:
    value = {
        "motion_source": "homography",
        "homography_inliers": 42,
        "dx": -0.5,
        "dy": -6.1,
        "magnitude": 6.12,
        "valid_ratio": 0.31,
    }
    value.update(overrides)
    return value


class BuildMotionLogRowTest(unittest.TestCase):
    def test_row_has_all_expected_fields(self) -> None:
        row = build_motion_log_row(
            frame_id=10,
            timestamp=1750000000.123456,
            class_name="escalator",
            detection_confidence=0.9123,
            raw_direction="UP",
            stable_direction="UP",
            stable_ratio=0.8333,
            direction_confidence=0.75,
            analysis=sample_analysis(),
            yolo_ms=12.345,
            processing_fps=14.9,
        )
        self.assertEqual(set(row.keys()), set(MOTION_LOG_FIELDS))
        self.assertEqual(row["motion_source"], "homography")
        self.assertEqual(row["homography_inliers"], 42)

    def test_missing_analysis_keys_default_to_zero(self) -> None:
        row = build_motion_log_row(
            frame_id=1,
            timestamp=0.0,
            class_name="escalator",
            detection_confidence=0.5,
            raw_direction="STATIONARY",
            stable_direction="ANALYZING",
            stable_ratio=0.0,
            direction_confidence=0.0,
            analysis={},
            yolo_ms=0.0,
            processing_fps=0.0,
        )
        self.assertEqual(row["dx"], 0.0)
        self.assertEqual(row["motion_source"], "")
        self.assertEqual(row["homography_inliers"], "")


class MotionLogWriterTest(unittest.TestCase):
    def test_creates_file_with_header_and_appends_rows(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "motion.csv"
            writer = MotionLogWriter(path)
            writer.write(
                build_motion_log_row(
                    frame_id=1,
                    timestamp=1.0,
                    class_name="escalator",
                    detection_confidence=0.9,
                    raw_direction="UP",
                    stable_direction="UP",
                    stable_ratio=0.8,
                    direction_confidence=0.7,
                    analysis=sample_analysis(),
                    yolo_ms=10.0,
                    processing_fps=15.0,
                )
            )
            writer.close()

            with path.open(encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["frame_id"], "1")
            self.assertEqual(rows[0]["motion_source"], "homography")

    def test_reopening_existing_file_does_not_duplicate_header(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "motion.csv"
            first = MotionLogWriter(path)
            first.write(
                build_motion_log_row(
                    frame_id=1,
                    timestamp=1.0,
                    class_name="escalator",
                    detection_confidence=0.9,
                    raw_direction="UP",
                    stable_direction="UP",
                    stable_ratio=0.8,
                    direction_confidence=0.7,
                    analysis=sample_analysis(),
                    yolo_ms=10.0,
                    processing_fps=15.0,
                )
            )
            first.close()

            second = MotionLogWriter(path)
            second.write(
                build_motion_log_row(
                    frame_id=2,
                    timestamp=2.0,
                    class_name="escalator",
                    detection_confidence=0.9,
                    raw_direction="DOWN",
                    stable_direction="DOWN",
                    stable_ratio=0.9,
                    direction_confidence=0.8,
                    analysis=sample_analysis(),
                    yolo_ms=11.0,
                    processing_fps=15.0,
                )
            )
            second.close()

            with path.open(encoding="utf-8") as handle:
                lines = handle.read().splitlines()
            header_lines = [line for line in lines if line.startswith("frame_id")]
            self.assertEqual(len(header_lines), 1)
            self.assertEqual(len(lines), 3)  # header + 2 rows


if __name__ == "__main__":
    unittest.main()
