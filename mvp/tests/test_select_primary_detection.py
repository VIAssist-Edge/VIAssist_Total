"""`EscalatorMVP._select_primary_detection`의 track_id 우선 선택 테스트.

Flow ROI가 프레임마다 다른 물체로 튀지 않도록, 한 번 추적을 시작한
track_id를 그 track이 사라지기 전까지는 계속 우선한다. YOLO·카메라
없이도 검증 가능한 순수 선택 로직만 다룬다.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

MVP_ROOT = Path(__file__).resolve().parents[1]
if str(MVP_ROOT) not in sys.path:
    sys.path.insert(0, str(MVP_ROOT))

import escalator_mvp  # noqa: E402


def detection(**overrides) -> dict:
    value = {
        "class_id": 0,
        "class_name": "escalator",
        "confidence": 0.9,
        "bbox": (100, 100, 200, 200),
        "area": 100 * 100,
        "track_id": 1,
    }
    value.update(overrides)
    return value


def _bare_engine() -> escalator_mvp.EscalatorMVP:
    engine = escalator_mvp.EscalatorMVP.__new__(escalator_mvp.EscalatorMVP)
    engine.primary_track_id = None
    return engine


class SelectPrimaryDetectionTest(unittest.TestCase):
    def test_no_detections_returns_none_and_resets_track(self) -> None:
        engine = _bare_engine()
        engine.primary_track_id = 5
        self.assertIsNone(engine._select_primary_detection([]))
        self.assertIsNone(engine.primary_track_id)

    def test_first_call_picks_largest_area_times_confidence(self) -> None:
        engine = _bare_engine()
        small = detection(track_id=1, area=100 * 100, confidence=0.9)
        large = detection(track_id=2, area=300 * 300, confidence=0.9)

        primary = engine._select_primary_detection([small, large])

        self.assertIs(primary, large)
        self.assertEqual(engine.primary_track_id, 2)

    def test_sticks_to_previous_track_even_if_no_longer_largest(self) -> None:
        """추적하던 물체가 화면에서 작아져도(또는 다른 물체가 커져도) ROI가 안 튄다."""

        engine = _bare_engine()
        tracked = detection(track_id=2, area=300 * 300, confidence=0.9)
        engine._select_primary_detection([detection(track_id=1), tracked])
        self.assertEqual(engine.primary_track_id, 2)

        # 다음 프레임: track_id=1 detection이 훨씬 커졌다(예: 사람이 카메라에
        # 가까워짐). 그래도 계속 track_id=2(에스컬레이터)를 따라가야 한다.
        now_dominant = detection(track_id=1, area=500 * 500, confidence=0.95)
        still_tracked = detection(track_id=2, area=200 * 200, confidence=0.7)

        primary = engine._select_primary_detection([now_dominant, still_tracked])

        self.assertIs(primary, still_tracked)
        self.assertEqual(engine.primary_track_id, 2)

    def test_reselects_when_tracked_object_disappears(self) -> None:
        engine = _bare_engine()
        engine._select_primary_detection(
            [detection(track_id=2, area=300 * 300, confidence=0.9)]
        )
        self.assertEqual(engine.primary_track_id, 2)

        # track_id=2가 이번 프레임에는 없다 — 새로 골라야 한다.
        replacement = detection(track_id=3, area=250 * 250, confidence=0.8)
        primary = engine._select_primary_detection([replacement])

        self.assertIs(primary, replacement)
        self.assertEqual(engine.primary_track_id, 3)

    def test_untracked_detections_fall_back_to_per_frame_max(self) -> None:
        """track_id가 전부 None이면(트래커 미동작) 매 프레임 재선정한다."""

        engine = _bare_engine()
        first = engine._select_primary_detection(
            [detection(track_id=None, area=100 * 100)]
        )
        self.assertIsNone(engine.primary_track_id)

        bigger = detection(track_id=None, area=400 * 400)
        second = engine._select_primary_detection([first, bigger])
        self.assertIs(second, bigger)


if __name__ == "__main__":
    unittest.main()
