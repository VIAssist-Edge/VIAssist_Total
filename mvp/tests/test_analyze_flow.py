"""`EscalatorMVP._analyze_flow`의 Stage 1 ego-motion 보정 통합 테스트.

`background_motion.py` 단위 테스트는 배경 정렬 자체만 검증한다. 여기서는
카메라가 흔들리는(회전+이동) 상황에서도 ROI 안의 진짜 물체 움직임을
`_analyze_flow`가 올바르게 뽑아내는지 합성 이미지로 확인한다. 카메라·YOLO
가중치는 필요 없다.
"""

from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path

import cv2
import numpy as np

MVP_ROOT = Path(__file__).resolve().parents[1]
if str(MVP_ROOT) not in sys.path:
    sys.path.insert(0, str(MVP_ROOT))

import escalator_mvp  # noqa: E402


FLOW_WIDTH = 320
FLOW_HEIGHT = 240
PAD = 60
BBOX = (100, 50, 220, 190)
PATCH_HEIGHT = 140
PATCH_WIDTH = 120


def _bare_engine() -> escalator_mvp.EscalatorMVP:
    engine = escalator_mvp.EscalatorMVP.__new__(escalator_mvp.EscalatorMVP)
    engine.args = types.SimpleNamespace(
        roi_margin=0.08,
        min_motion=0.35,
        max_motion=20.0,
        direction_threshold=0.55,
        raw_direction_confidence=0.55,
    )
    engine.orb_detector = cv2.ORB_create(nfeatures=500)
    return engine


def _escalator_patch() -> np.ndarray:
    """물체(에스컬레이터) 자체를 나타내는 비주기 텍스처.

    주기적인 줄무늬를 쓰면 이동량이 그 주기의 배수일 때 optical flow가
    "움직임 없음"과 구분하지 못하는 aliasing이 생긴다(에스컬레이터 계단도
    실제로 주기적이라 실기 검증에서 유의해야 할 지점이지만, 여기서는 코드
    경로 자체를 검증하는 게 목적이라 비주기 텍스처로 그 문제를 피한다).
    """

    rng = np.random.default_rng(99)
    return rng.integers(40, 220, (PATCH_HEIGHT, PATCH_WIDTH), dtype=np.uint8)


def _build_frames(
    *,
    object_shift_y: int,
    camera_angle_deg: float = 1.2,
    camera_translate: tuple[float, float] = (3.0, -2.0),
    seed: int = 7,
) -> tuple[np.ndarray, np.ndarray]:
    """카메라 흔들림 + ROI 안 독립 물체 움직임이 섞인 prev/curr 프레임을 만든다.

    `object_shift_y`가 음수면 물체가 위로, 양수면 아래로 움직인 것이다.
    """

    rng = np.random.default_rng(seed)
    big_background = rng.integers(
        0, 255, (FLOW_HEIGHT + 2 * PAD, FLOW_WIDTH + 2 * PAD), dtype=np.uint8
    )
    big_background = cv2.GaussianBlur(big_background, (3, 3), 0)

    def crop(source: np.ndarray) -> np.ndarray:
        return source[PAD : PAD + FLOW_HEIGHT, PAD : PAD + FLOW_WIDTH].copy()

    prev_gray = crop(big_background)

    center = (big_background.shape[1] / 2, big_background.shape[0] / 2)
    affine = cv2.getRotationMatrix2D(center, camera_angle_deg, 1.0)
    affine[0, 2] += camera_translate[0]
    affine[1, 2] += camera_translate[1]
    rotated_background = cv2.warpAffine(
        big_background, affine, (big_background.shape[1], big_background.shape[0])
    )
    curr_gray = crop(rotated_background)

    patch = _escalator_patch()
    x1, y1, _x2, _y2 = BBOX
    prev_gray[y1 : y1 + PATCH_HEIGHT, x1 : x1 + PATCH_WIDTH] = patch
    curr_y1 = y1 + object_shift_y
    curr_gray[curr_y1 : curr_y1 + PATCH_HEIGHT, x1 : x1 + PATCH_WIDTH] = patch

    return prev_gray, curr_gray


class AnalyzeFlowEgoMotionTest(unittest.TestCase):
    def test_upward_object_motion_survives_camera_shake(self) -> None:
        """카메라가 흔들려도(회전+이동) 실제 위쪽 움직임을 UP으로 판정해야 한다."""

        prev_gray, curr_gray = _build_frames(object_shift_y=-8)
        engine = _bare_engine()

        analysis = engine._analyze_flow(
            prev_gray=prev_gray,
            curr_gray=curr_gray,
            bbox=BBOX,
            frame_width=FLOW_WIDTH,
            frame_height=FLOW_HEIGHT,
            scale_x=1.0,
            scale_y=1.0,
        )

        self.assertEqual(analysis["motion_source"], "homography")
        self.assertGreaterEqual(analysis["homography_inliers"], 15)
        self.assertEqual(analysis["direction"], "UP")
        self.assertLess(analysis["dy"], 0)

    def test_downward_object_motion_survives_camera_shake(self) -> None:
        prev_gray, curr_gray = _build_frames(object_shift_y=8)
        engine = _bare_engine()

        analysis = engine._analyze_flow(
            prev_gray=prev_gray,
            curr_gray=curr_gray,
            bbox=BBOX,
            frame_width=FLOW_WIDTH,
            frame_height=FLOW_HEIGHT,
            scale_x=1.0,
            scale_y=1.0,
        )

        self.assertEqual(analysis["motion_source"], "homography")
        self.assertEqual(analysis["direction"], "DOWN")
        self.assertGreater(analysis["dy"], 0)

    def test_no_object_motion_leaves_only_small_residual(self) -> None:
        """물체가 안 움직여도 homography 추정 오차만큼의 잔차는 남을 수 있다.

        단일 프레임에서 STATIONARY로 딱 떨어지길 기대하지 않는다 — 실제
        위쪽/아래쪽 8px 이동(다른 테스트에서 magnitude≈6~8)과 비교해, 잔차
        크기가 그보다 뚜렷하게 작은지만 확인한다. 프레임 단위 잔차 노이즈를
        흡수하는 건 `DirectionStabilizer`(다수결)의 역할이라 여기서는
        다루지 않는다.
        """

        prev_gray, curr_gray = _build_frames(object_shift_y=0)
        engine = _bare_engine()

        analysis = engine._analyze_flow(
            prev_gray=prev_gray,
            curr_gray=curr_gray,
            bbox=BBOX,
            frame_width=FLOW_WIDTH,
            frame_height=FLOW_HEIGHT,
            scale_x=1.0,
            scale_y=1.0,
        )

        self.assertEqual(analysis["motion_source"], "homography")
        self.assertLess(analysis["magnitude"], 4.0)

    def test_textureless_background_falls_back_to_median(self) -> None:
        """배경 feature가 없으면 기존 median 기반 방식으로 안전하게 degrade한다."""

        prev_gray = np.full((FLOW_HEIGHT, FLOW_WIDTH), 100, dtype=np.uint8)
        curr_gray = np.full((FLOW_HEIGHT, FLOW_WIDTH), 100, dtype=np.uint8)
        patch = _escalator_patch()
        x1, y1, _x2, _y2 = BBOX
        prev_gray[y1 : y1 + PATCH_HEIGHT, x1 : x1 + PATCH_WIDTH] = patch
        curr_gray[y1 - 8 : y1 - 8 + PATCH_HEIGHT, x1 : x1 + PATCH_WIDTH] = patch
        engine = _bare_engine()

        analysis = engine._analyze_flow(
            prev_gray=prev_gray,
            curr_gray=curr_gray,
            bbox=BBOX,
            frame_width=FLOW_WIDTH,
            frame_height=FLOW_HEIGHT,
            scale_x=1.0,
            scale_y=1.0,
        )

        self.assertEqual(analysis["motion_source"], "background_median_fallback")
        self.assertEqual(analysis["direction"], "UP")


if __name__ == "__main__":
    unittest.main()
