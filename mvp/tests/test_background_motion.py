"""Stage 1 ego-motion 보정(homography 기반 배경 정렬) 순수 로직 테스트.

카메라, YOLO, VLM 없이 합성 이미지로만 검증한다. 실제 걸음걸이 흔들림
검증은 하드웨어(카메라)가 있어야 하므로 여기서는 다루지 않는다 — 이건
"팀이 준비해야 할 것" 중 실측 검증 프로토콜의 몫이다.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import cv2
import numpy as np

MVP_ROOT = Path(__file__).resolve().parents[1]
if str(MVP_ROOT) not in sys.path:
    sys.path.insert(0, str(MVP_ROOT))

from background_motion import (  # noqa: E402
    build_background_mask,
    derotate_background_motion,
    estimate_background_homography,
    warp_previous_frame,
)


WIDTH = 320
HEIGHT = 240


def textured_frame(seed: int = 0) -> np.ndarray:
    """ORB feature가 충분히 나오는 합성 배경(랜덤 노이즈 텍스처)을 만든다."""

    rng = np.random.default_rng(seed)
    frame = rng.integers(0, 255, (HEIGHT, WIDTH), dtype=np.uint8)
    return cv2.GaussianBlur(frame, (3, 3), 0)


def apply_camera_motion(
    frame: np.ndarray,
    *,
    angle_deg: float = 0.0,
    translate: tuple[float, float] = (0.0, 0.0),
) -> tuple[np.ndarray, np.ndarray]:
    """회전+이동으로 카메라가 움직인 다음 프레임과 진짜 homography를 만든다."""

    center = (WIDTH / 2, HEIGHT / 2)
    affine = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    affine[0, 2] += translate[0]
    affine[1, 2] += translate[1]
    homography = np.vstack([affine, [0.0, 0.0, 1.0]])
    warped = cv2.warpPerspective(frame, homography, (WIDTH, HEIGHT))
    return warped, homography


class BuildBackgroundMaskTest(unittest.TestCase):
    def test_no_roi_is_all_background(self) -> None:
        mask = build_background_mask((HEIGHT, WIDTH), None)
        self.assertTrue((mask == 255).all())

    def test_roi_is_excluded(self) -> None:
        mask = build_background_mask((HEIGHT, WIDTH), (50, 40, 150, 140))
        self.assertTrue((mask[40:140, 50:150] == 0).all())
        self.assertEqual(mask[0, 0], 255)
        self.assertEqual(mask[HEIGHT - 1, WIDTH - 1], 255)

    def test_roi_outside_frame_is_clamped(self) -> None:
        # 음수·초과 좌표가 들어와도 예외 없이 클램프되어야 한다.
        mask = build_background_mask((HEIGHT, WIDTH), (-50, -50, WIDTH + 50, HEIGHT + 50))
        self.assertTrue((mask == 0).all())


class EstimateBackgroundHomographyTest(unittest.TestCase):
    def test_recovers_small_rotation_and_translation(self) -> None:
        prev = textured_frame(seed=1)
        curr, true_homography = apply_camera_motion(
            prev, angle_deg=2.0, translate=(4.0, -3.0)
        )
        mask = build_background_mask((HEIGHT, WIDTH), None)

        homography, inlier_count = estimate_background_homography(prev, curr, mask)

        self.assertIsNotNone(homography)
        self.assertGreaterEqual(inlier_count, 15)

        # 추정된 homography로 prev를 warp하면 curr와 거의 일치해야 한다.
        aligned = warp_previous_frame(prev, homography, (WIDTH, HEIGHT))
        raw_diff = np.abs(prev.astype(int) - curr.astype(int)).mean()
        aligned_diff = np.abs(aligned.astype(int) - curr.astype(int)).mean()
        self.assertLess(aligned_diff, raw_diff * 0.5)

    def test_textureless_background_returns_none(self) -> None:
        prev = np.full((HEIGHT, WIDTH), 128, dtype=np.uint8)
        curr = np.full((HEIGHT, WIDTH), 128, dtype=np.uint8)
        mask = build_background_mask((HEIGHT, WIDTH), None)

        homography, inlier_count = estimate_background_homography(prev, curr, mask)

        self.assertIsNone(homography)
        self.assertEqual(inlier_count, 0)

    def test_masked_out_roi_does_not_contribute_matches(self) -> None:
        """ROI 안 feature만 있고 배경엔 없으면 homography를 못 구해야 한다."""

        prev = np.full((HEIGHT, WIDTH), 128, dtype=np.uint8)
        prev[40:140, 50:150] = textured_frame(seed=2)[40:140, 50:150]
        curr, _ = apply_camera_motion(prev, angle_deg=2.0, translate=(4.0, -3.0))
        # ROI를 배경 마스크에서 제외 → 텍스처가 있던 유일한 영역이 사라짐
        mask = build_background_mask((HEIGHT, WIDTH), (50, 40, 150, 140))

        homography, inlier_count = estimate_background_homography(prev, curr, mask)

        self.assertIsNone(homography)


class DerotateBackgroundMotionTest(unittest.TestCase):
    def test_success_path_returns_homography_source(self) -> None:
        prev = textured_frame(seed=3)
        curr, _ = apply_camera_motion(prev, angle_deg=1.5, translate=(-2.0, 3.0))

        result = derotate_background_motion(prev, curr, exclude_roi=None)

        self.assertEqual(result["motion_source"], "homography")
        self.assertIsNotNone(result["aligned_prev"])
        self.assertIsNotNone(result["homography"])
        self.assertGreaterEqual(result["inlier_count"], 15)

    def test_fallback_path_when_background_has_no_texture(self) -> None:
        prev = np.full((HEIGHT, WIDTH), 100, dtype=np.uint8)
        curr = np.full((HEIGHT, WIDTH), 100, dtype=np.uint8)

        result = derotate_background_motion(prev, curr, exclude_roi=None)

        self.assertEqual(result["motion_source"], "background_median_fallback")
        self.assertIsNone(result["aligned_prev"])
        self.assertIsNone(result["homography"])


if __name__ == "__main__":
    unittest.main()
