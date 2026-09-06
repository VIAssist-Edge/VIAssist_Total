"""배경 feature 매칭 기반 homography로 카메라 자체 움직임(ego-motion)을 추정한다.

Stage 1 (IMU 없는 소프트웨어 전용 개선): 기존 `escalator_mvp.py`는 ROI 밖
배경 optical flow 벡터의 median 하나를 "카메라 움직임"으로 취급해 뺐다.
이 방식은 카메라의 평행이동만 대충 보정하고 회전·기울어짐은 전혀 다루지
못한다(median 벡터 하나로는 회전을 표현할 수 없음).

이 모듈은 그 대신 배경 영역에서 ORB feature를 매칭하고 RANSAC으로
homography(8-DOF: 회전+이동+원근 근사)를 추정한다. 이전 프레임을 이
homography로 현재 프레임 기준에 맞춰 warp하면, 배경이 정렬된 상태에서
ROI 안의 남은 optical flow는 카메라 움직임이 아니라 실제 객체(에스컬레이터
계단)의 움직임만 반영하게 된다.

카메라, YOLO 모델, VLM에 의존하지 않는 순수 OpenCV/NumPy 계층이므로
하드웨어 없이 합성 이미지로 단위 테스트할 수 있다.
"""

from __future__ import annotations

from typing import Any, Optional

import cv2
import numpy as np


# 이 개수 미만으로 매칭되면 homography를 신뢰하지 않고 fallback한다.
MIN_MATCH_COUNT = 15
RANSAC_REPROJ_THRESHOLD = 3.0
LOWE_RATIO = 0.75


def build_background_mask(
    shape: tuple[int, int],
    exclude_roi: Optional[tuple[float, float, float, float]],
) -> np.ndarray:
    """`exclude_roi`(스케일된 좌표) 밖을 255(feature 탐지 대상)로 표시한다.

    `exclude_roi`가 None이면 전체 프레임을 배경으로 취급한다. ROI는 탐지된
    객체(에스컬레이터 등) 자체이므로, 그 위의 feature는 카메라 움직임이
    아니라 객체 자체의 움직임을 담고 있어 homography 추정에서 제외해야
    한다.
    """

    height, width = shape
    mask = np.full((height, width), 255, dtype=np.uint8)
    if exclude_roi is None:
        return mask

    # ROI 하나만 오던 것을 여러 개도 받도록 넓혔다. 탐지된 객체가 많을 때
    # 그 객체들의 feature가 전부 카메라 움직임 추정에 섞이면 안 된다.
    first = exclude_roi[0] if len(exclude_roi) else None
    rois = exclude_roi if isinstance(first, (list, tuple)) else [exclude_roi]

    for roi in rois:
        if roi is None or len(roi) != 4:
            continue
        x1, y1, x2, y2 = roi
        x1 = max(0, min(int(x1), width))
        x2 = max(0, min(int(x2), width))
        y1 = max(0, min(int(y1), height))
        y2 = max(0, min(int(y2), height))
        if x2 > x1 and y2 > y1:
            mask[y1:y2, x1:x2] = 0
    return mask


def estimate_background_homography(
    prev_gray: np.ndarray,
    curr_gray: np.ndarray,
    background_mask: np.ndarray,
    *,
    detector: Optional[cv2.ORB] = None,
    min_matches: int = MIN_MATCH_COUNT,
    ransac_reproj_threshold: float = RANSAC_REPROJ_THRESHOLD,
) -> tuple[Optional[np.ndarray], int]:
    """배경 영역 feature 매칭으로 prev→curr 방향 homography를 추정한다.

    매칭이 부족하거나(배경에 feature가 거의 없는 빈 벽 등) homography가
    RANSAC inlier를 충분히 못 얻으면 `(None, 실제 매칭/inlier 수)`를
    반환한다. 호출자는 이 경우 기존 median 기반 보정으로 fallback해야
    한다 — 이 함수 자체는 fallback을 하지 않는다.
    """

    orb = detector or cv2.ORB_create(nfeatures=500)

    keypoints_prev, descriptors_prev = orb.detectAndCompute(prev_gray, background_mask)
    keypoints_curr, descriptors_curr = orb.detectAndCompute(curr_gray, background_mask)

    if (
        descriptors_prev is None
        or descriptors_curr is None
        or len(keypoints_prev) < min_matches
        or len(keypoints_curr) < min_matches
    ):
        return None, 0

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    raw_matches = matcher.knnMatch(descriptors_prev, descriptors_curr, k=2)

    good_matches = []
    for pair in raw_matches:
        if len(pair) < 2:
            continue
        best, second_best = pair
        if best.distance < LOWE_RATIO * second_best.distance:
            good_matches.append(best)

    if len(good_matches) < min_matches:
        return None, len(good_matches)

    src_points = np.float32(
        [keypoints_prev[match.queryIdx].pt for match in good_matches]
    ).reshape(-1, 1, 2)
    dst_points = np.float32(
        [keypoints_curr[match.trainIdx].pt for match in good_matches]
    ).reshape(-1, 1, 2)

    homography, inlier_mask = cv2.findHomography(
        src_points, dst_points, cv2.RANSAC, ransac_reproj_threshold
    )
    if homography is None or inlier_mask is None:
        return None, len(good_matches)

    inlier_count = int(inlier_mask.sum())
    if inlier_count < min_matches:
        return None, inlier_count

    return homography, inlier_count


def warp_previous_frame(
    prev_gray: np.ndarray,
    homography: np.ndarray,
    output_size: tuple[int, int],
) -> np.ndarray:
    """이전 프레임을 homography로 현재 프레임 기준에 맞춰 정렬(warp)한다."""

    width, height = output_size
    return cv2.warpPerspective(prev_gray, homography, (width, height))


def derotate_background_motion(
    prev_gray: np.ndarray,
    curr_gray: np.ndarray,
    exclude_roi: Optional[tuple[float, float, float, float]],
    *,
    detector: Optional[cv2.ORB] = None,
) -> dict[str, Any]:
    """카메라 움직임을 homography로 보정한 "정렬된 이전 프레임"을 만든다.

    성공하면 `prev_gray`를 현재 프레임 배경에 맞춰 warp한 이미지를
    반환한다. 배경 feature가 부족해 homography를 못 구하면 `aligned_prev`가
    None이므로, 호출자는 기존 median 기반 방식으로 fallback해야 한다.
    """

    height, width = curr_gray.shape[:2]
    background_mask = build_background_mask((height, width), exclude_roi)
    homography, inlier_count = estimate_background_homography(
        prev_gray, curr_gray, background_mask, detector=detector
    )

    if homography is None:
        return {
            "aligned_prev": None,
            "homography": None,
            "inlier_count": inlier_count,
            "motion_source": "background_median_fallback",
        }

    aligned_prev = warp_previous_frame(prev_gray, homography, (width, height))
    return {
        "aligned_prev": aligned_prev,
        "homography": homography,
        "inlier_count": inlier_count,
        "motion_source": "homography",
    }
