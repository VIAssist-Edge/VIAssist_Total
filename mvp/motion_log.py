"""Stage 1 ego-motion 보정 실측 검증용 CSV 로거.

`escalator_mvp.py --motion-log PATH`로 활성화한다. 프레임마다
`_analyze_flow`가 실제로 도는 경우(=primary detection이 있는 경우)에만 한
줄씩 남겨서, 실기 보행 테스트에서 raw_direction(프레임 단위)과
stable_direction(다수결 이후) 차이, homography/median fallback 비율,
잔차 크기를 나중에 분석할 수 있게 한다.

카메라·YOLO에 의존하지 않는 순수 row 생성 함수와, 파일 I/O만 하는 얇은
writer로 나눠서 row 생성 로직을 하드웨어 없이 테스트할 수 있게 했다.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


MOTION_LOG_FIELDS = [
    "frame_id",
    "timestamp",
    "class_name",
    "detection_confidence",
    "raw_direction",
    "stable_direction",
    "stable_ratio",
    "direction_confidence",
    "motion_source",
    "homography_inliers",
    "dx",
    "dy",
    "magnitude",
    "valid_ratio",
    "yolo_ms",
    "processing_fps",
]


def build_motion_log_row(
    *,
    frame_id: int,
    timestamp: float,
    class_name: str,
    detection_confidence: float,
    raw_direction: str,
    stable_direction: str,
    stable_ratio: float,
    direction_confidence: float,
    analysis: dict[str, Any],
    yolo_ms: float,
    processing_fps: float,
) -> dict[str, Any]:
    """`_analyze_flow`가 반환한 analysis dict를 CSV 한 줄로 정리한다."""

    return {
        "frame_id": frame_id,
        "timestamp": round(timestamp, 3),
        "class_name": class_name,
        "detection_confidence": round(float(detection_confidence), 4),
        "raw_direction": raw_direction,
        "stable_direction": stable_direction,
        "stable_ratio": round(float(stable_ratio), 4),
        "direction_confidence": round(float(direction_confidence), 4),
        "motion_source": analysis.get("motion_source", ""),
        "homography_inliers": analysis.get("homography_inliers", ""),
        "dx": round(float(analysis.get("dx", 0.0)), 4),
        "dy": round(float(analysis.get("dy", 0.0)), 4),
        "magnitude": round(float(analysis.get("magnitude", 0.0)), 4),
        "valid_ratio": round(float(analysis.get("valid_ratio", 0.0)), 4),
        "yolo_ms": round(float(yolo_ms), 3),
        "processing_fps": round(float(processing_fps), 3),
    }


class MotionLogWriter:
    """`--motion-log`로 지정한 CSV 파일에 한 줄씩 이어 쓴다.

    프로세스 시작 시 한 번 연다. 파일이 없으면 헤더를 먼저 쓰고, 있으면
    이어 쓴다(같은 실행을 여러 세션에 나눠 기록할 때 유용).
    """

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not self.path.exists() or self.path.stat().st_size == 0
        self._file = self.path.open("a", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._file, fieldnames=MOTION_LOG_FIELDS)
        if write_header:
            self._writer.writeheader()
            self._file.flush()

    def write(self, row: dict[str, Any]) -> None:
        self._writer.writerow(row)
        self._file.flush()

    def close(self) -> None:
        if not self._file.closed:
            self._file.close()
