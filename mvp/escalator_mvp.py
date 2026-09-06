#!/usr/bin/env python3
"""
YOLO + Optical Flow escalator MVP

Pipeline:
    USB camera -> YOLO escalator detection -> dense optical flow inside bbox
    -> direction stabilization -> MJPEG web stream on port 5000

Run example:
    python3 escalator_mvp.py --model best.pt --camera 0 --port 5000
"""

from __future__ import annotations

import argparse
import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
from flask import Flask, Response, jsonify, render_template_string, request
from ultralytics import YOLO

from background_motion import derotate_background_motion
from motion_log import MotionLogWriter, build_motion_log_row
from class_rules import history_for
from perception_payload import build_flow_payload, build_yolo_payload


LOGGER = logging.getLogger("escalator_mvp")


@dataclass
class RuntimeStatus:
    camera_ok: bool = False
    escalator_detected: bool = False
    class_name: str = ""
    confidence: float = 0.0
    raw_direction: str = "ANALYZING"
    stable_direction: str = "ANALYZING"
    motion_dx: float = 0.0
    motion_dy: float = 0.0
    motion_magnitude: float = 0.0
    direction_confidence: float = 0.0
    motion_source: str = ""
    processing_fps: float = 0.0
    yolo_inference_ms: float = 0.0
    timestamp: float = 0.0
    # 처리 루프에서 잡힌 예외 누적 수와 마지막 메시지(대시보드 확인용).
    error_count: int = 0
    last_error: str = ""


class LatestFrameCamera:
    """Continuously reads the camera and keeps only the newest frame."""

    def __init__(self, camera_index: int, width: int, height: int, fps: int) -> None:
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.fps = fps
        self._frame: Optional[np.ndarray] = None
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._capture: Optional[cv2.VideoCapture] = None

    def start(self) -> None:
        self._open_camera()
        self._thread = threading.Thread(target=self._reader_loop, daemon=True)
        self._thread.start()

    def _open_camera(self) -> None:
        capture = cv2.VideoCapture(self.camera_index)
        if not capture.isOpened():
            # 카메라가 없어도 서버는 떠야 한다(영상 입력 모드·다른 모듈 점검이 가능하도록).
            # 리더 루프는 capture가 None이면 대기만 한다. 상태는 camera_ok=False로 남는다.
            LOGGER.error("카메라를 열 수 없습니다: index=%d — 캡처 없이 기동합니다(영상 입력 모드로 대체 가능)", self.camera_index)
            self._capture = None
            return

        capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        capture.set(cv2.CAP_PROP_FPS, self.fps)
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self._capture = capture

    def _reader_loop(self) -> None:
        while not self._stop_event.is_set():
            if self._capture is None:
                time.sleep(0.1)
                continue

            ok, frame = self._capture.read()
            if not ok or frame is None:
                LOGGER.warning("카메라 프레임 읽기 실패. 재시도합니다.")
                time.sleep(0.05)
                continue

            with self._lock:
                self._frame = frame

    def get_latest(self) -> Optional[np.ndarray]:
        with self._lock:
            if self._frame is None:
                return None
            return self._frame.copy()

    def stop(self) -> None:
        """여러 번 호출해도 안전하도록 종료 절차를 한 번만 수행한다."""

        if self._stop_event.is_set() and self._capture is None:
            return
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        self._thread = None
        if self._capture is not None:
            self._capture.release()
            self._capture = None


class DirectionStabilizer:
    """Converts noisy frame-level directions into a stable state."""

    VALID_DIRECTIONS = {"UP", "DOWN", "STATIONARY"}

    def __init__(self, history_size: int, majority_ratio: float) -> None:
        self.history: deque[str] = deque(maxlen=history_size)
        self.majority_ratio = majority_ratio

    def update(self, direction: str) -> tuple[str, float]:
        if direction in self.VALID_DIRECTIONS:
            self.history.append(direction)
        else:
            # Keep a little history during momentary uncertainty, but do not add noise.
            if not self.history:
                return "ANALYZING", 0.0

        if len(self.history) < max(4, self.history.maxlen // 3):
            return "ANALYZING", 0.0

        counts = {name: self.history.count(name) for name in self.VALID_DIRECTIONS}
        winner = max(counts, key=counts.get)
        ratio = counts[winner] / len(self.history)

        if ratio >= self.majority_ratio:
            return winner, ratio
        return "UNCERTAIN", ratio

    def reset(self) -> None:
        self.history.clear()


class EscalatorMVP:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        # TensorRT 엔진(.engine)은 task를 메타데이터에서 못 읽는 경우가 있어
        # 명시한다. .pt는 기존대로 자동 판별에 맡긴다.
        self.model = (
            YOLO(args.model, task="detect")
            if str(args.model).endswith(".engine")
            else YOLO(args.model)
        )
        self.class_names: dict[int, str] = self.model.names

        self.camera = LatestFrameCamera(
            camera_index=args.camera,
            width=args.width,
            height=args.height,
            fps=args.camera_fps,
        )

        self.status = RuntimeStatus()
        self.status_lock = threading.Lock()
        self.output_lock = threading.Lock()
        self.latest_jpeg: Optional[bytes] = None

        self.stop_event = threading.Event()
        self.processing_thread: Optional[threading.Thread] = None

        self.previous_gray_small: Optional[np.ndarray] = None
        # Stage 1 ego-motion 보정: 배경 feature로 카메라 움직임을 추정한다.
        # 매 프레임 새로 만들지 않고 재사용해 오버헤드를 줄인다.
        self.orb_detector = cv2.ORB_create(nfeatures=500)
        # 실측 검증용 CSV 로거. --motion-log가 없으면 비활성 상태로 둔다.
        self.motion_log_writer: Optional[MotionLogWriter] = (
            MotionLogWriter(args.motion_log) if getattr(args, "motion_log", None) else None
        )
        self.last_detections: list[dict[str, Any]] = []
        self.primary_track_id: Optional[int] = None
        self.frame_index = 0
        self.direction_stabilizer = DirectionStabilizer(
            history_size=args.direction_history,
            majority_ratio=args.direction_majority,
        )

        # YOLO가 실제로 실행된 프레임만 VLM 요청 대상으로 남긴다.
        # 이렇게 해야 VLM에 전달하는 이미지와 detection payload가 같은 프레임이다.
        self.snapshot_lock = threading.Lock()
        self.latest_snapshot: Optional[dict[str, Any]] = None
        # 객체(track_id)마다 별도 안정화기를 둔다. 히스토리 길이는 클래스가
        # 정한다 — 에스컬레이터는 길게(오판 방지), 사람·차량은 짧게(반응 속도).
        self.track_stabilizers: dict[int, DirectionStabilizer] = {}
        self.vlm_bridge: Any = None
        self.voice: Any = None
        self.stable_frames = 0
        self.previous_stable_direction = ""
        self.error_count = 0
        self.last_error = ""

    def start(self) -> None:
        LOGGER.info("YOLO classes: %s", self.class_names)
        if getattr(self.args, "enable_vlm", False):
            self._start_vlm()
        if getattr(self.args, "enable_voice", False):
            self._start_voice()
        self.camera.start()
        self.processing_thread = threading.Thread(target=self._processing_loop, daemon=True)
        self.processing_thread.start()

    def _start_vlm(self) -> None:
        """모델은 프로세스 시작 시 한 번만 로드한다."""

        from vlm_bridge import VLMBridge  # 지연 import: VLM 미사용 실행을 가볍게 유지

        LOGGER.info("VLM 로딩 중: %s", self.args.vlm_config)
        self.vlm_bridge = VLMBridge.from_config(
            self.args.vlm_config,
            timeout_seconds=self.args.vlm_timeout,
        )
        LOGGER.info("VLM 로딩 완료: %s", self.vlm_bridge.model_id)

    def _start_voice(self) -> None:
        """STT/TTS 모델도 프로세스 시작 시 한 번만 로드한다."""

        from voice_endpoints import VoiceServices  # 지연 import: 음성 미사용 실행 지원

        LOGGER.info("음성 모듈 로딩 중 (STT=%s)", self.args.stt_model)
        self.voice = VoiceServices.from_args(self.args)
        LOGGER.info("음성 모듈 로딩 완료")

    def stop(self) -> None:
        """여러 번 호출해도 안전하다."""

        if self.stop_event.is_set():
            return
        self.stop_event.set()
        self.camera.stop()
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=2.0)
        self.processing_thread = None
        if self.motion_log_writer is not None:
            self.motion_log_writer.close()

    def get_snapshot(self) -> Optional[dict[str, Any]]:
        """최신 프레임과 동기화된 Perception payload 묶음을 반환한다."""

        with self.snapshot_lock:
            if self.latest_snapshot is None:
                return None
            snapshot = dict(self.latest_snapshot)
        snapshot["frame"] = snapshot["frame"].copy()
        return snapshot

    def request_vlm_guidance(self, user_query: Optional[str]) -> dict[str, Any]:
        """사용자 요청이 있을 때만 최신 동기화 payload로 VLM을 실행한다."""

        if self.vlm_bridge is None:
            raise RuntimeError("VLM이 비활성화되어 있습니다. --enable-vlm으로 실행하세요.")
        snapshot = self.get_snapshot()
        if snapshot is None:
            raise RuntimeError("아직 분석된 프레임이 없습니다.")
        return self.vlm_bridge.describe(
            frame=snapshot["frame"],
            yolo_payload=snapshot["yolo_payload"],
            flow_payload=snapshot["flow_payload"],
            user_query=user_query,
        )

    def request_vlm_scene_description(self, user_query: Optional[str]) -> dict[str, Any]:
        """YOLO/Optical Flow와 무관하게 현재 프레임 전체를 설명한다.

        elevator_button/escalator 탐지가 없어도 동작하며, 사람·차량 등
        SUPPORTED_TARGETS 밖의 객체도 안내에 포함될 수 있다.
        """

        if self.vlm_bridge is None:
            raise RuntimeError("VLM이 비활성화되어 있습니다. --enable-vlm으로 실행하세요.")
        snapshot = self.get_snapshot()
        if snapshot is None:
            raise RuntimeError("아직 분석된 프레임이 없습니다.")
        return self.vlm_bridge.describe_scene(
            frame=snapshot["frame"],
            user_query=user_query,
        )

    def build_rule_guidance(self, *, max_items: int = 2) -> dict[str, Any]:
        """VLM 없이 탐지 결과만으로 안내 문장을 만든다.

        모델 호출이 없으므로 지연이 사실상 0이다. 학습된 33개 클래스
        안에서는 이 경로만으로 충분하고, VLM은 학습 범위 밖일 때만 쓴다.
        """

        import time as _time

        from class_rules import build_guidance

        started = _time.perf_counter()
        snapshot = self.get_snapshot()
        snapshot_ms = (_time.perf_counter() - started) * 1000

        if snapshot is None:
            return {
                "message": "아직 분석된 프레임이 없습니다.",
                "source": "rule",
                "detections": 0,
                "snapshot_ms": round(snapshot_ms, 3),
                "render_ms": 0.0,
            }

        detections = snapshot["yolo_payload"].get("detections", [])
        started = _time.perf_counter()
        message = build_guidance(detections, max_items=max_items)
        render_ms = (_time.perf_counter() - started) * 1000

        return {
            "message": message,
            "source": "rule",
            "frame_id": snapshot["frame_id"],
            "detections": len(detections),
            "objects": [d.get("class_name") for d in detections][:6],
            "snapshot_ms": round(snapshot_ms, 3),
            "render_ms": round(render_ms, 3),
        }

    def get_status(self) -> dict[str, Any]:
        with self.status_lock:
            return asdict(self.status)

    def get_jpeg(self) -> Optional[bytes]:
        with self.output_lock:
            return self.latest_jpeg

    def _processing_loop(self) -> None:
        """프레임 하나의 예외로 루프 전체가 죽지 않게 한다.

        예외는 traceback과 함께 로그(→ 대시보드 오류 기록)에 남기고, flow
        이전 프레임을 버린 뒤 잠시 쉬었다가 루프를 다시 시작한다.
        """

        while not self.stop_event.is_set():
            try:
                self._processing_loop_inner()
            except Exception as error:  # noqa: BLE001 - 프레임 단위 격리
                self.error_count += 1
                self.last_error = f"{type(error).__name__}: {error}"[:300]
                LOGGER.exception(
                    "처리 루프 예외 (frame=%d, 누적 %d회) — 0.5초 후 재개",
                    self.frame_index, self.error_count,
                )
                with self.status_lock:
                    self.status.error_count = self.error_count
                    self.status.last_error = self.last_error
                self.previous_gray_small = None
                time.sleep(0.5)

    def _processing_loop_inner(self) -> None:
        previous_tick = time.perf_counter()
        fps_ema = 0.0

        while not self.stop_event.is_set():
            frame = self.camera.get_latest()
            if frame is None:
                time.sleep(0.01)
                continue

            self.frame_index += 1
            annotated = frame.copy()
            frame_h, frame_w = frame.shape[:2]

            yolo_ms = 0.0
            yolo_ran = False
            if self.frame_index % self.args.yolo_every == 0 or not self.last_detections:
                started = time.perf_counter()
                self.last_detections = self._run_yolo(frame)
                yolo_ms = (time.perf_counter() - started) * 1000.0
                yolo_ran = True

            gray_small, scale_x, scale_y = self._prepare_flow_frame(frame)
            previous_gray_small = self.previous_gray_small
            self.previous_gray_small = gray_small

            primary = self._select_primary_detection(self.last_detections)
            raw_direction = "NO_ESCALATOR"
            stable_direction = "NO_ESCALATOR"
            analysis: Optional[dict[str, Any]] = None
            dx = dy = magnitude = direction_confidence = stable_ratio = 0.0

            for detection in self.last_detections:
                self._draw_detection(annotated, detection, is_primary=detection is primary)

            # 탐지된 모든 객체의 움직임을 잰다. flow 필드는 프레임당 한 번만
            # 계산하므로, 객체가 늘어나도 추가 비용은 bbox 슬라이싱뿐이다.
            flow_field = None
            if previous_gray_small is not None and self.last_detections:
                flow_field = self._compute_flow_field(
                    prev_gray=previous_gray_small,
                    curr_gray=gray_small,
                    detections=self.last_detections,
                    scale_x=scale_x,
                    scale_y=scale_y,
                )
                for detection in self.last_detections:
                    detection["motion"] = self._analyze_flow(
                        field=flow_field,
                        bbox=detection["bbox"],
                        frame_width=frame_w,
                        frame_height=frame_h,
                        scale_x=scale_x,
                        scale_y=scale_y,
                    )
                    self._stabilize_detection(detection)

                # 사라진 track의 안정화기는 버린다. 계속 쌓이면 누수가 된다.
                alive = {
                    d.get("track_id") for d in self.last_detections
                    if d.get("track_id") is not None
                }
                for stale in set(self.track_stabilizers) - alive:
                    del self.track_stabilizers[stale]

            if primary is not None and flow_field is not None:
                analysis = primary["motion"]
                raw_direction = analysis["direction"]
                dx = analysis["dx"]
                dy = analysis["dy"]
                magnitude = analysis["magnitude"]
                direction_confidence = analysis["confidence"]
                stable_direction, stable_ratio = self.direction_stabilizer.update(raw_direction)
                direction_confidence = min(direction_confidence, stable_ratio) if stable_ratio > 0 else 0.0
                self._draw_motion_result(annotated, primary["bbox"], analysis, stable_direction)
            elif primary is not None:
                raw_direction = "ANALYZING"
                stable_direction = "ANALYZING"
            else:
                self.direction_stabilizer.reset()

            if stable_direction == self.previous_stable_direction:
                self.stable_frames += 1
            else:
                self.previous_stable_direction = stable_direction
                self.stable_frames = 1

            if yolo_ran:
                self._publish_snapshot(
                    frame=frame,
                    detections=self.last_detections,
                    analysis=analysis,
                    stable_direction=stable_direction,
                    yolo_ms=yolo_ms,
                )

            now = time.perf_counter()
            elapsed = max(now - previous_tick, 1e-6)
            instant_fps = 1.0 / elapsed
            fps_ema = instant_fps if fps_ema == 0.0 else 0.9 * fps_ema + 0.1 * instant_fps
            previous_tick = now

            if analysis is not None and self.motion_log_writer is not None:
                self.motion_log_writer.write(
                    build_motion_log_row(
                        frame_id=self.frame_index,
                        timestamp=time.time(),
                        class_name=primary["class_name"] if primary else "",
                        detection_confidence=float(primary["confidence"]) if primary else 0.0,
                        raw_direction=raw_direction,
                        stable_direction=stable_direction,
                        stable_ratio=stable_ratio,
                        direction_confidence=direction_confidence,
                        analysis=analysis,
                        yolo_ms=yolo_ms,
                        processing_fps=fps_ema,
                    )
                )

            self._draw_header(
                annotated,
                stable_direction=stable_direction,
                raw_direction=raw_direction,
                fps=fps_ema,
                yolo_ms=yolo_ms,
            )

            with self.status_lock:
                self.status = RuntimeStatus(
                    camera_ok=True,
                    escalator_detected=primary is not None,
                    class_name=primary["class_name"] if primary else "",
                    confidence=float(primary["confidence"]) if primary else 0.0,
                    raw_direction=raw_direction,
                    stable_direction=stable_direction,
                    motion_dx=float(dx),
                    motion_dy=float(dy),
                    motion_magnitude=float(magnitude),
                    direction_confidence=float(direction_confidence),
                    motion_source=analysis.get("motion_source", "") if analysis else "",
                    processing_fps=float(fps_ema),
                    yolo_inference_ms=float(yolo_ms),
                    timestamp=time.time(),
                    error_count=self.error_count,
                    last_error=self.last_error,
                )

            encode_ok, encoded = cv2.imencode(
                ".jpg",
                annotated,
                [cv2.IMWRITE_JPEG_QUALITY, self.args.jpeg_quality],
            )
            if encode_ok:
                with self.output_lock:
                    self.latest_jpeg = encoded.tobytes()

    def _publish_snapshot(
        self,
        *,
        frame: np.ndarray,
        detections: list[dict[str, Any]],
        analysis: Optional[dict[str, Any]],
        stable_direction: str,
        yolo_ms: float,
    ) -> None:
        """YOLO가 실행된 프레임의 payload 쌍을 계약 형식으로 보관한다."""

        frame_h, frame_w = frame.shape[:2]
        captured_at = time.time()
        snapshot = {
            "frame": frame,
            "frame_id": self.frame_index,
            "captured_at": captured_at,
            "yolo_payload": build_yolo_payload(
                frame_id=self.frame_index,
                timestamp=captured_at,
                image_width=frame_w,
                image_height=frame_h,
                detections=detections,
                latency_ms=yolo_ms,
            ),
            "flow_payload": build_flow_payload(
                frame_id=self.frame_index,
                image_width=frame_w,
                image_height=frame_h,
                direction=stable_direction,
                analysis=analysis,
                stable_frames=self.stable_frames,
            ),
        }
        with self.snapshot_lock:
            self.latest_snapshot = snapshot

    def _run_yolo(self, frame: np.ndarray) -> list[dict[str, Any]]:
        # `.track(persist=True)`로 프레임 간 ByteTrack 상태를 유지해
        # 각 detection에 안정적인 track_id를 붙인다. `--yolo-every`로 일부
        # 프레임을 건너뛰어도 다음 호출 시 이전 tracker 상태를 그대로
        # 이어서 쓴다.
        results = self.model.track(
            source=frame,
            conf=self.args.conf,
            imgsz=self.args.imgsz,
            device=self.args.device,
            persist=True,
            tracker=self.args.tracker,
            verbose=False,
        )

        detections: list[dict[str, Any]] = []
        if not results:
            return detections

        boxes = results[0].boxes
        if boxes is None:
            return detections

        for box in boxes:
            xyxy = box.xyxy[0].detach().cpu().numpy().astype(int)
            class_id = int(box.cls[0].detach().cpu().item())
            confidence = float(box.conf[0].detach().cpu().item())
            x1, y1, x2, y2 = map(int, xyxy.tolist())
            track_id = int(box.id[0].item()) if box.id is not None else None

            detections.append(
                {
                    "class_id": class_id,
                    "class_name": str(self.class_names.get(class_id, class_id)),
                    "confidence": confidence,
                    "bbox": (x1, y1, x2, y2),
                    "area": max(0, x2 - x1) * max(0, y2 - y1),
                    "track_id": track_id,
                }
            )
        return detections

    def _stabilize_detection(self, detection: dict[str, Any]) -> None:
        """이 객체의 raw 방향을 클래스별 히스토리로 안정화해 motion에 넣는다."""

        track_id = detection.get("track_id")
        if track_id is None:
            # 추적 ID가 없으면 프레임 단위 판정을 그대로 쓴다.
            detection["motion"]["stable_direction"] = detection["motion"]["direction"]
            detection["motion"]["stable_ratio"] = 0.0
            detection["motion"]["history_size"] = 0
            return

        class_name = str(detection.get("class_name", ""))
        size = history_for(class_name)

        stabilizer = self.track_stabilizers.get(track_id)
        if stabilizer is None or stabilizer.history.maxlen != size:
            stabilizer = DirectionStabilizer(
                history_size=size,
                majority_ratio=self.args.direction_majority,
            )
            self.track_stabilizers[track_id] = stabilizer

        stable, ratio = stabilizer.update(detection["motion"]["direction"])
        detection["motion"]["stable_direction"] = stable
        detection["motion"]["stable_ratio"] = ratio
        detection["motion"]["history_size"] = size

    def _select_primary_detection(
        self, detections: list[dict[str, Any]]
    ) -> Optional[dict[str, Any]]:
        """Flow ROI가 프레임마다 다른 물체로 튀지 않도록 track_id를 우선한다.

        `docs/perception_integration_contract.md` §14의 우선순위(track_id →
        detection ID → ROI overlap → confidence)를 따른다. 이 MVP는
        detection ID·ROI overlap 매칭까지는 구현하지 않고, 이전에 추적하던
        track_id가 이번 프레임에도 있으면 그것을 그대로 쓰고, 없으면(추적을
        놓쳤거나 첫 프레임이면) 기존처럼 area*confidence가 가장 큰 detection을
        새로 고른다.
        """

        if not detections:
            self.primary_track_id = None
            return None

        if self.primary_track_id is not None:
            for detection in detections:
                if detection.get("track_id") == self.primary_track_id:
                    return detection
            # 추적하던 track_id가 이번 프레임에 없다 — 놓친 것이므로 새로 고른다.

        primary = max(detections, key=lambda item: item["area"] * item["confidence"])
        self.primary_track_id = primary.get("track_id")
        return primary

    def _prepare_flow_frame(self, frame: np.ndarray) -> tuple[np.ndarray, float, float]:
        frame_h, frame_w = frame.shape[:2]
        flow_width = min(self.args.flow_width, frame_w)
        flow_height = max(1, int(frame_h * flow_width / frame_w))
        small = cv2.resize(frame, (flow_width, flow_height), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        return gray, flow_width / frame_w, flow_height / frame_h

    def _compute_flow_field(
        self,
        prev_gray: np.ndarray,
        curr_gray: np.ndarray,
        detections: list[dict[str, Any]],
        scale_x: float,
        scale_y: float,
    ) -> dict[str, Any]:
        """프레임당 한 번만 부르는 무거운 부분.

        ego-motion 보정과 Farneback을 프레임 전체에 대해 수행한다. 객체별
        측정은 이 결과를 잘라 쓰기만 하므로 탐지 개수와 무관하게 비용이
        일정하다.
        """

        flow_h, flow_w = curr_gray.shape[:2]

        # 탐지된 객체 전부를 배경에서 제외한다. 예전에는 primary 하나만
        # 제외했는데, 그러면 나머지 움직이는 물체가 카메라 움직임으로
        # 잘못 추정된다.
        exclude_rois = [
            (
                float(np.clip(det["bbox"][0] * scale_x, 0, flow_w)),
                float(np.clip(det["bbox"][1] * scale_y, 0, flow_h)),
                float(np.clip(det["bbox"][2] * scale_x, 0, flow_w)),
                float(np.clip(det["bbox"][3] * scale_y, 0, flow_h)),
            )
            for det in detections
        ]

        derotation = derotate_background_motion(
            prev_gray,
            curr_gray,
            exclude_roi=exclude_rois or None,
            detector=self.orb_detector,
        )
        flow_source_prev = (
            derotation["aligned_prev"]
            if derotation["aligned_prev"] is not None
            else prev_gray
        )

        flow = cv2.calcOpticalFlowFarneback(
            flow_source_prev,
            curr_gray,
            None,
            pyr_scale=0.5,
            levels=3,
            winsize=21,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0,
        )
        return {
            "flow": flow,
            "motion_source": derotation["motion_source"],
            "homography_inliers": derotation["inlier_count"],
            "flow_width": flow_w,
            "flow_height": flow_h,
        }

    def _analyze_flow(
        self,
        field: dict[str, Any],
        bbox: tuple[int, int, int, int],
        frame_width: int,
        frame_height: int,
        scale_x: float,
        scale_y: float,
    ) -> dict[str, Any]:
        """미리 계산된 flow 필드에서 bbox 하나의 움직임만 뽑는다."""

        flow = field["flow"]
        motion_source = field["motion_source"]
        derotation = {"inlier_count": field["homography_inliers"]}
        flow_h = field["flow_height"]
        flow_w = field["flow_width"]
        x1, y1, x2, y2 = bbox

        # Shrink bbox slightly to reduce edge/background contamination.
        margin_x = int((x2 - x1) * self.args.roi_margin)
        margin_y = int((y2 - y1) * self.args.roi_margin)
        x1 += margin_x
        x2 -= margin_x
        y1 += margin_y
        y2 -= margin_y

        sx1 = int(np.clip(x1 * scale_x, 0, flow_w - 1))
        sx2 = int(np.clip(x2 * scale_x, sx1 + 1, flow_w))
        sy1 = int(np.clip(y1 * scale_y, 0, flow_h - 1))
        sy2 = int(np.clip(y2 * scale_y, sy1 + 1, flow_h))

        roi_full = (float(x1), float(y1), float(x2), float(y2))

        roi_flow = flow[sy1:sy2, sx1:sx2]
        if roi_flow.size == 0:
            return {**self._empty_analysis("UNCERTAIN"), "motion_source": motion_source}

        if motion_source == "homography":
            # 배경은 이미 homography로 정렬됐으므로 추가 median 보정은 하지
            # 않는다 — 남은 flow는 회전까지 보정된 순수 물체 움직임이다.
            corrected = roi_flow
            background_dx = 0.0
            background_dy = 0.0
        else:
            # Fallback: 배경 feature가 부족할 때만 쓰는 기존 median 보정.
            # 평행이동만 대충 보정하고 회전은 반영하지 못한다.
            background_mask = np.ones((flow_h, flow_w), dtype=bool)
            background_mask[sy1:sy2, sx1:sx2] = False
            background_vectors = flow[background_mask]

            background_dx = float(np.median(background_vectors[:, 0])) if background_vectors.size else 0.0
            background_dy = float(np.median(background_vectors[:, 1])) if background_vectors.size else 0.0

            corrected = roi_flow.copy()
            corrected[..., 0] -= background_dx
            corrected[..., 1] -= background_dy

        magnitude_map = np.linalg.norm(corrected, axis=2)
        valid_mask = (
            (magnitude_map >= self.args.min_motion)
            & (magnitude_map <= self.args.max_motion)
        )
        valid_vectors = corrected[valid_mask]

        roi_pixels = max(1, roi_flow.shape[0] * roi_flow.shape[1])
        valid_ratio = float(len(valid_vectors)) / roi_pixels

        minimum_vectors = max(30, int(roi_flow.shape[0] * roi_flow.shape[1] * 0.01))
        if len(valid_vectors) < minimum_vectors:
            return {
                **self._empty_analysis("STATIONARY"),
                "roi_small": (sx1, sy1, sx2, sy2),
                "roi": roi_full,
                "valid_ratio": valid_ratio,
                "background_dx": background_dx,
                "background_dy": background_dy,
                "motion_source": motion_source,
                "homography_inliers": derotation["inlier_count"],
            }

        dx = float(np.median(valid_vectors[:, 0]))
        dy = float(np.median(valid_vectors[:, 1]))
        magnitude = float(np.hypot(dx, dy))

        threshold = self.args.direction_threshold
        # 에스컬레이터만 볼 때는 수직 성분만 봤다. 사람·차량까지 대상이
        # 되면서 좌우 이동도 판정해야 하므로 우세 축을 먼저 고른다.
        vertical_dominant = abs(dy) >= abs(dx)

        if max(abs(dx), abs(dy)) < threshold:
            direction = "STATIONARY"
            confidence = max(
                0.0, 1.0 - max(abs(dx), abs(dy)) / max(threshold, 1e-6)
            )
        elif vertical_dominant:
            direction = "UP" if dy < 0 else "DOWN"
            matching = valid_vectors[:, 1] < 0 if direction == "UP" else valid_vectors[:, 1] > 0
            confidence = float(np.mean(matching))

            # 수평 성분이 크면 상하 판정의 신뢰도를 깎는다.
            axis_purity = abs(dy) / max(abs(dx) + abs(dy), 1e-6)
            confidence *= float(np.clip(axis_purity * 1.5, 0.0, 1.0))

            if confidence < self.args.raw_direction_confidence:
                direction = "UNCERTAIN"
        else:
            direction = "LEFT" if dx < 0 else "RIGHT"
            matching = valid_vectors[:, 0] < 0 if direction == "LEFT" else valid_vectors[:, 0] > 0
            confidence = float(np.mean(matching))

            axis_purity = abs(dx) / max(abs(dx) + abs(dy), 1e-6)
            confidence *= float(np.clip(axis_purity * 1.5, 0.0, 1.0))

            if confidence < self.args.raw_direction_confidence:
                direction = "UNCERTAIN"

        # Representative points for optional visualization.
        sample_vectors: list[tuple[int, int, float, float]] = []
        step = max(8, min(roi_flow.shape[:2]) // 8)
        for local_y in range(step // 2, roi_flow.shape[0], step):
            for local_x in range(step // 2, roi_flow.shape[1], step):
                vx, vy = corrected[local_y, local_x]
                mag = float(np.hypot(vx, vy))
                if self.args.min_motion <= mag <= self.args.max_motion:
                    sample_vectors.append((sx1 + local_x, sy1 + local_y, float(vx), float(vy)))

        return {
            "direction": direction,
            "dx": dx,
            "dy": dy,
            "magnitude": magnitude,
            "confidence": confidence,
            "valid_ratio": valid_ratio,
            "roi": roi_full,
            "roi_small": (sx1, sy1, sx2, sy2),
            "background_dx": background_dx,
            "background_dy": background_dy,
            "samples": sample_vectors,
            "flow_width": flow_w,
            "flow_height": flow_h,
            "frame_width": frame_width,
            "frame_height": frame_height,
            "motion_source": motion_source,
            "homography_inliers": derotation["inlier_count"],
        }

    @staticmethod
    def _empty_analysis(direction: str) -> dict[str, Any]:
        return {
            "direction": direction,
            "dx": 0.0,
            "dy": 0.0,
            "magnitude": 0.0,
            "confidence": 0.0,
            "valid_ratio": 0.0,
            "roi": None,
            "samples": [],
        }

    @staticmethod
    def _draw_detection(frame: np.ndarray, detection: dict[str, Any], is_primary: bool) -> None:
        x1, y1, x2, y2 = detection["bbox"]
        thickness = 3 if is_primary else 1
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 220, 0), thickness)
        label = f'{detection["class_name"]} {detection["confidence"]:.2f}'
        cv2.putText(
            frame,
            label,
            (x1, max(25, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 220, 0),
            2,
            cv2.LINE_AA,
        )

    def _draw_motion_result(
        self,
        frame: np.ndarray,
        bbox: tuple[int, int, int, int],
        analysis: dict[str, Any],
        stable_direction: str,
    ) -> None:
        x1, y1, x2, y2 = bbox
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2

        dx = analysis["dx"]
        dy = analysis["dy"]
        arrow_scale = self.args.arrow_scale
        end_x = int(center_x + dx * arrow_scale)
        end_y = int(center_y + dy * arrow_scale)

        if analysis["magnitude"] > 0:
            cv2.arrowedLine(
                frame,
                (center_x, center_y),
                (end_x, end_y),
                (0, 165, 255),
                4,
                tipLength=0.25,
            )

        text = (
            f"Motion: {stable_direction} | "
            f"raw={analysis['direction']} | "
            f"dx={dx:.2f}, dy={dy:.2f} | "
            f"conf={analysis['confidence']:.2f} | "
            f"ego={analysis.get('motion_source', 'n/a')}"
        )
        cv2.putText(
            frame,
            text,
            (20, frame.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 165, 255),
            2,
            cv2.LINE_AA,
        )

    @staticmethod
    def _draw_header(
        frame: np.ndarray,
        stable_direction: str,
        raw_direction: str,
        fps: float,
        yolo_ms: float,
    ) -> None:
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 58), (20, 20, 20), -1)
        cv2.putText(
            frame,
            f"Escalator: {stable_direction}   Raw: {raw_direction}",
            (15, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            f"Processing FPS: {fps:.1f}   Last YOLO: {yolo_ms:.1f} ms",
            (15, 49),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (210, 210, 210),
            1,
            cv2.LINE_AA,
        )


HTML_PAGE = """
<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Escalator YOLO + Optical Flow MVP</title>
  <style>
    body { margin: 0; background: #111; color: #eee; font-family: Arial, sans-serif; }
    main { width: min(1100px, 96vw); margin: 24px auto; }
    h1 { font-size: 24px; }
    img { width: 100%; border: 1px solid #444; border-radius: 8px; background: #000; }
    pre { background: #1d1d1d; padding: 14px; border-radius: 8px; overflow-x: auto; }
    .guidance { background: #14261c; border: 1px solid #2f6b47; padding: 16px;
                border-radius: 8px; font-size: 20px; line-height: 1.5; }
    button { padding: 10px 18px; border: 0; border-radius: 8px; background: #36a269;
             color: #fff; font-weight: bold; font-size: 15px; }
    button:disabled { opacity: .5; }
    details { margin-top: 12px; color: #b8bfca; }
  </style>
</head>
<body>
  <main>
    <h1>Escalator YOLO + Optical Flow MVP</h1>
    <img src="/video_feed" alt="camera stream">

    <h2>안내 요청 (VLM)</h2>
    <p>
      <button id="describe" type="button">현재 상황 안내 요청</button>
      <button id="describe-scene" type="button">주변 장면 설명 요청</button>
      <span id="vlm-meta"></span>
    </p>
    <div class="guidance" id="guidance">아직 요청하지 않았습니다.</div>
    <details>
      <summary>디버깅 정보 (사용자 안내에 사용 금지)</summary>
      <pre id="vlm-debug">-</pre>
    </details>

    <h2>음성 (STT / TTS)</h2>
    <p>
      <button id="voice-ask" type="button">말로 물어보기</button>
      <span id="voice-meta"></span>
    </p>
    <div class="guidance" id="voice-heard">아직 듣지 않았습니다.</div>

    <h2>Current status</h2>
    <pre id="status">loading...</pre>
  </main>
  <script>
    async function refreshStatus() {
      try {
        const response = await fetch('/status');
        const data = await response.json();
        document.getElementById('status').textContent = JSON.stringify(data, null, 2);
      } catch (error) {
        document.getElementById('status').textContent = String(error);
      }
    }
    refreshStatus();
    setInterval(refreshStatus, 500);

    const describeButton = document.getElementById('describe');
    describeButton.addEventListener('click', async () => {
      describeButton.disabled = true;
      document.getElementById('guidance').textContent = '분석 중...';
      document.getElementById('vlm-meta').textContent = '';
      try {
        const response = await fetch('/vlm/describe', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({})
        });
        const data = await response.json();
        if (!response.ok) throw new Error(data.error || 'VLM 요청 실패');
        // 사용자 안내에는 검증이 끝난 message만 사용한다.
        document.getElementById('guidance').textContent = data.message;
        document.getElementById('vlm-meta').textContent =
          `${data.status} | ${data.service_status} | ${data.request_latency_ms} ms`;
        document.getElementById('vlm-debug').textContent = JSON.stringify(data, null, 2);
      } catch (error) {
        document.getElementById('guidance').textContent = `오류: ${error.message}`;
      } finally { describeButton.disabled = false; }
    });

    const describeSceneButton = document.getElementById('describe-scene');
    describeSceneButton.addEventListener('click', async () => {
      describeSceneButton.disabled = true;
      document.getElementById('guidance').textContent = '분석 중...';
      document.getElementById('vlm-meta').textContent = '';
      try {
        const response = await fetch('/vlm/describe_scene', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({})
        });
        const data = await response.json();
        if (!response.ok) throw new Error(data.error || 'VLM 요청 실패');
        // 사용자 안내에는 검증이 끝난 message만 사용한다.
        document.getElementById('guidance').textContent = data.message;
        document.getElementById('vlm-meta').textContent =
          `${data.mode} | ${data.service_status} | ${data.request_latency_ms} ms`;
        document.getElementById('vlm-debug').textContent = JSON.stringify(data, null, 2);
      } catch (error) {
        document.getElementById('guidance').textContent = `오류: ${error.message}`;
      } finally { describeSceneButton.disabled = false; }
    });

    const voiceAskButton = document.getElementById('voice-ask');
    voiceAskButton.addEventListener('click', async () => {
      voiceAskButton.disabled = true;
      document.getElementById('voice-heard').textContent = '듣는 중...';
      document.getElementById('voice-meta').textContent = '';
      try {
        const response = await fetch('/voice/ask', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ mode: 'auto' }),  // 라우터: 규칙/VLM 상태/VLM 장면/고정 응답
        });
        const data = await response.json();
        if (!response.ok) throw new Error(data.error || '음성 요청 실패');
        const heard = data.heard || {};
        document.getElementById('voice-heard').textContent =
          heard.text ? `들은 말: ${heard.text}` : '말소리를 감지하지 못했습니다.';
        document.getElementById('guidance').textContent = data.message || '-';
        const spoken = data.spoken || {};
        document.getElementById('voice-meta').textContent =
          `STT ${heard.transcribe_ms || 0}ms | 재생 ${spoken.spoken ? '성공' : '실패'}`;
        document.getElementById('vlm-debug').textContent = JSON.stringify(data, null, 2);
      } catch (error) {
        document.getElementById('voice-heard').textContent = `오류: ${error.message}`;
      } finally { voiceAskButton.disabled = false; }
    });
  </script>
</body>
</html>
"""


def create_app(engine: EscalatorMVP) -> Flask:
    app = Flask(__name__)

    @app.get("/")
    def index() -> str:
        return render_template_string(HTML_PAGE)

    @app.get("/status")
    def status() -> Response:
        return jsonify(engine.get_status())

    @app.get("/perception_payload")
    def perception_payload() -> Response:
        snapshot = engine.get_snapshot()
        if snapshot is None:
            return jsonify(error="아직 분석된 프레임이 없습니다."), 503
        return jsonify(
            frame_id=snapshot["frame_id"],
            yolo_payload=snapshot["yolo_payload"],
            flow_payload=snapshot["flow_payload"],
        )

    @app.post("/guide")
    def guide() -> Response:
        """규칙 기반 즉시 안내. VLM을 부르지 않는다."""

        data = request.get_json(silent=True) or {}
        result = engine.build_rule_guidance(
            max_items=int(data.get("max_items", 2))
        )
        return jsonify(result)

    @app.post("/vlm/describe")
    def vlm_describe() -> Response:
        from vlm_bridge import VLMBusyError  # 지연 import: VLM 미사용 실행 지원

        if engine.vlm_bridge is None:
            return jsonify(error="VLM이 비활성화되어 있습니다."), 503
        data = request.get_json(silent=True) or {}
        query = data.get("query")
        try:
            result = engine.request_vlm_guidance(
                str(query) if query is not None else None
            )
        except VLMBusyError as error:
            return jsonify(error=str(error)), 429
        except ValueError as error:
            return jsonify(error=str(error)), 400
        except RuntimeError as error:
            return jsonify(error=str(error)), 503
        except Exception as error:  # noqa: BLE001 - 요청 단위로 격리한다
            LOGGER.exception("VLM 요청 실패")
            return jsonify(error=f"VLM 요청에 실패했습니다: {error}"), 500
        return jsonify(result)

    @app.post("/vlm/describe_scene")
    def vlm_describe_scene() -> Response:
        from vlm_bridge import VLMBusyError  # 지연 import: VLM 미사용 실행 지원

        if engine.vlm_bridge is None:
            return jsonify(error="VLM이 비활성화되어 있습니다."), 503
        data = request.get_json(silent=True) or {}
        query = data.get("query")
        try:
            result = engine.request_vlm_scene_description(
                str(query) if query is not None else None
            )
        except VLMBusyError as error:
            return jsonify(error=str(error)), 429
        except ValueError as error:
            return jsonify(error=str(error)), 400
        except RuntimeError as error:
            return jsonify(error=str(error)), 503
        except Exception as error:  # noqa: BLE001 - 요청 단위로 격리한다
            LOGGER.exception("VLM 장면 설명 요청 실패")
            return jsonify(error=f"VLM 요청에 실패했습니다: {error}"), 500
        return jsonify(result)

    @app.get("/video_feed")
    def video_feed() -> Response:
        def generate():
            while not engine.stop_event.is_set():
                jpeg = engine.get_jpeg()
                if jpeg is None:
                    time.sleep(0.03)
                    continue
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
                )
                time.sleep(0.01)

        return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")

    # 음성 경로는 --enable-voice로 켰을 때만 등록된다.
    from voice_endpoints import register_voice_routes
    from guidance_router import register_router_routes

    register_voice_routes(app, engine)
    # 규칙/VLM(상태)/VLM(장면) 자동 분기. /guide/auto, /guide/decide
    register_router_routes(app, engine)

    # 모듈별 점검 대시보드(/test). 등록에 실패해도 본 서비스는 그대로 뜬다.
    try:
        from module_test import register_test_routes

        register_test_routes(app, engine)
    except Exception:  # noqa: BLE001
        LOGGER.exception("모듈 점검 대시보드 등록 실패")

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLO + Optical Flow escalator MVP")
    parser.add_argument("--model", default="best.pt", help="YOLO model path")
    parser.add_argument("--camera", type=int, default=0, help="OpenCV camera index")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--camera-fps", type=int, default=30)
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--conf", type=float, default=0.45, help="YOLO confidence threshold")
    parser.add_argument("--imgsz", type=int, default=640, help="YOLO input size")
    parser.add_argument("--device", default="0", help="Ultralytics device, e.g. 0 or cpu")
    parser.add_argument("--yolo-every", type=int, default=3, help="Run YOLO every N frames")
    parser.add_argument(
        "--tracker",
        default="bytetrack.yaml",
        help="ultralytics tracker config (예: bytetrack.yaml, botsort.yaml)",
    )
    parser.add_argument("--flow-width", type=int, default=320, help="Optical flow working width")
    parser.add_argument("--roi-margin", type=float, default=0.08, help="Shrink YOLO bbox ratio")
    parser.add_argument("--min-motion", type=float, default=0.35)
    parser.add_argument("--max-motion", type=float, default=20.0)
    parser.add_argument("--direction-threshold", type=float, default=0.55)
    parser.add_argument("--raw-direction-confidence", type=float, default=0.55)
    parser.add_argument("--direction-history", type=int, default=12)
    parser.add_argument("--direction-majority", type=float, default=0.67)
    parser.add_argument("--arrow-scale", type=float, default=25.0)
    parser.add_argument("--jpeg-quality", type=int, default=80)
    parser.add_argument(
        "--enable-vlm",
        action="store_true",
        help="요청 기반 VLM 안내를 켠다. 시작 시 모델을 한 번 로드한다.",
    )
    parser.add_argument(
        "--vlm-config",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "vlm" / "config" / "jetson.json",
        help="VLM 설정 파일 경로",
    )
    parser.add_argument(
        "--vlm-timeout",
        type=float,
        default=10.0,
        help="VLM soft timeout(초)",
    )
    parser.add_argument(
        "--enable-voice",
        action="store_true",
        help=(
            "음성 입출력(STT/TTS)을 켠다. 시작 시 STT 모델을 한 번 로드하고 "
            "/voice/listen, /voice/say, /voice/ask 경로가 열린다."
        ),
    )
    parser.add_argument(
        "--stt-model",
        default="base",
        help=(
            "faster-whisper 모델 크기. 이 젯슨에서는 base가 1.3초, "
            "small이 4.1초였다(ctranslate2가 CUDA 미지원이라 CPU 고정)."
        ),
    )
    parser.add_argument("--stt-device", default="cpu", help="faster-whisper device")
    parser.add_argument(
        "--stt-compute-type", default="int8", help="faster-whisper compute type"
    )
    parser.add_argument("--stt-language", default="ko", help="STT 인식 언어")
    parser.add_argument(
        "--stt-noise-multiplier",
        type=float,
        default=2.5,
        help=(
            "발화 판정 임계값 = 노이즈 플로어 x 이 배수. 이 마이크는 게인이 "
            "높아 노이즈 플로어가 4500 안팎이므로, 말소리가 인식되지 않으면 "
            "1.5~2.0으로 낮춘다."
        ),
    )
    parser.add_argument("--tts-language", default="ko", help="TTS 합성 언어")
    parser.add_argument(
        "--tts-engine",
        default="melo",
        choices=["melo", "gtts"],
        help=(
            "melo: 온디바이스 MeloTTS 한국어(기본, 네트워크 불필요). "
            "gtts: 클라우드 폴백. melo 기동 실패 시 자동으로 gtts가 된다."
        ),
    )
    parser.add_argument(
        "--tts-melo-device", default="cuda",
        help="MeloTTS 추론 장치. cpu는 RTF 3.0으로 느리므로 cuda를 쓴다.",
    )
    parser.add_argument(
        "--tts-speed", type=float, default=1.0, help="TTS 발화 속도 배율"
    )
    parser.add_argument(
        "--tts-alsa-device",
        default=None,
        help="mpg123에 넘길 ALSA 출력 장치(예: hw:0,3). 미지정 시 기본 장치.",
    )
    parser.add_argument(
        "--motion-log",
        type=Path,
        default=None,
        help=(
            "Stage 1 ego-motion 보정 실측 검증용 CSV 로그 경로. 지정하면 "
            "프레임마다 raw_direction/stable_direction/motion_source 등을 "
            "한 줄씩 기록한다(성능에 큰 영향 없음, 기본은 비활성)."
        ),
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    args = parse_args()
    engine = EscalatorMVP(args)

    # 종료 정리는 finally 한 곳에서만 수행한다(stop()은 재호출해도 안전하다).
    try:
        engine.start()
        app = create_app(engine)
        LOGGER.info("웹 페이지: http://0.0.0.0:%d", args.port)
        app.run(host="0.0.0.0", port=args.port, threaded=True, use_reloader=False)
    except KeyboardInterrupt:
        LOGGER.info("종료 요청을 받았습니다.")
    finally:
        engine.stop()


if __name__ == "__main__":
    main()
