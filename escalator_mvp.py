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
import atexit
import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, asdict
from typing import Any, Optional

import cv2
import numpy as np
from flask import Flask, Response, jsonify, render_template_string
from ultralytics import YOLO


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
    processing_fps: float = 0.0
    yolo_inference_ms: float = 0.0
    timestamp: float = 0.0


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
            raise RuntimeError(f"카메라를 열 수 없습니다: index={self.camera_index}")

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
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
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
        self.model = YOLO(args.model)
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
        self.last_detections: list[dict[str, Any]] = []
        self.frame_index = 0
        self.direction_stabilizer = DirectionStabilizer(
            history_size=args.direction_history,
            majority_ratio=args.direction_majority,
        )

    def start(self) -> None:
        LOGGER.info("YOLO classes: %s", self.class_names)
        self.camera.start()
        self.processing_thread = threading.Thread(target=self._processing_loop, daemon=True)
        self.processing_thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        self.camera.stop()
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=2.0)

    def get_status(self) -> dict[str, Any]:
        with self.status_lock:
            return asdict(self.status)

    def get_jpeg(self) -> Optional[bytes]:
        with self.output_lock:
            return self.latest_jpeg

    def _processing_loop(self) -> None:
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
            if self.frame_index % self.args.yolo_every == 0 or not self.last_detections:
                started = time.perf_counter()
                self.last_detections = self._run_yolo(frame)
                yolo_ms = (time.perf_counter() - started) * 1000.0

            gray_small, scale_x, scale_y = self._prepare_flow_frame(frame)
            flow: Optional[np.ndarray] = None
            if self.previous_gray_small is not None:
                flow = cv2.calcOpticalFlowFarneback(
                    self.previous_gray_small,
                    gray_small,
                    None,
                    pyr_scale=0.5,
                    levels=3,
                    winsize=21,
                    iterations=3,
                    poly_n=5,
                    poly_sigma=1.2,
                    flags=0,
                )
            self.previous_gray_small = gray_small

            primary = self._select_primary_detection(self.last_detections)
            raw_direction = "NO_ESCALATOR"
            stable_direction = "NO_ESCALATOR"
            dx = dy = magnitude = direction_confidence = 0.0

            for detection in self.last_detections:
                self._draw_detection(annotated, detection, is_primary=detection is primary)

            if primary is not None and flow is not None:
                analysis = self._analyze_flow(
                    flow=flow,
                    bbox=primary["bbox"],
                    frame_width=frame_w,
                    frame_height=frame_h,
                    scale_x=scale_x,
                    scale_y=scale_y,
                )
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

            now = time.perf_counter()
            elapsed = max(now - previous_tick, 1e-6)
            instant_fps = 1.0 / elapsed
            fps_ema = instant_fps if fps_ema == 0.0 else 0.9 * fps_ema + 0.1 * instant_fps
            previous_tick = now

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
                    processing_fps=float(fps_ema),
                    yolo_inference_ms=float(yolo_ms),
                    timestamp=time.time(),
                )

            encode_ok, encoded = cv2.imencode(
                ".jpg",
                annotated,
                [cv2.IMWRITE_JPEG_QUALITY, self.args.jpeg_quality],
            )
            if encode_ok:
                with self.output_lock:
                    self.latest_jpeg = encoded.tobytes()

    def _run_yolo(self, frame: np.ndarray) -> list[dict[str, Any]]:
        results = self.model.predict(
            source=frame,
            conf=self.args.conf,
            imgsz=self.args.imgsz,
            device=self.args.device,
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

            detections.append(
                {
                    "class_id": class_id,
                    "class_name": str(self.class_names.get(class_id, class_id)),
                    "confidence": confidence,
                    "bbox": (x1, y1, x2, y2),
                    "area": max(0, x2 - x1) * max(0, y2 - y1),
                }
            )
        return detections

    @staticmethod
    def _select_primary_detection(detections: list[dict[str, Any]]) -> Optional[dict[str, Any]]:
        if not detections:
            return None
        # Prefer a large, confident escalator region.
        return max(detections, key=lambda item: item["area"] * item["confidence"])

    def _prepare_flow_frame(self, frame: np.ndarray) -> tuple[np.ndarray, float, float]:
        frame_h, frame_w = frame.shape[:2]
        flow_width = min(self.args.flow_width, frame_w)
        flow_height = max(1, int(frame_h * flow_width / frame_w))
        small = cv2.resize(frame, (flow_width, flow_height), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        return gray, flow_width / frame_w, flow_height / frame_h

    def _analyze_flow(
        self,
        flow: np.ndarray,
        bbox: tuple[int, int, int, int],
        frame_width: int,
        frame_height: int,
        scale_x: float,
        scale_y: float,
    ) -> dict[str, Any]:
        flow_h, flow_w = flow.shape[:2]
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

        roi_flow = flow[sy1:sy2, sx1:sx2]
        if roi_flow.size == 0:
            return self._empty_analysis("UNCERTAIN")

        # Estimate global camera motion using pixels outside the detected bbox.
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

        minimum_vectors = max(30, int(roi_flow.shape[0] * roi_flow.shape[1] * 0.01))
        if len(valid_vectors) < minimum_vectors:
            return {
                **self._empty_analysis("STATIONARY"),
                "roi_small": (sx1, sy1, sx2, sy2),
                "background_dx": background_dx,
                "background_dy": background_dy,
            }

        dx = float(np.median(valid_vectors[:, 0]))
        dy = float(np.median(valid_vectors[:, 1]))
        magnitude = float(np.hypot(dx, dy))

        if abs(dy) < self.args.direction_threshold:
            direction = "STATIONARY"
            confidence = max(0.0, 1.0 - abs(dy) / max(self.args.direction_threshold, 1e-6))
        else:
            direction = "UP" if dy < 0 else "DOWN"
            matching = valid_vectors[:, 1] < 0 if direction == "UP" else valid_vectors[:, 1] > 0
            confidence = float(np.mean(matching))

            # A strong horizontal component means up/down classification is less reliable.
            verticality = abs(dy) / max(abs(dx) + abs(dy), 1e-6)
            confidence *= float(np.clip(verticality * 1.5, 0.0, 1.0))

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
            "roi_small": (sx1, sy1, sx2, sy2),
            "background_dx": background_dx,
            "background_dy": background_dy,
            "samples": sample_vectors,
            "flow_width": flow_w,
            "flow_height": flow_h,
            "frame_width": frame_width,
            "frame_height": frame_height,
        }

    @staticmethod
    def _empty_analysis(direction: str) -> dict[str, Any]:
        return {
            "direction": direction,
            "dx": 0.0,
            "dy": 0.0,
            "magnitude": 0.0,
            "confidence": 0.0,
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
            f"conf={analysis['confidence']:.2f}"
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
  </style>
</head>
<body>
  <main>
    <h1>Escalator YOLO + Optical Flow MVP</h1>
    <img src="/video_feed" alt="camera stream">
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
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    args = parse_args()
    engine = EscalatorMVP(args)
    atexit.register(engine.stop)

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
