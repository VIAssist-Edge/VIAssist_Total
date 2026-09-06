#!/usr/bin/env python3
"""마이크 입력을 한국어 텍스트로 바꾸는 STT 경계.

- 모델은 프로세스 시작 시 한 번만 로드하고 재사용한다.
- 녹음은 webrtcvad로 말이 끝나면 스스로 멈춘다. 고정 길이로 기다리지 않는다.
- faster-whisper는 이 젯슨 빌드에서 CUDA를 못 쓴다(ctranslate2가 CUDA 없이
  컴파일됨). base/cpu/int8이 1.3초로 가장 빠르므로 그것을 기본값으로 둔다.
"""

from __future__ import annotations

import audioop
import queue
import threading
import time
import wave
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Optional


# webrtcvad가 허용하는 프레임 길이는 10/20/30ms뿐이다.
FRAME_MS = 30
SAMPLE_RATE = 16000
FRAME_SAMPLES = SAMPLE_RATE * FRAME_MS // 1000
FRAME_BYTES = FRAME_SAMPLES * 2  # int16 mono

# 이 이름이 들어간 입력 장치를 우선 고른다. USB 카메라 내장 마이크.
PREFERRED_INPUT_KEYWORDS = ("USB 2.0 Camera", "USB Audio")

# webrtcvad만으로는 이 마이크의 배경 소음을 말소리로 오인한다. 이 마이크는
# 게인이 높아 무음 구간 RMS가 2000을 넘기도 하므로 고정 임계값은 쓸 수 없다.
# 녹음 시작 시 노이즈 플로어를 재고, 그 배수를 넘는 프레임만 발화로 본다.
NOISE_CALIBRATION_FRAMES = 12  # 30ms * 12 = 360ms
NOISE_MULTIPLIER = 2.5
ABSOLUTE_MIN_RMS = 300  # 완전한 무음 환경에서 하한선

# 말하는 도중 이 간격마다 부분 인식을 한 번씩 돌린다. 너무 짧게 잡으면
# 워커가 계속 밀리고, 너무 길면 부분 결과가 늦게 나온다.
PARTIAL_INTERVAL_MS = 700

# Whisper는 무음·소음 구간에서도 그럴듯한 문장을 만들어낸다. 세그먼트의
# no_speech_prob이 이 값을 넘으면 환각으로 보고 버린다.
MAX_NO_SPEECH_PROB = 0.6


class STTUnavailableError(RuntimeError):
    """마이크나 모델을 쓸 수 없을 때 발생한다."""


class STTBusyError(RuntimeError):
    """이미 다른 녹음이 진행 중일 때 발생한다."""


def find_input_device() -> Optional[int]:
    """마이크로 쓸 입력 장치 인덱스를 찾는다. 못 찾으면 None(기본 장치)."""

    try:
        import sounddevice as sd
    except Exception:  # noqa: BLE001 - 오디오 스택이 없을 수도 있다
        return None

    try:
        devices = sd.query_devices()
    except Exception:  # noqa: BLE001
        return None

    fallback: Optional[int] = None
    for index, device in enumerate(devices):
        if device.get("max_input_channels", 0) < 1:
            continue
        name = str(device.get("name", ""))
        if any(keyword in name for keyword in PREFERRED_INPUT_KEYWORDS):
            return index
        if fallback is None:
            fallback = index
    return fallback


def _native_sample_rate(device: Optional[int]) -> int:
    """장치가 실제로 지원하는 샘플레이트를 확인한다.

    이 USB 카메라 마이크는 44.1kHz 전용이다. 16kHz로 직접 열면 ALSA
    리샘플러가 깨져 스트림이 첫 프레임만 잡고 그 뒤로는 무음에 가까운
    고정값만 들어온다(arecord로 44100Hz 네이티브로 열면 정상 녹음되는
    것으로 확인함 — 마이크/USB 자체는 정상). 그래서 캡처는 항상 장치
    네이티브 레이트로 열고, 그 결과를 여기서 직접 16kHz로
    다운샘플링한다.
    """

    if device is None:
        return SAMPLE_RATE
    try:
        import sounddevice as sd

        rate = sd.query_devices(device).get("default_samplerate")
        return int(round(rate)) if rate else SAMPLE_RATE
    except Exception:  # noqa: BLE001
        return SAMPLE_RATE


class STTService:
    """마이크 녹음과 음성 인식을 담당하는 단일 인스턴스 경계."""

    def __init__(
        self,
        model_size: str = "base",
        *,
        device: str = "cpu",
        compute_type: str = "int8",
        language: str = "ko",
        input_device: Optional[int] = None,
        beam_size: int = 1,
        noise_multiplier: float = NOISE_MULTIPLIER,
    ) -> None:
        try:
            from faster_whisper import WhisperModel
        except ImportError as error:
            raise STTUnavailableError(
                "faster-whisper를 불러오지 못했습니다."
            ) from error

        self.model_size = model_size
        self.language = language
        self.beam_size = beam_size
        self.noise_multiplier = noise_multiplier
        self.input_device = (
            input_device if input_device is not None else find_input_device()
        )
        self.native_rate = _native_sample_rate(self.input_device)
        self.native_frame_samples = max(1, round(self.native_rate * FRAME_MS / 1000))
        self._lock = threading.Lock()

        # 마지막 녹음에서 잡은 값. 임계값이 왜 그렇게 잡혔는지 확인용.
        self.last_noise_floor: Optional[int] = None
        self.last_speech_threshold: Optional[float] = None

        try:
            self.model = WhisperModel(
                model_size, device=device, compute_type=compute_type
            )
        except Exception as error:  # noqa: BLE001
            raise STTUnavailableError(
                f"STT 모델을 로드하지 못했습니다: {error}"
            ) from error

        print(f"[STT] model: {model_size} ({device}/{compute_type})")
        print(
            f"[STT] input device index: {self.input_device} "
            f"(native {self.native_rate}Hz -> {SAMPLE_RATE}Hz)"
        )

    @property
    def is_busy(self) -> bool:
        return self._lock.locked()

    def record(
        self,
        *,
        max_seconds: float = 8.0,
        silence_ms: int = 800,
        start_timeout_s: float = 4.0,
        aggressiveness: int = 3,
        noise_multiplier: Optional[float] = None,
    ) -> bytes:
        """말이 끝날 때까지 녹음해 PCM(int16 mono 16kHz) 바이트를 돌려준다.

        `start_timeout_s` 안에 아무 말도 들어오지 않으면 빈 바이트를 돌려준다.
        VAD 판정과 RMS 임계값을 모두 넘어야 발화로 인정한다. VAD만 쓰면
        이 마이크의 배경 소음이 발화로 잡힌다.
        """

        try:
            import sounddevice as sd
            import webrtcvad
        except ImportError as error:
            raise STTUnavailableError(
                "sounddevice 또는 webrtcvad를 불러오지 못했습니다."
            ) from error

        if noise_multiplier is None:
            noise_multiplier = self.noise_multiplier

        vad = webrtcvad.Vad(aggressiveness)
        frames: queue.Queue[bytes] = queue.Queue()

        def callback(indata, _frames, _time, status) -> None:
            # status는 오버플로 등 경고. 버리지 않고 그대로 쌓는다.
            frames.put(bytes(indata))

        collected: list[bytes] = []
        noise_samples: list[int] = []
        speech_threshold: Optional[float] = None
        speech_started = False
        silence_run_ms = 0
        started_at = time.monotonic()

        native_frame_bytes = self.native_frame_samples * 2
        resample_state = None
        resample_buffer = bytearray()
        done = False

        with sd.RawInputStream(
            samplerate=self.native_rate,
            blocksize=self.native_frame_samples,
            device=self.input_device,
            dtype="int16",
            channels=1,
            callback=callback,
        ):
            while not done:
                elapsed = time.monotonic() - started_at
                if elapsed > max_seconds:
                    break
                if not speech_started and elapsed > start_timeout_s:
                    break

                try:
                    native_frame = frames.get(timeout=0.5)
                except queue.Empty:
                    continue
                if len(native_frame) != native_frame_bytes:
                    continue

                if self.native_rate != SAMPLE_RATE:
                    converted, resample_state = audioop.ratecv(
                        native_frame, 2, 1, self.native_rate, SAMPLE_RATE, resample_state
                    )
                else:
                    converted = native_frame
                resample_buffer.extend(converted)

                while len(resample_buffer) >= FRAME_BYTES:
                    frame = bytes(resample_buffer[:FRAME_BYTES])
                    del resample_buffer[:FRAME_BYTES]

                    rms = audioop.rms(frame, 2)

                    # 시작 직후 몇 프레임으로 이 환경의 노이즈 플로어를 잡는다.
                    if speech_threshold is None:
                        noise_samples.append(rms)
                        if len(noise_samples) < NOISE_CALIBRATION_FRAMES:
                            continue
                        noise_samples.sort()
                        noise_floor = noise_samples[len(noise_samples) // 2]
                        speech_threshold = max(
                            ABSOLUTE_MIN_RMS, noise_floor * noise_multiplier
                        )
                        self.last_noise_floor = noise_floor
                        self.last_speech_threshold = speech_threshold
                        continue

                    is_speech = (
                        vad.is_speech(frame, SAMPLE_RATE) and rms >= speech_threshold
                    )
                    if is_speech:
                        speech_started = True
                        silence_run_ms = 0
                        collected.append(frame)
                    elif speech_started:
                        silence_run_ms += FRAME_MS
                        # 말 끝의 여운도 인식에 도움이 되므로 함께 담는다.
                        collected.append(frame)
                        if silence_run_ms >= silence_ms:
                            done = True
                            break

        return b"".join(collected) if speech_started else b""

    @staticmethod
    def _write_wav(path: Path, pcm: bytes) -> None:
        with wave.open(str(path), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(SAMPLE_RATE)
            handle.writeframes(pcm)

    def transcribe_pcm(self, pcm: bytes) -> str:
        if not pcm:
            return ""
        with TemporaryDirectory(prefix="viassist_stt_") as temp_dir:
            wav_path = Path(temp_dir) / "utterance.wav"
            self._write_wav(wav_path, pcm)
            return self.transcribe_file(wav_path)

    def transcribe_file(self, wav_path: Path) -> str:
        segments, _info = self.model.transcribe(
            str(wav_path),
            language=self.language,
            beam_size=self.beam_size,
            # 소음 구간에서 문장을 지어내는 것을 한 겹 더 막는다.
            vad_filter=True,
            condition_on_previous_text=False,
        )
        kept = [
            segment.text
            for segment in segments
            if getattr(segment, "no_speech_prob", 0.0) <= MAX_NO_SPEECH_PROB
        ]
        return "".join(kept).strip()

    def listen(
        self,
        *,
        max_seconds: float = 8.0,
        silence_ms: int = 800,
        start_timeout_s: float = 4.0,
        wait: bool = False,
    ) -> dict[str, Any]:
        """한 번 듣고 텍스트로 돌려준다. 동시에 두 번 녹음하지 않는다."""

        if not self._lock.acquire(blocking=wait):
            raise STTBusyError("이전 음성 인식이 아직 처리 중입니다.")

        try:
            record_started = time.perf_counter()
            pcm = self.record(
                max_seconds=max_seconds,
                silence_ms=silence_ms,
                start_timeout_s=start_timeout_s,
            )
            record_ms = (time.perf_counter() - record_started) * 1000

            transcribe_started = time.perf_counter()
            text = self.transcribe_pcm(pcm)
            transcribe_ms = (time.perf_counter() - transcribe_started) * 1000
        finally:
            self._lock.release()

        return {
            "text": text,
            "heard_speech": bool(pcm),
            "audio_seconds": round(len(pcm) / (SAMPLE_RATE * 2), 2),
            "record_ms": round(record_ms, 2),
            "transcribe_ms": round(transcribe_ms, 2),
            "model_size": self.model_size,
            "noise_floor": self.last_noise_floor,
            "speech_threshold": (
                round(self.last_speech_threshold, 1)
                if self.last_speech_threshold is not None
                else None
            ),
        }

    def listen_streaming(
        self,
        *,
        max_seconds: float = 8.0,
        silence_ms: int = 800,
        start_timeout_s: float = 4.0,
        partial_interval_ms: int = PARTIAL_INTERVAL_MS,
        aggressiveness: int = 3,
        wait: bool = False,
    ):
        """말하는 동안 부분 인식 결과를 내보내는 제너레이터.

        `{"type": "partial", ...}`을 여러 번 내보낸 뒤 마지막에
        `{"type": "final", ...}`을 한 번 내보낸다. 부분 인식은 워커
        스레드에서 돌기 때문에 녹음이 끊기지 않는다.
        """

        try:
            import sounddevice as sd
            import webrtcvad
        except ImportError as error:
            raise STTUnavailableError(
                "sounddevice 또는 webrtcvad를 불러오지 못했습니다."
            ) from error

        if not self._lock.acquire(blocking=wait):
            raise STTBusyError("이전 음성 인식이 아직 처리 중입니다.")

        vad = webrtcvad.Vad(aggressiveness)
        frames: queue.Queue[bytes] = queue.Queue()

        def callback(indata, _frames, _time, _status) -> None:
            frames.put(bytes(indata))

        collected: list[bytes] = []
        noise_samples: list[int] = []
        speech_threshold: Optional[float] = None
        speech_started = False
        silence_run_ms = 0
        started_at = time.monotonic()

        partial_text = ""
        partial_worker: Optional[threading.Thread] = None
        partial_result: dict[str, Any] = {}
        last_partial_at = 0.0
        partial_count = 0

        def transcribe_worker(pcm_bytes: bytes) -> None:
            try:
                partial_result["text"] = self.transcribe_pcm(pcm_bytes)
            except Exception:  # noqa: BLE001 - 부분 인식 실패는 무시한다
                partial_result["text"] = ""

        native_frame_bytes = self.native_frame_samples * 2
        resample_state = None
        resample_buffer = bytearray()
        done = False

        try:
            with sd.RawInputStream(
                samplerate=self.native_rate,
                blocksize=self.native_frame_samples,
                device=self.input_device,
                dtype="int16",
                channels=1,
                callback=callback,
            ):
                while not done:
                    elapsed = time.monotonic() - started_at
                    if elapsed > max_seconds:
                        break
                    if not speech_started and elapsed > start_timeout_s:
                        break

                    try:
                        native_frame = frames.get(timeout=0.5)
                    except queue.Empty:
                        continue
                    if len(native_frame) != native_frame_bytes:
                        continue

                    if self.native_rate != SAMPLE_RATE:
                        converted, resample_state = audioop.ratecv(
                            native_frame, 2, 1, self.native_rate, SAMPLE_RATE, resample_state
                        )
                    else:
                        converted = native_frame
                    resample_buffer.extend(converted)

                    while len(resample_buffer) >= FRAME_BYTES:
                        frame = bytes(resample_buffer[:FRAME_BYTES])
                        del resample_buffer[:FRAME_BYTES]

                        rms = audioop.rms(frame, 2)

                        if speech_threshold is None:
                            noise_samples.append(rms)
                            if len(noise_samples) < NOISE_CALIBRATION_FRAMES:
                                continue
                            noise_samples.sort()
                            noise_floor = noise_samples[len(noise_samples) // 2]
                            speech_threshold = max(
                                ABSOLUTE_MIN_RMS, noise_floor * self.noise_multiplier
                            )
                            self.last_noise_floor = noise_floor
                            self.last_speech_threshold = speech_threshold
                            continue

                        is_speech = (
                            vad.is_speech(frame, SAMPLE_RATE) and rms >= speech_threshold
                        )
                        if is_speech:
                            speech_started = True
                            silence_run_ms = 0
                            collected.append(frame)
                        elif speech_started:
                            silence_run_ms += FRAME_MS
                            collected.append(frame)
                            if silence_run_ms >= silence_ms:
                                done = True
                                break

                        # 워커가 비어 있고 주기가 됐으면 지금까지 버퍼를 인식한다.
                        now = time.monotonic()
                        worker_idle = partial_worker is None or not partial_worker.is_alive()
                        if (
                            speech_started
                            and worker_idle
                            and (now - last_partial_at) * 1000 >= partial_interval_ms
                            and len(collected) > NOISE_CALIBRATION_FRAMES
                        ):
                            if partial_result.get("text"):
                                new_text = partial_result["text"]
                                if new_text != partial_text:
                                    partial_text = new_text
                                    partial_count += 1
                                    yield {
                                        "type": "partial",
                                        "text": partial_text,
                                        "elapsed_ms": round(
                                            (now - started_at) * 1000, 1
                                        ),
                                    }
                            partial_result.clear()
                            last_partial_at = now
                            partial_worker = threading.Thread(
                                target=transcribe_worker,
                                args=(b"".join(collected),),
                                daemon=True,
                            )
                            partial_worker.start()

            # 발화 종료. 마지막 인식만 마저 돌린다.
            if partial_worker is not None and partial_worker.is_alive():
                partial_worker.join(timeout=3.0)

            pcm = b"".join(collected) if speech_started else b""
            final_started = time.perf_counter()
            final_text = self.transcribe_pcm(pcm)
            final_ms = (time.perf_counter() - final_started) * 1000
        finally:
            self._lock.release()

        yield {
            "type": "final",
            "text": final_text,
            "heard_speech": bool(pcm),
            "audio_seconds": round(len(pcm) / (SAMPLE_RATE * 2), 2),
            "final_transcribe_ms": round(final_ms, 2),
            "total_ms": round((time.monotonic() - started_at) * 1000, 1),
            "partial_count": partial_count,
            "model_size": self.model_size,
            "noise_floor": self.last_noise_floor,
            "speech_threshold": (
                round(self.last_speech_threshold, 1)
                if self.last_speech_threshold is not None
                else None
            ),
        }
