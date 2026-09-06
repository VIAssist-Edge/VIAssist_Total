#!/usr/bin/env python3
"""안내 문장을 한국어 음성으로 바꿔 재생하는 TTS 경계.

기본 엔진은 **MeloTTS 한국어(온디바이스, GPU)** 다. 젯슨 실측으로
RTF 0.20(3초 문장을 0.7초에 합성)이고 발음 검증도 3/3 통과했다.
네트워크가 필요 없다는 점이 보행 보조 기기에서 결정적이다.

MeloTTS는 transformers 4.27.4를 요구하는데 메인 환경은 4.49.0을 쓰므로
같은 프로세스에서 import할 수 없다. `~/melo_env` 안의 워커 프로세스를
한 번 띄워 두고 파이프로 요청을 보낸다.

워커를 못 띄우면 gTTS(클라우드)로 자동 폴백한다. 인터넷이 없으면 그것도
실패하지만, 그때는 `play_error`만 남기고 안내 자체는 계속 진행한다.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Optional


DEFAULT_CACHE_DIR = Path.home() / ".cache" / "viassist" / "tts"
MELO_VENV_PYTHON = Path.home() / "melo_env" / "bin" / "python"
MELO_WORKER = Path(__file__).resolve().parent / "melo_worker.py"

# 워커 기동은 모델 로드 + 워밍업까지 포함한다. 넉넉히 준다.
MELO_STARTUP_TIMEOUT_S = 120.0
MELO_REQUEST_TIMEOUT_S = 30.0

# 앞에서부터 있는 것을 쓴다.
PLAYER_CANDIDATES = (
    ("aplay", ("-q",)),           # wav 재생 (MeloTTS 출력)
    ("mpg123", ("-q",)),          # mp3 재생 (gTTS 폴백)
    ("ffplay", ("-nodisp", "-autoexit", "-loglevel", "quiet")),
)


class TTSUnavailableError(RuntimeError):
    """합성기를 쓸 수 없을 때 발생한다."""


class MeloWorker:
    """venv 안에서 도는 MeloTTS 합성 프로세스를 감싼다."""

    def __init__(self, device: str = "cuda") -> None:
        if not MELO_VENV_PYTHON.is_file():
            raise TTSUnavailableError(f"MeloTTS venv가 없습니다: {MELO_VENV_PYTHON}")
        if not MELO_WORKER.is_file():
            raise TTSUnavailableError(f"워커 스크립트가 없습니다: {MELO_WORKER}")

        environment = dict(os.environ)
        environment["MELO_DEVICE"] = device
        self.device = device
        self.process = subprocess.Popen(
            [str(MELO_VENV_PYTHON), str(MELO_WORKER)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
            env=environment,
        )

        ready = self._read_json(MELO_STARTUP_TIMEOUT_S)
        if not ready or not ready.get("ready"):
            self.close()
            detail = (ready or {}).get("error", "응답 없음")
            raise TTSUnavailableError(f"MeloTTS 워커 기동 실패: {detail}")

        self.load_ms = ready.get("load_ms", 0.0)
        self.speaker = ready.get("speaker", "KR")

    def _read_json(self, timeout_s: float) -> Optional[dict[str, Any]]:
        """워커의 한 줄 응답을 기다린다. 타임아웃이면 None."""

        result: dict[str, Any] = {}

        def reader() -> None:
            line = self.process.stdout.readline() if self.process.stdout else ""
            if line:
                try:
                    result.update(json.loads(line))
                except json.JSONDecodeError:
                    result["error"] = line.strip()[:200]

        thread = threading.Thread(target=reader, daemon=True)
        thread.start()
        thread.join(timeout=timeout_s)
        return result or None

    @property
    def is_alive(self) -> bool:
        return self.process.poll() is None

    def synthesize(self, text: str, path: Path, speed: float = 1.0) -> dict[str, Any]:
        if not self.is_alive:
            raise TTSUnavailableError("MeloTTS 워커가 종료되었습니다.")
        request = json.dumps(
            {"text": text, "path": str(path), "speed": speed}, ensure_ascii=False
        )
        self.process.stdin.write(request + "\n")
        self.process.stdin.flush()

        response = self._read_json(MELO_REQUEST_TIMEOUT_S)
        if not response:
            raise TTSUnavailableError("MeloTTS 워커 응답이 없습니다.")
        if not response.get("ok"):
            raise TTSUnavailableError(
                f"MeloTTS 합성 실패: {response.get('error', '알 수 없음')}"
            )
        return response

    def close(self) -> None:
        try:
            if self.process.stdin:
                self.process.stdin.close()
            self.process.terminate()
            self.process.wait(timeout=5)
        except Exception:  # noqa: BLE001
            try:
                self.process.kill()
            except Exception:  # noqa: BLE001
                pass


class TTSService:
    """문장을 음성으로 만들고 재생하는 단일 인스턴스 경계."""

    def __init__(
        self,
        *,
        language: str = "ko",
        cache_dir: Path = DEFAULT_CACHE_DIR,
        alsa_device: Optional[str] = None,
        engine: str = "melo",
        melo_device: str = "cuda",
        speed: float = 1.0,
    ) -> None:
        self.language = language
        self.speed = speed
        self.alsa_device = alsa_device
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

        self.engine = "gtts"
        self.melo: Optional[MeloWorker] = None
        self.engine_error: Optional[str] = None

        if engine == "melo":
            try:
                started = time.perf_counter()
                self.melo = MeloWorker(device=melo_device)
                self.engine = "melo"
                print(
                    f"[TTS] engine: melo ({melo_device}, "
                    f"기동 {time.perf_counter() - started:.1f}s)"
                )
            except Exception as error:  # noqa: BLE001
                self.engine_error = str(error)
                print(f"[TTS] MeloTTS 사용 불가 → gTTS로 폴백: {error}")

        if self.engine == "gtts":
            try:
                from gtts import gTTS  # noqa: F401 - 가용성만 확인한다
            except ImportError as error:
                raise TTSUnavailableError(
                    "MeloTTS도 gTTS도 쓸 수 없습니다."
                ) from error
            print("[TTS] engine: gtts (클라우드)")

        self.player = self._find_player()
        print(f"[TTS] language: {language}")
        print(f"[TTS] cache: {self.cache_dir}")
        print(f"[TTS] player: {self.player[0] if self.player else '없음(재생 불가)'}")

    @staticmethod
    def _find_player() -> Optional[tuple[str, tuple[str, ...]]]:
        for name, args in PLAYER_CANDIDATES:
            path = shutil.which(name)
            if path:
                return (path, args)
        return None

    def _cache_path(self, text: str) -> Path:
        digest = hashlib.sha256(
            f"{self.engine}|{self.language}|{self.speed}|{text}".encode("utf-8")
        ).hexdigest()[:32]
        suffix = ".wav" if self.engine == "melo" else ".mp3"
        return self.cache_dir / f"{digest}{suffix}"

    def synthesize(self, text: str) -> tuple[Path, bool]:
        """문장을 오디오 파일로 만든다. (경로, 캐시적중여부)를 돌려준다."""

        text = (text or "").strip()
        if not text:
            raise ValueError("합성할 문장이 비어 있습니다.")

        path = self._cache_path(text)
        if path.is_file() and path.stat().st_size > 0:
            return path, True

        # soundfile은 확장자로 포맷을 정하므로 임시 파일도 .wav/.mp3로 끝나야 한다.
        temp_path = path.parent / f"{path.stem}.part{path.suffix}"
        try:
            if self.engine == "melo" and self.melo is not None:
                self.melo.synthesize(text, temp_path, speed=self.speed)
            else:
                from gtts import gTTS

                gTTS(text=text, lang=self.language).save(str(temp_path))
            temp_path.replace(path)
        except Exception as error:  # noqa: BLE001
            temp_path.unlink(missing_ok=True)
            raise TTSUnavailableError(f"음성 합성에 실패했습니다: {error}") from error

        return path, False

    def play(self, path: Path, *, timeout_seconds: float = 30.0) -> Optional[str]:
        """재생한다. 성공하면 None, 실패하면 사유 문자열을 돌려준다."""

        if self.player is None:
            return "재생기(aplay/mpg123/ffplay)가 없습니다."

        binary, args = self.player
        name = Path(binary).name
        # wav는 aplay, mp3는 mpg123으로. 확장자와 맞지 않으면 ffplay를 쓴다.
        if path.suffix == ".wav" and name == "mpg123":
            binary = shutil.which("aplay") or binary
            name = Path(binary).name
            args = ("-q",)
        elif path.suffix == ".mp3" and name == "aplay":
            binary = shutil.which("mpg123") or shutil.which("ffplay") or binary
            name = Path(binary).name
            args = ("-q",) if name == "mpg123" else (
                "-nodisp", "-autoexit", "-loglevel", "quiet"
            )

        command = [binary, *args]
        if self.alsa_device and name in ("mpg123", "aplay"):
            command += (["-a", self.alsa_device] if name == "mpg123"
                        else ["-D", self.alsa_device])
        command.append(str(path))

        try:
            completed = subprocess.run(
                command,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                timeout=timeout_seconds,
            )
        except subprocess.TimeoutExpired:
            return f"재생이 {timeout_seconds}초 안에 끝나지 않았습니다."
        except Exception as error:  # noqa: BLE001
            return f"재생 실행 실패: {error}"

        if completed.returncode != 0:
            detail = completed.stderr.decode("utf-8", errors="ignore").strip()
            return f"재생기 오류(코드 {completed.returncode}): {detail[:200]}"
        return None

    def speak(
        self,
        text: str,
        *,
        play_audio: bool = True,
        wait: bool = True,
    ) -> dict[str, Any]:
        """문장을 합성하고 재생까지 시도한다.

        재생에 실패해도 예외를 올리지 않는다. 호출부가 `play_error`로 판단한다.
        """

        if not self._lock.acquire(blocking=wait):
            return {
                "text": text,
                "spoken": False,
                "play_error": "이전 음성 재생이 아직 진행 중입니다.",
            }

        try:
            started = time.perf_counter()
            path, cached = self.synthesize(text)
            synth_ms = (time.perf_counter() - started) * 1000

            play_error: Optional[str] = None
            play_ms = 0.0
            if play_audio:
                play_started = time.perf_counter()
                play_error = self.play(path)
                play_ms = (time.perf_counter() - play_started) * 1000
        finally:
            self._lock.release()

        return {
            "text": text,
            "engine": self.engine,
            "audio_path": str(path),
            "cached": cached,
            "synthesize_ms": round(synth_ms, 2),
            "play_ms": round(play_ms, 2),
            "spoken": play_audio and play_error is None,
            "play_error": play_error,
        }

    def close(self) -> None:
        if self.melo is not None:
            self.melo.close()
            self.melo = None
