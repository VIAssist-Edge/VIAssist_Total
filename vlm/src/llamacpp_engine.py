"""llama.cpp(llama-server) 백엔드 VLM 엔진.

같은 SmolVLM-500M을 HF transformers 대신 llama.cpp GGUF로 돌린다. 젯슨 Orin Nano 실측(2026-09-06):
호출 4.3~9 s → 0.62 s, 상주 1.7 GB → ~0.4 GB, 생성 89 tok/s. 프롬프트·안전 규칙·결과 파서는 그대로 쓰고
`SmolVLMEngine`과 같은 인터페이스(generate / model_id / get_peak_memory_mb)만 맞춘다.

llama-server는 이 프로세스가 자식으로 띄우고(autostart) 종료 시 함께 내린다. 이미 떠 있는 서버를 쓰려면
autostart=false 로 두고 server_url 만 준다.

주의(젯슨): --cache-ram 0 필수(없으면 호출마다 KV가 쌓여 OOM), 입력 이미지는 고정 크기 정사각으로 보낸다
(모양이 바뀔 때마다 비전 그래프 버퍼를 다시 잡아 메모리가 튄다).
"""

from __future__ import annotations

import atexit
import base64
import json
import os
import subprocess
import time
import urllib.error
import urllib.request
from io import BytesIO
from pathlib import Path
from typing import Any, Optional

from src.exceptions import (
    VLMConfigurationError,
    VLMImageError,
    VLMInferenceError,
    VLMModelLoadError,
)

DEFAULT_BINARY = "~/vlm8b/llama.cpp/build/bin/llama-server"
DEFAULT_ARGS = ["-ngl", "99", "-np", "1", "-b", "512", "-ub", "512", "-ctk", "q8_0", "-ctv", "q8_0",
                "--cache-ram", "0", "-t", "4", "--no-warmup"]


class LlamaCppVLMEngine:
    """OpenAI 호환 /v1/chat/completions 로 llama-server를 부른다."""

    device = "llamacpp"  # vlm_service가 torch 동기화를 건너뛰도록 cuda가 아닌 값

    def __init__(
        self,
        *,
        model_id: str,
        max_new_tokens: int,
        server_url: str = "http://127.0.0.1:8082",
        model_path: Optional[str] = None,
        mmproj_path: Optional[str] = None,
        binary: str = DEFAULT_BINARY,
        autostart: bool = True,
        ctx: int = 2048,
        extra_args: Optional[list[str]] = None,
        image_longest_edge: int = 512,
        jpeg_quality: int = 85,
        request_timeout_seconds: float = 20.0,
        startup_timeout_seconds: float = 180.0,
        temperature: float = 0.0,
    ) -> None:
        self.model_id = model_id
        self.max_new_tokens = int(max_new_tokens)
        self.server_url = server_url.rstrip("/")
        self.image_longest_edge = int(image_longest_edge)
        self.jpeg_quality = int(jpeg_quality)
        self.request_timeout_seconds = float(request_timeout_seconds)
        self.temperature = float(temperature)
        self.last_timings: dict[str, Any] = {}
        self._process: Optional[subprocess.Popen] = None

        if autostart:
            if not model_path or not mmproj_path:
                raise VLMConfigurationError(
                    "llamacpp 엔진 autostart에는 model_path와 mmproj_path가 필요합니다."
                )
            self._start_server(
                binary=Path(os.path.expanduser(binary)),
                model_path=Path(os.path.expanduser(model_path)),
                mmproj_path=Path(os.path.expanduser(mmproj_path)),
                ctx=int(ctx),
                extra_args=list(extra_args or DEFAULT_ARGS),
                startup_timeout_seconds=float(startup_timeout_seconds),
            )
        else:
            self._wait_healthy(float(startup_timeout_seconds))

        print(f"[VLM] engine: llama.cpp ({self.server_url})")
        print(f"[VLM] model: {self.model_id}")
        self._warmup()

    # ------------------------------------------------------------------ server
    def _start_server(self, *, binary: Path, model_path: Path, mmproj_path: Path,
                      ctx: int, extra_args: list[str], startup_timeout_seconds: float) -> None:
        for path, label in ((binary, "llama-server 실행 파일"), (model_path, "GGUF 모델"), (mmproj_path, "mmproj")):
            if not path.exists():
                raise VLMModelLoadError(f"{label}을 찾을 수 없습니다: {path}")
        host, port = self._host_port()
        command = [str(binary), "-m", str(model_path), "--mmproj", str(mmproj_path),
                   "-c", str(ctx), "--host", host, "--port", str(port), *extra_args]
        log_path = Path.home() / "vlm_llamacpp_server.log"
        self._log_handle = open(log_path, "ab")
        started = time.perf_counter()
        try:
            # 부모가 죽어도 정리되도록 atexit에 등록한다. setsid는 쓰지 않는다(같은 세션에서 함께 종료).
            self._process = subprocess.Popen(
                command, stdout=self._log_handle, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
            )
        except OSError as error:
            raise VLMModelLoadError("llama-server를 시작하지 못했습니다.", original_exception=error) from error
        atexit.register(self.close)
        self._wait_healthy(startup_timeout_seconds)
        print(f"[VLM] llama-server 기동 {(time.perf_counter() - started) * 1000:.0f} ms (log: {log_path})")

    def _host_port(self) -> tuple[str, int]:
        rest = self.server_url.split("//", 1)[-1]
        host, _, port = rest.partition(":")
        return host or "127.0.0.1", int(port or 8080)

    def _wait_healthy(self, timeout_seconds: float) -> None:
        deadline = time.monotonic() + timeout_seconds
        last_error: Optional[str] = None
        while time.monotonic() < deadline:
            if self._process is not None and self._process.poll() is not None:
                raise VLMModelLoadError(
                    f"llama-server가 종료됐습니다(exit {self._process.returncode}). ~/vlm_llamacpp_server.log 확인"
                )
            try:
                with urllib.request.urlopen(self.server_url + "/health", timeout=3) as response:
                    if b'"ok"' in response.read():
                        return
            except Exception as error:  # noqa: BLE001
                last_error = str(error)
            time.sleep(1.0)
        raise VLMModelLoadError(f"llama-server가 {timeout_seconds:.0f}초 안에 준비되지 않았습니다: {last_error}")

    def close(self) -> None:
        process = self._process
        self._process = None
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
        handle = getattr(self, "_log_handle", None)
        if handle is not None:
            try:
                handle.close()
            except Exception:  # noqa: BLE001
                pass

    # --------------------------------------------------------------- inference
    def _encode_image(self, image_path: Path) -> str:
        try:
            from PIL import Image
        except ImportError as error:
            raise VLMImageError("이미지를 처리할 수 없습니다.", original_exception=error) from error
        if not image_path.exists():
            raise VLMImageError("추론할 이미지를 찾을 수 없습니다.", error_code="IMAGE_NOT_FOUND")
        try:
            with Image.open(image_path) as source:
                image = source.convert("RGB")
                edge = self.image_longest_edge
                ratio = edge / float(max(image.size))
                image = image.resize((max(1, round(image.width * ratio)), max(1, round(image.height * ratio))), Image.BILINEAR)
                # 고정 크기 정사각 캔버스: 모양이 바뀌면 llama.cpp가 비전 그래프 버퍼를 다시 잡는다.
                canvas = Image.new("RGB", (edge, edge), (0, 0, 0))
                canvas.paste(image, ((edge - image.width) // 2, (edge - image.height) // 2))
                buffer = BytesIO()
                canvas.save(buffer, format="JPEG", quality=self.jpeg_quality)
        except Exception as error:  # noqa: BLE001
            raise VLMImageError("이미지를 열거나 변환할 수 없습니다.", original_exception=error) from error
        return base64.b64encode(buffer.getvalue()).decode("ascii")

    def generate(self, image_path: Path | None, prompt: str, metadata: dict[str, Any]) -> str:
        if image_path is None:
            raise VLMImageError("추론할 이미지가 제공되지 않았습니다.", error_code="IMAGE_NOT_FOUND")
        image_b64 = self._encode_image(Path(image_path))
        payload = {
            "model": self.model_id,
            "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
                {"type": "text", "text": prompt},
            ]}],
            "max_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "stop": ["\n\n"],
        }
        request = urllib.request.Request(
            self.server_url + "/v1/chat/completions",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.request_timeout_seconds) as response:
                data = json.loads(response.read())
        except urllib.error.HTTPError as error:
            body = error.read().decode("utf-8", "replace")[:300]
            raise VLMInferenceError(f"llama-server 오류 {error.code}: {body}", original_exception=error) from error
        except Exception as error:  # noqa: BLE001
            raise VLMInferenceError("llama-server 요청에 실패했습니다.", original_exception=error) from error

        self.last_timings = {
            "prompt_tokens": data.get("usage", {}).get("prompt_tokens"),
            "completion_tokens": data.get("usage", {}).get("completion_tokens"),
            "prompt_ms": data.get("timings", {}).get("prompt_ms"),
            "predicted_ms": data.get("timings", {}).get("predicted_ms"),
            "predicted_per_second": data.get("timings", {}).get("predicted_per_second"),
        }
        try:
            return str(data["choices"][0]["message"]["content"]).strip()
        except (KeyError, IndexError, TypeError) as error:
            raise VLMInferenceError("llama-server 응답 형식이 예상과 다릅니다.", original_exception=error) from error

    def get_peak_memory_mb(self) -> float:
        # 별도 프로세스라 torch 통계가 없다. 대시보드는 시스템 DRAM을 따로 본다.
        return 0.0

    def _warmup(self) -> None:
        try:
            from tempfile import TemporaryDirectory

            from PIL import Image

            started = time.perf_counter()
            with TemporaryDirectory(prefix="viassist_vlm_warm_") as temp_dir:
                warm_path = Path(temp_dir) / "warm.jpg"
                Image.new("RGB", (self.image_longest_edge, self.image_longest_edge), (40, 40, 40)).save(warm_path, format="JPEG")
                saved = self.max_new_tokens
                self.max_new_tokens = 4
                try:
                    self.generate(warm_path, "이 사진을 한 단어로 말하세요.", {})
                finally:
                    self.max_new_tokens = saved
            print(f"[VLM] warmup {(time.perf_counter() - started) * 1000:.0f} ms")
        except Exception as error:  # noqa: BLE001 - 워밍업 실패는 치명적이지 않다
            print(f"[VLM] warmup skipped: {error}")
