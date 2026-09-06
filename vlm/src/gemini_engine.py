from __future__ import annotations

import base64
import json
import os
import time
from io import BytesIO
from pathlib import Path
from typing import Any, Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from src.exceptions import (
    VLMConfigurationError,
    VLMImageError,
    VLMInferenceError,
)


API_ENDPOINT = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    "{model_id}:generateContent"
)

# SSE 스트리밍 엔드포인트. alt=sse가 없으면 JSON 배열을 통째로 준다.
STREAM_ENDPOINT = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    "{model_id}:streamGenerateContent?alt=sse"
)

# 키 이름 우선순위. 앞에서 찾으면 뒤는 보지 않는다.
API_KEY_ENV_VARS = ("GEMINI_API_KEY", "GOOGLE_API_KEY")

# 환경 변수가 없을 때 읽는 파일. `~/.bashrc`는 비대화형 SSH 실행에서
# 읽히지 않으므로, 어떤 실행 경로에서도 같은 값을 쓰도록 파일도 본다.
# VIASSIST_ENV_FILE로 경로를 덮어쓸 수 있다.
ENV_FILE_PATHS = (
    Path(__file__).resolve().parents[1] / ".env",
    Path.home() / ".config" / "viassist" / "env",
)

# 2.5 계열은 thinking 모델이라 thinkingBudget을 0으로 두지 않으면
# maxOutputTokens를 사고 과정에 모두 써버리고 빈 응답을 돌려준다.
# 3.x는 이 필드를 거부(400 INVALID_ARGUMENT)하므로 넣지 않는다.
THINKING_MODEL_PREFIXES = ("gemini-2.5",)

# 응답이 잘리면 안내 문장이 중간에서 끊긴다. 짧은 안내문이라도
# 여유를 두고 받아서 Safety Validator가 온전한 문장을 보게 한다.
MIN_OUTPUT_TOKENS = 256


class GeminiVLMEngine:
    """Gemini API를 쓰는 VLM 엔진.

    `SmolVLMEngine`과 같은 `generate(image_path, prompt, metadata)` 계약을
    지키므로 `VLMService`, `prompt_builder`, `safety_rules`는 그대로 쓴다.

    로컬 GPU를 쓰지 않으므로 `device`는 `"api"`이고 peak GPU memory는 0이다.
    API 키는 설정 파일이 아니라 환경 변수에서만 읽는다.
    """

    device = "api"

    def __init__(
        self,
        model_id: str = "gemini-2.5-flash",
        max_new_tokens: int = 256,
        *,
        api_key: Optional[str] = None,
        fallback_model_ids: Optional[list[str]] = None,
        request_timeout_seconds: float = 20.0,
        max_image_size: int = 768,
        jpeg_quality: int = 85,
        stream: bool = True,
    ) -> None:
        resolved_key = api_key or self._resolve_api_key()
        if not resolved_key:
            raise VLMConfigurationError(
                "Gemini API 키가 없습니다. "
                f"{' 또는 '.join(API_KEY_ENV_VARS)} 환경 변수를 설정하거나 "
                f"{ENV_FILE_PATHS[0]} 파일에 적어 주세요."
            )

        self.model_id = model_id
        self.stream = stream
        self.max_new_tokens = max(int(max_new_tokens), MIN_OUTPUT_TOKENS)
        self.request_timeout_seconds = float(request_timeout_seconds)
        self.max_image_size = int(max_image_size)
        self.jpeg_quality = int(jpeg_quality)
        self._api_key = resolved_key

        self.model_candidates = [model_id]
        for candidate in fallback_model_ids or []:
            if candidate and candidate not in self.model_candidates:
                self.model_candidates.append(candidate)

        # 마지막으로 실제 응답을 만든 모델. 폴백이 일어났는지 추적한다.
        self.last_used_model_id: str = model_id
        self.last_latency_ms: float = 0.0
        # 첫 토큰까지 걸린 시간. 스트리밍일 때만 의미가 있다.
        self.last_ttft_ms: Optional[float] = None
        self.last_chunk_count: int = 0

        mode = "stream(SSE)" if stream else "unary"
        print(f"[VLM] engine: gemini (device={self.device}, {mode})")
        print(f"[VLM] model: {self.model_id}")
        print(f"[VLM] fallback models: {self.model_candidates[1:]}")

    @staticmethod
    def _read_api_key_from_env() -> Optional[str]:
        for name in API_KEY_ENV_VARS:
            value = os.environ.get(name, "").strip()
            if value:
                return value
        return None

    @staticmethod
    def _read_api_key_from_file() -> Optional[str]:
        """KEY=VALUE 형식 파일에서 키를 읽는다. 값은 절대 로깅하지 않는다."""

        candidates: list[Path] = []
        override = os.environ.get("VIASSIST_ENV_FILE", "").strip()
        if override:
            candidates.append(Path(override))
        candidates.extend(ENV_FILE_PATHS)

        for path in candidates:
            try:
                if not path.is_file():
                    continue
                for raw_line in path.read_text(encoding="utf-8").splitlines():
                    line = raw_line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    name, _, value = line.partition("=")
                    name = name.strip().removeprefix("export").strip()
                    if name not in API_KEY_ENV_VARS:
                        continue
                    value = value.strip().strip("\"'")
                    if value:
                        return value
            except OSError:
                continue
        return None

    @classmethod
    def _resolve_api_key(cls) -> Optional[str]:
        return cls._read_api_key_from_env() or cls._read_api_key_from_file()

    def get_peak_memory_mb(self) -> float:
        """API 엔진은 로컬 GPU를 쓰지 않는다."""

        return 0.0

    def _encode_image(self, image_path: Path) -> str:
        try:
            from PIL import Image
        except ImportError as error:
            raise VLMImageError(
                "이미지를 처리할 수 없습니다.",
                original_exception=error,
            ) from error

        if not image_path.exists():
            raise VLMImageError(
                "추론할 이미지를 찾을 수 없습니다.",
                error_code="IMAGE_NOT_FOUND",
            )

        try:
            with Image.open(image_path) as source:
                image = source.convert("RGB")
                longest = max(image.size)
                if longest > self.max_image_size:
                    ratio = self.max_image_size / float(longest)
                    resized = (
                        max(1, round(image.width * ratio)),
                        max(1, round(image.height * ratio)),
                    )
                    image = image.resize(resized, Image.BILINEAR)

                buffer = BytesIO()
                image.save(buffer, format="JPEG", quality=self.jpeg_quality)
        except Exception as error:
            raise VLMImageError(
                "이미지를 열거나 변환할 수 없습니다.",
                original_exception=error,
            ) from error

        return base64.b64encode(buffer.getvalue()).decode("ascii")

    def _build_payload(
        self,
        model_id: str,
        prompt: str,
        image_b64: str,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {"text": prompt},
                        {
                            "inlineData": {
                                "mimeType": "image/jpeg",
                                "data": image_b64,
                            }
                        },
                    ],
                }
            ],
            "generationConfig": {
                # 로컬 엔진의 greedy decoding과 동작을 맞춘다.
                "temperature": 0.0,
                "maxOutputTokens": self.max_new_tokens,
            },
        }
        if model_id.startswith(THINKING_MODEL_PREFIXES):
            payload["generationConfig"]["thinkingConfig"] = {"thinkingBudget": 0}
        return payload

    def _call_model(self, model_id: str, payload: dict[str, Any]) -> str:
        request = Request(
            API_ENDPOINT.format(model_id=model_id),
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                # 키를 URL 쿼리에 넣으면 로그·프록시에 남는다. 헤더로 보낸다.
                "x-goog-api-key": self._api_key,
            },
            method="POST",
        )

        try:
            with urlopen(request, timeout=self.request_timeout_seconds) as response:
                body = response.read().decode("utf-8")
        except HTTPError as error:
            detail = error.read().decode("utf-8", errors="ignore")[:400]
            raise RuntimeError(f"HTTP {error.code}: {detail}") from error
        except URLError as error:
            raise RuntimeError(f"네트워크 오류: {error.reason}") from error
        except TimeoutError as error:
            raise RuntimeError(
                f"응답이 {self.request_timeout_seconds}초 안에 오지 않았습니다."
            ) from error

        try:
            parsed = json.loads(body)
        except json.JSONDecodeError as error:
            raise RuntimeError(f"응답 파싱 실패: {body[:200]}") from error

        candidates = parsed.get("candidates") or []
        if not candidates:
            blocked = parsed.get("promptFeedback", {}).get("blockReason")
            if blocked:
                raise RuntimeError(f"요청이 차단되었습니다: {blocked}")
            raise RuntimeError(f"응답이 비어 있습니다: {body[:200]}")

        candidate = candidates[0]
        parts = candidate.get("content", {}).get("parts") or []
        text = "".join(
            part.get("text", "") for part in parts if isinstance(part, dict)
        ).strip()

        if not text:
            reason = candidate.get("finishReason", "UNKNOWN")
            raise RuntimeError(f"응답 텍스트가 없습니다 (finishReason={reason}).")

        return text

    def _call_model_stream(self, model_id: str, payload: dict[str, Any]) -> str:
        """SSE로 받아 조각을 이어 붙인다. 첫 조각 도착 시각을 기록한다."""

        request = Request(
            STREAM_ENDPOINT.format(model_id=model_id),
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Accept": "text/event-stream",
                "x-goog-api-key": self._api_key,
            },
            method="POST",
        )

        started = time.perf_counter()
        pieces: list[str] = []
        first_at: Optional[float] = None
        finish_reason = "UNKNOWN"
        blocked: Optional[str] = None

        try:
            with urlopen(request, timeout=self.request_timeout_seconds) as response:
                for raw_line in response:
                    line = raw_line.decode("utf-8", errors="ignore").strip()
                    if not line or not line.startswith("data:"):
                        continue
                    body = line[5:].strip()
                    if not body or body == "[DONE]":
                        continue
                    try:
                        chunk = json.loads(body)
                    except json.JSONDecodeError:
                        continue

                    feedback = chunk.get("promptFeedback") or {}
                    if feedback.get("blockReason"):
                        blocked = feedback["blockReason"]
                        break

                    for candidate in chunk.get("candidates") or []:
                        finish_reason = candidate.get("finishReason", finish_reason)
                        for part in candidate.get("content", {}).get("parts") or []:
                            text = part.get("text") or ""
                            if not text:
                                continue
                            if first_at is None:
                                first_at = time.perf_counter()
                            pieces.append(text)
        except HTTPError as error:
            detail = error.read().decode("utf-8", errors="ignore")[:400]
            raise RuntimeError(f"HTTP {error.code}: {detail}") from error
        except URLError as error:
            raise RuntimeError(f"네트워크 오류: {error.reason}") from error
        except TimeoutError as error:
            raise RuntimeError(
                f"응답이 {self.request_timeout_seconds}초 안에 오지 않았습니다."
            ) from error

        if blocked:
            raise RuntimeError(f"요청이 차단되었습니다: {blocked}")

        text = "".join(pieces).strip()
        if not text:
            raise RuntimeError(f"응답 텍스트가 없습니다 (finishReason={finish_reason}).")

        self.last_ttft_ms = (
            round((first_at - started) * 1000, 1) if first_at else None
        )
        self.last_chunk_count = len(pieces)
        return text

    def generate(
        self,
        image_path: Path | None,
        prompt: str,
        metadata: dict[str, Any],
    ) -> str:
        """프롬프트와 이미지를 Gemini에 보내고 안내 문장 후보를 받는다.

        반환값은 검증 전 raw 텍스트다. 형식·과잉 주장 차단은 상위의
        Safety Validator가 담당한다.
        """

        if image_path is None:
            raise VLMImageError(
                "추론할 이미지가 제공되지 않았습니다.",
                error_code="IMAGE_NOT_FOUND",
            )

        image_b64 = self._encode_image(Path(image_path))

        started = time.perf_counter()
        errors: list[str] = []
        for model_id in self.model_candidates:
            try:
                caller = (
                    self._call_model_stream if self.stream else self._call_model
                )
                text = caller(
                    model_id,
                    self._build_payload(model_id, prompt, image_b64),
                )
            except Exception as error:  # noqa: BLE001 - 모델 단위로 폴백한다
                errors.append(f"{model_id}: {error}")
                continue

            self.last_used_model_id = model_id
            self.last_latency_ms = round(
                (time.perf_counter() - started) * 1000, 2
            )
            return text

        raise VLMInferenceError(
            "VLM 추론에 실패했습니다.",
            original_exception=RuntimeError(" | ".join(errors)),
        )
