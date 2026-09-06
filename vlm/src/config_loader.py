from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.exceptions import VLMConfigurationError


class ConfigValidationError(VLMConfigurationError):
    """VLM 설정 파일이 필수 형식이나 값을 만족하지 않을 때 발생한다."""


# 엔진이 실제로 반영하지 않는 설정은 구현된 것처럼 보이지 않도록 거부한다.
UNSUPPORTED_KEYS = {
    "temperature": (
        "현재 엔진은 greedy decoding(do_sample=False)만 사용하므로 "
        "이 값은 추론에 반영되지 않습니다. 설정에서 제거하세요."
    ),
}

# 4bit 양자화는 아직 구현되어 있지 않다. 값을 조용히 무시하지 않는다.
SUPPORTED_QUANTIZATION = {"none"}

ENGINE_LOCAL = "local"
ENGINE_GEMINI = "gemini"
ENGINE_LLAMACPP = "llamacpp"
SUPPORTED_ENGINES = {ENGINE_LOCAL, ENGINE_GEMINI, ENGINE_LLAMACPP}

# engine 키가 없는 기존 설정(jetson.json, pc.json)은 로컬 엔진으로 본다.
DEFAULT_ENGINE = ENGINE_LOCAL

BASE_REQUIRED_KEYS = {
    "environment",
    "max_new_tokens",
    "use_mock_model",
    "model_id",
}

# 로컬 엔진은 GPU와 이미지 프로세서 설정이 모두 필요하다.
LOCAL_REQUIRED_KEYS = BASE_REQUIRED_KEYS | {
    "device",
    "image_longest_edge",
    "max_image_size",
}

# API 엔진은 로컬 device·이미지 프로세서 설정을 쓰지 않는다.
# max_image_size는 업로드 크기를 제한하는 용도로 계속 쓴다.
GEMINI_REQUIRED_KEYS = BASE_REQUIRED_KEYS | {"max_image_size"}

# llama.cpp 엔진: 로컬 device 대신 llamacpp 블록(서버 주소·모델 경로)을 쓴다. 이미지 크기는 클라이언트에서 맞춘다.
LLAMACPP_REQUIRED_KEYS = BASE_REQUIRED_KEYS | {"image_longest_edge", "max_image_size", "llamacpp"}

REQUIRED_KEYS_BY_ENGINE = {
    ENGINE_LOCAL: LOCAL_REQUIRED_KEYS,
    ENGINE_GEMINI: GEMINI_REQUIRED_KEYS,
    ENGINE_LLAMACPP: LLAMACPP_REQUIRED_KEYS,
}


def _validate_positive_int(config: dict[str, Any], key: str) -> None:
    value = config[key]
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ConfigValidationError(f"{key}: 양의 정수여야 합니다.")


def _validate_gemini_options(config: dict[str, Any]) -> None:
    """Gemini 엔진에서만 의미가 있는 선택 키를 검증한다."""

    fallbacks = config.get("fallback_model_ids")
    if fallbacks is not None:
        if not isinstance(fallbacks, list) or not all(
            isinstance(item, str) and item.strip() for item in fallbacks
        ):
            raise ConfigValidationError(
                "fallback_model_ids: 비어 있지 않은 문자열의 배열이어야 합니다."
            )

    timeout = config.get("request_timeout_seconds")
    if timeout is not None:
        if (
            isinstance(timeout, bool)
            or not isinstance(timeout, (int, float))
            or timeout <= 0
        ):
            raise ConfigValidationError(
                "request_timeout_seconds: 0보다 큰 숫자여야 합니다."
            )

    stream = config.get("stream")
    if stream is not None and not isinstance(stream, bool):
        raise ConfigValidationError("stream: boolean이어야 합니다.")

    quality = config.get("jpeg_quality")
    if quality is not None:
        if (
            isinstance(quality, bool)
            or not isinstance(quality, int)
            or not 1 <= quality <= 100
        ):
            raise ConfigValidationError("jpeg_quality: 1~100 사이의 정수여야 합니다.")


def load_config(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"설정 파일을 찾을 수 없습니다: {path}")

    try:
        with path.open("r", encoding="utf-8") as file:
            config = json.load(file)
    except json.JSONDecodeError as error:
        raise ConfigValidationError(
            f"설정 JSON 형식이 올바르지 않습니다: {path} "
            f"(line {error.lineno}, column {error.colno})"
        ) from error

    if not isinstance(config, dict):
        raise ConfigValidationError("설정 JSON의 최상위 구조는 객체여야 합니다.")

    engine = config.get("engine", DEFAULT_ENGINE)
    if not isinstance(engine, str) or engine not in SUPPORTED_ENGINES:
        raise ConfigValidationError(
            f"engine: {sorted(SUPPORTED_ENGINES)} 중 하나여야 합니다."
        )
    config["engine"] = engine

    required_keys = REQUIRED_KEYS_BY_ENGINE[engine]
    missing = sorted(required_keys - config.keys())
    if missing:
        raise ConfigValidationError(f"필수 설정 키가 누락되었습니다: {', '.join(missing)}")

    for key in ("environment", "model_id"):
        if not isinstance(config[key], str) or not config[key].strip():
            raise ConfigValidationError(f"{key}: 비어 있지 않은 문자열이어야 합니다.")

    # device는 로컬 엔진에서만 필수다. API 엔진에 들어 있으면 값만 확인한다.
    if "device" in config:
        if not isinstance(config["device"], str) or not config["device"].strip():
            raise ConfigValidationError("device: 비어 있지 않은 문자열이어야 합니다.")
        if config["device"] not in {"auto", "cuda", "cpu"}:
            raise ConfigValidationError("device: auto, cuda, cpu 중 하나여야 합니다.")

    if not isinstance(config["use_mock_model"], bool):
        raise ConfigValidationError("use_mock_model: boolean이어야 합니다.")

    for key in ("max_new_tokens", "image_longest_edge", "max_image_size"):
        if key in config:
            _validate_positive_int(config, key)

    for key, guidance in UNSUPPORTED_KEYS.items():
        if key in config:
            raise ConfigValidationError(f"{key}: {guidance}")

    if engine == ENGINE_LOCAL:
        quantization = config.get("quantization")
        if quantization is not None and (
            not isinstance(quantization, str)
            or quantization.strip().lower() not in SUPPORTED_QUANTIZATION
        ):
            raise ConfigValidationError(
                "quantization: 현재 엔진은 양자화를 구현하지 않았습니다. "
                f"허용 값은 {sorted(SUPPORTED_QUANTIZATION)}입니다."
            )
    elif engine == ENGINE_GEMINI:
        _validate_gemini_options(config)
    else:
        block = config.get("llamacpp")
        if not isinstance(block, dict):
            raise ConfigValidationError("llamacpp: 객체(server_url, model_path, mmproj_path ...)여야 합니다.")

    return config
