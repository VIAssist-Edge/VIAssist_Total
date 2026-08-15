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

REQUIRED_KEYS = {
    "environment",
    "device",
    "max_new_tokens",
    "use_mock_model",
    "model_id",
    "image_longest_edge",
    "max_image_size",
}


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

    missing = sorted(REQUIRED_KEYS - config.keys())
    if missing:
        raise ConfigValidationError(f"필수 설정 키가 누락되었습니다: {', '.join(missing)}")

    for key in ("environment", "device", "model_id"):
        if not isinstance(config[key], str) or not config[key].strip():
            raise ConfigValidationError(f"{key}: 비어 있지 않은 문자열이어야 합니다.")

    if config["device"] not in {"auto", "cuda", "cpu"}:
        raise ConfigValidationError("device: auto, cuda, cpu 중 하나여야 합니다.")

    if not isinstance(config["use_mock_model"], bool):
        raise ConfigValidationError("use_mock_model: boolean이어야 합니다.")

    for key in ("max_new_tokens", "image_longest_edge", "max_image_size"):
        value = config[key]
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ConfigValidationError(f"{key}: 양의 정수여야 합니다.")

    for key, guidance in UNSUPPORTED_KEYS.items():
        if key in config:
            raise ConfigValidationError(f"{key}: {guidance}")

    quantization = config.get("quantization")
    if quantization is not None and (
        not isinstance(quantization, str)
        or quantization.strip().lower() not in SUPPORTED_QUANTIZATION
    ):
        raise ConfigValidationError(
            "quantization: 현재 엔진은 양자화를 구현하지 않았습니다. "
            f"허용 값은 {sorted(SUPPORTED_QUANTIZATION)}입니다."
        )

    return config
