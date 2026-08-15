from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

from src.config_loader import ConfigValidationError, load_config
from src.exceptions import (
    VLMConfigurationError,
    VLMError,
    VLMModelLoadError,
)
from src.metadata_schema import validate_metadata
from src.vlm_service import VLMService


DEFAULT_MODEL_ID = "HuggingFaceTB/SmolVLM-500M-Instruct"


def positive_timeout(value: str) -> float:
    timeout = float(value)
    if not math.isfinite(timeout) or timeout <= 0:
        raise argparse.ArgumentTypeError("0보다 큰 실수여야 합니다.")
    return timeout


def load_json(path: Path, *, validate: bool = True) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(
            f"입력 파일을 찾을 수 없습니다: {path}"
        )

    try:
        with path.open(
            "r",
            encoding="utf-8",
        ) as file:
            data = json.load(file)

    except json.JSONDecodeError as error:
        raise ValueError(
            f"JSON 형식이 올바르지 않습니다: {path}"
        ) from error

    if not isinstance(data, dict):
        raise ValueError(
            "입력 JSON의 최상위 구조는 객체여야 합니다."
        )

    return validate_metadata(data) if validate else data


def save_json(
    path: Path,
    data: dict[str, Any],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            data,
            file,
            ensure_ascii=False,
            indent=2,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="시각장애인 안전 보행 VLM MVP"
    )

    parser.add_argument(
        "--safe",
        action="store_true",
        help="VLM 오류 발생 시 metadata 기반 안전 fallback 반환",
    )

    parser.add_argument(
        "--timeout-seconds",
        type=positive_timeout,
        default=None,
        help="완료 후 경과 시간을 검사하는 soft timeout",
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="VLM JSON 설정 파일",
    )

    parser.add_argument(
        "--metadata",
        type=Path,
        required=True,
        help="YOLO 및 Optical Flow 결과 JSON",
    )

    parser.add_argument(
        "--image",
        type=Path,
        default=None,
        help="입력 이미지 경로",
    )

    parser.add_argument(
        "--engine",
        choices=["mock", "smolvlm"],
        default="mock",
        help="사용할 추론 엔진",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/result.json"),
        help="결과 JSON 저장 경로",
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    metadata = load_json(args.metadata, validate=not args.safe)
    config = load_config(args.config) if args.config else {
        "environment": "legacy-cli",
        "device": "auto",
        "max_new_tokens": 80,
        "use_mock_model": args.engine == "mock",
        "model_id": DEFAULT_MODEL_ID,
        "image_longest_edge": 1024,
        "max_image_size": 512,
    }
    service = VLMService.from_config(config)
    result = (
        service.infer_safe(
            args.image,
            metadata,
            timeout_seconds=args.timeout_seconds,
        )
        if args.safe
        else service.infer(args.image, metadata)
    )

    save_json(args.output, result)

    print(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2,
        )
    )

    print(
        f"\n결과 저장 위치: "
        f"{args.output.resolve()}"
    )

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except VLMModelLoadError as error:
        print(f"모델 로딩 오류: {error}", file=sys.stderr)
        raise SystemExit(3) from None
    except (VLMConfigurationError, FileNotFoundError, ValueError, ConfigValidationError) as error:
        print(f"입력 오류: {error}", file=sys.stderr)
        raise SystemExit(2) from None
    except VLMError as error:
        print(f"추론 오류: {error}", file=sys.stderr)
        raise SystemExit(4) from None
    except OSError as error:
        print(f"결과 저장 오류: {error}", file=sys.stderr)
        raise SystemExit(5) from None
