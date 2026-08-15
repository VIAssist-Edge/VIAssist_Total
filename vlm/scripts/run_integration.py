from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config_loader import ConfigValidationError, load_config
from src.exceptions import (
    IntegrationError,
    PerceptionUnavailableError,
    VLMError,
    VLMModelLoadError,
    YoloResultValidationError,
)
from src.integration_pipeline import VIAssistVLMPipeline
from src.metadata_adapter import DEFAULT_USER_QUERY
from src.perception_adapter import build_perception_metadata
from src.vlm_service import VLMService


def positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("0보다 큰 실수여야 합니다.")
    return parsed


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="VIAssist vision-to-VLM integration")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--image", type=Path, default=None)
    parser.add_argument("--yolo-json", type=Path, required=True)
    parser.add_argument("--motion-json", type=Path, default=None)
    parser.add_argument("--quality-json", type=Path, default=None)
    parser.add_argument("--user-query", default=DEFAULT_USER_QUERY)
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/integration_result.json")
    )
    parser.add_argument("--metadata-output", type=Path, default=None)
    parser.add_argument("--strict-frame-sync", action="store_true")
    parser.add_argument("--strict-inference", action="store_true")
    parser.add_argument(
        "--perception",
        action="store_true",
        help=(
            "Perception process_split() payload로 해석한다. "
            "항상 안전 추론 경로를 사용한다."
        ),
    )
    parser.add_argument("--timeout-seconds", type=positive_float, default=None)
    return parser.parse_args(argv)


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"JSON 파일을 찾을 수 없습니다: {path}")
    try:
        with path.open("r", encoding="utf-8") as file:
            value = json.load(file)
    except json.JSONDecodeError as error:
        raise ValueError(
            f"JSON 형식이 올바르지 않습니다: {path} "
            f"(line {error.lineno}, column {error.colno})"
        ) from error
    if not isinstance(value, dict):
        raise ValueError(f"JSON 최상위 구조는 객체여야 합니다: {path}")
    return value


def save_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(value, file, ensure_ascii=False, indent=2)


def run_integration(
    *,
    config: dict[str, Any],
    image_path: Path | None,
    yolo_result: dict[str, Any],
    motion_result: dict[str, Any] | None,
    image_quality: dict[str, Any] | None,
    user_query: str,
    output_path: Path,
    metadata_output_path: Path | None,
    strict_frame_sync: bool,
    strict_inference: bool,
    timeout_seconds: float | None,
    service_factory: Callable[[dict[str, Any]], Any] | None = None,
    perception: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if image_path is not None and not image_path.is_file():
        raise FileNotFoundError(f"이미지 파일을 찾을 수 없습니다: {image_path}")
    if image_path is None and not config["use_mock_model"]:
        raise FileNotFoundError("실제 VLM integration에는 --image가 필요합니다.")
    if perception and strict_inference:
        raise ValueError(
            "--perception은 항상 안전 추론 경로를 사용하므로 "
            "--strict-inference와 함께 쓸 수 없습니다."
        )

    factory = service_factory or VLMService.from_config
    try:
        service = factory(config)
    except VLMError:
        raise
    except Exception as error:
        raise VLMModelLoadError(
            "VLM 모델을 초기화하지 못했습니다.",
            original_exception=error,
        ) from error
    pipeline = VIAssistVLMPipeline(
        service,
        strict_frame_sync=strict_frame_sync,
    )
    if perception:
        try:
            metadata = build_perception_metadata(
                yolo_payload=yolo_result,
                flow_payload=motion_result,
                image_quality=image_quality,
                user_query=user_query,
                strict_frame_sync=strict_frame_sync,
            )
        except (PerceptionUnavailableError, YoloResultValidationError):
            # Perception 실패는 오류 종료가 아니라 degraded 결과로 처리한다.
            metadata = {}
    else:
        metadata = pipeline.build_metadata(
            yolo_result=yolo_result,
            motion_result=motion_result,
            image_quality=image_quality,
            user_query=user_query,
        )
    if metadata_output_path is not None and metadata:
        save_json(metadata_output_path, metadata)

    if perception:
        result = pipeline.process_perception(
            image_path=image_path,
            yolo_payload=yolo_result,
            flow_payload=motion_result,
            image_quality=image_quality,
            user_query=user_query,
            timeout_seconds=timeout_seconds,
        )
    else:
        result = pipeline.process(
            image_path=image_path,
            yolo_result=yolo_result,
            motion_result=motion_result,
            image_quality=image_quality,
            user_query=user_query,
            safe=not strict_inference,
            timeout_seconds=timeout_seconds,
        )
    save_json(output_path, result)
    return metadata, result


def main(argv: list[str] | None = None) -> int:
    try:
        args = parse_args(argv)
        config = load_config(args.config)
        yolo_result = load_json(args.yolo_json)
        motion_result = load_json(args.motion_json) if args.motion_json else None
        image_quality = load_json(args.quality_json) if args.quality_json else None
        metadata, result = run_integration(
            config=config,
            image_path=args.image,
            yolo_result=yolo_result,
            motion_result=motion_result,
            image_quality=image_quality,
            user_query=args.user_query,
            output_path=args.output,
            metadata_output_path=args.metadata_output,
            strict_frame_sync=args.strict_frame_sync,
            strict_inference=args.strict_inference,
            timeout_seconds=args.timeout_seconds,
            perception=args.perception,
        )
    except VLMModelLoadError as error:
        print(f"모델 로딩 오류: {error}", file=sys.stderr)
        return 3
    except (IntegrationError, ConfigValidationError, FileNotFoundError, ValueError) as error:
        print(f"입력/계약 오류: {error}", file=sys.stderr)
        return 2
    except VLMError as error:
        print(f"추론 오류: {error}", file=sys.stderr)
        return 4
    except OSError as error:
        print(f"결과 저장 오류: {error}", file=sys.stderr)
        return 5

    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.metadata_output is not None:
        print(f"\nmetadata 저장 위치: {args.metadata_output.resolve()}")
    print(f"결과 저장 위치: {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
