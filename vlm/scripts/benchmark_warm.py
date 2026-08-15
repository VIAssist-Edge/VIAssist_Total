from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config_loader import ConfigValidationError, load_config
from src.metadata_schema import MetadataValidationError, validate_metadata
from src.vlm_service import VLMService


class BenchmarkError(RuntimeError):
    """Warm benchmark 실행 단계가 실패했을 때 발생한다."""


class ModelLoadError(BenchmarkError):
    pass


class InferenceError(BenchmarkError):
    pass


class ReportSaveError(BenchmarkError):
    pass


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("1 이상의 정수여야 합니다.")
    return parsed


def nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("0 이상의 정수여야 합니다.")
    return parsed


def nonnegative_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0:
        raise argparse.ArgumentTypeError("0 이상의 실수여야 합니다.")
    return parsed


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Persistent VLM warm inference benchmark"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--runs", type=positive_int, default=3)
    parser.add_argument("--warmup-runs", type=nonnegative_int, default=1)
    parser.add_argument(
        "--output", type=Path, default=Path("reports/warm_benchmark.json")
    )
    parser.add_argument("--label", default="")
    parser.add_argument("--sleep-seconds", type=nonnegative_float, default=0.0)
    return parser.parse_args(argv)


def load_metadata(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"metadata 파일을 찾을 수 없습니다: {path}")
    try:
        with path.open("r", encoding="utf-8") as file:
            value = json.load(file)
    except json.JSONDecodeError as error:
        raise MetadataValidationError(
            f"metadata JSON 형식이 올바르지 않습니다: {path} "
            f"(line {error.lineno}, column {error.colno})"
        ) from error
    return validate_metadata(value)


def save_report(path: Path, report: dict[str, Any]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as file:
            json.dump(report, file, ensure_ascii=False, indent=2)
    except OSError as error:
        raise ReportSaveError(f"결과 보고서를 저장할 수 없습니다: {path}: {error}") from error


def display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def rounded_stats(values: list[float]) -> dict[str, float]:
    return {
        "mean": round(statistics.mean(values), 2),
        "median": round(statistics.median(values), 2),
        "min": round(min(values), 2),
        "max": round(max(values), 2),
        "stdev": round(statistics.stdev(values), 2) if len(values) > 1 else 0.0,
    }


def run_benchmark(
    *,
    config: dict[str, Any],
    config_path: Path,
    metadata: dict[str, Any],
    metadata_path: Path,
    image_path: Path,
    runs: int,
    warmup_runs: int,
    output_path: Path,
    label: str,
    sleep_seconds: float,
    service_factory: Callable[[dict[str, Any]], Any] | None = None,
    script_start_time: float | None = None,
) -> dict[str, Any]:
    if runs < 1:
        raise ValueError("runs는 1 이상이어야 합니다.")
    if warmup_runs < 0:
        raise ValueError("warmup_runs는 0 이상이어야 합니다.")
    if not math.isfinite(sleep_seconds) or sleep_seconds < 0:
        raise ValueError("sleep_seconds는 0 이상이어야 합니다.")
    if not image_path.is_file():
        raise FileNotFoundError(f"이미지 파일을 찾을 수 없습니다: {image_path}")
    validate_metadata(metadata)

    factory = service_factory or VLMService.from_config
    overall_start = (
        script_start_time if script_start_time is not None else time.perf_counter()
    )
    service_start = time.perf_counter()
    try:
        service = factory(config)
    except Exception as error:
        raise ModelLoadError(f"모델/service 로딩 실패: {type(error).__name__}: {error}") from error
    model_load_ms = round((time.perf_counter() - service_start) * 1000, 2)

    warmup_latencies: list[float] = []
    for index in range(warmup_runs):
        try:
            result = service.infer(image_path, metadata)
        except Exception as error:
            raise InferenceError(
                f"warm-up 추론 {index + 1} 실패: {type(error).__name__}: {error}"
            ) from error
        warmup_latencies.append(round(float(result["latency_ms"]), 2))
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)

    inference_latencies: list[float] = []
    memory_values: list[float] = []
    results: list[dict[str, Any]] = []
    fallback_count = 0
    vlm_output_count = 0

    for index in range(runs):
        try:
            result = service.infer(image_path, metadata)
        except Exception as error:
            raise InferenceError(
                f"측정 추론 {index + 1} 실패: {type(error).__name__}: {error}"
            ) from error

        latency = round(float(result["latency_ms"]), 2)
        memory = round(float(result.get("peak_gpu_memory_mb", 0.0)), 2)
        inference_latencies.append(latency)
        memory_values.append(memory)
        fallback_count += int(result.get("used_fallback") is True)
        vlm_output_count += int(result.get("message_source") == "vlm")
        results.append(
            {
                "run_index": index + 1,
                "latency_ms": latency,
                "peak_gpu_memory_mb": memory,
                "used_fallback": bool(result.get("used_fallback", False)),
                "message_source": result.get("message_source"),
                "message": result.get("message"),
                "raw_vlm_message": result.get("raw_vlm_message"),
                "status": result.get("status"),
                "validation_reasons": result.get("validation_reasons", []),
            }
        )
        if sleep_seconds > 0 and index + 1 < runs:
            time.sleep(sleep_seconds)

    inference_end = time.perf_counter()
    report = {
        "benchmark_version": 1,
        "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
        "environment": config["environment"],
        "label": label,
        "config_path": display_path(config_path),
        "image_path": display_path(image_path),
        "metadata_path": display_path(metadata_path),
        "model_id": config["model_id"],
        "device": config["device"],
        "runs": runs,
        "warmup_runs": warmup_runs,
        "model_load_ms": model_load_ms,
        "warmup_latencies_ms": warmup_latencies,
        "inference_latencies_ms": inference_latencies,
        "latency_stats_ms": rounded_stats(inference_latencies),
        "peak_gpu_memory_mb": {
            "max": round(max(memory_values), 2),
            "mean": round(statistics.mean(memory_values), 2),
            "values": memory_values,
        },
        "success_count": len(results),
        "failure_count": 0,
        "fallback_count": fallback_count,
        "vlm_output_count": vlm_output_count,
        "failures": [],
        "results": results,
        "total_benchmark_ms": round((inference_end - service_start) * 1000, 2),
        "script_total_ms": round((inference_end - overall_start) * 1000, 2),
    }
    save_report(output_path, report)
    return report


def main(argv: list[str] | None = None) -> int:
    script_start = time.perf_counter()
    try:
        args = parse_args(argv)
        config = load_config(args.config)
        metadata = load_metadata(args.metadata)
        report = run_benchmark(
            config=config,
            config_path=args.config,
            metadata=metadata,
            metadata_path=args.metadata,
            image_path=args.image,
            runs=args.runs,
            warmup_runs=args.warmup_runs,
            output_path=args.output,
            label=args.label,
            sleep_seconds=args.sleep_seconds,
            script_start_time=script_start,
        )
    except ModelLoadError as error:
        print(f"모델 로딩 오류: {error}", file=sys.stderr)
        return 3
    except InferenceError as error:
        print(f"추론 오류: {error}", file=sys.stderr)
        return 4
    except ReportSaveError as error:
        print(f"결과 저장 오류: {error}", file=sys.stderr)
        return 5
    except (FileNotFoundError, ConfigValidationError, MetadataValidationError, ValueError) as error:
        print(f"입력 오류: {error}", file=sys.stderr)
        return 2

    summary = {
        key: report[key]
        for key in (
            "model_load_ms",
            "warmup_latencies_ms",
            "latency_stats_ms",
            "peak_gpu_memory_mb",
            "success_count",
            "failure_count",
            "fallback_count",
            "vlm_output_count",
            "total_benchmark_ms",
            "script_total_ms",
        )
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\n결과 저장 위치: {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
