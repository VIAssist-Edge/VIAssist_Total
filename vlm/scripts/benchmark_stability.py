from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from collections import Counter
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.benchmark_warm import (
    ReportSaveError,
    display_path,
    load_metadata,
    nonnegative_float,
    nonnegative_int,
    positive_int,
    rounded_stats,
    save_report,
)
from src.config_loader import ConfigValidationError, load_config
from src.exceptions import VLMError, VLMModelLoadError
from src.metadata_schema import MetadataValidationError, validate_metadata
from src.vlm_service import VLMService


def positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("0보다 큰 실수여야 합니다.")
    return parsed


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Persistent VLM long-running stability benchmark"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--runs", type=positive_int, default=20)
    parser.add_argument("--warmup-runs", type=nonnegative_int, default=1)
    parser.add_argument("--sleep-seconds", type=nonnegative_float, default=1.0)
    parser.add_argument("--timeout-seconds", type=positive_float, default=None)
    parser.add_argument(
        "--safe", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--label", default="")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/stability_benchmark.json"),
    )
    parser.add_argument("--stop-on-error", action="store_true")
    return parser.parse_args(argv)


def _window_size(length: int) -> int:
    return 5 if length >= 10 else max(1, length // 2)


def drift_stats(values: list[float], *, include_percent: bool) -> dict[str, float]:
    size = _window_size(len(values))
    first_mean = statistics.mean(values[:size])
    last_mean = statistics.mean(values[-size:])
    difference = last_mean - first_mean
    result = {
        "first_window_mean": round(first_mean, 2),
        "last_window_mean": round(last_mean, 2),
        "difference": round(difference, 2),
    }
    if include_percent:
        result["percent"] = (
            round(difference / first_mean * 100, 2) if first_mean != 0 else 0.0
        )
    return result


def run_stability_benchmark(
    *,
    config: dict[str, Any],
    config_path: Path,
    metadata: dict[str, Any],
    metadata_path: Path,
    image_path: Path,
    runs: int,
    warmup_runs: int,
    sleep_seconds: float,
    timeout_seconds: float | None,
    safe: bool,
    label: str,
    output_path: Path,
    stop_on_error: bool,
    service_factory: Callable[[dict[str, Any]], Any] | None = None,
) -> dict[str, Any]:
    if isinstance(runs, bool) or runs < 1:
        raise ValueError("runs는 1 이상이어야 합니다.")
    if isinstance(warmup_runs, bool) or warmup_runs < 0:
        raise ValueError("warmup_runs는 0 이상이어야 합니다.")
    if (
        isinstance(sleep_seconds, bool)
        or not isinstance(sleep_seconds, (int, float))
        or not math.isfinite(float(sleep_seconds))
        or sleep_seconds < 0
    ):
        raise ValueError("sleep_seconds는 0 이상이어야 합니다.")
    VLMService._validate_timeout(timeout_seconds)
    if not image_path.is_file():
        raise FileNotFoundError(f"이미지 파일을 찾을 수 없습니다: {image_path}")
    validate_metadata(metadata)

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
    start_time = time.perf_counter()

    infer_method = service.infer_safe if safe else service.infer
    for index in range(warmup_runs):
        if safe:
            infer_method(image_path, metadata, timeout_seconds=timeout_seconds)
        else:
            infer_method(image_path, metadata)
        if sleep_seconds > 0 and (index + 1 < warmup_runs or runs > 0):
            time.sleep(sleep_seconds)

    latencies: list[float] = []
    memory_values: list[float] = []
    results: list[dict[str, Any]] = []
    status_counts: Counter[str] = Counter()
    error_counts: Counter[str] = Counter()
    fallback_count = 0

    for index in range(runs):
        if safe:
            result = infer_method(
                image_path, metadata, timeout_seconds=timeout_seconds
            )
        else:
            result = infer_method(image_path, metadata)

        status = str(result.get("service_status", "ok"))
        error_value = result.get("error")
        error_code = (
            str(error_value.get("code"))
            if isinstance(error_value, dict) and error_value.get("code")
            else None
        )
        latency = round(float(result["latency_ms"]), 2)
        memory = round(float(result.get("peak_gpu_memory_mb", 0.0)), 2)
        status_counts[status] += 1
        if error_code is not None:
            error_counts[error_code] += 1
        fallback_count += int(result.get("used_fallback") is True)
        latencies.append(latency)
        memory_values.append(memory)
        results.append(
            {
                "run_index": index + 1,
                "service_status": status,
                "error_code": error_code,
                "latency_ms": latency,
                "peak_gpu_memory_mb": memory,
                "used_fallback": bool(result.get("used_fallback", False)),
                "message_source": result.get("message_source"),
                "status": result.get("status"),
                "message": result.get("message"),
                "validation_reasons": result.get("validation_reasons", []),
            }
        )
        if stop_on_error and status != "ok":
            break
        if sleep_seconds > 0 and index + 1 < runs:
            time.sleep(sleep_seconds)

    total_benchmark_ms = round((time.perf_counter() - start_time) * 1000, 2)
    final_service_status = (
        "unavailable"
        if status_counts["unavailable"]
        else "degraded" if status_counts["degraded"] else "ok"
    )
    report = {
        "benchmark_version": 1,
        "benchmark_type": "stability",
        "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
        "environment": config["environment"],
        "label": label,
        "config_path": display_path(config_path),
        "metadata_path": display_path(metadata_path),
        "image_path": display_path(image_path),
        "model_id": config["model_id"],
        "device": config["device"],
        "runs_requested": runs,
        "runs_completed": len(results),
        "warmup_runs": warmup_runs,
        "success_count": status_counts["ok"],
        "degraded_count": status_counts["degraded"],
        "unavailable_count": status_counts["unavailable"],
        "fallback_count": fallback_count,
        "fallback_ratio": round(fallback_count / len(results), 4),
        "error_counts": dict(error_counts),
        "latencies_ms": latencies,
        "latency_stats_ms": rounded_stats(latencies),
        "latency_drift_ms": drift_stats(latencies, include_percent=True),
        "gpu_memory_mb": {
            "values": memory_values,
            "max": round(max(memory_values), 2),
            "mean": round(statistics.mean(memory_values), 2),
            "drift": drift_stats(memory_values, include_percent=False),
        },
        "final_service_status": final_service_status,
        "results": results,
        "total_benchmark_ms": total_benchmark_ms,
    }
    save_report(output_path, report)
    return report


def main(argv: list[str] | None = None) -> int:
    try:
        args = parse_args(argv)
        config = load_config(args.config)
        metadata = load_metadata(args.metadata)
        report = run_stability_benchmark(
            config=config,
            config_path=args.config,
            metadata=metadata,
            metadata_path=args.metadata,
            image_path=args.image,
            runs=args.runs,
            warmup_runs=args.warmup_runs,
            sleep_seconds=args.sleep_seconds,
            timeout_seconds=args.timeout_seconds,
            safe=args.safe,
            label=args.label,
            output_path=args.output,
            stop_on_error=args.stop_on_error,
        )
    except VLMModelLoadError as error:
        print(f"모델 로딩 오류: {error}", file=sys.stderr)
        return 3
    except (FileNotFoundError, ConfigValidationError, MetadataValidationError, ValueError) as error:
        print(f"입력 오류: {error}", file=sys.stderr)
        return 2
    except VLMError as error:
        print(f"추론 오류: {error}", file=sys.stderr)
        return 4
    except ReportSaveError as error:
        print(f"결과 저장 오류: {error}", file=sys.stderr)
        return 5
    except OSError as error:
        print(f"결과 저장 오류: {error}", file=sys.stderr)
        return 5

    summary_keys = (
        "runs_requested",
        "runs_completed",
        "success_count",
        "degraded_count",
        "unavailable_count",
        "fallback_count",
        "error_counts",
        "latency_stats_ms",
        "latency_drift_ms",
        "gpu_memory_mb",
        "final_service_status",
        "total_benchmark_ms",
    )
    print(json.dumps({key: report[key] for key in summary_keys}, ensure_ascii=False, indent=2))
    print(f"\n결과 저장 위치: {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
