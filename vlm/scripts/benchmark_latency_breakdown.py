from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.benchmark_warm import load_metadata, positive_int, nonnegative_int, save_report
from src.config_loader import load_config
from src.latency_experiments import load_experiment_catalog
from src.latency_profiler import DetailedVLMRunner, run_latency_benchmark
from src.runtime_inspector import collect_jetson_runtime, inspect_model_state


DEFAULT_SAMPLES = (
    ("elevator", ROOT / "samples/elevator_button.json", ROOT / "samples/elevator_button.png"),
    ("escalator", ROOT / "samples/escalator.json", ROOT / "samples/escalator.png"),
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 5 VLM latency breakdown benchmark (production config unchanged)"
    )
    parser.add_argument("--config", type=Path, default=ROOT / "config/jetson.json")
    parser.add_argument(
        "--experiments",
        type=Path,
        default=ROOT / "config/latency_experiments.json",
    )
    parser.add_argument("--variant", default="baseline")
    parser.add_argument("--runs", type=positive_int, default=20)
    parser.add_argument("--warmup-runs", type=nonnegative_int, default=2)
    parser.add_argument(
        "--sample",
        nargs=3,
        action="append",
        metavar=("LABEL", "METADATA", "IMAGE"),
        help="repeatable; defaults to elevator and escalator samples",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "reports/latency_optimization/baseline.json",
    )
    return parser.parse_args(argv)


def resolve_samples(values: list[list[str]] | None) -> list[tuple[str, Path, dict[str, Any]]]:
    configured = values or [
        [label, str(metadata), str(image)] for label, metadata, image in DEFAULT_SAMPLES
    ]
    samples = []
    labels: set[str] = set()
    for label, metadata_value, image_value in configured:
        if label in labels:
            raise ValueError(f"sample label이 중복되었습니다: {label}")
        labels.add(label)
        metadata_path = Path(metadata_value)
        image_path = Path(image_value)
        samples.append((label, image_path, load_metadata(metadata_path)))
    return samples


def build_report(
    *,
    args: argparse.Namespace,
    config: dict[str, Any],
    options: Any,
    runner: Any,
    benchmark: dict[str, Any],
    runtime: dict[str, Any],
    model_load_ms: float,
) -> dict[str, Any]:
    return {
        "benchmark_version": 2,
        "schema": "viassist.phase5.latency_breakdown.v1",
        "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
        "variant": options.name,
        "experimental_only": True,
        "production_config_modified": False,
        "config_path": str(args.config),
        "experiment_config_path": str(args.experiments),
        "model_id": config["model_id"],
        "device": runner.device,
        "runs": args.runs,
        "warmup_runs": args.warmup_runs,
        "model_load_ms": round(model_load_ms, 2),
        "options": options.__dict__,
        "generation_contract": {"do_sample": False, "num_beams": 1},
        "processor_api": {
            "transformers_4_49_verified_option": "Idefics3ImageProcessor.do_image_splitting",
            "supported": hasattr(runner.processor.image_processor, "do_image_splitting"),
        },
        "runtime": runtime,
        "model_state": inspect_model_state(runner.model, runner.torch),
        **benchmark,
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    runtime: dict[str, Any] | None = None
    try:
        config = load_config(args.config)
        variants = load_experiment_catalog(args.experiments)
        if args.variant not in variants:
            raise ValueError(
                f"알 수 없는 variant입니다: {args.variant}; 선택: {', '.join(sorted(variants))}"
            )
        options = variants[args.variant]
        samples = resolve_samples(args.sample)
        runtime = collect_jetson_runtime()
        load_start = time.perf_counter()
        runner = DetailedVLMRunner(config, options)
        model_load_ms = (time.perf_counter() - load_start) * 1000
        benchmark = run_latency_benchmark(
            samples=samples,
            runs=args.runs,
            warmup_runs=args.warmup_runs,
            runner=runner,
            progress=lambda message: print(f"[benchmark] {message}", flush=True),
        )
        report = build_report(
            args=args,
            config=config,
            options=options,
            runner=runner,
            benchmark=benchmark,
            runtime=runtime,
            model_load_ms=model_load_ms,
        )
        save_report(args.output, report)
    except Exception as error:
        failure_report = {
            "benchmark_version": 2,
            "schema": "viassist.phase5.latency_breakdown.v1",
            "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
            "variant": args.variant,
            "experimental_only": True,
            "production_config_modified": False,
            "status": "failed",
            "error": {"type": type(error).__name__, "message": str(error)},
            "runtime": runtime,
            "samples": {},
        }
        try:
            save_report(args.output, failure_report)
        except Exception:
            pass
        print(f"benchmark 실패: {type(error).__name__}: {error}", file=sys.stderr)
        return 2
    print(json.dumps({
        "variant": report["variant"],
        "model_state": report["model_state"],
        "samples": {
            label: {
                "total_ms": value["timings_ms"]["total_ms"],
                "generate_ms": value["timings_ms"]["generate_ms"],
                "validator_pass_rate": value["validator_pass_rate"],
                "fallback_rate": value["fallback_rate"],
            }
            for label, value in report["samples"].items()
        },
    }, ensure_ascii=False, indent=2))
    print(f"결과 저장 위치: {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
