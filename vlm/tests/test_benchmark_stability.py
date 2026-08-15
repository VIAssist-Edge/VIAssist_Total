from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.benchmark_stability import parse_args, run_stability_benchmark


ROOT = Path(__file__).resolve().parents[1]


def config() -> dict:
    return {
        "environment": "test",
        "device": "cpu",
        "model_id": "fake-model",
    }


def metadata() -> dict:
    return json.loads(
        (ROOT / "samples" / "elevator_button.json").read_text(encoding="utf-8")
    )


def result(
    latency: float,
    memory: float,
    *,
    status: str = "ok",
    fallback: bool = False,
    error_code: str | None = None,
) -> dict:
    return {
        "service_status": status,
        "error": (
            {"code": error_code, "message": "일반화된 오류", "retryable": True}
            if error_code
            else None
        ),
        "latency_ms": latency,
        "peak_gpu_memory_mb": memory,
        "used_fallback": fallback,
        "message_source": "fallback" if fallback else "vlm",
        "status": "detected",
        "message": "안내 메시지입니다.",
        "validation_reasons": [],
    }


class FakeService:
    def __init__(self, results: list[dict]) -> None:
        self.results = iter(results)
        self.calls = 0

    def infer_safe(self, image_path, input_metadata, *, timeout_seconds=None):
        self.calls += 1
        return next(self.results)

    def infer(self, image_path, input_metadata):
        self.calls += 1
        return next(self.results)


class StabilityBenchmarkTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.output = Path(self.tempdir.name) / "reports" / "stability.json"
        self.image = ROOT / "samples" / "elevator_button.png"
        self.metadata_path = ROOT / "samples" / "elevator_button.json"

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def run_with(
        self,
        service: FakeService,
        *,
        runs: int,
        warmup_runs: int = 1,
        stop_on_error: bool = False,
    ) -> dict:
        factory_calls = 0

        def factory(input_config):
            nonlocal factory_calls
            factory_calls += 1
            return service

        report = run_stability_benchmark(
            config=config(),
            config_path=ROOT / "config" / "pc.json",
            metadata=metadata(),
            metadata_path=self.metadata_path,
            image_path=self.image,
            runs=runs,
            warmup_runs=warmup_runs,
            sleep_seconds=0.0,
            timeout_seconds=1.0,
            safe=True,
            label="test",
            output_path=self.output,
            stop_on_error=stop_on_error,
            service_factory=factory,
        )
        self.assertEqual(factory_calls, 1)
        return report

    def test_statistics_drift_counts_and_json(self) -> None:
        service = FakeService(
            [
                result(1, 1),
                result(10, 100),
                result(
                    20, 110, status="degraded", fallback=True,
                    error_code="INFERENCE_TIMEOUT",
                ),
                result(
                    30, 140, status="unavailable", fallback=True,
                    error_code="SERVICE_UNAVAILABLE",
                ),
                result(40, 160),
            ]
        )
        report = self.run_with(service, runs=4)

        self.assertEqual(service.calls, 5)
        self.assertEqual(report["runs_completed"], 4)
        self.assertEqual(report["success_count"], 2)
        self.assertEqual(report["degraded_count"], 1)
        self.assertEqual(report["unavailable_count"], 1)
        self.assertEqual(report["fallback_count"], 2)
        self.assertEqual(
            report["error_counts"],
            {"INFERENCE_TIMEOUT": 1, "SERVICE_UNAVAILABLE": 1},
        )
        self.assertEqual(report["latency_stats_ms"]["mean"], 25.0)
        self.assertEqual(report["latency_drift_ms"]["difference"], 20.0)
        self.assertEqual(report["latency_drift_ms"]["percent"], 133.33)
        self.assertEqual(
            report["gpu_memory_mb"]["drift"]["difference"], 45.0
        )
        self.assertEqual(report["final_service_status"], "unavailable")
        self.assertTrue(self.output.is_file())
        self.assertEqual(json.loads(self.output.read_text(encoding="utf-8")), report)

    def test_stop_on_error_records_failing_run_then_stops(self) -> None:
        service = FakeService(
            [
                result(1, 1),
                result(10, 100),
                result(
                    20, 110, status="degraded", fallback=True,
                    error_code="INFERENCE_ERROR",
                ),
                result(30, 120),
            ]
        )
        report = self.run_with(service, runs=3, stop_on_error=True)
        self.assertEqual(service.calls, 3)
        self.assertEqual(report["runs_completed"], 2)
        self.assertEqual(report["degraded_count"], 1)

    def test_single_run_has_zero_stdev_and_zero_drift(self) -> None:
        service = FakeService([result(12, 50)])
        report = self.run_with(service, runs=1, warmup_runs=0)
        self.assertEqual(report["latency_stats_ms"]["stdev"], 0.0)
        self.assertEqual(report["latency_drift_ms"]["difference"], 0.0)
        self.assertEqual(report["gpu_memory_mb"]["drift"]["difference"], 0.0)

    def test_invalid_function_arguments(self) -> None:
        cases = (
            {"runs": 0, "warmup_runs": 0, "sleep_seconds": 0.0, "timeout_seconds": None},
            {"runs": 1, "warmup_runs": -1, "sleep_seconds": 0.0, "timeout_seconds": None},
            {"runs": 1, "warmup_runs": 0, "sleep_seconds": -1.0, "timeout_seconds": None},
            {"runs": 1, "warmup_runs": 0, "sleep_seconds": 0.0, "timeout_seconds": 0},
        )
        for values in cases:
            with self.subTest(values=values), self.assertRaises(ValueError):
                run_stability_benchmark(
                    config=config(),
                    config_path=ROOT / "config" / "pc.json",
                    metadata=metadata(),
                    metadata_path=self.metadata_path,
                    image_path=self.image,
                    safe=True,
                    label="",
                    output_path=self.output,
                    stop_on_error=False,
                    service_factory=lambda _: FakeService([]),
                    **values,
                )

    def test_cli_rejects_invalid_numeric_arguments(self) -> None:
        required = [
            "--config", "config/pc.json",
            "--metadata", "samples/elevator_button.json",
            "--image", "samples/elevator_button.png",
        ]
        for option, value in (
            ("--runs", "0"),
            ("--warmup-runs", "-1"),
            ("--sleep-seconds", "-1"),
            ("--timeout-seconds", "0"),
        ):
            with self.subTest(option=option), self.assertRaises(SystemExit):
                parse_args(required + [option, value])


if __name__ == "__main__":
    unittest.main()
