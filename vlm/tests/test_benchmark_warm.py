from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.benchmark_warm import parse_args, run_benchmark


ROOT = Path(__file__).resolve().parents[1]


def test_config() -> dict:
    return {
        "environment": "test",
        "device": "cpu",
        "max_new_tokens": 10,
        "use_mock_model": True,
        "model_id": "fake-model",
        "image_longest_edge": 1024,
        "max_image_size": 512,
    }


def metadata() -> dict:
    return json.loads(
        (ROOT / "samples" / "elevator_button.json").read_text(encoding="utf-8")
    )


class FakeService:
    def __init__(self, results: list[dict]) -> None:
        self.results = iter(results)
        self.calls = 0

    def infer(self, image_path: Path, input_metadata: dict) -> dict:
        self.calls += 1
        return next(self.results)


def fake_result(
    latency: float,
    memory: float,
    *,
    fallback: bool = False,
) -> dict:
    return {
        "latency_ms": latency,
        "peak_gpu_memory_mb": memory,
        "used_fallback": fallback,
        "message_source": "fallback" if fallback else "vlm",
        "message": "오른쪽에 엘리베이터 버튼이 있습니다.",
        "raw_vlm_message": "raw",
        "status": "detected",
        "validation_reasons": ["question_form_output"] if fallback else [],
    }


class WarmBenchmarkTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.output_path = Path(self.temporary_directory.name) / "nested" / "report.json"
        self.image_path = ROOT / "samples" / "elevator_button.png"
        self.metadata_path = ROOT / "samples" / "elevator_button.json"

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def run_with(self, service: FakeService, *, runs: int, warmup_runs: int) -> dict:
        factory_calls = []

        def factory(config: dict) -> FakeService:
            factory_calls.append(config)
            return service

        report = run_benchmark(
            config=test_config(),
            config_path=ROOT / "config" / "pc.json",
            metadata=metadata(),
            metadata_path=self.metadata_path,
            image_path=self.image_path,
            runs=runs,
            warmup_runs=warmup_runs,
            output_path=self.output_path,
            label="elevator_button",
            sleep_seconds=0.0,
            service_factory=factory,
        )
        self.assertEqual(len(factory_calls), 1)
        return report

    def test_persistent_service_statistics_and_json_report(self) -> None:
        service = FakeService(
            [
                fake_result(99.0, 1.0),
                fake_result(10.0, 100.0),
                fake_result(20.0, 110.0, fallback=True),
                fake_result(30.0, 120.0),
            ]
        )
        report = self.run_with(service, runs=3, warmup_runs=1)

        self.assertEqual(service.calls, 4)
        self.assertEqual(report["warmup_latencies_ms"], [99.0])
        self.assertEqual(report["inference_latencies_ms"], [10.0, 20.0, 30.0])
        self.assertEqual(
            report["latency_stats_ms"],
            {"mean": 20.0, "median": 20.0, "min": 10.0, "max": 30.0, "stdev": 10.0},
        )
        self.assertEqual(report["peak_gpu_memory_mb"]["mean"], 110.0)
        self.assertEqual(report["fallback_count"], 1)
        self.assertEqual(report["vlm_output_count"], 2)
        self.assertEqual(report["success_count"], 3)
        self.assertEqual(report["failure_count"], 0)
        self.assertTrue(self.output_path.is_file())
        self.assertEqual(
            json.loads(self.output_path.read_text(encoding="utf-8")), report
        )

    def test_single_run_stdev_is_zero(self) -> None:
        report = self.run_with(
            FakeService([fake_result(12.34, 5.0)]), runs=1, warmup_runs=0
        )
        self.assertEqual(report["latency_stats_ms"]["stdev"], 0.0)

    def test_run_benchmark_rejects_invalid_numeric_values(self) -> None:
        cases = (
            {"runs": 0, "warmup_runs": 0, "sleep_seconds": 0.0},
            {"runs": 1, "warmup_runs": -1, "sleep_seconds": 0.0},
            {"runs": 1, "warmup_runs": 0, "sleep_seconds": -0.1},
        )
        for values in cases:
            with self.subTest(values=values), self.assertRaises(ValueError):
                run_benchmark(
                    config=test_config(),
                    config_path=ROOT / "config" / "pc.json",
                    metadata=metadata(),
                    metadata_path=self.metadata_path,
                    image_path=self.image_path,
                    output_path=self.output_path,
                    label="",
                    service_factory=lambda config: FakeService([]),
                    **values,
                )

    def test_cli_rejects_invalid_numeric_values(self) -> None:
        required = [
            "--config", "config/pc.json",
            "--metadata", "samples/elevator_button.json",
            "--image", "samples/elevator_button.png",
        ]
        for option, value in (
            ("--runs", "0"),
            ("--warmup-runs", "-1"),
            ("--sleep-seconds", "-0.1"),
        ):
            with self.subTest(option=option), self.assertRaises(SystemExit):
                parse_args(required + [option, value])


if __name__ == "__main__":
    unittest.main()
