from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path
from types import SimpleNamespace

from PIL import Image

from scripts.summarize_latency_reports import choose_best, summarize_report
from src.latency_experiments import (
    build_compact_prompt,
    crop_target_image,
    expanded_crop_box,
    load_experiment_catalog,
)
from src.latency_profiler import (
    PROFILE_TIMING_KEYS,
    latency_stats,
    run_latency_benchmark,
    validate_profile_record,
)
from src.runtime_inspector import inspect_model_state


ROOT = Path(__file__).resolve().parents[1]


def load_sample(name: str) -> dict:
    return json.loads((ROOT / "samples" / name).read_text(encoding="utf-8"))


def valid_profile(**overrides):
    profile = {key: 1.0 for key in PROFILE_TIMING_KEYS}
    profile.update(
        {
            "input_ids_shape": [1, 10],
            "input_token_count": 10,
            "pixel_values_shape": [1, 1, 3, 512, 512],
            "pixel_values_dtype": "torch.float32",
            "model_dtype": "torch.float16",
            "generated_token_count": 5,
            "max_new_tokens": 24,
            "attention_implementation": "sdpa",
            "use_cache": True,
            "cache_implementation": "dynamic_default",
            "tokens_per_second": 10.0,
            "gpu_allocated_mb": 100.0,
            "gpu_reserved_mb": 120.0,
            "gpu_peak_mb": 110.0,
        }
    )
    profile.update(overrides)
    return profile


class LatencyExperimentConfigTest(unittest.TestCase):
    def setUp(self) -> None:
        self.catalog = load_experiment_catalog(
            ROOT / "config/latency_experiments.json"
        )

    def test_experiments_do_not_change_production_baseline(self) -> None:
        production = json.loads(
            (ROOT / "config/jetson.json").read_text(encoding="utf-8")
        )
        self.assertEqual(production["image_longest_edge"], 1024)
        self.assertEqual(production["max_image_size"], 512)
        self.assertEqual(production["max_new_tokens"], 60)
        self.assertNotIn("variant", production)

    def test_max_new_tokens_overrides_are_independent(self) -> None:
        self.assertEqual(self.catalog["baseline"].max_new_tokens, 60)
        self.assertEqual(self.catalog["tokens_32"].max_new_tokens, 32)
        self.assertEqual(self.catalog["tokens_24"].max_new_tokens, 24)
        self.assertEqual(self.catalog["tokens_16"].max_new_tokens, 16)
        self.assertEqual(self.catalog["baseline"].image_longest_edge, 1024)

    def test_image_splitting_and_attention_variants(self) -> None:
        self.assertTrue(self.catalog["baseline"].do_image_splitting)
        self.assertFalse(self.catalog["no_image_splitting"].do_image_splitting)
        self.assertIsNone(self.catalog["baseline"].attention_implementation)
        self.assertEqual(self.catalog["sdpa"].attention_implementation, "sdpa")
        self.assertEqual(self.catalog["eager"].attention_implementation, "eager")


class CropAndPromptTest(unittest.TestCase):
    def test_crop_margin_and_boundary_clamp(self) -> None:
        self.assertEqual(
            expanded_crop_box([0, 10, 20, 30], 100, 80, 0.25),
            (0, 5, 25, 35),
        )
        self.assertEqual(
            expanded_crop_box([80, 60, 100, 80], 100, 80, 0.25),
            (75, 55, 100, 80),
        )

    def test_crop_preserves_original_metadata_position_and_coordinates(self) -> None:
        metadata = load_sample("elevator_button.json")
        original = copy.deepcopy(metadata)
        with Image.open(ROOT / "samples/elevator_button.png") as source:
            image = source.convert("RGB")
        cropped, info = crop_target_image(
            image, metadata, margin_ratio=0.25, max_edge=512
        )
        self.assertTrue(info["applied"])
        self.assertEqual(info["source_position"], "right")
        self.assertEqual(info["source_bbox"], [563, 335, 596, 408])
        self.assertLessEqual(max(cropped.size), 512)
        self.assertEqual(metadata, original)
        self.assertEqual(metadata["detections"][0]["position"], "right")

    def test_compact_prompt_keeps_verified_facts_and_safety_rules(self) -> None:
        elevator = build_compact_prompt(load_sample("elevator_button.json"))
        self.assertIn("TARGET=elevator_button", elevator)
        self.assertIn("POSITION=right(오른쪽)", elevator)
        self.assertIn("MOTION=none", elevator)
        self.assertIn("거리·안전 여부", elevator)
        self.assertIn("행동을 지시하지 마세요", elevator)
        self.assertNotIn("563", elevator)

        escalator = build_compact_prompt(load_sample("escalator.json"))
        self.assertIn("TARGET=escalator", escalator)
        self.assertIn("POSITION=front(정면)", escalator)
        self.assertIn("MOTION=up(위쪽)", escalator)


class ProfilerSchemaTest(unittest.TestCase):
    def test_profile_schema_and_p95(self) -> None:
        validate_profile_record(valid_profile())
        stats = latency_stats([1, 2, 3, 4, 5])
        self.assertEqual(stats["median"], 3)
        self.assertEqual(stats["p95"], 4.8)
        broken = valid_profile()
        del broken["processor_ms"]
        with self.assertRaisesRegex(ValueError, "processor_ms"):
            validate_profile_record(broken)

    def test_benchmark_aggregates_validator_and_fallback_contract(self) -> None:
        class FakeRunner:
            def __init__(self) -> None:
                self.calls = 0

            def infer_profiled(self, image_path, metadata):
                self.calls += 1
                fallback = self.calls % 2 == 0
                return {
                    "profile": valid_profile(total_ms=float(self.calls)),
                    "result": {
                        "used_fallback": fallback,
                        "message_source": "fallback" if fallback else "vlm",
                        "validation_reasons": ["test"] if fallback else [],
                        "raw_vlm_message": "raw",
                        "message": "오른쪽에 엘리베이터 버튼이 있습니다.",
                        "service_status": "ok",
                    },
                }

        runner = FakeRunner()
        report = run_latency_benchmark(
            samples=[(
                "elevator",
                ROOT / "samples/elevator_button.png",
                load_sample("elevator_button.json"),
            )],
            runs=2,
            warmup_runs=1,
            runner=runner,
        )
        sample = report["samples"]["elevator"]
        self.assertEqual(runner.calls, 3)
        self.assertEqual(sample["validator_pass_count"], 1)
        self.assertEqual(sample["fallback_count"], 1)
        self.assertEqual(sample["timings_ms"]["total_ms"]["p95"], 2.95)

    def test_summary_rejects_faster_variant_with_fallback(self) -> None:
        def sample(total: float, fallback: bool) -> dict:
            record = {
                "profile": valid_profile(total_ms=total),
                "result": {"used_fallback": fallback},
            }
            return {
                "timings_ms": {"total_ms": {"mean": total, "p95": total}},
                "errors": [],
                "records": [record],
            }

        safe = summarize_report({
            "variant": "safe",
            "samples": {"elevator": sample(10, False), "escalator": sample(12, False)},
        })
        unsafe = summarize_report({
            "variant": "fast_but_fallback",
            "samples": {"elevator": sample(5, True), "escalator": sample(6, True)},
        })
        self.assertEqual(choose_best([unsafe, safe]), "safe")


class QuantizationInspectorTest(unittest.TestCase):
    class FakeParameter:
        dtype = "torch.float16"

        @staticmethod
        def numel():
            return 10

        @staticmethod
        def element_size():
            return 2

    class Linear4bit:
        pass

    class FakeModel:
        config = SimpleNamespace(
            quantization_config=SimpleNamespace(load_in_4bit=True),
            _attn_implementation="sdpa",
            use_cache=True,
        )
        generation_config = SimpleNamespace(use_cache=True, cache_implementation=None)

        @classmethod
        def named_parameters(cls):
            return iter((
                ("model.vision_model.layer", QuantizationInspectorTest.FakeParameter()),
                ("model.text_model.layer", QuantizationInspectorTest.FakeParameter()),
            ))

        @classmethod
        def modules(cls):
            return iter((cls(), QuantizationInspectorTest.Linear4bit()))

    class FakeCuda:
        @staticmethod
        def is_available():
            return False

    def test_inspector_uses_real_module_evidence(self) -> None:
        fake_torch = SimpleNamespace(cuda=self.FakeCuda())
        state = inspect_model_state(self.FakeModel(), fake_torch)
        self.assertEqual(state["classification"], "4-bit")
        self.assertEqual(state["linear4bit_count"], 1)
        self.assertTrue(state["bitsandbytes_config_present"])
        self.assertEqual(state["vision_model_dtypes"], ["float16"])
        self.assertEqual(state["language_model_dtypes"], ["float16"])
        self.assertTrue(state["model_config_use_cache"])


if __name__ == "__main__":
    unittest.main()
