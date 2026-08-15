from __future__ import annotations

import statistics
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from src.latency_experiments import (
    ExperimentOptions,
    build_compact_prompt,
    crop_target_image,
    validate_experiment_options,
)
from src.metadata_schema import validate_metadata
from src.prompt_builder import build_prompt
from src.result_parser import build_result
from src.safety_rules import select_detection


PROFILE_TIMING_KEYS = (
    "image_load_ms",
    "image_preprocess_ms",
    "prompt_build_ms",
    "processor_ms",
    "host_to_device_ms",
    "generate_ms",
    "first_token_ms",
    "approx_decode_ms",
    "decode_ms",
    "validation_ms",
    "total_ms",
)


def percentile_95(values: list[float]) -> float:
    if not values:
        raise ValueError("통계를 계산할 값이 없습니다.")
    ordered = sorted(values)
    rank = 0.95 * (len(ordered) - 1)
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = rank - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def latency_stats(values: list[float]) -> dict[str, float]:
    if not values:
        raise ValueError("통계를 계산할 값이 없습니다.")
    return {
        "mean": round(statistics.mean(values), 2),
        "median": round(statistics.median(values), 2),
        "min": round(min(values), 2),
        "max": round(max(values), 2),
        "p95": round(percentile_95(values), 2),
        "stdev": round(statistics.stdev(values), 2) if len(values) > 1 else 0.0,
    }


def validate_profile_record(record: dict[str, Any]) -> None:
    required = {
        *PROFILE_TIMING_KEYS,
        "input_ids_shape",
        "input_token_count",
        "pixel_values_shape",
        "pixel_values_dtype",
        "model_dtype",
        "generated_token_count",
        "max_new_tokens",
        "attention_implementation",
        "use_cache",
        "cache_implementation",
        "tokens_per_second",
        "gpu_allocated_mb",
        "gpu_reserved_mb",
        "gpu_peak_mb",
    }
    missing = sorted(required - record.keys())
    if missing:
        raise ValueError(f"profiler 결과 필드가 누락되었습니다: {', '.join(missing)}")
    for key in PROFILE_TIMING_KEYS:
        value = record[key]
        if not isinstance(value, (int, float)) or value < 0:
            raise ValueError(f"{key}는 0 이상의 숫자여야 합니다.")


class DetailedVLMRunner:
    """Jetson 전용 상세 측정 runner. Production VLMService 흐름은 변경하지 않는다."""

    def __init__(self, config: dict[str, Any], options: ExperimentOptions) -> None:
        validate_experiment_options(options)
        try:
            import torch
            from transformers import (
                AutoConfig,
                AutoModelForImageTextToText,
                AutoProcessor,
            )
        except ImportError as error:
            raise RuntimeError("VLM profiler 의존성을 불러올 수 없습니다.") from error

        self.torch = torch
        self.options = options
        self.model_id = str(config["model_id"])
        requested_device = str(config["device"])
        if requested_device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("요청한 CUDA 장치를 사용할 수 없습니다.")
        self.device = (
            "cuda"
            if requested_device == "cuda"
            or (requested_device == "auto" and torch.cuda.is_available())
            else "cpu"
        )

        baseline_dtype = torch.float16 if self.device == "cuda" else torch.float32
        dtype_map = {
            "baseline": baseline_dtype,
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
            "int8": torch.float16,
            "nf4": torch.float16,
        }
        self.dtype = dtype_map[options.dtype]
        if self.device == "cpu" and options.dtype in {"fp16", "int8", "nf4"}:
            raise RuntimeError(f"{options.dtype} 실험은 CUDA 환경에서만 지원합니다.")
        if options.dtype == "bf16" and self.device == "cuda" and not torch.cuda.is_bf16_supported():
            raise RuntimeError("현재 CUDA 장치는 BF16을 지원하지 않습니다.")

        if options.cache_implementation == "static":
            model_config = AutoConfig.from_pretrained(self.model_id)
            model_class = AutoModelForImageTextToText._model_mapping[type(model_config)]
            if not getattr(model_class, "_supports_static_cache", False):
                raise RuntimeError(
                    "Transformers 4.49의 현재 Idefics3 모델은 StaticCache를 지원하지 않습니다."
                )

        self.processor = AutoProcessor.from_pretrained(self.model_id)
        image_processor = self.processor.image_processor
        if not hasattr(image_processor, "do_image_splitting"):
            raise RuntimeError("설치된 processor가 do_image_splitting을 지원하지 않습니다.")
        image_processor.size = {"longest_edge": options.image_longest_edge}
        image_processor.max_image_size = {"longest_edge": options.max_image_size}
        image_processor.do_image_splitting = options.do_image_splitting

        load_kwargs: dict[str, Any] = {
            "torch_dtype": self.dtype,
            "low_cpu_mem_usage": True,
        }
        if options.attention_implementation is not None:
            load_kwargs["attn_implementation"] = options.attention_implementation
        quantized = options.dtype in {"int8", "nf4"}
        if quantized:
            try:
                import bitsandbytes  # noqa: F401
                from transformers import BitsAndBytesConfig
            except ImportError as error:
                raise RuntimeError(
                    f"{options.dtype} 실험에는 기존에 설치된 bitsandbytes가 필요합니다. "
                    "profiler는 이를 자동 설치하지 않습니다."
                ) from error
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_8bit=options.dtype == "int8",
                load_in_4bit=options.dtype == "nf4",
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
            )
            load_kwargs["device_map"] = {"": self.device}

        self.model = AutoModelForImageTextToText.from_pretrained(
            self.model_id, **load_kwargs
        )
        if not quantized:
            self.model.to(self.device)
        self.model.eval()
        if options.torch_compile:
            if not hasattr(self.model, "compile"):
                raise RuntimeError("현재 PyTorch model.compile을 사용할 수 없습니다.")
            self.model.compile()

    def _sync(self) -> None:
        if self.device == "cuda":
            self.torch.cuda.synchronize()

    def _memory(self) -> dict[str, float]:
        if self.device != "cuda":
            return {
                "gpu_allocated_mb": 0.0,
                "gpu_reserved_mb": 0.0,
                "gpu_peak_mb": 0.0,
            }
        return {
            "gpu_allocated_mb": round(
                self.torch.cuda.memory_allocated() / 1024**2, 2
            ),
            "gpu_reserved_mb": round(
                self.torch.cuda.memory_reserved() / 1024**2, 2
            ),
            "gpu_peak_mb": round(
                self.torch.cuda.max_memory_allocated() / 1024**2, 2
            ),
        }

    def infer_profiled(
        self, image_path: Path, metadata: dict[str, Any]
    ) -> dict[str, Any]:
        from PIL import Image
        from transformers import StoppingCriteria, StoppingCriteriaList

        validate_metadata(metadata)
        total_start = time.perf_counter()

        stage_start = time.perf_counter()
        with Image.open(image_path) as source_image:
            image = source_image.convert("RGB")
            image.load()
        image_load_ms = (time.perf_counter() - stage_start) * 1000

        stage_start = time.perf_counter()
        crop_info: dict[str, Any] = {"applied": False, "reason": "disabled"}
        if self.options.crop_enabled:
            image, crop_info = crop_target_image(
                image,
                metadata,
                margin_ratio=self.options.crop_margin_ratio,
                max_edge=self.options.crop_max_edge,
            )
        image_preprocess_ms = (time.perf_counter() - stage_start) * 1000

        stage_start = time.perf_counter()
        prompt = (
            build_compact_prompt(metadata)
            if self.options.compact_prompt
            else build_prompt(metadata)
        )
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        chat_text = self.processor.apply_chat_template(
            messages, add_generation_prompt=True
        )
        prompt_build_ms = (time.perf_counter() - stage_start) * 1000

        stage_start = time.perf_counter()
        inputs = self.processor(
            text=chat_text, images=[image], return_tensors="pt"
        )
        processor_ms = (time.perf_counter() - stage_start) * 1000
        input_ids_shape = list(inputs["input_ids"].shape)
        input_token_count = int(inputs["input_ids"].shape[-1])
        pixel_values = inputs.get("pixel_values")
        pixel_values_shape = list(pixel_values.shape) if pixel_values is not None else []
        pixel_values_dtype = str(pixel_values.dtype) if pixel_values is not None else "none"

        if self.device == "cuda":
            self.torch.cuda.reset_peak_memory_stats()
        self._sync()
        stage_start = time.perf_counter()
        inputs = {
            key: value.to(self.device) if hasattr(value, "to") else value
            for key, value in inputs.items()
        }
        self._sync()
        host_to_device_ms = (time.perf_counter() - stage_start) * 1000

        runner = self

        class FirstTokenTimer(StoppingCriteria):
            first_token_ms: float | None = None

            def __call__(self, input_ids: Any, scores: Any, **kwargs: Any) -> Any:
                if self.first_token_ms is None:
                    runner._sync()
                    self.first_token_ms = (time.perf_counter() - generate_start) * 1000
                return runner.torch.zeros(
                    input_ids.shape[0], dtype=runner.torch.bool, device=input_ids.device
                )

        timer = FirstTokenTimer()
        generation_kwargs: dict[str, Any] = {
            "max_new_tokens": self.options.max_new_tokens,
            "do_sample": False,
            "num_beams": 1,
            "use_cache": True,
            "stopping_criteria": StoppingCriteriaList([timer]),
        }
        if self.options.cache_implementation not in {None, "dynamic"}:
            generation_kwargs["cache_implementation"] = self.options.cache_implementation
        self._sync()
        generate_start = time.perf_counter()
        with self.torch.inference_mode():
            generated_ids = self.model.generate(**inputs, **generation_kwargs)
        self._sync()
        generate_ms = (time.perf_counter() - generate_start) * 1000

        input_length = int(inputs["input_ids"].shape[1])
        output_ids = generated_ids[:, input_length:]
        generated_token_count = int(output_ids.shape[1])
        first_token_ms = timer.first_token_ms or generate_ms
        approx_decode_ms = max(0.0, generate_ms - first_token_ms)

        stage_start = time.perf_counter()
        message = self.processor.batch_decode(
            output_ids, skip_special_tokens=True
        )[0].strip()
        decode_ms = (time.perf_counter() - stage_start) * 1000

        stage_start = time.perf_counter()
        selected_detection = select_detection(metadata)
        result = build_result(
            message=message,
            metadata=metadata,
            latency_ms=0.0,
            peak_gpu_memory_mb=self._memory()["gpu_peak_mb"],
            model_id=self.model_id,
            selected_detection=selected_detection,
        )
        validation_ms = (time.perf_counter() - stage_start) * 1000
        total_ms = (time.perf_counter() - total_start) * 1000
        result["latency_ms"] = round(total_ms, 2)
        result.update({"service_status": "ok", "error": None, "fallback_reason": None})

        config = getattr(self.model, "config", None)
        profile: dict[str, Any] = {
            "image_load_ms": round(image_load_ms, 2),
            "image_preprocess_ms": round(image_preprocess_ms, 2),
            "prompt_build_ms": round(prompt_build_ms, 2),
            "processor_ms": round(processor_ms, 2),
            "host_to_device_ms": round(host_to_device_ms, 2),
            "generate_ms": round(generate_ms, 2),
            "first_token_ms": round(first_token_ms, 2),
            "approx_decode_ms": round(approx_decode_ms, 2),
            "decode_ms": round(decode_ms, 2),
            "validation_ms": round(validation_ms, 2),
            "total_ms": round(total_ms, 2),
            "input_ids_shape": input_ids_shape,
            "input_token_count": input_token_count,
            "pixel_values_shape": pixel_values_shape,
            "pixel_values_dtype": pixel_values_dtype,
            "model_dtype": str(self.dtype),
            "generated_token_count": generated_token_count,
            "max_new_tokens": self.options.max_new_tokens,
            "attention_implementation": getattr(
                config, "_attn_implementation", "unknown"
            ),
            "use_cache": True,
            "cache_implementation": self.options.cache_implementation or "dynamic_default",
            "tokens_per_second": round(
                generated_token_count / (generate_ms / 1000), 2
            ) if generate_ms > 0 else 0.0,
            "crop": crop_info,
            "processor_settings": {
                "size": self.processor.image_processor.size,
                "max_image_size": self.processor.image_processor.max_image_size,
                "do_image_splitting": self.processor.image_processor.do_image_splitting,
            },
            **self._memory(),
        }
        validate_profile_record(profile)
        return {"profile": profile, "result": result}


def aggregate_sample_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    profiles = [record["profile"] for record in records]
    results = [record["result"] for record in records]
    timings = {
        key: latency_stats([float(profile[key]) for profile in profiles])
        for key in PROFILE_TIMING_KEYS
    }
    return {
        "runs": len(records),
        "timings_ms": timings,
        "gpu_peak_mb": latency_stats(
            [float(profile["gpu_peak_mb"]) for profile in profiles]
        ),
        "input_tokens": latency_stats(
            [float(profile["input_token_count"]) for profile in profiles]
        ),
        "generated_tokens": latency_stats(
            [float(profile["generated_token_count"]) for profile in profiles]
        ),
        "visual_shapes": sorted({str(profile["pixel_values_shape"]) for profile in profiles}),
        "validator_pass_count": sum(not result["used_fallback"] for result in results),
        "validator_pass_rate": round(
            sum(not result["used_fallback"] for result in results) / len(results), 4
        ),
        "fallback_count": sum(bool(result["used_fallback"]) for result in results),
        "fallback_rate": round(
            sum(bool(result["used_fallback"]) for result in results) / len(results), 4
        ),
        "errors": [],
        "records": records,
    }


def run_latency_benchmark(
    *,
    samples: list[tuple[str, Path, dict[str, Any]]],
    runs: int,
    warmup_runs: int,
    runner: Any,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    if runs < 1 or warmup_runs < 0:
        raise ValueError("runs는 1 이상, warmup_runs는 0 이상이어야 합니다.")
    if not samples:
        raise ValueError("sample이 하나 이상 필요합니다.")
    notify = progress or (lambda _: None)
    for label, image_path, metadata in samples:
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        validate_metadata(metadata)
        for index in range(warmup_runs):
            notify(f"warmup {label} {index + 1}/{warmup_runs}")
            runner.infer_profiled(image_path, metadata)

    sample_reports: dict[str, Any] = {}
    for label, image_path, metadata in samples:
        records = []
        for index in range(runs):
            notify(f"measure {label} {index + 1}/{runs}")
            records.append(runner.infer_profiled(image_path, metadata))
        sample_reports[label] = aggregate_sample_records(records)
    return {"samples": sample_reports}
