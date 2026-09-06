from __future__ import annotations

from pathlib import Path
from typing import Any

from src.exceptions import (
    VLMConfigurationError,
    VLMImageError,
    VLMInferenceError,
    VLMModelLoadError,
    VLMOutOfMemoryError,
)
from src.quality_checker import build_safe_fallback
from src.safety_rules import select_detection


class MockVLMEngine:
    """
    실제 모델 없이 입출력 파이프라인을 검증하는 임시 엔진.
    """

    def generate(
        self,
        image_path: Path | None,
        prompt: str,
        metadata: dict[str, Any],
    ) -> str:
        if metadata.get("mode") == "scene_description":
            return "주변에 사람이 있습니다."
        selected_detection = select_detection(metadata)
        return build_safe_fallback(
            metadata=metadata,
            selected_detection=selected_detection,
            reason="mock_generation",
        )


class SmolVLMEngine:
    """
    HuggingFaceTB/SmolVLM-500M-Instruct 기반 실제 VLM 엔진.
    """

    def __init__(
        self,
        model_id: str = "HuggingFaceTB/SmolVLM-500M-Instruct",
        max_new_tokens: int = 80,
        device: str = "auto",
        image_longest_edge: int = 1024,
        max_image_size: int = 512,
    ) -> None:
        try:
            import torch
            from transformers import (
                AutoModelForImageTextToText,
                AutoProcessor,
            )
        except ImportError as error:
            raise VLMModelLoadError(
                "VLM 모델 실행 환경을 초기화하지 못했습니다.",
                original_exception=error,
            ) from error

        self.torch = torch
        self.model_id = model_id
        self.max_new_tokens = max_new_tokens

        if device not in {"auto", "cuda", "cpu"}:
            raise VLMConfigurationError(
                "VLM device 설정이 올바르지 않습니다."
            )
        if device == "cuda" and not torch.cuda.is_available():
            raise VLMModelLoadError(
                "요청한 CUDA 장치를 사용할 수 없습니다."
            )

        self.device = (
            "cuda" if device == "cuda" or (
                device == "auto" and torch.cuda.is_available()
            ) else "cpu"
        )

        self.dtype = (
            torch.float16
            if self.device == "cuda"
            else torch.float32
        )

        print(f"[VLM] device: {self.device}")
        print(f"[VLM] model: {self.model_id}")

        try:
            self.processor = AutoProcessor.from_pretrained(
                self.model_id,
            )
        except Exception as error:
            raise VLMModelLoadError(
                "VLM processor를 초기화하지 못했습니다.",
                original_exception=error,
            ) from error

        if str(self.device).startswith("cuda"):
            self.processor.image_processor.size = {
                "longest_edge": image_longest_edge
            }
            self.processor.image_processor.max_image_size = {
                "longest_edge": max_image_size
            }
            # 타일 분할 없이 한 장만 넣는다. 토큰 수가 고정되어 지연이 예측 가능하다.
            self.processor.image_processor.do_image_splitting = False

            print(
                "[VLM] image processor:",
                f"size={self.processor.image_processor.size},",
                f"max_image_size={self.processor.image_processor.max_image_size}",
            )

        try:
            try:
                # eager attention 대신 SDPA 커널. 지원 안 되는 버전이면 아래 폴백.
                self.model = AutoModelForImageTextToText.from_pretrained(
                    self.model_id,
                    torch_dtype=self.dtype,
                    low_cpu_mem_usage=True,
                    attn_implementation="sdpa",
                )
                self.attn_implementation = "sdpa"
            except (ValueError, TypeError):
                self.model = AutoModelForImageTextToText.from_pretrained(
                    self.model_id,
                    torch_dtype=self.dtype,
                    low_cpu_mem_usage=True,
                )
                self.attn_implementation = "eager(fallback)"
            self.model.to(self.device)
            self.model.eval()
            print(f"[VLM] attention: {self.attn_implementation}")
        except Exception as error:
            raise VLMModelLoadError(
                "VLM 모델을 로드하지 못했습니다.",
                original_exception=error,
            ) from error

        # 4) 문장 끝 정지 기준. 두 문장이 끝나거나 줄바꿈이 나오면 멈춘다(안내는 1~2문장).
        self._stop_criteria = self._build_stop_criteria()

        # 5) 워밍업: 첫 호출의 커널 초기화(8~10초)를 기동 시간으로 옮긴다.
        self._warmup()

    def _build_stop_criteria(self):
        try:
            from transformers import StoppingCriteria, StoppingCriteriaList
        except Exception:  # noqa: BLE001
            return None

        tokenizer = getattr(self.processor, "tokenizer", None)
        if tokenizer is None:
            return None

        class _SentenceStop(StoppingCriteria):
            """생성된 부분만 디코드해 문장 종결 2개 또는 줄바꿈이 보이면 멈춘다."""

            def __init__(self, tok, prompt_len_holder):
                self.tok = tok
                self.holder = prompt_len_holder

            def __call__(self, input_ids, scores, **kwargs) -> bool:
                start = self.holder.get("prompt_len", 0)
                tail = input_ids[0, start:]
                if tail.shape[0] < 4:
                    return False
                text = self.tok.decode(tail, skip_special_tokens=True)
                if "\n" in text.strip():
                    return True
                return sum(text.count(ch) for ch in ".!?。") >= 2

        self._prompt_len_holder = {"prompt_len": 0}
        return StoppingCriteriaList([_SentenceStop(tokenizer, self._prompt_len_holder)])

    def _warmup(self) -> None:
        try:
            import io
            import time as _time
            from tempfile import TemporaryDirectory

            from PIL import Image

            started = _time.perf_counter()
            with TemporaryDirectory(prefix="viassist_vlm_warm_") as temp_dir:
                warm_path = Path(temp_dir) / "warm.jpg"
                Image.new("RGB", (256, 256), (40, 40, 40)).save(warm_path, format="JPEG")
                saved = self.max_new_tokens
                self.max_new_tokens = 4
                try:
                    self.generate(warm_path, "이 사진을 한 단어로 말하세요.", {})
                finally:
                    self.max_new_tokens = saved
            print(f"[VLM] warmup {(_time.perf_counter() - started) * 1000:.0f} ms")
        except Exception as error:  # noqa: BLE001 - 워밍업 실패는 치명적이지 않다
            print(f"[VLM] warmup skipped: {error}")

    def _raise_inference_error(self, error: Exception) -> None:
        oom_type = getattr(self.torch.cuda, "OutOfMemoryError", ())
        is_oom = (
            (oom_type and isinstance(error, oom_type))
            or "out of memory" in str(error).lower()
        )
        if is_oom:
            if self.device == "cuda":
                try:
                    self.torch.cuda.empty_cache()
                except Exception:
                    pass
            raise VLMOutOfMemoryError(
                "VLM 추론 메모리가 부족합니다.",
                original_exception=error,
            ) from error
        raise VLMInferenceError(
            "VLM 추론에 실패했습니다.",
            original_exception=error,
        ) from error

    def generate(
        self,
        image_path: Path | None,
        prompt: str,
        metadata: dict[str, Any],
    ) -> str:
        try:
            from PIL import Image
        except ImportError as error:
            raise VLMImageError(
                "이미지를 처리할 수 없습니다.",
                original_exception=error,
            ) from error

        if image_path is None:
            raise VLMImageError(
                "추론할 이미지가 제공되지 않았습니다.",
                error_code="IMAGE_NOT_FOUND",
            )

        if not image_path.exists():
            raise VLMImageError(
                "추론할 이미지를 찾을 수 없습니다.",
                error_code="IMAGE_NOT_FOUND",
            )

        try:
            with Image.open(image_path) as source_image:
                image = source_image.convert("RGB")
        except Exception as error:
            raise VLMImageError(
                "이미지를 열거나 변환할 수 없습니다.",
                original_exception=error,
            ) from error

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                    },
                    {
                        "type": "text",
                        "text": prompt,
                    },
                ],
            }
        ]

        try:
            chat_text = self.processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
            )

            inputs = self.processor(
                text=chat_text,
                images=[image],
                return_tensors="pt",
            )
        except Exception as error:
            self._raise_inference_error(error)

        pixel_values = inputs.get("pixel_values")
        if pixel_values is not None:
            print(
                "[VLM] pixel_values:",
                f"shape={tuple(pixel_values.shape)},",
                f"dtype={pixel_values.dtype},",
                f"device={pixel_values.device}",
            )

        try:
            inputs = {
                key: (
                    value.to(self.device)
                    if hasattr(value, "to")
                    else value
                )
                for key, value in inputs.items()
            }

            if self.device == "cuda":
                self.torch.cuda.reset_peak_memory_stats()
                self.torch.cuda.synchronize()

            holder = getattr(self, "_prompt_len_holder", None)
            if holder is not None:
                holder["prompt_len"] = int(inputs["input_ids"].shape[1])

            with self.torch.inference_mode():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    stopping_criteria=self._stop_criteria,
                )

            input_length = inputs["input_ids"].shape[1]

            generated_text = self.processor.batch_decode(
                generated_ids[:, input_length:],
                skip_special_tokens=True,
            )[0]
        except Exception as error:
            self._raise_inference_error(error)

        return generated_text.strip()

    def get_peak_memory_mb(self) -> float:
        if self.device != "cuda":
            return 0.0

        return round(
            self.torch.cuda.max_memory_allocated()
            / 1024**2,
            2,
        )
