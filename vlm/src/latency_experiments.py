from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.metadata_schema import validate_metadata
from src.prompt_builder import DIRECTION_MAP, POSITION_MAP
from src.safety_rules import select_detection


@dataclass(frozen=True)
class ExperimentOptions:
    """Production defaults를 변경하지 않는 단일 benchmark variant 설정."""

    name: str = "baseline"
    max_new_tokens: int = 60
    image_longest_edge: int = 1024
    max_image_size: int = 512
    crop_enabled: bool = False
    crop_margin_ratio: float = 0.25
    crop_max_edge: int = 512
    compact_prompt: bool = False
    do_image_splitting: bool = True
    attention_implementation: str | None = None
    cache_implementation: str | None = None
    torch_compile: bool = False
    dtype: str = "baseline"


def validate_experiment_options(options: ExperimentOptions) -> None:
    for name in ("max_new_tokens", "image_longest_edge", "max_image_size", "crop_max_edge"):
        value = getattr(options, name)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name}는 양의 정수여야 합니다.")
    if not math.isfinite(options.crop_margin_ratio) or not 0 <= options.crop_margin_ratio <= 1:
        raise ValueError("crop_margin_ratio는 0 이상 1 이하이어야 합니다.")
    if options.attention_implementation not in {None, "eager", "sdpa", "flash_attention_2"}:
        raise ValueError("지원하지 않는 attention implementation입니다.")
    if options.cache_implementation not in {None, "dynamic", "static"}:
        raise ValueError("지원하지 않는 cache implementation입니다.")
    if options.dtype not in {"baseline", "fp16", "bf16", "int8", "nf4"}:
        raise ValueError("지원하지 않는 dtype/quantization variant입니다.")


def expanded_crop_box(
    bbox: list[float | int],
    image_width: int,
    image_height: int,
    margin_ratio: float,
) -> tuple[int, int, int, int]:
    """원본 frame bbox에 문맥 margin을 더한 뒤 image boundary로 clamp한다."""

    if len(bbox) != 4:
        raise ValueError("bbox는 좌표 4개여야 합니다.")
    if image_width <= 0 or image_height <= 0:
        raise ValueError("image 크기는 양수여야 합니다.")
    if not math.isfinite(margin_ratio) or margin_ratio < 0:
        raise ValueError("margin_ratio는 0 이상이어야 합니다.")
    x1, y1, x2, y2 = (float(value) for value in bbox)
    if x1 >= x2 or y1 >= y2:
        raise ValueError("bbox는 x1 < x2, y1 < y2여야 합니다.")
    margin_x = (x2 - x1) * margin_ratio
    margin_y = (y2 - y1) * margin_ratio
    return (
        max(0, int(math.floor(x1 - margin_x))),
        max(0, int(math.floor(y1 - margin_y))),
        min(image_width, int(math.ceil(x2 + margin_x))),
        min(image_height, int(math.ceil(y2 + margin_y))),
    )


def crop_target_image(
    image: Any,
    metadata: dict[str, Any],
    *,
    margin_ratio: float,
    max_edge: int,
) -> tuple[Any, dict[str, Any]]:
    """선택된 YOLO bbox로 VLM 입력만 crop하며 metadata 좌표/position은 보존한다."""

    validate_metadata(metadata)
    detection = select_detection(metadata)
    if detection is None:
        return image, {"applied": False, "reason": "no_selected_detection"}
    box = expanded_crop_box(
        detection["bbox"], image.width, image.height, margin_ratio
    )
    cropped = image.crop(box)
    original_crop_size = tuple(cropped.size)
    if max(cropped.size) > max_edge:
        cropped.thumbnail((max_edge, max_edge))
    return cropped, {
        "applied": True,
        "box_original_coordinates": list(box),
        "source_position": detection["position"],
        "source_bbox": list(detection["bbox"]),
        "crop_size_before_resize": list(original_crop_size),
        "vlm_image_size": list(cropped.size),
        "margin_ratio": margin_ratio,
        "max_edge": max_edge,
    }


def build_compact_prompt(metadata: dict[str, Any]) -> str:
    """검증된 필드만 전달하되 baseline의 핵심 Safety 지시는 유지한다."""

    validate_metadata(metadata)
    detection = select_detection(metadata)
    motion = metadata["motion"]
    if detection is None:
        target = "unknown"
        position = "unknown"
    else:
        target = str(detection["class_name"])
        position = str(detection["position"])
    direction = str(motion["direction"]) if motion["available"] else "none"
    blur = str(bool(metadata["image_quality"]["is_blurry"])).lower()
    facts = (
        f"TARGET={target}\n"
        f"POSITION={position}({POSITION_MAP.get(position, '알 수 없음')})\n"
        f"MOTION={direction}({DIRECTION_MAP.get(direction, '없음')})\n"
        f"BLUR={blur}"
    )
    return f"""시각장애인 보행 안내를 한국어 평서형 한 문장으로 작성하세요.
검증 사실:
{facts}
규칙: 객체와 위치는 검증 사실만 사용하세요. 움직임 방향은 MOTION이 none이 아닐 때만 사용하고, 에스컬레이터 방향이면 반드시 포함하세요. 거리·안전 여부·미확인 객체나 상태를 만들지 마세요. 탑승·통과·버튼 누르기 같은 행동을 지시하지 마세요. 질문·JSON·설명 없이 자연스러운 '~있습니다.' 안내만 출력하세요."""


def load_experiment_catalog(path: Path) -> dict[str, ExperimentOptions]:
    import json

    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not isinstance(value.get("variants"), dict):
        raise ValueError("experiment config에는 variants 객체가 필요합니다.")
    variants: dict[str, ExperimentOptions] = {}
    for name, overrides in value["variants"].items():
        if not isinstance(overrides, dict):
            raise ValueError(f"variant {name}은 객체여야 합니다.")
        options = ExperimentOptions(name=name, **overrides)
        validate_experiment_options(options)
        variants[name] = options
    return variants
