from __future__ import annotations

from typing import Any

from src.metadata_schema import MetadataValidationError, validate_metadata
from src.quality_checker import build_safe_fallback, validate_vlm_message
from src.safety_rules import CONFIDENCE_THRESHOLD, select_detection


def confidence_to_label(value: float) -> str:
    if value >= 0.8:
        return "high"

    if value >= 0.5:
        return "medium"

    return "low"


def build_result(
    message: str,
    metadata: dict[str, Any],
    latency_ms: float,
    peak_gpu_memory_mb: float,
    model_id: str,
    selected_detection: dict[str, Any] | None = None,
) -> dict[str, Any]:
    metadata_is_valid = True
    try:
        validate_metadata(metadata)
    except MetadataValidationError:
        metadata_is_valid = False

    if metadata_is_valid and selected_detection is None:
        selected_detection = select_detection(metadata)

    validation = validate_vlm_message(
        message=message,
        metadata=metadata,
        selected_detection=selected_detection,
        confidence_threshold=CONFIDENCE_THRESHOLD,
    )
    final_message = validation.normalized_message
    if not validation.is_valid:
        final_message = build_safe_fallback(
            metadata=metadata,
            selected_detection=selected_detection,
            reason=validation.fallback_reason or "validation_failed",
        )

    detections_value = metadata.get("detections", [])
    detections = detections_value if isinstance(detections_value, list) else []
    image_quality_value = metadata.get("image_quality", {})
    image_quality = (
        image_quality_value if isinstance(image_quality_value, dict) else {}
    )
    if selected_detection is None:
        confidence_value = 0.0
        target = "unknown"
        position = "unknown"
    else:
        confidence_value = float(selected_detection["confidence"])
        target = str(selected_detection["class_name"])
        position = str(selected_detection["position"])

    if not metadata_is_valid or image_quality.get("is_blurry") is True:
        status = "uncertain"
    elif not detections or selected_detection is None:
        status = "not_found"
    elif confidence_value < CONFIDENCE_THRESHOLD:
        status = "uncertain"
    else:
        status = "detected"

    return {
        "message": final_message,
        "target": target,
        "position": position,
        "status": status,
        "confidence": confidence_to_label(confidence_value),
        "detection_confidence": (
            confidence_value if selected_detection is not None else None
        ),
        "latency_ms": latency_ms,
        "peak_gpu_memory_mb": peak_gpu_memory_mb,
        "model_id": model_id,
        "safety_validated": True,
        "used_fallback": validation.used_fallback,
        "validation_reasons": validation.reasons,
        "message_source": validation.message_source,
        "raw_vlm_message": message,
    }
