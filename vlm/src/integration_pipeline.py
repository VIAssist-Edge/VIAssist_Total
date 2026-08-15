from __future__ import annotations

from pathlib import Path
from typing import Any

from src.exceptions import (
    MotionResultValidationError,
    PerceptionUnavailableError,
    YoloResultValidationError,
)
from src.integration_schema import (
    ImageQualityInput,
    MotionResultInput,
    YoloResultInput,
)
from src.metadata_adapter import DEFAULT_USER_QUERY, build_vlm_metadata
from src.perception_adapter import (
    PERCEPTION_UNAVAILABLE_MESSAGE,
    YOLO_UNAVAILABLE_MESSAGE,
    build_perception_metadata,
    build_perception_unavailable_result,
    is_flow_usable,
)
from src.vlm_service import VLMService


class VIAssistVLMPipeline:
    """외부 vision 결과를 기존 VLMService 계약에 연결한다."""

    def __init__(
        self,
        service: VLMService,
        *,
        strict_frame_sync: bool = False,
    ) -> None:
        self.service = service
        self.strict_frame_sync = strict_frame_sync

    def build_metadata(
        self,
        *,
        yolo_result: YoloResultInput | dict[str, Any],
        motion_result: MotionResultInput | dict[str, Any] | None = None,
        image_quality: ImageQualityInput | dict[str, Any] | None = None,
        user_query: str = DEFAULT_USER_QUERY,
    ) -> dict[str, Any]:
        return build_vlm_metadata(
            yolo_result=yolo_result,
            motion_result=motion_result,
            image_quality=image_quality,
            user_query=user_query,
            strict_frame_sync=self.strict_frame_sync,
        )

    def process(
        self,
        *,
        image_path: Path | None,
        yolo_result: YoloResultInput | dict[str, Any],
        motion_result: MotionResultInput | dict[str, Any] | None = None,
        image_quality: ImageQualityInput | dict[str, Any] | None = None,
        user_query: str = DEFAULT_USER_QUERY,
        safe: bool = True,
        timeout_seconds: float | None = None,
    ) -> dict[str, Any]:
        metadata = self.build_metadata(
            yolo_result=yolo_result,
            motion_result=motion_result,
            image_quality=image_quality,
            user_query=user_query,
        )
        if safe:
            return self.service.infer_safe(
                image_path,
                metadata,
                timeout_seconds=timeout_seconds,
            )
        return self.service.infer(image_path, metadata)

    def process_perception(
        self,
        *,
        image_path: Path | None,
        yolo_payload: dict[str, Any],
        flow_payload: dict[str, Any] | None = None,
        image_quality: ImageQualityInput | dict[str, Any] | None = None,
        user_query: str = DEFAULT_USER_QUERY,
        timeout_seconds: float | None = None,
    ) -> dict[str, Any]:
        """Perception `process_split()` payload로 안전 추론을 수행한다.

        YOLO 결과를 쓸 수 없으면 VLM을 호출하지 않고 결정적 fallback을 만든다.
        정상 경로는 항상 `VLMService.infer_safe()`를 통과한다.
        """

        try:
            metadata = build_perception_metadata(
                yolo_payload=yolo_payload,
                flow_payload=flow_payload,
                image_quality=image_quality,
                user_query=user_query,
                strict_frame_sync=self.strict_frame_sync,
            )
        except (
            PerceptionUnavailableError,
            YoloResultValidationError,
            MotionResultValidationError,
        ) as error:
            flow_usable = is_flow_usable(flow_payload)
            return build_perception_unavailable_result(
                reason=(
                    "yolo_unavailable" if flow_usable else "perception_unavailable"
                ),
                message=(
                    YOLO_UNAVAILABLE_MESSAGE
                    if flow_usable
                    else PERCEPTION_UNAVAILABLE_MESSAGE
                ),
                detail=str(error),
                model_id=getattr(
                    getattr(self.service, "engine", None),
                    "model_id",
                    "mock-rule-engine",
                ),
            )

        return self.service.infer_safe(
            image_path,
            metadata,
            timeout_seconds=timeout_seconds,
        )
