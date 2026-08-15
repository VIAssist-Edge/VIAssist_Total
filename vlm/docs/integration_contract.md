# VIAssist Vision Integration Contract

이 계약은 YOLO 및 Optical Flow 구현과 VLM 모듈 사이의 데이터 경계다. 외부 팀은
JSON 또는 동일한 Python `dict`를 전달하며, Adapter가 기존 metadata schema 1.0으로
정규화한다. 실제 detector나 motion 알고리즘 구현은 이 계약의 범위가 아니다.

## 좌표와 confidence

- bbox는 `[x1, y1, x2, y2]` 순서의 픽셀 좌표다.
- 원점은 이미지 좌상단이며 normalized 좌표는 사용하지 않는다.
- `x1 < x2`, `y1 < y2`이고 모든 좌표는 `image_width`, `image_height` 안에 있어야 한다.
- confidence는 문자열이나 boolean이 아닌 유한 숫자이며 `0.0 <= confidence <= 1.0`이다.
- YOLO confidence를 그대로 전달할 수 있다.

## YOLO class 계약

| 내부 class | 허용 외부 alias |
|---|---|
| `elevator_button` | `elevator_button`, `elevator-button`, `lift_button`, `lift-button`, `elevator_call_button`, `elevator-call-button` |
| `escalator` | `escalator`, `moving_stairs`, `moving-stairs` |

지원하지 않는 class는 최종 detections에서 제외한다. 지원 detection은 confidence
내림차순으로 정렬하며 같은 confidence의 입력 순서는 유지한다.

### Detection 필드 표기

Adapter는 두 표기를 모두 받는다.

| 내부 필드 | 허용 입력 키 |
|---|---|
| `class_name` | `class_name`, `cls_name` |
| `confidence` | `confidence`, `conf` |
| `bbox` | `bbox`, 또는 `x1`, `y1`, `x2`, `y2` 네 개 모두 |

규칙:

- 두 표기가 함께 있고 값이 같으면 허용한다. 문자열 대소문자와 앞뒤 공백 차이는
  같은 값으로 본다.
- 값이 서로 다르면 추측하지 않고 `YoloResultValidationError`를 발생시킨다.
- `x1, y1, x2, y2` 중 일부만 있으면 오류다.
- `bbox`와 좌표 키가 함께 있고 값이 다르면 오류다.
- `cls_id`, `track_id`, `clock`, `distance_m`, `area_ratio`, `center_offset`은
  받아들이지만 현재 내부 metadata에는 넣지 않는다. 특히 `distance_m`은 안내
  문장에 사용하지 않는다.

### Payload 수준 필드

| 필드 | 처리 |
|---|---|
| `ok=false` | `PerceptionUnavailableError`. `process_perception()`은 VLM을 호출하지 않고 degraded 결과를 만든다. |
| `ok`가 boolean이 아님 | `YoloResultValidationError` |
| `quality` | 내부 `image_quality`로 이동. 빈 dict는 정보 없음으로 보고 기본값을 쓴다. 지원하지 않는 필드는 버린다. |
| `error` | 진단용으로만 사용하며 안내 문장에 넣지 않는다. |
| `timestamp`, `latency_ms` | 현재 동기화·안내에 사용하지 않는다. |

`image_quality` 인자를 직접 넘기면 payload의 `quality`보다 우선한다.

YOLO 담당자의 최소 입력:

```json
{
  "image_width": 1280,
  "image_height": 720,
  "detections": [
    {
      "class_name": "escalator",
      "confidence": 0.95,
      "bbox": [400, 100, 900, 700]
    }
  ]
}
```

## Position 계약

| 내부 position | 허용 외부 alias |
|---|---|
| `left` | `left`, `LEFT`, `좌측`, `왼쪽` |
| `front` | `front`, `center`, `centre`, `middle`, `정면`, `중앙` |
| `right` | `right`, `RIGHT`, `우측`, `오른쪽` |

명시적인 position이 있으면 alias를 정규화해 우선 사용한다. 값이 있는데 유효하지
않으면 추정하지 않고 validation error를 발생시킨다. position이 없으면 bbox 중심
`center_x=(x1+x2)/2`로 계산한다.

- `center_x < image_width / 3`: `left`
- `center_x > image_width * 2 / 3`: `right`
- 나머지와 두 경계값: `front`

## Optical Flow 계약

| 내부 direction | 허용 외부 alias |
|---|---|
| `up` | `up`, `upward`, `ascending`, `상행`, `위`, `위쪽` |
| `down` | `down`, `downward`, `descending`, `하행`, `아래`, `아래쪽` |
| `left` | `left`, `좌`, `왼쪽` |
| `right` | `right`, `우`, `오른쪽` |
| `opening` | `opening`, `open`, `열림`, `열리는중` |
| `closing` | `closing`, `close`, `닫힘`, `닫히는중` |
| `stopped` | `stopped`, `stop`, `stationary`, `정지`, `멈춤` |
| `unknown` | `unknown`, `none`, `unavailable`, `알수없음` |

Optical Flow 담당자의 최소 입력:

```json
{
  "available": true,
  "direction": "up",
  "speed": 0.72
}
```

`available=false`이면 Adapter는 direction을 반드시 `unknown`으로 만든다.
`available=true`와 direction `unknown` 조합은 허용하지만 가능한 한 실제 방향 제공을
권장한다. motion 결과가 없으면 다음 기본값을 사용한다.

### Perception flow payload 정규화

`src/perception_adapter.normalize_flow_payload()`는 Perception 계약(§12)을 따르며
위 alias 표보다 좁은 `up`, `down`, `stopped`, `unknown`만 사용한다.

| 입력 상태 | 결과 |
|---|---|
| `ok` 없음 또는 `ok=false` | `available=false`, `direction=unknown` |
| `available=false` | `available=false`, `direction=unknown` (`direction` 값 무시) |
| `direction=unknown` | `available=false` |
| 지원하지 않는 direction | `available=false`, `direction=unknown` |
| `direction=stopped` | `available=true`, `direction=stopped` |

`available`이 boolean이 아니면 `MotionResultValidationError`를 발생시킨다.
`unknown`과 `stopped`는 서로 다른 상태이며 `speed`는 `px/frame @ 320px`이므로
물리 속도로 안내하지 않는다.

```json
{
  "available": false,
  "direction": "unknown",
  "speed": "unknown"
}
```

speed는 유한 숫자, 문자열 `unknown`, 또는 null을 받을 수 있다. 기존 metadata
schema와 호환하기 위해 숫자는 문자열로 변환하고 null은 `unknown`으로 변환한다.

## Image quality

최소 입력은 `{"is_blurry": false}`다. `blur_score`, `brightness_score`, `source`,
`timestamp` 같은 추가 입력은 허용하지만 현재 VLM metadata에는 기존 schema가
지원하는 `is_blurry`와 `blur_score`만 전달한다. blur score가 없으면 `0.0`을 쓴다.

## Frame 및 timestamp 동기화

YOLO와 motion은 선택적으로 `frame_id`, `timestamp_ms`를 제공할 수 있다. 양쪽
frame_id가 모두 존재하면 동일해야 한다.

- strict mode: 불일치 시 `FrameSynchronizationError`
- non-strict mode: motion을 unavailable/unknown으로 강등

timestamp만 있는 경우 이번 Phase에서는 동기화를 판정하지 않는다. 실제 비동기
pipeline이 연결되면 timestamp tolerance와 최대 지연 정책이 추가로 필요하다.
frame/timestamp는 기존 metadata schema에 새 필드로 강제 추가하지 않는다.

## Python handoff 예시

```python
from pathlib import Path

from src.config_loader import load_config
from src.integration_pipeline import VIAssistVLMPipeline
from src.vlm_service import VLMService

config = load_config(Path("config/jetson.json"))
service = VLMService.from_config(config)
pipeline = VIAssistVLMPipeline(service)

result = pipeline.process(
    image_path=Path("samples/escalator.png"),
    yolo_result=yolo_result,
    motion_result=motion_result,
    image_quality={"is_blurry": False},
    user_query="에스컬레이터 방향을 알려줘.",
    safe=True,
    timeout_seconds=10,
)
```

Adapter는 입력 객체를 변경하지 않고 결과를 만든 뒤 항상 기존
`validate_metadata()`를 호출한다. Adapter 내부의 `image_path`는 schema 호환용
placeholder이며 실제 추론 이미지는 Pipeline의 `image_path` 인자로 전달한다.
