# Perception 반환값 연동 계약

## 1. 목적

이 문서는 VIAssist VLM 모듈과 Perception 모듈 사이의 실제 데이터 전달 규격을 정의한다.

Perception 모듈은 하나의 원본 카메라 프레임에 대해 다음 두 결과를 생성한다.

- YOLO 객체 탐지 결과
- Optical Flow 움직임 분석 결과

VLM 연동에서는 두 결과를 독립적으로 전달받는 `process_split()` 방식을 사용한다.

전체 흐름은 다음과 같다.

```text
Camera Frame
  ↓
Perception.process_split(frame)
  ├─ yolo_payload
  └─ flow_payload
       ↓
Metadata Adapter
       ↓
VLMService.infer_safe()
       ↓
Safety Validator
       ↓
Final JSON
       ↓
TTS
```

---

## 2. Perception 실행 원칙

Perception 객체와 내부 모델은 프로세스 시작 시 한 번만 생성한다.

매 프레임마다 `Perception` 객체나 YOLO 모델을 다시 생성하지 않는다.

```python
from perception import Perception, PerceptionConfig

perception = Perception(
    weights="runs_combined_v3/yolo26s_unified_v3/weights/best.pt",
    config=PerceptionConfig(
        imgsz=640,
        conf=0.25,
        hfov_deg=78.0,
    ),
)

perception.warmup(1280, 720)
```

현재 YOLO v4 모델이 학습 중인 경우, 완료 전까지 v3 weight를 통합 테스트에 사용한다.

```text
runs_combined_v3/yolo26s_unified_v3/weights/best.pt
```

VLMService 역시 프로세스 시작 시 한 번 생성하고 반복 추론에서 재사용한다.

```python
from pathlib import Path

from src.config_loader import load_config
from src.integration_pipeline import VIAssistVLMPipeline
from src.vlm_service import VLMService

config = load_config(Path("config/jetson.json"))
service = VLMService.from_config(config)
pipeline = VIAssistVLMPipeline(service)
```

---

## 3. 권장 호출 방식

VLM 연동에서는 YOLO와 Optical Flow 결과를 분리해서 받는다.

```python
yolo_payload, flow_payload = perception.process_split(frame)
```

`frame`은 OpenCV 형식의 BGR 원본 해상도 이미지다.

```python
import numpy as np

frame: np.ndarray
```

두 payload는 다음 특징을 가진다.

- 각각 독립적인 `dict`
- JSON 직렬화 가능
- 하나가 없어도 다른 하나를 독립적으로 해석 가능
- 동일한 `frame_id`로 같은 프레임의 결과를 연결
- `process_split()`은 일반적인 처리 실패 시 예외를 외부로 던지지 않음
- 처리 실패 시 두 payload 모두 `ok=false`
- 같은 처리 실패에서 생성된 payload는 동일한 오류 정보를 가질 수 있음

---

## 4. YOLO Payload 계약

### 4.1 예시

```json
{
  "frame_id": 128,
  "timestamp": 1750000000.123,
  "image_width": 1280,
  "image_height": 720,
  "detections": [
    {
      "cls_id": 30,
      "cls_name": "disp_up",
      "conf": 0.88,
      "x1": 812,
      "y1": 301,
      "x2": 849,
      "y2": 339,
      "track_id": null,
      "clock": 12,
      "distance_m": null,
      "area_ratio": 0.0015,
      "center_offset": 0.05
    }
  ],
  "quality": {},
  "latency_ms": 32.6,
  "ok": true,
  "error": null
}
```

### 4.2 최상위 필드

| 필드 | 타입 | 필수 | 설명 |
|---|---|---:|---|
| `frame_id` | `int` | 예 | 프레임 식별자 |
| `timestamp` | `int` 또는 `float` | 예 | 프레임 또는 처리 시각 |
| `image_width` | `int` | 예 | 원본 프레임 너비 |
| `image_height` | `int` | 예 | 원본 프레임 높이 |
| `detections` | `list` | 예 | 객체 탐지 결과 |
| `quality` | `dict` | 예 | 촬영 품질 분석 결과 |
| `latency_ms` | `int` 또는 `float` | 예 | YOLO 처리 시간 |
| `ok` | `bool` | 예 | YOLO 처리 성공 여부 |
| `error` | `null`, `str`, `dict` | 예 | 실패 정보 |

### 4.3 Detection 필드

| 필드 | 타입 | 필수 | 설명 |
|---|---|---:|---|
| `cls_id` | `int` | 예 | YOLO 클래스 ID |
| `cls_name` | `str` | 예 | YOLO 클래스 이름 |
| `conf` | `float` | 예 | 객체 탐지 신뢰도, `0.0~1.0` |
| `x1` | `int` 또는 `float` | 예 | bbox 왼쪽 좌표 |
| `y1` | `int` 또는 `float` | 예 | bbox 위쪽 좌표 |
| `x2` | `int` 또는 `float` | 예 | bbox 오른쪽 좌표 |
| `y2` | `int` 또는 `float` | 예 | bbox 아래쪽 좌표 |
| `track_id` | `int` 또는 `null` | 예 | 추적 ID |
| `clock` | `int` 또는 `null` | 예 | 화면 기준 시계 방향 위치 |
| `distance_m` | `float` 또는 `null` | 예 | 추정 거리 |
| `area_ratio` | `float` 또는 `null` | 예 | 원본 화면 대비 bbox 면적 비율 |
| `center_offset` | `float` 또는 `null` | 예 | 화면 중앙 대비 객체 중심 오프셋 |

---

## 5. YOLO bbox 규칙

YOLO payload의 bbox는 다음 형식이다.

```text
[x1, y1, x2, y2]
```

좌표 규칙:

- 좌상단이 원점
- 단위는 픽셀
- 원본 프레임 좌표계
- YOLO 추론 resize 또는 letterbox 좌표가 아님
- Perception 내부에서 원본 프레임 좌표로 복원된 값
- `x1 < x2`
- `y1 < y2`
- 모든 좌표는 원본 이미지 범위 안에 있어야 함

VLM Adapter에서는 다음처럼 변환한다.

```python
bbox = [
    detection["x1"],
    detection["y1"],
    detection["x2"],
    detection["y2"],
]
```

변환 결과는 기존 내부 metadata 형식을 유지한다.

```json
{
  "class_name": "escalator",
  "confidence": 0.95,
  "bbox": [400, 100, 900, 700],
  "position": "front"
}
```

---

## 6. YOLO 필드 변환 규칙

Perception YOLO 결과와 VLM 내부 metadata의 필드 매핑은 다음과 같다.

| Perception 필드 | VLM 내부 필드 |
|---|---|
| `cls_name` | `class_name` |
| `conf` | `confidence` |
| `x1, y1, x2, y2` | `bbox` |
| `frame_id` | frame synchronization |
| `image_width` | position 계산 |
| `image_height` | bbox 검증 |
| `quality` | `image_quality` |
| `track_id` | 선택적 motion 연결 정보 |
| `clock` | 선택적 공간 표현 정보 |
| `distance_m` | 현재 안내에는 사용하지 않음 |
| `area_ratio` | 선택적 품질·우선순위 정보 |
| `center_offset` | 선택적 위치 보조 정보 |

---

## 7. Position 계산 규칙

초기 MVP에서는 bbox 중심을 이용해 `left`, `front`, `right`를 계산한다.

```python
center_x = (x1 + x2) / 2
```

판정 규칙:

```text
center_x < image_width / 3
→ left

center_x > image_width * 2 / 3
→ right

그 외
→ front
```

경계값은 `front`에 포함한다.

```text
center_x == image_width / 3
→ front

center_x == image_width * 2 / 3
→ front
```

내부 표준값:

```text
left
front
right
```

한국어 출력 예시:

```text
left  → 왼쪽
front → 정면
right → 오른쪽
```

`clock` 값은 입력으로 보존할 수 있지만, 초기 통합에서는 기존 bbox 3분할 위치 계산을 기본으로 사용한다.

---

## 8. 거리 정보 처리

`distance_m` 값은 현재 VLM 안내 문장 생성에 사용하지 않는다.

```json
{
  "distance_m": null
}
```

숫자 값이 제공되더라도 거리 산출 방식과 오차 검증이 완료되기 전에는 다음과 같은 문장을 생성하지 않는다.

```text
50센티미터 앞에 버튼이 있습니다.
```

현재 MVP에서는 검증된 화면 위치만 안내한다.

```text
오른쪽에 엘리베이터 버튼이 있습니다.
```

---

## 9. Optical Flow Payload 계약

### 9.1 예시

```json
{
  "frame_id": 128,
  "available": true,
  "direction": "up",
  "confidence": 0.83,
  "speed": 1.84,
  "mean_dx": -0.12,
  "mean_dy": -1.84,
  "magnitude": 1.84,
  "valid_ratio": 0.41,
  "roi": [420, 260, 900, 700],
  "roi_source": "detection",
  "stable_frames": 7,
  "image_width": 1280,
  "image_height": 720,
  "ok": true,
  "error": null
}
```

### 9.2 필드 설명

| 필드 | 타입 | 필수 | 설명 |
|---|---|---:|---|
| `frame_id` | `int` | 예 | YOLO payload와 연결하는 프레임 ID |
| `available` | `bool` | 예 | 방향 분석 결과를 사용할 수 있는지 표시 |
| `direction` | `str` | 예 | 움직임 방향 |
| `confidence` | `float` | 예 | 움직임 분석 신뢰도 |
| `speed` | `float` | 예 | `px/frame @ 320px` 기준 속도 |
| `mean_dx` | `float` | 예 | ROI 평균 수평 이동 |
| `mean_dy` | `float` | 예 | ROI 평균 수직 이동 |
| `magnitude` | `float` | 예 | Optical Flow 이동 크기 |
| `valid_ratio` | `float` | 예 | 유효 Flow 벡터 비율 |
| `roi` | `list` 또는 `null` | 예 | 분석한 원본 프레임 ROI |
| `roi_source` | `str` 또는 `null` | 예 | ROI 생성 기준 |
| `stable_frames` | `int` | 예 | 동일 판정이 안정적으로 유지된 프레임 수 |
| `image_width` | `int` | 예 | 원본 프레임 너비 |
| `image_height` | `int` | 예 | 원본 프레임 높이 |
| `ok` | `bool` | 예 | Flow 처리 성공 여부 |
| `error` | `null`, `str`, `dict` | 예 | 실패 정보 |

---

## 10. Optical Flow direction 규칙

지원하는 direction은 다음과 같다.

```text
up
down
stopped
unknown
```

각 값의 의미:

| direction | 의미 |
|---|---|
| `up` | 위쪽 방향 움직임이 감지됨 |
| `down` | 아래쪽 방향 움직임이 감지됨 |
| `stopped` | 움직임이 정지한 것으로 판정됨 |
| `unknown` | 방향을 판단할 수 없음 |

다음 두 상태는 서로 다르다.

```text
unknown != stopped
```

- `unknown`: 방향 판단 불가
- `stopped`: 정지 상태로 판정

신뢰도가 부족한 경우 `stopped`로 추측하지 않고 `unknown`으로 처리한다.

---

## 11. available 처리 규칙

`available`은 direction 사용 가능 여부를 결정하는 최우선 필드다.

```text
available == true
→ direction 사용 가능

available == false
→ direction 무시
```

다음과 같은 경우 `available=false`가 될 수 있다.

- 첫 프레임
- 이전 프레임 없음
- 적절한 ROI 없음
- 이미지 텍스처 부족
- Flow 분석 신뢰도 부족
- 분석 실패
- stale frame
- frame synchronization 실패

`available=false`인데 direction 값이 `up` 또는 `down`으로 들어와도 방향을 안내하면 안 된다.

```python
if not flow_payload["available"]:
    direction = "unknown"
```

---

## 12. Flow payload 정규화 규칙

VLM Adapter는 다음 순서로 Flow 결과를 처리한다.

```python
def normalize_flow_payload(flow_payload: dict) -> dict:
    if not flow_payload.get("ok", False):
        return {
            "available": False,
            "direction": "unknown",
        }

    if not flow_payload.get("available", False):
        return {
            "available": False,
            "direction": "unknown",
        }

    direction = flow_payload.get("direction", "unknown")

    if direction not in {"up", "down", "stopped", "unknown"}:
        direction = "unknown"

    if direction == "unknown":
        return {
            "available": False,
            "direction": "unknown",
        }

    return {
        "available": True,
        "direction": direction,
    }
```

`direction=unknown`은 방향 안내가 가능한 상태로 취급하지 않는다.

---

## 13. speed 처리 규칙

Flow의 `speed`는 물리 속도가 아니다.

```text
단위: px/frame @ 320px scale
```

예:

```json
{
  "speed": 1.84
}
```

이 값은 다음과 같은 의미가 아니다.

```text
1.84m/s
1.84km/h
초속 1.84미터
```

초당 픽셀 이동량이 필요한 경우 다음과 같이 계산할 수 있다.

```text
pixels_per_second = speed × FPS
```

하지만 이 값도 물리 거리로 보정된 속도가 아니다.

초기 MVP에서는 `speed`를 다음 용도로만 사용한다.

- 개발 로그
- 성능 분석
- 방향 판정 보조
- 안정성 판단
- 디버깅
- 향후 threshold 조정

사용자 안내 문장에는 speed 숫자를 직접 포함하지 않는다.

---

## 14. ROI 처리 규칙

Flow ROI는 원본 프레임 좌표계다.

```json
{
  "roi": [420, 260, 900, 700],
  "roi_source": "detection"
}
```

ROI 형식:

```text
[x1, y1, x2, y2]
```

규칙:

- 원본 프레임 픽셀 좌표
- YOLO bbox와 같은 좌표계
- `x1 < x2`
- `y1 < y2`
- 이미지 범위 안
- `roi_source="detection"`이면 객체 탐지 결과를 기반으로 생성된 ROI

Flow ROI는 어떤 객체의 움직임을 분석했는지 확인하는 근거로 사용할 수 있다.

현재 VLM Adapter는 ROI를 가장 높은 confidence의 지원 detection과 연결할 수 있다.

복수 객체를 정밀하게 연결할 수 있는 식별자가 있는 경우 다음 우선순위를 고려한다.

```text
track_id
→ detection ID
→ ROI overlap
→ confidence가 가장 높은 지원 detection
```

---

## 15. 프레임 동기화 규칙

YOLO와 Flow payload는 `frame_id`를 이용해 같은 프레임의 결과인지 확인한다.

정상 예시:

```json
{
  "yolo_frame_id": 128,
  "flow_frame_id": 128
}
```

동기화 성공:

```text
yolo_payload.frame_id == flow_payload.frame_id
```

### Non-strict 모드

프레임 ID가 다르면 YOLO 결과는 유지하되 Flow 결과를 사용하지 않는다.

```python
if yolo_frame_id != flow_frame_id:
    motion = {
        "available": False,
        "direction": "unknown",
    }
```

### Strict 모드

프레임 ID가 다르면 통합 오류를 발생시킨다.

```text
FrameSynchronizationError
```

실제 실시간 MVP에서는 일시적인 프레임 차이로 전체 안내가 중단되지 않도록 non-strict 모드를 기본으로 사용한다.

---

## 16. Timestamp 처리

현재 프레임 동기화의 기본 키는 `frame_id`다.

`timestamp`는 다음 용도로 보존할 수 있다.

- 처리 로그
- 지연시간 분석
- stale frame 검사
- 비동기 queue 분석
- End-to-End latency 측정

Timestamp 단독 동기화는 초기 MVP 범위에 포함하지 않는다.

---

## 17. Image Quality 처리

YOLO payload의 `quality`는 VLM 내부의 `image_quality`로 전달한다.

예:

```json
{
  "quality": {
    "is_blurry": false,
    "blur_score": 120.5
  }
}
```

내부 metadata:

```json
{
  "image_quality": {
    "is_blurry": false,
    "blur_score": 120.5
  }
}
```

촬영 품질이 낮으면 객체나 움직임 상태를 추측하지 않는다.

예상 fallback:

```text
화면이 흐려 확인하기 어렵습니다. 카메라를 잠시 고정해 주세요.
```

`quality`에 내부 metadata schema가 지원하지 않는 필드가 들어오면 Adapter에서 제거하거나 별도 diagnostic 정보로 보존한다.

---

## 18. YOLO 실패 처리

다음 조건 중 하나라도 만족하면 YOLO 결과를 사용할 수 없는 상태로 처리한다.

```text
yolo_payload가 없음
ok == false
detections 필드가 list가 아님
image_width 또는 image_height가 잘못됨
bbox가 원본 이미지 범위를 벗어남
```

YOLO가 실패하면 VLM 이미지 단독 판단으로 객체를 추측하지 않는다.

예상 결과:

```json
{
  "message": "주변 객체를 확인하기 어렵습니다. 카메라를 잠시 고정해 주세요.",
  "target": null,
  "position": "unknown",
  "status": "uncertain",
  "service_status": "degraded",
  "fallback_reason": "yolo_unavailable"
}
```

---

## 19. Optical Flow 실패 처리

다음 조건 중 하나라도 만족하면 방향 정보를 사용할 수 없는 상태로 처리한다.

```text
flow_payload가 없음
ok == false
available == false
direction == unknown
direction이 지원 목록에 없음
YOLO와 frame_id 불일치
ROI가 유효하지 않음
```

Flow만 실패하고 YOLO 객체가 유효하면 객체 존재와 위치는 안내할 수 있다.

예:

```text
정면에 에스컬레이터가 있습니다. 운행 방향은 확인하기 어렵습니다.
```

또는 한 문장 제한이 필요한 경우:

```text
정면에 에스컬레이터가 있지만 운행 방향은 확인하기 어렵습니다.
```

---

## 20. YOLO와 Flow 모두 실패한 경우

YOLO와 Flow 결과를 모두 사용할 수 없으면 객체나 방향을 추측하지 않는다.

```text
주변 상황을 확인하기 어렵습니다. 카메라를 잠시 고정해 주세요.
```

최종 JSON 예시:

```json
{
  "message": "주변 상황을 확인하기 어렵습니다. 카메라를 잠시 고정해 주세요.",
  "target": null,
  "position": "unknown",
  "status": "uncertain",
  "confidence": "low",
  "safety_validated": true,
  "used_fallback": true,
  "message_source": "fallback",
  "service_status": "degraded",
  "fallback_reason": "perception_unavailable"
}
```

외부 payload의 `error` 원문이나 traceback은 TTS 문장에 포함하지 않는다.

---

## 21. 에스컬레이터 안내 정책

### 상행

입력:

```json
{
  "available": true,
  "direction": "up"
}
```

출력:

```text
정면에 위쪽으로 운행하는 에스컬레이터가 있습니다.
```

### 하행

입력:

```json
{
  "available": true,
  "direction": "down"
}
```

출력:

```text
정면에 아래쪽으로 운행하는 에스컬레이터가 있습니다.
```

### 방향 판단 불가

입력:

```json
{
  "available": false,
  "direction": "unknown"
}
```

출력:

```text
정면에 에스컬레이터가 있지만 운행 방향은 확인하기 어렵습니다.
```

### unknown

입력:

```json
{
  "available": true,
  "direction": "unknown"
}
```

정규화 결과:

```json
{
  "available": false,
  "direction": "unknown"
}
```

출력:

```text
정면에 에스컬레이터가 있지만 운행 방향은 확인하기 어렵습니다.
```

### stopped

입력:

```json
{
  "available": true,
  "direction": "stopped"
}
```

안전한 표현:

```text
정면에 에스컬레이터가 있으며 현재 움직임이 감지되지 않습니다.
```

`stopped`를 근거로 다음과 같은 행동 지시를 생성하지 않는다.

```text
운행하지 않으니 타지 마세요.
고장 났습니다.
안전하니 탑승하세요.
```

---

## 22. 엘리베이터 버튼 안내 정책

YOLO detection이 엘리베이터 버튼으로 정규화되고 bbox 위치가 오른쪽인 경우:

```text
오른쪽에 엘리베이터 버튼이 있습니다.
```

버튼의 세부 종류가 탐지 class로 검증된 경우에만 세부 정보를 포함한다.

예:

```text
오른쪽에 위쪽 호출 버튼이 있습니다.
```

다음 정보는 근거가 없으면 생성하지 않는다.

- 버튼이 눌렸는지 여부
- 버튼 조명이 켜졌는지 여부
- 엘리베이터 도착 여부
- 정확한 거리
- 버튼을 누르라는 강한 행동 지시
- 현재 탑승이 안전하다는 판단

---

## 23. VLM 내부 Metadata 예시

### 에스컬레이터 상행

```json
{
  "schema_version": "1.0",
  "image_path": "samples/escalator.png",
  "user_query": "에스컬레이터 방향을 알려줘.",
  "detections": [
    {
      "class_name": "escalator",
      "confidence": 0.95,
      "position": "front",
      "bbox": [400, 100, 900, 700]
    }
  ],
  "motion": {
    "available": true,
    "direction": "up",
    "speed": "1.84",
    "target": "escalator",
    "confidence": 0.83
  },
  "image_quality": {
    "is_blurry": false
  }
}
```

### Flow unavailable

```json
{
  "schema_version": "1.0",
  "image_path": "samples/escalator.png",
  "user_query": "에스컬레이터 방향을 알려줘.",
  "detections": [
    {
      "class_name": "escalator",
      "confidence": 0.95,
      "position": "front",
      "bbox": [400, 100, 900, 700]
    }
  ],
  "motion": {
    "available": false,
    "direction": "unknown",
    "speed": null,
    "target": "escalator",
    "confidence": null
  },
  "image_quality": {
    "is_blurry": false
  }
}
```

---

## 24. Pipeline 호출 예시

```python
from pathlib import Path
from typing import Any

import numpy as np

from src.config_loader import load_config
from src.integration_pipeline import VIAssistVLMPipeline
from src.vlm_service import VLMService


def process_frame(
    frame: np.ndarray,
    perception: Any,
    pipeline: VIAssistVLMPipeline,
    image_path: Path,
    user_query: str | None = None,
) -> dict:
    yolo_payload, flow_payload = perception.process_split(frame)

    quality = yolo_payload.get("quality", {})

    return pipeline.process(
        image_path=image_path,
        yolo_result=yolo_payload,
        motion_result=flow_payload,
        image_quality=quality,
        user_query=user_query,
        safe=True,
        timeout_seconds=10,
    )


def build_pipeline() -> VIAssistVLMPipeline:
    config = load_config(Path("config/jetson.json"))
    service = VLMService.from_config(config)
    return VIAssistVLMPipeline(service)
```

실제 카메라 frame을 VLM에 전달하는 방식이 파일 경로 기반인 경우, 동일 프레임을 임시 이미지로 저장한 뒤 해당 경로를 전달한다.

```python
import cv2

image_path = Path("runtime/latest_frame.jpg")

if not cv2.imwrite(str(image_path), frame):
    raise RuntimeError(f"Failed to save frame: {image_path}")
```

YOLO가 분석한 frame과 VLM에 전달되는 이미지는 반드시 동일해야 한다.

---

## 25. Adapter 구현 시 유지할 원칙

외부 Perception payload 형식이 기존 VLM 내부 schema와 다르더라도 내부 schema를 직접 변경하지 않는다.

변경 범위는 다음 계층으로 제한한다.

```text
Perception Payload
  ↓
Metadata Adapter
  ↓
기존 VLM Metadata Schema
  ↓
기존 VLMService
  ↓
기존 Safety Validator
```

다음 구성요소의 기존 계약은 유지한다.

- `validate_metadata()`
- `VLMService.infer()`
- `VLMService.infer_safe()`
- Safety Validator
- fallback generator
- 최종 JSON 형식
- `service_status`
- `used_fallback`
- `message_source`
- `validation_reasons`

---

## 26. Safety Validator 필수 규칙

Perception 통합 이후에도 다음 검증을 유지한다.

- YOLO에 없는 객체를 VLM이 생성하면 차단
- YOLO 위치와 다른 위치를 말하면 차단
- `available=false`인데 움직임 방향을 말하면 차단
- `direction=unknown`인데 움직임 방향을 말하면 차단
- `up`인데 아래쪽이라고 말하면 차단
- `down`인데 위쪽이라고 말하면 차단
- 검증되지 않은 거리 표현 차단
- 물리 속도로 오해될 수 있는 속도 표현 차단
- “안전합니다”와 같은 근거 없는 안전 판단 차단
- “탑승하세요”와 같은 강한 행동 지시 차단
- 질문형 출력 차단
- 사용자 질문 반복 차단
- 장황한 배경 설명 차단
- 두 문장을 초과하는 출력 차단

검증 실패 시 VLM 원문을 사용하지 않고 deterministic fallback으로 교체한다.

---

## 27. 최종 출력 예시

### 정상 YOLO + 정상 Flow

```json
{
  "message": "정면에 위쪽으로 운행하는 에스컬레이터가 있습니다.",
  "target": "escalator",
  "position": "front",
  "status": "detected",
  "confidence": "high",
  "detection_confidence": 0.95,
  "latency_ms": 4763.87,
  "peak_gpu_memory_mb": 1567.16,
  "model_id": "HuggingFaceTB/SmolVLM-500M-Instruct",
  "safety_validated": true,
  "used_fallback": false,
  "validation_reasons": [],
  "message_source": "vlm",
  "raw_vlm_message": "정면에 위쪽으로 운행하는 에스컬레이터가 있습니다.",
  "service_status": "ok",
  "error": null,
  "fallback_reason": null
}
```

### VLM 질문형 출력이 Validator에서 교체된 경우

```json
{
  "message": "정면에 위쪽으로 운행하는 에스컬레이터가 있습니다.",
  "target": "escalator",
  "position": "front",
  "status": "detected",
  "confidence": "high",
  "detection_confidence": 0.95,
  "latency_ms": 4763.87,
  "peak_gpu_memory_mb": 1567.16,
  "model_id": "HuggingFaceTB/SmolVLM-500M-Instruct",
  "safety_validated": true,
  "used_fallback": true,
  "validation_reasons": [
    "question_form_output",
    "required_motion_direction_missing"
  ],
  "message_source": "fallback",
  "raw_vlm_message": "앞에 에스컬레이터가 있나요?",
  "service_status": "ok",
  "error": null,
  "fallback_reason": "safety_validation_failed"
}
```

Safety Validator의 문장 교체는 시스템 장애가 아니다.

```text
used_fallback = true
service_status = ok
```

### Flow unavailable

```json
{
  "message": "정면에 에스컬레이터가 있지만 운행 방향은 확인하기 어렵습니다.",
  "target": "escalator",
  "position": "front",
  "status": "detected",
  "confidence": "high",
  "detection_confidence": 0.95,
  "safety_validated": true,
  "used_fallback": true,
  "message_source": "fallback",
  "service_status": "ok",
  "error": null,
  "fallback_reason": "motion_unavailable"
}
```

### Perception 실패

```json
{
  "message": "주변 상황을 확인하기 어렵습니다. 카메라를 잠시 고정해 주세요.",
  "target": null,
  "position": "unknown",
  "status": "uncertain",
  "confidence": "low",
  "safety_validated": true,
  "used_fallback": true,
  "message_source": "fallback",
  "service_status": "degraded",
  "error": {
    "type": "perception_error",
    "message": "Perception result is unavailable."
  },
  "fallback_reason": "perception_unavailable"
}
```

---

## 28. 테스트 시나리오

Perception Adapter 통합 후 최소한 다음 테스트를 수행한다.

### YOLO

1. 정상 객체 1개
2. 정상 객체 여러 개
3. 객체 없음
4. 지원하지 않는 class
5. confidence 경계값
6. bbox 범위 초과
7. `x1 >= x2`
8. `y1 >= y2`
9. `ok=false`
10. `detections` 누락

### Optical Flow

1. `available=true`, `direction=up`
2. `available=true`, `direction=down`
3. `available=true`, `direction=stopped`
4. `available=true`, `direction=unknown`
5. `available=false`, `direction=up`
6. `available=false`, `direction=unknown`
7. `ok=false`
8. 지원하지 않는 direction
9. ROI 범위 초과
10. ROI 없음

### 동기화

1. YOLO와 Flow frame ID 일치
2. frame ID 불일치, non-strict
3. frame ID 불일치, strict
4. YOLO만 존재
5. Flow만 존재
6. 둘 다 실패

### Safety Validator

1. VLM이 YOLO에 없는 객체 생성
2. 객체 위치 불일치
3. Flow unavailable인데 방향 생성
4. `up/down` 방향 불일치
5. 물리 속도 표현 생성
6. 검증되지 않은 거리 생성
7. 질문형 출력
8. 행동 지시 출력
9. 안전 판단 출력
10. 두 문장 초과

---

## 29. 통합 완료 기준

다음 조건을 모두 만족하면 실제 Perception 연동이 완료된 것으로 판정한다.

- `process_split()`의 실제 payload를 Adapter가 처리함
- YOLO bbox가 원본 프레임 좌표로 검증됨
- YOLO와 VLM 이미지가 동일한 프레임임
- `frame_id` 기반 동기화가 동작함
- `available=false`에서 방향 안내가 나오지 않음
- `unknown`과 `stopped`가 구분됨
- speed가 물리 속도로 안내되지 않음
- YOLO 실패 시 객체를 추측하지 않음
- Flow 실패 시 객체 위치만 제한적으로 안내함
- Safety Validator가 기존과 동일하게 동작함
- 최종 JSON 계약이 유지됨
- 모델이 반복 요청마다 다시 로드되지 않음
- Jetson 안정화 설정 `1024/512`가 유지됨
- 단위 테스트와 mock integration test가 통과함
- 실제 Jetson 통합 추론이 성공함

---

## 30. 반드시 지켜야 할 핵심 규칙

1. `available=false`이면 direction을 무시한다.
2. speed는 `px/frame @ 320px`이며 물리 속도가 아니다.
3. `unknown`과 `stopped`를 같은 상태로 처리하지 않는다.
4. 판단할 수 없는 방향은 추측하지 않는다.
5. YOLO가 객체 존재와 위치의 Source of Truth다.
6. Optical Flow가 움직임 방향의 Source of Truth다.
7. VLM은 검증된 사실을 자연어로 표현한다.
8. 모든 VLM 출력은 Safety Validator를 거친다.
9. VLM 실패가 전체 시스템 장애로 이어지지 않도록 fallback을 제공한다.
10. 원본 이미지, YOLO bbox, Flow ROI는 동일한 좌표계를 사용한다.
11. Perception과 VLM 모델은 프로세스 시작 시 한 번만 로드한다.
12. 최종 안내 문장은 짧고 자연스러운 한국어 평서문으로 생성한다.