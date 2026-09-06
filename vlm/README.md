# 시각장애인 안전 보행 VLM 모듈

## 샘플 데이터

`samples/elevator_button.png`와 `samples/escalator.png`는 PNG 이미지이며,
같은 이름의 JSON에 원본 이미지 픽셀 좌표계의 bbox와 이미지 크기가 기록되어
있습니다. bbox는 기존의 범위 밖 좌표를 비율로 임의 보정한 값이 아니라, 원본
이미지를 직접 확인해 대상 패널과 에스컬레이터 설비의 외곽을 수동으로 지정한
값입니다. 따라서 학습용 정밀 annotation이 아니라 입력 schema와 실행 경로를
검증하기 위한 예시 annotation으로 사용해야 합니다.

## VLM 출력 안전 정책

VLM이 생성한 문장은 사용자 안내로 그대로 사용되지 않습니다. metadata schema
검증 후 대표 detection을 선택하고, 모든 raw 문장은 safety validator를 통과해야
최종 `message`로 사용됩니다. 검증에 실패하면 모델을 다시 호출하지 않고 YOLO와
Optical Flow metadata만 사용하는 결정적 fallback 문장으로 전체 교체합니다.

- 객체 존재와 위치의 source of truth: YOLO `detections`
- 움직임과 방향의 source of truth: Optical Flow `motion`
- 현재 지원 target: `elevator_button`, `escalator`
- detection confidence threshold: `0.5`
- 흐린 화면, detection 없음, 낮은 confidence, 미지원 대상은 fallback 처리
- metadata와 다른 객체·위치·방향, 거리 추측, 안전 판단, 과도한 행동 지시는 차단
- 안내는 최대 2문장, 기본 120자 이하의 한국어 문장만 허용
- 질문형, 사용자 질문 반복, 명사형 불완전 문장과 `있는 것입니다` 같은 번역체는 차단
- 에스컬레이터의 up/down Optical Flow가 있으면 최종 안내에 방향 표현을 필수로 포함
- 미지원 대상은 `대상이 보이지만 정확한 안내를 제공하기 어렵습니다.`로 처리

실제 SmolVLM smoke test에서 질문형 응답과 부자연스러운 번역체 응답이 발견되어,
TTS에 적합한 평서형 안내 종결과 사용자 질문 반복 여부를 강제 검증하도록 보완했습니다.

결과의 `raw_vlm_message`는 검증 전 모델 출력을 조사하기 위한 디버깅 전용
필드입니다. 사용자 화면이나 TTS에는 반드시 검증이 끝난 `message`만 사용해야
합니다. `validation_reasons`, `used_fallback`, `message_source`로 교체 여부와 이유를
확인할 수 있습니다.

## 주변 장면 설명 모드 (scene_description)

`elevator_button`/`escalator` 안내는 YOLO detection과 대조하는 Safety
Validator를 거치므로, `SUPPORTED_TARGETS`(`vlm/src/safety_rules.py`) 밖의
객체(사람, 차량, 자전거 등)는 metadata 단계에서 필터링되어 사라지고
`"목표 객체를 확인하기 어렵습니다..."` fallback만 나옵니다. 화면에 사람만
있어도 실제 상황을 설명받고 싶을 때는 이 fallback 대신
`VLMService.describe_scene()` 경로를 씁니다.

- YOLO/Optical Flow metadata와 대조하지 않으므로 사람·차량 등도 언급할 수 있음
- 문장 형식과 과잉 주장 차단(질문형, 안전 판단, 거리 추측, 강한 행동 지시,
  2문장 초과 등)은 기존 Safety Validator와 동일하게 유지
- 검증 실패나 VLM 오류 시 `"주변 상황을 정확히 설명하기 어렵습니다."`로 대체

```python
result = pipeline.process_scene_description(
    image_path=image_path,
    user_query="주변 상황을 설명해줘.",
    timeout_seconds=10,
)
```

MVP에서는 `VLMBridge.describe_scene()` / `POST /vlm/describe_scene`
(`mvp/escalator_mvp.py`)과 `mvp/vlm_webcam_test.py`의 "주변 장면 설명 모드"
패널로 확인할 수 있습니다. `dev_describe`와 달리 안전 검증을 거치므로
`message`를 그대로 TTS에 사용할 수 있습니다.

## 실행

모델을 로드하지 않는 mock 실행:

```bash
python main.py \
  --image samples/elevator_button.png \
  --metadata samples/elevator_button.json \
  --engine mock
```

실제 모델 smoke test는 기본 단위 테스트에 포함되지 않습니다. 모델과 실행 환경이
준비된 경우에만 다음 명령을 별도로 실행합니다.

```bash
python main.py \
  --image samples/elevator_button.png \
  --metadata samples/elevator_button.json \
  --engine smolvlm
```

단위 테스트:

```bash
python -m unittest discover -v
```

## Warm inference benchmark

하나의 `VLMService`와 모델을 재사용해 warm-up과 반복 추론을 수행하고, 모델 로딩
시간·추론 latency·GPU peak memory를 분리한 JSON 보고서를 생성합니다.

엘리베이터 버튼:

```bash
python scripts/benchmark_warm.py \
  --config config/jetson.json \
  --metadata samples/elevator_button.json \
  --image samples/elevator_button.png \
  --runs 3 \
  --warmup-runs 1 \
  --label elevator_button \
  --output reports/warm_elevator.json
```

에스컬레이터:

```bash
python scripts/benchmark_warm.py \
  --config config/jetson.json \
  --metadata samples/escalator.json \
  --image samples/escalator.png \
  --runs 3 \
  --warmup-runs 1 \
  --label escalator \
  --output reports/warm_escalator.json
```

## Safe inference

```bash
python main.py \
  --config config/jetson.json \
  --metadata samples/elevator_button.json \
  --image samples/elevator_button.png \
  --safe \
  --timeout-seconds 10 \
  --output outputs/safe_result.json
```

`--safe`는 VLM 이미지 처리나 추론이 실패하면 구조화된 YOLO·Optical Flow
metadata만 사용하는 fallback 결과를 저장합니다. `service_status`와 `error` 필드로
정상, degraded, unavailable 상태를 구분할 수 있습니다. timeout은 추론 완료 후
경과 시간을 판정하는 soft timeout이며 이미 시작된 CUDA kernel을 강제로 종료하지
않습니다. 강제 timeout에는 향후 프로세스 격리 worker가 필요하며, 자동 모델
재로딩은 현재 지원하지 않습니다.

## Stability benchmark

동일한 service와 모델로 장시간 반복 추론하면서 latency와 GPU memory drift,
fallback 및 오류 상태 비율을 기록합니다.

엘리베이터 버튼:

```bash
python scripts/benchmark_stability.py \
  --config config/jetson.json \
  --metadata samples/elevator_button.json \
  --image samples/elevator_button.png \
  --runs 20 \
  --warmup-runs 1 \
  --sleep-seconds 1 \
  --timeout-seconds 10 \
  --label elevator_button \
  --output reports/stability_elevator.json
```

에스컬레이터:

```bash
python scripts/benchmark_stability.py \
  --config config/jetson.json \
  --metadata samples/escalator.json \
  --image samples/escalator.png \
  --runs 20 \
  --warmup-runs 1 \
  --sleep-seconds 1 \
  --timeout-seconds 10 \
  --label escalator \
  --output reports/stability_escalator.json
```

## Phase 5 latency breakdown

Production baseline을 변경하지 않는 token, visual input, crop, compact prompt,
attention, cache, dtype/quantization 실험은
[`docs/phase5_latency_optimization.md`](docs/phase5_latency_optimization.md)를
따릅니다. 상세 profiler는 elevator와 escalator를 같은 모델 instance로 측정하며
Safety Validator 통과율과 fallback 결과까지 함께 저장합니다.

## Integration adapter

이 단계는 실제 YOLO/OpenCV 구현과 분리된 계약 Adapter입니다. 외부 모듈은 JSON
또는 같은 Python dict 구조로 결과를 전달할 수 있고, Adapter 출력은 기존
`validate_metadata()`를 반드시 통과합니다. 실제 모듈 출력 형식이 다르면 VLM
서비스가 아니라 Adapter alias 또는 얇은 wrapper만 수정합니다.

엘리베이터 mock 실행:

```bash
python scripts/run_integration.py \
  --config config/pc.json \
  --image samples/elevator_button.png \
  --yolo-json samples/integration/yolo_elevator.json \
  --motion-json samples/integration/motion_unavailable.json \
  --quality-json samples/integration/quality_clear.json \
  --metadata-output outputs/integration_elevator_metadata.json \
  --output outputs/integration_elevator_result.json
```

에스컬레이터 mock 실행:

```bash
python scripts/run_integration.py \
  --config config/pc.json \
  --image samples/escalator.png \
  --yolo-json samples/integration/yolo_escalator.json \
  --motion-json samples/integration/motion_escalator_up.json \
  --quality-json samples/integration/quality_clear.json \
  --user-query "에스컬레이터 방향을 알려줘." \
  --metadata-output outputs/integration_escalator_metadata.json \
  --output outputs/integration_escalator_result.json
```

실제 Jetson에서는 같은 명령의 config만 `config/jetson.json`으로 지정합니다. 전체
좌표, alias, frame 동기화 계약은 `docs/integration_contract.md`에 정리되어 있습니다.

## Perception 연동 실행 구조

실제 카메라 파이프라인은 다음 순서로 동작합니다.

```text
Camera Frame
  ↓
YOLO + Optical Flow (mvp/escalator_mvp.py)
  ↓
mvp/perception_payload.py        yolo_payload / flow_payload 생성
  ↓
src/perception_adapter.py        Perception payload 정규화
  ↓
src/metadata_adapter.py          metadata schema 1.0
  ↓
VLMService.infer_safe()          모델 1회 로드 후 재사용
  ↓
Safety Validator + fallback
  ↓
최종 JSON (message만 사용자 안내에 사용)
```

Perception payload는 두 표기를 모두 받습니다.

| 입력 | detection 키 |
|---|---|
| Perception `process_split()` | `cls_name`, `conf`, `x1`, `y1`, `x2`, `y2` |
| 기존 내부 형식 | `class_name`, `confidence`, `bbox` |

두 표기가 함께 들어오면 값이 같을 때만 허용하고, 서로 다른 값이면 추측하지 않고
`YoloResultValidationError`를 발생시킵니다. 내부 metadata 형식은 기존 한 가지를
그대로 유지합니다.

- `ok=false`: VLM을 호출하지 않고 `fallback_reason="yolo_unavailable"` 또는
  `"perception_unavailable"` 결과를 만듭니다(`service_status="degraded"`).
- Flow `available=false`, `direction=unknown`, 미지원 direction: 방향을 사용하지
  않고 객체 위치만 안내합니다. `stopped`와 `unknown`은 구분합니다.
- `frame_id` 불일치: non-strict는 motion만 내리고, strict는
  `FrameSynchronizationError`를 냅니다.
- YOLO payload의 `quality`는 내부 `image_quality`로 옮기고, schema가 지원하지
  않는 필드는 버립니다.

Python에서 직접 호출할 때는 `VIAssistVLMPipeline.process_perception()`을 씁니다.

```python
result = pipeline.process_perception(
    image_path=image_path,
    yolo_payload=yolo_payload,
    flow_payload=flow_payload,
    user_query="에스컬레이터 방향을 알려줘.",
    timeout_seconds=10,
)
```

## Mock 통합 실행과 테스트

카메라, CUDA, YOLO 모델 없이 전체 흐름을 검증합니다.

```bash
# vlm 단위 + Perception adapter + mock end-to-end
cd vlm
python -m unittest discover

# MVP payload 변환 + VLMBridge mock 통합
cd ../mvp
python -m unittest discover -s tests -t .
```

`vlm/tests/test_perception_end_to_end.py`와
`mvp/tests/test_mvp_vlm_integration.py`는 mock 엔진으로
Perception → Adapter → VLMService → Safety Validator → 최종 JSON 전 구간을
확인합니다.

## Jetson 실기 실행

요청 기반 VLM 안내를 포함한 MVP 실행(모델은 시작 시 한 번만 로드):

```bash
cd mvp
python3 escalator_mvp.py \
  --model best.pt \
  --camera 0 \
  --port 5000 \
  --enable-vlm \
  --vlm-config ../vlm/config/jetson.json \
  --vlm-timeout 10
```

- `POST /vlm/describe`: 현재 동기화된 payload로 1회 추론
- `GET /perception_payload`: 최신 yolo/flow payload 확인
- 동시 요청은 lock으로 직렬화하며 이미 실행 중이면 `429`를 반환합니다.

웹캠 VLM 점검 도구:

```bash
cd mvp
python3 vlm_webcam_test.py --config ../vlm/config/jetson.json          # 안전 안내 모드
python3 vlm_webcam_test.py --allow-dev-describe                        # 개발 전용 자유 설명 모드
```

기본 `guide` 모드는 `VLMService.infer_safe()`와 Safety Validator를 통과한
`message`만 노출합니다. `dev_describe`는 안전 검증을 거치지 않는 개발 전용
경로이며 결과를 `dev_raw_text`로만 반환합니다. 사용자 안내나 TTS에 사용하면
안 됩니다.

## 검증 상태

mock으로 검증된 항목:

- Perception payload 두 표기 호환과 alias 충돌 거부
- `ok=false`, 빈 detections, 미지원 class, malformed bbox, frame 불일치 처리
- Perception → Adapter → VLMService → Safety Validator → 최종 JSON 전체 흐름
- 임시 이미지 파일 생성·삭제와 동시 요청 직렬화
- 모델 1회 로드 후 재사용

Jetson 실기 검증 대기 항목:

- 실제 카메라·YOLO 가중치와 연결한 `--enable-vlm` 실행
- 실제 프레임 기준 end-to-end latency와 GPU peak memory
- 에스컬레이터 실측에서의 fallback 비율 재측정
- 실제 이미지 품질(`quality`) 지표 연결

## 미구현 설정

- `quantization`: 양자화는 구현되어 있지 않습니다. `"none"` 외의 값은
  `ConfigValidationError`로 거부합니다. 4-bit 양자화는 다음 단계 작업입니다.
- `temperature`: 엔진은 greedy decoding(`do_sample=False`)만 사용하므로 설정
  자체를 거부합니다. 설정 파일에서 제거하세요.
