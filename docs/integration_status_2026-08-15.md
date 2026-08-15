# VIAssist 통합 상태 (2026-08-15 기준)

이 문서는 2026년 8월 15일 시점의 실제 저장소 상태를 기록한다.
`vlm/reports/repository_review.md`는 과거 시점 문서이므로 현재 상태의 근거로
사용하지 않는다.

## 1. 저장소 구조

| 경로 | 상태 |
|---|---|
| `/home/edge/viassist` | Git 저장소 아님 |
| `/home/edge/viassist/vlm` | 독립 Git 저장소 (`main`) |
| `/home/edge/viassist/mvp` | 카메라·YOLO·Optical Flow MVP, Git 추적 없음 |
| `/home/edge/viassist/docs` | 보고서와 Perception 연동 계약 |

`vlm/reports/`의 stability·warm benchmark JSON은 미추적 상태이며 보존 대상이다.

## 2. 이번 단계에서 완료한 것

- Perception `process_split()` payload(`cls_name`/`conf`/`x1..y2`)와 기존 내부
  형식(`class_name`/`confidence`/`bbox`)을 모두 받는 호환 계층
- alias 충돌, 부분 좌표, `ok=false`, 빈 detections, 미지원 class, malformed bbox,
  frame_id 불일치에 대한 명시적 처리와 테스트
- Perception 전용 Flow 정규화(`available`, `unknown`/`stopped` 구분, 미지원
  direction 처리)
- `VIAssistVLMPipeline.process_perception()`: 실패 시 VLM을 호출하지 않는
  결정적 degraded 결과
- `mvp/perception_payload.py`: 카메라·모델에 의존하지 않는 payload 변환 함수
- `mvp/vlm_bridge.py`: 모델 1회 로드, 요청 기반 추론, lock 직렬화, 임시 이미지
  정리, `infer_safe()` 경로 강제
- `mvp/escalator_mvp.py`: YOLO가 실제로 실행된 프레임만 VLM 요청 대상으로 저장,
  `POST /vlm/describe`, `GET /perception_payload`, 중복 종료 호출 정리
- `mvp/vlm_webcam_test.py`: 안전 안내 모드와 개발 전용 자유 설명 모드 분리,
  raw 출력은 디버깅 영역에서만 노출
- 미구현 설정(`quantization`, `temperature`) 명시적 거부

## 3. 검증 상태

### mock으로 검증됨 (카메라·GPU 불필요)

- `vlm`: `python -m unittest discover`
- `mvp`: `python -m unittest discover -s tests -t .`
- Perception → Adapter → VLMService → Safety Validator → 최종 JSON 전체 흐름
- 안전하지 않은 VLM 문장이 사용자 안내로 나가지 않는 것
- 동시 요청 직렬화와 임시 파일 정리

### Jetson 실기 검증 대기

- 실제 카메라 + `best.pt` + `--enable-vlm` 통합 실행
- 실제 프레임 기준 end-to-end latency (기존 warm latency 4.6~5.6초)
- 에스컬레이터 실측 fallback 비율 재측정 (직전 실측은 100% fallback)
- GPU peak memory와 장시간 안정성
- 실제 `quality`(blur) 지표 연결 후 흐린 화면 정책 동작

## 4. 알려진 한계

- 현재 공식 지원 target은 `elevator_button`, `escalator` 두 가지뿐이다.
- `distance_m`은 안내에 사용하지 않는다.
- Flow `speed`는 `px/frame @ 320px`이며 물리 속도가 아니다.
- timeout은 추론 완료 후 판정하는 soft timeout이다. 실행 중인 CUDA kernel을
  강제 종료하지 않는다.
- MVP는 아직 blur 등 이미지 품질 지표를 만들지 않아 `quality`가 비어 있다.
- Gemini 연동(`mvp/vlm_webcam_test_gemini.py`)은 개발 실험용이며 production
  경로가 아니다.

## 5. 다음 단계 (이번 범위 밖)

- STT/TTS/GPIO 버튼 연동
- elevator door, gate 등 신규 탐지 클래스와 상태 안내
- process 격리 기반 hard timeout과 자동 모델 재로딩
- 4-bit 양자화 실제 구현
- Flow ROI ↔ detection 정밀 매칭(`track_id` 우선순위)
- 실제 이미지 품질 지표 산출과 흐린 화면 정책 연결
