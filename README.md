# VIAssist — 아이즈온(Eyes-On) 젯슨 통합 코드

시각장애인 보행 보조 웨어러블의 온디바이스 파이프라인. NVIDIA Jetson Orin Nano(8GB) 한 대에서
**YOLO 탐지 → Optical Flow 방향 판정 → 규칙/VLM 라우터 → 음성 안내(STT/TTS)** 를 전부 돌리고,
개발자용 웹 대시보드(`/test`)로 모듈 하나하나를 따로 확인할 수 있다.

```
USB 카메라 ─┬─ YOLO(best.pt, 32클래스) ─┐
            └─ Optical Flow(Farneback + ego-motion 보정) ─┤
                                                         ▼
마이크 ── STT(faster-whisper) ── 질문 ──▶ 라우터 ──▶ 규칙 안내(0 ms) ─────────┐
                                             ├──▶ VLM 상태 질문(문 열림·표시등)  ├──▶ TTS(MeloTTS) ──▶ 스피커
                                             └──▶ VLM 장면 설명(탐지 밖의 것)  ┘
```

## 저장소 구성

| 경로 | 내용 |
|---|---|
| `mvp/escalator_mvp.py` | 메인 서버(Flask :5000). 카메라 스레드, YOLO+Flow 처리 루프, `/status` `/video_feed` `/guide` `/vlm/*` |
| `mvp/guidance_router.py` | **규칙 / VLM 상태 / VLM 장면** 자동 분기. `/guide/auto`, `/guide/decide` |
| `mvp/class_rules.py` | 클래스 33개의 한국어 이름·문장 형태·히스토리 길이. 규칙 안내 템플릿. detection 표기 정규화 |
| `mvp/background_motion.py` | 카메라 자체 움직임(ego-motion) 제거 — 배경 특징점 homography, 실패 시 중앙값 폴백 |
| `mvp/perception_payload.py`, `motion_log.py` | Perception 계약(§4) payload 생성, 모션 로그 |
| `mvp/stt_service.py` | faster-whisper(base, int8, CPU). VAD, 44.1 kHz→16 kHz, 스트리밍 부분 인식 |
| `mvp/tts_service.py`, `melo_worker.py` | MeloTTS-Korean(CUDA, `~/melo_env` 별도 프로세스) + gTTS 폴백, 합성 캐시 |
| `mvp/voice_endpoints.py` | `/voice/listen` `/voice/stream` `/voice/say` `/voice/ask`(기본 mode=auto → 라우터) |
| `mvp/vlm_bridge.py`, `failure_notice.py`, `slot_schema.py` | VLM 파이프라인 연결, 실패 사유 안내, 슬롯 프롬프트 |
| `mvp/module_test.py`, `module_test.html` | **모듈 점검 대시보드** `/test` (아래) |
| `mvp/tests/` | 남은 모듈의 단위 테스트 |
| `vlm/` | VLM 파이프라인(SmolVLM-500M 기본, Gemini 엔진 선택). `src/`(config·프롬프트·안전 규칙·결과 파서), `config/{jetson,pc,gemini}.json`, `samples/`, `tests/`, `docs/` |
| `docs/` | 중간보고서, Perception 연동 계약, 연동 현황 |
| `scripts/jetson_setup.sh` | 시스템 안정화 1회 설정(sudo): earlyoom, journald 영속화, 25W 전원 모드 |
| `scripts/launch_mvp.sh` | 전체 스택 재기동 런처(세션 분리, 로그 `~/mvp_server.log`) |

추적하지 않는 것(.gitignore): 모델 가중치(`*.pt` `*.engine` `*.onnx`), `vlm/.venv`, `vlm/.env`(API 키), `mvp/reports/`(벤치·오류 기록·영상), 로그.
실험·벤치 코드는 젯슨 `~/han/archive_2026-09-05/`에 보관(목록 `MOVED.txt`).

## 젯슨에서 실행

전제: JetPack(L4T R36.5.2), Python 3.10, torch 2.3.0(CUDA 12.6), ultralytics, transformers 4.49, faster-whisper,
`~/melo_env`(MeloTTS 전용 venv), `mvp/best.pt`(팀 학습 YOLO 32클래스). VLM 의존성은 `vlm/requirements-jetson.txt`.

```bash
# 0) 처음 한 번: 시스템 안정화(earlyoom·journald 영속화·25W) — 아래 "안정성" 참조
sudo bash scripts/jetson_setup.sh

# 1) 전체 스택(YOLO+Flow+VLM+STT+TTS+대시보드) — 기동 약 2분 (VLM 30 s + 음성 60 s)
bash scripts/launch_mvp.sh            # 세션과 분리해 띄우고 ~/mvp_server.log 에 로그

# 직접 띄울 때
cd mvp && python3 escalator_mvp.py --enable-voice --enable-vlm --stt-model base \
    --vlm-config ../vlm/config/jetson.json --port 5000
```

주요 옵션: `--model`(기본 best.pt) `--camera N` `--conf 0.45` `--imgsz 640` `--yolo-every 3`(3프레임마다 추론)
`--flow-width 320` `--direction-history` `--direction-majority` `--stt-model {tiny,base,small}` `--tts-engine {melo,gtts}` `--vlm-timeout`.
`--enable-vlm`/`--enable-voice`를 빼면 해당 모듈 없이 뜬다(카메라+YOLO+Flow만: 메모리 0.9 GB).

카메라가 없어도 서버는 뜬다(`camera_ok=false`). 대시보드 **영상 입력** 탭에서 녹화 mp4를 카메라 대신 주입해 같은 파이프라인을 돌릴 수 있다.

## 웹 화면

- `http://<젯슨>:5000/` — 기존 MVP 화면(스트림, 방향, VLM 버튼, 말로 물어보기)
- `http://<젯슨>:5000/test` — **모듈 점검 대시보드**

| 탭 | 확인할 수 있는 것 |
|---|---|
| 개요 | 5모듈 상태 배지, 통합 DRAM/CUDA/스왑/온도, 처리 스레드 생존 |
| 영상 입력 | mp4 업로드·재생·구간 이동, 카메라 ↔ 영상 전환, 방향 판정 타임라인 |
| YOLO | 박스 스트림, 클래스·신뢰도·FPS·추론 ms, 이미지 업로드 추론 |
| Optical Flow | 벡터 시각화 스트림, dy·magnitude·ego 보정 상태, flow 계산 시간 |
| VLM | 현재 프레임 캡처 → 장면 설명/상황 안내, 응답 원문·지연, **설정별 벤치 이력**(파일 저장) |
| STT | 마이크 레벨, VAD/고정 녹음, 파형·인식 텍스트, wav 업로드 인식 |
| TTS | 문장 합성 → 브라우저 청취, 젯슨 스피커 재생, 캐시 여부 |
| 파이프라인 시간 | 한 번 실행하며 단계별 ms 워터폴 + **라우터가 고른 경로·이유** |
| 오류 | 라우트 예외·로거 ERROR·스레드 예외·브라우저 JS 오류를 traceback과 함께 영구 기록(`reports/module_test_errors.jsonl`) |
| 로그 | 최근 로그 링버퍼 |

## 라우터 규칙 (`guidance_router.decide`)

1. 질문에 상태 단어(열렸/작동/표시/글자/상행/하행…) → **VLM 상태 질문** (YOLO 컨텍스트 첨부)
2. 질문에 탐지된 시설의 이름("에스컬레이터 어디야") → **규칙** (위치·방향은 규칙이 0 ms로 답함)
3. 장면/주변/위험을 묻는 열린 질문 → **VLM 장면 설명**
4. 질문 없음·"안내해줘": 최우선 탐지가 signal(문 상태·표시등·신호등)이면 VLM 상태 질문, 다른 클래스면 규칙, 탐지 없으면 VLM 장면 설명
5. VLM 비활성이면 항상 규칙

실측(젯슨, SmolVLM): 규칙 1.4 ms, VLM 4~5 s. 응답에 `route`/`reason`이 실려 대시보드에서 확인할 수 있다.

## 실측 수치 (2026-09-05, Orin Nano 8GB)

**메모리(통합 DRAM 7,607 MB — GPU 가중치도 같은 예산)**

| 구성 | 사용(누적) | 여유 |
|---|---|---|
| OS·기본 서비스 | 1,293 MB | 6,088 MB |
| + YOLO + Optical Flow | 2,184 MB | 5,198 MB |
| + STT + TTS(MeloTTS 워커 ≈ 2 GB) | 4,295 MB | 3,085 MB |
| + SmolVLM-500M (전체 스택) | ≈ 6,000 MB | ≈ 1,300 MB |

**VLM 후보 비교(같은 벤치: 라벨 42장 슬롯 + 정성 44회)**

| 모델 | 상주 | 호출 | 생성 | 슬롯 파싱 / object / 둘 다 | 비고 |
|---|---|---|---|---|---|
| SmolVLM-500M (현재) | 1.7 GB | 4.3 s | — | 7% / 0% / 0% | 한국어 생성 불가 |
| Qwen3-VL-4B Q4_K_M (llama.cpp) | 3.0 GB | 2.8 s | 16.7 tok/s | 71% / 14% / 5% | 위험 요소 환각 |
| Qwen3-VL-8B Q4_K_M (llama.cpp) | 5.5 GB | 3.1~4.6 s | 11.4 tok/s | 93% / 24% / 14% | 닫힌 상태 질문에 정확, 열린 위험 질문에 약함 |

8B는 전체 스택과 동시 적재 불가(5.5 GB > 여유 3.1 GB). DFloat11(무손실 70%)은 8B=12.3 GB로 통합 메모리에 불가.
TTS 경량화 실험: piper(한국어 커뮤니티 음성)는 343 MB·0.7 s지만 음절 14% 탈락으로 기각, MMS-TTS-kor int8(CPU) 11~15 s로 기각,
MMS-TTS-kor fp32를 GPU·인프로세스로 돌리면 +1.4 GB·0.5~0.7 s. 벤치 스크립트와 결과는 젯슨 `~/vlm8b/`.

## 안정성 — 꼭 읽을 것

젯슨은 **microSD 루트, zram 스왑(RAM 안), 전원 모드 MAXN_SUPER** 상태였고 2026-09-05 하루 4번 완전 정지했다.
메모리가 바닥나면 OOM 킬러가 나서기 전에 zram 압축 스래싱으로 시스템 전체가 멈춘다(GPU 매핑 메모리는 스왑 불가).

- 가용 메모리 **350 MB 미만이면 VLM 호출을 차단**하는 가드가 대시보드·라우터에 있다(`MEMORY_GUARD_MB`).
- 새 모델을 올리는 실험은 **반드시 대시보드 서버를 내린 뒤**, 여유 1.5 GB 미만이면 중단.
- llama.cpp 서버는 `--cache-ram 0` 필수(없으면 호출마다 35 MB 누적 → OOM), 입력 이미지는 크기를 고정(모양이 바뀌면 그래프 버퍼 재할당).
- 시스템 설정은 `sudo bash scripts/jetson_setup.sh` 한 번: `earlyoom`(메모리 8%에서 개입, 스왑 무시), `journald` 영속화(`/var/log/journal`), `nvpmodel -m 1`(25 W).
- 서버 실행 중 카메라 핫플러그 금지 — 정지 사례 1회. 부팅 전에 꽂는다.
- 젯슨이 응답을 잃으면 USB-C(A-to-C) 케이블로 PC에 연결 → USB 장치 모드(SSH `192.168.55.1`, 시리얼 COM, `L4T-README`)로 Wi-Fi 없이 접속 가능.
- `pkill -f 패턴`이 자기 ssh 명령줄과 겹치면 세션이 죽는다 → 재시작은 스크립트 파일(`~/launch_mvp_test.sh`)로.

## 관련 문서

- `docs/perception_integration_contract.md` — Perception(YOLO/Flow) ↔ VLM payload 계약
- `vlm/docs/` — VLM 통합 계약, 지연 최적화 기록
- 프롬프트 v2 설계·개발보고서 수정 체크리스트 — 팀 공유 문서(2026-09-05)
