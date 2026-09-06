# Phase 5: Jetson VLM latency optimization

이 단계의 코드는 production `config/jetson.json`, `VLMService.infer()`,
`VLMService.infer_safe()`를 변경하지 않는다. 모든 변경은
`config/latency_experiments.json`의 독립 variant와 profiler에 한정된다.

## Baseline 확인 사항

- Production image processor 값은 `longest_edge=1024`, `max_image_size=512`로 유지된다.
- `SmolVLMEngine`은 CUDA에서 모델을 `torch.float16`으로 로드한다.
- 현재 production config의 quantization은 `none`이며 엔진은 quantized load를 하지
  않는다. 과거 또는 실험 config의 quantization 문자열만으로 실제 양자화를
  단정하지 않는다. 실제 판정은 parameter dtype, quantization config,
  `Linear4bit`/`Linear8bitLt` 모듈을 검사한다.
- Transformers 4.49의 `Idefics3ImageProcessor`는
  `do_image_splitting`을 지원하며 기본값은 `True`이다. 설치된 4.49.0 소스에서
  생성자와 `preprocess()` 인자, 실제 image splitting 분기를 확인했다. Runner도
  model load 전에 processor instance에 이 속성이 없으면 추측 적용하지 않고
  실패한다.
- 설치된 Transformers 4.49.0 소스와 실제 SmolVLM processor로 384/256을 확인했다.
  splitting ON에서 두 sample 모두 각각 `[1, 1, 3, 384, 384]`,
  `[1, 1, 3, 256, 256]`을 반환하며 내부적으로 512로 올리지 않는다. Model config의
  `patch_size=16`, `scale_factor=4`에도 두 해상도가 나누어떨어진다. Idefics3의
  문서화된 image sequence 계산식에 따라 `image_seq_len`은 각각 36/16이어야 한다.
  Runner는 이 값을 resolution에서 계산하고 실제 image token 수까지 model load 전에
  확인하며, shape/token 계약이 다르면 우회하지 않고 실패한다.
- KV cache는 profiler가 `model.config.use_cache`,
  `generation_config.use_cache`, 실제 generate의 `use_cache=True`를 각각 기록한다.
  로컬 config에서는 두 `use_cache` 값이 이미 `True`이므로 baseline은 cache 미사용이
  아니다. Transformers 4.49의 현재 Idefics3 class는 `_supports_static_cache=False`여서
  StaticCache/StaticCache+compile은 unsupported 결과로 남기며 강제 적용하지 않는다.

## 실행 원칙

각 명령은 variant 하나만 변경한다. 같은 power mode와 clock/thermal 조건에서
elevator/escalator 각각 warm-up 2회 후 20회 측정한다.

### Image resolution 실험

`resolution_768`과 `resolution_512`는 baseline의
`image_longest_edge=1024`, `max_image_size=512`를 기준으로 longest edge만 각각
768과 512로 변경한다. Catalog를 읽을 때 두 variant의 모든 나머지 inference
setting을 baseline과 필드별로 비교하므로, `max_new_tokens`, dtype, attention,
cache, prompt, crop 또는 image splitting 등 다른 값이 달라지면 모델을 로드하기
전에 실패한다. Production `config/jetson.json`은 읽기만 하며 수정하지 않는다.

```bash
python scripts/benchmark_latency_breakdown.py \
  --variant resolution_768 --warmup-runs 2 --runs 20 \
  --output reports/latency_optimization/resolution_768.json

python scripts/benchmark_latency_breakdown.py \
  --variant resolution_512 --warmup-runs 2 --runs 20 \
  --output reports/latency_optimization/resolution_512.json
```

각 report는 기존 baseline과 동일하게 run별 `total_ms`, `generate_ms`,
`first_token_ms`, input/generated token count, `pixel_values_shape`,
`tokens_per_second`, GPU peak memory, Validator/fallback 결과와 raw VLM output을
보존한다.

실측 결과 `resolution_768`은 baseline보다 느려 reject했고, `resolution_512`는
latency candidate로 유지한다. Production config는 변경하지 않는다.

### 512 미만 input resolution 실험

Prompt optimization은 종료한다. `compact_prompt_512`와 `dedup_prompt_512`는 모두
reject하며 production prompt를 현재 최적 prompt로 고정한다. 새
`resolution_384`/`resolution_256`은 `resolution_512`와 비교해 다음 두 resolution
값만 함께 낮춘다.

| Variant | image_longest_edge | max_image_size |
|---|---:|---:|
| resolution_512 | 512 | 512 |
| resolution_384 | 384 | 384 |
| resolution_256 | 256 | 256 |

Catalog load 단계에서 production prompt, splitting ON, crop OFF,
`max_new_tokens=60`, dtype, attention implementation, Dynamic KV cache, compile과 모든
generation option을 `resolution_512`와 필드별로 비교한다. Resolution 외 값이
다르면 model load 전에 실패한다. Processor도 model load 전에 requested/actual
`pixel_values.shape`, patch/scale compatibility와 image token 수를 검증한다.

Report는 기존 schema와 run별 timing/token/GPU/Validator/fallback/raw output 필드를
그대로 유지한다. Summary는 sample별 실제 shape, input tokens, first-token/generate/
total latency, generated tokens, GPU peak, raw output, Validator/fallback, validation
reasons, truncation과 metadata hallucination을 512/384/256 사이에 비교한다.

Elevator는 Validator 100%, fallback 0%를 반드시 유지해야 하며 target/position
불일치, hallucination, truncation 또는 의미 변화가 있으면 속도와 관계없이 reject한다.
Raw output이 달라지면 summary가 사람의 의미 검토 필요 여부를 표시한다. Escalator의
의미 검토가 끝나기 전 candidate verdict는 `pending_semantic_review`로 남는다. Escalator의
기존 fallback 100% 자체는 candidate reject 사유가 아니지만, baseline보다 Validator/
fallback/truncation/hallucination/target/position이 나빠지면 신규 regression으로
기록하고 reject한다.

```bash
python scripts/summarize_latency_reports.py \
  reports/latency_optimization/resolution_512.json \
  reports/latency_optimization/resolution_384.json \
  reports/latency_optimization/resolution_256.json
```

### Image splitting 단일 변수 실험

다음 실험은 `resolution_512`를 splitting ON 기준으로 사용하고
`splitting_off_512`에서 `do_image_splitting=False`만 변경한다. 두 variant 모두
`image_longest_edge=512`, `max_image_size=512`, `max_new_tokens=60`, FP16, SDPA,
Dynamic KV cache, 기존 prompt/Validator/fallback/result schema를 유지한다. Catalog
load 단계에서 모든 experiment option을 필드별로 비교하므로 splitting 이외 설정이
달라지면 model load 전에 실패한다.

```bash
python scripts/benchmark_latency_breakdown.py --variant splitting_off_512 --warmup-runs 2 --runs 20 --output reports/latency_optimization/splitting_off_512.json
```

ON/OFF report를 함께 summary에 전달하면 elevator/escalator별
`input_token_count` 평균과 `pixel_values.shape`가 별도 비교 표로 출력된다.

```bash
python scripts/summarize_latency_reports.py reports/latency_optimization/resolution_512.json reports/latency_optimization/splitting_off_512.json
```

### Compact prompt 단일 변수 실험

현재 production과 `resolution_512` benchmark가 공통으로 사용하는
`build_prompt()`의 전체 template은 다음과 같다. `{metadata_text}`는 들여쓰기한
JSON이며 `user_query`, `image_quality`, 모든 detection의 `class_name`, `confidence`,
`position`, `position_ko`, `bbox`, 그리고 motion의 `available`, `direction`,
`direction_ko`, `speed`를 포함한다. `{user_query}`는 같은 질문을 마지막에 한 번 더
넣으므로 실제 질문 문자열은 prompt에 두 번 나타난다.

```text
You are an assistant for safe walking guidance for a visually impaired user.

Analyze the image together with the structured vision analysis below.

Structured vision analysis:
{metadata_text}

Follow these rules strictly:

1. Answer in Korean.
2. Use only one or two short sentences.
3. Include only information needed for walking.
4. Use 왼쪽, 정면, 오른쪽 for object position.
5. The object class and position from detections take priority over your visual guess.
6. Escalator direction or door movement must only come from motion data.
7. Do not invent distance, movement, safety, or object state.
8. If information is uncertain, say "확인하기 어렵습니다."
9. Do not describe irrelevant background details.
10. Return only the guidance sentence, not JSON and not an explanation.
11. 사용자에게 질문하지 마세요.
12. 사용자 질문을 그대로 반복하지 마세요.
13. 반드시 평서형 안내 문장으로 답하세요.
14. "~있는 것입니다" 같은 번역체를 사용하지 마세요.
15. "~있습니다" 형태의 자연스러운 안내 문장을 사용하세요.
16. Optical Flow 방향이 제공되면 에스컬레이터 안내에 반드시 포함하세요.

User question:
{user_query}
```

Production engine과 profiler는 이 text 앞에 동일한 image content를 둔 user message를
만들고 `apply_chat_template(..., add_generation_prompt=True)`로 최종 model input을
조립한다. 따라서 아래 실험 override가 바꾸는 것은 image나 chat wrapper가 아니라
text prompt 내용뿐이다.

`compact_prompt_512`는 production prompt를 수정하지 않고 profiler의 기존
experimental prompt branch만 사용한다. `select_detection()`이 선택한 YOLO 대상과
metadata position, `motion.available=true`이고 direction이 유효할 때의 Optical Flow
방향만 전달한다. bbox, confidence 숫자, blur score, speed, user query와 중복된 영어
서론/규칙은 전달하지 않는다. 낮은 confidence, detection 없음, blurry image는
`확인상태=불확실`로만 전달하며 최종 판단과 fallback은 기존 Safety Validator가
그대로 수행한다. 실제 compact template은 다음과 같다.

```text
한국어 보행 안내를 평서문 한 문장으로만 출력하세요.
탐지값만 사실로 사용하세요: 대상={target_ko}, 위치={position_ko}[, 움직임={motion_ko}][, 확인상태=불확실].
객체·위치와 제공된 움직임 외에는 추측하지 마세요. 불확실하면 확인하기 어렵다고 안내하세요.
질문·설명·추론·서론·JSON·마크다운·행동 지시는 금지합니다. 거리·안전 여부·미제공 상태를 만들지 말고 자연스러운 평서문으로 끝내세요.
```

Catalog load 시 `resolution_512`와 `compact_prompt_512`의 image size, splitting,
crop, `max_new_tokens`, dtype, attention, cache, compile을 포함한 모든 option을
비교한다. prompt flag 외 차이가 있으면 model load 전에 실패한다. 두 variant는
CUDA runner에서 동일한 FP16 mapping과 기존 SDPA/Dynamic KV runtime을 사용하고,
`do_sample=False`, `num_beams=1`, `use_cache=True` generation 계약도 같은 공유
상수에서 가져온다.

두 report를 summary에 함께 전달하면 elevator/escalator 각각에 대해 input token
감소 수/감소율, generated token 변화, raw output, Validator/fallback 변화와 validation
reason을 비교한다. generation limit 도달, metadata와 다른 객체·위치·방향 주장,
Elevator Validator pass 하락은 compact verdict를 자동 reject한다. 또한 기존
`앞에 에스컬레이터가 있나요?`가 질문부호와 `question_form_output`이 없는 평서문으로
바뀌었는지 별도 boolean으로 기록한다.

```bash
python scripts/summarize_latency_reports.py reports/latency_optimization/resolution_512.json reports/latency_optimization/compact_prompt_512.json
```

실측에서 `compact_prompt_512`는 input token과 first-token latency는 줄였지만 두
sample 모두 `generated_tokens=60`에 도달했고 Validator가 실패했다. instruction
모방, 비정상 문장, possible truncation, metadata hallucination이 발생했으므로 이
variant는 reject한다.

### Deduplicated production prompt 단일 변수 실험

`dedup_prompt_512`는 새 prompt를 설계하지 않는다. 위에 기록한 production
`build_prompt()` 결과를 그대로 생성한 다음 structured JSON의 다음 한 줄만
제거한다.

```text
  "user_query": "...",
```

정확한 중복 감사 결과는 다음과 같다.

- 질문 문자열은 structured JSON의 `user_query`와 마지막 `User question:` 값에
  동일하게 두 번 들어가므로 앞의 JSON 필드만 제거한다.
- `position`/`position_ko`와 `direction`/`direction_ko`는 동일 문자열이 아니라
  source code와 한국어 표시값의 쌍이므로 유지한다.
- 규칙 2/10, 7/8, 11~15는 의미가 일부 겹쳐도 각각 문장 수, 출력 형식, 추측 금지,
  uncertainty fallback, 질문/반복 금지, 평서형과 문체를 별도로 규정하므로 모두
  유지한다.
- 영어 서론, 16개 rule의 wording와 순서, 마지막 `User question:`, structured
  metadata의 나머지 필드와 JSON formatting은 production prompt와 동일하다.

실험 함수는 production prompt 안에서 제거할 JSON line을 정확히 한 번 찾지 못하면
실패한다. Catalog load는 `resolution_512`와 비교해 `deduplicated_prompt` flag 외
모든 option이 같은지 확인한다. 따라서 image 512/512, splitting ON, crop OFF,
`max_new_tokens=60`, FP16 CUDA mapping, SDPA runtime, Dynamic KV cache와 generation
options가 달라지면 model load 전에 실패한다. Production prompt와
`config/jetson.json`은 변경하지 않는다.

Summary는 sample별 input tokens와 감소율, first-token/total/generate latency,
generated tokens, raw VLM message, Validator/fallback, validation reasons, truncation,
metadata hallucination을 비교한다. `Escalator known question -> statement`는 candidate가
Validator PASS이고 `question_form_output`이 없으며 truncation과 metadata
hallucination도 없을 때만 true이다.

Deduplicated prompt verdict는 다음 중 하나라도 발생하면 reject한다.

- Elevator Validator pass가 100% 미만
- Elevator fallback이 0% 초과
- sample output truncation 또는 metadata hallucination
- 같은 sample에서 generation limit에 2회 이상 도달
- sample total latency가 `resolution_512`보다 5% 초과 증가

```bash
python scripts/summarize_latency_reports.py reports/latency_optimization/resolution_512.json reports/latency_optimization/dedup_prompt_512.json
```

```bash
python scripts/benchmark_latency_breakdown.py \
  --variant baseline --warmup-runs 2 --runs 20 \
  --output reports/latency_optimization/baseline.json
```

동일한 명령에서 variant와 output만 다음 순서로 바꾼다.

```text
tokens_32, tokens_24, tokens_16
resolution_768, resolution_512
crop_512, no_image_splitting
compact_prompt
eager, sdpa
fp16, bf16, int8, nf4
```

`static_kv`, `static_kv_compile`은 catalog에 조사 후보로 남아 있지만 현재
Transformers 4.49 Idefics3가 지원하지 않아 runner가 명시적으로 거부한다.

`flash_attention_2`는 catalog 기본 실행 목록에 없다. 이미 설치되어 있고 현재
환경에서 Transformers의 지원 검사를 통과할 때만 별도 variant로 추가한다.
Profiler는 flash-attn이나 bitsandbytes를 자동 설치하지 않는다. INT8/NF4도 기존
bitsandbytes가 정상 동작할 때만 실행한다.

Profiler는 image load, crop/resize, prompt, processor, host-to-device, generate,
first token, approximate token decode, text decode, Safety Validator와 total 시간을
기록한다. CUDA 측정 전후 synchronize는 profiler 내부에만 있으며 production
inference에는 추가되지 않는다.

결과 표 생성:

```bash
python scripts/summarize_latency_reports.py \
  reports/latency_optimization/baseline.json \
  reports/latency_optimization/tokens_32.json \
  reports/latency_optimization/tokens_24.json \
  reports/latency_optimization/tokens_16.json
```

BEST는 Validator 100%, fallback 0%, error 0인 결과 중 평균 latency가 가장 낮은
variant만 자동 후보로 표시한다. 최종 채택 전에는 raw output의 객체·위치·방향
정확성과 반복 안정성을 사람이 함께 확인해야 한다.

## Jetson runtime 상태

Profiler는 값을 변경하지 않고 `nvpmodel -q`, `jetson_clocks --show`, tegrastats
1회 snapshot을 기록한다. 권한 때문에 상세 nvpmodel을 얻지 못하면 사용자가 직접
다음을 실행한다.

```bash
sudo nvpmodel -q --verbose
```

Profiler가 power mode나 clock을 자동 변경하지는 않는다. Thermal throttling을
명시적으로 판정할 근거가 없으면 `unknown`으로 기록한다.

## In-memory image 조사

현재 이 저장소의 integration path에는 `cv2.imwrite`가 없다. Pipeline과 engine
사이의 공개 계약은 `Path | None`이고 engine이 `PIL.Image.open()`을 호출한다.
따라서 외부 YOLO/OpenCV runtime이 ndarray를 파일로 저장하는지는 이 저장소만으로
확인할 수 없다. 우선 `image_load_ms`를 실측한다. 영향이 유의미할 때에만
`SmolVLMEngine.generate_image(PIL.Image.Image | np.ndarray, ...)` 같은 내부 선택
경로를 추가하고 기존 `generate(image_path, ...)`는 wrapper로 유지하는 방식이
backward compatible하다.

## Rule Fast Path 준비 분석

`src/quality_checker.py`의 `build_safe_fallback()`은 이미 검증 metadata와
`select_detection()`만으로 elevator/escalator 문장을 결정적으로 생성한다.
다음 Phase에서 Fast Path를 도입한다면 재사용 가능한 핵심 함수는
`validate_metadata()`, `select_detection()`, `build_safe_fallback()`,
`build_result()`이다.

예상 수정 범위는 `src/vlm_service.py` 또는 `src/integration_pipeline.py`의 선택 정책,
별도 fast-path policy 모듈, config, 테스트와 benchmark이다. VLM 호출 대부분을
건너뛰므로 절감 상한은 profiler의 VLM total과 거의 같지만 실제 수치는 Jetson에서
측정해야 한다. 반드시 YOLO 객체/원본 position, Optical Flow direction만 사용하고
Safety Validator/final JSON/service status/message source 계약을 유지해야 한다.
