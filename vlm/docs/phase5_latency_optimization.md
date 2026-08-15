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
  `do_image_splitting`을 지원하며 기본값은 `True`이다.
- KV cache는 profiler가 `model.config.use_cache`,
  `generation_config.use_cache`, 실제 generate의 `use_cache=True`를 각각 기록한다.
  로컬 config에서는 두 `use_cache` 값이 이미 `True`이므로 baseline은 cache 미사용이
  아니다. Transformers 4.49의 현재 Idefics3 class는 `_supports_static_cache=False`여서
  StaticCache/StaticCache+compile은 unsupported 결과로 남기며 강제 적용하지 않는다.

## 실행 원칙

각 명령은 variant 하나만 변경한다. 같은 power mode와 clock/thermal 조건에서
elevator/escalator 각각 warm-up 2회 후 20회 측정한다.

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
