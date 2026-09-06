#!/usr/bin/env bash
# 전체 스택(YOLO+Flow+VLM+STT+TTS+/test 대시보드) 재기동. 기동 약 2분.
#   bash scripts/launch_mvp.sh            # 기본
#   STT_MODEL=small bash scripts/launch_mvp.sh
# pkill -f 패턴이 이 스크립트 자신을 죽이지 않도록 [e]로 감싼다(자기 명령줄엔 escalator_mvp.py 문자열이 없다).
set -u
# VLM 유휴 언로드(초). 기본 0=끔. 실측(09-06) 언로드로 돌아오는 가용 메모리가 ~300 MB뿐이고 재로드 중 earlyoom에 죽을 수 있어 기본은 끈다.
export VLM_IDLE_UNLOAD_S=${VLM_IDLE_UNLOAD_S:-0}
export VLM_RELOAD_MIN_MB=${VLM_RELOAD_MIN_MB:-1200}
REPO=$(cd "$(dirname "$0")/.." && pwd)
LOG=${LOG:-$HOME/mvp_server.log}
STT_MODEL=${STT_MODEL:-base}
# 기본 VLM 백엔드는 llama.cpp(SmolVLM Q8, 0.6 s/호출). HF 경로로 돌리려면 VLM_CONFIG=../vlm/config/jetson.json
# 기본 VLM은 SmolVLM Q8(llama.cpp, 0.6 s/호출, 상주 1.1 GB) + MeloTTS. Qwen3-VL-4B Q4(상주 4.0 GB, 4.1 s/호출)로 쓰려면 MeloTTS(2.9 GB)를 못 띄우므로:
#   VLM_CONFIG=../vlm/config/jetson_qwen4b.json TTS_ENGINE=gtts bash scripts/launch_mvp.sh
# 8B Q4는 6.2 GB라 스택과 동시 불가(2026-09-06 실측).
VLM_CONFIG=${VLM_CONFIG:-../vlm/config/jetson_llamacpp.json}
PORT=${PORT:-5000}
# TTS 엔진: melo(기본, 워커 2.9 GB) 또는 gtts(클라우드, 메모리 거의 0). Qwen3-VL-4B를 쓸 땐 gtts 필수(메모리).
TTS_ENGINE=${TTS_ENGINE:-melo}

# llama-server는 VLM 엔진의 자식 프로세스라 부모가 죽으면 같이 정리되지만, 강제 종료 뒤 남았을 수 있어 함께 내린다.
pkill -f "[e]scalator_mvp.py" 2>/dev/null; pkill -f "[m]elo_worker.py" 2>/dev/null; pkill -x llama-server 2>/dev/null; sleep 3
cd "$REPO/mvp" || exit 1
: > "$LOG"
setsid nohup python3 escalator_mvp.py --enable-voice --enable-vlm --stt-model "$STT_MODEL" \
  --vlm-config "$VLM_CONFIG" --tts-engine "$TTS_ENGINE" --port "$PORT" >> "$LOG" 2>&1 < /dev/null &
disown
echo "launched pid=$!  log=$LOG  →  http://$(hostname -I | awk '{print $1}'):$PORT/test"
