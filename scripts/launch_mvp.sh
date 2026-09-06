#!/usr/bin/env bash
# 전체 스택(YOLO+Flow+VLM+STT+TTS+/test 대시보드) 재기동. 기동 약 2분.
#   bash scripts/launch_mvp.sh            # 기본
#   STT_MODEL=small bash scripts/launch_mvp.sh
# pkill -f 패턴이 이 스크립트 자신을 죽이지 않도록 [e]로 감싼다(자기 명령줄엔 escalator_mvp.py 문자열이 없다).
set -u
REPO=$(cd "$(dirname "$0")/.." && pwd)
LOG=${LOG:-$HOME/mvp_server.log}
STT_MODEL=${STT_MODEL:-base}
VLM_CONFIG=${VLM_CONFIG:-../vlm/config/jetson.json}
PORT=${PORT:-5000}

pkill -f "[e]scalator_mvp.py" 2>/dev/null; pkill -f "[m]elo_worker.py" 2>/dev/null; sleep 3
cd "$REPO/mvp" || exit 1
: > "$LOG"
setsid nohup python3 escalator_mvp.py --enable-voice --enable-vlm --stt-model "$STT_MODEL" \
  --vlm-config "$VLM_CONFIG" --port "$PORT" >> "$LOG" 2>&1 < /dev/null &
disown
echo "launched pid=$!  log=$LOG  →  http://$(hostname -I | awk '{print $1}'):$PORT/test"
