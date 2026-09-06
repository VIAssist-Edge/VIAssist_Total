#!/usr/bin/env bash
# 젯슨 Orin Nano 시스템 설정 — 2026-09-05 하루 4회 완전 정지를 겪고 정한 것들.
# 한 번만 root로 실행: sudo bash scripts/jetson_setup.sh
# 멱등이라 여러 번 실행해도 안전하다.
set -euo pipefail

if [ "$(id -u)" -ne 0 ]; then
  echo "root 권한이 필요합니다: sudo bash $0" >&2
  exit 1
fi

echo "== 1) earlyoom: 커널 OOM 킬러가 나서기 전에 가장 큰 프로세스를 죽여 시스템 정지를 막는다"
# zram 스왑은 RAM 안에 있어서 "스왑도 바닥날 때까지" 기다리면 이미 스래싱으로 멈춘 뒤다.
# -m 5: 가용 메모리 5%(≈380 MB) 미만이면 개입. 8%(608 MB)로 두면 전체 스택이 VLM 추론 중
#       정상적으로 880 MB까지 내려가고 VLM 재로드 때 그 밑으로 스쳐 서버가 죽었다(09-06 실측).
#       완전 정지가 났던 구간은 여유 50 MB 안쪽이라 5%면 충분히 앞서 개입한다.
# -s 100: 스왑 상태는 무시(zram은 RAM 안이라 기준이 못 됨), -r 60: 60초마다 상태 로그.
if ! command -v earlyoom >/dev/null 2>&1; then
  apt-get update -qq && apt-get install -y -qq earlyoom
fi
cat > /etc/default/earlyoom <<'EOF'
EARLYOOM_ARGS="-m 5 -s 100 -r 60"
EOF
systemctl enable --now earlyoom
systemctl restart earlyoom
sleep 1
journalctl -u earlyoom --no-pager -n 2 | tail -1

echo "== 2) journald 영속화: 재부팅 전 로그(정지 원인)가 남도록"
# Storage=auto만으로는 디렉터리를 만들어도 계속 /run(휘발)에 쓰는 경우가 있어 명시적으로 persistent로 둔다.
mkdir -p /var/log/journal /etc/systemd/journald.conf.d
cat > /etc/systemd/journald.conf.d/10-persistent.conf <<'EOF'
[Journal]
Storage=persistent
SystemMaxUse=300M
EOF
systemd-tmpfiles --create --prefix /var/log/journal || true
systemctl restart systemd-journald
journalctl --flush 2>/dev/null || true
if ls -d /var/log/journal/*/ >/dev/null 2>&1; then
  echo "  /var/log/journal/$(ls /var/log/journal | head -1) 준비됨 — 다음 정지 뒤 'journalctl -b -1' 로 원인 확인 가능"
else
  echo "  경고: 영속 저널 디렉터리가 아직 없음. 'journalctl --header' 로 Storage 확인 필요" >&2
fi

echo "== 3) 전원 모드 25W (MAXN_SUPER는 순간 25W+ → 약한 어댑터/배터리에서 리셋)"
if command -v nvpmodel >/dev/null 2>&1; then
  nvpmodel -m 1 >/dev/null 2>&1 || true
  nvpmodel -q | head -2
fi

echo "== 4) 확인"
echo "  earlyoom: $(systemctl is-active earlyoom)  ($(cat /etc/default/earlyoom))"
echo "  swap:"; cat /proc/swaps | tail -n +2 | awk '{print "    " $1, $3/1024 " MB"}'
free -m | awk '/Mem:/{print "  mem avail now: " $7 " MB"}'
echo "완료. 카메라는 부팅 전에 꽂고, 새 모델 실험은 대시보드 서버를 내린 뒤에."
