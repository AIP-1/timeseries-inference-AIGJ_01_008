#!/bin/bash
# 심화 6·7 GPU 작업 실행기. 아래 조건을 모두 만족할 때만 작업 하나를 돌린다.
#   - 가용 메모리 MIN_AVAIL(기본 2500MB) 이상. 15:26 보드 정지 뒤 1.5GB → 2.5GB로 올림. 작은 모델만 다루는 심화 7은 2000MB
#   - 내 계정에서 서비스 말고 다른 python 작업이 없음(다른 세션의 작업과 겹치지 않게)
#   - 잠금 파일로 한 번에 하나만
#   ./b_gpu_run.sh <로그이름> <명령...>
cd "$(dirname "$0")"
LOG=report/$1.log; shift
AV=$(free -m | awk "/Mem/{print \$7}")
MIN=${MIN_AVAIL:-2500}
if [ "$AV" -lt $MIN ]; then echo "가용 메모리 ${AV}MB < ${MIN}MB, 실행 안 함" | tee $LOG; exit 1; fi
OTHER=$(pgrep -af "^/home/team2-2/venv/bin/python" | grep -v "service.py")
if [ -n "$OTHER" ]; then echo "다른 python 작업이 실행 중, 실행 안 함: $OTHER" | tee $LOG; exit 1; fi
exec 9>logs/b_gpu.lock
flock -n 9 || { echo "다른 GPU 작업이 실행 중" | tee $LOG; exit 1; }
echo "=== 시작 $(date +%T) 가용=${AV}MB 명령: $*" > $LOG
# 실행 중 가용 메모리를 10초마다 기록(정지 원인 추적용)
( while true; do echo "[mem $(date +%T)] 가용=$(free -m | awk "/Mem/{print \$7}")MB" >> $LOG; sleep 10; done ) &
MON=$!
"$@" >> $LOG 2>&1
RC=$?
kill $MON
echo "=== 끝 exit=$RC $(date +%T) 가용=$(free -m | awk "/Mem/{print \$7}")MB" >> $LOG
