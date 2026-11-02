#!/bin/bash
# 과제 8 추론 서비스 감시 실행: 죽으면 10초 뒤 다시 띄운다. 재부팅 때는 cron @reboot가 이 스크립트를 실행한다.
cd "$(dirname "$0")"
# 보드가 막 켜졌으면 InfluxDB·브로커 컨테이너가 뜰 때까지 기다린다
for i in $(seq 1 60); do curl -sf http://localhost:8087/health >/dev/null && break; sleep 5; done
while true; do
  echo "$(date "+%F %T") 서비스 시작" >> logs/supervisor.log
  ~/venv/bin/python -W ignore service.py >> logs/stdout.log 2>&1
  echo "$(date "+%F %T") 서비스 종료(코드 $?), 10초 뒤 재시작" >> logs/supervisor.log
  sleep 10
done
