#!/bin/bash
# 과제 8 장애 시험: 수집기를 10분 멈췄다가 다시 켠다. 결과는 logs/failtest.log
cd "$(dirname "$0")"
echo "$(date "+%F %T") 수집기 정지" >> logs/failtest.log
docker stop t22-collector >/dev/null
sleep 600
docker start t22-collector >/dev/null
echo "$(date "+%F %T") 수집기 재시작" >> logs/failtest.log
sleep 300
echo "$(date "+%F %T") 시험 종료" >> logs/failtest.log
