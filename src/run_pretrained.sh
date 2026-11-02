#!/bin/bash
# 과제 6: Chronos-Bolt 3종을 GPU로, 모델마다 새 프로세스(메모리 비움)
cd "$(dirname "$0")"
LOG=report/pretrained/run_gpu.log
mkdir -p report/pretrained; : > $LOG
for m in tiny small base; do
  echo "=== $m $(date +%T) avail=$(free -m | awk "/Mem/{print \$7}")MB" >> $LOG
  DEV=cuda BATCH=128 ~/venv/bin/python -W ignore pretrained.py amazon/chronos-bolt-$m >> $LOG 2>&1
  echo "=== $m exit=$? $(date +%T)" >> $LOG
done
echo ALLDONE >> $LOG
