#!/bin/bash
# 과제 7: 모델마다 새 프로세스로 측정(GPU 메모리를 매번 비운다) → 결과 합치기
cd "$(dirname "$0")"
: > report/optimize_run.log
for m in fc_gru fc_lstm fc_tcn ae vent; do
  echo "=== $m $(date +%T)" >> report/optimize_run.log
  ONLY=$m ~/venv/bin/python -W ignore optimize.py >> report/optimize_run.log 2>&1
done
~/venv/bin/python - <<PY >> report/optimize_run.log 2>&1
import pandas as pd, glob
for k in ("latency", "diff"):
    fs = sorted(glob.glob(f"report/optimize_{k}_*.csv"))
    pd.concat([pd.read_csv(f) for f in fs]).to_csv(f"report/optimize_{k}.csv", index=False)
print("MERGED")
PY
