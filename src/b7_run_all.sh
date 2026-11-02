#!/bin/bash
# 심화 7-3·7-4를 하나씩 차례로. 실행기가 매번 가용 메모리(2GB 이상)·다른 python 작업 없음을 확인한다.
#   1) INT8 입력 준비(CPU)  2) 엔진 하나당 프로세스 하나(GPU)  3) 합치기  4) 4개 모델 동시 적재 메모리(변형마다 새 프로세스)
cd "$(dirname "$0")"
PY=~/venv/bin/python
step() {  # step <로그이름> <환경변수...> <스크립트>
  local log=$1; shift
  for try in 1 2 3 4 5 6; do MIN_AVAIL=2000 ./b_gpu_run.sh $log env "$@" && return 0; sleep 60; done
  echo "포기: $log" ; return 1
}
for m in fc_tcn ae fc_gru fc_lstm; do step b7_prep_$m STAGE=prep ONLY=$m $PY -W ignore b7_int8.py; done
for m in fc_tcn ae fc_gru fc_lstm; do
  for mode in fp32 fp16 int8; do step b7_int8_${m}_$mode STAGE=gpu ONLY=$m MODE=$mode $PY -W ignore b7_int8.py; sleep 15; done
done
STAGE=merge $PY -W ignore b7_int8.py > report/b7_int8_merge.log 2>&1
for v in torch ort trt; do step b7_memory_$v VARIANT=$v $PY -W ignore b7_memory.py; sleep 15; done
echo ALLDONE > report/b7_run_all.done
