# 심화(B) 과제 1·5·8 결과 요약 (2026-10-21)

이 문서는 보고서 본문에 옮기기 전 정리본이다. 스크립트는 `~/work/ts/`, 결과 파일은 `~/work/ts/report/` 아래에 있다.

| 과제 | 스크립트 | 결과 파일 | 계산 장치 |
|---|---|---|---|
| 1 저장소 크기·보관 정책 | `b1_storage.py --exact` | `b1_measure.json`, `b1_forecast.csv`, `b1_policy.csv` | CPU. 모델 계산이 없어 GPU로 옮길 것이 없다 |
| 5 노쇼·예약 없는 사용 | `b5_noshow.py` | `b5_summary.csv`, `b5_noshow_events.csv`, `b5_unbooked_events.csv` (+ `_val_test` 2주판) | CPU. 사용 여부 모델이 scikit-learn GBDT라 GPU 경로가 없다 |
| 8(a) 환기 제어 시뮬레이션 | `b8_control_sim.py` | `b8_control_sim.csv`, `b8_control_actions.csv`, `b8_mass_balance_fit.csv` | CPU. 과제 3에서 저장한 예측 결과를 다시 써서 새 torch 계산이 없다 |
| 8(b) 서비스 자기 성능 감시 | `service.py` (백업 `service.py.bak_20261021_b8`) | `logs/accuracy.jsonl`, `logs/accuracy_state.json`, `report/b8_accuracy_example.json` | 서비스 추론 장치는 그대로(CPU) |

---

## 과제 1. 저장소 크기 1년 추정과 보관 정책

**한 일**
- 지금 쌓이는 양을 실측했다(`python b1_storage.py --exact`). 토큰·비밀번호는 `.env`에서 읽기만 하고 출력하지 않았다.
  - InfluxDB `sensors` 버킷: 최근 24시간 포인트 수를 measurement별로 셌다(`count()`). WAL 파일 크기와 생성 시각으로 디스크 증가량도 구했다.
  - InfluxDB `history` 버킷(공용 iot-bucket 복사본, 445일): TSM 크기를 값 개수(내보낸 `sensors.csv`의 677만 값)로 나눠, 압축 뒤 값 하나의 크기를 구했다.
  - PostgreSQL `readings`: 전체 행 수, 테이블 크기(색인 포함), 최근 24시간 행 수를 measurement별로 셌다.
- 이 값으로 1년 크기를 추정하고, 단계별 보관 정책(원본 → 5분 집계 → 1시간 집계)을 설계했다. 실제 삭제나 보관 기간 변경은 하지 않았다.

**측정 조건**: 실데이터(보드 `t22-influxdb`, 과정 PostgreSQL). 측정 시점 2026-10-21 16시경 KST, 집계 구간은 최근 24시간이다. 수집 시작(10-20 15:16) 뒤 약 1.03일 분량이다. 모델 계산이 없어 CPU로 돌렸다.

**실측값**

| 항목 | 값 | 단위 |
|---|---:|---|
| InfluxDB `sensors` 하루 포인트 | **235,896** (uplink 68,319 · env 63,550 · hvac 60,246 · inference 23,998 · collector 19,763) | 값/일 |
| InfluxDB `sensors` WAL 증가 | 7.38 | MB/일 (압축 전) |
| InfluxDB 압축 후 크기(`history` 실측) | 39.5 MB ÷ 677만 값 = **5.83** | 바이트/값 |
| PostgreSQL `readings` 하루 행 | **43,062** (uplink 13,674 · env 13,566 · hvac 10,485 · inference 3,921 · collector 1,417) | 행/일 |
| PostgreSQL `readings` 전체 | 44,108행, 26.0 MB (테이블 16.0 + 색인 10.0) | |
| PostgreSQL 한 행 크기(색인 포함) | **589** | 바이트/행 |

- PostgreSQL은 센서 메시지 1건이 env 행과 uplink 행 두 줄로 저장된다. 여기에 냉난방(hvac), 추론 서비스 결과(inference), 수집기 상태(collector) 행이 더해져 하루 4.3만 행이 된다. 처음 가정(메시지 1건 = 1행, 500바이트, 하루 5.3MB)보다 약 4.8배 많다.

**1년 추정 (정책 없이 지금처럼 원본을 계속 쌓을 때)**

| 저장소 | 하루 | 1년 | 3년 |
|---|---:|---:|---:|
| InfluxDB `sensors`, 압축 후(TSM) | 1.38 MB | **0.50 GB** | 1.5 GB |
| InfluxDB `sensors`, 압축 전(WAL 형태, 상한) | 7.38 MB | 2.69 GB | 8.1 GB |
| PostgreSQL `readings` | 25.4 MB | **9.26 GB** | 27.8 GB |
| **합계(InfluxDB는 압축 후 기준)** | 26.8 MB | **약 9.8 GB** | **약 29 GB** |

**보관 정책 설계와 결과 크기**

| 단계 | 보관 기간 | 하루 건수 | 1년 뒤 | 3년 뒤 |
|---|---|---:|---:|---:|
| InfluxDB 원본(모든 값) | 30일 | 235,896 값 | 41.3 MB | 41.3 MB |
| InfluxDB 5분 집계(co2·온도·습도 28개 시리즈의 평균·최소·최대) | 400일 | 24,192 값 | 51.5 MB | 56.4 MB |
| InfluxDB 1시간 집계(평균·최소·최대) | 영구 | 2,016 값 | 4.3 MB | 12.9 MB |
| PostgreSQL 원본 env·hvac | 30일 | 24,051 행 | 425 MB | 425 MB |
| PostgreSQL 원본 uplink·collector·inference(진단용) | 7일 | 19,013 행 | 78 MB | 78 MB |
| PostgreSQL 5분 집계 `readings_5m` | 400일 | 8,064 행 | 353 MB | 387 MB |
| PostgreSQL 1시간 집계 `readings_1h` | 영구 | 672 행 | 29 MB | 88 MB |
| **합계** | | | **약 0.98 GB** | **약 1.09 GB** |

- 정책이 없으면 1년 9.8 GB, 3년 29 GB가 된다. 정책을 적용하면 1년 뒤 약 1 GB에서 거의 멈추고, 영구 보관하는 1시간 집계만 해마다 약 34 MB씩 는다.
- 집계 행 크기(120바이트)는 집계 테이블을 아직 만들지 않아 가정한 값이다. 컬럼 7개짜리 고정폭 행과 기본 키 색인을 합친 크기로 잡았다.
- 학습에 쓰는 5분 표(과제 2~7)는 5분 집계만으로 만들 수 있다. 원본 30일은 결측 원인 진단과 재학습 직전 확인에 쓰고, 신호 세기 같은 uplink 정보는 7일이면 충분하다.

**적용 명령(설계만 했고 실행하지 않았다)**

```bash
# InfluxDB: 집계 버킷 만들기 → 집계 작업 등록 → 원본 보관 기간 30일로 줄이기
docker exec t22-influxdb influx bucket create -n sensors_5m -r 400d
docker exec t22-influxdb influx bucket create -n sensors_1h -r 0
docker exec -i t22-influxdb influx task create <<'TASK'
option task = {name: "down_5m", every: 5m, offset: 1m}
data = from(bucket: "sensors") |> range(start: -10m)
  |> filter(fn: (r) => r._measurement == "env" and (r._field == "co2" or r._field == "temperature" or r._field == "humidity"))
union(tables: [
  data |> aggregateWindow(every: 5m, fn: mean) |> set(key: "agg", value: "mean"),
  data |> aggregateWindow(every: 5m, fn: min)  |> set(key: "agg", value: "min"),
  data |> aggregateWindow(every: 5m, fn: max)  |> set(key: "agg", value: "max")])
  |> to(bucket: "sensors_5m")
TASK
# (1시간 집계는 위와 같은 꼴로 every: 1h, 입력 sensors_5m 의 mean 을 다시 평균)
docker exec t22-influxdb influx bucket update -i a8d070ca677b632a -r 720h   # 집계가 30일 이상 쌓인 것을 확인한 뒤
```

```sql
-- PostgreSQL: 5분 집계 테이블 + 매시간 채우기(cron) + 원본 정리
CREATE TABLE readings_5m (bucket timestamptz, series text, field text, avg double precision, min double precision,
                          max double precision, n int, PRIMARY KEY (series, field, bucket));
INSERT INTO readings_5m
SELECT date_bin('5 minutes', ts, '2000-01-01') + interval '5 minutes', series, f.key,
       avg(f.value::float), min(f.value::float), max(f.value::float), count(*)
FROM readings, jsonb_each_text(fields) f
WHERE measurement = 'env' AND f.key IN ('co2','temperature','humidity') AND ts >= now() - interval '2 hours'
GROUP BY 1, 2, 3
ON CONFLICT (series, field, bucket) DO UPDATE SET avg = EXCLUDED.avg, min = EXCLUDED.min, max = EXCLUDED.max, n = EXCLUDED.n;
DELETE FROM readings WHERE measurement IN ('env','hvac') AND ts < now() - interval '30 days';
DELETE FROM readings WHERE measurement IN ('uplink','collector','inference') AND ts < now() - interval '7 days';
-- 데이터가 더 늘면 월별 파티션으로 바꿔 DROP PARTITION 으로 지운다(DELETE 는 VACUUM 전까지 공간을 돌려주지 않는다)
```

**해석**: 크기의 95%는 PostgreSQL 원본이다. 한 행이 589바이트로 InfluxDB 압축 값(5.8바이트)의 약 100배이고, 메시지 하나가 여러 행으로 나뉘어 저장되기 때문이다. 과정 PostgreSQL은 여러 팀이 같이 쓰는 서버이므로, 진단용 measurement(uplink·collector·inference)를 7일만 보관하는 규칙부터 먼저 적용하는 것이 효과가 크다. 이것만 해도 하루 증가량이 25.4 MB에서 14.2 MB로 줄어든다. 보드 디스크(233 GB 중 178 GB 남음)는 정책이 없어도 몇 년은 문제가 없다.

**한계**: 수집을 시작한 지 약 하루밖에 안 돼서, 주말이나 휴강일처럼 메시지 수가 다른 날은 반영되지 않았다. InfluxDB `sensors`는 아직 첫 주 샤드가 압축되기 전이라 압축률은 `history` 버킷 값으로 대신했다. 10-26 이후 샤드가 압축되면 `--exact`로 다시 확인한다.

---

## 과제 5. 회의실 노쇼와 예약 없는 사용 찾기

**한 일**
- 사용 여부는 과제 5의 분류 모델(`models/occ.pkl`, 센서값만 쓰는 GBDT, 5분 칸) 예측을 썼다. CO₂ 상승 규칙은 쓰지 않았다.
- **노쇼 규칙**: 예약 시작 뒤 15분(5분 칸 3개) 동안 모델이 계속 '비어있음'으로 예측하면 노쇼로 본다.
- **예약 없는 사용 규칙**: 모델이 '사용중'을 15분(3칸) 이상 이어서 예측했는데, 그 구간이 어떤 예약(시작 15분 전 ~ 끝난 뒤 15분)과도 겹치지 않으면 예약 없는 사용으로 본다.
- **정답**: 합성 데이터의 실제 인원(`truth.csv.gz`, 1분 단위)을 직접 관찰 기록 대신 썼다. 실제 노쇼는 예약 시작 뒤 15분 동안 실제 인원이 0명인 경우다. 실제 예약 없는 사용은 실제 인원이 1명 이상인 상태가 15분 이상 이어졌는데 예약이 없는 경우다.
- 비교를 위해 기존 CO₂ 상승 규칙(시작 뒤 30분 안에 +50ppm이 없으면 노쇼)도 같은 예약에 다시 적용했다.

**측정 조건**: 합성 데이터. 시험 주간(10-05~10-11)과, 표본을 늘리려고 모델 학습에 쓰지 않은 검증 주간을 합친 2주(09-28~10-11). 계산 장치는 CPU.

**노쇼 결과 (회의실)**

| 기간 | 규칙 | 예약 | 실제 노쇼 | 노쇼로 판정 | 맞힘 | 놓침 | 헛경보 |
|---|---|---:|---:|---:|---:|---:|---:|
| 시험 1주 | 모델 예측, 15분 비어있음 | 13 | 2 | 3 | 2 | 0 | **1** |
| 시험 1주 | (기존) CO₂ 30분 안 +50ppm 없음 | 13 | 2 | 4 | 2 | 0 | 2 |
| 검증+시험 2주 | 모델 예측, 15분 비어있음 | 27 | 3 | 4 | 3 | 0 | **1** |
| 검증+시험 2주 | (기존) CO₂ 30분 안 +50ppm 없음 | 27 | 3 | 6 | 3 | 0 | 3 |
| (참고) 전체 6주, 과제 2 진단 | (기존) CO₂ 규칙, 정답 = 30분 비어있음 | 90 | 10 | 19 | 10 | 0 | 9 |

- 정답을 '30분 동안 비어있음'으로 바꿔도 위 숫자는 같다.
- 모델 규칙의 헛경보 1건(10-07 15:30 예약)은 첫 사람이 **12분 뒤**에 들어온 경우다. 정답 기준으로는 노쇼가 아니지만, CO₂가 오르기 전이라 모델이 15분 안에 알아채지 못했다.
- 기존 CO₂ 규칙이 더 낸 헛경보(09-28 15:30, 10-08 14:00)는 직전 회의 바로 뒤라 CO₂가 이미 높아 +50ppm이 더 오르지 않은 경우다. 모델은 감소 속도 같은 특징도 함께 봐서 이 경우를 맞혔다.

**예약 없는 사용 결과**

| 기간 | 공간 | 실제 건수 | 판정 | 맞힘 | 놓침 | 헛경보 |
|---|---|---:|---:|---:|---:|---:|
| 검증+시험 2주 | 회의실 | 1 | 1 | 1 | 0 | 0 |
| 검증+시험 2주 | 라운지 | 5 | 0 | 0 | **5** | 0 |
| 검증+시험 2주 | 강의실 A·B | 0 | 0 | 0 | 0 | 0 |
| 시험 1주 | 회의실 | 0 | 0 | 0 | 0 | 0 |

- 회의실의 예약 없는 사용 1건(10-01 10:34~11:24, 2명)은 시작 1분 차이로 잡았다.
- 라운지에서 놓친 5건은 모두 저녁(18시 이후)이나 이른 아침에 2~3명이 20~40분 머문 경우다. 라운지(180 m³)는 넓어서 2~3명으로는 CO₂가 거의 오르지 않으므로, 센서만 쓰는 모델이 알아채기 어렵다.

**해석**: 사용 여부 모델로 판정하면 기존 CO₂ 상승 규칙보다 헛경보가 절반 이하로 줄고(2주 기준 3건 → 1건), 실제 노쇼는 모두 잡는다. 남는 헛경보는 늦게 온 회의라서, 노쇼 알림을 15분이 아니라 20분 뒤에 확정하거나 "15분째 비어있음 → 확인 요청, 30분째 → 예약 해제"처럼 두 단계로 두면 줄일 수 있다. 다만 시험 기간 노쇼가 2~3건뿐이라 표본이 작다. 넓은 공간의 소수 인원은 CO₂로 잡기 어려우므로 PIR(동작) 센서를 함께 쓰는 것이 좋다.

---

## 과제 8(a). 예측 기반 환기 제어를 지난 1주에 적용했다면

**한 일**
- **질량수지 가정(1분 단위)**: `C[t+1] = C[t] + a(t)·(420 − C[t]) + r(t)`
  - `a(t)`는 기본 환기율이다. 운영 시간(08:30~19:00) 안과 밖의 두 값을 실제 CO₂(`truth.co2_true`)에 최소제곱으로 맞췄다. 맞춘 값이 데이터 생성 설정(meta의 `ach_on`, `ach_off`)과 거의 같아서 가정이 맞는지 확인할 수 있었다(아래 표).
  - `r(t)`는 사람이 내는 CO₂와 고장 영향을 합친 나머지다. 원래 궤적에서 그대로 구했다. 그래서 제어가 없으면 원래 값이 정확히 재현되고, 제어는 환기만 바꾼다.
  - **환기 강화**는 기본 환기율에 시간당 공기 교환을 더하는 것으로 본다. 기본 +2회/시간(창문 열기·급기 팬 최대 정도)으로 두고, +1회와 +4회도 함께 계산했다.
- **제어 규칙(5분마다 판단)**
  - 예측형: 30분 안 예측 최댓값이 1,000ppm 이상이면 켠다. 켜면 최소 15분 유지하고, 예측 최댓값이 900 밑으로 내려가면 끈다.
    - 예측값은 지금(제어가 반영된) CO₂에 과제 3 최종 모델(`gru_L48`)이 시험 주간에 냈던 '변화량' 예측을 더한 값이다.
    - 예측이 없는 시각은 지금 값이 1,000 이상이면 켜는 방식으로 대신했다.
  - 비교용 반응형: 지금 CO₂가 1,000 이상이면 켜고 900 밑이면 끈다(최소 15분).

**측정 조건**: 합성 데이터 시험 주간(10-05~10-11), 1분 단위 실제 CO₂(잡음 없는 정답값) 기준. 예측은 과제 3의 저장된 결과를 다시 썼다(새 모델 계산 없음, CPU).

**질량수지 맞춤 확인**

| 공간 | 운영 중 환기(회/시간) 맞춤 / 설정 | 운영 밖 맞춤 / 설정 | 설명력 R² |
|---|---|---|---:|
| 강의실 A | 2.18 / 2.2 | 0.25 / 0.25 | 0.69 |
| 강의실 B | 2.00 / 2.0 | 0.25 / 0.25 | 0.64 |
| 회의실 | 3.01 / 3.0 | 0.32 / 0.30 | 0.89 |
| 라운지 | 2.39 / 2.5 | 0.35 / 0.35 | 0.83 |

**결과: 1,000ppm 이상인 시간 (환기 강화 +2회/시간 기준)**

| 공간 | 제어 없음(분) | 반응형(분) | 예측형(분) | 예측형 감소율 | 예측형 켠 횟수 | 예측형 환기 시간(분) |
|---|---:|---:|---:|---:|---:|---:|
| 강의실 A | 845 | 191 | **164** | 80.6% | 24 | 575 |
| 강의실 B | 1,432 | 234 | **161** | 88.8% | 43 | 1,075 |
| 회의실 | 52 | 10 | **10** | 80.8% | 2 | 35 |
| 라운지 | 137 | 69 | **41** | 70.1% | 4 | 180 |
| **합계** | **2,466** | 504 | **376** | **84.8%** | 73 | 1,865 |

**환기 강도에 따른 합계(민감도)**

| 환기 강화 | 반응형 초과(분) | 예측형 초과(분) | 예측형 감소율 | 예측형 켠 횟수 |
|---|---:|---:|---:|---:|
| +1회/시간 | 1,159 | 794 | 67.8% | 40 |
| +2회/시간 | 504 | 376 | 84.8% | 73 |
| +4회/시간 | 264 | 167 | 93.2% | 80 |

**해석**: 같은 환기 강도라면 30분 앞 예측으로 미리 켜는 쪽이 지금 값을 보고 켜는 쪽보다 초과 시간이 합계 기준 25~37% 더 적다(+2회 기준 504분 → 376분). 그 대신 환기 시간은 약 26% 늘어 냉난방 에너지가 더 든다. 회의실은 고장(냉방 고장) 구간 등으로 예측이 비어 있는 시각이 많아 반응형과 결과가 같았다. 강의실 B는 수업 중 계속 높아 +2회로도 161분이 남으므로, 수업 시작 전 미리 환기하거나 인원을 줄이는 운영 조치가 함께 필요하다. 이 결과는 "사람 수와 고장은 그대로 두고 환기만 바꾼다"는 가정에서 나온 것이다. 예측은 환기 효과를 모른 채 낸 값이라 실제보다 오래 켜는 보수적인 쪽으로 치우친다.

---

## 과제 8(b). 서비스가 스스로 성능을 감시하게 하기

**한 일** (`~/work/ts/service.py`, 원본은 `service.py.bak_20261021_b8`로 백업)
1. 매 주기 낸 60분 예측(5분 간격 12개)을 5분 칸마다 한 번씩 **저장**한다. 같은 칸 안에서 매분 다시 계산한 값은 저장하지 않는다.
2. 대상 시각의 실측이 들어오면 **채점**한다.
   - 실측은 그 5분 칸에 원본 메시지가 1건 이상 있는 값이다. 보간한 값은 쓰지 않는다.
   - 늦게 도착하는 메시지를 감안해, 대상 칸이 끝나고 5분이 더 지난 뒤에 채점한다.
   - 대상 시각 뒤 30분이 지나도 실측이 없으면 채점하지 않고 버린다. 회의실처럼 센서가 드물게 보내는 경우가 여기에 해당한다.
3. 최근 **6시간 이동 MAE**를 공간·예측 앞 시간(5~60분)별로 낸다.
4. **60분 앞 CO₂ MAE**가 **검증 MAE(34.29ppm) × 2 = 68.6ppm**을 넘고 채점 건수가 12건 이상이면 경보 `forecast_degraded`를 낸다. 기존 경보처럼 처음 한 번 알리고, 이후에는 30분마다 다시 알린다.
5. 채점 상태는 `logs/accuracy_state.json`에 저장하므로 재시작해도 이어진다. 처음 시작할 때 이 파일이 없으면 InfluxDB `inference` 기록의 `co2_60`으로 지난 7시간의 60분 앞 예측을 채워서 바로 채점을 시작한다. 채점한 건은 `logs/accuracy.jsonl`에 한 줄씩 남긴다.
6. 기존 API·MQTT 필드는 그대로 두고 아래만 더했다.
   - 공간 결과(`/spaces/{공간}`, MQTT `team2-2/inference/<공간>`)에 `accuracy` 필드
   - 새 엔드포인트 `/spaces/{공간}/accuracy`
   - InfluxDB `inference`에 `mae_co2_60`, `mae_n_60` 필드
   - 경보 종류 `forecast_degraded`
7. 기존 감시 스크립트(`run_service.sh`)로 재시작했다. 15:45 재시작 뒤 주기 실패 0회이고 `/health`도 정상이다.

**새 엔드포인트 형식**: `GET /spaces/{lab|meeting|office}/accuracy`

| 필드 | 뜻 |
|---|---|
| `window_hours` | 이동 MAE 창(시간), 기본 6 |
| `as_of` | 기준 5분 칸 시각(KST) |
| `mae_co2_60_ppm`, `n_60` | 60분 앞 CO₂ MAE(ppm)와 채점 건수 |
| `threshold_co2_60_ppm`, `threshold_rule` | 경보 기준(ppm)과 그 근거 |
| `degraded`, `enough_data` | 경보 여부, 채점 건수가 `min_n` 이상인지 |
| `pending_forecasts`, `scored_since_start`, `expired_since_start` | 채점 대기 수, 시작 뒤 채점한 수, 실측이 없어 버린 수 |
| `by_horizon[]` | 5~60분 앞마다 `n`, `mae_co2_ppm`, `bias_co2_ppm`(예측−실측 평균), `mae_temperature_c` |

**응답 예** (`report/b8_accuracy_example.json`, 2026-10-21 15:50, 실습실, 5~55분 항목은 줄임)

```json
{"space": "실습실", "topic_space": "lab", "window_hours": 6.0, "as_of": "2026-10-21T15:50:00",
 "mae_co2_60_ppm": 105.8, "n_60": 71, "threshold_co2_60_ppm": 68.6, "threshold_rule": "검증 MAE 34.29 × 2.0",
 "min_n": 12, "degraded": true, "enough_data": true,
 "pending_forecasts": 34, "scored_since_start": 72, "expired_since_start": 1,
 "by_horizon": [{"minutes_ahead": 5, "n": 0, "mae_co2_ppm": null, "bias_co2_ppm": null, "mae_temperature_c": null},
                {"minutes_ahead": 60, "n": 71, "mae_co2_ppm": 105.8, "bias_co2_ppm": 3.2, "mae_temperature_c": null}]}
```

공간 결과의 경보 예: `{"type": "forecast_degraded", "severity": "warning", "sensor": "model", "message": "최근 6시간 60분 앞 CO₂ 예측 MAE 165.4ppm, 기준 68.6ppm 초과 (재학습·센서 점검 필요)"}`

**측정 조건**: 실데이터(보드 InfluxDB `sensors`), 2026-10-21 약 09:45~15:50 KST의 60분 앞 예측. 처음 채운 기록에는 15:3x 재부팅 공백이 들어 있다. 5~55분 앞 MAE는 재시작 뒤 새로 쌓인다. 15:55에 5분 앞 첫 채점이 들어온 것을 확인했다(사무실 오차 −5.1ppm, 온도 −0.02℃).

**결과(재시작 직후)**

| 공간 | 60분 앞 CO₂ MAE | 채점 건수 | 기준 | 경보 |
|---|---:|---:|---:|---|
| 실습실 | 105.8 ppm | 71 | 68.6 ppm | 켜짐 |
| 사무실 | 165.4 ppm | 70 | 68.6 ppm | 켜짐 |
| 회의실 | 약 68 ppm | 33 | 68.6 ppm | 기준 근처(주기마다 켜졌다 꺼짐) |

**해석**: 경보가 켜진 것은 감시 기능이 의도대로 움직인다는 뜻이다. 서비스 모델(`gru_L48_generic`)은 합성 데이터로 학습했고, 실공간에서는 검증 오차의 3~5배가 나온다. 할 일은 실데이터(`~/work/ts_real`)로 학습한 모델로 바꾸고, 기준도 실데이터 검증 MAE로 다시 정하는 것이다. 지금은 기준이 합성 데이터 검증값이라 실공간에서는 경보가 계속 켜져 있을 것이다.
