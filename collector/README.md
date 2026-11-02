# team2-2 수집 스택 (과제 1)

MQTT(`data.withio.net`) → 수집기 → InfluxDB 2.7(보드 안) + PostgreSQL(과정 서버). 수집기와 InfluxDB는 Docker로 돌고, 보드 재부팅이나 장애 뒤 자동으로 다시 시작된다.
팀원 스택과는 완전히 따로 돈다. 컨테이너 이름, 포트, 데이터 폴더, MQTT client_id가 모두 다르다.

| 항목 | 값 |
|---|---|
| 위치 | `/home/team2-2/ingest` |
| InfluxDB | `http://localhost:8087` (보드 안에서만 열림), org `team2-2`, bucket `sensors` |
| 컨테이너 | `t22-influxdb`, `t22-collector` |
| MQTT client_id | `team2-2-collector` |
| PostgreSQL | 과정 서버(Crowfoot에서 발급, 주소·계정은 `.env`), 테이블 `readings`, 뷰 `env_readings` |
| 비밀값 | `.env` (본인만 읽기 가능, 공유·커밋 금지) |

## 받는 토픽과 저장 구조

| 토픽 | measurement | tags | fields |
|---|---|---|---|
| `application/+/device/+/event/up` (환경 센서) | `env` | location, device, model, gateway | `object`의 숫자 값 전부(co2, temperature, humidity, …), rx_delay_s |
| 위와 같음 | `uplink` | location, device, model | fcnt, rssi, snr, dr, rx_delay_s |
| `xflow/hvacr/+/+/status`, `xflow/controller/+/status` (냉난방 실내기·실외기·컨트롤러) | `hvac` | unit_type, unit_id, name | 숫자 값 전부(power·online 등 참거짓은 1/0), rx_delay_s |
| (수집기 자신) | `collector` | name | received, written, duplicate, write_error, mqtt_connected, last_msg_age_s, pending (1분마다) |

- 시각: 메시지에 찍힌 측정 시각(UTC)을 쓴다. 센서는 `rxInfo[0].nsTime`(네트워크 서버 수신 시각), 냉난방은 `ts`(ms)를 쓴다. 센서 메시지 최상위 `time`은 게이트웨이 시계라서 쓰지 않는다(아래 '발견' 참고).
- `rx_delay_s`: 수집기가 받은 시각과 측정 시각의 차이(초)다. 네트워크가 밀렸다가 한꺼번에 도착한 경우를 찾을 때 쓴다.
- 중복: 센서는 `deduplicationId`로, 냉난방은 (토픽, ts)로 거른다.

## PostgreSQL 저장 구조

InfluxDB와 같은 기록을 한 메시지 한 줄로 저장한다. 새 센서나 새 항목이 생겨도 `fields`(jsonb)에 그대로 들어가므로 테이블을 고치지 않는다.

```sql
readings(ts timestamptz, measurement text, series text, tags jsonb, fields jsonb, rx_delay_s double precision,
         inserted_at timestamptz, PRIMARY KEY (measurement, series, ts))
-- 자주 쓰는 형태로 펼친 뷰
SELECT ts_kst, location, device, co2, temperature, humidity FROM env_readings ORDER BY ts DESC LIMIT 10;
```

- 기본 키 (measurement, series, ts) 덕분에 같은 메시지를 두 번 써도 한 줄만 남는다(재전송·재동기화 안전).
- 시각은 ms 단위로 맞춰 InfluxDB와 같은 값을 쓴다.
- 역할 분담: InfluxDB는 보드 안 원본 겸 버퍼다. PostgreSQL은 외부 서버라 네트워크가 끊기면 쓰기가 실패한다. 실패분은 메모리에 보관했다가 다시 쓰고, 수집기가 재시작돼 보관분을 잃더라도 매시간 `backfill.py`가 InfluxDB에서 빈 곳을 채운다.
- 학습·분석은 PostgreSQL(또는 InfluxDB)에서 읽기만 한다. 수집기는 쓰기만 해서 서로 막지 않는다.

## 자주 쓰는 명령 (보드에서)

```bash
cd ~/ingest
docker compose ps                                     # 상태
docker compose logs -f --tail 50 collector            # 수집기 로그
docker compose exec collector python coverage.py 24          # 최근 24시간 커버율(co2)
docker compose exec collector python coverage.py 24 temperature
docker compose exec collector python coverage.py 1 power hvac  # 냉난방 전원 상태
docker compose restart collector                      # config.yaml 수정 후 반영
docker compose up -d --build                          # 코드 수정 후 반영
docker compose exec -T collector python backfill.py -24h     # InfluxDB → PostgreSQL 빈 곳 채우기(여러 번 돌려도 안전)
```

## 형식 변경 대응

`collector/config.yaml`의 `sources`만 고친다. 새 토픽이나 새 형식은 source를 추가하고, 필드 이름 변경은 `rename`, 제외는 `exclude`로 처리한다.
시각 형식은 `iso`, `epoch_ms`, `epoch_s` 중에서 고른다. 토픽 조각을 태그로 쓰려면 `topic.N`으로 적는다.
수정한 뒤 `docker compose restart collector`로 반영한다.

## 수집 현황 보고서

```bash
docker compose exec -T collector python report.py 24 > reports/coverage_$(date +%Y%m%d_%H%M).md
```

- 매일 09:05에 cron이 지난 24시간 보고서를 `reports/coverage_YYYYMMDD.md`로 남긴다.
- 커버율은 두 가지를 낸다. 1분 커버율은 안내문 기준이다. 주기 커버율은 센서 자신의 보고 주기를 기준으로 기대 건수 대비 실제 건수다.
- 결측 구간의 원인은 세 가지로 나눈다.
  - 수집기 끊김: 수집기 상태 기록으로 판단한다.
  - 전송 유실: 센서 전송 번호(fCnt)가 건너뛰었다.
  - 센서가 보내지 않음: fCnt가 이어진다.

## 복구 시험 기록

| 날짜 | 시험 | 결과 |
|---|---|---|
| 2026-10-20 | 수집기 컨테이너 네트워크 60초 차단 (`docker network disconnect t22-ingest_default t22-collector`, 15:21:59~15:22:59) | 복구 6초 뒤(15:23:05) MQTT 재접속(session_present=True). 차단 중 실패한 InfluxDB 쓰기 3번은 보관했다가 다시 써서 유실 0 |
| 2026-10-20 | 형식 변경 대응: 냉난방(xflow) 토픽 추가 | 코드 수정 없이 `config.yaml`에 source 3개(xflow-unit, xflow-controller, xflow-other)와 `subscribe`에 `xflow/#`를 추가했다. 메시지 구조(`{"ts": ms, "values": {이름: {필드}}}`)가 센서와 달라서 `time.format: epoch_ms`, `keyed_by: name`, 토픽 조각 태그(`topic.N`)로 해석했다 |
| 2026-10-20 | 보드 재부팅 (`sudo reboot`) | 15:32:45 수집기 정상 종료(남은 0건 쓰고 종료) → 15:33:08 부팅 → 15:33:29 MQTT 재접속(session_present=True), 사람 손 없이 수집 재개. 수집기 공백 약 44초. 공백 동안 브로커가 보관한 메시지(QoS 1, 고정 client_id)를 재접속 뒤 받아서 공백 중 측정값도 저장됐다(rx_delay_s 60~67초) |
| 2026-10-20 | PostgreSQL 저장 추가(16:20) | 설정(.env의 PG_*)만으로 켜진다. 15:16~16:20에 InfluxDB에만 있던 1,774건을 `backfill.py`로 옮겼다. 게이트웨이 시계가 빨랐던 196건은 수신 시각으로 고쳐서 넣었다. 다시 돌리면 새로 추가 0건으로 두 저장소가 일치한다 |

참고: 겹치는 토픽 패턴(`xflow/#`와 `xflow/hvacr/+/+/status`)을 따로 구독하면 브로커가 같은 메시지를 두 번 보낸다. 그래서 구독은 `mqtt.subscribe`에 넓은 패턴 하나만 두고, 해석 규칙은 `sources`에서 고른다.

## 발견: 게이트웨이 시계가 약 5분 빠르다 (2026-10-20)

`rx_delay_s`(수신 시각 - 메시지 시각)가 센서마다 두 무리로 나뉜다.

| 무리 | 센서 | rx_delay_s 중앙값 |
|---|---|---|
| 정상 | AM107-067999, EM300-TH-151547·151606·151836, EM320-TH-389818, EM500-CO2-152590, WS302-012436 | +0.24초 |
| 시계 빠름 | AM103-081175(사무실), AM103-089152(회의실), AM107-140101, EM320-TH-389010, EM500-CO2-152862 | **-295초** |

메시지의 `time`은 게이트웨이가 찍은 수신 시각이다. 한쪽 게이트웨이 시계가 약 4분 55초 빠른 것으로 보인다(EM300-TH-151523은 두 게이트웨이를 오간다).
그래서 이 센서들의 값은 미래 시각으로 저장된다. 그대로 쓰면 1분·5분 칸이 어긋나고, 실시간 서비스에서 "지금보다 미래" 데이터가 생긴다.
원본 메시지를 보면 `rxInfo[0].gwTime`(게이트웨이 시계)과 `time`이 같고, `rxInfo[0].nsTime`(네트워크 서버 시계)은 정확하다. 게이트웨이 `24e124fffef5dccc`만 약 295초 빠르고, `24e124fffef79304`는 정상이다.
대응(15:40, 설정만 변경): 시각 경로를 `time`에서 `rxInfo.0.nsTime`으로 바꿨고, `gateway` 태그를 추가했다. 15:16~15:40에 저장된 값 중 `24e124fffef5dccc`를 거친 값은 약 295초 미래로 찍혀 있으니 과제 2 정제 때 보정한다.
보고서용 정리: "시각은 센서가 찍은 것과 수신한 것 중 무엇을 저장할까"라는 질문에 답이 된다. 이 센서들은 측정 시각을 따로 보내지 않는다. 그래서 가장 이른 신뢰할 수 있는 시계인 네트워크 서버 수신 시각을 쓰고, 수집기 수신 시각과의 차이(rx_delay_s)를 함께 남겨 지연 도착을 찾는다.
