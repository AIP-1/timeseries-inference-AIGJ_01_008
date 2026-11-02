"""과제 1 심화: 저장소 크기 1년 추정 + 다운샘플링·보관 정책 설계 (읽기만 한다, 정책은 적용하지 않는다)

  python b1_storage.py            → report/b1_measure.json, b1_forecast.csv, b1_policy.csv  (비밀값 없이 디스크·보고서만 본다)
  python b1_storage.py --exact    → 위에 더해 InfluxDB count()·PostgreSQL 행 수/테이블 크기를 직접 잰다(.env 의 토큰·비밀번호 사용, 출력 안 함)

기본 모드에서 쓰는 실측값
  - sensors 버킷 WAL 파일 크기와 그 샤드가 생긴 시각 → 하루 디스크 증가량(압축 전 WAL 형태)
  - history 버킷(공용 iot-bucket 복사본, 약 14개월) TSM 크기 ÷ 그 안의 값 개수(내보낸 sensors.csv 의 값 수) → 압축 후 바이트/값
  - 수집 현황 보고서(reports/coverage_*.md)의 센서별 24시간 건수 → 하루 메시지 수, 주요 env 값(co2·온도·습도) 수
가정(실측 못 한 부분, 결과 표에 '가정'으로 표시)
  - 메시지 1건당 저장 값 수 PTS_PER_MSG: env 의 다른 숫자 필드·uplink 5개 필드·냉난방 등을 합쳐 10개로 둔다
  - PostgreSQL 한 행(메시지 1건) 크기 PG_ROW_BYTES: jsonb tags·fields + 기본 키 색인 포함 500바이트
"""
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

OUT = Path.home() / "work/ts/report"
REPORTS = Path.home() / "ingest/reports"
BUCKET_ID = {"sensors": "a8d070ca677b632a", "history": "6db32ee211d36289"}
PTS_PER_MSG = 10
PG_ROW_BYTES = 500
MAIN_SERIES = None   # 보고서에서 센다(co2·온도·습도 시리즈 수)


def dexec(*cmd):
    return subprocess.run(["docker", "exec", "t22-influxdb", *cmd], capture_output=True, text=True).stdout


def du_bytes(path):  # 실제 블록 기준(미리 잡아 둔 색인 파일의 겉보기 크기는 빼고 싶어서 TSM·WAL 파일만 센다)
    out = dexec("sh", "-c", f"find {path} -type f \\( -name '*.tsm' -o -name '*.wal' \\) -exec stat -c %s {{}} +")
    return sum(int(x) for x in out.split())


m = {}
# ---- 1. sensors 버킷: WAL 증가 속도
wal_files = dexec("sh", "-c", f"find /var/lib/influxdb2/engine/wal/{BUCKET_ID['sensors']} -name '*.wal' -size +1k -exec stat -c '%s %W %n' {{}} +").split("\n")
wal = [l.split() for l in wal_files if l.strip()]
born = int(dexec("stat", "-c", "%W", f"/var/lib/influxdb2/engine/wal/{BUCKET_ID['sensors']}/autogen/1").strip() or 0)
now = datetime.now(timezone.utc).timestamp()
wal_bytes = sum(int(w[0]) for w in wal)
age_days = (now - born) / 86400 if born > 0 else None
if not age_days or age_days > 30:   # 생성 시각을 못 읽으면 README 기록(수집 시작 2026-10-20 15:16 KST)을 쓴다
    age_days = (now - datetime(2026, 10, 20, 6, 16, tzinfo=timezone.utc).timestamp()) / 86400
m["sensors"] = dict(wal_bytes=wal_bytes, tsm_bytes=du_bytes(f"/var/lib/influxdb2/engine/data/{BUCKET_ID['sensors']}"), age_days=round(age_days, 3),
                    wal_bytes_per_day=wal_bytes / age_days)
# ---- 2. history 버킷: 압축 후 바이트/값
h_tsm = du_bytes(f"/var/lib/influxdb2/engine/data/{BUCKET_ID['history']}")
real = Path.home() / "work/data/real/sensors.csv"
n_vals = 0
for ch in pd.read_csv(real, usecols=["co2", "temperature", "humidity"], chunksize=500_000):
    n_vals += int(ch.notna().sum().sum())
rmeta = json.loads((real.parent / "meta.json").read_text())
h_days = (pd.Timestamp(rmeta["end"]) - pd.Timestamp(rmeta["start"])).total_seconds() / 86400
m["history"] = dict(tsm_bytes=h_tsm, values=n_vals, span=[rmeta["start"], rmeta["end"]], days=round(h_days, 1),
                    bytes_per_value=h_tsm / n_vals, mb_per_week=h_tsm / 1e6 / (h_days / 7))
# ---- 3. 수집 현황 보고서: 하루 메시지 수
rep = sorted(REPORTS.glob("coverage_*.md"))[-1]
txt = rep.read_text()
cov = txt.split("## 커버율", 1)[1].split("\n## ", 1)[0]   # 커버율 표만(결측 구간 표는 뺀다)
rows = [l.split("|") for l in cov.splitlines() if l.startswith("| ") and re.match(r"\|\s*\S+\s*\|\s*\S+-\S+", l)]
tab = pd.DataFrame([[c.strip() for c in r[1:6]] for r in rows], columns=["space", "device", "field", "count", "period"])
tab["count"] = tab["count"].astype(int)
# 보고서 구간 중 수집기가 실제 돈 시간(첫 상태 기록 ~ 구간 끝)으로 하루 값으로 바꾼다
mt = re.search(r"조회 구간: (\d\d-\d\d \d\d:\d\d) ~ (\d\d-\d\d \d\d:\d\d)", txt)
m0 = re.search(r"수집기 첫 상태 기록: (\d\d-\d\d \d\d:\d\d)", txt)
yr = 2026
t_end = pd.Timestamp(f"{yr}-{mt.group(2)}")
t_start = max(pd.Timestamp(f"{yr}-{mt.group(1)}"), pd.Timestamp(f"{yr}-{m0.group(1)}") if m0 else pd.Timestamp(f"{yr}-{mt.group(1)}"))
hours = (t_end - t_start).total_seconds() / 3600
msgs = tab.groupby("device")["count"].max().sum()
m["report"] = dict(file=rep.name, hours_running=round(hours, 2), devices=int(tab.device.nunique()), main_series=len(tab),
                   msgs_per_day=msgs * 24 / hours, main_values_per_day=tab["count"].sum() * 24 / hours)
MAIN_SERIES = len(tab)

if "--exact" in sys.argv:   # 권한이 있을 때만: 직접 센 값으로 가정을 바꾼다
    from influxdb_client import InfluxDBClient
    import psycopg
    ENV = dict(l.split("=", 1) for l in (Path.home() / "ingest/.env").read_text().splitlines() if "=" in l and not l.startswith("#"))
    with InfluxDBClient(url="http://localhost:8087", token=ENV["INFLUX_TOKEN"], org=ENV["INFLUX_ORG"], timeout=600000) as cli:
        tabs = cli.query_api().query('from(bucket:"sensors") |> range(start:-24h) |> group(columns:["_measurement"]) |> count()')
        cnt = {}
        for t in tabs:
            for r in t.records:
                cnt[r.values["_measurement"]] = cnt.get(r.values["_measurement"], 0) + int(r.get_value())
    m["sensors"]["points_24h_exact"] = cnt
    with psycopg.connect(host=ENV["PG_HOST"], port=ENV.get("PG_PORT", "5432"), dbname=ENV["PG_DB"], user=ENV["PG_USER"], password=ENV["PG_PASSWORD"],
                         connect_timeout=10, options=f"-c search_path={ENV.get('PG_SCHEMA', 'public')}") as conn:
        cur = conn.cursor()
        cur.execute("SELECT count(*), pg_total_relation_size('readings'), count(*) FILTER (WHERE ts > now() - interval '24 hours') FROM readings")
        n, size, n24 = cur.fetchone()
        cur.execute("SELECT measurement, count(*) FROM readings WHERE ts > now() - make_interval(hours => 24) GROUP BY 1")
        by_m = dict(cur.fetchall())
    m["pg"] = dict(rows=n, total_bytes=size, rows_24h=n24, bytes_per_row=size / n, rows_24h_by_measurement=by_m)

# ---- 1년 추정 (정책 없음 = 지금처럼 원본을 계속 쌓음)
pts_day = sum(m["sensors"]["points_24h_exact"].values()) if "points_24h_exact" in m["sensors"] else m["report"]["msgs_per_day"] * PTS_PER_MSG
bpv = m["history"]["bytes_per_value"]
pg_rows_day = m["pg"]["rows_24h"] if "pg" in m else m["report"]["msgs_per_day"]
pg_bpr = m["pg"]["bytes_per_row"] if "pg" in m else PG_ROW_BYTES
fc = pd.DataFrame([
    dict(저장소="InfluxDB sensors (압축 전, WAL 증가 실측)", 하루_건수=None, 단위="", 바이트_per=None, 하루_MB=m["sensors"]["wal_bytes_per_day"] / 1e6, 근거="실측"),
    dict(저장소="InfluxDB sensors (압축 후 TSM)", 하루_건수=pts_day, 단위="값", 바이트_per=bpv, 하루_MB=pts_day * bpv / 1e6,
         근거="실측" if "points_24h_exact" in m["sensors"] else f"가정: 메시지당 {PTS_PER_MSG}값, 압축률은 history 실측"),
    dict(저장소="PostgreSQL readings", 하루_건수=pg_rows_day, 단위="행", 바이트_per=pg_bpr, 하루_MB=pg_rows_day * pg_bpr / 1e6,
         근거="실측" if "pg" in m else f"가정: 한 행 {PG_ROW_BYTES}B, 행 수는 보고서 메시지 수"),
])
fc["한해_MB"] = fc["하루_MB"] * 365
fc.to_csv(OUT / "b1_forecast.csv", index=False)

# ---- 정책: 원본 30일 → 5분 집계(평균·최소·최대) 400일 → 1시간 집계 영구. 집계 대상 = env co2·온도·습도 시리즈
p5, p60 = MAIN_SERIES * 288 * 3, MAIN_SERIES * 24 * 3
raw_day_mb = pts_day * bpv / 1e6
def kept(keep_days, years):   # 보관 기간 상한을 둔 단계가 n년 뒤 들고 있는 날 수
    return min(keep_days, 365 * years)


INF = 10 ** 9
pg_by = m.get("pg", {}).get("rows_24h_by_measurement")
pg_main = pg_by.get("env", 0) + pg_by.get("hvac", 0) if pg_by else pg_rows_day
tiers = [  # (단계, 보관(일), 하루 값/행, 값·행당 바이트)
    ("InfluxDB 원본(모든 값)", 30, pts_day, bpv),
    ("InfluxDB 5분 집계 env(평균·최소·최대)", 400, p5, bpv),
    ("InfluxDB 1시간 집계 env(평균·최소·최대)", INF, p60, bpv),
    # PostgreSQL 원본: 분석에 쓰는 env·hvac 는 30일, 진단용 uplink·collector·inference 는 7일
    ("PostgreSQL 원본 readings (env·hvac)", 30, pg_main, pg_bpr),
    ("PostgreSQL 원본 readings (uplink·collector·inference)", 7, pg_rows_day - pg_main, pg_bpr),
    ("PostgreSQL 5분 집계 readings_5m(시리즈·5분 한 행, 약 120B)", 400, MAIN_SERIES * 288, 120),
    ("PostgreSQL 1시간 집계 readings_1h(시리즈·1시간 한 행, 약 120B)", INF, MAIN_SERIES * 24, 120),
]
pol = pd.DataFrame([dict(단계=n, 보관="영구" if k == INF else f"{k}일", 하루_건수=d, 한해뒤_MB=d * b * kept(k, 1) / 1e6, 삼년뒤_MB=d * b * kept(k, 3) / 1e6)
                    for n, k, d, b in tiers])
pol.to_csv(OUT / "b1_policy.csv", index=False)
m["assumptions"] = dict(PTS_PER_MSG=PTS_PER_MSG, PG_ROW_BYTES=PG_ROW_BYTES, exact="--exact" in sys.argv)
(OUT / "b1_measure.json").write_text(json.dumps(m, ensure_ascii=False, indent=1, default=str))
pd.set_option("display.width", 220)
print(json.dumps(m, ensure_ascii=False, indent=1, default=str))
print(fc.round(2).to_string(index=False))
print(pol.round(1).to_string(index=False))
nop = fc["한해_MB"].iloc[1] + fc["한해_MB"].iloc[2]
print(f"정책 없음: 1년 {nop:.0f} MB, 3년 {3 * nop:.0f} MB (압축 후 기준) / 정책 적용: 1년 뒤 {pol['한해뒤_MB'].sum():.0f} MB, 3년 뒤 {pol['삼년뒤_MB'].sum():.0f} MB")
