"""수집 현황 보고: 공간·센서·항목별 건수, 커버율, 최장 공백, 마지막 수신.

  python coverage.py [시간=24] [항목=co2] [measurement=env]

커버율은 두 가지로 낸다.
  1분 커버율   : 1분 칸 중 데이터가 있는 칸의 비율 (안내문 기준)
  주기 커버율  : 센서 자신의 보고 주기(중앙값) 기준으로 기대 건수 대비 실제 건수.
                 2~10분 주기 센서는 1분 커버율로는 95%를 넘을 수 없어서 함께 본다.
"""
import os
import sys
from datetime import datetime, timedelta, timezone

from influxdb_client import InfluxDBClient

hours = float(sys.argv[1]) if len(sys.argv) > 1 else 24
field = sys.argv[2] if len(sys.argv) > 2 else "co2"
meas = sys.argv[3] if len(sys.argv) > 3 else "env"
KST = timezone(timedelta(hours=9))

client = InfluxDBClient(url=os.environ["INFLUX_URL"], token=os.environ["INFLUX_TOKEN"], org=os.environ["INFLUX_ORG"])
q = f'''
from(bucket: "{os.environ["INFLUX_BUCKET"]}")
  |> range(start: -{int(hours * 60)}m)
  |> filter(fn: (r) => r._measurement == "{meas}" and r._field == "{field}")
  |> keep(columns: ["_time", "location", "device", "name"])
'''
rows = {}
for t in client.query_api().query(q):
    for r in t.records:
        key = (r.values.get("location") or r.values.get("name") or "-", r.values.get("device") or "-")
        rows.setdefault(key, []).append(r.get_time())

now = datetime.now(timezone.utc)
window_min = hours * 60
print(f"조회 구간: 최근 {hours:g}시간, 항목 {meas}.{field}, 기준 {now.astimezone(KST):%Y-%m-%d %H:%M} KST\n")
print(f"{'공간':<10}{'센서':<20}{'건수':>6}{'주기(분)':>9}{'1분커버율':>10}{'주기커버율':>10}{'최장공백(분)':>12}{'마지막(분 전)':>13}")
for (loc, dev), ts in sorted(rows.items()):
    ts.sort()
    span_min = min(window_min, (now - ts[0]).total_seconds() / 60 + 1)  # 첫 수신 이후만 분모로
    gaps = [(b - a).total_seconds() / 60 for a, b in zip(ts, ts[1:])]
    period = sorted(gaps)[len(gaps) // 2] if gaps else float("nan")
    minutes = {t.replace(second=0, microsecond=0) for t in ts}
    cov1 = min(1.0, len(minutes) / max(span_min, 1))
    covp = min(1.0, len(ts) / max(span_min / period, 1)) if gaps else float("nan")
    longest = max(gaps + [(now - ts[-1]).total_seconds() / 60])
    last = (now - ts[-1]).total_seconds() / 60
    print(f"{loc:<10}{dev:<20}{len(ts):>6}{period:>9.1f}{cov1:>10.1%}{covp:>10.1%}{longest:>12.1f}{last:>13.1f}")
client.close()
