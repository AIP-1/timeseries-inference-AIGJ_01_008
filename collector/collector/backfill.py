"""InfluxDB(보드 안 원본) → PostgreSQL 재동기화.

PostgreSQL이 오래 끊겼거나 수집기 재시작으로 메모리 보관분을 잃었을 때 빈 구간을 채운다.
기본 키(measurement, series, ts)가 같으면 건너뛰므로 같은 구간을 여러 번 돌려도 안전하다.

  python backfill.py [시작=-24h] [끝=now()]
  예) python backfill.py 2026-10-20T06:00:00Z 2026-10-20T07:30:00Z

시계 보정: rx_delay_s가 -60초보다 작은 값(게이트웨이 시계가 빨라 미래로 찍힌 값)은
측정 시각을 수신 시각(ts + rx_delay_s)으로 고쳐서 넣는다.
"""
import os
import sys
from datetime import timedelta

import psycopg
from influxdb_client import InfluxDBClient
from psycopg.types.json import Jsonb

from collector import PG_DDL, series_key

start = sys.argv[1] if len(sys.argv) > 1 else "-24h"
stop = sys.argv[2] if len(sys.argv) > 2 else "now()"
TAG_KEYS = {"location", "device", "model", "gateway", "name", "unit_type", "unit_id", "topic", "space"}

influx = InfluxDBClient(url=os.environ["INFLUX_URL"], token=os.environ["INFLUX_TOKEN"], org=os.environ["INFLUX_ORG"])
flux = f'''from(bucket:"{os.environ["INFLUX_BUCKET"]}") |> range(start:{start}, stop:{stop})
  |> pivot(rowKey:["_time"], columnKey:["_field"], valueColumn:"_value")'''
rows, fixed = [], 0
for table in influx.query_api().query(flux):
    for r in table.records:
        v = r.values
        meas = v["_measurement"]
        tags = {k: v[k] for k in TAG_KEYS if v.get(k) not in (None, "")}
        fields = {}
        for k, x in v.items():
            if k.startswith("_") or k in TAG_KEYS | {"result", "table", "rx_delay_s"} or x is None:
                continue
            try:
                fields[k] = float(x)
            except (TypeError, ValueError):   # 숫자가 아닌 값(새로 생긴 태그 등)은 건너뛴다 — 한 값 때문에 전체가 멈추지 않게
                continue
        if not fields:
            continue
        ts = r.get_time()
        delay = v.get("rx_delay_s")
        if delay is not None and delay < -60:
            ts = ts + timedelta(seconds=delay)
            delay = 0.0
            fixed += 1
        rows.append((ts, meas, series_key(tags), Jsonb(tags), Jsonb(fields), delay))
influx.close()

conn = psycopg.connect(host=os.environ["PG_HOST"], port=os.getenv("PG_PORT", "5432"), dbname=os.environ["PG_DB"],
                       user=os.environ["PG_USER"], password=os.environ["PG_PASSWORD"],
                       options=f"-c search_path={os.getenv('PG_SCHEMA', 'public')}")
with conn.cursor() as cur:
    cur.execute(PG_DDL)
    cur.execute("SELECT count(*) FROM readings")
    before = cur.fetchone()[0]
    # 예전 기록은 InfluxDB와 PostgreSQL 시각이 1ms 어긋난 경우가 있어 ±2ms 안에 같은 기록이 있으면 건너뛴다
    insert = """INSERT INTO readings (ts, measurement, series, tags, fields, rx_delay_s)
      SELECT %(ts)s, %(m)s, %(s)s, %(t)s, %(f)s, %(d)s
      WHERE NOT EXISTS (SELECT 1 FROM readings WHERE measurement = %(m)s AND series = %(s)s
                        AND ts BETWEEN %(ts)s - interval '2 milliseconds' AND %(ts)s + interval '2 milliseconds')
      ON CONFLICT DO NOTHING"""
    params = [dict(ts=r[0], m=r[1], s=r[2], t=r[3], f=r[4], d=r[5]) for r in rows]
    for i in range(0, len(params), 2000):
        cur.executemany(insert, params[i:i + 2000])
    cur.execute("SELECT count(*) FROM readings")
    after = cur.fetchone()[0]
conn.commit()
conn.close()
print(f"InfluxDB에서 {len(rows)}건 읽음(시계 보정 {fixed}건) → PostgreSQL 새로 추가 {after - before}건, 이미 있어 건너뜀 {len(rows) - (after - before)}건")
