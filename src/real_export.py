"""실데이터(보드 InfluxDB history 버킷: 공용 iot-bucket에서 복사)를 합성 데이터와 같은 파일 형식으로 내보낸다.
그러면 prep·baselines·train_forecast를 그대로 실데이터에 돌릴 수 있다.

  python real_export.py [시작=2026-07-27] [끝=2026-10-20] → ~/work/data/real/

- sensors.csv: time(KST), space, device, co2, temperature, humidity (센서별 원본, 5분 칸에서 공간 평균이 된다)
- schedule.csv: 비어 있음(실제 교육장 시간표·예약 자료를 아직 받지 못했다)
- faults.csv: 비어 있음(실데이터에는 고장 정답이 없다)
- meta.json: 기간과 시간순 분할(앞 8주 학습, 2주 검증, 마지막 2주 시험)
"""
import json
import sys
from pathlib import Path

import pandas as pd
from influxdb_client import InfluxDBClient

ENV = dict(l.split("=", 1) for l in (Path.home() / "ingest/.env").read_text().splitlines() if "=" in l and not l.startswith("#"))
OUT = Path.home() / "work/data/real"
OUT.mkdir(parents=True, exist_ok=True)
start = pd.Timestamp(sys.argv[1] if len(sys.argv) > 1 else "2026-07-27")
end = pd.Timestamp(sys.argv[2] if len(sys.argv) > 2 else "2026-10-20")
SPACES = ["실습실", "회의실", "사무실"]

cli = InfluxDBClient(url="http://localhost:8087", token=ENV["INFLUX_TOKEN"], org=ENV["INFLUX_ORG"], timeout=600000)
t0 = (start - pd.Timedelta(hours=9)).strftime("%Y-%m-%dT%H:%M:%SZ")
t1 = (end - pd.Timedelta(hours=9)).strftime("%Y-%m-%dT%H:%M:%SZ")
q = f'''from(bucket:"history") |> range(start:{t0}, stop:{t1})
  |> filter(fn:(r)=> r._measurement=="env" and (r._field=="co2" or r._field=="temperature" or r._field=="humidity"))
  |> filter(fn:(r)=> r.location=="실습실" or r.location=="회의실" or r.location=="사무실")
  |> pivot(rowKey:["_time","device"], columnKey:["_field"], valueColumn:"_value")
  |> keep(columns:["_time","location","device","co2","temperature","humidity"])'''
df = cli.query_api().query_data_frame(q)
df = pd.concat(df) if isinstance(df, list) else df
df = df.rename(columns={"_time": "time", "location": "space"})
df["time"] = pd.to_datetime(df.time).dt.tz_convert("Asia/Seoul").dt.tz_localize(None).dt.floor("s")
for v in ("co2", "temperature", "humidity"):
    if v not in df:
        df[v] = float("nan")
df = df[["time", "space", "device", "co2", "temperature", "humidity"]].sort_values(["time", "space"])
df.to_csv(OUT / "sensors.csv", index=False)
pd.DataFrame(columns=["space", "start", "end", "kind", "expected"]).to_csv(OUT / "schedule.csv", index=False)
pd.DataFrame(columns=["space", "type", "start", "end", "sensor", "note"]).to_csv(OUT / "faults.csv", index=False)
weeks = (end - start).days / 7
# 시험 = 마지막 2주, 검증 = 그 앞 2주, 학습 = 나머지 전부(중간에 복사하지 않은 기간은 결측으로 남는다)
va_end = end - pd.Timedelta(weeks=2)
tr_end = va_end - pd.Timedelta(weeks=2)
meta = dict(source="iot-bucket(공용 InfluxDB) → 보드 history 버킷", start=str(start), end=str(end - pd.Timedelta(minutes=1)),
            timezone="Asia/Seoul (KST, naive)",
            split=dict(train=[str(start), str(tr_end)], val=[str(tr_end), str(va_end)], test=[str(va_end), str(end - pd.Timedelta(minutes=1))]),
            spaces={s: dict(name=s, report_min=1) for s in SPACES},
            counts=df.groupby("space").size().to_dict(), devices=df.groupby("space").device.unique().apply(list).to_dict())
(OUT / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1, default=str))
print(json.dumps(meta, ensure_ascii=False, indent=1, default=str))
