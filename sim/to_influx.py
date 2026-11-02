"""sensors.csv(KST)를 InfluxDB line protocol로 바꿔 표준출력에 쓴다.
실데이터와 같은 구조(measurement env, tags location·device·model)로 synthetic 버킷에 넣는다.

  python3 to_influx.py data/synthetic/sensors.csv \
    | docker exec -i influxdb influx write --bucket synthetic --precision s
"""
import sys

import pandas as pd

df = pd.read_csv(sys.argv[1], parse_dates=["time"])
ts = (df.time.dt.tz_localize("Asia/Seoul").dt.tz_convert("UTC").astype("int64") // 10**9).astype(str)
lines = ("env,location=" + df.space + ",device=" + df.device + ",model=synthetic"
         + " co2=" + df.co2.astype(str) + ",temperature=" + df.temperature.astype(str)
         + ",humidity=" + df.humidity.astype(str) + " " + ts)
sys.stdout.write("\n".join(lines) + "\n")
