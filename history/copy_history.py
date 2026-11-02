"""공용 InfluxDB(iot-bucket, 읽기 전용) → Jetson team2-2 InfluxDB(history 버킷) 복사. Mac에서 실행한다.

공용 InfluxDB는 교육장 내부망(랜선)에서만 닿고, Jetson에서는 닿지 않는다.
그래서 Mac이 읽어서 SSH로 Jetson의 `influx write`에 흘려 넣는다(Jetson 쪽 토큰이 Mac에 필요 없다).

  python3 copy_history.py [시작=2025-08-01] [끝=지금]
  FIELDS=co2,temperature python3 copy_history.py …   # 항목만 골라서

- 저장 형태: measurement `env`, tags location·device·point·source=iot-bucket, field = 원래 measurement 이름(co2, temperature …)
  → 실시간 수집(`sensors` 버킷, env)과 같은 모양이라 같은 코드로 읽을 수 있다.
- device: 장치 EUI 끝 6자리가 실시간 수집의 장치 이름 끝 6자리와 같으면 그 이름(AM107-067999 등)으로 바꾼다.
- 한 달씩, 항목별로 나눠 옮긴다. 같은 구간을 다시 돌려도 InfluxDB가 같은 점을 덮어쓰므로 중복이 생기지 않는다.
- 시각: 공용 버킷은 네트워크 서버 수신 시각을 쓰고 있어 실시간 수집(nsTime)과 0.2초 차이뿐임을 확인했다(2026-10-20).
"""
import csv
import io
import os
import subprocess
import sys
import time
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ENV = dict(line.split("=", 1) for line in (HERE / "src.env").read_text().split() if "=" in line)
URL, TOKEN = ENV["SRC_URL"], ENV["SRC_TOKEN"]
ORG = ENV["SRC_ORG"]               # iot-bucket의 orgID (토큰이 org 목록 읽기 권한은 없어 버킷 정보에서 얻음)
JETSON = "team2-2@100.92.224.43"
FIELDS = ["co2", "temperature", "humidity", "activity", "illumination", "infrared", "infrared_and_visible",
          "pressure", "tvoc", "LAeq", "LAI", "LAImax", "magnet_status", "battery", "rssi", "snr"]
NAMES = {"067999": "AM107-067999", "140101": "AM107-140101", "081175": "AM103-081175", "089152": "AM103-089152",
         "152590": "EM500-CO2-152590", "152862": "EM500-CO2-152862", "389010": "EM320-TH-389010", "389818": "EM320-TH-389818",
         "151523": "EM300-TH-151523", "151547": "EM300-TH-151547", "151606": "EM300-TH-151606", "151836": "EM300-TH-151836",
         "012436": "WS302-012436"}


def esc(v):
    return str(v).replace("\\", "\\\\").replace(" ", "\\ ").replace(",", "\\,").replace("=", "\\=")


def query(flux):
    req = urllib.request.Request(f"{URL}/api/v2/query?orgID={ORG}", data=flux.encode(), method="POST",
                                 headers={"Authorization": f"Token {TOKEN}", "Content-Type": "application/vnd.flux", "Accept": "application/csv"})
    with urllib.request.urlopen(req, timeout=900) as r:
        return r.read().decode()


def to_lines(text, field):
    out = []
    header = None
    for row in csv.reader(io.StringIO(text)):
        if not row or len(row) < 4:
            header = None
            continue
        if header is None:
            header = {k: i for i, k in enumerate(row)}
            continue
        try:
            t, v = row[header["_time"]], float(row[header["_value"]])
        except (ValueError, KeyError):
            continue
        loc = row[header["location"]] if "location" in header else ""
        dev = row[header["device_id"]] if "device_id" in header else ""
        pt = row[header["point"]] if "point" in header else ""
        dev = NAMES.get(dev[-6:], dev) if dev else f"unknown-{loc}-{pt}"
        ts = parse_ms(t)
        tags = f"env,location={esc(loc or 'unknown')},device={esc(dev)},source=iot-bucket" + (f",point={esc(pt)}" if pt else "")
        out.append(f"{tags} {field}={v} {ts}")
    return out


def parse_ms(t):
    t = t.rstrip("Z")
    base, _, frac = t.partition(".")
    dt = datetime.fromisoformat(base).replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000) + int((frac + "000")[:3] or 0)


def write(lines):
    p = subprocess.run(["ssh", "-o", "BatchMode=yes", JETSON, "docker exec -i t22-influxdb influx write --bucket history --precision ms"],
                       input="\n".join(lines) + "\n", text=True, capture_output=True)
    if p.returncode != 0:
        raise RuntimeError(p.stderr[-500:])


def main():
    start = datetime.fromisoformat(sys.argv[1] if len(sys.argv) > 1 else "2025-08-01").replace(tzinfo=timezone.utc)
    end = datetime.fromisoformat(sys.argv[2]).replace(tzinfo=timezone.utc) if len(sys.argv) > 2 else datetime.now(timezone.utc)
    subprocess.run(["ssh", "-o", "BatchMode=yes", JETSON,
                    "docker exec t22-influxdb influx bucket list | grep -qw history || docker exec t22-influxdb influx bucket create --name history --retention 0 >/dev/null"],
                   check=True)
    total, t0 = 0, time.time()
    cur = start
    while cur < end:
        nxt = min((cur.replace(day=1) + timedelta(days=32)).replace(day=1), end)
        month_n = 0
        for f in (os.getenv("FIELDS").split(",") if os.getenv("FIELDS") else FIELDS):
            flux = f'''from(bucket:"iot-bucket") |> range(start:{cur.isoformat().replace("+00:00", "Z")}, stop:{nxt.isoformat().replace("+00:00", "Z")})
              |> filter(fn:(r)=> r._measurement=="{f}" and r._field=="value")
              |> filter(fn:(r)=> exists r._value)
              |> map(fn:(r)=> ({{r with _value: string(v: r._value)}}))
              |> group() |> keep(columns:["_time","_value","location","device_id","point"])'''
            try:
                lines = to_lines(query(flux), f)
            except Exception as e:
                print(f"  {cur:%Y-%m} {f}: 읽기 실패 {e}", flush=True)
                continue
            for i in range(0, len(lines), 50000):
                write(lines[i:i + 50000])
            month_n += len(lines)
        total += month_n
        print(f"{cur:%Y-%m}: {month_n:,}건 (누적 {total:,}, {time.time() - t0:.0f}초)", flush=True)
        cur = nxt
    print(f"완료: {total:,}건")


if __name__ == "__main__":
    main()
