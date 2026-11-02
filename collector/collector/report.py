"""수집 현황 보고서(마크다운): 공간·센서·항목별 커버율 표 + 결측 구간과 추정 원인.

  python report.py [시간=24] > reports/coverage_$(date +%Y%m%d_%H%M).md

결측 원인 추정 규칙
  - 그 시간에 수집기가 MQTT에서 끊겨 있었다        → 수집기·네트워크 끊김 (우리 쪽)
  - 공백 앞뒤 fCnt(센서 전송 번호)가 건너뛰었다       → 전송 유실 (무선 구간·게이트웨이, 센서는 보냈음)
  - fCnt가 이어진다(건너뛰지 않음)                    → 센서가 보내지 않음 (송신 주기 변화·전원·센서 쪽)
"""
import os
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone

from influxdb_client import InfluxDBClient

HOURS = float(sys.argv[1]) if len(sys.argv) > 1 else 24
KST = timezone(timedelta(hours=9))
FIELDS = ["co2", "temperature", "humidity"]
GAP_FACTOR = 3  # 자기 주기의 3배 이상 비면 '결측 구간'으로 본다

client = InfluxDBClient(url=os.environ["INFLUX_URL"], token=os.environ["INFLUX_TOKEN"], org=os.environ["INFLUX_ORG"])
qa = client.query_api()
B = os.environ["INFLUX_BUCKET"]
now = datetime.now(timezone.utc)
start = now - timedelta(hours=HOURS)


def q(flux):
    return [r for t in qa.query(flux) for r in t.records]


# 센서 값
series = defaultdict(list)  # (loc, dev, field) -> [time]
for r in q(f'''from(bucket:"{B}") |> range(start:-{int(HOURS*60)}m)
  |> filter(fn:(r)=> r._measurement=="env" and contains(value:r._field, set:{FIELDS!r}))
  |> keep(columns:["_time","_field","location","device"])'''.replace("'", '"')):
    series[(r["location"], r["device"], r["_field"])].append(r.get_time())

# 센서 전송 번호
fcnt = defaultdict(list)  # dev -> [(time, fcnt)]
for r in q(f'''from(bucket:"{B}") |> range(start:-{int(HOURS*60)}m)
  |> filter(fn:(r)=> r._measurement=="uplink" and r._field=="fcnt")
  |> keep(columns:["_time","_value","device"])'''):
    fcnt[r["device"]].append((r.get_time(), r.get_value()))

# 수집기 끊김 구간 (1분 상태 기록에서 mqtt_connected==0 이거나 상태 기록 자체가 빈 구간)
beats = sorted((r.get_time(), r.get_value()) for r in q(f'''from(bucket:"{B}") |> range(start:-{int(HOURS*60)}m)
  |> filter(fn:(r)=> r._measurement=="collector" and r._field=="mqtt_connected")'''))
down = []
for (t0, v0), (t1, _) in zip(beats, beats[1:]):
    if v0 == 0 or (t1 - t0) > timedelta(minutes=2.5):
        down.append((t0, t1))
collect_start = beats[0][0] if beats else start


def overlaps(a0, a1, ranges):
    return any(a0 < b1 and b0 < a1 for b0, b1 in ranges)


def cause(dev, g0, g1):
    if g0 < collect_start:
        return "수집기 가동 전"
    if overlaps(g0, g1, down):
        return "수집기·네트워크 끊김"
    seq = sorted(fcnt.get(dev, []))
    before = [f for t, f in seq if t <= g0]
    after = [f for t, f in seq if t >= g1]
    if before and after:
        jump = after[0] - before[-1]
        if jump > 1:
            return f"전송 유실 (fCnt {int(jump) - 1}건 건너뜀)"
        return "센서가 보내지 않음 (fCnt 연속)"
    return "판단 불가 (fCnt 없음)"


def kst(t):
    return t.astimezone(KST).strftime("%m-%d %H:%M")


print(f"# 수집 현황 보고\n")
print(f"- 조회 구간: {kst(start)} ~ {kst(now)} KST (최근 {HOURS:g}시간)")
print(f"- 수집기 첫 상태 기록: {kst(collect_start)} KST, 수집기 끊김 {len(down)}회")
print(f"- 데이터: 실데이터 (MQTT `data.withio.net`), 저장소 InfluxDB `{B}`\n")
print("## 커버율\n")
print("| 공간 | 센서 | 항목 | 건수 | 주기(분) | 1분 커버율 | 주기 커버율 | 최장 공백(분) | 마지막 수신 |")
print("|---|---|---|---:|---:|---:|---:|---:|---|")
gaps_out = []
seen_gap = set()
for (loc, dev, field), ts in sorted(series.items()):
    ts.sort()
    t_from = max(start, collect_start)
    span = (now - t_from).total_seconds() / 60
    d = [(b - a).total_seconds() / 60 for a, b in zip(ts, ts[1:])]
    period = sorted(d)[len(d) // 2] if d else float("nan")
    minutes = {t.replace(second=0, microsecond=0) for t in ts if t >= t_from}
    cov1 = min(1.0, len(minutes) / max(span, 1))
    covp = min(1.0, len([t for t in ts if t >= t_from]) / max(span / period, 1)) if d else float("nan")
    longest = max(d + [(now - ts[-1]).total_seconds() / 60]) if ts else 0
    print(f"| {loc} | {dev} | {field} | {len(ts)} | {period:.1f} | {cov1:.1%} | {covp:.1%} | {longest:.0f} | {kst(ts[-1])} |")
    for a, b, m in zip(ts, ts[1:], d):
        if m >= GAP_FACTOR * period and m >= 5 and (dev, a) not in seen_gap:  # 센서당 한 번만
            seen_gap.add((dev, a))
            gaps_out.append((loc, dev, a, b, m, cause(dev, a, b)))

print("\n1분 커버율 = 데이터가 있는 1분 칸 / 전체 1분 칸. 주기 커버율 = 실제 건수 / (구간 ÷ 센서 자신의 보고 주기).")
print("보고 주기가 1분보다 긴 센서는 1분 커버율로는 95%를 넘을 수 없어서 주기 커버율을 함께 본다.\n")
print(f"## 결측 구간 (자기 주기의 {GAP_FACTOR}배 이상, 5분 이상 빈 구간)\n")
if not gaps_out:
    print("없음")
else:
    print("| 공간 | 센서 | 시작 | 끝 | 길이(분) | 추정 원인 |")
    print("|---|---|---|---|---:|---|")
    for loc, dev, a, b, m, c in sorted(gaps_out, key=lambda x: x[2]):
        print(f"| {loc} | {dev} | {kst(a)} | {kst(b)} | {m:.0f} | {c} |")
    summary = defaultdict(int)
    for *_, c in gaps_out:
        summary[c.split(" (")[0]] += 1
    print("\n원인별 건수: " + ", ".join(f"{k} {v}건" for k, v in sorted(summary.items(), key=lambda x: -x[1])))
client.close()
