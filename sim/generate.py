"""합성 데이터 생성기 (과제 2~8용).

교육장 4개 공간(강의실 A·B, 회의실, 라운지)의 CO₂·온도·습도를 1분 단위 물리 모델로 만든다.
실제 사용(인원)은 일정(시간표·예약)을 따르되 결석·노쇼·연장·예약 없는 사용으로 어긋난다.
센서 값에는 기록 주기 차이, 게이트웨이 끊김, 깨진 값, 잡음을 넣고,
마지막 1주에는 정답을 아는 고장 5종을 심는다.

출력 (OUT 폴더)
  sensors.csv     센서가 실제로 보낸 값(결측·깨진 값·고장 포함). time(KST), space, device, co2, temperature, humidity
  schedule.csv    운영 자료: 수업 시간표·회의실 예약·라운지 혼잡 예상. 미래 값도 미리 알 수 있는 정보
  truth.csv       1분 단위 정답: 실제 인원(occupants), 사용 여부(occupied), 고장 없는 참값(co2_true, temp_true), 고장 종류(fault)
  faults.csv      고장 이벤트 목록: space, type, start, end, sensor, 설명
  meta.json       생성 조건(시드, 기간, 공간 크기, 분할 기준)

사용법
  python generate.py --out data/synthetic --weeks 6 --seed 42
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------- 공간 설정
# volume: 실내 공기 부피(m³), ach_on/off: 환기 가동/정지 시 시간당 환기 횟수,
# report_min: 센서 기록 주기(분), capacity: 정원
SPACES = {
    "classroom_a": dict(name="강의실 A", device="ENV-CA-01", volume=270, ach_on=2.2, ach_off=0.25,
                        report_min=1, capacity=30, ua=0.020, mass=1.0),
    "classroom_b": dict(name="강의실 B", device="ENV-CB-01", volume=200, ach_on=2.0, ach_off=0.25,
                        report_min=1, capacity=24, ua=0.022, mass=0.85),
    "meeting":     dict(name="회의실",   device="ENV-MT-01", volume=60,  ach_on=3.0, ach_off=0.3,
                        report_min=2, capacity=8,  ua=0.030, mass=0.6),
    "lounge":      dict(name="라운지",   device="ENV-LG-01", volume=180, ach_on=2.5, ach_off=0.35,
                        report_min=5, capacity=40, ua=0.028, mass=0.9),
}
CO2_OUT = 420.0          # 외기 CO₂ (ppm)
CO2_GEN = 0.0003         # 1인 CO₂ 배출량 (m³/분, 앉아서 공부 ≈ 0.3 L/분)
HEAT_PER_PERSON = 0.010  # 1인 발열이 만드는 온도 상승률 (℃/분, 부피 100m³ 기준)
T_SET = 24.0             # 냉방 설정 온도
HVAC_TAU = 12.0          # 냉방이 설정 온도로 끌어당기는 시간 상수 (분)
OPEN_HOURS = (8.5, 19.0) # 평일 냉난방·환기 가동 시간 (08:30~19:00)

# 수업 시간표 (강의실). 50분 수업 + 10분 쉬는 시간, 점심 12:00~13:00
CLASS_BLOCKS = [(9, 0, 9, 50), (10, 0, 10, 50), (11, 0, 11, 50),
                (13, 0, 13, 50), (14, 0, 14, 50), (15, 0, 15, 50), (16, 0, 16, 50), (17, 0, 17, 50)]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="data/synthetic")
    p.add_argument("--weeks", type=int, default=6)
    p.add_argument("--start", default="2026-08-31")  # 월요일
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# ---------------------------------------------------------------- 일정과 실제 사용
def build_schedule_and_occupancy(idx, rng):
    """일정(운영 자료)과 실제 인원(정답)을 함께 만든다. 둘은 일부러 어긋난다."""
    n = len(idx)
    occ = {s: np.zeros(n) for s in SPACES}
    rows = []
    minute_of = {t: i for i, t in enumerate(idx)}
    days = pd.date_range(idx[0].normalize(), idx[-1].normalize(), freq="D")

    def put(space, start, end, people):
        i0 = minute_of.get(start)
        i1 = minute_of.get(end)
        if i0 is None:
            return
        i1 = n if i1 is None else i1
        occ[space][i0:i1] = np.maximum(occ[space][i0:i1], people)

    for day in days:
        if day.weekday() >= 5:
            continue
        # --- 강의실: 과정마다 정원, 매일 출석 인원이 다르다
        for space, base in (("classroom_a", 26), ("classroom_b", 18)):
            if rng.random() < 0.04:          # 휴강일(시간표에는 있으나 아무도 안 옴)
                attendance = 0
            else:
                attendance = int(np.clip(rng.normal(base, 2.5), 4, SPACES[space]["capacity"]))
            half_day = rng.random() < 0.08   # 오후 자율학습 → 절반만 남음
            for bi, (h0, m0, h1, m1) in enumerate(CLASS_BLOCKS):
                start = day + pd.Timedelta(hours=h0, minutes=m0)
                end = day + pd.Timedelta(hours=h1, minutes=m1)
                rows.append(dict(space=space, start=start, end=end, kind="class", expected=base))
                if attendance == 0:
                    continue
                people = attendance
                if half_day and h0 >= 13:
                    people = max(2, attendance // 2)
                # 지각·조기 퇴실, 마지막 교시 연장
                s_shift = int(rng.integers(-3, 6))
                e_shift = int(rng.integers(-2, 4))
                if bi == len(CLASS_BLOCKS) - 1 and rng.random() < 0.15:
                    e_shift += int(rng.integers(10, 40))
                put(space, start + pd.Timedelta(minutes=s_shift), end + pd.Timedelta(minutes=e_shift), people)
                # 쉬는 시간에 일부만 남는다
                if bi < len(CLASS_BLOCKS) - 1 and h0 != 11:
                    br0 = end + pd.Timedelta(minutes=e_shift)
                    br1 = day + pd.Timedelta(hours=CLASS_BLOCKS[bi + 1][0], minutes=CLASS_BLOCKS[bi + 1][1])
                    put(space, br0, br1, max(1, int(people * rng.uniform(0.3, 0.6))))
            # 점심에 남아서 먹는 사람
            if attendance and rng.random() < 0.5:
                put("classroom_a" if space == "classroom_a" else space,
                    day + pd.Timedelta(hours=12, minutes=5), day + pd.Timedelta(hours=12, minutes=55),
                    int(rng.integers(1, 5)))

        # --- 회의실: 예약 2~5건, 노쇼·연장·예약 없는 사용
        t = day + pd.Timedelta(hours=9)
        n_book = int(rng.integers(2, 6))
        slots = sorted(rng.choice(np.arange(9 * 2, 18 * 2), size=n_book, replace=False))
        last_end = None
        for sl in slots:
            start = day + pd.Timedelta(minutes=int(sl) * 30)
            if last_end is not None and start < last_end:
                continue
            dur = int(rng.choice([30, 60, 60, 90, 120]))
            end = start + pd.Timedelta(minutes=dur)
            people = int(rng.integers(2, 9))
            rows.append(dict(space="meeting", start=start, end=end, kind="booking", expected=people))
            last_end = end
            if rng.random() < 0.15:          # 노쇼
                continue
            s_shift = int(rng.integers(-5, 12))
            e_shift = int(rng.integers(-15, 10)) if rng.random() > 0.2 else int(rng.integers(10, 30))
            real = max(1, people + int(rng.integers(-2, 2)))
            put("meeting", start + pd.Timedelta(minutes=s_shift), end + pd.Timedelta(minutes=e_shift), real)
        if rng.random() < 0.25:              # 예약 없는 사용
            start = day + pd.Timedelta(minutes=int(rng.integers(9 * 60, 17 * 60)))
            put("meeting", start, start + pd.Timedelta(minutes=int(rng.integers(20, 60))), int(rng.integers(2, 5)))

        # --- 라운지: 쉬는 시간·점심에 몰린다. 일정 자료는 '혼잡 예상 시간대'
        for (h0, m0, h1, m1) in CLASS_BLOCKS[:-1]:
            if h0 == 11:
                continue
            start = day + pd.Timedelta(hours=h1, minutes=m1)
            end = start + pd.Timedelta(minutes=10)
            rows.append(dict(space="lounge", start=start, end=end, kind="break", expected=12))
            put("lounge", start, end, int(rng.integers(5, 20)))
        lunch0 = day + pd.Timedelta(hours=12)
        lunch1 = day + pd.Timedelta(hours=13)
        rows.append(dict(space="lounge", start=lunch0, end=lunch1, kind="lunch", expected=25))
        put("lounge", lunch0, lunch1, int(rng.integers(15, 35)))
        # 그 밖에 드문드문 머무는 사람
        for _ in range(int(rng.integers(3, 8))):
            start = day + pd.Timedelta(minutes=int(rng.integers(8 * 60 + 30, 19 * 60)))
            put("lounge", start, start + pd.Timedelta(minutes=int(rng.integers(5, 40))), int(rng.integers(1, 4)))

    sched = pd.DataFrame(rows).sort_values(["space", "start"]).reset_index(drop=True)
    return sched, occ


# ---------------------------------------------------------------- 물리 모델
def outdoor_temp(idx, rng):
    doy = idx.dayofyear.values
    hour = idx.hour.values + idx.minute.values / 60
    seasonal = 27 - 0.17 * (doy - 243)                      # 9월 초 27℃ → 10월 중순 20℃
    daily = 4.5 * np.sin((hour - 9) / 24 * 2 * np.pi)       # 15시 최고, 3시 최저
    day_noise = np.repeat(rng.normal(0, 1.8, len(idx) // 1440 + 1), 1440)[: len(idx)]
    return seasonal + daily + day_noise


def simulate(idx, occ, faults_true, rng):
    n = len(idx)
    hour = idx.hour.values + idx.minute.values / 60
    weekday = idx.weekday.values < 5
    open_now = weekday & (hour >= OPEN_HOURS[0]) & (hour < OPEN_HOURS[1])
    t_out = outdoor_temp(idx, rng)
    h_out = np.clip(65 + 15 * np.sin((hour - 3) / 24 * 2 * np.pi) + rng.normal(0, 3, n), 35, 95)

    out = {}
    for s, cfg in SPACES.items():
        V = cfg["volume"]
        N = occ[s]
        q = np.where(open_now, cfg["ach_on"], cfg["ach_off"]) * V / 60.0     # m³/분
        # 쉬는 시간·점심에 가끔 창문을 연다(강의실·라운지)
        window = np.zeros(n)
        if s != "meeting":
            for day_start in range(0, n, 1440):
                for h in (10.83, 12.0, 14.83, 15.83):
                    if rng.random() < 0.18:
                        i0 = day_start + int(h * 60)
                        window[i0:i0 + int(rng.integers(8, 25))] = 1
        q = q + window * V * 6 / 60.0
        # 회의실은 사용 중일 때만 냉방, 나머지는 운영 시간 동안
        if s == "meeting":
            hvac = open_now & (np.convolve(N > 0, np.ones(20), "same") > 0)
        else:
            hvac = open_now.copy()
        vent_factor = np.ones(n)
        for f in faults_true:
            if f["space"] != s:
                continue
            i0, i1 = f["i0"], f["i1"]
            if f["type"] == "ac_failure":
                hvac[i0:i1] = False
            if f["type"] == "vent_failure":
                vent_factor[i0:i1] = 1 / 1.8 * 0.9  # 같은 인원에서 상승 폭 ≈ 1.8배 이상

        co2 = np.empty(n)
        temp = np.empty(n)
        hum = np.empty(n)
        c, t, h = CO2_OUT + 30, 24.0, 50.0
        heat = HEAT_PER_PERSON * 100 / V / cfg["mass"]
        for i in range(n):
            qi = q[i] * (vent_factor[i] if open_now[i] else 1.0)
            c += (CO2_GEN * N[i] * 1e6 - qi * (c - CO2_OUT)) / V
            dt = cfg["ua"] * (t_out[i] - t) + heat * N[i]
            if hvac[i]:
                dt += (T_SET - t) / HVAC_TAU
            t += dt
            target_h = 0.6 * h_out[i] + 0.4 * 45 + 0.35 * N[i] * 100 / V
            if hvac[i]:
                target_h = 0.75 * target_h + 0.25 * 50
            h += (target_h - h) / 40.0
            co2[i], temp[i], hum[i] = c, t, h
        out[s] = dict(co2=co2, temp=temp, hum=np.clip(hum, 15, 95))
    return out


# ---------------------------------------------------------------- 고장
def plan_faults(idx, occ, rng, test_start):
    """마지막 1주에 5종 고장을 심는다. 종류마다 2~4건, 공간을 돌아가며."""
    n = len(idx)
    i_test = int(np.searchsorted(idx, test_start))
    spaces = list(SPACES)
    hour = idx.hour.values + idx.minute.values / 60
    weekday = idx.weekday.values < 5
    faults = []
    used = {s: np.zeros(n, bool) for s in spaces}

    def free(s, i0, i1):
        return not used[s][max(0, i0 - 120):min(n, i1 + 120)].any()

    def add(s, typ, i0, i1, sensor, note, **kw):
        used[s][i0:i1] = True
        faults.append(dict(space=s, type=typ, i0=i0, i1=i1, sensor=sensor, note=note, **kw))

    def pick(cond, dur, s):
        cand = np.where(cond[i_test:n - dur])[0] + i_test
        rng.shuffle(cand)
        for i0 in cand[:500]:
            if free(s, i0, i0 + dur):
                return int(i0)
        return None

    # 1) 튄 값: 한 칸만 크게 튐 (6건)
    for k in range(6):
        s = spaces[k % 4]
        i0 = pick(np.ones(n, bool), 1, s)
        sensor = "co2" if k % 3 else "temperature"
        mag = float(rng.uniform(500, 1500)) if sensor == "co2" else float(rng.choice([1, -1]) * rng.uniform(6, 12))
        add(s, "spike", i0, i0 + 1, sensor, f"{sensor} {mag:+.0f}", magnitude=mag)
    # 2) 값 고정: 1~4시간 마지막 값 반복 (3건, 사용 시간대 포함)
    for k, s in enumerate(["classroom_a", "meeting", "lounge"]):
        dur = int(rng.integers(60, 241))
        i0 = pick(weekday & (hour > 9) & (hour < 15), dur, s)
        add(s, "stuck", i0, i0 + dur, "all", f"{dur}분 고정")
    # 3) 영점 틀어짐: 수 시간에 걸쳐 CO₂가 수백 ppm 오르고 그대로 유지 (2건)
    for s in ["classroom_b", "lounge"]:
        ramp = int(rng.integers(240, 480))
        hold = int(rng.integers(600, 1200))
        amount = float(rng.uniform(250, 450))
        i0 = pick(np.ones(n, bool), ramp + hold, s)
        add(s, "drift", i0, i0 + ramp + hold, "co2", f"{ramp}분 동안 +{amount:.0f}ppm", ramp=ramp, amount=amount)
    # 4) 냉방 정지: 수업 중 냉방이 꺼져 온도가 서서히 오름 (3건, 인원 있는 공간)
    for s in ["classroom_a", "classroom_b", "meeting"]:
        dur = int(rng.integers(120, 240))
        cond = weekday & (hour > 9.2) & (hour < 16) & (occ[s] > 0)
        i0 = pick(cond, dur, s)
        add(s, "ac_failure", i0, i0 + dur, "temperature", f"{dur}분 냉방 정지")
    # 5) 환기 정지: 사용 중 환기량 감소 → CO₂ 상승 폭 ≈ 1.8배 (3건)
    for s in ["classroom_b", "classroom_a", "meeting"]:
        dur = int(rng.integers(120, 240))
        cond = weekday & (hour > 9.2) & (hour < 16) & (occ[s] > 0)
        i0 = pick(cond, dur, s)
        add(s, "vent_failure", i0, i0 + dur, "co2", f"{dur}분 환기 정지")
    return faults


def apply_ac_rise(true_vals, faults):
    """냉방 정지: 안내문 정의대로 수업 중 온도가 서서히 3~5℃ 오르게 한다(실제 공기 온도, 참값에 적용).
    v1 생성기는 냉방만 꺼서, 바깥이 서늘한 10월에는 온도가 오르지 않아 고장이 데이터에 나타나지 않았다(2026-10-21 수정).
    난수를 쓰지 않으므로 냉방 정지 구간(과 끝난 뒤 60분 회복) 밖의 데이터는 v1과 같다."""
    for f in faults:
        if f["type"] != "ac_failure":
            continue
        i0, i1 = f["i0"], f["i1"]
        dur = i1 - i0
        amount = float(np.clip(3 + 2 * (dur - 120) / 120, 3, 5))      # 120분 → 3℃, 240분 → 5℃
        t = true_vals[f["space"]]["temp"]
        ramp = amount * np.arange(1, dur + 1) / dur                       # 서서히 오름
        t[i0:i1] += ramp
        rec = np.arange(1, 61)
        t[i1:i1 + 60] += amount * np.exp(-rec / HVAC_TAU)[: len(t[i1:i1 + 60])]   # 냉방 복구 후 12분 시간 상수로 회복
        f["amount"] = amount


def apply_sensor_faults(vals, faults, idx):
    for f in faults:
        v = vals[f["space"]]
        i0, i1 = f["i0"], f["i1"]
        if f["type"] == "spike":
            key = "co2" if f["sensor"] == "co2" else "temp"
            v[key][i0] += f["magnitude"]
        elif f["type"] == "stuck":
            for key in ("co2", "temp", "hum"):
                v[key][i0:i1] = v[key][i0 - 1]
        elif f["type"] == "drift":
            ramp = f["ramp"]
            off = np.concatenate([np.linspace(0, f["amount"], ramp), np.full(i1 - i0 - ramp, f["amount"])])
            v["co2"][i0:i1] += off


# ---------------------------------------------------------------- 메인
def main():
    a = parse_args()
    rng = np.random.default_rng(a.seed)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    start = pd.Timestamp(a.start)
    idx = pd.date_range(start, start + pd.Timedelta(weeks=a.weeks), freq="1min", inclusive="left")
    n = len(idx)
    test_start = idx[0] + pd.Timedelta(weeks=a.weeks - 1)
    val_start = test_start - pd.Timedelta(weeks=1)

    sched, occ = build_schedule_and_occupancy(idx, rng)
    faults = plan_faults(idx, occ, rng, test_start)
    true_vals = simulate(idx, occ, faults, rng)
    apply_ac_rise(true_vals, faults)

    # 측정 잡음 → 센서 고장 → 기록 주기·결측·깨진 값
    meas = {}
    for s, v in true_vals.items():
        meas[s] = dict(co2=v["co2"] + rng.normal(0, 8, n),
                       temp=v["temp"] + rng.normal(0, 0.06, n),
                       hum=v["hum"] + rng.normal(0, 0.6, n))
    apply_sensor_faults(meas, faults, idx)

    # 게이트웨이 끊김: 공간 공통 짧은 끊김 + 공간별 긴 끊김
    gap = {s: np.zeros(n, bool) for s in SPACES}
    for _ in range(a.weeks * 6):                      # 공통 짧은 끊김 2~12분
        i0 = int(rng.integers(0, n - 15))
        for s in SPACES:
            gap[s][i0:i0 + int(rng.integers(2, 13))] = True
    for _ in range(a.weeks):                          # 공간별 긴 끊김 1~6시간
        s = list(SPACES)[int(rng.integers(0, 4))]
        i0 = int(rng.integers(0, n - 400))
        gap[s][i0:i0 + int(rng.integers(60, 361))] = True
    for _ in range(a.weeks * 30):                     # 개별 메시지 유실
        s = list(SPACES)[int(rng.integers(0, 4))]
        gap[s][int(rng.integers(0, n))] = True

    rows = []
    broken = []
    for s, cfg in SPACES.items():
        step = cfg["report_min"]
        phase = int(rng.integers(0, step))
        keep = (np.arange(n) % step == phase) & ~gap[s]
        sel = np.where(keep)[0]
        co2 = np.round(meas[s]["co2"][sel]).astype(float)
        temp = np.round(meas[s]["temp"][sel], 1)
        hum = np.round(meas[s]["hum"][sel], 1)
        # 깨진 값: 0ppm, 65535ppm, -40℃, 60℃ 등 (약 0.1%)
        nb = max(3, int(len(sel) * 0.001))
        for j in rng.choice(len(sel), nb, replace=False):
            kind = rng.integers(0, 4)
            if kind == 0:
                co2[j] = 0
            elif kind == 1:
                co2[j] = 65535
            elif kind == 2:
                temp[j] = -40.0
            else:
                temp[j] = 60.0
            broken.append(dict(space=s, time=idx[sel[j]]))
        rows.append(pd.DataFrame(dict(time=idx[sel], space=s, device=cfg["device"],
                                      co2=co2, temperature=temp, humidity=hum)))
    sensors = pd.concat(rows).sort_values(["time", "space"]).reset_index(drop=True)

    # 정답 (1분 단위)
    truth_rows = []
    for s in SPACES:
        fault_col = np.full(n, "none", dtype=object)
        for f in faults:
            if f["space"] == s:
                fault_col[f["i0"]:f["i1"]] = f["type"]
        truth_rows.append(pd.DataFrame(dict(
            time=idx, space=s, occupants=occ[s].astype(int), occupied=(occ[s] > 0).astype(int),
            co2_true=np.round(true_vals[s]["co2"], 1), temp_true=np.round(true_vals[s]["temp"], 2),
            gap=gap[s].astype(int), fault=fault_col)))
    truth = pd.concat(truth_rows).sort_values(["time", "space"]).reset_index(drop=True)

    faults_df = pd.DataFrame([dict(space=f["space"], type=f["type"], start=idx[f["i0"]],
                                   end=idx[min(f["i1"], n - 1)], sensor=f["sensor"], note=f["note"] + (f" (+{f['amount']:.1f}℃)" if "amount" in f else ""))
                              for f in faults]).sort_values("start")

    sensors.to_csv(out / "sensors.csv", index=False)
    sched.to_csv(out / "schedule.csv", index=False)
    truth.to_csv(out / "truth.csv.gz", index=False)
    faults_df.to_csv(out / "faults.csv", index=False)
    meta = dict(
        seed=a.seed, start=str(idx[0]), end=str(idx[-1]), timezone="Asia/Seoul (KST, naive)",
        split=dict(train=[str(idx[0]), str(val_start)], val=[str(val_start), str(test_start)],
                   test=[str(test_start), str(idx[-1])]),
        spaces={s: {k: v for k, v in c.items()} for s, c in SPACES.items()},
        co2_outdoor=CO2_OUT, t_set=T_SET, open_hours=OPEN_HOURS,
        counts=dict(sensor_rows=len(sensors), broken_values=len(broken), faults=len(faults_df)),
    )
    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    print(json.dumps(meta["counts"], ensure_ascii=False))
    print(faults_df.to_string(index=False))


if __name__ == "__main__":
    main()
