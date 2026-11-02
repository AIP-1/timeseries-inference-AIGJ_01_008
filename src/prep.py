"""공통 전처리: 정제 → 5분 집계 → 짧은 결측 보간 → 분할 → 정규화.

학습(과제 3~6)과 상시 서비스(과제 8)가 이 모듈 하나만 쓴다. 학습과 서비스의 전처리가 다르면
모델은 학습 때 본 적 없는 분포를 입력받아 조용히 틀린다(training-serving skew).

정제 규칙 (근거는 report/diagnosis.md)
  R1 물리적으로 불가능한 값 → 결측. CO₂ 300~5000ppm, 온도 -10~50℃, 습도 0~100% 밖
  R2 5분 칸(t-5분, t] 평균으로 집계. 칸 이름은 끝 시각 t → 시각 t의 값은 t까지의 데이터만 쓴다(미래 누설 없음)
  R3 15분(3칸) 이하 결측은 시간 선형 보간, 더 긴 결측은 결측으로 둔다. 보간 여부는 *_imp 열로 남긴다
  R4 정규화 평균·표준편차는 학습 기간 값으로만 계산
  고장 의심 값(튄 값, 고정, 영점 틀어짐)은 여기서 지우지 않는다. 그건 과제 4 이상 탐지의 몫이다.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

import os

DATA = Path(os.getenv("TS_DATA", Path(__file__).resolve().parent.parent / "data" / "synthetic"))   # 실데이터: TS_DATA=~/work/data/real
VARS = ["co2", "temperature", "humidity"]
LIMITS = {"co2": (300, 5000), "temperature": (-10, 50), "humidity": (0, 100)}
FREQ = "5min"
MAX_GAP_BINS = 3          # 15분
HORIZONS = [1, 3, 6, 12]  # 5·15·30·60분 뒤 (5분 칸 수)


def load_raw(data=DATA):
    s = pd.read_csv(data / "sensors.csv", parse_dates=["time"])
    sched = pd.read_csv(data / "schedule.csv", parse_dates=["start", "end"])
    meta = json.loads((data / "meta.json").read_text())
    return s, sched, meta


def clean(raw):
    """R1. 불가능한 값을 결측으로 바꾸고, 바꾼 개수를 함께 돌려준다."""
    df = raw.copy()
    removed = {}
    for v, (lo, hi) in LIMITS.items():
        bad = df[v].notna() & ((df[v] < lo) | (df[v] > hi))
        removed[v] = df.loc[bad].groupby("space").size()
        df.loc[bad, v] = np.nan
    return df, pd.DataFrame(removed).fillna(0).astype(int)


def to_grid(df, start=None, end=None):
    """R2+R3. 공간별 5분 칸 표. index=(space, time), 열=VARS + VARS_imp(보간 여부) + VARS_n(칸 안 원본 건수)."""
    out = []
    for space, g in df.groupby("space"):
        lo = start or g.time.min()
        hi = end or g.time.max()
        full = pd.date_range(pd.Timestamp(lo).ceil(FREQ), pd.Timestamp(hi).floor(FREQ), freq=FREQ)
        cnt = (g.set_index("time")[VARS].sort_index().resample(FREQ, label="right", closed="right").count()
               .add_suffix("_n").reindex(full).fillna(0).astype(int))
        if "device" in g and g.device.nunique() > 1:
            # R2' 센서가 여러 대인 공간: 센서별로 5분 평균을 내고, 각 센서의 마지막 값을 15분까지 이어 쓴 뒤 센서 평균.
            # 칸마다 값을 보낸 센서 조합이 바뀌면(센서끼리 보정값이 달라) 평균이 출렁이는 것을 막는다. 이어 쓰기는 과거 값만 쓴다.
            per = []
            for _, gd in g.groupby("device"):
                r = gd.set_index("time")[VARS].sort_index().resample(FREQ, label="right", closed="right").mean().reindex(full)
                per.append(r)
            actual = pd.concat(per).groupby(level=0).count()
            agg = pd.concat([r.ffill(limit=MAX_GAP_BINS) for r in per]).groupby(level=0).mean().reindex(full)
            agg = agg.where(actual.reindex(full).fillna(0) > 0)   # 실제 값이 하나도 없는 칸은 결측으로(보간 규칙에 맡긴다)
        else:
            agg = g.set_index("time")[VARS].sort_index().resample(FREQ, label="right", closed="right").mean().reindex(full)
        filled = agg.copy()
        for v in VARS:
            filled[v] = interpolate_short(agg[v])
            filled[v + "_imp"] = agg[v].isna() & filled[v].notna()
        filled = pd.concat([filled, cnt], axis=1)
        filled.index.name = "time"
        filled.insert(0, "space", space)
        out.append(filled.reset_index())
    return pd.concat(out).set_index(["space", "time"]).sort_index()


def interpolate_short(s, max_bins=MAX_GAP_BINS):
    """max_bins 이하 길이의 결측만 선형 보간한다(긴 결측의 앞부분만 메우는 일을 막는다)."""
    na = s.isna()
    run_id = (na != na.shift()).cumsum()
    run_len = na.groupby(run_id).transform("sum")
    inner = s.interpolate(method="time", limit_area="inside")
    return s.where(~(na & (run_len <= max_bins)), inner)


def split_of(times, meta):
    t = pd.DatetimeIndex(times)
    sp = meta["split"]
    lab = np.full(len(t), "", dtype=object)
    lab[(t >= sp["train"][0]) & (t < sp["train"][1])] = "train"
    lab[(t >= sp["val"][0]) & (t < sp["val"][1])] = "val"
    lab[(t >= sp["test"][0])] = "test"
    return lab


def norm_stats(grid, meta):
    """R4. 학습 기간·관측값(보간 제외)으로만 평균·표준편차."""
    g = grid.reset_index()
    tr = g[split_of(g.time, meta) == "train"]
    return {v: dict(mean=float(tr.loc[~tr[v + "_imp"], v].mean()), std=float(tr.loc[~tr[v + "_imp"], v].std()))
            for v in VARS}


def schedule_on_grid(sched, grid):
    """일정 → 5분 칸별 '예정 인원'. 미래 값도 미리 알 수 있는 정보(과제 3 입력용)."""
    g = grid.reset_index()[["space", "time"]].copy()
    g["sched_people"] = 0.0
    for space, s in sched.groupby("space"):
        m = g.space == space
        times = g.loc[m, "time"].values
        val = np.zeros(m.sum())
        for _, r in s.iterrows():
            val = np.where((times > np.datetime64(r.start)) & (times <= np.datetime64(r.end)),
                           np.maximum(val, r.expected), val)
        g.loc[m, "sched_people"] = val
    return g.set_index(["space", "time"])["sched_people"]


def build(data=DATA):
    """전체 파이프라인. (5분 표, 정규화 값, 제거 건수, 원본, 일정, meta)"""
    raw, sched, meta = load_raw(data)
    cleaned, removed = clean(raw)
    grid = to_grid(cleaned, meta["start"], meta["end"])
    grid["split"] = split_of(grid.index.get_level_values("time"), meta)
    grid["sched_people"] = schedule_on_grid(sched, grid)
    return grid, norm_stats(grid, meta), removed, raw, sched, meta


def fault_mask(grid, data=DATA, pad_min=0):
    """평가에서 뺄 고장 구간(정답 faults.csv). 과제 2·3 평가는 고장 없는 구간에서 한다."""
    f = pd.read_csv(data / "faults.csv", parse_dates=["start", "end"])
    idx = grid.index
    mask = np.zeros(len(idx), bool)
    sp = idx.get_level_values("space")
    t = idx.get_level_values("time")
    for _, r in f.iterrows():
        mask |= (sp == r.space) & (t > r.start - pd.Timedelta(minutes=pad_min)) & (t <= r.end + pd.Timedelta(minutes=5 + pad_min))
    return pd.Series(mask, index=idx)
