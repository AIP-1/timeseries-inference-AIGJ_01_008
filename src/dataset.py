"""모델 입력 만들기(과제 3·4·5·7·8 공통). prep.build()의 5분 표 → 공간별 특징 배열 → 샘플.

시각 t의 샘플
  과거 입력  x_hist [L, F]   t-L+1 … t (정규화 값, 결측 표시, 시간·요일, 일정, 공간)
  미래 입력  x_fut  [12, 3]  t+1 … t+12 에 미리 알 수 있는 것(예정 인원, 시각 sin/cos)
  정답       y      [12, 2]  (t+h 값 - t 값) / 표준편차, CO₂·온도. '변화량'을 맞히게 한다
  정답 마스크 m     [12, 2]  t+h 값이 실제 관측일 때만 1
"""
import numpy as np
import pandas as pd

import prep

SPACES = ["classroom_a", "classroom_b", "meeting", "lounge"]
TARGETS = ["co2", "temperature"]
H = 12
CAP = 30.0  # 예정 인원 정규화


def features(grid, stats, use_sched=True, use_space=True):
    """공간별 dict: feat [N, F], fut [N, 3], y_raw [N, 2], obs [N, 2], times, split."""
    out = {}
    for space in grid.index.get_level_values("space").unique():
        g = grid.loc[space]
        t = g.index
        cols = []
        for v in prep.VARS:
            z = (g[v].values - stats[v]["mean"]) / stats[v]["std"]
            miss = np.isnan(z) | g[v + "_imp"].values.astype(bool)
            cols += [np.nan_to_num(z), miss.astype(float)]
        hour = (t.hour + t.minute / 60).values
        dow = t.weekday.values
        cols += [np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24),
                 np.sin(2 * np.pi * dow / 7), np.cos(2 * np.pi * dow / 7), (dow >= 5).astype(float)]
        sched = g["sched_people"].values / CAP if use_sched else np.zeros(len(g))
        cols.append(sched)
        onehot = np.zeros((len(g), len(SPACES)))
        if use_space and space in SPACES:
            onehot[:, SPACES.index(space)] = 1
        feat = np.column_stack(cols + [onehot]).astype(np.float32)
        fut = np.column_stack([sched, np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24)]).astype(np.float32)
        y_raw = g[TARGETS].values.astype(np.float32)
        obs = np.column_stack([g[v].notna().values & ~g[v + "_imp"].values.astype(bool) for v in TARGETS])
        out[space] = dict(feat=feat, fut=fut, y_raw=y_raw, obs=obs, times=t, split=g["split"].values)
    return out


def samples(fe, stats, L, split, fault=None, spaces=None, min_hist_obs=0.8):
    """조건에 맞는 (공간, t) 샘플 인덱스 목록과 배열."""
    xs, fs, ys, ms, keys = [], [], [], [], []
    std = np.array([stats[v]["std"] for v in TARGETS], np.float32)
    for space, d in fe.items():
        if spaces is not None and space not in spaces:
            continue
        n = len(d["feat"])
        flt = fault.loc[space].values if fault is not None else np.zeros(n, bool)
        cf = np.concatenate([[0], np.cumsum(flt)])
        obs_any = d["obs"].all(1).astype(float)
        co = np.concatenate([[0], np.cumsum(obs_any)])
        t = np.arange(L - 1, n - 1)
        ok = np.isin(d["split"][t], split if isinstance(split, (list, tuple)) else [split])
        ok &= d["obs"][t].all(1)
        ok &= (co[t + 1] - co[t + 1 - L]) / L >= min_hist_obs
        t_end = np.minimum(t + H, n - 1)
        ok &= (cf[t_end + 1] - cf[t + 1 - L]) == 0
        t = t[ok]
        idx_h = np.arange(L)[None, :] + (t - L + 1)[:, None]
        idx_f = np.clip(t[:, None] + np.arange(1, H + 1)[None, :], 0, n - 1)
        valid_f = (t[:, None] + np.arange(1, H + 1)[None, :]) < n
        y = (d["y_raw"][idx_f] - d["y_raw"][t][:, None, :]) / std
        m = d["obs"][idx_f] & valid_f[..., None]
        xs.append(d["feat"][idx_h])
        fs.append(d["fut"][idx_f])
        ys.append(np.nan_to_num(y))
        ms.append(m.astype(np.float32))
        keys.append(pd.DataFrame(dict(space=space, time=d["times"][t], y_now_co2=d["y_raw"][t, 0], y_now_temp=d["y_raw"][t, 1])))
    return (np.concatenate(xs), np.concatenate(fs), np.concatenate(ys).astype(np.float32),
            np.concatenate(ms), pd.concat(keys, ignore_index=True))


def sensor_features(g, space, use_space=True):
    """센서값만으로 만든 특징(일정 없음). g: 한 공간의 5분 표."""
    c, t, h = g["co2"], g["temperature"], g["humidity"]
    f = pd.DataFrame(index=g.index)
    f["co2"] = c
    f["co2_excess"] = c - 420
    for k in (1, 3, 6):
        f[f"dco2_{k * 5}"] = c - c.shift(k)
        f[f"dtemp_{k * 5}"] = t - t.shift(k)
    f["dhum_15"] = h - h.shift(3)
    # 비어 있으면 CO₂는 외기 쪽으로 지수적으로 줄어든다: 15분 변화 / 외기 대비 초과분 = 감소 비율
    f["decay_ratio_15"] = f["dco2_15"] / f["co2_excess"].clip(lower=20)
    f["co2_max_60"] = c.rolling(12, min_periods=6).max()
    f["temp"] = t
    hour = g.index.hour + g.index.minute / 60
    f["hour_sin"], f["hour_cos"] = np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24)
    f["weekend"] = (g.index.weekday >= 5).astype(float)
    if use_space:
        for s in SPACES:
            f[f"sp_{s}"] = float(space == s)
    return f
