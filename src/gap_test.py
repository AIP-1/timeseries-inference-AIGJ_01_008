"""정제 규칙 R3(15분 이하 결측만 보간)의 근거 실험.

  python gap_test.py            # 합성
  TS_DATA=~/work/data/real python gap_test.py   # 실데이터

실제 값이 있는 구간에서 일부러 k칸(5분 × k)을 지우고, 앞뒤 값을 직선으로 이어 메운 뒤 실제 값과 비교한다.
학습 기간만 쓴다(규칙을 정하는 데 시험 기간을 쓰지 않는다). 보간된 칸은 실제 값이 아니므로 쓰지 않는다.
함께 실제 결측 구간의 길이 분포를 세어, 기준을 몇 분으로 하면 결측 칸을 얼마나 살리는지 본다.
결과: report/gap_test_<데이터>.csv
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd

import prep

OUT = Path(__file__).resolve().parent / "report"
TAG = Path(prep.DATA).name
KS = [1, 2, 3, 4, 6, 9, 12, 24]       # 지우는 칸 수: 5·10·15·20·30·45·60·120분
VARS = ["co2", "temperature"]

grid, *_ = prep.build()
g = grid[grid.split == "train"]

rows = []
for space, gs in g.groupby(level="space"):
    gs = gs.droplevel("space")
    hour = gs.index.hour
    day = (gs.index.dayofweek < 5) & (hour >= 8) & (hour < 19)   # 평일 낮(사람이 드나드는 시간)
    for v in VARS:
        x = gs[v].where(~gs[v + "_imp"]).to_numpy(float)    # 실제 관측값만
        ok = ~np.isnan(x)
        for k in KS:
            n = len(x) - k - 1
            if n <= 0:
                continue
            # 시작 i: i-1(앞 값)과 i+k(뒤 값), 그 사이 k칸이 모두 실제 값인 자리
            win = np.lib.stride_tricks.sliding_window_view(ok, k + 2).all(axis=1)
            starts = np.flatnonzero(win) + 1
            if len(starts) == 0:
                continue
            a, b = x[starts - 1], x[starts + k]
            w = (np.arange(1, k + 1) / (k + 1))[None, :]
            est = a[:, None] + (b - a)[:, None] * w
            true = x[starts[:, None] + np.arange(k)[None, :]]
            err = np.abs(est - true).mean(axis=1)            # 구간 하나의 평균 오차
            dmask = day[starts]
            for part, m in (("전체", slice(None)), ("평일 낮", dmask)):
                e = err[m]
                if len(e) == 0:
                    continue
                rows.append(dict(data=TAG, space=space, var=v, part=part, gap_min=5 * k, n=len(e),
                                 MAE=round(float(e.mean()), 2), p95=round(float(np.percentile(e, 95)), 2)))

res = pd.DataFrame(rows)

# 실제 결측 구간 길이 분포(보간 전 기준, 학습 기간, CO₂)
gaps = []
for space, gs in g.groupby(level="space"):
    na = (gs["co2"].isna() | gs["co2_imp"]).to_numpy()
    run = (na != np.roll(na, 1)).cumsum()
    lens = pd.Series(na).groupby(run).agg(["sum", "first"])
    lens = lens[lens["first"]]["sum"]
    for L in lens:
        gaps.append(dict(space=space, bins=int(L)))
gp = pd.DataFrame(gaps)

res.to_csv(OUT / f"gap_test_{TAG}.csv", index=False)
gp.to_csv(OUT / f"gap_lengths_{TAG}.csv", index=False)

pd.set_option("display.width", 200)
print(f"== {TAG}: 직선으로 메웠을 때 평균 오차 (학습 기간, 공간 평균)")
for v in VARS:
    t = res[res["var"] == v].groupby(["part", "gap_min"]).apply(
        lambda d: pd.Series(dict(MAE=np.average(d.MAE, weights=d.n), p95=d.p95.mean(), n=d.n.sum())), include_groups=False)
    print(f"-- {v}")
    print(t.unstack("part").round(2).to_string())
print("-- 공간별 CO₂ 오차 (전체)")
print(res[(res["var"] == "co2") & (res.part == "전체")].pivot(index="gap_min", columns="space", values="MAE").to_string())
if len(gp):
    tot = gp.bins.sum()
    print(f"-- 실제 결측 구간: {len(gp)}개, 결측 칸 {tot}개")
    for k in (1, 2, 3, 6, 12):
        sel = gp[gp.bins <= k]
        print(f"   {5 * k:>3}분 이하: 구간 {len(sel) / len(gp):.0%}, 칸 {sel.bins.sum() / tot:.0%}")
    print("   공간별 최장:", gp.groupby("space").bins.max().mul(5).to_dict(), "분")
