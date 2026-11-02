"""과제 2 기준선: 현재값 유지 / 최근 1시간 평균 / 어제 같은 시각. 5·15·30·60분 뒤 CO₂·온도 MAE.

평가 지점(과제 3 이후 모든 모델이 같은 지점에서 평가한다 → report/eval_points.csv)
  - 5분 칸 시각 t, 검증·시험 기간
  - t의 값과 t+h의 값이 모두 '실제 관측'(보간·결측 아님)
  - 어제 같은 시각 값(t+h-24h)이 있다 (세 기준선 모두 같은 지점에서 비교하기 위해)
  - t-4시간 ~ t+h 사이에 정답 고장이 없다 (고장 구간은 과제 4에서 따로 본다)

  python baselines.py → report/baseline_*.csv, report/fig_baseline.png
"""
from pathlib import Path

import numpy as np
import pandas as pd

import prep
from viz import SPACE_KO, plt

OUT = Path(__file__).resolve().parent / "report"
HIST_BINS = 48  # 4시간: 과제 3에서 쓸 수 있는 가장 긴 입력 길이

grid, stats, removed, raw, sched, meta = prep.build()
fault = prep.fault_mask(grid)
METHODS = ["현재값 유지", "최근 1시간 평균", "어제 같은 시각"]


def per_space(space, g):
    g = g.droplevel("space")
    n = len(g)
    f = fault.loc[space].values
    # t-4h ~ t 사이에 고장이 있었나 (누적합으로 빠르게)
    cf = np.concatenate([[0], np.cumsum(f)])
    rows = []
    for v in ("co2", "temperature"):
        y = g[v].values
        obs = g[v].notna().values & ~g[v + "_imp"].values
        mean1h = pd.Series(y).rolling(12, min_periods=6).mean().values
        for h in prep.HORIZONS:
            t = np.arange(HIST_BINS, n - h)
            yest = t + h - 288
            ok = (obs[t] & obs[t + h] & (yest >= 0))
            ok &= np.where(yest >= 0, ~np.isnan(y[np.clip(yest, 0, None)]), False)
            ok &= (cf[t + h + 1] - cf[t - HIST_BINS]) == 0
            t = t[ok]
            pred = {"현재값 유지": y[t], "최근 1시간 평균": mean1h[t], "어제 같은 시각": y[t + h - 288]}
            rows.append(pd.DataFrame(dict(space=space, var=v, h=h * 5, time=g.index[t], split=g["split"].values[t],
                                          y_now=y[t], y_true=y[t + h], **{m: pred[m] for m in METHODS})))
    return rows


ev = pd.concat([r for sp, g in grid.groupby(level="space") for r in per_space(sp, g)])
ev = ev[ev.split.isin(["val", "test"])].dropna(subset=METHODS)
ev[["space", "var", "h", "time", "split"]].to_csv(OUT / "eval_points.csv", index=False)

hour = ev.time.dt.hour
ev["daytime"] = (ev.time.dt.weekday < 5) & (hour >= 9) & (hour < 18)


def mae_table(d, by):
    out = d.groupby(by).apply(lambda x: pd.Series({m: np.abs(x[m] - x.y_true).mean() for m in METHODS} | {"지점 수": len(x)}))
    return out


main = mae_table(ev[ev.split == "test"], ["var", "h"]).round(2)
day = mae_table(ev[(ev.split == "test") & ev.daytime], ["var", "h"]).round(2)
by_space = mae_table(ev[ev.split == "test"], ["var", "space", "h"]).round(2)
val = mae_table(ev[ev.split == "val"], ["var", "h"]).round(2)
for name, t in (("test", main), ("test_daytime", day), ("test_by_space", by_space), ("val", val)):
    t.to_csv(OUT / f"baseline_{name}.csv")

# 어제 같은 시각이 약한 이유: 월요일(어제=일요일)과 화~금 비교
ev["weekday"] = ev.time.dt.weekday
mon = ev[(ev.split == "test") & (ev["var"] == "co2") & (ev.h == 60) & ev.daytime]
yest_by_dow = mon.groupby(mon.weekday).apply(lambda x: pd.Series({m: np.abs(x[m] - x.y_true).mean() for m in METHODS})).round(1)
yest_by_dow.index = [["월", "화", "수", "목", "금", "토", "일"][i] for i in yest_by_dow.index]
yest_by_dow.to_csv(OUT / "baseline_co2_60_by_weekday.csv")

# 그래프: 예측 시점별 MAE
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for ax, v, unit in ((axes[0], "co2", "ppm"), (axes[1], "temperature", "℃")):
    t = main.loc[v]
    for m in METHODS:
        ax.plot(t.index, t[m], marker="o", label=m)
    ax.set_title(f"{'CO₂' if v == 'co2' else '온도'} 기준선 MAE (시험 기간, 고장 구간 제외)")
    ax.set_xlabel("몇 분 뒤")
    ax.set_ylabel(f"MAE ({unit})")
    ax.set_xticks([5, 15, 30, 60])
    ax.grid(alpha=0.3)
    ax.legend()
fig.tight_layout()
fig.savefig(OUT / "fig_baseline.png", dpi=130)

pd.set_option("display.width", 200)
print("== 시험 기간 전체 ==\n", main.to_string())
print("== 시험 기간 평일 9~18시 ==\n", day.to_string())
print("== 공간별 (시험) ==\n", by_space.to_string())
print("== 검증 기간 ==\n", val.to_string())
print("== CO2 60분, 평일 낮, 요일별 ==\n", yest_by_dow.to_string())
