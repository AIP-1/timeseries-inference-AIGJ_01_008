"""심화 6-2: 직접 학습한 GRU(과거 4시간)와 사전학습 Chronos-Bolt(과거 1주)를 섞은 예측이 둘 다보다 나은지.

  python b6_ensemble.py      (CPU)

  - GRU 검증·시험 예측: 저장된 models/forecast_gru_L48.pt로 다시 만든다(train_forecast.py와 같은 입력 규칙)
  - Chronos 시험 예측: report/pretrained/chronos-bolt-<m>_cuda_pred_168h.csv.gz (과제 6 결과)
  - Chronos 검증 예측: report/b6_bolt_<m>_val_168h.csv.gz (b6_bolt_val.py)
  섞는 법: ① 단순 평균(0.5:0.5) ② 검증 60분 MAE가 가장 작은 GRU 비중 w(0~1, 0.05 간격, 모든 예측 거리에 같은 w)
  ③ 예측 거리마다 따로 고른 w. 비중은 검증으로만 고르고 시험은 평가만 한다.
  모든 방법을 '둘 다 예측이 있는 같은 지점'에서 잰다.
결과: report/b6_ensemble_results.csv, b6_ensemble_weights.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import dataset
import models
import prep

ROOT = Path(__file__).resolve().parent
OUT, MD = ROOT / "report", ROOT / "models"
torch.set_num_threads(4)
HS = (5, 15, 30, 60)

grid, stats, *_ = prep.build()
fault = prep.fault_mask(grid)
EVAL = pd.read_csv(OUT / "eval_points.csv", parse_dates=["time"])
STD = np.array([stats[v]["std"] for v in dataset.TARGETS], np.float32)

ck = torch.load(MD / "forecast_gru_L48.pt", weights_only=False)
gru = models.make(ck["kind"], ck["n_in"])
gru.load_state_dict(ck["state"])
gru.eval()
fe = dataset.features(grid, stats, use_sched=ck["use_sched"], use_space=ck["use_space"])


def gru_pred(split):
    X, F, _, _, K = dataset.samples(fe, stats, ck["L"], split, fault, min_hist_obs=0.0)
    with torch.no_grad():
        P = np.concatenate([gru(torch.from_numpy(X[i:i + 4096]), torch.from_numpy(F[i:i + 4096])).numpy() for i in range(0, len(X), 4096)])
    rows = [pd.DataFrame(dict(space=K.space, time=K.time, h=(hi + 1) * 5, gru=K.y_now_co2.values + P[:, hi, 0] * STD[0])) for hi in range(12)]
    ev = EVAL[(EVAL.split == split) & (EVAL["var"] == "co2")]
    return ev.merge(pd.concat(rows), on=["space", "time", "h"], how="left")


def table(split, chronos_file):
    g = gru_pred(split)
    c = pd.read_csv(chronos_file, parse_dates=["time"])[["space", "time", "h", "q50"]].rename(columns=dict(q50="chronos"))
    m = g.merge(c, on=["space", "time", "h"], how="inner")
    ix = pd.MultiIndex.from_arrays([m.space, m.time + pd.to_timedelta(m.h, "min")])
    m["true"] = grid["co2"].reindex(ix).values
    return m.dropna(subset=["gru", "chronos", "true"])


def mae(m, w):
    """w: GRU 비중(스칼라 또는 h→w dict)."""
    ww = m.h.map(w) if isinstance(w, dict) else w
    return ((ww * m.gru + (1 - ww) * m.chronos) - m.true).abs().groupby(m.h).mean()


rows, wrows = [], []
grid_w = np.round(np.arange(0, 1.0001, 0.05), 2)
gru_val_check = None
for cm in ("small", "base"):
    va = table("val", OUT / f"b6_bolt_{cm}_val_168h.csv.gz")
    te = table("test", OUT / "pretrained" / f"chronos-bolt-{cm}_cuda_pred_168h.csv.gz")
    va_curve = pd.DataFrame({w: mae(va, w) for w in grid_w})       # index h, 열 w
    w_all = float(va_curve.loc[60].idxmin())                          # 60분 기준 하나의 w
    w_h = {int(h): float(va_curve.loc[h].idxmin()) for h in va_curve.index}
    for h in HS:
        wrows.append(dict(chronos=f"bolt-{cm}", h=h, 검증최적_w_GRU=w_h[h], 검증MAE_최적=round(va_curve.loc[h].min(), 2),
                          검증MAE_GRU=round(va_curve.loc[h, 1.0], 2), 검증MAE_Chronos=round(va_curve.loc[h, 0.0], 2),
                          검증MAE_평균=round(va_curve.loc[h, 0.5], 2)))
    methods = {"GRU 단독": 1.0, f"Chronos-Bolt-{cm} 단독": 0.0, "단순 평균(0.5:0.5)": 0.5,
               f"검증 비중(GRU {w_all:.2f}, 60분 기준)": w_all, "검증 비중(예측 거리별)": w_h}
    for name, w in methods.items():
        for split, m in (("val", va), ("test", te)):
            r = mae(m, w)
            rows.append(dict(조합=f"GRU+bolt-{cm}", 방법=name, split=split, **{f"MAE_{h}분": round(r[h], 2) for h in HS},
                             지점수_60분=int((m.h == 60).sum())))
    print(pd.DataFrame(rows[-10:]).to_string(index=False), flush=True)

res = pd.DataFrame(rows)
res.to_csv(OUT / "b6_ensemble_results.csv", index=False)
pd.DataFrame(wrows).to_csv(OUT / "b6_ensemble_weights.csv", index=False)
pd.set_option("display.width", 200)
print(res[res.split == "test"].to_string(index=False))
print(pd.DataFrame(wrows).to_string(index=False))
