"""과제 6: 사전학습 시계열 모델(Chronos-Bolt) 학습 없이 CO₂ 60분 예측.

  DEV=cuda python pretrained.py [모델=amazon/chronos-bolt-small]

같은 평가 지점(report/eval_points.csv)에서 기준선·직접 학습 모델과 비교한다.
과거 길이(1·4·24·96·168시간)마다 검증·시험 기간 정확도를 재고, 과거 길이는 검증 MAE로 고른다(시험은 고르는 데 쓰지 않는다).
10~90% 예측 구간 적중률, 1회 추론 시간(배치 1·4)과 메모리를 잰다.
결과: report/pretrained/<모델>_<장치>_results.csv, _meta.json
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
import torch

import prep

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "report"
RES = OUT / "pretrained"
RES.mkdir(exist_ok=True)
MODEL = sys.argv[1] if len(sys.argv) > 1 else "amazon/chronos-bolt-small"
torch.set_num_threads(int(os.getenv("THREADS", "4")))
DEV = os.getenv("DEV", "cpu")
BATCH = int(os.getenv("BATCH", "256"))
CTX_HOURS = (1, 4, 24, 96, 168)
TAG = f"{MODEL.split('/')[-1]}_{DEV}"
GPU = DEV.startswith("cuda")

grid, stats, *_ = prep.build()
EV_ALL = pd.read_csv(OUT / "eval_points.csv", parse_dates=["time"])
EV_ALL = EV_ALL[EV_ALL["var"] == "co2"]

proc = psutil.Process()
avail0 = psutil.virtual_memory().available
rss0 = proc.memory_info().rss
from chronos import BaseChronosPipeline  # noqa: E402

t0 = time.time()
pipe = BaseChronosPipeline.from_pretrained(MODEL, device_map=DEV, torch_dtype=torch.float32)
load_sec = time.time() - t0
rss_model = proc.memory_info().rss - rss0
gpu_model = torch.cuda.memory_allocated() if GPU else 0
n_params = sum(p.numel() for p in pipe.model.parameters())


def series(space):
    """모델에 줄 과거 값: 5분 CO₂(정제·짧은 결측 보간). 긴 결측은 앞 값으로 채운다(모델은 NaN을 받지만 앞쪽 결측만 허용)."""
    s = grid.loc[space]["co2"]
    return s.ffill().bfill()


def quantiles(ctx):
    q_all = []
    for i in range(0, len(ctx), BATCH):
        q, _ = pipe.predict_quantiles(ctx[i:i + BATCH], prediction_length=12, quantile_levels=[0.1, 0.5, 0.9])
        q_all.append(q.cpu().numpy())
    return np.concatenate(q_all)  # [N, 12, 3]


def evaluate(split, ctx_h):
    EV = EV_ALL[EV_ALL.split == split]
    pts = EV[["space", "time"]].drop_duplicates()
    L = ctx_h * 12
    preds = []
    for space, p in pts.groupby("space"):
        s = series(space)
        pos = s.index.get_indexer(p.time)
        ctx = torch.tensor(np.stack([s.values[i - L + 1:i + 1] if i - L + 1 >= 0
                                     else np.pad(s.values[:i + 1], (L - i - 1, 0), constant_values=np.nan) for i in pos]), dtype=torch.float32)
        q = quantiles(ctx)
        for hi in range(12):
            preds.append(pd.DataFrame(dict(space=space, time=p.time.values, h=(hi + 1) * 5, q10=q[:, hi, 0], q50=q[:, hi, 1], q90=q[:, hi, 2])))
    m = EV.merge(pd.concat(preds), on=["space", "time", "h"])
    idx = pd.MultiIndex.from_arrays([m.space, m.time + pd.to_timedelta(m.h, "min")])
    m["true"] = grid["co2"].reindex(idx).values
    m["err"] = (m.q50 - m.true).abs()
    m["inside"] = (m.true >= m.q10) & (m.true <= m.q90)
    return m


rows, eval_sec = [], {}
for ctx_h in CTX_HOURS:
    for split in ("val", "test"):
        t = time.time()
        m = evaluate(split, ctx_h)
        eval_sec[f"{split}_{ctx_h}h"] = round(time.time() - t, 1)
        for h in (5, 15, 30, 60):
            mm = m[m.h == h]
            rows.append(dict(model=MODEL, device=DEV, split=split, context_h=ctx_h, h=h, MAE=round(mm.err.mean(), 2),
                             구간적중률_10_90=round(mm.inside.mean(), 3), 구간폭_중앙값=round((mm.q90 - mm.q10).median(), 1), 지점수=len(mm)))
        print(rows[-1], flush=True)
        if split == "test":
            m.to_csv(RES / f"{TAG}_pred_{ctx_h}h.csv.gz", index=False)

res = pd.DataFrame(rows)
v60 = res[(res.split == "val") & (res.h == 60)]
best_ctx = int(v60.loc[v60.MAE.idxmin(), "context_h"])

# 1회 추론 시간: 배치 1(한 공간)과 배치 4(네 공간). 과거 24시간과 검증으로 고른 과거 길이 두 가지.
timing = {}
for ctx_h in sorted({24, best_ctx}):
    ctx1 = torch.tensor(series("classroom_a").values[-ctx_h * 12:], dtype=torch.float32)[None]
    for name, c in ((f"배치1_{ctx_h}h", ctx1), (f"배치4_{ctx_h}h", ctx1.repeat(4, 1))):
        for _ in range(20):
            pipe.predict_quantiles(c, prediction_length=12, quantile_levels=[0.1, 0.5, 0.9])
        ts = []
        for _ in range(100):
            if GPU:
                torch.cuda.synchronize()
            t = time.perf_counter()
            q, _ = pipe.predict_quantiles(c, prediction_length=12, quantile_levels=[0.1, 0.5, 0.9])
            q.cpu()
            if GPU:
                torch.cuda.synchronize()
            ts.append((time.perf_counter() - t) * 1000)
        timing[name] = dict(mean_ms=round(float(np.mean(ts)), 2), p95_ms=round(float(np.percentile(ts, 95)), 2))

res.to_csv(RES / f"{TAG}_results.csv", index=False)
info = dict(model=MODEL, device=DEV, threads=torch.get_num_threads(), batch=BATCH, params=n_params, load_sec=round(load_sec, 1),
            best_context_h_by_val=best_ctx,
            rss_model_mb=round(rss_model / 2**20, 1), rss_total_mb=round(proc.memory_info().rss / 2**20, 1),
            gpu_model_mb=round(gpu_model / 2**20, 1),
            gpu_peak_mb=round(torch.cuda.max_memory_allocated() / 2**20, 1) if GPU else 0,
            board_avail_drop_mb=round((avail0 - psutil.virtual_memory().available) / 2**20, 1),
            eval_sec=eval_sec, timing=timing)
(RES / f"{TAG}_meta.json").write_text(json.dumps(info, ensure_ascii=False, indent=1))
print(res[res.h == 60].to_string(index=False))
print(info)
