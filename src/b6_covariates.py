"""심화 6-1: 공변량을 받는 사전학습 모델(Chronos-2)에 온습도·일정 정보를 함께 넣어 CO₂만 넣었을 때와 비교.

  DEV=cuda python b6_covariates.py

Chronos-Bolt는 목표 계열 하나만 받는다. Chronos-2(amazon/chronos-2, chronos-forecasting 2.x)는
과거 공변량(과거 값만 아는 것)과 미래 공변량(미래 값도 미리 아는 것)을 함께 받는다. 학습 없이(zero-shot) 쓴다.
  A  CO₂만
  B  + 과거 온도·습도
  C  + 과거 온도·습도 + 예정 인원(sched_people)·시각 sin/cos (이 셋은 미래 12칸 값도 준다)
과거 길이(24·168시간)는 검증 기간 60분 MAE로 고르고, 고른 길이로만 시험 기간을 잰다.
결과: report/b6_covariates_results.csv, b6_covariates_meta.json, b6_chronos2_<변형>_<split>_<ctx>h.csv.gz
"""
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
import torch

import prep

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "report"
DEV = os.getenv("DEV", "cuda")
CTX_HOURS = [int(c) for c in os.getenv("CTX", "24,168").split(",")]
CHUNK = 64           # 한 번에 만드는 지점 수(파이썬 메모리 제한)
SERIES_BATCH = int(os.getenv("SERIES_BATCH", "32"))   # 모델에 한 번에 넣는 계열 수(목표+공변량 합)
VARIANTS = {"A_co2만": ([], []),
            "B_온습도": (["temperature", "humidity"], []),
            "C_온습도_일정시각": (["temperature", "humidity", "sched", "hsin", "hcos"], ["sched", "hsin", "hcos"])}
ONLY_VARIANTS = os.getenv("VARIANTS")
if ONLY_VARIANTS:
    VARIANTS = {k: v for k, v in VARIANTS.items() if k in ONLY_VARIANTS.split(",")}

grid, stats, *_ = prep.build()
EV_ALL = pd.read_csv(OUT / "eval_points.csv", parse_dates=["time"])
EV_ALL = EV_ALL[EV_ALL["var"] == "co2"]

proc = psutil.Process()
from chronos import Chronos2Pipeline  # noqa: E402

t0 = time.time()
pipe = Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map=DEV, torch_dtype=torch.float32)
load_sec = time.time() - t0
gpu_model = torch.cuda.memory_allocated() if DEV == "cuda" else 0
n_params = sum(p.numel() for p in pipe.model.parameters())


def space_arrays(space):
    g = grid.loc[space]
    hour = (g.index.hour + g.index.minute / 60).values
    return dict(co2=g["co2"].ffill().bfill().values, temperature=g["temperature"].ffill().bfill().values,
                humidity=g["humidity"].ffill().bfill().values, sched=g["sched_people"].fillna(0).values.astype(float),
                hsin=np.sin(2 * np.pi * hour / 24), hcos=np.cos(2 * np.pi * hour / 24)), g.index


def window(a, i, L):
    if i - L + 1 >= 0:
        return a[i - L + 1:i + 1]
    return np.concatenate([np.full(L - i - 1, np.nan), a[:i + 1]])


def evaluate(split, ctx_h, past, fut):
    EV = EV_ALL[EV_ALL.split == split]
    L = ctx_h * 12
    preds = []
    for space, p in EV[["space", "time"]].drop_duplicates().groupby("space"):
        arr, idx = space_arrays(space)
        n = len(idx)
        pos = idx.get_indexer(p.time)
        qs = []
        for c in range(0, len(pos), CHUNK):
            inputs = []
            for i in pos[c:c + CHUNK]:
                d = dict(target=window(arr["co2"], i, L))
                if past:
                    d["past_covariates"] = {k: window(arr[k], i, L) for k in past}
                if fut:
                    j = np.clip(np.arange(i + 1, i + 13), 0, n - 1)
                    d["future_covariates"] = {k: arr[k][j] for k in fut}
                inputs.append(d)
            q, _ = pipe.predict_quantiles(inputs, prediction_length=12, quantile_levels=[0.1, 0.5, 0.9], batch_size=SERIES_BATCH)
            qs.append(np.stack([x[0].float().cpu().numpy() for x in q]))   # [N, 12, 3]
        q = np.concatenate(qs)
        for hi in range(12):
            preds.append(pd.DataFrame(dict(space=space, time=p.time.values, h=(hi + 1) * 5, q10=q[:, hi, 0], q50=q[:, hi, 1], q90=q[:, hi, 2])))
    m = EV.merge(pd.concat(preds), on=["space", "time", "h"])
    ix = pd.MultiIndex.from_arrays([m.space, m.time + pd.to_timedelta(m.h, "min")])
    m["true"] = grid["co2"].reindex(ix).values
    m["err"] = (m.q50 - m.true).abs()
    m["inside"] = (m.true >= m.q10) & (m.true <= m.q90)
    return m


def summarize(m, variant, split, ctx_h, sec):
    out = []
    for h in (5, 15, 30, 60):
        mm = m[m.h == h]
        out.append(dict(variant=variant, split=split, context_h=ctx_h, h=h, MAE=round(mm.err.mean(), 2),
                        구간적중률_10_90=round(mm.inside.mean(), 3), 지점수=int(mm.err.notna().sum()), 실행초=sec))
    return out


rows = []
for variant, (past, fut) in VARIANTS.items():
    val60 = {}
    for ctx_h in CTX_HOURS:
        t = time.time()
        m = evaluate("val", ctx_h, past, fut)
        sec = round(time.time() - t, 1)
        r = summarize(m, variant, "val", ctx_h, sec)
        rows += r
        val60[ctx_h] = r[-1]["MAE"]
        print(r[-1], flush=True)
        pd.DataFrame(rows).to_csv(OUT / "b6_covariates_results.csv", index=False)
    best = min(val60, key=val60.get)
    t = time.time()
    m = evaluate("test", best, past, fut)
    sec = round(time.time() - t, 1)
    rows += summarize(m, variant, "test", best, sec)
    print(rows[-1], flush=True)
    m.to_csv(OUT / f"b6_chronos2_{variant}_test_{best}h.csv.gz", index=False)
    pd.DataFrame(rows).to_csv(OUT / "b6_covariates_results.csv", index=False)

# 1회 추론 시간(배치 1 = 한 공간, 과거 24시간), 변형별
timing = {}
arr, idx = space_arrays("classroom_a")
i = len(idx) - 13
for variant, (past, fut) in VARIANTS.items():
    d = dict(target=window(arr["co2"], i, 288))
    if past:
        d["past_covariates"] = {k: window(arr[k], i, 288) for k in past}
    if fut:
        d["future_covariates"] = {k: arr[k][i + 1:i + 13] for k in fut}
    for _ in range(10):
        pipe.predict_quantiles([d], prediction_length=12, quantile_levels=[0.5])
    ts = []
    for _ in range(50):
        if DEV == "cuda":
            torch.cuda.synchronize()
        s = time.perf_counter()
        pipe.predict_quantiles([d], prediction_length=12, quantile_levels=[0.5])
        if DEV == "cuda":
            torch.cuda.synchronize()
        ts.append((time.perf_counter() - s) * 1000)
    timing[variant] = dict(mean_ms=round(float(np.mean(ts)), 1), p95_ms=round(float(np.percentile(ts, 95)), 1))

meta = dict(model="amazon/chronos-2", device=DEV, params=n_params, load_sec=round(load_sec, 1), contexts_h=CTX_HOURS,
            gpu_model_mb=round(gpu_model / 2**20, 1),
            gpu_peak_mb=round(torch.cuda.max_memory_allocated() / 2**20, 1) if DEV == "cuda" else 0,
            rss_mb=round(proc.memory_info().rss / 2**20, 1), timing_batch1_24h=timing,
            chronos_version=__import__("chronos").__version__ if hasattr(__import__("chronos"), "__version__") else None)
(OUT / "b6_covariates_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1))
print(pd.DataFrame(rows).to_string(index=False))
print(meta)
