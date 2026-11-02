"""심화 6-2 준비: Chronos-Bolt small·base의 '검증 기간' 예측(과거 1주)을 지점별로 저장한다.

  DEV=cuda BATCH=16 python b6_bolt_val.py small   (모델 하나씩, 프로세스를 나눠 실행)

pretrained.py는 시험 기간 예측만 저장했다. 섞는 비율을 검증으로 고르려면 검증 예측이 필요하다.
같은 입력 규칙(pretrained.py의 series·evaluate)을 그대로 쓴다.
결과: report/b6_bolt_<모델>_val_168h.csv.gz
"""
import gc
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import prep

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "report"
DEV = os.getenv("DEV", "cuda")
BATCH = int(os.getenv("BATCH", "16"))   # 15:26 보드 정지 뒤 128 → 16으로 줄임
CTX_H = 168

grid, stats, *_ = prep.build()
EV = pd.read_csv(OUT / "eval_points.csv", parse_dates=["time"])
EV = EV[(EV["var"] == "co2") & (EV.split == "val")]
from chronos import BaseChronosPipeline  # noqa: E402


def series(space):
    return grid.loc[space]["co2"].ffill().bfill()


for name in sys.argv[1:] or ["small"]:
    pipe = BaseChronosPipeline.from_pretrained(f"amazon/chronos-bolt-{name}", device_map=DEV, torch_dtype=torch.float32)
    L = CTX_H * 12
    preds = []
    for space, p in EV[["space", "time"]].drop_duplicates().groupby("space"):
        s = series(space)
        pos = s.index.get_indexer(p.time)
        ctx = torch.tensor(np.stack([s.values[i - L + 1:i + 1] if i - L + 1 >= 0
                                     else np.pad(s.values[:i + 1], (L - i - 1, 0), constant_values=np.nan) for i in pos]), dtype=torch.float32)
        q_all = []
        for i in range(0, len(ctx), BATCH):
            q, _ = pipe.predict_quantiles(ctx[i:i + BATCH], prediction_length=12, quantile_levels=[0.1, 0.5, 0.9])
            q_all.append(q.cpu().numpy())
        q = np.concatenate(q_all)
        for hi in range(12):
            preds.append(pd.DataFrame(dict(space=space, time=p.time.values, h=(hi + 1) * 5, q50=q[:, hi, 1])))
    m = EV.merge(pd.concat(preds), on=["space", "time", "h"])
    idx = pd.MultiIndex.from_arrays([m.space, m.time + pd.to_timedelta(m.h, "min")])
    m["true"] = grid["co2"].reindex(idx).values
    m["err"] = (m.q50 - m.true).abs()
    m.to_csv(OUT / f"b6_bolt_{name}_val_{CTX_H}h.csv.gz", index=False)
    print(name, "val 60분 MAE", round(m[m.h == 60].err.mean(), 2), "지점", (m.h == 60).sum(), flush=True)
    del pipe
    gc.collect()
    torch.cuda.empty_cache()
print("DONE")
