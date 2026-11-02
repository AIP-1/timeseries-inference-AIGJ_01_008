"""과제 3: 60분 예측 모델 학습·평가.

  python train_forecast.py            # 실험 전부(모델 3종 × 과거 길이 3종 + 일정 제외 + 처음 보는 공간)
  python train_forecast.py gru 24     # 하나만

모델 선택은 검증 기간 60분 CO₂ MAE로만 한다. 시험 기간은 마지막에 한 번 평가만 한다.
평가는 과제 2와 같은 지점(report/eval_points.csv)에서 잰다.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import dataset
import models
import prep

ROOT = Path(__file__).resolve().parent
OUT, MD = ROOT / "report", ROOT / "models"
MD.mkdir(exist_ok=True)
import os
# 보드 메모리(8GB, 스왑 없음)를 팀원 작업과 나눠 쓰므로 기본은 CPU 4스레드로 학습한다(CUDA 컨텍스트만 1GB 가까이 쓴다)
DEV = os.getenv("DEV", "cpu")
torch.set_num_threads(int(os.getenv("THREADS", "4")))

grid, stats, *_ = prep.build()
fault = prep.fault_mask(grid)
EVAL = pd.read_csv(OUT / "eval_points.csv", parse_dates=["time"])
STD = np.array([stats[v]["std"] for v in dataset.TARGETS], np.float32)


def predict(model, X, F, bs=4096):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            out.append(model(torch.from_numpy(X[i:i + bs]).to(DEV), torch.from_numpy(F[i:i + bs]).to(DEV)).cpu().numpy())
    return np.concatenate(out)


def masked_mae(p, y, m):
    return (np.abs(p - y) * m).sum((0)) / np.maximum(m.sum(0), 1)  # [12, 2] (정규화 단위)


def run(kind="gru", L=24, use_sched=True, use_space=True, holdout=None, epochs=40, patience=6, seed=0, name=None, save=True):
    torch.manual_seed(seed)
    np.random.seed(seed)
    name = name or f"{kind}_L{L}{'' if use_sched else '_nosched'}{'' if holdout is None else '_hold-' + holdout}"
    fe = dataset.features(grid, stats, use_sched=use_sched, use_space=use_space and holdout is None)
    train_sp = None if holdout is None else [s for s in dataset.SPACES if s != holdout]
    Xtr, Ftr, Ytr, Mtr, _ = dataset.samples(fe, stats, L, "train", fault, spaces=train_sp)
    Xva, Fva, Yva, Mva, _ = dataset.samples(fe, stats, L, "val", fault, spaces=train_sp)
    Xte, Fte, Yte, Mte, Kte = dataset.samples(fe, stats, L, "test", fault, min_hist_obs=0.0)  # 평가 지점은 모두 예측

    model = models.make(kind, Xtr.shape[2]).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), 1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=2)
    tr = [torch.from_numpy(a) for a in (Xtr, Ftr, Ytr, Mtr)]
    best, best_state, bad, t0, log = 1e9, None, 0, time.time(), []
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(Xtr))
        for i in range(0, len(perm), 256):
            b = perm[i:i + 256]
            x, f, y, m = (a[b].to(DEV) for a in tr)
            loss = ((model(x, f) - y).abs() * m).sum() / m.sum().clamp(min=1)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        val = masked_mae(predict(model, Xva, Fva), Yva, Mva)
        v60 = float(val[-1, 0] * STD[0])  # 검증 60분 CO₂ MAE(ppm) → 선택 기준
        sched.step(v60)
        log.append(v60)
        if v60 < best - 0.05:
            best, bad = v60, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    train_sec = time.time() - t0

    # 시험: 같은 평가 지점에서
    P = predict(model, Xte, Fte)
    rows = []
    for j, var in enumerate(dataset.TARGETS):
        now = Kte["y_now_co2" if var == "co2" else "y_now_temp"].values
        for hi in range(dataset.H):
            rows.append(pd.DataFrame(dict(space=Kte.space, time=Kte.time, var=var, h=(hi + 1) * 5,
                                          pred=now + P[:, hi, j] * STD[j],
                                          true=np.where(Mte[:, hi, j] > 0, now + Yte[:, hi, j] * STD[j], np.nan))))
    pr = pd.concat(rows)
    pr = EVAL[EVAL.split == "test"].merge(pr, on=["space", "time", "var", "h"], how="left")
    cover = pr.pred.notna().mean()
    pr["err"] = (pr.pred - pr.true).abs()
    res = dict(name=name, kind=kind, L=L, use_sched=use_sched, holdout=holdout, epochs=len(log), train_sec=round(train_sec, 1),
               val_co2_60=round(best, 2), coverage=round(float(cover), 4),
               params=sum(p.numel() for p in model.parameters()), n_train=len(Xtr))
    for var in dataset.TARGETS:
        for h in (5, 15, 30, 60):
            res[f"test_{var}_{h}"] = round(float(pr[(pr["var"] == var) & (pr.h == h)].err.mean()), 3)
    if save:
        torch.save(dict(state=model.state_dict(), kind=kind, L=L, use_sched=use_sched, use_space=use_space and holdout is None,
                        n_in=Xtr.shape[2], stats=stats), MD / f"forecast_{name}.pt")
        pr.to_csv(OUT / f"pred_{name}.csv.gz", index=False)
    print(json.dumps(res, ensure_ascii=False), flush=True)
    return res


if __name__ == "__main__":
    if len(sys.argv) > 2:
        run(sys.argv[1], int(sys.argv[2]))
        sys.exit()
    # 중간에 끊겨도 이어서 할 수 있게, 로그에 이미 있는 실험은 건너뛴다
    done = {}
    log = OUT / "forecast_run.log"
    if log.exists():
        for line in log.read_text().splitlines():
            if line.startswith("{"):
                r = json.loads(line)
                if (MD / f"forecast_{r['name']}.pt").exists():
                    done[r["name"]] = r
    results = []
    for kind in ("gru", "lstm", "tcn"):
        for L in (12, 24, 48):
            results.append(done.get(f"{kind}_L{L}") or run(kind, L))
    best = min(results, key=lambda r: r["val_co2_60"])
    results.append(done.get(f"{best['kind']}_L{best['L']}_nosched") or run(best["kind"], best["L"], use_sched=False))
    for hold in ("meeting", "classroom_b"):
        results.append(done.get(f"{best['kind']}_L{best['L']}_hold-{hold}") or run(best["kind"], best["L"], holdout=hold))
    pd.DataFrame(results).to_csv(OUT / "forecast_results.csv", index=False)
    (OUT / "forecast_best.json").write_text(json.dumps(best, ensure_ascii=False, indent=1))
    print("BEST", best["name"])
