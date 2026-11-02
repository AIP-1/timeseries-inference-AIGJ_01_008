"""과제 8 심화(a): 예측 기반 환기 제어 규칙을 지난 1주(합성 시험 주간)에 적용했을 때 1,000ppm 초과 시간이 얼마나 줄었을지 추정

  python b8_control_sim.py → report/b8_control_sim.csv, b8_control_actions.csv, b8_mass_balance_fit.csv

질량수지 가정(1분 단위)
  C[t+1] = C[t] + a(t)·(C_out − C[t]) + r(t)
    a(t) : 기본 환기율(분당 교환 비율). 운영 시간(meta open_hours) 안/밖 두 값을 실제 CO₂(truth.co2_true)에 최소제곱으로 맞춘다.
    r(t) : 사람이 내는 CO₂와 고장 영향 등 나머지. 원래 궤적에서 r = ΔC − a·(C_out − C) 로 그대로 구한다
           → 제어가 없으면 원래 궤적이 정확히 재현된다. 사람 수·고장은 그대로 두고 환기만 바꾼다고 보는 것이다.
  환기 강화 중: a(t) 에 extra_ach/60 을 더한다(기본 +2회/시간, 창문 개방·급기 팬 최대 정도). +1, +4 도 민감도로 함께 낸다.

제어 규칙(5분마다 판단)
  예측 규칙: 30분 안 예측 최댓값 ≥ 1,000ppm 이면 켠다. 켜면 최소 15분 유지, 예측 최댓값 < 900 이 되면 끈다.
    예측값 = 지금(제어 반영된) CO₂ + 과제 3 최종 모델(gru_L48)이 시험 주간에 냈던 '변화량' 예측.
    (모델은 원래 변화량을 맞히므로 출발점만 바꿔 쓴다. 환기 자체의 효과는 예측에 안 들어가 보수적으로 오래 켠다.)
    예측이 없는 시각(평가 지점 밖·고장 직후)은 지금 값 ≥ 1,000 이면 켜는 반응 규칙으로 대신한다.
  비교 규칙(반응형): 지금 CO₂ ≥ 1,000 이면 켜고 < 900 이면 끈다(최소 15분).
계산 장치: CPU. 예측은 과제 3에서 저장한 시험 주간 예측(pred_gru_L48.csv.gz)을 그대로 써서 새 모델 계산(torch)이 없다.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

import prep
from viz import SPACE_KO

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "report"
meta = json.loads((prep.DATA / "meta.json").read_text())
COUT = meta["co2_outdoor"]
OPEN0, OPEN1 = meta["open_hours"]
t0, t1 = pd.Timestamp(meta["split"]["test"][0]), pd.Timestamp(meta["split"]["test"][1])

truth = pd.read_csv(prep.DATA / "truth.csv.gz", parse_dates=["time"], low_memory=False)
best = json.loads((OUT / "forecast_best.json").read_text())["name"]
pr = pd.read_csv(OUT / f"pred_{best}.csv.gz", parse_dates=["time"])
pr = pr[(pr["var"] == "co2") & (pr.h <= 30) & pr.pred.notna()]
grid = prep.build()[0]

fit_rows, sim_rows, act_rows = [], [], []
for space in meta["spaces"]:
    tr = truth[truth.space == space].set_index("time").sort_index()
    c = tr.co2_true.astype(float)
    hr = c.index.hour + c.index.minute / 60
    is_open = ((hr >= OPEN0) & (hr < OPEN1)).astype(float)
    # ---- 기본 환기율 맞추기(학습 기간, 고장 없는 분만): ΔC = a_open·(Cout−C)·open + a_closed·(Cout−C)·(1−open) + g·n
    dC = c.shift(-1) - c
    X = np.column_stack([(COUT - c) * is_open, (COUT - c) * (1 - is_open), tr.occupants.astype(float)])
    ok = (c.index < pd.Timestamp(meta["split"]["val"][0])) & (tr.fault == "none").values & dC.notna().values
    coef, *_ = np.linalg.lstsq(X[ok], dC.values[ok], rcond=None)
    a_open, a_closed, g = coef
    fit_rows.append(dict(space=SPACE_KO[space], ach_open_fit=a_open * 60, ach_closed_fit=a_closed * 60, ach_on_meta=meta["spaces"][space]["ach_on"],
                         ach_off_meta=meta["spaces"][space]["ach_off"], ppm_per_person_min=g,
                         r2=1 - np.sum((dC.values[ok] - X[ok] @ coef) ** 2) / np.sum((dC.values[ok] - dC.values[ok].mean()) ** 2)))
    # ---- 시험 주간 시뮬레이션
    w = (c.index >= t0) & (c.index <= t1)
    cw, aw = c[w].values, np.where(is_open[w] > 0, a_open, a_closed)
    r = np.r_[cw[1:] - cw[:-1] - aw[:-1] * (COUT - cw[:-1]), 0.0]
    times = c[w].index
    # 예측 변화량: (발행 시각 t) → 30분 안 최대 변화량. 지금 값은 5분 표의 관측 CO₂
    g5 = grid.loc[space].co2
    p = pr[pr.space == space].groupby("time").pred.max()
    dmax = (p - g5.reindex(p.index)).dropna()
    dmax = dmax.reindex(times)

    def simulate(rule, extra_ach):
        x = np.empty_like(cw); x[0] = cw[0]
        on, on_since, n_act, on_min, n_fb = False, None, 0, 0, 0
        log = []
        for i in range(len(cw) - 1):
            t = times[i]
            if t.minute % 5 == 0:   # 5분마다 판단
                if rule == "forecast" and not np.isnan(dmax.iloc[i]):
                    level = max(x[i], x[i] + dmax.iloc[i])
                else:
                    level = x[i]
                    n_fb += rule == "forecast"
                if not on and level >= 1000:
                    on, on_since, n_act = True, t, n_act + 1
                elif on and level < 900 and (t - on_since) >= pd.Timedelta(minutes=15):
                    log.append(dict(space=SPACE_KO[space], rule=rule, extra_ach=extra_ach, start=on_since, end=t))
                    on = False
            a = aw[i] + (extra_ach / 60 if on else 0)
            on_min += on
            x[i + 1] = x[i] + a * (COUT - x[i]) + r[i]
        if on:
            log.append(dict(space=SPACE_KO[space], rule=rule, extra_ach=extra_ach, start=on_since, end=times[-1]))
        return x, n_act, on_min, n_fb, log

    base_over = int((cw >= 1000).sum())
    for extra in (1, 2, 4):
        for rule in ("reactive", "forecast"):
            x, n_act, on_min, n_fb, log = simulate(rule, extra)
            act_rows += log
            sim_rows.append(dict(space=SPACE_KO[space], rule={"reactive": "반응형(지금 ≥1000)", "forecast": "예측형(30분 안 ≥1000)"}[rule],
                                 extra_ach=extra, over1000_before_min=base_over, over1000_after_min=int((x >= 1000).sum()),
                                 reduction_pct=100 * (1 - (x >= 1000).sum() / base_over) if base_over else np.nan,
                                 peak_before=float(cw.max()), peak_after=float(x.max()), actions=n_act, vent_minutes=int(on_min),
                                 no_forecast_decisions=int(n_fb)))
    print(f"[{SPACE_KO[space]}] 기본 환기 맞춤 R²={fit_rows[-1]['r2']:.3f}, 시험 주간 1,000ppm 이상 {base_over}분", flush=True)

fit = pd.DataFrame(fit_rows)
sim = pd.DataFrame(sim_rows)
tot = sim.groupby(["rule", "extra_ach"], as_index=False)[["over1000_before_min", "over1000_after_min", "actions", "vent_minutes", "no_forecast_decisions"]].sum()
tot["space"] = "합계"
tot["reduction_pct"] = 100 * (1 - tot.over1000_after_min / tot.over1000_before_min)
sim = pd.concat([sim, tot], ignore_index=True)
fit.to_csv(OUT / "b8_mass_balance_fit.csv", index=False)
sim.to_csv(OUT / "b8_control_sim.csv", index=False)
pd.DataFrame(act_rows).to_csv(OUT / "b8_control_actions.csv", index=False)
pd.set_option("display.width", 250)
print(fit.round(3).to_string(index=False))
print(sim.round(1).to_string(index=False))
