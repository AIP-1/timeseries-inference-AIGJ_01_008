"""과제 5 심화: 예약 vs 추정 사용 여부 → 회의실 노쇼·예약 없는 사용 찾기 (합성 데이터 시험 주간)

  python b5_noshow.py → report/b5_noshow_events.csv, b5_unbooked_events.csv, b5_summary.csv

- 사용 여부 = 과제 5 분류 모델(models/occ.pkl, 센서만, 5분 칸) 예측. CO₂ 상승 규칙이 아니다.
- 노쇼 규칙: 예약 시작 후 15분(5분 칸 3개) 동안 모델이 계속 '비어있음' → 노쇼.
- 예약 없는 사용 규칙: 모델이 '사용중'을 15분(3칸) 이상 연속으로 내고, 그 구간이 어떤 예약(시작~끝+15분)과도 겹치지 않음.
- 정답('직접 관찰' 대신): 합성 데이터의 실제 인원(truth.csv.gz). 1분 단위.
- 비교: 기존 CO₂ 상승 규칙(예약 시작 후 30분 안에 +50ppm 없음 → 노쇼)을 같은 시험 주간 예약에 다시 적용.
- 계산 장치: CPU (scikit-learn GBDT라 GPU 경로가 없다).
"""
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

import state  # table(), truth, grid 를 그대로 쓴다(모듈 로드 때 prep.build 실행)
import prep

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "report"
M15 = pd.Timedelta(minutes=15)

d = state.table()
occ = pickle.load(open(ROOT / "models/occ.pkl", "rb"))
SPLITS = os.environ.get("B5_SPLITS", "test").split(",")   # 기본: 시험 주간. "val,test" 면 모델이 학습에 안 쓴 2주
SUF = "" if SPLITS == ["test"] else "_" + "_".join(SPLITS)
te = d[d.split.isin(SPLITS)].copy()
ok = te[occ["feats"]].notna().all(1) & te.obs_ok
te["pred"] = np.nan
te.loc[ok, "pred"] = occ["model"].predict(te.loc[ok, occ["feats"]])
print(f"[예측] 시험 주간 5분 칸 {len(te)}개 중 예측 가능 {ok.sum()}개", flush=True)

truth = state.truth
tr1 = {s: g.set_index("time").occupants.sort_index() for s, g in truth.groupby("space")}
sched = state.sched
meta = state.meta
ts0, ts1 = pd.Timestamp(meta["split"][SPLITS[0]][0]), pd.Timestamp(meta["split"][SPLITS[-1]][1])

# ------------------------------------------------------------ 1. 노쇼 (회의실)
cleaned, _ = prep.clean(state.raw)
co2_1m = cleaned[cleaned.space == "meeting"].set_index("time").co2.sort_index()
pm = te[te.space == "meeting"].set_index("time").pred
bk = sched[(sched.space == "meeting") & (sched.start >= ts0) & (sched.start <= ts1)].copy()
rows = []
for _, b in bk.iterrows():
    slots = pm[(pm.index > b.start) & (pm.index <= b.start + M15)]          # (s, s+15] 의 5분 칸 3개
    est = bool(len(slots) and slots.notna().all() and (slots == 0).all())
    occ1 = tr1["meeting"][(tr1["meeting"].index > b.start - pd.Timedelta(minutes=1)) & (tr1["meeting"].index < b.start + M15)]
    occ30 = tr1["meeting"][(tr1["meeting"].index > b.start - pd.Timedelta(minutes=1)) & (tr1["meeting"].index < b.start + 2 * M15)]
    before = co2_1m[(co2_1m.index > b.start - pd.Timedelta(minutes=10)) & (co2_1m.index <= b.start)].dropna()
    w = co2_1m[(co2_1m.index > b.start) & (co2_1m.index <= b.start + 2 * M15)].dropna()
    co2_rule = bool(len(before) and not (w >= before.mean() + 50).any())
    rows.append(dict(start=b.start, end=b.end, expected=b.expected, slots=len(slots), slots_pred="".join("?" if np.isnan(x) else str(int(x)) for x in slots),
                     est_noshow=est, co2rule_noshow=co2_rule, truth_noshow15=bool(occ1.max() == 0), truth_noshow30=bool(occ30.max() == 0),
                     truth_first_in_min=float((occ1[occ1 > 0].index[0] - b.start).total_seconds() / 60 + 1) if (occ1 > 0).any() else np.nan))
ns = pd.DataFrame(rows)
ns.to_csv(OUT / f"b5_noshow_events{SUF}.csv", index=False)


def cm(est, truth_):
    return dict(예약수=len(est), 실제노쇼=int(truth_.sum()), 찾아냄=int(est.sum()), 맞힘=int((est & truth_).sum()),
                놓침=int((~est & truth_).sum()), 헛경보=int((est & ~truth_).sum()))


summ = [dict(항목="노쇼", 규칙="모델 예측: 시작 후 15분 계속 비어있음", 정답="15분 비어있음") | cm(ns.est_noshow, ns.truth_noshow15),
        dict(항목="노쇼", 규칙="모델 예측: 시작 후 15분 계속 비어있음", 정답="30분 비어있음") | cm(ns.est_noshow, ns.truth_noshow30),
        dict(항목="노쇼", 규칙="(기존) CO₂ 상승: 30분 안 +50ppm 없음", 정답="15분 비어있음") | cm(ns.co2rule_noshow, ns.truth_noshow15),
        dict(항목="노쇼", 규칙="(기존) CO₂ 상승: 30분 안 +50ppm 없음", 정답="30분 비어있음") | cm(ns.co2rule_noshow, ns.truth_noshow30)]
print(f"[노쇼] {SPLITS} 회의실 예약 {len(ns)}건", flush=True)


# ------------------------------------------------------------ 2. 예약 없는 사용 (모든 공간, 회의실이 주 대상)
def runs(mask_series, min_len):
    """True 가 min_len 칸 이상 이어진 구간 [(시작, 끝)]. 시각은 칸 끝 시각."""
    m = mask_series.fillna(False).astype(bool).values
    t = mask_series.index
    out, i = [], 0
    while i < len(m):
        if m[i]:
            j = i
            while j + 1 < len(m) and m[j + 1] and (t[j + 1] - t[j]) <= pd.Timedelta(minutes=5):
                j += 1
            if j - i + 1 >= min_len:
                out.append((t[i], t[j]))
            i = j + 1
        else:
            i += 1
    return out


def booked(space, a, b, pad=M15):
    s = sched[sched.space == space]
    return ((s.start - pad < b) & (s.end + pad > a)).any()   # 앞뒤 15분 여유(일찍 오기·CO₂ 잔류)


ub_rows = []
for space in state.SPACES:
    p = te[te.space == space].set_index("time").pred
    est_ev = [(a, b) for a, b in runs(p == 1, 3) if not booked(space, a - pd.Timedelta(minutes=5), b)]
    o = tr1[space][(tr1[space].index >= ts0) & (tr1[space].index <= ts1)]
    tru_ev = [(a, b) for a, b in runs(o > 0, 15) if not booked(space, a, b)]     # 1분 단위 15칸 = 15분
    hit_est = [any(a <= tb and b >= ta for ta, tb in tru_ev) for a, b in est_ev]
    hit_tru = [any(a <= tb and b >= ta for ta, tb in est_ev) for a, b in tru_ev]
    for (a, b), h in zip(est_ev, hit_est):
        ub_rows.append(dict(space=space, kind="추정", start=a - pd.Timedelta(minutes=5), end=b, minutes=(b - a).total_seconds() / 60 + 5, matched=h,
                            truth_max_people=int(tr1[space][(tr1[space].index > a - pd.Timedelta(minutes=5)) & (tr1[space].index <= b)].max())))
    for (a, b), h in zip(tru_ev, hit_tru):
        ub_rows.append(dict(space=space, kind="정답", start=a, end=b, minutes=(b - a).total_seconds() / 60 + 1, matched=h,
                            truth_max_people=int(tr1[space][(tr1[space].index >= a) & (tr1[space].index <= b)].max())))
    summ.append(dict(항목="예약 없는 사용", 규칙=f"{state.SPACE_KO[space]}: 모델 예측 사용중 15분 이상 + 예약 없음", 정답="실제 인원>0 15분 이상 + 예약 없음",
                     예약수=int(((sched.space == space) & (sched.start >= ts0) & (sched.start <= ts1)).sum()),
                     실제노쇼=len(tru_ev), 찾아냄=len(est_ev), 맞힘=int(sum(hit_est)), 놓침=int(len(hit_tru) - sum(hit_tru)),
                     헛경보=int(len(hit_est) - sum(hit_est))))
ub = pd.DataFrame(ub_rows)
ub.to_csv(OUT / f"b5_unbooked_events{SUF}.csv", index=False)
sm = pd.DataFrame(summ).rename(columns={"실제노쇼": "실제건수"})
sm.to_csv(OUT / f"b5_summary{SUF}.csv", index=False)
pd.set_option("display.width", 250)
print(sm.to_string(index=False))
print(ns.to_string(index=False))
print(ub.to_string(index=False))
