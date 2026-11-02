"""과제 5: 상태 추정 — 사용 여부(센서만) / 환기 선제 경보(30분 안 1,000ppm 초과).

  python state.py → report/state_*.csv, report/fig_state_cm.png, models/occ.pkl, models/vent.pkl

정답
  사용 여부: 학습 정답은 일정(시간표·예약)에서 만든다. '직접 관찰'은 합성 데이터의 실제 인원(truth)으로
            흉내 낸다: 시험 기간 평일 중 공간마다 이틀, 9~18시를 관찰했다고 가정한다.
  환기 경보: 지금 CO₂ < 1,000이고 다음 30분(5분 칸 6개) 안에 1,000 이상이 되는 시각만 양성.
            이미 넘은 시각은 데이터에서 뺀다(그때는 '선제' 경보가 아니라 이미 늦은 경보다).
"""
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

import prep
from dataset import SPACES, sensor_features
from viz import SPACE_KO, plt

ROOT = Path(__file__).resolve().parent
OUT, MD = ROOT / "report", ROOT / "models"
grid, stats, removed, raw, sched, meta = prep.build()
fault = prep.fault_mask(grid, pad_min=60)
truth = pd.read_csv(prep.DATA / "truth.csv.gz", parse_dates=["time"], low_memory=False)


def table(use_space=True):
    rows = []
    for space in SPACES:
        g = grid.loc[space]
        f = sensor_features(g, space, use_space)
        f["space"], f["split"] = space, g["split"].values
        f["sched_occ"] = (g["sched_people"] > 0).astype(int).values
        f["sched_next30"] = g["sched_people"][::-1].rolling(6, min_periods=1).max()[::-1].shift(-1).fillna(0).values / 30
        f["sched_now"] = g["sched_people"].values / 30
        c = g["co2"]
        fut_max = c[::-1].rolling(6, min_periods=1).max()[::-1].shift(-1)
        f["vent_label"] = ((c < 1000) & (fut_max >= 1000)).astype(int)
        f["vent_ok"] = (c < 1000) & fut_max.notna() & ~g["co2_imp"].astype(bool)
        f["obs_ok"] = g["co2"].notna() & ~g["co2_imp"].astype(bool)
        f["fault"] = fault.loc[space].values
        rows.append(f.reset_index())
    d = pd.concat(rows, ignore_index=True)
    tr = truth[["space", "time", "occupied"]].copy()
    tr["time"] = tr.time + pd.Timedelta(minutes=1)  # truth는 (t-1, t] 사용 → 5분 칸 끝 시각으로 맞춘다
    tr5 = tr.set_index("time").groupby("space").occupied.resample("5min", label="right", closed="right").max().reset_index()
    d = d.merge(tr5.rename(columns={"occupied": "true_occ"}), on=["space", "time"], how="left")
    return d


def observed_slots(d, rng):
    """시험 기간 평일 중 공간마다 이틀, 9~18시를 '직접 관찰'한 것으로 둔다."""
    te = d[(d.split == "test") & (d.time.dt.weekday < 5)]
    days = sorted(te.time.dt.normalize().unique())
    keep = []
    for s in SPACES:
        for day in rng.choice(days, 2, replace=False):
            keep.append(te[(te.space == s) & (te.time.dt.normalize() == day) & (te.time.dt.hour >= 9) & (te.time.dt.hour < 18)].index)
    return np.concatenate(keep)


def metrics(y, p, pos_name):
    cm = confusion_matrix(y, p, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return dict(정확도=(tp + tn) / cm.sum(), F1=f1_score(y, p, zero_division=0), macro_F1=f1_score(y, p, average="macro", zero_division=0),
                재현율=recall_score(y, p, zero_division=0), 정밀도=precision_score(y, p, zero_division=0),
                빈방을_사용중으로=fp / max(fp + tn, 1), 지점수=int(cm.sum()), 양성비율=float(np.mean(y))), cm


X_OCC = None


def occupancy(d, rng):
    global X_OCC
    feats = [c for c in d.columns if c not in ("time", "space", "split", "sched_occ", "sched_next30", "sched_now", "vent_label",
                                               "vent_ok", "obs_ok", "fault", "true_occ")]
    X_OCC = feats
    tr = d[(d.split == "train") & d.obs_ok & ~d.fault].dropna(subset=feats)
    clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=0).fit(tr[feats], tr.sched_occ)
    obs_idx = observed_slots(d, rng)
    ob = d.loc[obs_idx]
    ob = ob[ob.obs_ok & ~ob.fault].dropna(subset=feats)
    res, cms = [], {}
    # 일정 기반 정답이 관찰과 얼마나 다른가
    m, cm = metrics(ob.true_occ.astype(int), ob.sched_occ, "")
    res.append(dict(방법="일정 그대로(정답 품질 확인)") | m); cms["일정"] = cm
    rule = (ob["dco2_15"] >= 30).astype(int)
    m, cm = metrics(ob.true_occ.astype(int), rule, ""); res.append(dict(방법="규칙: 15분 CO₂ +30ppm 이상") | m); cms["규칙"] = cm
    pred = clf.predict(ob[feats])
    m, cm = metrics(ob.true_occ.astype(int), pred, ""); res.append(dict(방법="분류 모델(센서만)") | m); cms["모델"] = cm
    # 오류 사례: 언제 틀리나
    ob = ob.assign(pred=pred)
    ob["case"] = np.select([(ob.pred == 1) & (ob.true_occ == 0), (ob.pred == 0) & (ob.true_occ == 1)], ["빈 방→사용중", "사용중→빈 방"], "맞음")
    # 직전 사용 여부(막 나간 직후·막 들어온 직후)
    full = d.set_index(["space", "time"]).true_occ
    prev = [full.get((s, t - pd.Timedelta(minutes=15)), np.nan) for s, t in zip(ob.space, ob.time)]
    ob["prev15"] = prev
    ob["상황"] = np.select([(ob.true_occ == 0) & (ob.prev15 == 1), (ob.true_occ == 1) & (ob.prev15 == 0)],
                         ["막 나간 직후(15분 안)", "막 들어온 직후(15분 안)"], "그 밖")
    err = ob[ob.case != "맞음"].groupby(["case", "상황", "space"]).size().rename("건수").reset_index()
    err["space"] = err.space.map(SPACE_KO)
    err.to_csv(OUT / "state_occ_errors.csv", index=False)
    # 시험 기간 전체(모든 시간대)에서 실제 인원 대비 성능도 함께
    te = d[(d.split == "test") & d.obs_ok & ~d.fault].dropna(subset=feats)
    m_all, _ = metrics(te.true_occ.astype(int), clf.predict(te[feats]), "")
    m_rule_all, _ = metrics(te.true_occ.astype(int), (te["dco2_15"] >= 30).astype(int), "")
    res.append(dict(방법="(참고) 모델, 시험 기간 전체 시간대") | m_all)
    res.append(dict(방법="(참고) 규칙, 시험 기간 전체 시간대") | m_rule_all)
    # 일정 기반 정답 vs 관찰: 어디가 다른가
    diff = ob[ob.sched_occ != ob.true_occ.astype(int)]
    sd = diff.groupby(["space"]).apply(lambda x: pd.Series({"일정=사용, 실제=빈 방": int(((x.sched_occ == 1) & (x.true_occ == 0)).sum()),
                                                          "일정=빈 방, 실제=사용": int(((x.sched_occ == 0) & (x.true_occ == 1)).sum())}), include_groups=False)
    sd.index = sd.index.map(SPACE_KO)
    sd.to_csv(OUT / "state_sched_vs_obs.csv")
    pickle.dump(dict(model=clf, feats=feats), open(MD / "occ.pkl", "wb"))
    return pd.DataFrame(res), cms, err, sd


def leave_one_out_occ(d):
    rows = []
    d2 = table(use_space=False)
    feats = [c for c in X_OCC if not c.startswith("sp_")]
    for hold in SPACES:
        tr = d2[(d2.split == "train") & d2.obs_ok & ~d2.fault & (d2.space != hold)].dropna(subset=feats)
        te = d2[(d2.split == "test") & d2.obs_ok & ~d2.fault & (d2.space == hold)].dropna(subset=feats)
        clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=0).fit(tr[feats], tr.sched_occ)
        same = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=0).fit(
            d2[(d2.split == "train") & d2.obs_ok & ~d2.fault].dropna(subset=feats)[feats],
            d2[(d2.split == "train") & d2.obs_ok & ~d2.fault].dropna(subset=feats).sched_occ)
        m1, _ = metrics(te.true_occ.astype(int), clf.predict(te[feats]), "")
        m0, _ = metrics(te.true_occ.astype(int), same.predict(te[feats]), "")
        rows.append({"뺀 공간": SPACE_KO[hold], "F1(학습에 포함)": m0["F1"], "F1(처음 보는 공간)": m1["F1"],
                     "빈방→사용중(포함)": m0["빈방을_사용중으로"], "빈방→사용중(처음)": m1["빈방을_사용중으로"]})
    return pd.DataFrame(rows)


def ventilation(d):
    feats = X_OCC + ["sched_now", "sched_next30"]   # 환기 경보에는 미래 일정도 쓴다(미리 알 수 있는 정보)
    base = d[d.vent_ok & ~d.fault].dropna(subset=feats)
    tr, va, te = (base[base.split == s] for s in ("train", "val", "test"))
    clf = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05, class_weight="balanced", random_state=0).fit(tr[feats], tr.vent_label)
    # 임계값: 검증에서 재현율 ≥ 0.90을 지키는 범위에서 F1이 가장 높은 값
    pv = clf.predict_proba(va[feats])[:, 1]
    best = (0.5, -1)
    for th in np.linspace(0.05, 0.95, 91):
        p = (pv >= th).astype(int)
        if recall_score(va.vent_label, p) >= 0.90 and f1_score(va.vent_label, p) > best[1]:
            best = (th, f1_score(va.vent_label, p))
    th = best[0]
    res, cms = [], {}
    y = te.vent_label.values
    m, cm = metrics(y, np.zeros_like(y), ""); res.append(dict(방법="항상 '경보 없음'") | m); cms["없음"] = cm
    m, cm = metrics(y, (te.co2 >= 850).astype(int).values, ""); res.append(dict(방법="규칙: 지금 CO₂ ≥ 850") | m); cms["규칙"] = cm
    pt = (clf.predict_proba(te[feats])[:, 1] >= th).astype(int)
    m, cm = metrics(y, pt, ""); res.append(dict(방법=f"분류 모델(임계 {th:.2f})") | m); cms["모델"] = cm
    # 과제 3 예측 모델로 판단: 예측한 30분 안 최댓값 ≥ 1,000
    import json
    best_name = json.loads((OUT / "forecast_best.json").read_text())["name"]
    pr = pd.read_csv(OUT / f"pred_{best_name}.csv.gz", parse_dates=["time"])
    pr = pr[(pr["var"] == "co2") & (pr.h <= 30)].groupby(["space", "time"]).pred.max().rename("fc_max30").reset_index()
    tf = te.merge(pr, on=["space", "time"], how="inner")
    m, cm = metrics(tf.vent_label.values, (tf.fc_max30 >= 1000).astype(int).values, ""); res.append(dict(방법="과제 3 예측 모델(30분 안 최대 예측 ≥ 1,000)") | m); cms["예측"] = cm
    pickle.dump(dict(model=clf, feats=feats, threshold=th), open(MD / "vent.pkl", "wb"))
    # 가장 많이 틀리는 경우
    te = te.assign(pred=pt)
    miss = te[(te.vent_label == 1) & (te.pred == 0)]
    fa = te[(te.vent_label == 0) & (te.pred == 1)]
    cases = pd.DataFrame({"놓침(공간)": miss.space.map(SPACE_KO).value_counts(), "헛경보(공간)": fa.space.map(SPACE_KO).value_counts()}).fillna(0).astype(int)
    return pd.DataFrame(res), cms, cases, th, fa


if __name__ == "__main__":
    rng = np.random.default_rng(7)
    d = table()
    occ, occ_cm, occ_err, sched_vs_obs = occupancy(d, rng)
    loo = leave_one_out_occ(d)
    vent, vent_cm, vent_cases, th, fa = ventilation(d)
    for name, t in (("occupancy", occ), ("occupancy_loo", loo), ("vent", vent), ("vent_cases", vent_cases)):
        t.to_csv(OUT / f"state_{name}.csv", index=name == "vent_cases")
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    for ax, cm, title, labels in ((axes[0], occ_cm["모델"], "사용 여부 (관찰 시간대)", ["비어있음", "사용중"]),
                                  (axes[1], vent_cm["모델"], "환기 선제 경보 (시험 기간)", ["정상", "곧 환기 필요"])):
        ax.imshow(cm, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=12)
        ax.set_xticks([0, 1], [f"예측: {l}" for l in labels], fontsize=8)
        ax.set_yticks([0, 1], [f"실제: {l}" for l in labels], fontsize=8)
        ax.set_title(title, fontsize=10)
    fig.tight_layout(); fig.savefig(OUT / "fig_state_cm.png", dpi=130)
    pd.set_option("display.width", 220)
    for t in (occ, sched_vs_obs, occ_err, loo, vent, vent_cases):
        print(t.round(3).to_string()); print()
    print("occ cm", {k: v.tolist() for k, v in occ_cm.items()})
    print("vent cm", {k: v.tolist() for k, v in vent_cm.items()}, "th", th)
