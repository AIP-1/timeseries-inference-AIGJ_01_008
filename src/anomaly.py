"""과제 4: 이상 탐지 — 오토인코더(모델) / 규칙 / 예측 오차(심화) / 결합.

  python anomaly.py   → report/anomaly_*.csv, report/fig_anomaly_*.png, models/ae.pt

탐지기는 정상 데이터(학습 기간)로만 학습하고, 임계값은 검증 기간(고장 없음)의 헛경보 수로 정한다.
시험 기간에 심어진 고장 17건(faults.csv)으로 평가한다.

경보 → 사건: 같은 공간에서 30분 안에 이어지는 경보는 한 사건으로 묶는다(같은 고장에 매분 울리지 않게).
탐지 성공: 고장 시작 ~ 끝+30분 사이에 시작한 경보 사건이 있으면. 지연 = 첫 경보 - 고장 시작.
헛경보: 어느 고장 구간(앞뒤 30분 포함)과도 겹치지 않는 경보 사건.
"""
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

import dataset
import prep
from models import AE
from viz import SPACE_KO, plt

ROOT = Path(__file__).resolve().parent
OUT, MD = ROOT / "report", ROOT / "models"
DEV = os.getenv("DEV", "cpu")   # 보드 GPU 메모리를 다른 작업과 나눠 쓰므로 기본은 CPU
MERGE = pd.Timedelta(minutes=30)
TYPE_KO = {"spike": "튄 값", "stuck": "값 고정", "drift": "영점 틀어짐", "ac_failure": "냉방 정지", "vent_failure": "환기 정지"}

grid, stats, removed, raw, sched, meta = prep.build()
cleaned, _ = prep.clean(raw)
faults = pd.read_csv(prep.DATA / "faults.csv", parse_dates=["start", "end"])


def _observable(r):
    """고장 구간에 센서 기록이 한 건이라도 있나(튄 값은 그 1분에 기록이 있어야 보인다)."""
    d = raw[(raw.space == r.space) & (raw.time >= r.start) & (raw.time < max(r.end, r.start + pd.Timedelta(minutes=1)))]
    return len(d) > 0


faults["observable"] = [_observable(r) for r in faults.itertuples()]
VAL0, TEST0 = pd.Timestamp(meta["split"]["val"][0]), pd.Timestamp(meta["split"]["test"][0])
END = pd.Timestamp(meta["end"])
VAL_DAYS = (TEST0 - VAL0).days
TEST_DAYS = (END - TEST0).total_seconds() / 86400


# ================================================================ 경보 → 사건, 평가
def to_events(alarms):
    """alarms: DataFrame(space, time, source, culprit). 30분 안에 이어지면 한 사건."""
    if alarms.empty:
        return pd.DataFrame(columns=["space", "start", "end", "source", "culprit"])
    ev = []
    for space, a in alarms.sort_values("time").groupby("space"):
        cur = None
        for r in a.itertuples():
            if cur and r.time - cur["end"] <= MERGE:
                cur["end"] = r.time
                cur["culprits"].append(r.culprit)
                cur["sources"].add(r.source)
            else:
                if cur:
                    ev.append(cur)
                cur = dict(space=space, start=r.time, end=r.time, culprits=[r.culprit], sources={r.source})
        ev.append(cur)
    df = pd.DataFrame(ev)
    df["culprit"] = df.culprits.apply(lambda c: pd.Series(c).mode().iloc[0])
    df["source"] = df.sources.apply(lambda s: "+".join(sorted(s)))
    return df.drop(columns=["culprits", "sources"])


def evaluate(alarms, period="test"):
    t0, t1, days = (TEST0, END, TEST_DAYS) if period == "test" else (VAL0, TEST0, VAL_DAYS)
    a = alarms[(alarms.time >= t0) & (alarms.time < t1)]
    ev = to_events(a)
    f = faults if period == "test" else faults.iloc[0:0]
    det = []
    for r in f.itertuples():
        hit = ev[(ev.space == r.space) & (ev.start >= r.start - pd.Timedelta(minutes=5)) & (ev.start <= r.end + MERGE)] if len(ev) else ev
        # 고장이 시작되기 전에 이미 열려 있던 사건이 고장 구간까지 이어진 경우도 탐지로 본다
        if len(ev) and hit.empty:
            hit = ev[(ev.space == r.space) & (ev.start < r.start) & (ev.end >= r.start)]
        first = hit.start.min() if len(hit) else pd.NaT
        delay = max(0.0, (first - r.start).total_seconds() / 60) if pd.notna(first) else np.nan
        culprit_ok = None
        if len(hit):
            c = hit.sort_values("start").iloc[0].culprit
            culprit_ok = (r.sensor == "all") or (c == r.sensor)
        det.append(dict(space=r.space, type=r.type, start=r.start, observable=r.observable, detected=pd.notna(first), delay_min=delay, culprit_ok=culprit_ok))
    det = pd.DataFrame(det)
    fa = 0
    for e in ev.itertuples() if len(ev) else []:
        ff = f[(f.space == e.space) & (f.start - MERGE <= e.end) & (f.end + MERGE >= e.start)]
        fa += ff.empty
    return det, fa / days, ev


def summarize(det, fa_per_day, name):
    by = det.groupby("type").agg(건수=("detected", "size"), 탐지=("detected", "sum"))
    obs = det[det.observable]
    row = {"탐지기": name, "이벤트 탐지율": det.detected.mean(), "탐지율(관측 가능 고장)": obs.detected.mean(), "평균 지연(분)": det.delay_min.mean(),
           "헛경보/일": fa_per_day, "원인 센서 적중": det.culprit_ok.dropna().mean() if det.culprit_ok.notna().any() else np.nan}
    for t in TYPE_KO:
        if t in by.index:
            row[TYPE_KO[t]] = f"{int(by.loc[t, '탐지'])}/{int(by.loc[t, '건수'])}"
    return row


# ================================================================ 1. 오토인코더 (정상만 학습)
W = 12  # 1시간 창
fe = dataset.features(grid, stats)




def ae_windows(space):
    d = fe[space]
    f = d["feat"]
    vals = f[:, [0, 2, 4]]          # co2, temperature, humidity (정규화)
    miss = f[:, [1, 3, 5]]
    cond = f[:, 6:]                 # 시각·요일·일정·공간 (재구성하지 않고 조건으로만)
    n = len(f)
    idx = np.arange(W - 1, n)
    win = np.stack([vals[i - W + 1:i + 1] for i in idx])         # [N, W, 3]
    mwin = np.stack([miss[i - W + 1:i + 1] for i in idx])
    return idx, win.reshape(len(idx), -1).astype(np.float32), mwin.reshape(len(idx), -1).astype(np.float32), cond[idx].astype(np.float32), d


def train_ae():
    Xs, Ms, Cs = [], [], []
    for s in dataset.SPACES:
        idx, x, m, c, d = ae_windows(s)
        ok = (d["split"][idx] == "train") & (m.mean(1) < 0.2)
        Xs.append(x[ok]); Ms.append(m[ok]); Cs.append(c[ok])
    X, M, C = (torch.from_numpy(np.concatenate(a)) for a in (Xs, Ms, Cs))
    torch.manual_seed(0)
    model = AE(X.shape[1], C.shape[1]).to(DEV)
    opt = torch.optim.Adam(model.parameters(), 1e-3)
    for ep in range(30):
        perm = torch.randperm(len(X))
        for i in range(0, len(X), 512):
            b = perm[i:i + 512]
            x, m, c = X[b].to(DEV), M[b].to(DEV), C[b].to(DEV)
            loss = (((model(x, c) - x) ** 2) * (1 - m)).mean()
            opt.zero_grad(); loss.backward(); opt.step()
    torch.save(dict(state=model.state_dict(), n_in=X.shape[1], n_cond=C.shape[1], W=W), MD / "ae.pt")
    return model


def ae_scores(model):
    rows = []
    model.eval()
    for s in dataset.SPACES:
        idx, x, m, c, d = ae_windows(s)
        with torch.no_grad():
            r = model(torch.from_numpy(x).to(DEV), torch.from_numpy(c).to(DEV)).cpu().numpy()
        err = ((r - x) ** 2 * (1 - m)).reshape(len(idx), W, 3)
        last = err[:, -3:, :].mean(1)                               # 최근 15분 재구성 오차(변수별)
        rows.append(pd.DataFrame(dict(space=s, time=d["times"][idx], split=d["split"][idx],
                                      score=last.sum(1), e_co2=last[:, 0], e_temperature=last[:, 1], e_humidity=last[:, 2],
                                      missing=m[:, -3:].mean(1))))
    return pd.concat(rows, ignore_index=True)


def ae_alarms(sc, thr, k=2):
    out = []
    for s, g in sc.groupby("space"):
        hit = (g.score > thr) & (g.missing < 0.5)
        run = hit.rolling(k).sum() >= k                             # k칸 연속 넘을 때만
        a = g[run]
        cul = a[["e_co2", "e_temperature", "e_humidity"]].idxmax(1).str[2:]
        out.append(pd.DataFrame(dict(space=s, time=a.time, source="AE", culprit=cul)))
    return pd.concat(out, ignore_index=True)


# ================================================================ 2. 규칙 (1분 원본)
def rule_alarms(spike_co2=300, spike_temp=3.0, stuck_min=30, drift_level=520, drift_flat=15):
    out = []
    for s, d in cleaned.groupby("space"):
        d = d.set_index("time").sort_index()
        period = meta["spaces"][s]["report_min"]
        # (1) 급변: 직전 3건 중앙값과 다음 1건 모두에서 크게 벗어난 한 건 (다음 값이 와야 판단 → 1주기 지연)
        for v, th in (("co2", spike_co2), ("temperature", spike_temp)):
            x = d[v]
            prev_med = x.shift(1).rolling(3, min_periods=2).median()
            nxt = x.shift(-1)
            sp = ((x - prev_med).abs() > th) & ((x - nxt).abs() > th)
            out.append(pd.DataFrame(dict(space=s, time=x.index[sp] + pd.Timedelta(minutes=period), source="규칙:급변", culprit=v)))
        # (2) 값 고정: 세 값이 모두 그대로인 상태가 stuck_min분 이상
        same = (d[prep.VARS].diff().abs().sum(1) == 0)
        grp = (~same).cumsum()
        runlen = same.groupby(grp).cumsum() * period
        st = runlen >= stuck_min
        out.append(pd.DataFrame(dict(space=s, time=d.index[st], source="규칙:고정", culprit="all")))
        # (3) 영점 틀어짐: 일정이 없고(최근 2시간) CO₂가 평평한데 외기보다 한참 높게 1시간 넘게 머문다
        co2 = d["co2"].resample("5min", label="right", closed="right").mean()
        roll_mean = co2.rolling(12, min_periods=8).mean()
        roll_std = co2.rolling(12, min_periods=8).std()
        busy = pd.Series(0.0, index=co2.index)
        for r in sched[sched.space == s].itertuples():
            busy[(busy.index > r.start) & (busy.index <= r.end + pd.Timedelta(hours=2))] = 1
        dr = (busy == 0) & (roll_mean > drift_level) & (roll_std < drift_flat)
        out.append(pd.DataFrame(dict(space=s, time=co2.index[dr], source="규칙:영점", culprit="co2")))
    return pd.concat(out, ignore_index=True)


# ================================================================ 2-1. 정상 범위 규칙 (설비 고장용)
def envelope_alarms(co2_margin=100.0, temp_margin=0.7, k=2, days=7):
    """직전 days일(같은 공간·평일/주말·같은 시각)의 최댓값 + 여유 위로 k칸(10분) 연속 올라가면 경보.
    냉방 정지(온도가 평소보다 3~5℃ 높아짐)와 환기 정지(CO₂ 상승 폭 1.8배)를 노린다.
    학습 기간 전체가 아니라 '최근 며칠'을 기준으로 삼는 이유: 9월 초 더위로 잡힌 온도 상한은 10월에는 너무 높다(계절 변화)."""
    g = grid.reset_index()
    g["hour"] = g.time.dt.hour
    g["weekend"] = g.time.dt.weekday >= 5
    g["date"] = g.time.dt.normalize()
    daily = g.groupby(["space", "weekend", "hour", "date"])[["co2", "temperature"]].max().reset_index().sort_values("date")
    # 그날을 빼고 직전 days일 중 같은 조건의 최댓값
    daily[["co2_max", "temp_max"]] = (daily.groupby(["space", "weekend", "hour"])[["co2", "temperature"]]
                                      .transform(lambda x: x.shift(1).rolling(days, min_periods=2).max()))
    g = g.merge(daily[["space", "weekend", "hour", "date", "co2_max", "temp_max"]], on=["space", "weekend", "hour", "date"], how="left")
    out = []
    for s, d in g.groupby("space"):
        d = d.sort_values("time")
        for v, mx, mg in (("co2", "co2_max", co2_margin), ("temperature", "temp_max", temp_margin)):
            over = (d[v] > d[mx] + mg) & ~d[v + "_imp"].astype(bool)
            run = over.astype(int).rolling(k).sum() >= k
            out.append(pd.DataFrame(dict(space=s, time=d.time[run], source="규칙:정상범위", culprit=v)))
    return pd.concat(out, ignore_index=True)


# ================================================================ 3. 예측 오차 탐지기 (심화)
def forecast_residuals():
    import models
    best = json.loads((OUT / "forecast_best.json").read_text())
    ck = torch.load(MD / f"forecast_{best['name']}.pt", weights_only=False)
    model = models.make(ck["kind"], ck["n_in"]).to(DEV)
    model.load_state_dict(ck["state"]); model.eval()
    L, hstep = ck["L"], 6   # 30분 전에 낸 예측과 비교
    std = np.array([stats[v]["std"] for v in dataset.TARGETS])
    rows = []
    for s, d in fe.items():
        n = len(d["feat"])
        t = np.arange(L - 1, n - hstep)
        X = np.stack([d["feat"][i - L + 1:i + 1] for i in t]).astype(np.float32)
        F = np.stack([d["fut"][i + 1:i + 13] if i + 13 <= n else np.pad(d["fut"][i + 1:], ((0, i + 13 - n), (0, 0))) for i in t]).astype(np.float32)
        with torch.no_grad():
            P = np.concatenate([model(torch.from_numpy(X[i:i + 4096]).to(DEV), torch.from_numpy(F[i:i + 4096]).to(DEV)).cpu().numpy()
                                for i in range(0, len(X), 4096)])
        pred = d["y_raw"][t] + P[:, hstep - 1, :] * std
        tgt = t + hstep
        res = d["y_raw"][tgt] - pred
        ok = d["obs"][tgt] & d["obs"][t]
        rows.append(pd.DataFrame(dict(space=s, time=d["times"][tgt], split=d["split"][tgt],
                                      r_co2=np.where(ok[:, 0], res[:, 0], np.nan), r_temperature=np.where(ok[:, 1], res[:, 1], np.nan))))
    return pd.concat(rows, ignore_index=True)


def fc_alarms(rs, z, sig, k=6):
    """30분 전 예측보다 실제가 z·σ(시간대별) 넘게 높은 상태가 k칸(30분) 이어지면 경보."""
    out = []
    for s, g in rs.groupby("space"):
        h = g.time.dt.hour
        zc = g.r_co2 / h.map(sig["co2"])
        zt = g.r_temperature / h.map(sig["temperature"])
        # 실제가 예측보다 높게 벗어날 때(설비 고장은 값을 올린다)만 본다
        hit_c = (zc > z).astype(int).rolling(k).sum() >= k
        hit_t = (zt > z).astype(int).rolling(k).sum() >= k
        for hit, v in ((hit_c, "co2"), (hit_t, "temperature")):
            out.append(pd.DataFrame(dict(space=s, time=g.time[hit], source="예측오차", culprit=v)))
    return pd.concat(out, ignore_index=True)


# ================================================================ 실행
if __name__ == "__main__":
    ae = train_ae()
    sc = ae_scores(ae)
    val_sc = sc[sc.split == "val"]
    # 임계값 후보: 검증 점수 분위수. 검증 헛경보 ≤ 0.5회/일인 가장 낮은 값을 고른다
    sweep = []
    for q in (0.99, 0.995, 0.998, 0.999, 0.9995, 0.9999, 1.0):
        thr = float(val_sc.score.quantile(q)) * (1.0 if q < 1 else 1.2)
        a = ae_alarms(sc, thr)
        _, fa_val, _ = evaluate(a, "val")
        det, fa, _ = evaluate(a, "test")
        sweep.append(dict(분위수=q, 임계값=round(thr, 3), 검증_헛경보_일=round(fa_val, 2), 시험_탐지율=round(det.detected.mean(), 3), 시험_헛경보_일=round(fa, 2)))
    sweep = pd.DataFrame(sweep)
    sweep.to_csv(OUT / "anomaly_ae_threshold.csv", index=False)
    print(sweep.to_string(index=False), flush=True)
    okq = sweep[sweep["검증_헛경보_일"] <= 0.5]
    chosen = okq.iloc[0] if len(okq) else sweep.sort_values("검증_헛경보_일").iloc[0]
    thr = float(chosen["임계값"])
    A_ae = ae_alarms(sc, thr)

    A_rule = rule_alarms()
    rule_val = evaluate(A_rule, "val")[1]
    env_sweep = []
    for cm, tm in ((50, 0.5), (100, 0.7), (150, 1.0), (200, 1.5), (300, 2.0)):
        a = envelope_alarms(cm, tm)
        env_sweep.append(dict(CO2_여유=cm, 온도_여유=tm, 검증_헛경보_일=round(evaluate(a, "val")[1], 2),
                              시험_탐지율=round(evaluate(a, "test")[0].detected.mean(), 3)))
    env_sweep = pd.DataFrame(env_sweep)
    env_sweep.to_csv(OUT / "anomaly_envelope_threshold.csv", index=False)
    print(env_sweep.to_string(index=False), flush=True)
    okm = env_sweep[env_sweep["검증_헛경보_일"] <= 0.15]
    em = okm.iloc[0] if len(okm) else env_sweep.sort_values("검증_헛경보_일").iloc[0]
    A_env = envelope_alarms(float(em["CO2_여유"]), float(em["온도_여유"]))
    # 오토인코더는 결합에서 헛경보를 거의 늘리지 않는 높은 기준(검증 헛경보 0)으로 쓴다
    q_hi = sweep[sweep["검증_헛경보_일"] == 0].iloc[0]
    A_ae_hi = ae_alarms(sc, float(q_hi["임계값"]))

    results, details = [], {}
    rs = forecast_residuals()
    v = rs[rs.split == "val"]
    # 시간대별 잔차 크기(검증, 중앙값 기반 σ). 수업 시작·점심처럼 원래 예측이 어려운 시각은 σ가 커져 덜 울린다
    v = v.assign(hour=v.time.dt.hour)
    sig = {c: (v.groupby("hour")["r_" + c].apply(lambda x: x.abs().quantile(0.5) / 0.6745).clip(lower=1e-3)).to_dict()
           for c in ("co2", "temperature")}
    fsweep = []
    for z in (3, 4, 5, 6, 8, 10, 12, 15):
        a = fc_alarms(rs, z, sig)
        fsweep.append(dict(z=z, 검증_헛경보_일=round(evaluate(a, "val")[1], 2), 시험_탐지율=round(evaluate(a, "test")[0].detected.mean(), 3)))
    fsweep = pd.DataFrame(fsweep)
    fsweep.to_csv(OUT / "anomaly_fc_threshold.csv", index=False)
    ok = fsweep[fsweep["검증_헛경보_일"] <= 0.3]
    zbest = float(ok.iloc[0].z if len(ok) else fsweep.sort_values("검증_헛경보_일").iloc[0].z)
    print(fsweep.to_string(index=False), flush=True)
    A_fc = fc_alarms(rs, zbest, sig)

    combos = (("오토인코더", A_ae), ("규칙(급변·고정·영점)", A_rule), ("규칙: 정상 범위", A_env), ("예측 오차(심화)", A_fc),
              ("결합 v1: 오토인코더+규칙+예측오차", pd.concat([A_ae, A_rule, A_fc])),
              ("결합 v2: 오토인코더(높은 기준)+규칙+정상 범위", pd.concat([A_ae_hi, A_rule, A_env])),
              ("결합 v3: 오토인코더(높은 기준)+규칙+정상 범위+예측오차", pd.concat([A_ae_hi, A_rule, A_env, A_fc])))
    for name, a in combos:
        det, fa, ev = evaluate(a, "test")
        details[name] = det
        results.append(summarize(det, fa, name))
    res = pd.DataFrame(results)
    res.to_csv(OUT / "anomaly_scorecard.csv", index=False)
    full = details["결합 v2: 오토인코더(높은 기준)+규칙+정상 범위"].copy()
    for name in ("오토인코더", "규칙(급변·고정·영점)", "규칙: 정상 범위", "예측 오차(심화)"):
        full[name] = details[name].detected.values
        full[name + " 지연"] = details[name].delay_min.values
    full.to_csv(OUT / "anomaly_by_event.csv", index=False)

    # 임계값-헛경보 그래프
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(sweep["시험_헛경보_일"], sweep["시험_탐지율"], marker="o")
    for r in sweep.itertuples():
        ax.annotate(f"q={r.분위수}", (r.시험_헛경보_일, r.시험_탐지율), fontsize=7)
    ax.axvline(1, color="red", ls=":", lw=0.8)
    ax.set_xlabel("하루당 헛경보 (시험, 네 공간 합)")
    ax.set_ylabel("이벤트 탐지율 (시험)")
    ax.set_title("오토인코더 임계값에 따른 탐지율과 헛경보")
    ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(OUT / "fig_anomaly_threshold.png", dpi=130)

    # 서비스(과제 8)는 결합 v2의 오토인코더 기준(높은 기준)을 쓴다
    meta_out = dict(ae_threshold=float(q_hi["임계값"]), ae_threshold_low=thr, ae_quantile=float(chosen["분위수"]), rule_val_fa_per_day=rule_val,
                    fc_z=zbest, fc_sigma={k: {int(h): round(float(x), 4) for h, x in d.items()} for k, d in sig.items()}, envelope_margin=dict(co2=float(em["CO2_여유"]), temperature=float(em["온도_여유"])))
    (OUT / "anomaly_meta.json").write_text(json.dumps(meta_out, ensure_ascii=False, indent=1))
    pd.set_option("display.width", 220)
    print(sweep.to_string(index=False)); print(fsweep.to_string(index=False))
    print(res.round(3).to_string(index=False))
    print(full.to_string(index=False))
    print(meta_out)
