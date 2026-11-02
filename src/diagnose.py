"""과제 2 데이터 진단: 결측률·제거 이상값, 평일 시간대 패턴, 일정–CO₂ 반응 시간, 일정과 어긋난 날.

  python diagnose.py   → report/diag_*.csv, report/fig_*.png, report/diag.json
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

import prep
from viz import SPACE_KO, plt

OUT = Path(__file__).resolve().parent / "report"
OUT.mkdir(exist_ok=True)

grid, stats, removed, raw, sched, meta = prep.build()
cleaned, _ = prep.clean(raw)
res = {}

# ---------------------------------------------------------------- 1. 결측률·제거 이상값
period = {s: c["report_min"] for s, c in meta["spaces"].items()}
minutes = (pd.Timestamp(meta["end"]) - pd.Timestamp(meta["start"])).total_seconds() / 60 + 1
rows = []
for space, g in grid.groupby(level="space"):
    r = {"공간": SPACE_KO[space], "기록 주기(분)": period[space]}
    n_raw = (raw.space == space).sum()
    r["메시지 수신률"] = n_raw / (minutes / period[space])
    for v in prep.VARS:
        before = g[v].isna() | g[v + "_imp"]
        r[f"{v} 5분칸 결측(보간 전)"] = before.mean()
        r[f"{v} 5분칸 결측(보간 후)"] = g[v].isna().mean()
        r[f"{v} 제거 이상값"] = int(removed.loc[space, v]) if space in removed.index else 0
    rows.append(r)
miss = pd.DataFrame(rows)
miss.to_csv(OUT / "diag_missing.csv", index=False)
res["missing"] = miss.to_dict("records")

# 결측 구간 길이 분포 (5분 칸, 보간 전)
gap_rows = []
for space, g in grid.groupby(level="space"):
    na = (g["co2"].isna() | g["co2_imp"]).values
    runs, cur = [], 0
    for x in na:
        if x:
            cur += 1
        elif cur:
            runs.append(cur)
            cur = 0
    runs = np.array(runs) * 5
    gap_rows.append({"공간": SPACE_KO[space], "결측 구간 수": len(runs),
                     "15분 이하(보간)": int((runs <= 15).sum()), "15분 초과(결측 유지)": int((runs > 15).sum()),
                     "최장(분)": int(runs.max()) if len(runs) else 0})
gaps = pd.DataFrame(gap_rows)
gaps.to_csv(OUT / "diag_gaps.csv", index=False)
res["gaps"] = gap_rows

# ---------------------------------------------------------------- 2. 평일 시간대별 평균 패턴 (학습 기간)
g = grid.reset_index()
tr = g[(g.split == "train") & (g.time.dt.weekday < 5)].copy()
tr["hour"] = tr.time.dt.hour + tr.time.dt.minute / 60
fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
for space, d in tr.groupby("space"):
    prof = d.groupby("hour")[["co2", "temperature"]].mean()
    axes[0].plot(prof.index, prof.co2, label=SPACE_KO[space])
    axes[1].plot(prof.index, prof.temperature, label=SPACE_KO[space])
for ax, title, unit in ((axes[0], "평일 시간대별 CO₂ 평균", "ppm"), (axes[1], "평일 시간대별 온도 평균", "℃")):
    for x, lab in ((9, "수업 시작"), (12, "점심"), (13, "오후 수업"), (18, "퇴실")):
        ax.axvline(x, color="gray", lw=0.6, ls="--")
    ax.set_title(title)
    ax.set_xlabel("시각")
    ax.set_ylabel(unit)
    ax.set_xticks(range(0, 25, 3))
    ax.grid(alpha=0.3)
axes[0].axhline(1000, color="red", lw=0.8, ls=":", label="환기 권고 1,000ppm")
axes[0].legend(fontsize=8)
fig.tight_layout()
fig.savefig(OUT / "fig_weekday_profile.png", dpi=130)
plt.close(fig)
prof_tab = tr.pivot_table(index=tr.time.dt.hour, columns="space", values="co2", aggfunc="mean").round(0)
prof_tab.to_csv(OUT / "diag_profile_co2.csv")

# ---------------------------------------------------------------- 3. 일정–CO₂ 반응 시간 (1분 원본, 정제 후)
co2 = {s: d.set_index("time")["co2"].sort_index() for s, d in cleaned.groupby("space")}


def first_cross(series, t0, base, delta, within=45, up=True):
    w = series[(series.index > t0) & (series.index <= t0 + pd.Timedelta(minutes=within))].dropna()
    hit = w[w >= base + delta] if up else w[w <= base - delta]
    return (hit.index[0] - t0).total_seconds() / 60 if len(hit) else np.nan


lag_rows = []
for space in ("classroom_a", "classroom_b"):
    s = co2[space]
    for day in pd.date_range(meta["start"], meta["end"], freq="D"):
        if day.weekday() >= 5:
            continue
        for label, hh, up, delta in (("수업 시작 09:00", 9, True, 50), ("점심 시작 12:00", 12 - 10 / 60, False, 100),
                                     ("오후 수업 13:00", 13, True, 50), ("퇴실 18:00", 18 - 10 / 60, False, 100)):
            t0 = day + pd.Timedelta(hours=hh)
            before = s[(s.index > t0 - pd.Timedelta(minutes=10)) & (s.index <= t0)].dropna()
            if before.empty:
                continue
            lag = first_cross(s, t0, before.mean(), delta, up=up)
            lag_rows.append(dict(space=space, day=day.date(), event=label, base=before.mean(), lag_min=lag))
for space, kind, delta in (("meeting", "booking", 50), ("lounge", "lunch", 50)):
    s = co2[space]
    for _, r in sched[(sched.space == space) & (sched.kind == kind)].iterrows():
        before = s[(s.index > r.start - pd.Timedelta(minutes=10)) & (s.index <= r.start)].dropna()
        if before.empty:
            continue
        lag_rows.append(dict(space=space, day=r.start.date(), start=r.start, event=f"{'회의 시작' if kind == 'booking' else '라운지 점심'}",
                             base=before.mean(), lag_min=first_cross(s, r.start, before.mean(), delta, within=30)))
lags = pd.DataFrame(lag_rows)
lag_sum = (lags.groupby(["space", "event"]).lag_min
           .agg(사건수="size", 반응없음=lambda x: int(x.isna().sum()), 중앙값="median",
                p25=lambda x: x.quantile(.25), p75=lambda x: x.quantile(.75)).round(1).reset_index())
lag_sum["space"] = lag_sum.space.map(SPACE_KO)
lag_sum.to_csv(OUT / "diag_reaction_lag.csv", index=False)
res["reaction_lag"] = lag_sum.to_dict("records")

# ---------------------------------------------------------------- 4. 일정과 실제가 어긋난 날·회의
truth = pd.read_csv(prep.DATA / "truth.csv.gz", parse_dates=["time"], low_memory=False)
mis = []
for space in ("classroom_a", "classroom_b"):
    s = co2[space]
    for day in pd.date_range(meta["start"], meta["end"], freq="D"):
        if day.weekday() >= 5:
            continue
        w = s[(s.index >= day + pd.Timedelta(hours=9)) & (s.index < day + pd.Timedelta(hours=18))].dropna()
        if w.empty:
            continue
        rise = w.quantile(0.9) - w.quantile(0.05)
        t = truth[(truth.space == space) & (truth.time >= day + pd.Timedelta(hours=9)) & (truth.time < day + pd.Timedelta(hours=18))]
        if rise < 150:
            mis.append(dict(공간=SPACE_KO[space], 날짜=day.date(), 일정="수업", 관측=f"CO₂ 상승 {rise:.0f}ppm뿐 (수업 없음 추정)",
                            정답_실제인원최대=int(t.occupants.max())))
mtg = lags[(lags.space == "meeting")]
noshow = mtg[mtg.lag_min.isna()]
for _, r in noshow.iterrows():
    b = sched[(sched.space == "meeting") & (sched.start.dt.date == r.day)]
    mis.append(dict(공간="회의실", 날짜=r.day, 일정="예약", 관측="예약 시작 후 30분 동안 CO₂ +50ppm 미만 (노쇼 추정)",
                    정답_실제인원최대=None))
mis = pd.DataFrame(mis)
mis.to_csv(OUT / "diag_mismatch.csv", index=False)
res["mismatch_count"] = mis.groupby("공간").size().to_dict() if len(mis) else {}

# 회의실 노쇼 추정이 정답과 얼마나 맞나
mt = truth[truth.space == "meeting"].set_index("time").occupants
mb = lags[lags.space == "meeting"].copy()
mb["truth_noshow"] = [mt[(mt.index > t) & (mt.index <= t + pd.Timedelta(minutes=30))].max() == 0 for t in mb.start]
mb["est_noshow"] = mb.lag_min.isna()
cm = pd.crosstab(mb.truth_noshow.map({True: "실제 노쇼", False: "실제 사용"}),
                 mb.est_noshow.map({True: "노쇼로 추정", False: "사용으로 추정"}))
cm.to_csv(OUT / "diag_meeting_noshow_cm.csv")
res["meeting_noshow_cm"] = cm.to_dict()
# 노쇼로 잘못 추정한 예약: 직전 회의 직후라 CO₂가 이미 높았나?
fp = mb[mb.est_noshow & ~mb.truth_noshow]
res["meeting_fp_base_median"] = float(fp.base.median()) if len(fp) else None
res["meeting_tn_base_median"] = float(mb[mb.est_noshow & mb.truth_noshow].base.median())

# ---------------------------------------------------------------- 5. 하루 예시 그래프 (일정 vs CO₂)
day = pd.Timestamp(meta["split"]["val"][0]) + pd.Timedelta(days=1)
fig, axes = plt.subplots(4, 1, figsize=(12, 9), sharex=True)
for ax, space in zip(axes, ["classroom_a", "classroom_b", "meeting", "lounge"]):
    s = co2[space][(co2[space].index >= day) & (co2[space].index < day + pd.Timedelta(days=1))]
    ax.plot(s.index, s.values, lw=0.9, label="CO₂ (정제 후 1분)")
    for _, r in sched[(sched.space == space) & (sched.start >= day) & (sched.start < day + pd.Timedelta(days=1))].iterrows():
        ax.axvspan(r.start, r.end, color="orange", alpha=0.15)
    tt = truth[(truth.space == space) & (truth.time >= day) & (truth.time < day + pd.Timedelta(days=1))]
    ax2 = ax.twinx()
    ax2.plot(tt.time, tt.occupants, color="green", lw=0.7, alpha=0.6)
    ax2.set_ylabel("실제 인원", color="green", fontsize=8)
    ax.set_ylabel("ppm")
    ax.set_title(f"{SPACE_KO[space]} — 주황: 일정, 초록: 실제 인원(정답)", fontsize=9)
    ax.axhline(1000, color="red", lw=0.6, ls=":")
fig.tight_layout()
fig.savefig(OUT / "fig_day_example.png", dpi=120)
plt.close(fig)

res["norm_stats_train"] = stats
res["grid_rows"] = int(len(grid))
(OUT / "diag.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str))
pd.set_option("display.width", 200)
print(miss.round(3).to_string(index=False))
print(gaps.to_string(index=False))
print(lag_sum.to_string(index=False))
print(mis.to_string(index=False))
print(cm.to_string()); print("오추정 예약의 시작 직전 CO2 중앙값", res["meeting_fp_base_median"], "/ 진짜 노쇼", res["meeting_tn_base_median"])
print(prof_tab.to_string())
