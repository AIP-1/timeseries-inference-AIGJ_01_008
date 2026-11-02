"""과제 6 보완: 직접 학습한 GRU에 예측 구간(10~90%)을 붙이고 적중률을 잰다. CPU.

GRU는 예측값 하나만 낸다. 검증 기간에서 "실제 - 예측" 오차의 10%·90% 지점을 예측 거리(5~60분)별로 구해
시험 기간 예측값에 더한 것을 10~90% 구간으로 쓴다(시험 기간은 구간을 만드는 데 쓰지 않는다).
같은 평가 지점(eval_points.csv)에서 Chronos와 같은 방식으로 적중률·구간 폭을 잰다.
함께 GRU 모델 파일 크기와, ONNX Runtime CPU로 올렸을 때 프로세스 메모리 증가를 잰다.
결과: report/b6_gru_interval.csv, report/b6_gru_interval_meta.json
"""
import json, os, resource
from pathlib import Path
import numpy as np, pandas as pd, psutil, torch
import dataset, models, prep

ROOT = Path(__file__).resolve().parent
OUT, MD = ROOT / "report", ROOT / "models"
torch.set_num_threads(4)
grid, stats, *_ = prep.build()
fault = prep.fault_mask(grid)
EVAL = pd.read_csv(OUT / "eval_points.csv", parse_dates=["time"])
STD = np.array([stats[v]["std"] for v in dataset.TARGETS], np.float32)
ck = torch.load(MD / "forecast_gru_L48.pt", weights_only=False)
gru = models.make(ck["kind"], ck["n_in"]); gru.load_state_dict(ck["state"]); gru.eval()
fe = dataset.features(grid, stats, use_sched=ck["use_sched"], use_space=ck["use_space"])

def pred(split):
    X, F, _, _, K = dataset.samples(fe, stats, ck["L"], split, fault, min_hist_obs=0.0)
    with torch.no_grad():
        P = np.concatenate([gru(torch.from_numpy(X[i:i+4096]), torch.from_numpy(F[i:i+4096])).numpy() for i in range(0, len(X), 4096)])
    rows = [pd.DataFrame(dict(space=K.space, time=K.time, h=(hi+1)*5, pred=K.y_now_co2.values + P[:, hi, 0]*STD[0])) for hi in range(12)]
    ev = EVAL[(EVAL.split == split) & (EVAL["var"] == "co2")]
    m = ev.merge(pd.concat(rows), on=["space", "time", "h"], how="inner")
    ix = pd.MultiIndex.from_arrays([m.space, m.time + pd.to_timedelta(m.h, "min")])
    m["true"] = grid["co2"].reindex(ix).values
    return m.dropna(subset=["pred", "true"])

va, te = pred("val"), pred("test")
va["res"] = va.true - va.pred
q = va.groupby("h").res.quantile([0.1, 0.9]).unstack()          # 예측 거리별 오차 10%·90% 지점(검증만)
te = te.join(q, on="h")
te["lo"], te["hi"] = te.pred + te[0.1], te.pred + te[0.9]
te["inside"] = (te.true >= te.lo) & (te.true <= te.hi)
te["width"] = te.hi - te.lo
te["err"] = (te.pred - te.true).abs()
rows = []
for h in (5, 15, 30, 60):
    d = te[te.h == h]
    rows.append(dict(h=h, MAE=round(d.err.mean(), 2), 구간적중률_10_90=round(d.inside.mean(), 3),
                     구간폭_중앙값=round(d.width.median(), 1), 검증_오차10=round(q.loc[h, 0.1], 1), 검증_오차90=round(q.loc[h, 0.9], 1), 지점수=len(d)))
res = pd.DataFrame(rows); res.to_csv(OUT / "b6_gru_interval.csv", index=False)

# 모델 크기와 ONNX Runtime CPU 메모리
import onnxruntime as ort
proc = psutil.Process()
r0 = proc.memory_info().rss
sess = ort.InferenceSession(str(MD / "fc_gru.onnx"), providers=["CPUExecutionProvider"])
names = [i.name for i in sess.get_inputs()]
X, F, *_ = dataset.samples(fe, stats, ck["L"], "test", fault, min_hist_obs=0.0)
for _ in range(20):
    sess.run(None, {names[0]: X[:1], names[1]: F[:1]})
r1 = proc.memory_info().rss
meta = dict(params=sum(p.numel() for p in gru.parameters()),
            pt_file_kb=round((MD / "forecast_gru_L48.pt").stat().st_size / 1024, 1),
            onnx_file_kb=round((MD / "fc_gru.onnx").stat().st_size / 1024, 1),
            ort_session_rss_mb=round((r1 - r0) / 2**20, 1))
(OUT / "b6_gru_interval_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1))
print(res.to_string(index=False)); print(meta)
