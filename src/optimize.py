"""과제 7: 엣지 최적화 — PyTorch CPU / PyTorch GPU / ONNX Runtime(CPU) / TensorRT FP16.

  python optimize.py   → report/optimize_*.csv, models/*.onnx, models/*.engine

대상 모델
  - 예측: GRU·LSTM·TCN (같은 과거 길이) — 순환망과 합성곱의 변환 이득 비교
  - 이상 탐지: 오토인코더(models/ae.pt)
  - 분류: 환기 선제 경보 MLP(여기서 학습. 과제 5의 GBDT는 TensorRT로 바꿀 수 없어 같은 특징으로 신경망 판을 만든다)
측정: 배치 1(한 공간)·배치 4(네 공간), 워밍업 50회 후 500회, 평균과 느린 쪽 5%(p95).
"""
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
import torch
from torch import nn

import dataset
import models
import prep

ROOT = Path(__file__).resolve().parent
OUT, MD = ROOT / "report", ROOT / "models"
torch.set_num_threads(4)
WARM, ITERS = 50, 500
TRT_MAX_BATCH = 64   # 공간 수(3~4)보다 넉넉하게. 4096으로 잡으면 실행 메모리를 200MB 넘게 잡아 보드에서 실패했다
grid, stats, *_ = prep.build()
fault = prep.fault_mask(grid)
best = json.loads((OUT / "forecast_best.json").read_text())
L = best["L"]
STD = np.array([stats[v]["std"] for v in dataset.TARGETS], np.float32)


# ---------------------------------------------------------------- 대상 모델과 시험 입력
def load_forecaster(kind):
    ck = torch.load(MD / f"forecast_{kind}_L{L}.pt", weights_only=False)
    m = models.make(kind, ck["n_in"])
    m.load_state_dict(ck["state"])
    return m.eval()


fe = dataset.features(grid, stats)
Xte, Fte, Yte, Mte, Kte = dataset.samples(fe, stats, L, "test", fault, min_hist_obs=0.0)
Xte, Fte = Xte[:2000], Fte[:2000]

from models import AE  # noqa: E402

ck = torch.load(MD / "ae.pt", weights_only=False)
ae = AE(ck["n_in"], ck["n_cond"])
ae.load_state_dict(ck["state"])
ae.eval()
rng = np.random.default_rng(0)
Xae = rng.normal(0, 1, (2000, ck["n_in"])).astype(np.float32)
Cae = rng.normal(0, 1, (2000, ck["n_cond"])).astype(np.float32)


class VentMLP(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_vent_mlp():
    import state
    d = state.table()
    state.occupancy(d, np.random.default_rng(7))  # 특징 목록(X_OCC) 채우기
    feats = state.X_OCC + ["sched_now", "sched_next30"]
    base = d[d.vent_ok & ~d.fault].dropna(subset=feats)
    tr, te = base[base.split == "train"], base[base.split == "test"]
    mu, sd = tr[feats].mean().values, tr[feats].std().replace(0, 1).values
    X = torch.tensor(((tr[feats].values - mu) / sd), dtype=torch.float32)
    y = torch.tensor(tr.vent_label.values, dtype=torch.float32)
    torch.manual_seed(0)
    m = VentMLP(len(feats))
    opt = torch.optim.Adam(m.parameters(), 1e-3)
    pos_w = torch.tensor((1 - y.mean()) / y.mean())
    for _ in range(40):
        perm = torch.randperm(len(X))
        for i in range(0, len(X), 512):
            b = perm[i:i + 512]
            loss = nn.functional.binary_cross_entropy_with_logits(m(X[b]), y[b], pos_weight=pos_w)
            opt.zero_grad(); loss.backward(); opt.step()
    m.eval()
    Xt = ((te[feats].values - mu) / sd).astype(np.float32)
    from sklearn.metrics import f1_score, recall_score
    with torch.no_grad():
        p = (torch.sigmoid(m(torch.from_numpy(Xt))) >= 0.5).int().numpy()
    q = dict(recall=float(recall_score(te.vent_label, p)), macro_f1=float(f1_score(te.vent_label, p, average="macro")))
    torch.save(dict(state=m.state_dict(), feats=feats, mu=mu, sd=sd, n=len(feats), quality=q), MD / "vent_mlp.pt")
    return m, Xt, q


if (MD / "vent_mlp.pt").exists() and (OUT / "vent_mlp_X.npy").exists():   # 모델별로 따로 실행할 때 다시 학습하지 않는다
    vk = torch.load(MD / "vent_mlp.pt", weights_only=False)
    vent = VentMLP(vk["n"]); vent.load_state_dict(vk["state"]); vent.eval()
    Xvent, vent_q = np.load(OUT / "vent_mlp_X.npy"), vk["quality"]
else:
    vent, Xvent, vent_q = train_vent_mlp()
    np.save(OUT / "vent_mlp_X.npy", Xvent[:2000])

TARGETS = {
    "예측 GRU": (load_forecaster("gru"), (Xte, Fte)),
    "예측 LSTM": (load_forecaster("lstm"), (Xte, Fte)),
    "예측 TCN": (load_forecaster("tcn"), (Xte, Fte)),
    "이상 탐지 AE": (ae, (Xae, Cae)),
    "분류 환기경보 MLP": (vent, (Xvent[:2000],)),
}
SLUG = {"예측 GRU": "fc_gru", "예측 LSTM": "fc_lstm", "예측 TCN": "fc_tcn", "이상 탐지 AE": "ae", "분류 환기경보 MLP": "vent"}


# ---------------------------------------------------------------- 백엔드
def bench(fn, inputs):
    for _ in range(WARM):
        fn(*inputs)
    ts = []
    for _ in range(ITERS):
        t = time.perf_counter()
        fn(*inputs)
        ts.append((time.perf_counter() - t) * 1000)
    return float(np.mean(ts)), float(np.percentile(ts, 95))


def torch_fn(model, dev):
    model = model.to(dev)

    def f(*xs):
        with torch.no_grad():
            out = model(*[torch.from_numpy(x).to(dev) for x in xs])
            if dev == "cuda":
                torch.cuda.synchronize()
            return out.cpu().numpy()
    return f


def export_onnx(name, model, inputs):
    path = MD / f"{SLUG[name]}.onnx"
    model = model.cpu().eval()
    names = [f"in{i}" for i in range(len(inputs))]
    torch.onnx.export(model, tuple(torch.from_numpy(x[:1]) for x in inputs), str(path), input_names=names, output_names=["out"],
                      dynamic_axes={n: {0: "batch"} for n in names + ["out"]}, opset_version=17, dynamo=False)
    return path, names


def ort_fn(path, names):
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.intra_op_num_threads = 4
    sess = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
    return lambda *xs: sess.run(None, {n: x for n, x in zip(names, xs)})[0]


def trt_build(path, names, inputs, fp16=True):
    import tensorrt as trt
    logger = trt.Logger(trt.Logger.ERROR)
    builder = trt.Builder(logger)
    net = builder.create_network(0)
    parser = trt.OnnxParser(net, logger)
    assert parser.parse(path.read_bytes()), [parser.get_error(i) for i in range(parser.num_errors)]
    cfg = builder.create_builder_config()
    cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 256 << 20)
    if fp16:
        cfg.set_flag(trt.BuilderFlag.FP16)
    prof = builder.create_optimization_profile()
    for n, x in zip(names, inputs):
        shp = x.shape[1:]
        prof.set_shape(n, (1, *shp), (4, *shp), (TRT_MAX_BATCH, *shp))
    cfg.add_optimization_profile(prof)
    plan = builder.build_serialized_network(net, cfg)
    eng_path = MD / f"{path.stem}{'_fp16' if fp16 else ''}.engine"
    eng_path.write_bytes(bytes(plan))
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(plan)
    ctx = engine.create_execution_context()
    stream = torch.cuda.Stream()

    def f(*xs):
        if len(xs[0]) > TRT_MAX_BATCH:  # 비교용 큰 입력은 나눠서
            return np.concatenate([f(*(x[i:i + TRT_MAX_BATCH] for x in xs)) for i in range(0, len(xs[0]), TRT_MAX_BATCH)])
        ins = [torch.from_numpy(np.ascontiguousarray(x)).cuda() for x in xs]
        for n, t in zip(names, ins):
            ctx.set_input_shape(n, tuple(t.shape))
            ctx.set_tensor_address(n, t.data_ptr())
        out = torch.empty(tuple(ctx.get_tensor_shape("out")), device="cuda", dtype=torch.float32)
        ctx.set_tensor_address("out", out.data_ptr())
        ctx.execute_async_v3(stream.cuda_stream)
        stream.synchronize()
        return out.cpu().numpy()
    return f, eng_path.stat().st_size


class Power:
    """tegrastats로 측정 중 보드 전체 전력(VDD_IN)·GPU 사용률을 1초마다 기록."""

    def __enter__(self):
        self.p = subprocess.Popen(["tegrastats", "--interval", "500"], stdout=subprocess.PIPE, text=True)
        self.lines = []
        return self

    def __exit__(self, *a):
        self.p.terminate()
        self.lines = self.p.communicate(timeout=5)[0].splitlines()

    def summary(self):
        import re
        pw = [int(m.group(1)) for line in self.lines for m in [re.search(r"VDD_IN (\d+)mW", line)] if m]
        gpu = [int(m.group(1)) for line in self.lines for m in [re.search(r"GR3D_FREQ (\d+)%", line)] if m]
        return dict(power_mW=round(float(np.mean(pw)), 0) if pw else None, gpu_util=round(float(np.mean(gpu)), 1) if gpu else None)


# ---------------------------------------------------------------- 실행
proc = psutil.Process()
rows, diffs = [], []
ONLY = os.getenv("ONLY")   # 모델 하나만(fc_gru, fc_lstm, fc_tcn, ae, vent). GPU 메모리를 모델마다 새 프로세스로 비운다
for name, (model, inputs) in TARGETS.items():
    if ONLY and SLUG[name] != ONLY:
        continue
    ref = torch_fn(model, "cpu")(*inputs)                 # 기준: PyTorch CPU FP32 출력
    path, names = export_onnx(name, model, inputs)
    backends = {"PyTorch CPU": lambda: torch_fn(model, "cpu"), "PyTorch GPU": lambda: torch_fn(model, "cuda"),
                "ONNX Runtime CPU": lambda: ort_fn(path, names)}
    eng_size = None
    for bname, make in list(backends.items()) + [("TensorRT FP16", None)]:
        if bname == "TensorRT FP16":
            model.cpu()
            torch.cuda.empty_cache()   # PyTorch가 잡은 GPU 메모리를 돌려준 뒤 엔진을 만든다
            fn, eng_size = trt_build(path, names, inputs)
        else:
            fn = make()
        out = fn(*inputs)
        # 변환 전후 차이(원래 단위)
        if name.startswith("예측"):
            d = np.abs(out - ref)  # 정규화된 변화량 → ppm·℃
            diffs.append(dict(모델=name, 실행=bname, CO2_최대차_ppm=float((d[..., 0] * STD[0]).max()),
                              CO2_60분_평균차_ppm=float((d[:, -1, 0] * STD[0]).mean()), 온도_최대차_C=float((d[..., 1] * STD[1]).max())))
        elif name.startswith("분류"):
            diffs.append(dict(모델=name, 실행=bname, 판정_바뀐_비율=float(((out >= 0) != (ref >= 0)).mean()), 로짓_최대차=float(np.abs(out - ref).max())))
        else:
            diffs.append(dict(모델=name, 실행=bname, 재구성_최대차=float(np.abs(out - ref).max())))
        for bs in (1, 4):
            x = tuple(i[:bs] for i in inputs)
            rss_before = proc.memory_info().rss
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            with Power() as pw:
                mean, p95 = bench(fn, x)
            rows.append(dict(모델=name, 실행=bname, 배치=bs, 평균_ms=round(mean, 3), p95_ms=round(p95, 3),
                             RSS_MB=round(proc.memory_info().rss / 2**20, 0), GPU_peak_MB=round(torch.cuda.max_memory_allocated() / 2**20, 1),
                             엔진_KB=round(eng_size / 1024, 1) if eng_size and bname.startswith("TensorRT") else None, **pw.summary()))
            print(rows[-1], flush=True)

lat = pd.DataFrame(rows)
dif = pd.DataFrame(diffs)
suffix = f"_{ONLY}" if ONLY else ""
lat.to_csv(OUT / f"optimize_latency{suffix}.csv", index=False)
dif.to_csv(OUT / f"optimize_diff{suffix}.csv", index=False)
cond = dict(power_mode=subprocess.run(["nvpmodel", "-q"], capture_output=True, text=True).stdout.split("\n")[0],
            clocks_locked=False, warmup=WARM, iters=ITERS, threads=4, torch=torch.__version__,
            tensorrt=__import__("tensorrt").__version__, vent_mlp_quality=vent_q, forecaster_L=L,
            note="jetson_clocks(클럭 고정)는 sudo가 필요해 쓰지 않았다. 팀원 작업이 같은 보드에서 돌고 있어 수치에 잡음이 있을 수 있다")
(OUT / "optimize_meta.json").write_text(json.dumps(cond, ensure_ascii=False, indent=1))
pd.set_option("display.width", 220)
print(lat.to_string(index=False)); print(dif.to_string(index=False)); print(cond)
