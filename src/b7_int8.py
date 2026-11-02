"""심화 7-3: TensorRT INT8 변환 — 정확도 손실과 속도 이득을 FP32·FP16과 비교.

  STAGE=prep ONLY=fc_tcn python b7_int8.py              (CPU) 보정·시험 입력, PyTorch CPU FP32 기준 출력 → report/b7_int8_inputs_<모델>.npz
  STAGE=gpu  ONLY=fc_tcn MODE=int8 python b7_int8.py    (GPU) 엔진 하나 만들고 정확도·속도 측정
  STAGE=merge python b7_int8.py                         결과 합치기
  모델: fc_tcn, fc_gru, fc_lstm, ae

  - ONNX: optimize.py가 만든 models/<모델>.onnx를 그대로 쓴다
  - INT8 보정(calibration): 학습 기간 입력 1024개(무작위, seed 0), 엔트로피 보정(IInt8EntropyCalibrator2), 보정 묶음 32.
    INT8로 못 도는 층은 TensorRT가 FP16/FP32로 남긴다(FP16 플래그도 켬)
  - 정확도 기준은 PyTorch CPU FP32 출력
      예측 모델  시험 기간 전체 샘플의 CO₂ 최대차·60분 평균차(ppm)와 60분 CO₂ MAE(ppm, 관측 정답)
      AE        시험 기간 창의 이상 점수(최근 15분 재구성 오차) 차이와 임계값 판정이 바뀐 비율
  - 속도: optimize.py와 같은 조건(워밍업 50회 + 500회, 배치 1·4, 평균·p95). 입출력 복사 포함
  메모리: 17:19 첫 시도(한 프로세스에서 PyTorch CUDA + 엔진 3개)에 보드 가용 메모리가 647MB까지 떨어져 중단했다.
  그래서 GPU 단계는 PyTorch·pandas 없이 TensorRT + cuda-python만 쓰고, 엔진 하나당 프로세스 하나, 작업 공간 64MB로 줄였다.
"""
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
OUT, MD = ROOT / "report", ROOT / "models"
WARM, ITERS = 50, 500
TRT_MAX_BATCH = 64
CALIB_N, CALIB_BS = 1024, 32
STAGE = os.environ["STAGE"]
ONLY = os.getenv("ONLY")


def prep_stage():
    import torch
    import dataset
    import models
    import prep
    torch.set_num_threads(4)
    grid, stats, *_ = prep.build()
    fault = prep.fault_mask(grid)
    std = np.array([stats[v]["std"] for v in dataset.TARGETS], np.float32)
    L = json.loads((OUT / "forecast_best.json").read_text())["L"]
    fe = dataset.features(grid, stats)
    rng = np.random.default_rng(0)
    extra = {}
    if ONLY.startswith("fc_"):
        kind = ONLY[3:]
        ck = torch.load(MD / f"forecast_{kind}_L{L}.pt", weights_only=False)
        model = models.make(kind, ck["n_in"]); model.load_state_dict(ck["state"]); model.eval()
        Xtr, Ftr, *_ = dataset.samples(fe, stats, L, "train", fault)
        pick = rng.choice(len(Xtr), CALIB_N, replace=False)
        calib = (Xtr[pick], Ftr[pick])
        Xte, Fte, Yte, Mte, _ = dataset.samples(fe, stats, L, "test", fault, min_hist_obs=0.0)
        test = (Xte, Fte)
        extra = dict(Y=Yte, M=Mte, std=std)
    else:   # ae: anomaly.py의 ae_windows와 같은 창(1시간 × CO₂·온도·습도, 조건 = 시각·요일·일정·공간)
        W = 12
        ak = torch.load(MD / "ae.pt", weights_only=False)
        model = models.AE(ak["n_in"], ak["n_cond"]); model.load_state_dict(ak["state"]); model.eval()
        parts = []
        for s in dataset.SPACES:
            f = fe[s]["feat"]
            idx = np.arange(W - 1, len(f))
            win = np.stack([f[i - W + 1:i + 1][:, [0, 2, 4]] for i in idx]).reshape(len(idx), -1)
            miss = np.stack([f[i - W + 1:i + 1][:, [1, 3, 5]] for i in idx]).reshape(len(idx), -1)
            parts.append((win.astype(np.float32), miss.astype(np.float32), f[idx, 6:].astype(np.float32), fe[s]["split"][idx]))
        X, Mm, C, sp = (np.concatenate([p[k] for p in parts]) for k in range(4))
        tr = np.where((sp == "train") & (Mm.mean(1) < 0.2))[0]
        pick = rng.choice(tr, CALIB_N, replace=False)
        calib = (X[pick], C[pick])
        te = sp == "test"
        test = (X[te], C[te])
        extra = dict(M=Mm[te], thr=np.float32(json.loads((OUT / "anomaly_meta.json").read_text())["ae_threshold"]))
    with torch.no_grad():
        ref = np.concatenate([model(*[torch.from_numpy(x[i:i + 4096]) for x in test]).numpy() for i in range(0, len(test[0]), 4096)])
    np.savez(OUT / f"b7_int8_inputs_{ONLY}.npz", c0=calib[0], c1=calib[1], t0=test[0], t1=test[1], ref=ref, **extra)
    print(ONLY, "입력 저장", [a.shape for a in test], flush=True)


def accuracy(out, d):
    ref = d["ref"]
    if ONLY.startswith("fc_"):
        std = d["std"]
        dd = np.abs(out - ref)[..., 0] * std[0]
        m = d["M"][:, -1, 0] > 0
        mae60 = float((np.abs(out[:, -1, 0] - d["Y"][:, -1, 0]) * std[0])[m].mean())
        return dict(CO2_최대차_ppm=round(float(dd.max()), 3), CO2_60분_평균차_ppm=round(float(dd[:, -1].mean()), 4),
                    CO2_60분_MAE_ppm=round(mae60, 3), 샘플수=len(out))
    x, M, thr = d["t0"], d["M"], float(d["thr"])

    def score(r):
        return (((r - x) ** 2 * (1 - M)).reshape(len(x), 12, 3)[:, -3:, :].mean(1)).sum(1)
    s, s0 = score(out), score(ref)
    return dict(재구성_최대차=round(float(np.abs(out - ref).max()), 4), 점수_평균차=round(float(np.abs(s - s0).mean()), 5),
                점수_상대차_중앙값=round(float(np.median(np.abs(s - s0) / np.maximum(s0, 1e-6))), 4),
                임계값=thr, 판정바뀐비율=round(float(((s > thr) != (s0 > thr)).mean()), 5), 기준_이상판정수=int((s0 > thr).sum()), 창수=len(out))


def gpu_stage():
    import psutil
    import tensorrt as trt
    from cuda.bindings import runtime as cudart
    MODE = os.environ["MODE"]
    d = dict(np.load(OUT / f"b7_int8_inputs_{ONLY}.npz"))
    calib, test = (d["c0"], d["c1"]), (d["t0"], d["t1"])
    names = ["in0", "in1"]
    avail0 = psutil.virtual_memory().available

    def malloc(n):
        err, p = cudart.cudaMalloc(n)
        assert err == cudart.cudaError_t.cudaSuccess, err
        return p

    class Calib(trt.IInt8EntropyCalibrator2):
        def __init__(self):
            trt.IInt8EntropyCalibrator2.__init__(self)
            self.i = 0
            self.buf = [malloc(CALIB_BS * a[0].nbytes) for a in calib]

        def get_batch_size(self):
            return CALIB_BS

        def get_batch(self, nm):
            if self.i + CALIB_BS > len(calib[0]):
                return None
            for b, a in zip(self.buf, calib):
                x = np.ascontiguousarray(a[self.i:self.i + CALIB_BS])
                cudart.cudaMemcpy(b, x.ctypes.data, x.nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)
            self.i += CALIB_BS
            return [self.buf[names.index(n)] for n in nm]

        def read_calibration_cache(self):
            return None

        def write_calibration_cache(self, cache):
            (MD / f"{ONLY}_int8.calib").write_bytes(bytes(cache))

    logger = trt.Logger(trt.Logger.ERROR)
    builder = trt.Builder(logger)
    net = builder.create_network(0)
    parser = trt.OnnxParser(net, logger)
    assert parser.parse((MD / f"{ONLY}.onnx").read_bytes())
    cfg = builder.create_builder_config()
    cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 64 << 20)
    prof = builder.create_optimization_profile()
    for n, a in zip(names, test):
        prof.set_shape(n, (1, *a.shape[1:]), (4, *a.shape[1:]), (TRT_MAX_BATCH, *a.shape[1:]))
    cfg.add_optimization_profile(prof)
    if MODE in ("fp16", "int8"):
        cfg.set_flag(trt.BuilderFlag.FP16)
    if MODE == "int8":
        cfg.set_flag(trt.BuilderFlag.INT8)
        cp = builder.create_optimization_profile()
        for n, a in zip(names, test):
            cp.set_shape(n, *[(CALIB_BS, *a.shape[1:])] * 3)
        cfg.set_calibration_profile(cp)
        cfg.int8_calibrator = Calib()
    t = time.time()
    plan = builder.build_serialized_network(net, cfg)
    assert plan is not None, f"{MODE} 엔진 생성 실패"
    build_sec = time.time() - t
    eng_path = MD / f"{ONLY}_{MODE}_b7.engine"
    eng_path.write_bytes(bytes(plan))
    del builder, cfg, net, parser
    engine = trt.Runtime(logger).deserialize_cuda_engine(plan)
    ctx = engine.create_execution_context()
    _, stream = cudart.cudaStreamCreate()
    out_shape1 = (TRT_MAX_BATCH,) + tuple(d["ref"].shape[1:])
    bufs = [malloc(TRT_MAX_BATCH * a[0].nbytes) for a in test] + [malloc(int(np.prod(out_shape1)) * 4)]
    for n, b in zip(names + ["out"], bufs):
        ctx.set_tensor_address(n, b)

    def run(*xs):
        bs = len(xs[0])
        if bs > TRT_MAX_BATCH:
            return np.concatenate([run(*(x[i:i + TRT_MAX_BATCH] for x in xs)) for i in range(0, bs, TRT_MAX_BATCH)])
        for n, b, x in zip(names, bufs, xs):
            x = np.ascontiguousarray(x)
            ctx.set_input_shape(n, x.shape)
            cudart.cudaMemcpyAsync(b, x.ctypes.data, x.nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, stream)
        ctx.execute_async_v3(stream)
        out = np.empty((bs,) + out_shape1[1:], np.float32)
        cudart.cudaMemcpyAsync(out.ctypes.data, bufs[-1], out.nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, stream)
        cudart.cudaStreamSynchronize(stream)
        return out

    out = run(*test)
    acc = dict(모델=ONLY, 실행=f"TensorRT {MODE.upper()}", **accuracy(out, d))
    rows = []
    for bs in (1, 4):
        x = tuple(a[:bs] for a in test)
        for _ in range(WARM):
            run(*x)
        ts = []
        for _ in range(ITERS):
            t = time.perf_counter()
            run(*x)
            ts.append((time.perf_counter() - t) * 1000)
        rows.append(dict(모델=ONLY, 실행=f"TensorRT {MODE.upper()}", 배치=bs, 평균_ms=round(float(np.mean(ts)), 3),
                         p95_ms=round(float(np.percentile(ts, 95)), 3), 엔진_KB=round(eng_path.stat().st_size / 1024, 1),
                         빌드_초=round(build_sec, 1), RSS_MB=round(psutil.Process().memory_info().rss / 2**20, 1),
                         보드가용감소_MB=round((avail0 - psutil.virtual_memory().available) / 2**20, 1)))
    res = dict(acc=acc, lat=rows)
    (OUT / f"b7_int8_{ONLY}_{MODE}.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print(res, flush=True)


def merge_stage():
    import pandas as pd
    lat, acc = [], []
    for f in sorted(OUT.glob("b7_int8_*_*.json")):
        r = json.loads(f.read_text())
        lat += r["lat"]; acc.append(r["acc"])
    pd.DataFrame(lat).to_csv(OUT / "b7_int8_latency.csv", index=False)
    pd.DataFrame(acc).to_csv(OUT / "b7_int8_accuracy.csv", index=False)
    meta = dict(power_mode=subprocess.run(["nvpmodel", "-q"], capture_output=True, text=True).stdout.split("\n")[0],
                clocks_locked=False, warmup=WARM, iters=ITERS, calib_n=CALIB_N, calib_batch=CALIB_BS, workspace_mb=64,
                tensorrt=__import__("tensorrt").__version__, 기준="PyTorch CPU FP32 출력")
    (OUT / "b7_int8_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1))
    pd.set_option("display.width", 220)
    print(pd.DataFrame(lat).to_string(index=False)); print(pd.DataFrame(acc).to_string(index=False))


dict(prep=prep_stage, gpu=gpu_stage, merge=merge_stage)[STAGE]()
