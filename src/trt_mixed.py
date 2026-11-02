"""과제 7 보완: 순환망(GRU·LSTM)을 TensorRT 혼합 정밀도로 변환한다.
순환층(48칸을 도는 반복 계산)만 FP32로 두고 나머지(입력·출력 층)는 FP16으로 둔다.
FP16 전체 변환에서 CO₂ 최대 차이가 1.03~1.10ppm으로 기준(1ppm)을 조금 넘었기 때문이다.

  python trt_mixed.py   → report/optimize_mixed.csv
"""
import json
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pandas as pd
import tensorrt as trt
import torch

import dataset
import prep

ROOT = Path(__file__).resolve().parent
OUT, MD = ROOT / "report", ROOT / "models"
TRT_MAX_BATCH, WARM, ITERS = 64, 50, 500
grid, stats, *_ = prep.build()
STD = np.array([stats[v]["std"] for v in dataset.TARGETS], np.float32)
L = json.loads((OUT / "forecast_best.json").read_text())["L"]
fe = dataset.features(grid, stats)
X, F, *_ = dataset.samples(fe, stats, L, "test", prep.fault_mask(grid), min_hist_obs=0.0)
X, F = X[:2000], F[:2000]
LOOP_TYPES = {trt.LayerType.RECURRENCE, trt.LayerType.ITERATOR, trt.LayerType.LOOP_OUTPUT, trt.LayerType.TRIP_LIMIT}


def build(onnx_path, mode):
    logger = trt.Logger(trt.Logger.ERROR)
    builder = trt.Builder(logger)
    net = builder.create_network(0)
    parser = trt.OnnxParser(net, logger)
    assert parser.parse(onnx_path.read_bytes())
    cfg = builder.create_builder_config()
    cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 256 << 20)
    n_fp32 = 0
    if mode in ("fp16", "mixed"):
        cfg.set_flag(trt.BuilderFlag.FP16)
    if mode == "mixed":
        cfg.set_flag(trt.BuilderFlag.PREFER_PRECISION_CONSTRAINTS)   # 강제(OBEY)는 TensorRT가 묶어 최적화한 RNN 노드와 충돌해 빌드가 실패했다
        # 반복(loop) 안에 있는 층과 반복 구조 자체를 FP32로 고정한다
        for i in range(net.num_layers):
            layer = net.get_layer(i)
            name = layer.name.lower()
            is_float = all(layer.get_output(j).dtype in (trt.float32, trt.float16) for j in range(layer.num_outputs))
            if is_float and (layer.type in LOOP_TYPES or "loop" in name or "gru" in name or "lstm" in name or "rnn" in name):
                layer.precision = trt.float32   # 정수(모양·인덱스) 계산 층은 건드리지 않는다
                for j in range(layer.num_outputs):
                    if layer.get_output(j).dtype in (trt.float32, trt.float16):
                        layer.set_output_type(j, trt.float32)
                n_fp32 += 1
    prof = builder.create_optimization_profile()
    for name, x in (("in0", X), ("in1", F)):
        prof.set_shape(name, (1, *x.shape[1:]), (4, *x.shape[1:]), (TRT_MAX_BATCH, *x.shape[1:]))
    cfg.add_optimization_profile(prof)
    plan = builder.build_serialized_network(net, cfg)
    engine = trt.Runtime(logger).deserialize_cuda_engine(plan)
    ctx = engine.create_execution_context()
    stream = torch.cuda.Stream()

    def f(x, fu):
        if len(x) > TRT_MAX_BATCH:
            return np.concatenate([f(x[i:i + TRT_MAX_BATCH], fu[i:i + TRT_MAX_BATCH]) for i in range(0, len(x), TRT_MAX_BATCH)])
        a, b = torch.from_numpy(np.ascontiguousarray(x)).cuda(), torch.from_numpy(np.ascontiguousarray(fu)).cuda()
        for n, t in (("in0", a), ("in1", b)):
            ctx.set_input_shape(n, tuple(t.shape))
            ctx.set_tensor_address(n, t.data_ptr())
        out = torch.empty(tuple(ctx.get_tensor_shape("out")), device="cuda", dtype=torch.float32)
        ctx.set_tensor_address("out", out.data_ptr())
        ctx.execute_async_v3(stream.cuda_stream)
        stream.synchronize()
        return out.cpu().numpy()
    return f, plan.nbytes, n_fp32, net.num_layers


def bench(fn, bs):
    x, fu = X[:bs], F[:bs]
    for _ in range(WARM):
        fn(x, fu)
    ts = []
    for _ in range(ITERS):
        t = time.perf_counter(); fn(x, fu); ts.append((time.perf_counter() - t) * 1000)
    return round(float(np.mean(ts)), 3), round(float(np.percentile(ts, 95)), 3)


rows = []
for slug, name in (("fc_gru", "예측 GRU"), ("fc_lstm", "예측 LSTM"), ("fc_tcn", "예측 TCN")):
    path = MD / f"{slug}.onnx"
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    ref = sess.run(None, {"in0": X, "in1": F})[0]          # FP32 기준(PyTorch CPU와 차이 0.0002ppm 이하)
    for mode in ("fp32", "fp16", "mixed"):
        fn, size, n32, nl = build(path, mode)
        d = np.abs(fn(X, F) - ref)
        m1, p1 = bench(fn, 1)
        m4, p4 = bench(fn, 4)
        rows.append(dict(모델=name, 정밀도={"fp32": "FP32", "fp16": "FP16", "mixed": "혼합(순환층 FP32)"}[mode],
                         FP32_고정_층=f"{n32}/{nl}" if mode == "mixed" else "", CO2_최대차_ppm=round(float((d[..., 0] * STD[0]).max()), 3),
                         CO2_60분_평균차_ppm=round(float((d[:, -1, 0] * STD[0]).mean()), 4), 온도_최대차_C=round(float((d[..., 1] * STD[1]).max()), 4),
                         배치1_평균_ms=m1, 배치1_p95_ms=p1, 배치4_평균_ms=m4, 배치4_p95_ms=p4, 엔진_KB=round(size / 1024, 1)))
        print(rows[-1], flush=True)
        del fn
        torch.cuda.empty_cache()
pd.DataFrame(rows).to_csv(OUT / "optimize_mixed.csv", index=False)
print(pd.DataFrame(rows).to_string(index=False))
