"""심화 7-4: 모델 4개(예측·이상 탐지 AE·재실 판정·환기 경보)를 한 프로세스에 함께 올렸을 때의 메모리.

  VARIANT=ort python b7_memory.py     (변형마다 새 프로세스: torch, ort, trt)

  torch  PyTorch CPU   예측 GRU(.pt) + AE(.pt)                 + 재실 occ.pkl + 환기 vent.pkl(sklearn)   ← 지금 서비스 방식
  ort    ONNX Runtime  예측 GRU(.onnx) + AE(.onnx)             + 재실 occ.pkl + 환기 vent.pkl
  trt    TensorRT FP16 예측 GRU·AE·환기 MLP(.engine, torch 없이 cuda-python으로 버퍼) + 재실 occ.pkl
재실 판정은 결정트리 앙상블(sklearn)이라 GPU로 바꿀 수 없어 모든 변형에서 CPU로 둔다.
각 단계(공통 라이브러리만 / 모델 적재 후 / 추론 20회 후)에서 프로세스 RSS, 보드 가용 메모리(free의 available),
cudaMemGetInfo의 여유량(Jetson은 CPU·GPU가 같은 메모리를 쓴다)을 잰다.
결과: report/b7_memory_<변형>.json
"""
import gc
import json
import os
import pickle
import time
from pathlib import Path

import numpy as np
import psutil

ROOT = Path(__file__).resolve().parent
OUT, MD = ROOT / "report", ROOT / "models"
VARIANT = os.environ["VARIANT"]
proc = psutil.Process()
rng = np.random.default_rng(0)


def snap(label, rec):
    gc.collect()
    time.sleep(1)
    s = dict(단계=label, RSS_MB=round(proc.memory_info().rss / 2**20, 1),
             보드가용_MB=round(psutil.virtual_memory().available / 2**20, 1))
    if VARIANT == "trt":
        from cuda.bindings import runtime as cudart
        _, free, total = cudart.cudaMemGetInfo()
        s["cuda_free_MB"] = round(free / 2**20, 1)
    rec.append(s)
    print(s, flush=True)


rec = []
snap("시작(numpy만)", rec)

# 입력 모양: optimize.py에서 쓴 것과 같다
L, N_IN = 48, 16   # fc_gru.onnx 입력 [배치, 48, 16]
occ = pickle.load(open(MD / "occ.pkl", "rb"))
vent_pkl = pickle.load(open(MD / "vent.pkl", "rb"))
x_occ = rng.normal(size=(4, len(occ["feats"])))
x_vent = rng.normal(size=(4, len(vent_pkl["feats"])))
x_fc = rng.normal(size=(4, L, N_IN)).astype(np.float32)
f_fc = rng.normal(size=(4, 12, 3)).astype(np.float32)

if VARIANT == "torch":
    import torch
    torch.set_num_threads(4)
    import models
    snap("라이브러리 import 후", rec)
    ck = torch.load(MD / "forecast_gru_L48.pt", weights_only=False)
    fc = models.make("gru", ck["n_in"]); fc.load_state_dict(ck["state"]); fc.eval()
    ak = torch.load(MD / "ae.pt", weights_only=False)
    ae = models.AE(ak["n_in"], ak["n_cond"]); ae.load_state_dict(ak["state"]); ae.eval()
    x_ae, c_ae = rng.normal(size=(4, ak["n_in"])).astype(np.float32), rng.normal(size=(4, ak["n_cond"])).astype(np.float32)

    def run():
        with torch.no_grad():
            fc(torch.from_numpy(x_fc), torch.from_numpy(f_fc)); ae(torch.from_numpy(x_ae), torch.from_numpy(c_ae))
        occ["model"].predict_proba(x_occ); vent_pkl["model"].predict_proba(x_vent)
elif VARIANT == "ort":
    import onnxruntime as ort
    snap("라이브러리 import 후", rec)
    so = ort.SessionOptions(); so.intra_op_num_threads = 4
    sfc = ort.InferenceSession(str(MD / "fc_gru.onnx"), so, providers=["CPUExecutionProvider"])
    sae = ort.InferenceSession(str(MD / "ae.onnx"), so, providers=["CPUExecutionProvider"])
    n_in, n_cond = (i.shape[1] for i in sae.get_inputs())
    x_ae, c_ae = rng.normal(size=(4, n_in)).astype(np.float32), rng.normal(size=(4, n_cond)).astype(np.float32)

    def run():
        sfc.run(None, dict(in0=x_fc, in1=f_fc)); sae.run(None, dict(in0=x_ae, in1=c_ae))
        occ["model"].predict_proba(x_occ); vent_pkl["model"].predict_proba(x_vent)
else:
    import tensorrt as trt
    from cuda.bindings import runtime as cudart
    snap("라이브러리 import 후", rec)
    logger = trt.Logger(trt.Logger.ERROR)
    rt = trt.Runtime(logger)
    _, stream = cudart.cudaStreamCreate()

    class Eng:
        def __init__(self, path):
            self.e = rt.deserialize_cuda_engine(path.read_bytes())
            self.ctx = self.e.create_execution_context()
            self.names = [self.e.get_tensor_name(i) for i in range(self.e.num_io_tensors)]
            self.bufs = {}

        def __call__(self, **xs):
            for n, x in xs.items():
                self.ctx.set_input_shape(n, x.shape)
            outs = {}
            for n in self.names:
                shape = xs[n].shape if n in xs else tuple(self.ctx.get_tensor_shape(n))
                nbytes = int(np.prod(shape)) * 4
                if n not in self.bufs:
                    self.bufs[n] = cudart.cudaMalloc(64 * nbytes // shape[0])[1]   # 배치 64까지
                self.ctx.set_tensor_address(n, self.bufs[n])
                if n in xs:
                    cudart.cudaMemcpyAsync(self.bufs[n], xs[n].ctypes.data, nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, stream)
                else:
                    outs[n] = (np.empty(shape, np.float32), nbytes)
            self.ctx.execute_async_v3(stream)
            for n, (o, nb) in outs.items():
                cudart.cudaMemcpyAsync(o.ctypes.data, self.bufs[n], nb, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, stream)
            cudart.cudaStreamSynchronize(stream)
            return outs

    efc, eae, eve = Eng(MD / "fc_gru_fp16.engine"), Eng(MD / "ae_fp16.engine"), Eng(MD / "vent_fp16.engine")
    a_in, a_cond = (eae.e.get_tensor_shape(n)[1] for n in ("in0", "in1"))
    v_in = eve.e.get_tensor_shape("in0")[1]
    x_ae, c_ae = rng.normal(size=(4, a_in)).astype(np.float32), rng.normal(size=(4, a_cond)).astype(np.float32)
    x_vm = rng.normal(size=(4, v_in)).astype(np.float32)

    def run():
        efc(in0=x_fc, in1=f_fc); eae(in0=x_ae, in1=c_ae); eve(in0=x_vm)
        occ["model"].predict_proba(x_occ)

snap("모델 적재 후", rec)
t = time.perf_counter()
for _ in range(20):
    run()
cyc = (time.perf_counter() - t) / 20 * 1000
snap("추론 20회 후", rec)
res = dict(변형=VARIANT, 단계=rec, 모델적재_RSS증가_MB=round(rec[-2]["RSS_MB"] - rec[1]["RSS_MB"], 1),
           전체_RSS_MB=rec[-1]["RSS_MB"], 보드가용_감소_MB=round(rec[0]["보드가용_MB"] - rec[-1]["보드가용_MB"], 1),
           한바퀴_ms=round(cyc, 2), 보드전체_MB=round(psutil.virtual_memory().total / 2**20, 1))
(OUT / f"b7_memory_{VARIANT}.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
print(res)
