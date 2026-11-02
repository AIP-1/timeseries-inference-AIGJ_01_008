"""과제 8: 상시 추론 서비스 v1.

매분 0초 직후에 공간마다 계산한다.
  1) 60분 예측(CO₂·온도, 5분 간격) + "몇 분 뒤 1,000ppm 초과" 경보
  2) 이상: 오토인코더 점수·원인 센서 + 규칙 경보(데이터 끊김·값 고정·범위 밖)
  3) 사용 여부, 환기 선제 경보, 빈 방 냉난방 가동
  4) 자기 성능 감시: 낸 예측을 저장했다가 실제 값이 들어오면(5~60분 뒤) 채점, 최근 6시간 MAE를 앞 시간별로 낸다.
     60분 앞 CO₂ MAE가 검증 MAE × 2 를 넘으면 경보 forecast_degraded (상태는 logs/accuracy_state.json 에 남아 재시작에도 이어진다)
결과 제공
  - MQTT(보드 안 브로커 t22-mosquitto, 포트 1884) 토픽 team2-2/inference/<공간>, retain=True → 새 구독자도 최신값을 바로 받는다
  - 조회 API http://<보드>:8090  (/health, /spaces, /spaces/{공간}, /spaces/{공간}/co2?minutes=30, /spaces/{공간}/accuracy)
  - 보드 InfluxDB measurement "inference", 주기 기록 logs/cycles.jsonl
전처리는 학습과 같은 prep.clean / prep.to_grid 를 쓴다(학습·서비스 전처리 불일치 방지).

  python service.py            # 상시 실행
  python service.py --once     # 한 번만 계산하고 결과 출력
"""
import json
import logging
import os
import pickle
import sys
import threading
import time
import traceback
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import dataset
import models
import prep
from models import AE

ROOT = Path(__file__).resolve().parent
MD, LOGS = ROOT / "models", ROOT / "logs"
LOGS.mkdir(exist_ok=True)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                    handlers=[logging.StreamHandler(), logging.FileHandler(LOGS / "service.log")])
log = logging.getLogger("service")
torch.set_num_threads(2)
KST = timezone(timedelta(hours=9))
VERSION = "v1"

# ---------------------------------------------------------------- 설정
ENV = {}
for line in (Path.home() / "ingest/.env").read_text().splitlines():
    if "=" in line and not line.startswith("#"):
        k, v = line.split("=", 1)
        ENV[k] = v
SPACES = {"실습실": "lab", "회의실": "meeting", "사무실": "office"}   # 실데이터 공간 → 토픽 이름
HVAC_NAME = {"회의실": "회의실", "사무실": "사무실"}                 # 공간 → 냉난방 실내기 이름
GAP_MIN = {"실습실": 5, "사무실": 5, "회의실": 90}   # 공간별 센서 주기에 맞춘 끊김 기준(회의실 센서는 1시간에 1~2번 보낸다)
STUCK_MIN, ALERT_REPEAT_MIN = 60, 30
MQTT_HOST, MQTT_PORT, TOPIC = os.getenv("OUT_MQTT_HOST", "localhost"), int(os.getenv("OUT_MQTT_PORT", "1884")), "team2-2/inference"

# ---------------------------------------------------------------- 모델
gen = json.loads((ROOT / "report/forecast_service.json").read_text())   # 실공간용 일반 모델(공간·일정 입력 없음)
ck = torch.load(MD / f"forecast_{gen['name']}.pt", weights_only=False)
STATS = ck["stats"]
FC = models.make(ck["kind"], ck["n_in"]); FC.load_state_dict(ck["state"]); FC.eval()
L = ck["L"]
ak = torch.load(MD / "ae.pt", weights_only=False)
AEM = AE(ak["n_in"], ak["n_cond"]); AEM.load_state_dict(ak["state"]); AEM.eval()
AMETA = json.loads((ROOT / "report/anomaly_meta.json").read_text())
OCC = pickle.load(open(MD / "occ.pkl", "rb"))
VENT = pickle.load(open(MD / "vent.pkl", "rb"))
STD = np.array([STATS[v]["std"] for v in dataset.TARGETS])

from influxdb_client import InfluxDBClient  # noqa: E402
import paho.mqtt.client as mqtt  # noqa: E402

influx = InfluxDBClient(url="http://localhost:8087", token=ENV["INFLUX_TOKEN"], org=ENV["INFLUX_ORG"], timeout=20000)
from influxdb_client.client.write_api import SYNCHRONOUS  # noqa: E402
write_api = influx.write_api(write_options=SYNCHRONOUS)
mq = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="team2-2-inference")
mq.reconnect_delay_set(1, 30)
if ENV.get("OUT_MQTT_USER"):
    mq.username_pw_set(ENV["OUT_MQTT_USER"], ENV["OUT_MQTT_PASS"])   # 결과 브로커 쓰기 계정(읽기는 누구나)
LATEST = {}          # 공간 → 최신 결과(API가 읽는다)
ALERT_STATE = {}     # (공간, 경보) → 마지막 발송 시각
HEALTH = dict(started=datetime.now(KST).isoformat(), cycles=0, failures=0, last_cycle=None, last_error=None)

# ---------------------------------------------------------------- 자기 성능 감시 설정
ACC_WINDOW_H = float(os.getenv("ACC_WINDOW_H", "6"))      # 이동 MAE 창(시간)
ACC_MIN_N = 12                                            # 이 건수 미만이면 경보 판단을 하지 않는다
ACC_FACTOR = float(os.getenv("ACC_MAE_FACTOR", "2.0"))
ACC_THRESHOLD_CO2_60 = round(ACC_FACTOR * gen["val_co2_60"], 1)   # 검증 기간 60분 앞 CO₂ MAE × 2
ACC_EXPIRE_MIN = 30                                       # 대상 시각 뒤 30분 안에 실측이 없으면 채점하지 않고 버린다
ACC_FILE = LOGS / "accuracy_state.json"
PENDING = {}         # 공간 → {기준 시각: {"h": [...], "co2": [...], "temp": [...]}}  아직 채점 안 된 예측
SCORED = {}          # 공간 → [[기준 시각, h, 대상 시각, CO₂ 오차, 온도 오차], ...]       창 안의 채점 결과
ACC_COUNT = {}       # 공간 → dict(scored=, expired=)  시작 뒤 누계


# ---------------------------------------------------------------- 데이터 읽기 (보드 InfluxDB, 실데이터)
def read_raw(hours=6):
    q = f'''from(bucket:"{ENV["INFLUX_BUCKET"]}") |> range(start:-{hours}h)
      |> filter(fn:(r)=> r._measurement=="env" and (r._field=="co2" or r._field=="temperature" or r._field=="humidity"))
      |> pivot(rowKey:["_time","device"], columnKey:["_field"], valueColumn:"_value")
      |> keep(columns:["_time","location","device","co2","temperature","humidity"])'''
    df = influx.query_api().query_data_frame(q)
    if isinstance(df, list):
        df = pd.concat(df) if df else pd.DataFrame()
    if df.empty:
        return df
    df = df.rename(columns={"_time": "time", "location": "space"})
    df["time"] = pd.to_datetime(df.time).dt.tz_convert("Asia/Seoul").dt.tz_localize(None)
    for v in prep.VARS:
        if v not in df:
            df[v] = np.nan
    return df[df.space.isin(SPACES)][["time", "space", "device", *prep.VARS]]


def read_hvac():
    q = f'''from(bucket:"{ENV["INFLUX_BUCKET"]}") |> range(start:-30m)
      |> filter(fn:(r)=> r._measurement=="hvac" and r._field=="power") |> last() |> keep(columns:["name","_value","_time"])'''
    out = {}
    for t in influx.query_api().query(q):
        for r in t.records:
            out[r["name"]] = int(r.get_value())
    return out


# ---------------------------------------------------------------- 자기 성능 감시: 예측 저장 → 실측 도착 시 채점 → 이동 MAE
def acc_load():
    """재시작해도 채점이 이어지게 저장 파일을 읽는다. 없으면 InfluxDB 'inference' 의 co2_60 기록으로 60분 앞 예측만 채운다."""
    global PENDING, SCORED
    try:
        st = json.loads(ACC_FILE.read_text())
        PENDING, SCORED = st["pending"], st["scored"]
        log.info("성능 감시 상태 복원: 대기 %s, 채점 %s", {s: len(v) for s, v in PENDING.items()}, {s: len(v) for s, v in SCORED.items()})
        return
    except FileNotFoundError:
        pass
    except Exception as e:
        log.warning("성능 감시 상태 파일을 읽지 못함(새로 시작): %s", e)
    try:
        q = f'''from(bucket:"{ENV["INFLUX_BUCKET"]}") |> range(start:-{int(ACC_WINDOW_H) + 1}h)
          |> filter(fn:(r)=> r._measurement=="inference" and r._field=="co2_60") |> keep(columns:["_time","space","_value"])'''
        n = 0
        for t in influx.query_api().query(q):
            for r in t.records:
                tt = r.get_time().astimezone(KST).replace(tzinfo=None)
                base = (tt - timedelta(minutes=tt.minute % 5)).replace(second=0, microsecond=0)
                PENDING.setdefault(r["space"], {}).setdefault(base.isoformat(), dict(h=[60], co2=[float(r.get_value())], temp=[None]))
                n += 1
        log.info("성능 감시: InfluxDB 기록에서 60분 앞 예측 %d건을 채점 대기로 채움", n)
    except Exception as e:
        log.warning("성능 감시 초기 채우기 실패: %s", e)


def acc_save():
    tmp = ACC_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(dict(pending=PENDING, scored=SCORED, saved_at=datetime.now(KST).isoformat()), ensure_ascii=False))
    tmp.replace(ACC_FILE)


def acc_score(space, gl, end):
    """실측이 들어온 대상 시각의 예측을 채점한다. 실측 = 그 5분 칸에 원본이 1건 이상 있는 값(보간 값은 쓰지 않는다).
    늦게 도착하는 메시지를 감안해 대상 칸이 끝나고 5분이 더 지난 뒤에 채점한다."""
    pend, sc = PENDING.setdefault(space, {}), SCORED.setdefault(space, [])
    cnt = ACC_COUNT.setdefault(space, dict(scored=0, expired=0))
    obs = gl[(gl["co2_n"] > 0) & gl["co2"].notna()]
    new = []
    for base in list(pend):
        e = pend[base]
        keep = dict(h=[], co2=[], temp=[])
        for h, pc, pt in zip(e["h"], e["co2"], e["temp"]):
            tgt = datetime.fromisoformat(base) + timedelta(minutes=h)
            if tgt > end - timedelta(minutes=5):
                keep["h"].append(h); keep["co2"].append(pc); keep["temp"].append(pt)
            elif tgt in obs.index:
                a = obs.loc[tgt]
                et = None if pt is None or np.isnan(a["temperature"]) else round(float(pt - a["temperature"]), 3)
                row = [base, h, tgt.isoformat(), round(float(pc - a["co2"]), 1), et]
                sc.append(row); new.append(row); cnt["scored"] += 1
            elif tgt < end - timedelta(minutes=ACC_EXPIRE_MIN):
                cnt["expired"] += 1
            else:
                keep["h"].append(h); keep["co2"].append(pc); keep["temp"].append(pt)
        if keep["h"]:
            pend[base] = keep
        else:
            pend.pop(base)
    lo = (end - timedelta(hours=ACC_WINDOW_H)).isoformat()
    SCORED[space] = [r for r in sc if r[2] > lo]
    if new:
        with open(LOGS / "accuracy.jsonl", "a") as f:
            for r in new:
                f.write(json.dumps(dict(space=space, base_time=r[0], minutes_ahead=r[1], target_time=r[2], err_co2=r[3], err_temp=r[4]), ensure_ascii=False) + "\n")


def acc_register(space, end, forecast):
    """이번 기준 시각(5분 칸)의 예측을 처음 한 번만 저장한다(같은 칸 안의 매분 재계산은 저장하지 않는다)."""
    pend = PENDING.setdefault(space, {})
    key = end.isoformat()
    if forecast and key not in pend:
        pend[key] = dict(h=[f["minutes_ahead"] for f in forecast], co2=[f["co2_ppm"] for f in forecast], temp=[f["temperature_c"] for f in forecast])


def acc_summary(space, end):
    sc = SCORED.get(space, [])
    rows = []
    for h in range(5, 65, 5):
        e = [r for r in sc if r[1] == h]
        ec = np.array([r[3] for r in e], float)
        et = np.array([r[4] for r in e if r[4] is not None], float)
        rows.append(dict(minutes_ahead=h, n=len(e), mae_co2_ppm=round(float(np.abs(ec).mean()), 1) if len(ec) else None,
                         bias_co2_ppm=round(float(ec.mean()), 1) if len(ec) else None,
                         mae_temperature_c=round(float(np.abs(et).mean()), 3) if len(et) else None))
    h60 = rows[-1]
    degraded = h60["n"] >= ACC_MIN_N and h60["mae_co2_ppm"] > ACC_THRESHOLD_CO2_60
    return dict(window_hours=ACC_WINDOW_H, as_of=end.isoformat(), mae_co2_60_ppm=h60["mae_co2_ppm"], n_60=h60["n"],
                threshold_co2_60_ppm=ACC_THRESHOLD_CO2_60, threshold_rule=f"검증 MAE {gen['val_co2_60']} × {ACC_FACTOR}",
                min_n=ACC_MIN_N, degraded=bool(degraded), enough_data=h60["n"] >= ACC_MIN_N,
                pending_forecasts=sum(len(e["h"]) for e in PENDING.get(space, {}).values()),
                scored_since_start=ACC_COUNT.get(space, {}).get("scored", 0), expired_since_start=ACC_COUNT.get(space, {}).get("expired", 0),
                by_horizon=rows)


# ---------------------------------------------------------------- 공간 하나 계산
def space_grid(raw, space, now):
    d = raw[raw.space == space]
    # 한 공간에 센서가 여러 대면 같은 시각대 값을 센서별로 정제한 뒤 5분 칸에서 중앙값으로 합친다
    cleaned, removed = prep.clean(d.assign(space=space))
    now = now.replace(tzinfo=None)   # 저장된 시각은 KST(시간대 표기 없음)
    end = (now - timedelta(minutes=now.minute % 5)).replace(second=0, microsecond=0)   # 막 시작된 5분 칸은 쓰지 않는다
    per_dev = []
    for dev, g in cleaned.groupby("device"):
        gg = prep.to_grid(g.assign(space=space), start=end - timedelta(hours=6), end=end)
        per_dev.append(gg.loc[space][prep.VARS])
    agg = pd.concat(per_dev).groupby(level=0).median()
    full = prep.to_grid(cleaned.assign(space=space), start=end - timedelta(hours=6), end=end).loc[space]
    full[prep.VARS] = agg.reindex(full.index)[prep.VARS].combine_first(full[prep.VARS])
    full["sched_people"] = 0.0
    full["split"] = "live"
    full.index = pd.MultiIndex.from_product([[space], full.index], names=["space", "time"])
    return full, cleaned, int(removed.values.sum()), end


def compute(space, raw, hvac, now):
    g, cleaned, n_removed, end = space_grid(raw, space, now)
    fe = dataset.features(g, STATS, use_sched=False, use_space=False)[space]
    gl = g.loc[space]
    last_msg = cleaned.time.max() if len(cleaned) else None
    age = (now.replace(tzinfo=None) - last_msg).total_seconds() / 60 if last_msg is not None else None
    res = dict(space=space, topic_space=SPACES[space], time=end.isoformat(), computed_at=now.isoformat(), model_version=VERSION,
               data=dict(last_message_kst=last_msg.isoformat() if last_msg is not None else None, minutes_since_last=round(age, 1) if age is not None else None,
                         devices=sorted(cleaned.device.unique().tolist()), removed_invalid_6h=n_removed))
    now_co2, now_t = gl["co2"].iloc[-1], gl["temperature"].iloc[-1]
    res["now"] = dict(co2_ppm=None if np.isnan(now_co2) else round(float(now_co2), 1),
                      temperature_c=None if np.isnan(now_t) else round(float(now_t), 2),
                      humidity_pct=None if np.isnan(gl["humidity"].iloc[-1]) else round(float(gl["humidity"].iloc[-1]), 1))
    alerts = []
    # --- 규칙: 데이터 끊김 / 범위 밖 / 값 고정
    if age is None or age > GAP_MIN[space]:
        alerts.append(dict(type="data_gap", severity="warning", sensor="all", message=f"{GAP_MIN[space]}분 넘게 데이터가 들어오지 않음 (마지막 {res['data']['last_message_kst']})"))
    if n_removed and len(cleaned) and cleaned[cleaned.time > end - timedelta(minutes=10)][prep.VARS].isna().any().any():
        alerts.append(dict(type="out_of_range", severity="info", sensor="unknown", message="최근 10분 안에 물리적으로 불가능한 값이 들어옴(정제에서 제외)"))
    recent = cleaned[cleaned.time > end - timedelta(minutes=STUCK_MIN)]
    for dev, gdev in recent.groupby("device"):
        vals = gdev[prep.VARS].dropna(axis=1, how="all")
        if len(gdev) >= 6 and not vals.empty and (vals.nunique() <= 1).all():
            alerts.append(dict(type="stuck", severity="warning", sensor=dev, message=f"{dev}: {STUCK_MIN}분 동안 값이 전혀 바뀌지 않음"))
    # --- 60분 예측 (최근 L칸 중 관측이 절반 이상일 때만)
    hist_ok = (~np.isnan(gl["co2"].values[-L:])).mean() >= 0.5 and not np.isnan(now_co2)
    if hist_ok:
        x = torch.from_numpy(fe["feat"][-L:][None])
        hour = [(end + timedelta(minutes=5 * k)) for k in range(1, 13)]
        hh = np.array([t.hour + t.minute / 60 for t in hour])
        fut = torch.from_numpy(np.column_stack([np.zeros(12), np.sin(2 * np.pi * hh / 24), np.cos(2 * np.pi * hh / 24)]).astype(np.float32)[None])
        with torch.no_grad():
            p = FC(x, fut).numpy()[0]
        co2 = now_co2 + p[:, 0] * STD[0]
        temp = now_t + p[:, 1] * STD[1]
        res["forecast"] = [dict(minutes_ahead=5 * (k + 1), time=hour[k].isoformat(), co2_ppm=round(float(co2[k]), 1),
                                temperature_c=round(float(temp[k]), 2)) for k in range(12)]
        over = np.where(co2 >= 1000)[0]
        res["co2_over_1000_in_min"] = int(5 * (over[0] + 1)) if len(over) and now_co2 < 1000 else (0 if now_co2 >= 1000 else None)
        if now_co2 >= 1000:
            alerts.append(dict(type="co2_high", severity="warning", sensor="co2", message=f"지금 CO₂ {now_co2:.0f}ppm, 환기 권고 기준 초과"))
        elif len(over):
            alerts.append(dict(type="co2_forecast", severity="info", sensor="co2", message=f"약 {5 * (over[0] + 1)}분 뒤 1,000ppm 초과 예상"))
    else:
        res["forecast"] = None
        res["co2_over_1000_in_min"] = None
    # --- 자기 성능 감시: 지난 예측 채점 → 이번 예측 저장 → 이동 MAE
    acc_score(space, gl, end)
    acc_register(space, end, res["forecast"])
    res["accuracy"] = acc_summary(space, end)
    if res["accuracy"]["degraded"]:
        alerts.append(dict(type="forecast_degraded", severity="warning", sensor="model",
                           message=f"최근 {ACC_WINDOW_H:g}시간 60분 앞 CO₂ 예측 MAE {res['accuracy']['mae_co2_60_ppm']}ppm, 기준 {ACC_THRESHOLD_CO2_60}ppm 초과 (재학습·센서 점검 필요)"))
    # --- 오토인코더 이상 점수 (최근 1시간 창)
    W = ak["W"]
    f = fe["feat"][-W:]
    # 데이터가 끊겼거나 최근 15분에 결측이 있으면 이상 점수를 내지 않는다(결측을 이상으로 오판한다 — 2026-10-20 장애 시험에서 확인)
    gap_now = any(a["type"] == "data_gap" for a in alerts)
    if len(f) == W and f[:, [1, 3, 5]].mean() < 0.5 and not gap_now and f[-3:, [1, 3, 5]].max() == 0:
        xw = torch.from_numpy(f[:, [0, 2, 4]].reshape(1, -1).astype(np.float32))
        mw = f[:, [1, 3, 5]].reshape(1, -1)
        cw = torch.from_numpy(f[-1:, 6:].astype(np.float32))
        with torch.no_grad():
            r = AEM(xw, cw).numpy()
        err = (((r - xw.numpy()) ** 2) * (1 - mw)).reshape(W, 3)[-3:].mean(0)
        score = float(err.sum())
        culprit = ["co2", "temperature", "humidity"][int(err.argmax())]
        res["anomaly"] = dict(score=round(score, 3), threshold=round(AMETA["ae_threshold"], 3), is_anomaly=score > AMETA["ae_threshold"],
                              culprit_sensor=culprit, per_sensor={k: round(float(v), 3) for k, v in zip(["co2", "temperature", "humidity"], err)})
        if score > AMETA["ae_threshold"]:
            alerts.append(dict(type="anomaly", severity="warning", sensor=culprit, message=f"평소와 다른 패턴(점수 {score:.2f}), 원인 후보: {culprit}"))
    else:
        res["anomaly"] = None
    # --- 상태: 사용 여부, 환기 선제 경보, 빈 방 냉난방
    sf = dataset.sensor_features(gl, "unknown", use_space=True).iloc[[-1]]
    for s in dataset.SPACES:
        sf[f"sp_{s}"] = 0.0
    sf["sched_now"] = 0.0
    sf["sched_next30"] = 0.0
    if sf[OCC["feats"]].notna().all(axis=None):
        occ_p = float(OCC["model"].predict_proba(sf[OCC["feats"]])[0, 1])
        vent_p = float(VENT["model"].predict_proba(sf[VENT["feats"]])[0, 1])
        res["state"] = dict(occupied=occ_p >= 0.5, occupied_prob=round(occ_p, 3),
                            ventilation_warning=bool(vent_p >= VENT["threshold"] and (now_co2 < 1000)), ventilation_prob=round(vent_p, 3))
        if res["state"]["ventilation_warning"]:
            alerts.append(dict(type="ventilation_soon", severity="info", sensor="co2", message="30분 안에 1,000ppm을 넘을 가능성이 높음, 미리 환기 권장"))
    else:
        res["state"] = None
    hv = hvac.get(HVAC_NAME.get(space, ""), None)
    res["hvac_power"] = None if hv is None else bool(hv)
    if hv == 1 and res["state"] and not res["state"]["occupied"]:
        alerts.append(dict(type="hvac_empty_room", severity="info", sensor="hvac", message="빈 방으로 추정되는데 냉난방이 켜져 있음"))
    res["alerts"] = alerts
    return res


# ---------------------------------------------------------------- 경보 다시 보내기 규칙
def decide_notify(space, alerts, now):
    """같은 경보는 처음 생길 때, 그리고 30분마다 한 번만 '알림'으로 표시. 사라지면 해제를 알린다."""
    active = {a["type"] for a in alerts}
    notes = []
    for a in alerts:
        key = (space, a["type"])
        last = ALERT_STATE.get(key)
        a["notify"] = last is None or (now - last) >= timedelta(minutes=ALERT_REPEAT_MIN)
        a["since"] = (ALERT_STATE.get((space, a["type"], "since")) or now).isoformat()
        ALERT_STATE.setdefault((space, a["type"], "since"), now)
        if a["notify"]:
            ALERT_STATE[key] = now
    for key in [k for k in ALERT_STATE if len(k) == 2 and k[0] == space and k[1] not in active]:
        notes.append(dict(type=key[1], status="resolved", at=now.isoformat()))
        ALERT_STATE.pop(key, None)
        ALERT_STATE.pop((space, key[1], "since"), None)
    return notes


def cycle():
    t0 = time.time()
    now = datetime.now(KST)
    raw = read_raw()
    hvac = read_hvac()
    out = {}
    for space in SPACES:
        try:
            r = compute(space, raw, hvac, now)
        except Exception as e:  # 한 공간이 실패해도 다른 공간과 다음 주기는 계속
            log.warning("공간 %s 계산 실패: %s", space, e)
            r = dict(space=space, topic_space=SPACES[space], computed_at=now.isoformat(), error=str(e), alerts=[
                dict(type="compute_error", severity="warning", sensor="service", message=str(e)[:200])])
        r["resolved"] = decide_notify(space, r["alerts"], now)
        out[space] = r
        LATEST[space] = r
        payload = json.dumps(r, ensure_ascii=False, default=str)
        mq.publish(f"{TOPIC}/{SPACES[space]}", payload, qos=1, retain=True)
        for a in r["alerts"]:
            if a.get("notify"):
                mq.publish(f"{TOPIC}/{SPACES[space]}/alert", json.dumps(dict(space=space, **a), ensure_ascii=False), qos=1)
    elapsed = time.time() - t0
    summary = dict(cycle_start=now.isoformat(), elapsed_s=round(elapsed, 3),
                   alerts={s: [a["type"] for a in r["alerts"]] for s, r in out.items()},
                   notified={s: [a["type"] for a in r["alerts"] if a.get("notify")] for s, r in out.items()},
                   co2_60={s: (r.get("forecast") or [{}])[-1].get("co2_ppm") for s, r in out.items()})
    with open(LOGS / "cycles.jsonl", "a") as f:
        f.write(json.dumps(summary, ensure_ascii=False) + "\n")
    try:
        acc_save()
    except Exception as e:
        log.warning("성능 감시 상태 저장 실패: %s", e)
    mq.publish(f"{TOPIC}/_status", json.dumps(dict(HEALTH, last_cycle=now.isoformat(), elapsed_s=round(elapsed, 3)), ensure_ascii=False), retain=True)
    try:
        from influxdb_client import Point
        pts = []
        for s, r in out.items():
            p = Point("inference").tag("space", s).field("elapsed_s", elapsed).field("n_alerts", len(r["alerts"]))
            if r.get("forecast"):
                p = p.field("co2_60", r["forecast"][-1]["co2_ppm"]).field("temp_60", r["forecast"][-1]["temperature_c"])
            if r.get("anomaly"):
                p = p.field("anomaly_score", r["anomaly"]["score"])
            if r.get("accuracy") and r["accuracy"]["mae_co2_60_ppm"] is not None:
                p = p.field("mae_co2_60", r["accuracy"]["mae_co2_60_ppm"]).field("mae_n_60", r["accuracy"]["n_60"])
            if r.get("state"):
                p = p.field("occupied_prob", r["state"]["occupied_prob"]).field("vent_prob", r["state"]["ventilation_prob"])
            pts.append(p)
        write_api.write(ENV["INFLUX_BUCKET"], ENV["INFLUX_ORG"], pts)
    except Exception as e:
        log.warning("결과 저장 실패: %s", e)
    return summary


def loop():
    while True:
        nxt = (datetime.now(KST) + timedelta(minutes=1)).replace(second=5, microsecond=0)
        time.sleep(max(0.0, (nxt - datetime.now(KST)).total_seconds()))
        try:
            s = cycle()
            HEALTH.update(cycles=HEALTH["cycles"] + 1, last_cycle=s["cycle_start"], last_elapsed_s=s["elapsed_s"])
            log.info("주기 완료 %.2fs 경보 %s", s["elapsed_s"], s["alerts"])
        except Exception as e:  # 한 번 실패해도 다음 주기에 다시 시도
            HEALTH.update(failures=HEALTH["failures"] + 1, last_error=f"{datetime.now(KST).isoformat()} {e}")
            log.error("주기 실패: %s\n%s", e, traceback.format_exc())


# ---------------------------------------------------------------- 조회 API
def api():
    from fastapi import FastAPI, HTTPException
    import uvicorn
    app = FastAPI(title="team2-2 교육장 추론 서비스", version=VERSION)
    alias = {v: k for k, v in SPACES.items()} | {k: k for k in SPACES}

    def get(space):
        s = alias.get(space)
        if s is None or s not in LATEST:
            raise HTTPException(404, f"공간 '{space}' 없음. 가능한 값: {list(SPACES.values())}")
        return LATEST[s]

    @app.get("/health")
    def health():
        return HEALTH

    @app.get("/spaces")
    def spaces():
        return {r["topic_space"]: dict(space=k, time=r.get("time"), now=r.get("now"), alerts=[a["type"] for a in r["alerts"]]) for k, r in LATEST.items()}

    @app.get("/spaces/{space}")
    def one(space: str):
        return get(space)

    @app.get("/spaces/{space}/co2")
    def co2(space: str, minutes: int = 30):
        r = get(space)
        if not r.get("forecast"):
            raise HTTPException(503, "예측 불가(최근 데이터 부족)")
        m = min(60, max(5, int(round(minutes / 5)) * 5))
        f = next(x for x in r["forecast"] if x["minutes_ahead"] == m)
        return dict(space=r["space"], base_time=r["time"], minutes_ahead=m, co2_ppm=f["co2_ppm"], temperature_c=f["temperature_c"],
                    co2_over_1000_in_min=r["co2_over_1000_in_min"])

    @app.get("/spaces/{space}/accuracy")
    def accuracy(space: str):
        r = get(space)
        if not r.get("accuracy"):
            raise HTTPException(503, "성능 감시 결과 없음(이번 주기 계산 실패)")
        return dict(space=r["space"], topic_space=r["topic_space"], **r["accuracy"])

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("API_PORT", "8090")), log_level="warning")


if __name__ == "__main__":
    mq.connect_async(MQTT_HOST, MQTT_PORT)
    mq.loop_start()
    acc_load()
    if "--once" in sys.argv:
        time.sleep(1)
        print(json.dumps(cycle(), ensure_ascii=False, indent=1))
        print(json.dumps(LATEST, ensure_ascii=False, indent=1, default=str)[:4000])
        sys.exit()
    threading.Thread(target=loop, daemon=True).start()
    api()
