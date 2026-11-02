"""MQTT → InfluxDB 상시 수집기 (team2-2).

- 메시지 해석 규칙은 모두 config.yaml에 있다. 새 토픽·새 형식은 설정만 바꾼다.
- 시각은 메시지에 찍힌 측정 시각을 쓰고, 수신 시각과의 차이(rx_delay_s)를 함께 남긴다.
- 브로커가 끊기면 자동 재접속하고, InfluxDB 쓰기가 실패하면 메모리에 보관했다가 다시 쓴다.
- 1분마다 자기 상태(collector measurement)를 기록한다.
- PG_HOST가 .env에 있으면 같은 기록을 PostgreSQL에도 쓴다(InfluxDB는 보드 안 버퍼 겸 원본).
  PostgreSQL이 끊겨도 InfluxDB 수집은 계속되고, 밀린 기록은 메모리에 보관했다가 다시 쓴다.
"""
import collections
import json
import logging
import os
import signal
import threading
import time
from datetime import datetime, timedelta, timezone

import paho.mqtt.client as mqtt
import yaml
from influxdb_client import InfluxDBClient, Point, WritePrecision
from influxdb_client.client.write_api import SYNCHRONOUS

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("collector")

CFG = yaml.safe_load(open(os.getenv("CONFIG", "config.yaml"), encoding="utf-8"))
MQ, IX = CFG["mqtt"], CFG["influx"]
BUCKET, ORG = os.environ["INFLUX_BUCKET"], os.environ["INFLUX_ORG"]

stats = collections.Counter()
pending = collections.deque(maxlen=IX.get("max_pending", 200000))
pg_pending = collections.deque(maxlen=IX.get("max_pending", 200000))
PG = bool(os.getenv("PG_HOST"))
lock = threading.Lock()
seen = collections.OrderedDict()
state = {"connected": False, "last_msg": None}
stop = threading.Event()


# ---------------------------------------------------------------- 경로 해석
def get_path(msg, topic_parts, path):
    """'a.b.0.c' 또는 'topic.N' 경로로 값을 꺼낸다. 없으면 None."""
    if path is None:
        return None
    if path == "topic":
        return "/".join(topic_parts)
    keys = path.split(".")
    if keys[0] == "topic":
        i = int(keys[1])
        return topic_parts[i] if i < len(topic_parts) else None
    cur = msg
    for k in keys:
        if isinstance(cur, list) and k.isdigit() and int(k) < len(cur):
            cur = cur[int(k)]
        elif isinstance(cur, dict) and k in cur:
            cur = cur[k]
        else:
            return None
    return cur


def parse_time(raw, fmt):
    if raw is None:
        return None
    try:
        if fmt == "epoch_ms":
            return datetime.fromtimestamp(float(raw) / 1000, tz=timezone.utc)
        if fmt == "epoch_s":
            return datetime.fromtimestamp(float(raw), tz=timezone.utc)
        return datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None


def numeric_fields(obj, rename, exclude):
    """숫자·참거짓 값만 float 필드로. 참거짓은 1/0."""
    out = {}
    if not isinstance(obj, dict):
        return out
    for k, v in obj.items():
        if k in exclude:
            continue
        if isinstance(v, bool):
            v = 1.0 if v else 0.0
        elif isinstance(v, (int, float)):
            v = float(v)
        else:
            continue
        out[rename.get(k, k)] = v
    return out


def topic_matches(pattern, topic):
    return mqtt.topic_matches_sub(pattern, topic)


# ---------------------------------------------------------------- 메시지 처리
def is_duplicate(key):
    if key in seen:
        return True
    seen[key] = None
    if len(seen) > CFG.get("dedup_window", 20000):
        seen.popitem(last=False)
    return False


def handle(topic, payload):
    rx = datetime.now(timezone.utc)
    stats["received"] += 1
    state["last_msg"] = time.time()
    parts = topic.split("/")
    src = next((s for s in CFG["sources"] if topic_matches(s["topic"], topic)), None)
    if src is None:
        stats["unmatched"] += 1
        return
    try:
        msg = json.loads(payload)
    except (ValueError, UnicodeDecodeError):
        stats["parse_error"] += 1
        return

    if src.get("dedup"):
        key = (src["name"],) + tuple(str(get_path(msg, parts, p)) for p in src["dedup"])
        if is_duplicate(key):
            stats["duplicate"] += 1
            return

    tcfg = src.get("time", {})
    ts = parse_time(get_path(msg, parts, tcfg.get("path")), tcfg.get("format", "iso"))
    if ts is None:
        ts = rx
        stats["no_timestamp"] += 1
    delay = (rx - ts).total_seconds()
    tags = {k: get_path(msg, parts, p) for k, p in src.get("tags", {}).items()}
    tags = {k: str(v) for k, v in tags.items() if v not in (None, "")}

    records = []
    for out in src["outputs"]:
        rename, exclude = out.get("rename") or {}, set(out.get("exclude") or [])
        groups = []
        if "fields_from" in out:
            base = msg if out["fields_from"] == "" else get_path(msg, parts, out["fields_from"])
            if out.get("keyed_by") and isinstance(base, dict):
                groups = [({out["keyed_by"]: str(k)}, numeric_fields(v, rename, exclude)) for k, v in base.items()]
            else:
                groups = [({}, numeric_fields(base, rename, exclude))]
        else:
            f = {k: get_path(msg, parts, p) for k, p in out["fields"].items()}
            groups = [({}, numeric_fields(f, {}, set()))]
        for extra_tags, fields in groups:
            if fields:
                records.append(dict(measurement=out["measurement"], ts=ts, tags={**tags, **extra_tags},
                                    fields=fields, rx_delay_s=round(delay, 3)))
    if not records:
        stats["empty"] += 1
    emit(records)


EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)


def to_point(r):
    # 정수 ms로 넘겨 InfluxDB와 PostgreSQL의 시각이 1ms도 어긋나지 않게 한다(부동소수 반올림 방지)
    p = Point(r["measurement"]).time((r["ts"] - EPOCH) // timedelta(milliseconds=1), WritePrecision.MS)
    for k, v in r["tags"].items():
        p = p.tag(k, v)
    for k, v in r["fields"].items():
        p = p.field(k, v)
    if r.get("rx_delay_s") is not None:
        p = p.field("rx_delay_s", float(r["rx_delay_s"]))
    return p


def emit(records):
    # InfluxDB가 ms 단위로 저장하므로 PostgreSQL도 ms로 맞춘다(재동기화 때 같은 기본 키가 되도록)
    for r in records:
        r["ts"] = r["ts"].replace(microsecond=r["ts"].microsecond // 1000 * 1000)
    with lock:
        pending.extend(to_point(r) for r in records)
        if PG:
            pg_pending.extend(records)


# ---------------------------------------------------------------- MQTT
def on_connect(client, userdata, flags, reason_code, properties):
    if reason_code.is_failure:
        log.warning("MQTT 접속 실패: %s", reason_code)
        return
    state["connected"] = True
    log.info("MQTT 접속 (session_present=%s)", flags.session_present)
    for topic in MQ.get("subscribe") or [s["topic"] for s in CFG["sources"]]:
        client.subscribe(topic, qos=MQ.get("qos", 1))
        log.info("구독: %s", topic)


def on_disconnect(client, userdata, flags, reason_code, properties):
    state["connected"] = False
    stats["disconnects"] += 1
    log.warning("MQTT 끊김: %s (자동 재접속)", reason_code)


def on_message(client, userdata, m):
    try:
        handle(m.topic, m.payload)
    except Exception:  # 한 메시지 오류로 수집기가 죽지 않게
        stats["handler_error"] += 1
        log.exception("메시지 처리 오류: %s", m.topic)


# ---------------------------------------------------------------- InfluxDB 쓰기
def writer(write_api):
    last_flush = time.time()
    while not stop.is_set():
        time.sleep(0.2)
        with lock:
            n = len(pending)
        if n == 0 or (n < IX.get("batch_size", 500) and time.time() - last_flush < IX.get("flush_seconds", 2)):
            continue
        with lock:
            batch = [pending.popleft() for _ in range(min(n, 5000))]
        try:
            write_api.write(bucket=BUCKET, org=ORG, record=batch)
            stats["written"] += len(batch)
            last_flush = time.time()
        except Exception as e:
            stats["write_error"] += 1
            log.warning("InfluxDB 쓰기 실패(%d건 보관 후 재시도): %s", len(batch), e)
            with lock:
                pending.extendleft(reversed(batch))
            time.sleep(5)


# ---------------------------------------------------------------- PostgreSQL 쓰기
PG_DDL = """
CREATE TABLE IF NOT EXISTS readings (
    ts           timestamptz      NOT NULL,             -- 측정 시각(UTC 저장, 조회 시 KST로 변환)
    measurement  text             NOT NULL,             -- env / uplink / hvac / collector …
    series       text             NOT NULL,             -- 같은 출처를 가리키는 키(태그를 이은 문자열)
    tags         jsonb            NOT NULL DEFAULT '{}', -- location, device, model, gateway, name …
    fields       jsonb            NOT NULL,             -- 숫자 값 전부(co2, temperature …)
    rx_delay_s   double precision,                      -- 수집기 수신 시각 - 측정 시각
    inserted_at  timestamptz      NOT NULL DEFAULT now(),
    PRIMARY KEY (measurement, series, ts)               -- 같은 메시지를 두 번 써도 한 줄만 남는다
);
CREATE INDEX IF NOT EXISTS readings_ts_idx ON readings (ts);
CREATE OR REPLACE VIEW env_readings AS
SELECT ts AT TIME ZONE 'Asia/Seoul'          AS ts_kst,
       ts,
       tags->>'location'                      AS location,
       tags->>'device'                        AS device,
       tags->>'model'                         AS model,
       (fields->>'co2')::double precision          AS co2,
       (fields->>'temperature')::double precision  AS temperature,
       (fields->>'humidity')::double precision     AS humidity,
       fields
FROM readings WHERE measurement = 'env';
"""
PG_INSERT = """INSERT INTO readings (ts, measurement, series, tags, fields, rx_delay_s)
VALUES (%s, %s, %s, %s, %s, %s) ON CONFLICT DO NOTHING"""


def series_key(tags):
    return "|".join(f"{k}={tags[k]}" for k in sorted(tags))


def pg_writer():
    import psycopg
    from psycopg.types.json import Jsonb
    conn = None
    while not stop.is_set():
        time.sleep(0.5)
        if not pg_pending:
            continue
        try:
            if conn is None or conn.closed:
                conn = psycopg.connect(host=os.environ["PG_HOST"], port=os.getenv("PG_PORT", "5432"),
                                       dbname=os.environ["PG_DB"], user=os.environ["PG_USER"],
                                       password=os.environ["PG_PASSWORD"], connect_timeout=10,
                                       options=f"-c search_path={os.getenv('PG_SCHEMA', 'public')}")
                with conn.cursor() as cur:
                    cur.execute(PG_DDL)
                conn.commit()
                log.info("PostgreSQL 접속 (%s:%s/%s)", os.environ["PG_HOST"], os.getenv("PG_PORT"), os.environ["PG_DB"])
            with lock:
                batch = [pg_pending.popleft() for _ in range(min(len(pg_pending), 2000))]
            try:
                with conn.cursor() as cur:
                    cur.executemany(PG_INSERT, [(r["ts"], r["measurement"], series_key(r["tags"]), Jsonb(r["tags"]),
                                                 Jsonb(r["fields"]), r["rx_delay_s"]) for r in batch])
                conn.commit()
                stats["pg_written"] += len(batch)
            except Exception:
                with lock:
                    pg_pending.extendleft(reversed(batch))
                raise
        except Exception as e:
            stats["pg_error"] += 1
            log.warning("PostgreSQL 쓰기 실패(보관 %d건, 10초 뒤 재시도): %s", len(pg_pending), e)
            try:
                conn and conn.close()
            except Exception:
                pass
            conn = None
            time.sleep(10)


def heartbeat():
    while not stop.wait(60):
        age = time.time() - state["last_msg"] if state["last_msg"] else -1
        fields = {"mqtt_connected": 1.0 if state["connected"] else 0.0,
                  "last_msg_age_s": float(round(age, 1)), "pending": float(len(pending)),
                  "pg_pending": float(len(pg_pending))}
        for k in ("received", "written", "duplicate", "parse_error", "unmatched", "write_error",
                  "handler_error", "disconnects", "pg_written", "pg_error"):
            fields[k] = float(stats[k])
        emit([dict(measurement="collector", ts=datetime.now(timezone.utc),
                   tags={"name": MQ["client_id"]}, fields=fields, rx_delay_s=None)])
        log.info("상태 %s pending=%d pg_pending=%d", dict(stats), len(pending), len(pg_pending))


def main():
    influx = InfluxDBClient(url=os.environ["INFLUX_URL"], token=os.environ["INFLUX_TOKEN"], org=ORG)
    write_api = influx.write_api(write_options=SYNCHRONOUS)

    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=MQ["client_id"],
                         clean_session=MQ.get("clean_session", False))
    client.username_pw_set(os.environ["MQTT_USER"], os.environ["MQTT_PASS"])
    client.reconnect_delay_set(MQ.get("reconnect_min", 1), MQ.get("reconnect_max", 30))
    client.on_connect, client.on_disconnect, client.on_message = on_connect, on_disconnect, on_message

    def shutdown(*_):
        log.info("종료 신호 수신, 남은 %d건 쓰고 종료", len(pending))
        stop.set()
        client.disconnect()
    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)

    threading.Thread(target=writer, args=(write_api,), daemon=True).start()
    threading.Thread(target=heartbeat, daemon=True).start()
    if PG:
        threading.Thread(target=pg_writer, daemon=True).start()
    client.connect_async(os.environ["MQTT_HOST"], int(os.getenv("MQTT_PORT", 1883)), keepalive=MQ.get("keepalive", 30))
    client.loop_start()
    stop.wait()
    client.loop_stop()
    with lock:
        rest = list(pending)
    if rest:
        try:
            write_api.write(bucket=BUCKET, org=ORG, record=rest)
        except Exception as e:
            log.warning("종료 중 쓰기 실패: %s", e)
    influx.close()


if __name__ == "__main__":
    main()
