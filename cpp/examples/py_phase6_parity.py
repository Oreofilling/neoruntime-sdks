#!/usr/bin/env python3
"""py_phase6_parity.py — Python-SDK side of the Phase 6 smoke (冒烟④).

Runs the exact same observable sequence as cpp/examples/phase6_smoke.cpp
against the same live daemons, so the two outputs can be diffed:

  [1] connectivity + endpoints
  [2] typed model registration (pre-clean)
  [3] overlay event observer on inference/third
  [4] StreamPipeline(stream, model, fps=5, ttl=2000, session, 1 zone):
      start -> zone push; results (or the daemon's documented -2811 DMA
      rejection surfaced through status().last_error); stop -> clears
  [5] observer verdict: operator writes untagged, result writes tagged

Exit 0 only if no section FAILed (SKIP is acceptable and explained).
"""
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeout

from neoruntime_ipc_sdk import (
    Config,
    EventClient,
    InferenceClient,
    StreamPipeline,
)

MODEL_PATH = "/data/aipc-data/models/hailo_yolov8n_384_640.hef"
MODEL_ID = "py-p6-y8n"
STREAM = "third"
SESSION = "py-p6"

failures = 0
skips = 0

seen_mu = threading.Lock()
seen = []


def pass_(what):
    print(f"  [PASS] {what}")


def fail(what, err):
    global failures
    # stdout, not stderr: both smoke logs get diffed, and cross-stream
    # interleaving scrambles the section order
    print(f"  [FAIL] {what}: {err}")
    failures += 1


def skip_(what, why):
    global skips
    print(f"  [SKIP] {what}: {why}")
    skips += 1


def record(ev):
    payload = ev.payload or {}
    seen.append(
        {
            "session": ev.metadata.get("session_id", ""),
            "ttl": ev.metadata.get("result_ttl_ms", ""),
            "has_polygons": "polygons" in payload,
            "polygons_empty": not payload.get("polygons"),
            "has_detections": "num_detections" in payload,
            "num_detections": payload.get("num_detections", -1),
        }
    )


def seen_snapshot():
    with seen_mu:
        return list(seen)


def wait_seen(count, timeout_s=8.0):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if len(seen_snapshot()) >= count:
            return True
        time.sleep(0.05)
    return len(seen_snapshot()) >= count


def main():
    print("[1] connectivity")
    print(f"  inference={Config.get_inference_endpoint()}")
    print(f"  events={Config.get_event_bus_endpoint()}")
    print(f"  camera={Config.get_camera_control_endpoint()}")
    inf = InferenceClient()
    try:
        inf.connect()
        models_before = len(inf.list_models())
        pass_(f"connect + list_models ({models_before} models)")
    except Exception as exc:  # noqa: BLE001 — smoke reports everything
        fail("connect", exc)
        print(f"=== Phase 6 parity: FAIL (failures=1) ===")
        return 1

    print("[2] typed model registration")
    registered = False
    try:
        for m in inf.list_models():
            if m.model_id == MODEL_ID:
                inf.unregister_model(MODEL_ID)
        inf.register_model(MODEL_PATH, MODEL_ID, owner_id="", model_type="detection")
        registered = True
        pass_("register_model(typed detection, no owner_id)")
    except Exception as exc:  # noqa: BLE001
        fail("register_model", exc)

    print(f"[3] overlay event observer on '{STREAM}'")
    bus = EventClient()
    # Pull subscription + registry handshake, NOT on_event: on_event's
    # detached thread races start()'s synchronous zone push (observed on
    # 93.72: the push is published before the daemon registered the
    # subscriber and is missed). Python's subscribe() is a lazy generator,
    # so the drainer thread (which starts the RPC) must run before the
    # handshake; polling get_topic_info() until subscriber_count grows
    # proves the daemon's registry holds us.
    try:
        bus.connect()
        topic = f"inference/{STREAM}"
        baseline = 0
        for t in bus.list_topics():
            if t.topic == topic:
                baseline = t.subscriber_count
        events = bus.subscribe(topic)

        def drain():
            for ev in events:
                record(ev)

        threading.Thread(target=drain, daemon=True).start()
        live = False
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            time.sleep(0.05)
            info = bus.get_topic_info(topic)
            if info is not None and info.subscriber_count > baseline:
                live = True
                break
        if live:
            pass_("subscribed + registry handshake (outside vantage)")
        else:
            fail("observer handshake", "subscriber never appeared in topic registry")
    except Exception as exc:  # noqa: BLE001
        fail("event observer", exc)

    print("[4] StreamPipeline live run")
    dma_reject = ""
    try:
        # python polygons are plain dicts: {"points": [[x, y], ...], ...}
        # (the C++ smoke uses the typed OverlayPolygon/OverlayPoint structs)
        zone = {
            "points": [[0.05, 0.05], [0.95, 0.05], [0.95, 0.95], [0.05, 0.95]],
            "label": "smoke-zone",
        }
        pipe = StreamPipeline(
            STREAM,
            MODEL_ID,
            fps=5,
            session_id=SESSION,
            raw_output_only=False,
            max_consecutive_failures=10,
            queue_size=8,
            draw=True,
            overlay_config=None,
            ttl_ms=2000,
            min_score=0.3,
            labels=None,
            polygons=[zone],
        )
        pipe.start()
        zone_pushed = wait_seen(1)
        if zone_pushed and seen_snapshot()[0]["has_polygons"]:
            pass_("start(): static zone pushed on the bus (untagged)")
        else:
            fail("zone push", "observer saw no polygon event after start()")

        got_result = False
        try:
            # python's read API is the results() iterator (0.8.0 and 0.8.1);
            # the C++ port is next_result() per the C++ SDK's pull-stream
            # next() -> optional convention. Boxed with a timeout so a
            # daemon stall cannot hang the smoke.
            with ThreadPoolExecutor(max_workers=1) as pool:
                fut = pool.submit(lambda: next(pipe.results(), None))
                try:
                    item = fut.result(timeout=30)
                    got_result = item is not None
                    if got_result:
                        seq, result = item
                        print(f"  first result: seq={seq} objects={len(result.objects)}")
                except FuturesTimeout:
                    pipe.stop()  # stop marker ends the blocked pull
                    fut.result(timeout=10)
                    fail("results", "no terminal state within 30s")
        except Exception as exc:  # noqa: BLE001 — the daemon's stream error
            print(f"  results() raised: {exc}")

        st = pipe.status()
        print(
            f"  status: running={int(st.running)} seen={st.results_seen} "
            f"annotated={st.results_annotated} drops={st.result_queue_drops} "
            f"skew_last={st.last_skew_us}us latency_last={st.last_latency_ms:.1f}ms"
        )
        if st.last_error:
            print(f"  last_error: {st.last_error}")

        if got_result and st.results_seen > 0:
            pass_("results flowed end-to-end (subscribe -> overlay -> app queue)")
        elif "2811" in str(st.last_error) or "DMA input rejected" in str(st.last_error):
            dma_reject = st.last_error
            skip_(
                "camera-fed results",
                "daemon rejects DMA inputs on this build (documented -2811); "
                "SDK surfaced it through status().last_error after "
                "max_consecutive_failures, the parity contract",
            )
        else:
            fail("results", st.last_error or "no results and no error")

        before_stop = len(seen_snapshot())
        pipe.stop()
        cleared = wait_seen(before_stop + 2)
        if cleared:
            pass_("stop(): box + zone clears published")
        else:
            fail("teardown clears", "observer did not see both clear events")
    except Exception as exc:  # noqa: BLE001
        fail("StreamPipeline run", exc)

    print("[5] observer verdict")
    try:
        evs = seen_snapshot()
        zone_push = clear_boxes = clear_zones = tagged = 0
        for s in evs:
            if s["has_polygons"] and not s["polygons_empty"] and not s["session"]:
                zone_push += 1
            if (
                s["has_detections"]
                and s["num_detections"] == 0
                and not s["has_polygons"]
                and not s["session"]
            ):
                clear_boxes += 1
            if s["has_polygons"] and s["polygons_empty"] and not s["session"]:
                clear_zones += 1
            if s["session"]:
                tagged += 1
                if s["session"] != SESSION:
                    fail("session tag", f"unexpected session '{s['session']}'")
                if s["ttl"] != "2000":
                    fail("result ttl", f"metadata ttl='{s['ttl']}' expected 2000")
        if zone_push == 1:
            pass_("exactly one untagged zone push")
        else:
            fail("zone push count", str(zone_push))
        if clear_boxes == 1 and clear_zones == 1:
            pass_("exactly one untagged box clear + one zone clear at teardown")
        else:
            fail("clear counts", f"boxes={clear_boxes} zones={clear_zones}")
        if tagged == 0 and not dma_reject:
            fail("result tagging", "results flowed but none carried the session tag")
        elif tagged > 0:
            pass_(f"{tagged} tagged result write(s), ttl forwarded")
        else:
            skip_("result tagging", "no results on this daemon build (see above)")
        print(f"  observer total events: {len(evs)}")
    except Exception as exc:  # noqa: BLE001
        fail("observer verdict", exc)

    if registered:
        # retry: the daemon releases the model reference from the just-
        # stopped subscription asynchronously (observed: immediate call
        # after stop() fails with "Failed to unregister")
        unregistered = False
        last_err = ""
        for _ in range(6):
            time.sleep(0.5)
            try:
                inf.unregister_model(MODEL_ID)
                unregistered = True
                break
            except Exception as exc:  # noqa: BLE001
                last_err = str(exc)
        if unregistered:
            pass_("unregister_model")
        else:
            fail("unregister_model", last_err)

    verdict = "PASS" if failures == 0 else "FAIL"
    print(f"=== Phase 6 parity: {verdict} (failures={failures} skips={skips}) ===")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
