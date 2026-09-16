#!/usr/bin/env python3
"""perf-demo: dual-chain inference performance showcase, burned into video.

Chain A (subscribe): platform-scheduled inference on the infer stream,
detections + metric lines burned by the daemon overlay onto the main
stream — zero pixels reach the app. Chain B (keep-fd): the app pulls
frames, runs its own pipeline, composites via DSP and pushes the sub
stream back. Same camera, same model, two integration paths, each chain
burning its own metrics through its own drawing path.

Run on-device: python3 app.py --model-path /path/to/model.hef
Needs the daemon yaml ``injection: enabled: true`` section for chain B
(deploy/install.sh appends it with a backup; uninstall.sh restores).
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import pathlib
import re
import signal
import sys
import threading
import time
import uuid

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from chains.a_subscribe import ChainA  # noqa: E402
from chains.b_keepfd import ChainB  # noqa: E402
from metrics import MetricsHub  # noqa: E402
from statusline import StatusLine, start_encoded_watcher  # noqa: E402

logger = logging.getLogger("perf_demo")

A_FIRST_RESULT_CANCEL_S = 20.0   # watchdog cancels a never-yielding subscribe
A_FIRST_RESULT_DEAD_S = 60.0     # ... then declares the path dead (NA display)
WATCHDOG_PERIOD_S = 1.0


def model_id_for(path: str) -> str:
    """Filename-derived id (perf_common.perf_model_id semantics): a fixed
    id is a register trap — daemon keeps old bindings and self-heal
    resurrects owner-less entries under the same id."""
    stem = os.path.basename(path).rsplit(".", 1)[0]
    return f"perf-demo-{stem}"


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model-path", required=True, help="HEF/ONNX model file")
    p.add_argument("--a-stream", default="third", help="chain A infer stream")
    p.add_argument("--a-display", default="main", help="chain A burn-in display stream")
    p.add_argument("--a-fps", type=int, default=10, help="chain A subscribe fps")
    p.add_argument("--b-stream", default="sub", help="chain B keep-fd source stream")
    p.add_argument("--b-skip", type=int, default=0,
                   help="chain B skip_frames pacing (0 = every frame: the "
                        "publisher must match the encoder rate or gaps pass "
                        "the live picture through, flickering the burn-in)")
    p.add_argument("--annotate-hz", type=float, default=3.0, help="chain A OSD refresh")
    p.add_argument("--status-interval", type=float, default=5.0, help="status tick (s)")
    p.add_argument("--duration", type=float, default=0.0, help="stop after N s (0 = run)")
    p.add_argument("--pool-depth", type=int, default=4, help="injection pool depth")
    p.add_argument("--min-score", type=float, default=0.3, help="detection threshold")
    p.add_argument("--zero-copy", action="store_true",
                   help="EXPERIMENTAL: blend directly on the keep-fd handle")
    p.add_argument("--json-path", default="/run/aipc/perf-demo.json",
                   help="operator status JSON ('' disables)")
    p.add_argument("--watch-encoded", default="main,sub",
                   help="encoded streams to gap-watch ('' disables)")
    p.add_argument("--session-id", default="perf-demo", help="overlay session tag")
    p.add_argument("-v", "--verbose", action="store_true")
    p.add_argument("--chains", choices=("none", "a", "b", "ab"), default="ab")
    p.add_argument("--b-fps", type=float, default=0, help="admission fps; 0 = unlimited")
    p.add_argument("--publish-hz", type=float, default=40)
    p.add_argument("--no-metrics-overlay", action="store_true")
    p.add_argument("--no-detections-overlay", action="store_true")
    p.add_argument("--samples-path", default="", help="append-only JSONL; empty disables")
    p.add_argument("--samples-max-mb", type=int, default=1024)
    p.add_argument("--run-id", default=uuid.uuid4().hex)
    p.add_argument("--phase", default="measure")
    p.add_argument("--model-id", default=None, help="test-exclusive model registration ID")
    p.add_argument("--variant", default=None,
                   help="postprocess variant (real dlsym function name, e.g. "
                        "hailo_yolov8n); without it vendor HEFs draw empty "
                        "boxes — inference runs but no detections come back")
    p.add_argument("--reuse-model", action="store_true", help="require matching existing model; never unregister")
    p.add_argument("--keep-model", action="store_true", help="leave our new registration resident")
    args = p.parse_args(argv)
    bounds = dict(a_fps=(1, 120), b_fps=(0, 120), publish_hz=(1, 120),
                  annotate_hz=(0.1, 120), status_interval=(0.1, 3600),
                  duration=(0, 604800), pool_depth=(1, 8), min_score=(0, 1),
                  b_skip=(0, 10000), samples_max_mb=(1, 65536))
    for name, (low, high) in bounds.items():
        value = getattr(args, name)
        if not math.isfinite(value) or not low <= value <= high:
            p.error(f"--{name.replace('_', '-')} must be in [{low}, {high}]")
    for name in ("run_id", "phase", "session_id", "a_stream", "a_display", "b_stream"):
        value = getattr(args, name)
        if not value or len(value) > 128 or any(ord(c) < 32 for c in value):
            p.error(f"{name} must be 1..128 printable characters")
    if args.model_id is not None and not re.fullmatch(r"[A-Za-z0-9_.-]{1,128}", args.model_id):
        p.error("model-id must use 1..128 letters/digits/underscore/dot/hyphen")
    if args.variant is not None and not re.fullmatch(r"[A-Za-z0-9_.:-]{1,128}", args.variant):
        p.error("variant must use 1..128 letters/digits/underscore/dot/colon/hyphen")
    if args.reuse_model and not args.model_id:
        p.error("--reuse-model requires an explicit --model-id")
    if args.samples_path and args.json_path and os.path.abspath(args.samples_path) == os.path.abspath(args.json_path):
        p.error("samples-path and json-path must differ")
    return args


class PerfDemoApp:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.hub = MetricsHub()
        self.model_id = args.model_id or f"{model_id_for(args.model_path)[:80]}-{uuid.uuid4().hex[:12]}"
        self._owns_model = False
        self._model_state = dict(model_kept=None, model_kept_reason="not_inspected",
                                 model_ownership="unknown")
        self._cleanup_errors = []
        self._threads = []
        self._clients = []
        self._stop = threading.Event()
        self._chain_a = None
        self._watchers: list = []

    def run(self) -> int:
        from neoruntime_ipc_sdk import (
            CameraClient,
            DspClient,
            FdMediaClient,
            InferenceClient,
        )

        args = self.args
        logging.basicConfig(
            level=logging.DEBUG if args.verbose else logging.INFO,
            format="%(asctime)s %(name)s %(levelname)s %(message)s")
        logger.info("perf-demo starting: model=%s id=%s", args.model_path, self.model_id)

        from recorder import SampleRecorder
        if args.samples_path:
            self.hub.recorder = SampleRecorder(args.samples_path, run_id=args.run_id,
                                               phase=args.phase, max_bytes=args.samples_max_mb * 1024 * 1024)
        infer = status = None
        exit_code = 0
        old_signals = {}
        try:
            self.hub.emit("run_start", chains=args.chains, model_id=self.model_id,
                          controls={k: v for k, v in vars(args).items()
                                    if k not in ("model_path", "samples_path", "json_path")},
                          clock_domains={"monotonic_ns": "local_CLOCK_MONOTONIC",
                                         "a_result_timestamp_ns": "unknown",
                                         "b_source_timestamp_ns": "device_CLOCK_MONOTONIC"},
                          b_source_age_requires_same_host=True, display_latency_ms=None)
            camera = self._client(CameraClient)
            infer = self._client(InferenceClient)
            self._prepare_model(infer)
            self._start_chains(camera, infer, DspClient, FdMediaClient)
            status = StatusLine(hub=self.hub, camera=camera, infer=infer, model_id=self.model_id,
                                json_path=args.json_path, interval=args.status_interval,
                                a_display=args.a_display, b_stream=args.b_stream)
            self._threads.append(status)
            status.start()
            for stream_id in filter(None, (s.strip() for s in args.watch_encoded.split(","))):
                self._watchers.append(start_encoded_watcher(stream_id, self.hub))
            for sig in (signal.SIGTERM, signal.SIGINT):
                old_signals[sig] = signal.signal(sig, self._on_signal)
            self._chains_started_at = time.monotonic()
            self._watchdog(args.duration)
        except Exception:
            exit_code = 1
            logger.exception("perf-demo failed")
        finally:
            exit_code = self._teardown(infer, status, exit_code)
            for sig, handler in old_signals.items():
                signal.signal(sig, handler)
        return exit_code

    def _client(self, factory):
        client = factory()
        self._clients.append(client)
        return client

    def _prepare_model(self, infer) -> None:
        import grpc

        from neoruntime_ipc_sdk.config import Config
        try:
            existing = infer.get_model_info(self.model_id)
        except grpc.aio.AioRpcError as exc:
            if exc.code() != grpc.StatusCode.NOT_FOUND:
                raise
            existing = None
        if existing is not None:
            self._model_state = dict(model_kept=True, model_kept_reason="existing_registration_untouched",
                                     model_ownership="preexisting")
            expected = os.path.normpath(Config.translate_path_to_host(self.args.model_path))
            if not self.args.reuse_model or os.path.normpath(existing.model_path) != expected:
                raise ValueError("model ID exists; explicit reuse with matching path required")
            self._model_state = dict(model_kept=True, model_kept_reason="reused_registration",
                                     model_ownership="reused")
            self.hub.emit("model_state", model_id=self.model_id, reused=True,
                          load_timestamp=getattr(existing, "load_timestamp", None))
            return
        self._model_state = dict(model_kept=False, model_kept_reason="not_registered",
                                 model_ownership="none")
        if self.args.reuse_model:
            raise ValueError("reuse requested but model ID does not exist")
        self._model_state = dict(model_kept=None, model_kept_reason="registration_outcome_unknown",
                                 model_ownership="unknown")
        infer.register_model(self.args.model_path, model_id=self.model_id,
                             owner_id=self.args.session_id, model_type="detection",
                             model_variant=self.args.variant or None)
        self._owns_model = True
        self._model_state = dict(model_kept=True, model_kept_reason="registered_by_run",
                                 model_ownership="created_by_run")
        self.hub.emit("model_state", model_id=self.model_id, reused=False)

    def _cleanup_model(self, infer) -> None:
        if not self._owns_model:
            return
        if self.args.keep_model:
            self._model_state = {**self._model_state, "model_kept": True,
                                 "model_kept_reason": "keep_requested"}
            return
        # A failed RPC may have reached the daemon. Do not equate an exception
        # with either successful unregister or a definitely retained model.
        self._model_state = {**self._model_state, "model_kept": None,
                             "model_kept_reason": "unregister_outcome_unknown"}
        infer.unregister_model(self.model_id)  # SDK gRPC, never REST DELETE
        self._owns_model = False
        self._model_state = {**self._model_state, "model_kept": False,
                             "model_kept_reason": "unregistered"}

    def _start_chains(self, camera, infer, dsp_factory, media_factory) -> None:
        args = self.args
        common = dict(camera=camera, infer=infer, hub=self.hub, model_id=self.model_id,
                      min_score=args.min_score, session_id=args.session_id,
                      metrics_overlay=not args.no_metrics_overlay,
                      detections_overlay=not args.no_detections_overlay)
        if "b" in args.chains:
            b = ChainB(**common, dsp=self._client(dsp_factory), media=self._client(media_factory),
                       stream_id=args.b_stream, skip=args.b_skip, pool_depth=args.pool_depth,
                       zero_copy=args.zero_copy, fps=args.b_fps, publish_hz=args.publish_hz)
            self._threads.append(b)
            b.start()
        if "a" in args.chains:
            self._chain_a = ChainA(**common, stream_id=args.a_stream, display_stream=args.a_display,
                                   fps=args.a_fps, annotate_hz=args.annotate_hz)
            self._threads.append(self._chain_a)
            self._chain_a.start()

    # -- lifecycle --

    def _on_signal(self, signum, _frame) -> None:
        logger.info("signal %s received", signum)
        self._stop.set()

    def _watchdog(self, duration: float) -> None:
        """Probe/stall policy for chain A + lifetime control.

        A blocked subscribe (dead -2814 path) never returns to this loop
        on its own: after A_FIRST_RESULT_CANCEL_S we cancel the iterator
        (wakes the chain thread, which reconnects), and after
        A_FIRST_RESULT_DEAD_S with still zero results we degrade chain A
        — the OSD shows NA and chain B keeps the demo alive.
        """
        cancelled_once = False
        while not self._stop.wait(WATCHDOG_PERIOD_S):
            now = time.monotonic()
            a = self.hub.a
            results = a.counters.get("results") or 0
            degraded = a.counters.get("degraded_reason")
            elapsed = now - self._chains_started_at
            if results == 0 and not degraded and self._chain_a is not None:
                if elapsed > A_FIRST_RESULT_DEAD_S:
                    reason = (f"subscribe path dead: no first result "
                              f"in {A_FIRST_RESULT_DEAD_S:.0f}s")
                    logger.warning("chain A: %s", reason)
                    self._chain_a.degrade(reason)
                elif elapsed > A_FIRST_RESULT_CANCEL_S and not cancelled_once:
                    cancelled_once = True
                    logger.warning(
                        "chain A: no first result in %.0fs, cancelling iterator",
                        A_FIRST_RESULT_CANCEL_S)
                    self._chain_a.cancel_iter()
            if duration and elapsed > duration:
                logger.info("duration %.0fs reached", duration)
                self._stop.set()
                break

    def _teardown(self, infer, status, exit_code: int) -> int:
        workers = self._threads + self._watchers
        for worker in workers:
            self._cleanup_step(worker.stop)
        deadline = time.monotonic() + 15.0
        for worker in workers:
            if worker.ident is not None:
                worker.join(timeout=max(0, deadline - time.monotonic()))
        alive = [w.name for w in workers if w.is_alive()]
        b_cleanup = self.hub.b.counters.get("cleanup_error")
        if b_cleanup:
            self._cleanup_errors.append(b_cleanup)
        a_cleanup = self.hub.a.counters.get("cleanup_error")
        if a_cleanup:
            self._cleanup_errors.append(a_cleanup)
        self._cleanup_errors.extend(w.cleanup_error or w.terminal_error for w in self._watchers
                                    if w.cleanup_error or w.terminal_error)
        # A timed-out worker can still own an FD/lease/RPC. Do not close its
        # resources underneath it or unregister a model it may still use.
        if not alive and not b_cleanup:
            if infer is not None:
                self._cleanup_step(lambda: self._cleanup_model(infer))
            for client in reversed(self._clients):
                self._cleanup_step(client.close)
        elif self._owns_model:
            reason = "cleanup_skipped_active_workers" if alive else "cleanup_skipped_b_error"
            self._model_state = {**self._model_state, "model_kept": True,
                                 "model_kept_reason": reason}
        degraded = any(self.hub.snapshot()[c]["degraded_reason"] for c in ("a", "b"))
        exit_code = int(bool(exit_code or alive or self._cleanup_errors or degraded))
        final = dict(exit_code=exit_code, alive_threads=alive, cleanup_errors=self._cleanup_errors,
                     **self._model_state)
        self.hub.record_final("failed" if exit_code else "stop")
        self.hub.emit("run_exit", **final, snapshot=self.hub.snapshot())
        recorder = self.hub.recorder
        if recorder:
            if not recorder.close(exit_code=exit_code) or recorder.snapshot()["error"]:
                exit_code = 1
                self.hub.record_final("recorder_failed")
        if status is not None:
            status._write_json({**self.hub.snapshot(), "exit_status": {**final, "exit_code": exit_code}})
        return exit_code

    def _cleanup_step(self, action) -> None:
        try:
            action()
        except Exception as exc:
            self._cleanup_errors.append(type(exc).__name__)
            logger.warning("cleanup failed: %s", type(exc).__name__)


def main() -> int:
    args = parse_args()
    if not os.path.isfile(args.model_path):
        print(f"model not found: {args.model_path}", file=sys.stderr)
        return 2
    app = PerfDemoApp(args)
    try:
        return app.run()
    except Exception:  # noqa: BLE001 - top-level: log and exit non-zero
        logger.exception("perf-demo fatal")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
