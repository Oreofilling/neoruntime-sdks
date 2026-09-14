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
import os
import signal
import sys
import pathlib
import threading
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from metrics import MetricsHub  # noqa: E402

from chains.a_subscribe import ChainA  # noqa: E402
from chains.b_keepfd import ChainB  # noqa: E402
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
    return p.parse_args(argv)


class PerfDemoApp:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.hub = MetricsHub()
        self.model_id = model_id_for(args.model_path)
        self._stop = threading.Event()
        self._chain_a = None
        self._watchers: list = []

    def run(self) -> int:
        from neoruntime_ipc_sdk import (
            CameraClient,
            DspClient,
            FdMediaClient,
            InferenceClient,
            OverlayClient,
        )

        args = self.args
        logging.basicConfig(
            level=logging.DEBUG if args.verbose else logging.INFO,
            format="%(asctime)s %(name)s %(levelname)s %(message)s")
        logger.info("perf-demo starting: model=%s id=%s", args.model_path, self.model_id)

        camera = CameraClient()
        infer = InferenceClient()
        dsp = DspClient()
        media = FdMediaClient()
        overlay = OverlayClient()

        infer.register_model(
            args.model_path,
            model_id=self.model_id,
            owner_id=args.session_id,
            model_type="detection",
        )
        overlay.enable(show_label=True, show_confidence=False, line_thickness=2)

        hub = self.hub
        chain_a = ChainA(
            camera=camera, infer=infer, hub=hub,
            stream_id=args.a_stream, display_stream=args.a_display,
            model_id=self.model_id, fps=args.a_fps,
            annotate_hz=args.annotate_hz, min_score=args.min_score,
            session_id=args.session_id)
        chain_b = ChainB(
            camera=camera, infer=infer, dsp=dsp, media=media, hub=hub,
            stream_id=args.b_stream, model_id=self.model_id,
            min_score=args.min_score, skip=args.b_skip,
            pool_depth=args.pool_depth, zero_copy=args.zero_copy,
            session_id=args.session_id)
        status = StatusLine(
            hub=hub, camera=camera, infer=infer, model_id=self.model_id,
            json_path=args.json_path, interval=args.status_interval,
            a_display=args.a_display, b_stream=args.b_stream)
        for stream_id in filter(None, args.watch_encoded.split(",")):
            self._watchers.append(start_encoded_watcher(stream_id.strip(), hub))

        signal.signal(signal.SIGTERM, self._on_signal)
        signal.signal(signal.SIGINT, self._on_signal)

        self._chain_a = chain_a
        self._chains_started_at = time.monotonic()
        chain_b.start()
        chain_a.start()
        status.start()

        try:
            self._watchdog(args.duration)
        finally:
            logger.info("tearing down")
            self._teardown(chain_a, chain_b, status, overlay, infer, media,
                           dsp, camera)
        return 0

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

    def _teardown(self, chain_a, chain_b, status, overlay, infer, media,
                  dsp, camera) -> None:
        self.hub.record_final("stop")
        chain_b.stop()
        chain_a.stop()
        status.stop()
        for t in (chain_b, chain_a, status):
            t.join(timeout=15.0)
        try:
            status.tick()  # final JSON with the 'final' section
        except Exception:  # noqa: BLE001
            pass
        try:
            overlay.disable()
        except Exception:  # noqa: BLE001
            logger.debug("overlay disable failed", exc_info=True)
        try:
            infer.unregister_model(self.model_id)  # before infer.close()
        except Exception as exc:  # noqa: BLE001
            logger.warning("model unregister failed: %s", exc)
        for closer in (media.close, infer.close, dsp.close, camera.close):
            try:
                closer()
            except Exception:  # noqa: BLE001
                logger.debug("close step failed", exc_info=True)
        logger.info("perf-demo stopped (up %.0fs)", self.hub.uptime_s())


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
