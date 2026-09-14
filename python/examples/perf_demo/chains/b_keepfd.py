"""Chain B: app-owned pipeline over keep-fd frames, composited via DSP.

Per frame: keep-fd subscribe → DSP stretch-resize to the model's exact
NV12 input geometry → infer (raw tensor path) → render_overlay_rgba
boxes + a text chip for the metric line → blend_hw → the latest
composed frame goes to the FramePublisher.

Input contract (learned the hard way, -2811 on every frame): the raw
infer path checks byte_size, it does NOT resize — the tensor must be
the model's exact NV12 geometry (e.g. 640x384 = 368640 bytes for this
model). The SDK Preprocessor is RGB-only and cannot produce that, so
this chain resizes on the DSP itself (stretch, NOT letterbox: a
stretch keeps model-normalized boxes identical to source-normalized,
so server-decoded objects render on the source frame unmapped). The
resize is the app's pixel cost — paid deliberately, that cost IS the
chain-B story the demo shows.

Publishing is PACED off the compose loop: the pipeline sustains ~13 fps
while the sub encoder runs at 30 — a publish-per-compose would leave
gaps that pass the live picture through, flickering the burn-in (the
daemon bakes each pushed frame and falls back to live between pushes).
So a pacer thread republishes the newest composed frame at PUBLISH_HZ,
each stamped PUBLISH_LEAD_NS into the future: the daemon HOLDS
not-yet-due frames and its encoder pickup takes the newest due one, so
a 1-2 deep queue of future frames rides out arrival gaps (GIL storms
in the compose loop) — measured 100% slot coverage, where due-now
publishing left 5-15% of slots flashing the live picture. The B line
honestly reports the compose rate as its fps. (Zero-copy mode
publishes inline per compose instead: its composed output IS the
frame's dmabuf, which the frame lock releases.)

Frame discipline: every frame lives inside ``with frame:`` — never more
than one retained (the daemon pool starves at >=8). End-to-end latency
subtracts the frame's device-CLOCK_MONOTONIC timestamp from a local
clock_gettime reading: same clock domain on-device, so the subtraction
is exact.

Failure policy: per-frame inference errors are counted and skipped
(stream keeps flowing); blend/compose failures degrade the chain for a
30 s cooldown while the pipeline and publisher rebuild, then retry; ten
consecutive paced-publish failures degrade terminally. Teardown always
publishes EOS so the encoder returns to the live source.
"""

from __future__ import annotations

import logging
import threading
import time

from display import format_b_line, text_chip_rgba
from neoruntime_ipc_sdk import InferenceClient
from neoruntime_ipc_sdk.draw import render_overlay_rgba

logger = logging.getLogger("perf_demo.chain_b")

CHIP_REFRESH_S = 0.333   # metric text re-renders at ~3 Hz, not per frame
DEGRADED_COOLDOWN_S = 30.0
CHIP_MARGIN_PX = 16
PUBLISH_HZ = 40.0        # ~33% over the 30 fps encoder: every pickup is
                        # guaranteed a fresh arrival (p90 publish tail)
PUBLISH_LEAD_NS = 40_000_000  # stamp each frame ~1 slot ahead: the daemon
                        # holds not-yet-due frames, so a small queue of
                        # future frames rides out publish arrival gaps
                        # (GIL storms in the compose loop) that a
                        # due-immediately stream would miss
PUBLISH_FAIL_LIMIT = 10


class ChainB(threading.Thread):
    """Own pipeline: pull keep-fd frames, infer, composite, push back."""

    def __init__(self, *, camera, infer, dsp, media, hub,
                 stream_id: str = "sub", model_id: str = "",
                 min_score: float = 0.3, skip: int = 1,
                 pool_depth: int = 4, zero_copy: bool = False,
                 session_id: str = "perf-demo") -> None:
        super().__init__(name="chain-b", daemon=True)
        self.camera = camera
        self.infer = infer
        self.dsp = dsp
        self.media = media
        self.hub = hub
        self.stream_id = stream_id
        self.model_id = model_id
        self.min_score = min_score
        self.skip = skip
        self.pool_depth = pool_depth
        self.zero_copy = zero_copy
        self.session_id = session_id
        self._stop = threading.Event()
        self._publisher = None
        self._infer_wh: tuple[int, int] | None = None
        self._chip: tuple | None = None   # (rgba, x, y), rebuilt at 3 Hz
        self._chip_t = 0.0
        self._latest_lock = threading.Lock()
        self._latest = None               # (composed, pts_ns) for the pacer
        self._pacer: threading.Thread | None = None
        self._publish_failures = 0

    def stop(self) -> None:
        self._stop.set()

    # -- thread body --

    def run(self) -> None:  # noqa: C901 - rebuild loop + teardown is the story
        try:
            while not self._stop.is_set() \
                    and self.hub.b.counters.get("degraded_reason") is None:
                try:
                    self._prepare_infer()
                    self._ensure_publisher()
                    self._frame_loop()
                except Exception as exc:  # noqa: BLE001 - degrade, cool, rebuild
                    reason = f"{type(exc).__name__}: {exc}"
                    logger.warning("chain B degraded: %s", reason)
                    self.hub.b.counters.set("degraded_reason", reason[:120])
                    self._close_publisher(eos=False)
                    self._sleep_stop_aware(DEGRADED_COOLDOWN_S)
                    if self._stop.is_set():
                        break
                    self.hub.b.counters.set("degraded_reason", None)
                    self.hub.b.counters.set(
                        "rebuilds", (self.hub.b.counters.get("rebuilds") or 0) + 1)
        finally:
            self._close_publisher(eos=True)

    def _prepare_infer(self) -> None:
        """Resolve the model's exact NV12 input geometry once.

        The raw infer path is a byte_size contract (no daemon-side
        resize), so the tensor must match the model input to the byte.
        """
        info = self.infer.get_model_info(self.model_id)
        inp = (getattr(info, "inputs", None) or [None])[0]
        shape = list(getattr(inp, "shape", None) or []) \
            if not isinstance(inp, dict) else list(inp.get("shape") or [])
        if shape and shape[0] == 1:
            shape = shape[1:]
        if len(shape) != 3:
            raise ValueError(
                f"model {self.model_id!r} input shape not NHWC: {shape}")
        _h, w, _c = shape
        self._infer_wh = (int(w), int(_h))

    def _ensure_publisher(self) -> None:
        from neoruntime_ipc_sdk import FramePublisher

        self._publisher = FramePublisher(
            self.camera, self.dsp,
            stream_id=self.stream_id,
            pool_depth=self.pool_depth,
            mode="replace", fmt="nv12",
            session_id=self.session_id,
        )
        self._publish_failures = 0
        if not self.zero_copy:  # zero-copy publishes inline (see _frame_loop)
            self._pacer = threading.Thread(
                target=self._pacer_loop, name="chain-b-pub", daemon=True)
            self._pacer.start()
        b = self.hub.b
        b.counters.set("pool_depth", self.pool_depth)
        b.counters.set("lease_mode",
                       "lease" if self._publisher.lease_mode else "legacy")

    def _pacer_loop(self) -> None:
        """Republish the newest composed frame at PUBLISH_HZ so every
        encode slot bakes composed pixels (a slot with no arrival since
        the previous pickup passes the live picture through — that was
        the 42%-duty flicker).

        Deliberately FREE-RUNNING, not packet-pulse-locked: the encoded
        packet reaches this app only ~46 ms after its pts (encode
        pipeline latency), leaving ~17 ms before the next pickup — less
        than the publish tail (p90 26 ms, GIL/daemon jitter), so
        pulse-locked publishing missed ~13% of slots anyway (measured).
        Over-rate + the daemon's newest-due-wins pickup (older due
        frames are superseded and merely counted as drops) covers every
        slot without needing to win the arrival race. The 0.1 s
        stop-poll keeps teardown responsive."""
        interval = 1.0 / PUBLISH_HZ
        while not self._stop.is_set() and self._publisher is not None:
            t0 = time.perf_counter()
            with self._latest_lock:
                latest = self._latest
            pub = self._publisher
            if latest is not None and pub is not None:
                composed, _pts = latest
                try:
                    due = time.clock_gettime_ns(time.CLOCK_MONOTONIC) \
                        + PUBLISH_LEAD_NS
                    pub.publish(composed, pts_ns=due)
                    self._publish_failures = 0
                    self.hub.b.pub_ms.add(
                        (time.perf_counter() - t0) * 1000.0)
                    n = (self.hub.b.counters.get("publishes") or 0) + 1
                    self.hub.b.counters.set("publishes", n)
                except Exception as exc:  # noqa: BLE001
                    self._publish_failures += 1
                    logger.debug("paced publish failed: %s", exc)
                    if self._publish_failures >= PUBLISH_FAIL_LIMIT:
                        self.hub.b.counters.set(
                            "degraded_reason",
                            f"publish dead: {exc}"[:120])
                        return
            self._stop.wait(max(0.0, interval - (time.perf_counter() - t0)))

    def _frame_loop(self) -> None:
        for frame in self.media.subscribe(
                self.stream_id, skip_frames=self.skip, keep_fd=True):
            if self._stop.is_set() \
                    or self.hub.b.counters.get("degraded_reason") is not None:
                break
            with frame:
                t0 = time.perf_counter()
                base = frame if self.zero_copy else frame.to_array()
                t1 = time.perf_counter()
                out = self._infer_or_count(frame)
                t2 = time.perf_counter()
                overlays = self._overlays_for(frame, out)
                composed = self.dsp.blend_hw(base, overlays,
                                             zero_copy=self.zero_copy)
                t3 = time.perf_counter()
                if self.zero_copy:
                    # composed IS the frame's dmabuf here, released at
                    # with-exit — publish inline; the pacer must never
                    # touch a released handle
                    self._publisher.publish(composed, pts_ns=frame.timestamp_ns)
                    pub_ms = (time.perf_counter() - t3) * 1000.0
                else:
                    with self._latest_lock:
                        self._latest = (composed, frame.timestamp_ns)
                    pub_ms = None  # paced publishes time themselves
                e2e_ms = (time.clock_gettime_ns(time.CLOCK_MONOTONIC)
                          - frame.timestamp_ns) / 1e6
            objs = (out.objects or []) if out is not None else []
            hw_us = getattr(out, "infer_time_us", 0) if out is not None else 0
            self.hub.b.record_frame(
                pull_ms=(t1 - t0) * 1000.0,
                infer_ms=(t2 - t1) * 1000.0,
                hw_infer_ms=hw_us / 1000.0 if hw_us else None,
                draw_ms=(t3 - t2) * 1000.0,
                pub_ms=pub_ms,
                e2e_ms=e2e_ms,
                objects=len(objs),
            )
            self.hub.b.counters.set(
                "frames_ok", (self.hub.b.counters.get("frames_ok") or 0) + 1)

    def _infer_or_count(self, frame):
        """Stretch-resize to the model's NV12 geometry on the DSP, then
        infer on the raw tensor. A failed frame is counted and skipped
        (the stream must keep flowing) — None renders an empty pass."""
        try:
            scaled = self.dsp.resize_hw(
                frame, self._infer_wh[0], self._infer_wh[1],
                scaling="stretch")
            return self.infer.infer(scaled, self.model_id)
        except Exception as exc:  # noqa: BLE001
            self.hub.b.counters.set(
                "frames_err", (self.hub.b.counters.get("frames_err") or 0) + 1)
            logger.debug("chain B inference failed on one frame: %s", exc)
            return None

    def _overlays_for(self, frame, out) -> list[tuple]:
        overlays: list[tuple] = []
        objs = [o for o in (out.objects or [])
                if (getattr(o, "score", 0.0) or 0.0) >= self.min_score] \
            if out is not None else []
        if objs:
            try:
                rgba, x0, y0 = render_overlay_rgba(
                    frame.width, frame.height,
                    boxes=objs,
                    labels=[getattr(o, "label", None) for o in objs],
                    scores=[getattr(o, "score", None) for o in objs],
                )
                overlays.append((rgba, x0, y0))
            except Exception as exc:  # noqa: BLE001
                logger.debug("render_overlay_rgba failed: %s", exc)
        overlays.append(self._chip_for(frame.width, frame.height))
        return overlays

    def _chip_for(self, frame_w: int, frame_h: int) -> tuple:
        now = time.monotonic()
        if self._chip is None or now - self._chip_t >= CHIP_REFRESH_S:
            snap = self.hub.snapshot()
            chip, _, _ = text_chip_rgba(format_b_line(snap["b"]))
            x = CHIP_MARGIN_PX
            y = max(frame_h - chip.shape[0] - CHIP_MARGIN_PX, 0)
            self._chip = (chip, x, y)
            self._chip_t = now
        return self._chip

    def _close_publisher(self, *, eos: bool) -> None:
        pub, self._publisher = self._publisher, None  # pacer sees None and exits
        pacer, self._pacer = self._pacer, None
        if pacer is not None:
            pacer.join(timeout=2.0)
        if pub is None:
            return
        try:
            if eos:
                pub.publish_eos()  # encoder returns to the live source
        except Exception:  # noqa: BLE001 - teardown path
            pass
        try:
            pub.close()
        except Exception:  # noqa: BLE001
            pass

    def _sleep_stop_aware(self, seconds: float) -> None:
        self._stop.wait(seconds)
