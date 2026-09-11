"""
Tests for the SDK side of app frame injection (PushFrame P0-P2):
the CameraClient.push_frame / push_frame_stream / injection_status /
stop_injection RPC wrappers, FramePublisher (DSP pool + round-robin +
RPC compose, REPLACE + OVERLAY), and the pip_dest preset helper.

The daemon contract mirrored here (injection_service.cpp):
- buffer must be a DSP-registry id (never a raw fd),
- width/height nonzero and even, stride >= width,
- REPLACE consumes NV12 matching the target stream's encode resolution;
  OVERLAY composes an NV12 (opaque paste) or ARGB32 (alpha blend) inset
  at dest_x/dest_y inside the stream_id-named frame,
- queue cap 3 drop-oldest (never backpressures), counters in status,
- end_of_stream closes the session (buffer_id 0 only valid with EOS),
- PushFrameStream: first rejection ends the stream with the count of
  accepted frames; an EOS request closes; a clean half-close keeps the
  session open.
"""

from collections.abc import Iterator
from unittest.mock import patch

import numpy as np
import pytest

from neoruntime_ipc_sdk import (
    CameraClient,
    FramePublisher,
    InjectionResult,
    InjectionStatus,
)
from neoruntime_ipc_sdk.camera import pip_dest
from neoruntime_ipc_sdk.camera_types import StreamStatus
from neoruntime_ipc_sdk.proto import camera_pb2


def _stream(stream_id="sub", width=1280, height=720, has_encoder=True, fps=15):
    return StreamStatus(
        stream_id=stream_id,
        status="active",
        has_encoder=has_encoder,
        codec="h264",
        width=width,
        height=height,
        fps=fps,
        bitrate_bps=1_000_000,
        gop=30,
    )


class TestInjectionResultTypes:
    def test_result_fields(self):
        r = InjectionResult(
            success=True, message="ok", error_code=0, injected_frame_id=7
        )
        assert (r.success, r.message, r.error_code, r.injected_frame_id) == (
            True,
            "ok",
            0,
            7,
        )

    def test_status_fields(self):
        s = InjectionStatus(
            success=True,
            message="OK",
            active=True,
            mode="replace",
            frames_injected=3,
            frames_dropped=1,
            queue_depth=2,
        )
        assert s.active and s.mode == "replace"
        assert (s.frames_injected, s.frames_dropped, s.queue_depth) == (3, 1, 2)


class TestPushFrameRpc:
    @patch("neoruntime_ipc_sdk.camera.grpc.insecure_channel")
    def test_request_field_mapping(self, _mock_channel):
        stub = _RecordingStub(
            camera_pb2.PushFrameResponse(
                success=True, message="ok", error_code=0, injected_frame_id=7
            )
        )
        client = CameraClient()
        client._stub = stub

        res = client.push_frame(
            buffer_id=42,
            width=1280,
            height=720,
            stride=1280,
            pts_ns=12345,
        )

        req = stub.calls["PushFrame"]
        assert req.buffer_id == 42
        assert (req.width, req.height, req.stride) == (1280, 720, 1280)
        assert req.mode == camera_pb2.INJECT_REPLACE
        assert req.pts_ns == 12345
        assert req.dest_x == 0 and req.dest_y == 0
        assert req.end_of_stream is False
        assert res == InjectionResult(
            success=True, message="ok", error_code=0, injected_frame_id=7
        )

    @patch("neoruntime_ipc_sdk.camera.grpc.insecure_channel")
    def test_overlay_mode_maps_enum(self, _mock_channel):
        stub = _RecordingStub(
            camera_pb2.PushFrameResponse(success=True, message="ok")
        )
        client = CameraClient()
        client._stub = stub
        client.push_frame(buffer_id=1, width=64, height=48, stride=64, mode="overlay")
        assert stub.calls["PushFrame"].mode == camera_pb2.INJECT_OVERLAY

    @patch("neoruntime_ipc_sdk.camera.grpc.insecure_channel")
    def test_unknown_mode_rejected_client_side(self, _mock_channel):
        client = CameraClient()
        client._stub = _RecordingStub(camera_pb2.PushFrameResponse(success=True))
        with pytest.raises(ValueError, match="mode"):
            client.push_frame(buffer_id=1, width=64, height=48, stride=64, mode="blend")

    @patch("neoruntime_ipc_sdk.camera.grpc.insecure_channel")
    def test_failure_raises_with_code(self, _mock_channel):
        stub = _RecordingStub(
            camera_pb2.PushFrameResponse(
                success=False, message="unknown or foreign buffer id", error_code=-7011
            )
        )
        client = CameraClient()
        client._stub = stub
        with pytest.raises(RuntimeError) as ei:
            client.push_frame(buffer_id=9, width=64, height=48, stride=64)
        assert "unknown or foreign buffer id" in str(ei.value)
        assert "-7011" in str(ei.value)

    @patch("neoruntime_ipc_sdk.camera.grpc.insecure_channel")
    def test_timeout_forwarded(self, _mock_channel):
        stub = _RecordingStub(camera_pb2.PushFrameResponse(success=True))
        client = CameraClient()
        client._stub = stub
        client.push_frame(buffer_id=1, width=64, height=48, stride=64, timeout_s=2.0)
        assert stub.timeouts["PushFrame"] == 2.0

    @patch("neoruntime_ipc_sdk.camera.grpc.insecure_channel")
    def test_injection_status_mapping(self, _mock_channel):
        stub = _RecordingStub(
            camera_pb2.InjectionStatusResponse(
                success=True,
                message="OK",
                active=True,
                mode=camera_pb2.INJECT_REPLACE,
                frames_injected=12,
                frames_dropped=2,
                queue_depth=1,
            )
        )
        client = CameraClient()
        client._stub = stub
        st = client.injection_status()
        assert st == InjectionStatus(
            success=True,
            message="OK",
            active=True,
            mode="replace",
            frames_injected=12,
            frames_dropped=2,
            queue_depth=1,
        )
        assert isinstance(stub.calls["GetInjectionStatus"], camera_pb2.Empty)

    @patch("neoruntime_ipc_sdk.camera.grpc.insecure_channel")
    def test_injection_status_failure_raises(self, _mock_channel):
        stub = _RecordingStub(
            camera_pb2.InjectionStatusResponse(
                success=False, message="frame injection unavailable"
            )
        )
        client = CameraClient()
        client._stub = stub
        with pytest.raises(RuntimeError, match="unavailable"):
            client.injection_status()

    @patch("neoruntime_ipc_sdk.camera.grpc.insecure_channel")
    def test_stop_injection(self, _mock_channel):
        stub = _RecordingStub(
            camera_pb2.InjectionStatusResponse(success=True, message="OK")
        )
        client = CameraClient()
        client._stub = stub
        client.stop_injection()
        assert isinstance(stub.calls["StopInjection"], camera_pb2.Empty)

    @patch("neoruntime_ipc_sdk.camera.grpc.insecure_channel")
    def test_stop_injection_failure_raises(self, _mock_channel):
        stub = _RecordingStub(
            camera_pb2.InjectionStatusResponse(success=False, message="no daemon")
        )
        client = CameraClient()
        client._stub = stub
        with pytest.raises(RuntimeError, match="no daemon"):
            client.stop_injection()


# ---------------------------------------------------------------------------
# FramePublisher fakes
# ---------------------------------------------------------------------------


class FakePool:
    """Stands in for DspBufferPool: records writes, exposes ids/strides."""

    def __init__(self, width, height, depth):
        self.width, self.height = width, height
        self.ids = [100 + i for i in range(depth)]
        # padded luma stride (aligned up), like a real DMA layout
        self.strides = ((width + 15) // 16 * 16, (width + 15) // 16 * 16 // 2)
        self.writes = []
        self.release_count = 0

    def write(self, index, arr):
        assert 0 <= index < len(self.ids)
        self.writes.append((index, arr))

    def buffer_id(self, index):
        return self.ids[index]

    def release(self):
        self.release_count += 1


class FakeDsp:
    """Hands out pools FIFO (one per alloc/import call) and fakes convert_hw."""

    def __init__(self, *pools):
        self._pools = list(pools)
        self.alloc_args = None
        self.allocs = []
        self.import_args = None
        self.imports = []
        self.last_used_hw = True
        self.convert_calls = []

    def alloc_buffers(self, width, height, fmt="nv12", count=1):
        self.alloc_args = (width, height, fmt, count)
        self.allocs.append(self.alloc_args)
        return self._pools.pop(0)

    def import_shared_buffers(self, width, height, fmt="argb", count=1):
        self.import_args = (width, height, fmt, count)
        self.imports.append(self.import_args)
        return self._pools.pop(0)

    def convert_hw(self, src, dst_fmt, **kwargs):
        self.convert_calls.append((src.shape, dst_fmt, kwargs.get("src_pool"), kwargs.get("dst_pool")))
        # shaped like a real CONVERT read-back: nv12 (h*3//2, w)
        h, w = kwargs["src_pool"].height, kwargs["src_pool"].width
        return np.full((h * 3 // 2, w), 7, dtype=np.uint8)


class FakeCamera:
    def __init__(self, streams, results=None, stream_result=None):
        self.streams = streams
        self.results = list(results or [])
        self.stream_result = stream_result
        self.pushes = []
        self.stream_pushes = None
        self.stream_timeouts = None

    def get_stream_status(self, timeout_s=None):
        return self.streams

    def push_frame(self, **kwargs):
        self.pushes.append(kwargs)
        if self.results:
            return self.results.pop(0)
        return InjectionResult(
            success=True, message="ok", error_code=0, injected_frame_id=len(self.pushes)
        )

    def push_frame_stream(self, requests, timeout_s=None):
        self.stream_pushes = [dict(r) for r in requests]
        self.stream_timeouts = timeout_s
        if self.stream_result is not None:
            if not self.stream_result.success:
                # Mirror CameraClient.push_frame_stream's contract: the
                # first rejection ends the stream as RuntimeError.
                raise RuntimeError(
                    f"PushFrameStream failed after "
                    f"{self.stream_result.accepted_frame_count} accepted "
                    f"frame(s): {self.stream_result.message} "
                    f"(error_code={self.stream_result.error_code})"
                )
            return self.stream_result
        return InjectionResult(
            success=True,
            message="ok",
            error_code=0,
            accepted_frame_count=len(self.stream_pushes),
        )


class TestFramePublisherGeometry:
    def test_resolves_geometry_and_allocates_pool(self):
        pool = FakePool(1280, 720, 2)
        dsp = FakeDsp(pool)
        cam = FakeCamera([_stream()])
        pub = FramePublisher(cam, dsp, stream_id="sub")

        assert (pub.width, pub.height) == (1280, 720)
        assert dsp.alloc_args == (1280, 720, "nv12", 2)

    def test_custom_pool_depth(self):
        pool = FakePool(1280, 720, 3)
        pub = FramePublisher(FakeCamera([_stream()]), FakeDsp(pool), pool_depth=3)
        assert pub._pool_depth == 3

    def test_unknown_stream_raises(self):
        pool = FakePool(1280, 720, 2)
        with pytest.raises(RuntimeError, match="sub"):
            FramePublisher(FakeCamera([_stream("main")]), FakeDsp(pool), stream_id="sub")
        assert pool.release_count == 0  # ctor failed before alloc — nothing to leak

    def test_stream_without_encoder_raises(self):
        pool = FakePool(1280, 720, 2)
        with pytest.raises(RuntimeError, match="encoder"):
            FramePublisher(
                FakeCamera([_stream(has_encoder=False)]), FakeDsp(pool)
            )

    def test_zero_geometry_raises(self):
        pool = FakePool(1280, 720, 2)
        with pytest.raises(ValueError, match="even"):
            FramePublisher(FakeCamera([_stream(width=0)]), FakeDsp(pool))

    def test_odd_dims_raise(self):
        pool = FakePool(1281, 720, 2)
        with pytest.raises(ValueError, match="even"):
            FramePublisher(FakeCamera([_stream(width=1281)]), FakeDsp(pool))


class TestFramePublisherPublish:
    def _pub(self, results=None, depth=2):
        pool = FakePool(1280, 720, depth)
        dsp = FakeDsp(pool)
        cam = FakeCamera([_stream()], results=results)
        return FramePublisher(cam, dsp, stream_id="sub"), cam, pool

    @staticmethod
    def _frame(pub):
        return np.zeros((pub.height * 3 // 2, pub.width), dtype=np.uint8)

    def test_round_robin_slots_and_rpc_fields(self):
        pub, cam, pool = self._pub()
        f = self._frame(pub)

        pub.publish(f)
        pub.publish(f)
        pub.publish(f)

        slots = [i for i, _ in pool.writes]
        assert slots == [0, 1, 0]
        ids = [p["buffer_id"] for p in cam.pushes]
        assert ids == [pool.ids[0], pool.ids[1], pool.ids[0]]
        first = cam.pushes[0]
        assert (first["width"], first["height"]) == (1280, 720)
        assert first["stride"] == pool.strides[0]
        assert first["mode"] == "replace"
        assert first["end_of_stream"] is False

    def test_accepts_tight_nv12_bytes(self):
        pub, cam, pool = self._pub()
        raw = bytes(pub.height * 3 // 2 * pub.width)
        pub.publish(raw)
        idx, arr = pool.writes[0]
        assert arr.shape == (pub.height * 3 // 2, pub.width)
        assert arr.dtype == np.uint8

    def test_rejects_wrong_ndarray_shape(self):
        pub, _, _ = self._pub()
        with pytest.raises(ValueError, match="shape"):
            pub.publish(np.zeros((pub.height, pub.width), dtype=np.uint8))

    def test_rejects_wrong_ndarray_dtype(self):
        pub, _, _ = self._pub()
        with pytest.raises(ValueError, match="dtype"):
            pub.publish(
                np.zeros((pub.height * 3 // 2, pub.width), dtype=np.uint16)
            )

    def test_rejects_wrong_byte_length(self):
        pub, _, _ = self._pub()
        with pytest.raises(ValueError, match="bytes"):
            pub.publish(b"\x00" * (pub.width * pub.height))

    def test_pts_propagates(self):
        pub, cam, _ = self._pub()
        pub.publish(self._frame(pub), pts_ns=987654321)
        assert cam.pushes[0]["pts_ns"] == 987654321

    def test_result_propagates(self):
        want = InjectionResult(
            success=True, message="ok", error_code=0, injected_frame_id=9
        )
        pub, _, _ = self._pub(results=[want])
        got = pub.publish(self._frame(pub))
        assert got is want


class TestFramePublisherLifecycle:
    def _pub(self):
        pool = FakePool(1280, 720, 2)
        cam = FakeCamera([_stream()])
        pub = FramePublisher(cam, FakeDsp(pool), stream_id="sub")
        return pub, pool

    def test_publish_eos(self):
        pub, _ = self._pub()
        # capture the kwargs via the fake camera
        pub.publish_eos()
        push = pub._camera.pushes[0]
        assert push["buffer_id"] == 0
        assert push["end_of_stream"] is True
        assert (push["width"], push["height"]) == (1280, 720)

    def test_close_releases_pool_idempotently(self):
        pub, pool = self._pub()
        pub.close()
        pub.close()
        assert pool.release_count == 1

    def test_publish_after_close_raises(self):
        pub, _ = self._pub()
        pub.close()
        with pytest.raises(RuntimeError, match="closed"):
            pub.publish(np.zeros((720 * 3 // 2, 1280), dtype=np.uint8))

    def test_context_manager_closes(self):
        pub, pool = self._pub()
        with pub:
            pub.publish(np.zeros((720 * 3 // 2, 1280), dtype=np.uint8))
        assert pool.release_count == 1

    def _frame(self, pub):
        return np.zeros((pub.height * 3 // 2, pub.width), dtype=np.uint8)

    def test_exit_sends_best_effort_eos_after_publishing(self):
        # A with-block that published frames but never sent EOS: __exit__
        # flushes the session — a clean half-close would keep it open
        # (daemon contract) until the connection drops.
        pub, pool = self._pub()
        with pub:
            pub.publish(self._frame(pub))
        pushes = pub._camera.pushes
        assert len(pushes) == 2  # frame push + auto EOS
        eos = pushes[-1]
        assert eos["buffer_id"] == 0
        assert eos["end_of_stream"] is True
        assert pool.release_count == 1

    def test_exit_sends_no_eos_when_never_published(self):
        pub, _ = self._pub()
        with pub:
            pass
        assert pub._camera.pushes == []

    def test_exit_does_not_resend_eos_after_explicit_publish_eos(self):
        pub, _ = self._pub()
        with pub:
            pub.publish(self._frame(pub))
            pub.publish_eos()
        assert len(pub._camera.pushes) == 2  # frame + explicit EOS, no third

    def test_exit_does_not_resend_eos_after_publish_stream_end_with_eos(self):
        pub, _ = self._pub()
        with pub:
            pub.publish_stream([self._frame(pub), self._frame(pub)],
                               end_with_eos=True)
        assert pub._camera.pushes == []  # no unary auto-EOS on top
        assert pub._camera.stream_pushes[-1]["end_of_stream"] is True

    def test_exit_survives_eos_failure(self, caplog):
        # A failing EOS is a warning, never an exception out of the
        # with-block; the pools still release.
        pub, pool = self._pub()
        cam = pub._camera
        original_push = cam.push_frame

        def push_failing_eos(**kwargs):
            if kwargs.get("end_of_stream"):
                raise RuntimeError("connection gone")
            return original_push(**kwargs)

        cam.push_frame = push_failing_eos
        with pub:
            pub.publish(self._frame(pub))
        assert pool.release_count == 1
        assert pub._closed
        assert "best-effort EOS failed" in caplog.text


class TestPipDest:
    def test_four_corners(self):
        # 1280x720 frame, 320x180 inset, 16 margin
        assert pip_dest(1280, 720, 320, 180, "top-left", 16) == (16, 16)
        assert pip_dest(1280, 720, 320, 180, "top-right", 16) == (944, 16)
        assert pip_dest(1280, 720, 320, 180, "bottom-left", 16) == (16, 524)
        assert pip_dest(1280, 720, 320, 180, "bottom-right", 16) == (944, 524)

    def test_rounds_down_to_even(self):
        # odd inset on an even frame yields odd raw coords -> floored
        x, y = pip_dest(1280, 720, 321, 181, "bottom-right", 16)
        assert (x % 2, y % 2) == (0, 0)
        assert (x, y) == (943 & ~1, 523 & ~1)  # 942, 522

    def test_unknown_corner_raises(self):
        with pytest.raises(ValueError, match="corner"):
            pip_dest(1280, 720, 320, 180, "middle")

    def test_oversized_inset_raises(self):
        with pytest.raises(ValueError, match="fit"):
            pip_dest(640, 360, 640, 180, "bottom-right", 16)

    def test_nonpositive_inset_raises(self):
        with pytest.raises(ValueError, match="positive"):
            pip_dest(1280, 720, 0, 180, "top-left", 16)


class TestPushFrameStreamRpc:
    @patch("neoruntime_ipc_sdk.camera.grpc.insecure_channel")
    def test_request_stream_mapping(self, _mock_channel):
        stub = _RecordingStub(
            camera_pb2.PushFrameResponse(
                success=True, message="ok", accepted_frame_count=3
            )
        )
        client = CameraClient()
        client._stub = stub
        frames = [
            {"buffer_id": 10, "width": 320, "height": 180, "stride": 320,
             "mode": "overlay", "stream_id": "main", "dest_x": 944, "dest_y": 524,
             "pts_ns": 111},
            {"buffer_id": 11, "width": 320, "height": 180, "stride": 320,
             "mode": "overlay", "stream_id": "main", "dest_x": 944, "dest_y": 524},
            {"buffer_id": 0, "width": 320, "height": 180, "stride": 320,
             "end_of_stream": True},
        ]
        res = client.push_frame_stream(frames)
        reqs = list(stub.calls["PushFrameStream"])
        assert len(reqs) == 3
        assert reqs[0].mode == camera_pb2.INJECT_OVERLAY
        assert (reqs[0].dest_x, reqs[0].dest_y) == (944, 524)
        assert reqs[0].stream_id == "main"
        assert reqs[0].pts_ns == 111 and reqs[1].pts_ns == 0
        assert reqs[2].end_of_stream is True and reqs[2].buffer_id == 0
        assert res.accepted_frame_count == 3
        assert res.success and res.message == "ok"

    @patch("neoruntime_ipc_sdk.camera.grpc.insecure_channel")
    def test_first_rejection_raises_with_count(self, _mock_channel):
        stub = _RecordingStub(
            camera_pb2.PushFrameResponse(
                success=False, message="inset exceeds stream bounds",
                error_code=-7012, accepted_frame_count=2,
            )
        )
        client = CameraClient()
        client._stub = stub
        with pytest.raises(RuntimeError) as ei:
            client.push_frame_stream([{"buffer_id": 1, "width": 64, "height": 48, "stride": 64}])
        assert "exceeds stream bounds" in str(ei.value)
        assert "2 accepted" in str(ei.value)

    @patch("neoruntime_ipc_sdk.camera.grpc.insecure_channel")
    def test_unknown_stream_mode_raises_before_rpc(self, _mock_channel):
        client = CameraClient()
        client._stub = _RecordingStub(camera_pb2.PushFrameResponse(success=True))
        with pytest.raises(ValueError, match="mode"):
            client.push_frame_stream(
                [{"buffer_id": 1, "width": 64, "height": 48, "stride": 64,
                  "mode": "blend"}]
            )


class TestPushFrameStreamId:
    @patch("neoruntime_ipc_sdk.camera.grpc.insecure_channel")
    def test_stream_id_passthrough(self, _mock_channel):
        stub = _RecordingStub(camera_pb2.PushFrameResponse(success=True, message="ok"))
        client = CameraClient()
        client._stub = stub
        client.push_frame(
            buffer_id=5, width=1280, height=720, stride=1280,
            stream_id="sub", mode="replace",
        )
        assert stub.calls["PushFrame"].stream_id == "sub"
        assert stub.calls["PushFrame"].mode == camera_pb2.INJECT_REPLACE


class TestFramePublisherOverlayGeometry:
    def _dsp(self, *pools):
        return FakeDsp(*pools)

    def test_default_inset_is_quarter_even(self):
        pool = FakePool(320, 180, 2)
        dsp = self._dsp(pool)
        pub = FramePublisher(
            FakeCamera([_stream()]), dsp, stream_id="sub", mode="overlay"
        )
        assert (pub.width, pub.height) == (320, 180)  # 1280/4, 720/4
        assert dsp.alloc_args == (320, 180, "nv12", 2)
        assert pub.mode == "overlay"
        assert pub.dest == (0, 0)

    def test_custom_inset_and_dest(self):
        pool = FakePool(480, 270, 2)
        dsp = self._dsp(pool)
        pub = FramePublisher(
            FakeCamera([_stream()]), dsp, stream_id="sub",
            mode="overlay", inset=(480, 270), dest=(64, 96),
        )
        assert (pub.width, pub.height) == (480, 270)
        assert pub.dest == (64, 96)

    def test_argb_pool_fmt(self):
        # The deployed HAL refuses ARGB32 pool allocation, so argb overlays
        # ride the shared memfd-import ring instead of alloc_buffers.
        pool = FakePool(480, 270, 2)
        dsp = self._dsp(pool)
        FramePublisher(
            FakeCamera([_stream()]), dsp, stream_id="sub",
            mode="overlay", fmt="argb", inset=(480, 270),
        )
        assert dsp.import_args == (480, 270, "argb", 2)
        assert dsp.allocs == []  # no doomed dma-buf pool was requested

    def test_odd_inset_raises(self):
        with pytest.raises(ValueError, match="even"):
            FramePublisher(
                FakeCamera([_stream()]), self._dsp(FakePool(480, 271, 2)),
                stream_id="sub", mode="overlay", inset=(480, 271),
            )

    def test_odd_dest_raises(self):
        with pytest.raises(ValueError, match="even"):
            FramePublisher(
                FakeCamera([_stream()]), self._dsp(FakePool(480, 270, 2)),
                stream_id="sub", mode="overlay", inset=(480, 270), dest=(63, 0),
            )

    def test_out_of_bounds_dest_raises(self):
        with pytest.raises(ValueError, match="exceeds"):
            FramePublisher(
                FakeCamera([_stream()]), self._dsp(FakePool(480, 270, 2)),
                stream_id="sub", mode="overlay", inset=(480, 270),
                dest=(1280 - 478, 0),  # 2 px past the right edge
            )

    def test_replace_rejects_argb(self):
        with pytest.raises(ValueError, match="NV12"):
            FramePublisher(
                FakeCamera([_stream()]), self._dsp(FakePool(1280, 720, 2)),
                stream_id="sub", mode="replace", fmt="argb",
            )

    def test_unknown_mode_raises(self):
        with pytest.raises(ValueError, match="mode"):
            FramePublisher(
                FakeCamera([_stream()]), self._dsp(FakePool(1280, 720, 2)),
                stream_id="sub", mode="blend",
            )


class TestFramePublisherOverlayPublish:
    def _pub(self, fmt="nv12", depth=2):
        pool = FakePool(480, 270, depth)
        cam = FakeCamera([_stream()])
        pub = FramePublisher(
            cam, FakeDsp(pool), stream_id="sub",
            mode="overlay", fmt=fmt, inset=(480, 270), dest=(64, 96),
        )
        return pub, cam, pool

    def test_push_carries_overlay_fields(self):
        pub, cam, _ = self._pub()
        pub.publish(np.zeros((270 * 3 // 2, 480), dtype=np.uint8))
        push = cam.pushes[0]
        assert push["mode"] == "overlay"
        assert push["stream_id"] == "sub"
        assert (push["dest_x"], push["dest_y"]) == (64, 96)
        assert (push["width"], push["height"]) == (480, 270)

    def test_argb_shape_accepted(self):
        pub, cam, pool = self._pub(fmt="argb")
        pub.publish(np.zeros((270, 480, 4), dtype=np.uint8))
        assert pool.writes[0][1].shape == (270, 480, 4)
        assert cam.pushes[0]["mode"] == "overlay"

    def test_argb_bytes_length_checked(self):
        pub, _, _ = self._pub(fmt="argb")
        with pytest.raises(ValueError, match="bytes"):
            pub.publish(b"\x00" * (480 * 270))  # half the needed bytes

    def test_nv12_shape_still_enforced(self):
        pub, _, _ = self._pub()
        with pytest.raises(ValueError, match="shape"):
            pub.publish(np.zeros((270, 480), dtype=np.uint8))  # missing UV


class TestFramePublisherPublishStream:
    def _pub(self, depth=3, mode="replace"):
        w, h = (1280, 720) if mode == "replace" else (480, 270)
        pool = FakePool(w, h, depth)
        cam = FakeCamera([_stream()])
        pub = FramePublisher(
            cam, FakeDsp(pool), stream_id="sub", pool_depth=depth,
            **({"mode": "overlay", "inset": (480, 270), "dest": (64, 96)} if mode == "overlay" else {}),
        )
        frame = (
            np.zeros((720 * 3 // 2, 1280), dtype=np.uint8)
            if mode == "replace"
            else np.zeros((270 * 3 // 2, 480), dtype=np.uint8)
        )
        return pub, cam, pool, frame

    def test_all_frames_one_rpc(self):
        pub, cam, pool, frame = self._pub()
        res = pub.publish_stream([frame, frame, frame])
        assert len(cam.pushes) == 0          # unary never touched
        assert len(cam.stream_pushes) == 3
        assert res.accepted_frame_count == 3
        slots = [i for i, _ in pool.writes]
        assert slots == [0, 1, 2]
        ids = [r["buffer_id"] for r in cam.stream_pushes]
        assert ids == [pool.ids[0], pool.ids[1], pool.ids[2]]

    def test_request_fields(self):
        pub, cam, _, frame = self._pub()
        pub.publish_stream([frame], pts_ns=[1234])
        req = cam.stream_pushes[0]
        assert req["mode"] == "replace"
        assert req["stream_id"] == "sub"
        assert req["pts_ns"] == 1234
        assert (req["dest_x"], req["dest_y"]) == (0, 0)

    def test_pts_defaults_to_zero(self):
        pub, cam, _, frame = self._pub()
        pub.publish_stream([frame, frame])
        assert [r["pts_ns"] for r in cam.stream_pushes] == [0, 0]

    def test_end_with_eos_appends_closing_request(self):
        pub, cam, _, frame = self._pub()
        pub.publish_stream([frame], end_with_eos=True)
        assert len(cam.stream_pushes) == 2
        eos = cam.stream_pushes[1]
        assert eos["buffer_id"] == 0
        assert eos["end_of_stream"] is True

    def test_overlay_fields_flow(self):
        pub, cam, _, frame = self._pub(mode="overlay")
        pub.publish_stream([frame])
        req = cam.stream_pushes[0]
        assert req["mode"] == "overlay"
        assert (req["dest_x"], req["dest_y"]) == (64, 96)
        assert (req["width"], req["height"]) == (480, 270)

    def test_closed_publisher_raises(self):
        pub, _, _, frame = self._pub()
        pub.close()
        with pytest.raises(RuntimeError, match="closed"):
            pub.publish_stream([frame])

    def test_failure_result_propagates(self):
        pub, cam, _, frame = self._pub()
        cam.stream_result = InjectionResult(
            success=False, message="permission denied", error_code=-6,
            accepted_frame_count=0,
        )
        with pytest.raises(RuntimeError, match="permission denied"):
            pub.publish_stream([frame])


class TestFramePublisherPublishRgb:
    def _pub(self):
        pool = FakePool(1280, 720, 2)
        rgb_pool = FakePool(1280, 720, 1)
        stage = FakePool(1280, 720, 1)
        cam = FakeCamera([_stream()])
        dsp = FakeDsp(pool, rgb_pool, stage)
        pub = FramePublisher(cam, dsp, stream_id="sub")
        return pub, cam, dsp, rgb_pool, stage

    def test_hw_path_pushes_stage_buffer(self):
        pub, cam, dsp, rgb_pool, stage = self._pub()
        rgb = np.zeros((720, 1280, 3), dtype=np.uint8)
        res = pub.publish_rgb(rgb)
        assert res.success
        # rgb staged + hardware convert rgb->nv12
        assert dsp.allocs == [(1280, 720, "nv12", 2), (1280, 720, "rgb24", 1), (1280, 720, "nv12", 1)]
        src_shape, dst_fmt, src_pool, dst_pool = dsp.convert_calls[0]
        assert src_shape == (720, 1280, 3) and dst_fmt == "nv12"
        assert src_pool is rgb_pool and dst_pool is stage
        # HW path: no client-side write into the staging pool
        assert stage.writes == []
        push = cam.pushes[0]
        assert push["buffer_id"] == stage.ids[0]
        assert push["mode"] == "replace" and push["stream_id"] == "sub"

    def test_cpu_fallback_writes_stage(self):
        pub, cam, dsp, _, stage = self._pub()
        dsp.last_used_hw = False  # firmware refused / DSP down
        pub.publish_rgb(np.zeros((720, 1280, 3), dtype=np.uint8))
        assert len(stage.writes) == 1
        idx, arr = stage.writes[0]
        assert idx == 0
        assert arr.shape == (720 * 3 // 2, 1280)
        assert cam.pushes[0]["buffer_id"] == stage.ids[0]

    def test_rgb_bytes_reshaped(self):
        pub, cam, dsp, rgb_pool, _ = self._pub()
        raw = bytes(720 * 1280 * 3)
        pub.publish_rgb(raw)
        # the reshape happens before the convert boundary (the real
        # convert_hw stages src into the rgb pool itself)
        src_shape, dst_fmt, _, _ = dsp.convert_calls[0]
        assert src_shape == (720, 1280, 3) and dst_fmt == "nv12"

    def test_pools_allocated_once(self):
        pub, _, dsp, _, _ = self._pub()
        rgb = np.zeros((720, 1280, 3), dtype=np.uint8)
        pub.publish_rgb(rgb)
        pub.publish_rgb(rgb)
        assert len(dsp.allocs) == 3  # ctor pool + rgb + stage, no growth

    def test_overlay_publisher_rejects_rgb(self):
        pool = FakePool(480, 270, 2)
        pub = FramePublisher(
            FakeCamera([_stream()]), FakeDsp(pool), stream_id="sub",
            mode="overlay", inset=(480, 270),
        )
        with pytest.raises(RuntimeError, match="REPLACE"):
            pub.publish_rgb(np.zeros((270, 480, 3), dtype=np.uint8))

    def test_close_releases_staging_pools(self):
        pub, _, _, rgb_pool, stage = self._pub()
        pub.publish_rgb(np.zeros((720, 1280, 3), dtype=np.uint8))
        pub.close()
        assert rgb_pool.release_count == 1
        assert stage.release_count == 1


class _RecordingStub:
    """Records the RPC name -> request (and timeout); returns preset responses.

    Client-streaming requests arrive as a generator; real gRPC drains it,
    so the fake does too (a raise inside the generator then surfaces at
    call time, like the real transport) and stores the materialized list.
    """

    def __init__(self, response):
        self.response = response
        self.calls = {}
        self.timeouts = {}

    def __getattr__(self, name):
        def _call(req, timeout=None):
            if isinstance(req, Iterator):
                req = list(req)
            self.calls[name] = req
            self.timeouts[name] = timeout
            return self.response

        return _call


class TestSessionTagRpc:
    """P2-13: session_id rides PushFrame requests and echoes back; the
    live session's tag surfaces in GetInjectionStatus. Ownership itself
    stays fd-anchored daemon-side — the tag is correlation/observability
    only (a killed app's frames are reclaimed by UDS disconnect, not by
    this string)."""

    @patch("neoruntime_ipc_sdk.camera.grpc.insecure_channel")
    def test_push_frame_session_id_on_request_and_echo(self, _mock_channel):
        stub = _RecordingStub(
            camera_pb2.PushFrameResponse(success=True, message="ok", session_id="s1")
        )
        client = CameraClient()
        client._stub = stub
        res = client.push_frame(
            buffer_id=42, width=64, height=48, stride=64, session_id="s1"
        )
        assert stub.calls["PushFrame"].session_id == "s1"
        assert res.session_id == "s1"

    @patch("neoruntime_ipc_sdk.camera.grpc.insecure_channel")
    def test_push_frame_default_session_id_empty(self, _mock_channel):
        stub = _RecordingStub(camera_pb2.PushFrameResponse(success=True, message="ok"))
        client = CameraClient()
        client._stub = stub
        res = client.push_frame(buffer_id=1, width=64, height=48, stride=64)
        assert stub.calls["PushFrame"].session_id == ""
        assert res.session_id == ""

    @patch("neoruntime_ipc_sdk.camera.grpc.insecure_channel")
    def test_push_frame_stream_passthrough_and_echo(self, _mock_channel):
        stub = _RecordingStub(
            camera_pb2.PushFrameResponse(
                success=True, message="ok", accepted_frame_count=2, session_id="s9"
            )
        )
        client = CameraClient()
        client._stub = stub
        frame = {"buffer_id": 1, "width": 64, "height": 48, "stride": 64}
        res = client.push_frame_stream(
            [dict(frame, session_id="s9"), dict(frame, buffer_id=2, session_id="s9")]
        )
        reqs = stub.calls["PushFrameStream"]
        assert [r.session_id for r in reqs] == ["s9", "s9"]
        assert res.session_id == "s9"
        assert res.accepted_frame_count == 2

    @patch("neoruntime_ipc_sdk.camera.grpc.insecure_channel")
    def test_injection_status_reports_live_session_tag(self, _mock_channel):
        stub = _RecordingStub(
            camera_pb2.InjectionStatusResponse(
                success=True,
                message="OK",
                active=True,
                mode=camera_pb2.INJECT_OVERLAY,
                frames_injected=3,
                frames_dropped=1,
                queue_depth=2,
                session_id="live-tag",
            )
        )
        client = CameraClient()
        client._stub = stub
        st = client.injection_status()
        assert st.active and st.mode == "overlay"
        assert st.session_id == "live-tag"


class TestFramePublisherSessionTag:
    """P2-13: one FramePublisher = one session tag on every request."""

    @staticmethod
    def _frame(w=1280, h=720):
        return np.zeros((h * 3 // 2, w), dtype=np.uint8)

    def test_publish_forwards_session_id(self):
        pool = FakePool(1280, 720, 2)
        cam = FakeCamera([_stream()])
        pub = FramePublisher(cam, FakeDsp(pool), session_id="sess-A")
        pub.publish(self._frame())
        assert cam.pushes[0]["session_id"] == "sess-A"

    def test_publish_default_is_untagged(self):
        pool = FakePool(1280, 720, 2)
        cam = FakeCamera([_stream()])
        pub = FramePublisher(cam, FakeDsp(pool))
        pub.publish(self._frame())
        assert cam.pushes[0]["session_id"] == ""

    def test_publish_stream_tags_every_request_including_eos(self):
        pool = FakePool(1280, 720, 2)
        cam = FakeCamera([_stream()])
        pub = FramePublisher(cam, FakeDsp(pool), session_id="sess-A")
        pub.publish_stream([self._frame(), self._frame()], end_with_eos=True)
        # 2 frames + the EOS request, all carrying the publisher's tag
        assert len(cam.stream_pushes) == 3
        assert all(r.get("session_id") == "sess-A" for r in cam.stream_pushes)

    def test_publish_rgb_forwards_session_id(self):
        dsp = FakeDsp(
            FakePool(1280, 720, 2), FakePool(1280, 720, 1), FakePool(1280, 720, 1)
        )
        cam = FakeCamera([_stream()])
        pub = FramePublisher(cam, dsp, session_id="sess-B")
        pub.publish_rgb(np.zeros((720, 1280, 3), dtype=np.uint8))
        assert cam.pushes[0]["session_id"] == "sess-B"

    def test_publish_eos_carries_session_id(self):
        pool = FakePool(1280, 720, 2)
        cam = FakeCamera([_stream()])
        pub = FramePublisher(cam, FakeDsp(pool), session_id="sess-A")
        pub.publish_eos()
        assert cam.pushes[0]["end_of_stream"] is True
        assert cam.pushes[0]["session_id"] == "sess-A"
