"""Tests for the P2 async surface: the SubmitDspJobAsync/WaitDspJob
pair, the PendingDspJob lifecycle (wait / wait_result / done / release,
timeout-then-retry, failure surfacing), the UNIMPLEMENTED -> sync
fallback for old daemons, the encode_jpeg_hw(src_buffer_id=) zero-copy
chain tail, and convert_hw's learned gray8 quiet short-circuit.

The fake wire mirrors tests/test_dsp.py + test_dsp_blend.py: memfd
pools, a stub with the async rpc pair, and scripted WaitDspJob
responses (timeout -4 first, completion after).
"""

import warnings
from unittest import mock

import grpc
import numpy as np
import pytest

from neoruntime_ipc_sdk.dsp import DspClient, DspError, PendingDspJob
from neoruntime_ipc_sdk.proto import camera_pb2
from tests.test_dsp import memfd, nv12_array, patched_alloc


# ---------------------------------------------------------------- helpers --
def unimplemented_error():
    err = grpc.RpcError("no such method")
    err.code = lambda: grpc.StatusCode.UNIMPLEMENTED
    return err


def wait_timeout(job_id, ms):
    return camera_pb2.DspJobResponse(
        success=False, error_code=-4, message=f"job {job_id} not done after {ms}ms"
    )


def wait_ok(job_id, elapsed=2):
    return camera_pb2.DspJobResponse(
        success=True, elapsed_ms=elapsed, job_id=job_id, done=True
    )


class AsyncStub:
    """Fake gRPC stub with the async pair and a scripted wait script.

    ``wait_script`` responses are consumed one per WaitDspJob call; once
    empty, waits complete successfully. ``sync_response`` optionally
    overrides SubmitDspJob's answer (used for refused CONVERT jobs).
    """

    def __init__(
        self,
        wait_script=None,
        unimplemented_async=False,
        wait_unimplemented=False,
        encode_unimplemented=False,
        sync_response=None,
    ):
        self.async_requests = []
        self.sync_requests = []
        self.wait_requests = []
        self.encode_requests = []
        self.next_job_id = 11
        self.wait_script = list(wait_script or [])
        self.unimplemented_async = unimplemented_async
        self.wait_unimplemented = wait_unimplemented
        self.encode_unimplemented = encode_unimplemented
        self.sync_response = sync_response

    def SubmitDspJob(self, request, timeout=None):
        self.sync_requests.append(request)
        if self.sync_response is not None:
            return self.sync_response(request)
        return camera_pb2.DspJobResponse(success=True, elapsed_ms=5)

    def SubmitDspJobAsync(self, request, timeout=None):
        self.async_requests.append(request)
        if self.unimplemented_async:
            raise unimplemented_error()
        job_id = self.next_job_id
        self.next_job_id += 1
        return camera_pb2.DspJobResponse(success=True, job_id=job_id, done=False)

    def WaitDspJob(self, request, timeout=None):
        self.wait_requests.append(request)
        if self.wait_unimplemented:
            raise unimplemented_error()
        if self.wait_script:
            return self.wait_script.pop(0)
        return wait_ok(request.job_id)

    def EncodeImage(self, request, timeout=None):
        self.encode_requests.append(request)
        if self.encode_unimplemented:
            raise unimplemented_error()
        return camera_pb2.EncodeImageResponse(success=True, jpeg=b"\xff\xd8FAKE")


def async_client(**stub_kw):
    """Client whose daemon answers the async pair via an AsyncStub."""
    client = DspClient()
    patched_alloc(client)
    client._send_release = mock.Mock()
    stub = AsyncStub(**stub_kw)
    client._stub = stub
    return client, stub


def overlay_solid(width, height):
    rgba = np.zeros((height, width, 4), np.uint8)
    rgba[..., 0] = 200
    rgba[..., 3] = 255
    return rgba


def importing_overlays(client):
    """Patch _import_memfd for blend tests: ids from 2000, no socket."""
    state = {"next": 2000}

    def fake(wire, width, height, fmt, stride, timeout_s=5.0):
        bid = state["next"]
        state["next"] += 1
        return bid

    client._import_memfd = fake


# ------------------------------------------------------------ submit/wait --
class TestAsyncSubmit:
    def test_resize_async_request_and_wait_roundtrip(self):
        client, stub = async_client()
        nv12 = nv12_array(64, 32, seed=3)

        job = client.resize_hw(nv12, 32, 16, fmt="nv12", wait=False)

        assert isinstance(job, PendingDspJob)
        assert stub.async_requests and not stub.sync_requests
        req = stub.async_requests[0]
        assert req.op == camera_pb2.DSP_OP_RESIZE
        assert req.src_buffer_id == 1000  # src pool
        assert list(req.dst_buffer_ids) == [1001]  # dst pool
        assert job.job_id == 11
        assert client.last_used_hw is True

        # the daemon fills the destination; wait() then reads it back
        expect = nv12_array(32, 16, seed=9)
        dst_pool, dst_i = job._reads[0]
        dst_pool.write(dst_i, expect)
        out = job.wait()

        wr = stub.wait_requests[0]
        assert (wr.job_id, wr.timeout_ms) == (11, 5000)
        assert out.shape == (24, 32)  # 16 * 3 // 2 rows
        assert np.array_equal(out, expect)

    def test_wait_timeout_then_longer_retry_succeeds(self):
        client, stub = async_client(
            wait_script=[wait_timeout(11, 0), wait_timeout(11, 200)]
        )
        job = client.resize_hw(nv12_array(64, 32), 32, 16, fmt="nv12", wait=False)

        assert job.done() is False  # non-blocking poll
        with pytest.raises(DspError, match="still pending"):
            job.wait_result(timeout_s=0.2)
        # the daemon kept the entry — a later wait completes
        out = job.wait(timeout_s=5.0)
        assert out.shape == (24, 32)
        assert len(stub.wait_requests) == 3

    def test_done_poll_caches_completion(self):
        client, stub = async_client(wait_script=[wait_ok(11)])
        job = client.resize_hw(nv12_array(64, 32), 32, 16, fmt="nv12", wait=False)

        assert job.done() is True
        assert job.done() is True  # cached — no extra rpc
        out = job.wait()
        assert len(stub.wait_requests) == 1
        assert out.shape == (24, 32)

    def test_multi_crop_async_wait_returns_list(self):
        client, stub = async_client()
        rects = [(0, 0, 32, 16, 16, 16), (0, 16, 32, 16, 16, 16)]

        job = client.multi_crop_hw(nv12_array(64, 32), rects, fmt="nv12", wait=False)
        req = stub.async_requests[0]
        assert req.op == camera_pb2.DSP_OP_MULTI_CROP_AND_RESIZE
        # one dst pool (both rects share geometry), two slots
        assert list(req.dst_buffer_ids) == [1001, 1002]

        out = job.wait()
        assert isinstance(out, list) and len(out) == 2
        assert all(part.shape == (24, 16) for part in out)

    def test_release_is_idempotent_and_reaps(self):
        client, stub = async_client()
        job = client.resize_hw(nv12_array(64, 32), 32, 16, fmt="nv12", wait=False)

        job.release()
        job.release()

        # one bounded reap wait, then both buffers returned exactly once
        assert len(stub.wait_requests) == 1
        assert client._send_release.call_count == 2  # src + dst pool

    def test_wait_after_release_raises(self):
        client, _ = async_client()
        job = client.resize_hw(nv12_array(64, 32), 32, 16, fmt="nv12", wait=False)
        job.release()
        with pytest.raises(DspError, match="already released"):
            job.wait_result()


class TestFailures:
    def test_failed_job_error_surfaced_from_wait(self):
        client, _ = async_client(
            wait_script=[camera_pb2.DspJobResponse(
                success=False, error_code=-2801, message="HAL -2801"
            )]
        )
        job = client.resize_hw(nv12_array(64, 32), 32, 16, fmt="nv12", wait=False)
        # the poll reports the terminal state without raising…
        assert job.done() is True
        # …and the error surfaces from the consuming wait
        with pytest.raises(DspError, match="HAL -2801"):
            job.wait()

    def test_unknown_job_id_raises(self):
        client, _ = async_client(
            wait_script=[camera_pb2.DspJobResponse(
                success=False, error_code=-2, message="unknown or reaped job id"
            )]
        )
        job = client.resize_hw(nv12_array(64, 32), 32, 16, fmt="nv12", wait=False)
        with pytest.raises(DspError, match="unknown or reaped"):
            job.wait_result()

    def test_wait_rpc_unimplemented_surfaces(self):
        client, _ = async_client(wait_unimplemented=True)
        job = client.resize_hw(nv12_array(64, 32), 32, 16, fmt="nv12", wait=False)
        from neoruntime_ipc_sdk.dsp_wire import _DspUnavailable

        with pytest.raises(_DspUnavailable):
            job.wait_result()


class TestSyncFallback:
    def test_old_daemon_falls_back_to_sync_submit(self):
        client, stub = async_client(unimplemented_async=True)

        job = client.resize_hw(nv12_array(64, 32), 32, 16, fmt="nv12", wait=False)

        # the async rpc was tried, the sync rpc ran the job
        assert stub.async_requests and stub.sync_requests
        assert job.job_id is None  # no daemon job id — born done
        assert job.done() is True
        out = job.wait()
        assert not stub.wait_requests  # no WaitDspJob needed
        assert out.shape == (24, 32)

    def test_blend_async_array_base_single_job(self):
        client, stub = async_client()
        importing_overlays(client)

        job = client.blend_hw(nv12_array(64, 32), [(overlay_solid(16, 16), 0, 0)],
                              wait=False)

        req = stub.async_requests[0]
        assert req.op == camera_pb2.DSP_OP_BLEND
        assert req.src_buffer_id == 1000
        assert list(req.dst_buffer_ids) == [2000]  # the overlay import
        out = job.wait()
        assert out.shape == (48, 64)

    def test_blend_handle_base_copy_leg_stays_sync(self):
        # the zero-copy chain's RESIZE leg runs synchronously even under
        # wait=False: an async job nobody waits would leak its registry
        # entry (and eat one of the per-client pending slots)
        from neoruntime_ipc_sdk.frame import FrameHandle

        client, stub = async_client()
        importing_overlays(client)

        def fake_source(handle, width, height, fmt, timeout_s=5.0):
            return 3000

        client._import_source = fake_source
        handle = FrameHandle(
            [memfd(64 * 32 * 3 // 2)], (64,), (64 * 32 * 3 // 2,), 9,
            width=64, height=32, format="NV12",
        )

        job = client.blend_hw(handle, [(overlay_solid(16, 16), 0, 0)],
                              wait=False, zero_copy=True)

        sync_req = stub.sync_requests[0]
        assert sync_req.op == camera_pb2.DSP_OP_RESIZE  # the copy leg
        assert sync_req.src_buffer_id == 3000
        async_req = stub.async_requests[0]
        assert async_req.op == camera_pb2.DSP_OP_BLEND  # only the blend async
        assert job.job_id == 11


class TestEncodeChain:
    def test_buffer_id_feeds_encode_without_readback(self):
        client, stub = async_client()
        rgb = np.zeros((32, 64, 3), np.uint8)

        job = client.convert_hw(rgb, "nv12", wait=False)
        job.wait_result()  # completion without reading the array

        jpeg = client.encode_jpeg_hw(None, quality=80, src_buffer_id=job.buffer_id)

        req = stub.encode_requests[0]
        assert req.src_buffer_id == job.buffer_id == 1001  # the dst pool
        assert req.quality == 80
        assert jpeg == b"\xff\xd8FAKE"
        job.release()

    def test_buffer_id_with_src_rejected(self):
        client, _ = async_client()
        with pytest.raises(DspError, match="not both"):
            client.encode_jpeg_hw(
                nv12_array(64, 32), src_buffer_id=1001
            )

    def test_buffer_id_unavailable_raises_no_fallback(self):
        # no client pixels exist on this leg — unavailability must raise,
        # not silently fall back
        client, _ = async_client(encode_unimplemented=True)
        with pytest.raises(DspError, match="no client pixels"):
            client.encode_jpeg_hw(None, src_buffer_id=1001)


class TestGray8Quiet:
    def test_gray8_refusal_learned_then_quiet(self):
        def reject_gray8(request):
            if request.op == camera_pb2.DSP_OP_CONVERT_FORMAT:
                return camera_pb2.DspJobResponse(
                    success=False, error_code=-1, message="HAL refuses gray8"
                )
            return camera_pb2.DspJobResponse(success=True, elapsed_ms=1)

        client, stub = async_client(sync_response=reject_gray8)
        g = np.full((32, 64), 128, np.uint8)

        # first refusal: warned, CPU fallback, degradation recorded
        with pytest.warns(UserWarning, match="rejected gray8->rgb24"):
            out1 = client.convert_hw(g, "rgb24", fmt="gray8")
        assert client.last_used_hw is False
        assert out1.shape == (32, 64, 3)

        # later pairs: same pixels, no warning, no doomed submit
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            out2 = client.convert_hw(g, "rgb24", fmt="gray8")
        assert np.array_equal(out1, out2)
        converts = [r for r in stub.sync_requests
                    if r.op == camera_pb2.DSP_OP_CONVERT_FORMAT]
        assert len(converts) == 1  # only the first, refused attempt

    def test_gray8_handle_after_refusal_still_raises(self):
        from neoruntime_ipc_sdk.frame import FrameHandle

        def reject_gray8(request):
            if request.op == camera_pb2.DSP_OP_CONVERT_FORMAT:
                return camera_pb2.DspJobResponse(
                    success=False, error_code=-1, message="HAL refuses gray8"
                )
            return camera_pb2.DspJobResponse(success=True, elapsed_ms=1)

        client, _ = async_client(sync_response=reject_gray8)
        arr = np.full((32, 64), 128, np.uint8)
        with pytest.warns(UserWarning, match="gray8"):
            client.convert_hw(arr, "rgb24", fmt="gray8")  # learn the firmware gap

        # a zero-copy source can never take the quiet CPU leg — it must
        # keep failing loudly instead of pretending to fall back
        client._import_source = lambda *a, **k: 3000
        handle = FrameHandle(
            [memfd(64 * 32)], (64,), (64 * 32,), 10,
            width=64, height=32, format="GRAY8",
        )
        with pytest.raises(DspError, match="gray8 conversions are firmware-refused"):
            client.convert_hw(handle, "rgb24")
