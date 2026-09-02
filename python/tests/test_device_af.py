"""
Tests for DeviceClient native autofocus APIs:
start_oneshot_af / start_zoom_follow / get_autofocus_status /
cancel_autofocus / set_af_windows / get_af_measurement
"""

from unittest.mock import Mock, patch

import pytest

from neoruntime_ipc_sdk import DeviceClient
from neoruntime_ipc_sdk.device import AfJob, AfMeasurement, AfStatus


def make_client(**stub_returns):
    mock_stub = Mock()
    for name, value in stub_returns.items():
        setattr(mock_stub, name, Mock(return_value=value))
    client = DeviceClient()
    client.stub = mock_stub
    return client, mock_stub


class TestStartOneshotAf:
    @patch("neoruntime_ipc_sdk.device.grpc.insecure_channel")
    def test_returns_af_job(self, _ch):
        client, stub = make_client(
            StartOneShotAf=Mock(accepted=True, job_id=42, message="ok"))
        job = client.start_oneshot_af()
        assert isinstance(job, AfJob)
        assert job.accepted is True
        assert job.job_id == 42
        assert job.message == "ok"

    @patch("neoruntime_ipc_sdk.device.grpc.insecure_channel")
    def test_rejected_raises(self, _ch):
        client, _ = make_client(
            StartOneShotAf=Mock(accepted=False, job_id=0, message="busy"))
        with pytest.raises(RuntimeError, match="busy"):
            client.start_oneshot_af()


class TestStartZoomFollow:
    @patch("neoruntime_ipc_sdk.device.grpc.insecure_channel")
    def test_sends_ratio(self, _ch):
        client, stub = make_client(
            StartZoomFollow=Mock(accepted=True, job_id=7, message=""))
        job = client.start_zoom_follow(1.5)
        assert job.job_id == 7
        req = stub.StartZoomFollow.call_args[0][0]
        assert req.ratio == pytest.approx(1.5)


class TestGetAutofocusStatus:
    @patch("neoruntime_ipc_sdk.device.grpc.insecure_channel")
    def test_maps_full_response(self, _ch):
        resp = Mock(
            job_id=11, operation="oneshot", state="done", progress=1.0,
            busy=False, anchor_valid=True, requested_ratio=2.0,
            effective_ratio=1.9, zoom_pos=1000, focus_pos=500,
            best_focus=480, metric=12.5, confidence=0.9,
            reproducibility=0.8, estimated_distance_m=4.2,
            elapsed_ms=1500, error_code=0, message="")
        client, _ = make_client(GetAutofocusStatus=resp)
        st = client.get_autofocus_status()
        assert isinstance(st, AfStatus)
        assert st.job_id == 11
        assert st.operation == "oneshot"
        assert st.state == "done"
        assert st.progress == 1.0
        assert st.busy is False
        assert st.anchor_valid is True
        assert st.requested_ratio == pytest.approx(2.0)
        assert st.effective_ratio == pytest.approx(1.9)
        assert st.zoom_pos == 1000
        assert st.focus_pos == 500
        assert st.best_focus == 480
        assert st.metric == pytest.approx(12.5)
        assert st.confidence == pytest.approx(0.9)
        assert st.reproducibility == pytest.approx(0.8)
        assert st.estimated_distance_m == pytest.approx(4.2)
        assert st.elapsed_ms == 1500
        assert st.error_code == 0
        assert st.message == ""


class TestCancelAutofocus:
    @patch("neoruntime_ipc_sdk.device.grpc.insecure_channel")
    def test_default_cancels_active_job(self, _ch):
        client, stub = make_client(
            CancelAutofocus=Mock(success=True, message=""))
        client.cancel_autofocus()
        req = stub.CancelAutofocus.call_args[0][0]
        assert req.job_id == 0

    @patch("neoruntime_ipc_sdk.device.grpc.insecure_channel")
    def test_explicit_job_id(self, _ch):
        client, stub = make_client(
            CancelAutofocus=Mock(success=True, message=""))
        client.cancel_autofocus(job_id=42)
        req = stub.CancelAutofocus.call_args[0][0]
        assert req.job_id == 42

    @patch("neoruntime_ipc_sdk.device.grpc.insecure_channel")
    def test_failure_raises(self, _ch):
        client, _ = make_client(
            CancelAutofocus=Mock(success=False, message="no job"))
        with pytest.raises(RuntimeError, match="no job"):
            client.cancel_autofocus()


class TestSetAfWindows:
    @patch("neoruntime_ipc_sdk.device.grpc.insecure_channel")
    def test_sends_pixel_windows(self, _ch):
        client, stub = make_client(
            SetAfWindows=Mock(success=True, message=""))
        client.set_af_windows(True, [(100, 200, 300, 150), (0, 0, 640, 480)],
                              stream_id="sub")
        req = stub.SetAfWindows.call_args[0][0]
        assert req.enabled is True
        assert req.stream_id == "sub"
        assert len(req.windows) == 2
        w = req.windows[0]
        assert (w.x, w.y, w.w, w.h) == (100, 200, 300, 150)

    @patch("neoruntime_ipc_sdk.device.grpc.insecure_channel")
    def test_default_stream_is_main(self, _ch):
        client, stub = make_client(
            SetAfWindows=Mock(success=True, message=""))
        client.set_af_windows(False, [])
        req = stub.SetAfWindows.call_args[0][0]
        assert req.stream_id == "main"
        assert len(req.windows) == 0

    @patch("neoruntime_ipc_sdk.device.grpc.insecure_channel")
    def test_rejects_more_than_three_windows(self, _ch):
        client, _ = make_client(SetAfWindows=Mock(success=True, message=""))
        with pytest.raises(ValueError):
            client.set_af_windows(True, [(0, 0, 10, 10)] * 4)

    @patch("neoruntime_ipc_sdk.device.grpc.insecure_channel")
    def test_failure_raises(self, _ch):
        client, _ = make_client(
            SetAfWindows=Mock(success=False, message="rejected"))
        with pytest.raises(RuntimeError, match="rejected"):
            client.set_af_windows(True, [(0, 0, 10, 10)])


class TestGetAfMeasurement:
    @patch("neoruntime_ipc_sdk.device.grpc.insecure_channel")
    def test_maps_measurement(self, _ch):
        client, _ = make_client(
            GetAfMeasurement=Mock(focus_energy=[10, 20, 30],
                                  mean_luma=[50, 60, 70], frame_id=999))
        m = client.get_af_measurement()
        assert isinstance(m, AfMeasurement)
        assert m.focus_energy == [10, 20, 30]
        assert m.mean_luma == [50, 60, 70]
        assert m.frame_id == 999
