"""diagnostics() snapshot — never raises, stable structure."""

import pytest

from neoruntime_ipc_sdk import diagnostics
from neoruntime_ipc_sdk.diagnostics import _uds_path


class TestUdsPath:
    def test_strips_unix_scheme(self):
        assert _uds_path("unix:///run/aipc/x.sock") == "/run/aipc/x.sock"

    def test_plain_path_unchanged(self):
        assert _uds_path("/run/aipc/x.sock") == "/run/aipc/x.sock"


class TestDiagnostics:
    @pytest.fixture(autouse=True)
    def _isolate_env(self, monkeypatch, tmp_path):
        # Point every endpoint at a nonexistent dir: the probes must report
        # exists=False without raising (no daemons on a dev machine).
        for var in (
            "AI_RUNTIME_ENDPOINT",
            "EVENT_BUS_ENDPOINT",
            "DEVICE_CONTROL_ENDPOINT",
            "CAMERA_CONTROL_ENDPOINT",
            "APP_MANAGER_ENDPOINT",
        ):
            monkeypatch.setenv(var, f"unix://{tmp_path}/nope/{var}.sock")
        monkeypatch.setenv("CAMERA_SOCK_PATH", str(tmp_path / "nope" / "camera.sock"))
        monkeypatch.setenv("ENCODED_SOCKET_DIR", str(tmp_path / "nope" / "encoded"))

    def test_structure_and_never_raises(self):
        report = diagnostics()

        assert set(report) >= {
            "sdk_version", "python", "optional_deps", "services",
            "media", "accel_router", "accel_probes",
        }
        assert report["sdk_version"]
        deps = report["optional_deps"]
        assert deps["numpy"]["available"] is True
        assert deps["pillow"]["available"] is True  # hard dependency
        assert isinstance(deps["cv2"]["available"], bool)  # dev-machine dependent
        for svc in report["services"].values():
            assert svc["exists"] is False
            assert svc["connectable"] is False
        assert report["media"]["camera_sock"]["exists"] is False
        assert report["media"]["encoded_dir"]["streams"] == []
        assert report["accel_router"]["policy"]

    def test_lists_encoded_sockets(self, monkeypatch, tmp_path):
        enc = tmp_path / "encoded"
        enc.mkdir()
        for name in ("main.sock", "sub.sock", "notes.txt"):
            (enc / name).touch()
        monkeypatch.setenv("ENCODED_SOCKET_DIR", str(enc))

        report = diagnostics()

        assert report["media"]["encoded_dir"]["streams"] == ["main", "sub"]


class TestCheck:
    @pytest.fixture(autouse=True)
    def _isolate_env(self, monkeypatch, tmp_path):
        for var in (
            "AI_RUNTIME_ENDPOINT", "EVENT_BUS_ENDPOINT", "DEVICE_CONTROL_ENDPOINT",
            "CAMERA_CONTROL_ENDPOINT", "APP_MANAGER_ENDPOINT",
        ):
            monkeypatch.setenv(var, f"unix://{tmp_path}/nope/{var}.sock")
        monkeypatch.setenv("CAMERA_SOCK_PATH", str(tmp_path / "nope" / "camera.sock"))
        monkeypatch.setenv("ENCODED_SOCKET_DIR", str(tmp_path / "nope" / "encoded"))

    def test_healthy_when_nothing_required(self):
        from neoruntime_ipc_sdk.diagnostics import check

        report = check()
        assert report["healthy"] is True and report["problems"] == []
        assert set(report["checks"]) >= {"inference", "camera", "cv2", "dsp"}

    def test_failed_requirements_carry_hints(self):
        from neoruntime_ipc_sdk.diagnostics import check

        report = check(["inference", "camera_sock"])
        assert report["healthy"] is False
        assert report["checks"]["inference"] is False
        assert any("ai-runtime" in p for p in report["problems"])
        assert any("video frames" in p for p in report["problems"])
        with pytest.raises(RuntimeError, match="health check failed"):
            report.raise_if_unhealthy()

    def test_unknown_check_name_rejected(self):
        from neoruntime_ipc_sdk.diagnostics import check

        with pytest.raises(ValueError, match="unknown check"):
            check(["quantum_link"])

    def test_model_and_stream_probes(self):
        from types import SimpleNamespace

        from neoruntime_ipc_sdk.diagnostics import check

        class _FakeInference:
            def get_model_info(self, model_id):
                return None if model_id == "missing" else object()

        class _FakeCamera:
            def get_stream_status(self, timeout_s=None):
                return [SimpleNamespace(stream_id="main", status="active"),
                        SimpleNamespace(stream_id="sub", status="starting")]

        ok = check(["model_registered", "stream_active"],
                   inference=_FakeInference(), model_id="person_v1",
                   camera=_FakeCamera(), stream_id="main")
        assert ok["healthy"] is True

        bad = check(["model_registered", "stream_active"],
                    inference=_FakeInference(), model_id="missing",
                    camera=_FakeCamera(), stream_id="sub")
        assert bad["healthy"] is False
        assert any("not registered" in p for p in bad["problems"])
        assert any("status: starting" in p for p in bad["problems"])
