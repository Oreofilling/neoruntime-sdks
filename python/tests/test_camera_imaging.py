"""
Tests for CameraClient imaging/IR/preset/privacy-mask/OSD/config-field APIs.
"""

from unittest.mock import Mock, patch

import pytest

from neoruntime_ipc_sdk import CameraClient
from neoruntime_ipc_sdk.camera import InfraredStatus, IrPreset, PrivacyMaskSettings
from neoruntime_ipc_sdk.proto import camera_pb2


def make_client(**stub_returns):
    mock_stub = Mock()
    for name, value in stub_returns.items():
        setattr(mock_stub, name, Mock(return_value=value))
    client = CameraClient()
    client._stub = mock_stub
    return client, mock_stub


def ir_response(**over):
    kw = dict(success=True, message="", mode="infrared", transition="done",
              output_source="ir", auto_follow=True, follow_active=True,
              manual_override=False, degraded=False,
              requested_near_pwm=80, requested_far_pwm=60,
              applied_near_pwm=78, applied_far_pwm=58,
              zoom_ratio=1.5, active_profile="night",
              selected_mode="auto", light_percent=10, light_mv=120,
              light_milli=900, light_valid=True, night_enter=15, day_enter=40)
    kw.update(over)
    return camera_pb2.InfraredStatusResponse(**kw)


class TestImagingMode:
    @patch("neoruntime_ipc_sdk.camera.grpc.insecure_channel")
    @pytest.mark.parametrize("mode", ["day", "infrared", "auto"])
    def test_sends_mode(self, _ch, mode):
        client, stub = make_client(SetImagingMode=ir_response(mode=mode))
        st = client.set_imaging_mode(mode)
        req = stub.SetImagingMode.call_args[0][0]
        assert req.mode == mode
        assert isinstance(st, InfraredStatus)
        assert st.mode == mode

    @patch("neoruntime_ipc_sdk.camera.grpc.insecure_channel")
    def test_invalid_mode_raises(self, _ch):
        client, _ = make_client(SetImagingMode=ir_response())
        with pytest.raises(ValueError):
            client.set_imaging_mode("night")

    @patch("neoruntime_ipc_sdk.camera.grpc.insecure_channel")
    def test_failure_raises(self, _ch):
        client, _ = make_client(
            SetImagingMode=ir_response(success=False, message="hw busy"))
        with pytest.raises(RuntimeError, match="hw busy"):
            client.set_imaging_mode("day")


class TestInfraredStatus:
    @patch("neoruntime_ipc_sdk.camera.grpc.insecure_channel")
    def test_get_maps_all_fields(self, _ch):
        client, _ = make_client(GetInfraredStatus=ir_response())
        st = client.get_infrared_status()
        assert isinstance(st, InfraredStatus)
        assert st.mode == "infrared"
        assert st.transition == "done"
        assert st.output_source == "ir"
        assert st.auto_follow is True
        assert st.follow_active is True
        assert st.manual_override is False
        assert st.degraded is False
        assert st.requested_near_pwm == 80
        assert st.requested_far_pwm == 60
        assert st.applied_near_pwm == 78
        assert st.applied_far_pwm == 58
        assert st.zoom_ratio == pytest.approx(1.5)
        assert st.active_profile == "night"
        assert st.selected_mode == "auto"
        assert st.light_percent == 10
        assert st.light_mv == 120
        assert st.light_milli == 900
        assert st.light_valid is True
        assert st.night_enter == 15
        assert st.day_enter == 40

    @patch("neoruntime_ipc_sdk.camera.grpc.insecure_channel")
    def test_set_settings_sends_only_given_fields(self, _ch):
        client, stub = make_client(SetInfraredSettings=ir_response())
        client.set_infrared_settings(near_pwm=90, night_enter=20)
        req = stub.SetInfraredSettings.call_args[0][0]
        assert req.near_pwm == 90
        assert req.night_enter == 20
        assert not req.HasField("auto_follow")
        assert not req.HasField("far_pwm")
        assert not req.HasField("day_enter")

    @patch("neoruntime_ipc_sdk.camera.grpc.insecure_channel")
    def test_clear_manual(self, _ch):
        client, stub = make_client(ClearInfraredManual=ir_response())
        st = client.clear_infrared_manual()
        stub.ClearInfraredManual.assert_called_once()
        assert st.manual_override is False


def preset_list():
    return camera_pb2.IrPresetListResponse(
        success=True, presets=[
            camera_pb2.IrPreset(name="gate", zoom_ratio=1.0,
                                near_pwm=70, far_pwm=50),
            camera_pb2.IrPreset(name="far", zoom_ratio=2.5,
                                near_pwm=30, far_pwm=80),
        ])


class TestIrPresets:
    @patch("neoruntime_ipc_sdk.camera.grpc.insecure_channel")
    def test_list(self, _ch):
        client, _ = make_client(ListIrPresets=preset_list())
        presets = client.list_ir_presets()
        assert len(presets) == 2
        assert all(isinstance(p, IrPreset) for p in presets)
        assert presets[0].name == "gate"
        assert presets[1].zoom_ratio == pytest.approx(2.5)

    @patch("neoruntime_ipc_sdk.camera.grpc.insecure_channel")
    def test_save_sends_fields(self, _ch):
        client, stub = make_client(SaveIrPreset=preset_list())
        presets = client.save_ir_preset("door", 1.8, 40, 60)
        req = stub.SaveIrPreset.call_args[0][0]
        assert req.name == "door"
        assert req.zoom_ratio == pytest.approx(1.8)
        assert req.near_pwm == 40
        assert req.far_pwm == 60
        assert len(presets) == 2

    @patch("neoruntime_ipc_sdk.camera.grpc.insecure_channel")
    def test_delete_sends_name(self, _ch):
        client, stub = make_client(DeleteIrPreset=preset_list())
        client.delete_ir_preset("gate")
        req = stub.DeleteIrPreset.call_args[0][0]
        assert req.name == "gate"


def mask_config():
    cfg = camera_pb2.PrivacyMaskConfig(
        color=0x00FF00, blur_radius=8, enabled=False,
        dpm_enabled=True, dpm_labels="person,vehicle",
        dpm_mode="mosaic", dpm_color=0xFF0000)
    cfg.regions.add(id="r1", name="yard", enabled=True,
                        points_x=[0.1, 0.5, 0.5], points_y=[0.2, 0.2, 0.6])
    return cfg


class TestPrivacyMask:
    @patch("neoruntime_ipc_sdk.camera.grpc.insecure_channel")
    def test_get_maps_settings(self, _ch):
        client, _ = make_client(GetPrivacyMaskConfig=mask_config())
        s = client.get_privacy_mask()
        assert isinstance(s, PrivacyMaskSettings)
        assert s.color == 0x00FF00
        assert s.blur_radius == 8
        assert s.enabled is False
        assert s.dpm_enabled is True
        assert s.dpm_labels == "person,vehicle"
        assert s.dpm_mode == "mosaic"
        assert s.dpm_color == 0xFF0000
        assert len(s.regions) == 1
        region = s.regions[0]
        assert region["id"] == "r1"
        assert region["name"] == "yard"
        assert region["enabled"] is True
        assert list(region["points_x"]) == pytest.approx([0.1, 0.5, 0.5])

    @patch("neoruntime_ipc_sdk.camera.grpc.insecure_channel")
    def test_set_overrides_keep_unspecified_fields(self, _ch):
        client, stub = make_client(
            GetPrivacyMaskConfig=mask_config(),
            SetPrivacyMaskConfig=Mock(success=True, message=""))
        client.set_privacy_mask(enabled=True)
        req = stub.SetPrivacyMaskConfig.call_args[0][0]
        assert req.enabled is True
        # untouched fields preserved from the current config
        assert req.blur_radius == 8
        assert req.color == 0x00FF00
        assert len(req.regions) == 1
        assert req.regions[0].id == "r1"
        assert req.dpm_labels == "person,vehicle"

    @patch("neoruntime_ipc_sdk.camera.grpc.insecure_channel")
    def test_set_new_regions(self, _ch):
        client, stub = make_client(
            GetPrivacyMaskConfig=mask_config(),
            SetPrivacyMaskConfig=Mock(success=True, message=""))
        client.set_privacy_mask(
            regions=[{"id": "r2", "name": "win", "enabled": True,
                      "points_x": [0.0, 0.1], "points_y": [0.0, 0.1]}])
        req = stub.SetPrivacyMaskConfig.call_args[0][0]
        assert len(req.regions) == 1
        assert req.regions[0].id == "r2"

    @patch("neoruntime_ipc_sdk.camera.grpc.insecure_channel")
    def test_set_failure_raises(self, _ch):
        client, _ = make_client(
            GetPrivacyMaskConfig=mask_config(),
            SetPrivacyMaskConfig=camera_pb2.Status(
                success=False, message="too many"))
        with pytest.raises(RuntimeError, match="too many"):
            client.set_privacy_mask(enabled=False)


class TestOsdReadback:
    @patch("neoruntime_ipc_sdk.camera.grpc.insecure_channel")
    def test_get_osd_symmetric_with_set_osd(self, _ch):
        resp = camera_pb2.OsdConfigResponse(streams=[
            camera_pb2.StreamOsdConfig(
                stream_name="main",
                text_overlays=[camera_pb2.OsdTextOverlayConfig(
                    id="t1", text="hello", x=1.0, y=2.0, font_size=16,
                    text_color=16777215, enabled=True)],
                datetime_overlays=[camera_pb2.OsdDateTimeOverlayConfig(
                    id="d1", x=0.0, y=10.0, format="%Y", enabled=True)]),
        ])
        client, _ = make_client(GetOsdConfig=resp)
        streams = client.get_osd()
        assert len(streams) == 1
        s = streams[0]
        assert s["stream_name"] == "main"
        assert s["text_overlays"][0]["text"] == "hello"
        assert s["text_overlays"][0]["enabled"] is True
        assert s["datetime_overlays"][0]["format"] == "%Y"


class TestConfigField:
    @patch("neoruntime_ipc_sdk.camera.grpc.insecure_channel")
    def test_get_returns_value(self, _ch):
        client, stub = make_client(
            GetConfigField=camera_pb2.GetConfigFieldResponse(
                success=True, type=0, value="true"))
        assert client.get_config_field("frontend.hailort.enabled") == "true"
        req = stub.GetConfigField.call_args[0][0]
        assert req.field_path == "frontend.hailort.enabled"

    @patch("neoruntime_ipc_sdk.camera.grpc.insecure_channel")
    def test_get_failure_raises(self, _ch):
        client, _ = make_client(
            GetConfigField=camera_pb2.GetConfigFieldResponse(
                success=False, message="no such field"))
        with pytest.raises(RuntimeError, match="no such field"):
            client.get_config_field("bogus.path")

    @patch("neoruntime_ipc_sdk.camera.grpc.insecure_channel")
    def test_set_encodes_value_as_string(self, _ch):
        client, stub = make_client(
            SetConfigField=Mock(success=True, message=""))
        client.set_config_field("frontend.hailort.enabled", "false")
        req = stub.SetConfigField.call_args[0][0]
        assert req.field_path == "frontend.hailort.enabled"
        assert req.value == "false"

    @patch("neoruntime_ipc_sdk.camera.grpc.insecure_channel")
    def test_set_failure_raises(self, _ch):
        client, _ = make_client(
            SetConfigField=camera_pb2.Status(
                success=False, message="read-only"))
        with pytest.raises(RuntimeError, match="read-only"):
            client.set_config_field("x.y", "1")
