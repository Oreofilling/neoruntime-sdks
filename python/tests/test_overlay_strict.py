"""Tests for OverlayClient strict frame-lock configuration (P1-6).

The daemon's UpdateAiOverlay handler treats AiOverlayConfig.strict_frame_lock
(field 9) and strict_wait_cap_ms (field 10) as optional: absent = keep the
current (yaml) setting. So the client must send each field only when the
caller passed it — a falsy-but-explicit value (False, 0) still goes on the
wire, and a plain configure() call sends neither. No real socket is touched:
the stub is a Mock.
"""

from __future__ import annotations

from unittest.mock import Mock, patch

import pytest

from neoruntime_ipc_sdk import OverlayClient, OverlayConfig


@pytest.fixture
def client():
    with patch("neoruntime_ipc_sdk.overlay.grpc.insecure_channel"):
        oc = OverlayClient()
        mock_stub = Mock()
        mock_stub.UpdateAiOverlay.return_value = Mock(success=True, message="")
        oc._stub = mock_stub
        return oc


def _sent_config(client) -> object:
    """The AiOverlayConfig handed to the daemon by the last _update call."""
    client._stub.UpdateAiOverlay.assert_called_once()
    return client._stub.UpdateAiOverlay.call_args[0][0]


class TestConfigureStrict:
    def test_opt_in_sets_field_cap_absent(self, client):
        client.configure(strict_frame_lock=True)
        cfg = _sent_config(client)
        assert cfg.HasField("strict_frame_lock") and cfg.strict_frame_lock
        assert not cfg.HasField("strict_wait_cap_ms")

    def test_explicit_opt_out_is_sent_not_dropped(self, client):
        # False is meaningful (leave strict mode); the falsy check pattern
        # used for box_color must not swallow it
        client.configure(strict_frame_lock=False)
        cfg = _sent_config(client)
        assert cfg.HasField("strict_frame_lock") and not cfg.strict_frame_lock

    def test_cap_zero_means_derive_from_fps(self, client):
        client.configure(strict_wait_cap_ms=0)
        cfg = _sent_config(client)
        assert cfg.HasField("strict_wait_cap_ms") and cfg.strict_wait_cap_ms == 0

    def test_explicit_cap_value(self, client):
        client.configure(strict_frame_lock=True, strict_wait_cap_ms=66)
        cfg = _sent_config(client)
        assert cfg.HasField("strict_frame_lock") and cfg.strict_frame_lock
        assert cfg.HasField("strict_wait_cap_ms") and cfg.strict_wait_cap_ms == 66

    def test_plain_configure_keeps_daemon_settings(self, client):
        client.configure()
        cfg = _sent_config(client)
        assert not cfg.HasField("strict_frame_lock")
        assert not cfg.HasField("strict_wait_cap_ms")

    def test_enable_disable_leave_strict_untouched(self, client):
        client.enable()
        cfg = _sent_config(client)
        assert not cfg.HasField("strict_frame_lock")
        client._stub.UpdateAiOverlay.reset_mock()
        client.disable()
        cfg = _sent_config(client)
        assert not cfg.HasField("strict_frame_lock")


class TestConfigureStrictValidation:
    @pytest.mark.parametrize("bad", [-1, 5001, True, "66", 1.5])
    def test_bad_cap_rejected_without_rpc(self, client, bad):
        with pytest.raises(ValueError, match="strict_wait_cap_ms"):
            client.configure(strict_wait_cap_ms=bad)
        client._stub.UpdateAiOverlay.assert_not_called()

    def test_boundary_caps_accepted(self, client):
        client.configure(strict_wait_cap_ms=5000)
        assert _sent_config(client).strict_wait_cap_ms == 5000


class TestOverlayConfigStrict:
    def test_config_object_sets_both_fields(self, client):
        cfg_obj = OverlayConfig(strict_frame_lock=True, strict_wait_cap_ms=80)
        client.apply(cfg_obj)
        cfg = _sent_config(client)
        assert cfg.HasField("strict_frame_lock") and cfg.strict_frame_lock
        assert cfg.HasField("strict_wait_cap_ms") and cfg.strict_wait_cap_ms == 80

    def test_config_object_defaults_absent(self, client):
        client.apply(OverlayConfig())
        cfg = _sent_config(client)
        assert not cfg.HasField("strict_frame_lock")
        assert not cfg.HasField("strict_wait_cap_ms")

    def test_to_proto_direct(self):
        proto = OverlayConfig(strict_frame_lock=False).to_proto()
        assert proto.HasField("strict_frame_lock") and not proto.strict_frame_lock
