"""Channel-option tests: every gRPC channel the SDK creates must lift grpc's
4 MiB default receive limit to the ai-runtime server's 64 MiB cap, or large
inference responses fail client-side with ResourceExhausted.
"""

from __future__ import annotations

from unittest.mock import patch

from neoruntime_ipc_sdk import CameraClient, InferenceClient, OverlayClient
from neoruntime_ipc_sdk._transport import (
    MAX_GRPC_MESSAGE_LENGTH,
    max_message_length_options,
)


def test_max_message_length_matches_server_limit():
    # ai-runtime main.cpp: SetMax{Send,Receive}MessageSize(64 MiB)
    assert MAX_GRPC_MESSAGE_LENGTH == 64 * 1024 * 1024
    assert max_message_length_options() == [
        ("grpc.max_receive_message_length", MAX_GRPC_MESSAGE_LENGTH)
    ]


def test_grpc_client_base_applies_limit_on_connect():
    # Arrange — patch grpc so no real channel/socket is created
    with patch("neoruntime_ipc_sdk._transport.grpc") as grpc_mod:
        client = CameraClient(endpoint="unix:///tmp/test.sock")
        client.connect()

        # Act / Assert — the option reached insecure_channel
        _args, kwargs = grpc_mod.insecure_channel.call_args
        assert ("grpc.max_receive_message_length", MAX_GRPC_MESSAGE_LENGTH) in (
            kwargs.get("options") or []
        )


def test_overlay_override_keeps_limit_and_adds_epoll1():
    options = OverlayClient(endpoint="unix:///tmp/test.sock").channel_options

    assert ("grpc.max_receive_message_length", MAX_GRPC_MESSAGE_LENGTH) in options
    assert ("grpc.poll_strategy", 1) in options


def test_inference_client_aio_channel_applies_limit():
    # Arrange — patch the grpc module inference.py sees before any connect
    with patch("neoruntime_ipc_sdk.inference.grpc") as grpc_mod:
        import asyncio

        client = InferenceClient(endpoint="unix:///tmp/test.sock")
        asyncio.run(client._connect_async())

        # Act / Assert
        _args, kwargs = grpc_mod.aio.insecure_channel.call_args
        options = dict(kwargs.get("options") or [])
        assert options.get("grpc.max_receive_message_length") == MAX_GRPC_MESSAGE_LENGTH
        assert options.get("grpc.poll_strategy") == 1
