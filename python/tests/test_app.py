"""Tests for AppClient against the real proto surface.

Regression (device run 2026-09-09): ``list_apps`` built its request with
``app_pb2.Empty()``, but app_pb2 defines no ``Empty`` — the generated
stub declares ``google.protobuf.empty_pb2.Empty`` as the ListApps
request type, and every call died with AttributeError before reaching
the daemon (5 device cases). These tests pin the request type.
"""

from unittest.mock import Mock, patch

from google.protobuf import empty_pb2

from neoruntime_ipc_sdk import AppClient
from neoruntime_ipc_sdk.proto import app_pb2


class TestAppClientList:
    @patch('neoruntime_ipc_sdk.app.grpc.insecure_channel')
    def test_list_apps_sends_protobuf_empty(self, mock_channel):
        client = AppClient()
        mock_stub = Mock()
        mock_stub.ListApps.return_value = app_pb2.AppList()
        client.stub = mock_stub

        apps = client.list_apps()

        assert apps == []
        request = mock_stub.ListApps.call_args[0][0]
        assert isinstance(request, empty_pb2.Empty)
        assert request.SerializeToString() == b""

    @patch('neoruntime_ipc_sdk.app.grpc.insecure_channel')
    def test_list_apps_parses_entries(self, mock_channel):
        client = AppClient()
        mock_stub = Mock()
        mock_stub.ListApps.return_value = app_pb2.AppList(apps=[
            app_pb2.AppInfo(id="demo"),
        ])
        client.stub = mock_stub

        apps = client.list_apps()

        assert len(apps) == 1
        assert apps[0].id == "demo"
