"""Phase 5 — plugin SDK surface (deprecated module, self-hosted).

``neoruntime_ipc_sdk.plugin`` is deprecated (the platform does not ship
``/run/aipc/plugins``), so this module is verified *self-hosted*: a
PluginServer runs on a socket under the test tmp dir, a hand-written
discovery.json sits beside it, and PluginDiscovery resolves against
that directory. The deprecation warning itself is part of the contract
and is asserted. Nothing touches the platform's /run/aipc.
"""

from __future__ import annotations

import json
import os
import shutil
import threading
import unittest
import warnings

import grpc

from neoruntime_ipc_sdk.plugin import (
    PluginDiscovery,
    PluginEndpoint,
    PluginServer,
)

from common import DEVICE_TMP_DIR, DeviceTestCase

CAP = "sdk-test-capability"
APP = "sdk-test-app"


def _write_discovery(plugin_dir: str, socket_path: str, state: str = "running"):
    data = {
        "plugins": {
            APP: {
                "app_id": APP,
                "state": state,
                "capabilities": [{
                    "id": CAP,
                    "version": "1.0.0",
                    "transport": "grpc",
                    "grpc": {"socket_path": socket_path,
                             "service": "sdktest.SdkTest"},
                    "event": {"publish": ["sdk-test/ping"],
                              "subscribe": []},
                }],
            }
        }
    }
    path = os.path.join(plugin_dir, "discovery.json")
    with open(path, "w") as fh:
        json.dump(data, fh)
    return path


class _PluginArea(DeviceTestCase):
    area = "plugin"
    timeout_s = 90

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.plugin_dir = os.path.join(DEVICE_TMP_DIR, "plugins")
        os.makedirs(cls.plugin_dir, exist_ok=True)
        cls.sock = os.path.join(cls.plugin_dir, "sdk-test-plugin.sock")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            cls.server = PluginServer("sdk-test-plugin",
                                      socket_dir=cls.plugin_dir)
            cls.grpc_server = cls.server.create_server()
            cls.server.start()
            cls.discovery = PluginDiscovery(discovery_dir=cls.plugin_dir)
        cls.deprecations = [str(w.message) for w in caught
                            if issubclass(w.category, DeprecationWarning)]
        _write_discovery(cls.plugin_dir, cls.sock)
        cls.discovery.reload()

    @classmethod
    def tearDownClass(cls):
        cls.discovery.close()
        cls.server.stop(0.0)
        shutil.rmtree(cls.plugin_dir, ignore_errors=True)


class T01ServerLifecycle(_PluginArea):
    def test_01_deprecation_warning(self):
        self.mark("plugin module deprecation contract")
        self.evidence(warnings=self.deprecations)
        self.assertTrue(self.deprecations,
                        "module use must emit a DeprecationWarning")

    def test_02_create_start_stop(self):
        self.mark("PluginServer.create_server/start/wait/stop")
        ephemeral = PluginServer("sdk-test-ephemeral",
                                 socket_dir=self.plugin_dir)
        server = ephemeral.create_server(max_workers=1)
        ephemeral.start()
        alive = os.path.exists(ephemeral.socket_path)
        waited = threading.Thread(target=ephemeral.wait, daemon=True)
        waited.start()
        ephemeral.stop(grace=0.0)
        waited.join(timeout=5.0)
        removed = not os.path.exists(ephemeral.socket_path)
        self.evidence(server_type=type(server).__name__,
                      socket=ephemeral.socket_path,
                      socket_alive=alive, socket_removed=removed)
        self.assertIsInstance(server, grpc.Server)
        self.assertTrue(alive, "plugin socket not created on start")
        self.assertTrue(removed, "plugin socket not cleaned up on stop")


class T02Discovery(_PluginArea):
    def test_01_list_plugins_and_capabilities(self):
        self.mark("PluginDiscovery.reload/list_plugins/list_capabilities")
        self.timed(self.discovery.reload, label="reload")
        plugins = self.discovery.list_plugins()
        caps = self.discovery.list_capabilities()
        self.evidence(plugins=list(plugins), capabilities=caps)
        self.assertIn(APP, plugins)
        self.assertIn(CAP, caps)

    def test_02_get(self):
        self.mark("PluginDiscovery.get/PluginEndpoint.is_available")
        endpoint = self.timed(self.discovery.get, CAP, label="get")
        self.evidence(app_id=endpoint.app_id,
                      capability=endpoint.capability_id,
                      version=endpoint.version, transport=endpoint.transport,
                      state=endpoint.state,
                      is_available=endpoint.is_available)
        self.assertIsInstance(endpoint, PluginEndpoint)
        self.assertTrue(endpoint.is_available)

    def test_03_connect(self):
        self.mark("PluginEndpoint.connect")
        endpoint = self.discovery.get(CAP)
        channel = self.timed(endpoint.connect, label="connect")
        self.evidence(channel_type=type(channel).__name__,
                      target=endpoint.socket_path)
        self.assertIsInstance(channel, grpc.Channel)
        channel.close()

    def test_04_require(self):
        self.mark("PluginDiscovery.require")
        endpoint = self.timed(self.discovery.require, CAP, 5.0,
                              label="require")
        self.evidence(app_id=endpoint.app_id)
        self.assertEqual(endpoint.app_id, APP)

    def test_05_require_timeout(self):
        self.mark("PluginDiscovery.require (timeout path)")
        with self.assertRaises(TimeoutError):
            self.discovery.require("no-such-capability", timeout=0.5)
        self.evidence(raised="TimeoutError")

    def test_06_watch(self):
        self.mark("PluginDiscovery.watch")
        fired = threading.Event()
        self.discovery.watch(fired.set)
        # Bump mtime so the 2s poll loop notices the change.
        path = os.path.join(self.plugin_dir, "discovery.json")
        os.utime(path, None)
        joined = fired.wait(timeout=10.0)
        self.evidence(fired=joined)
        self.assertTrue(joined, "watch callback never fired after "
                                "discovery.json changed")


if __name__ == "__main__":
    unittest.main()
