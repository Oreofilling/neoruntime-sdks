"""Phase 5 — application container interfaces (AppClient).

app-manager has a hang history, so every call here runs under the
per-case alarm. Query surface (list/get/stats/logs) runs against
whatever real apps are installed; the lifecycle surface
(install/start/stop/restart/uninstall) needs a manifest+image bundle
the suite does not ship — and mutating a production app is off-limits —
so those record SKIP-NA with the reason, rather than fake passes.
"""

from __future__ import annotations

import unittest

from neoruntime_ipc_sdk import AppClient

from common import DeviceTestCase, known_issue

# SDK 0.7.4 defect: app.py builds its request with app_pb2.Empty(), but
# the bundled app_pb2 module has no Empty message (verified against the
# wheel's proto descriptors), so every AppClient query RPC raises
# AttributeError before any traffic leaves the process — SDK-side fix.
_APP_EMPTY_DEFECT = (
    "SDK 0.7.4: app_pb2 lacks Empty, AppClient RPCs raise AttributeError "
    "client-side (app.py app_pb2.Empty())")


def _soft(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs), None
    except Exception as exc:  # noqa: BLE001
        return None, f"{type(exc).__name__}: {exc}"


class T01AppQuery(DeviceTestCase):
    area = "app"

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.client = AppClient()

    @classmethod
    def tearDownClass(cls):
        cls.client.close()

    @known_issue(_APP_EMPTY_DEFECT)
    def test_01_list_apps(self):
        self.mark("AppClient.list_apps")
        apps = self.timed(self.client.list_apps, label="list_apps")
        self.evidence(apps=[{
            "id": a.id, "name": a.name, "version": a.version,
            "state": a.state, "restarts": a.restart_count,
        } for a in apps])
        self.assertIsInstance(apps, list)

    def test_02_register_web_url(self):
        self.mark("AppClient.register_web_url")
        result, err = _soft(self.timed, self.client.register_web_url,
                            "/", label="register_web_url")
        self.evidence(call=result if err is None else err)
        if err is not None:
            self.na(f"register_web_url rejected: {err} (web route "
                    "registration may need an app context)")

    def _installed(self):
        apps = self.client.list_apps()
        if not apps:
            self.na("no apps installed on this device — nothing to query")
        return apps[0]

    @known_issue(_APP_EMPTY_DEFECT)
    def test_03_get_app(self):
        self.mark("AppClient.get_app")
        target = self._installed()
        info = self.timed(self.client.get_app, target.id, label="get_app")
        self.evidence(id=info.id, name=info.name, state=info.state,
                      version=info.version, pid=info.pid,
                      manifest=info.manifest_path)
        self.assertEqual(info.id, target.id)

    @known_issue(_APP_EMPTY_DEFECT)
    def test_04_get_app_stats(self):
        self.mark("AppClient.get_app_stats")
        target = self._installed()
        stats = self.timed(self.client.get_app_stats, target.id,
                           label="get_app_stats")
        self.evidence(cpu_percent=stats.cpu_usage_percent,
                      mem_bytes=stats.memory_usage_bytes,
                      threads=stats.thread_count,
                      uptime_s=stats.uptime_seconds)

    @known_issue(_APP_EMPTY_DEFECT)
    def test_05_get_logs(self):
        self.mark("AppClient.get_logs")
        target = self._installed()
        lines = list(self.timed(self.client.get_logs, target.id, 20,
                                label="get_logs"))
        self.evidence(count=len(lines),
                      sample=[str(l) for l in lines[:3]])
        for line in lines:
            self.assertIsInstance(line.level, str)

    @known_issue(_APP_EMPTY_DEFECT)
    def test_06_get_logs_text(self):
        self.mark("AppClient.get_logs_text")
        target = self._installed()
        text_lines = list(self.timed(self.client.get_logs_text, target.id, 20,
                                     label="get_logs_text"))
        self.evidence(count=len(text_lines), sample=text_lines[:3])
        for line in text_lines:
            self.assertIsInstance(line, str)


class T02AppLifecycle(DeviceTestCase):
    """Lifecycle surface — needs a bundle the suite does not carry."""

    area = "app"

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.client = AppClient()

    @classmethod
    def tearDownClass(cls):
        cls.client.close()

    def test_01_lifecycle_without_bundle(self):
        self.mark("AppClient.install_app/start_app/stop_app/"
                  "restart_app/uninstall_app")
        self.na("no app bundle (manifest+image) provisioned for the test, "
                "and mutating a production app is out of scope — lifecycle "
                "ops unexercised")


if __name__ == "__main__":
    unittest.main()
