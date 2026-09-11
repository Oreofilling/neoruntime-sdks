"""Tests for neoruntime_ipc_sdk.accel — hardware/software routing."""

from __future__ import annotations

import numpy as np
import pytest

from neoruntime_ipc_sdk.accel import (
    AccelRouter,
    DegradationRecord,
    HardwareUnavailable,
    RoutePolicy,
    get_default_router,
)


def _sw_upper(x):
    return x.upper()


def _hw_upper(x):
    return x.upper() + "!"


def _hw_broken(x):
    raise HardwareUnavailable("service down")


class TestRoute:
    def test_prefer_hardware_when_registered(self):
        r = AccelRouter()
        r.register("greet", software=_sw_upper, hardware=_hw_upper)
        d = r.route("greet")
        assert (d.op, d.backend) == ("greet", "hardware")
        assert d.provider == "_hw_upper"

    def test_software_only_policy_forces_software(self):
        r = AccelRouter(policy=RoutePolicy.SOFTWARE_ONLY)
        r.register("greet", software=_sw_upper, hardware=_hw_upper)
        d = r.route("greet")
        assert d.backend == "software"
        assert "software-only" in d.reason

    def test_missing_hardware_leg_reports_note(self):
        r = AccelRouter()
        r.register("greet", software=_sw_upper, note="pending platform exposure")
        d = r.route("greet")
        assert d.backend == "software"
        assert d.reason == "pending platform exposure"

    def test_no_providers_is_unavailable(self):
        r = AccelRouter()
        r.register("greet")
        assert r.route("greet").backend == "unavailable"

    def test_software_only_without_software_leg_is_unavailable(self):
        r = AccelRouter(policy=RoutePolicy.SOFTWARE_ONLY)
        r.register("bare", hardware=_hw_upper)
        assert r.route("bare").backend == "unavailable"

    def test_unregistered_op_raises(self):
        r = AccelRouter()
        with pytest.raises(KeyError, match="not registered"):
            r.route("nope")


class TestRun:
    def test_hardware_success(self):
        r = AccelRouter()
        r.register("greet", software=_sw_upper, hardware=_hw_upper)
        assert r.run("greet", "hi") == "HI!"
        assert r.health()["ops"]["greet"]["hardware_calls"] == 1

    def test_degrades_to_software(self):
        r = AccelRouter()
        seen = []
        r.register("greet", software=_sw_upper, hardware=_hw_broken)
        r.on_degradation = seen.append
        assert r.run("greet", "hi") == "HI"
        h = r.health()["ops"]["greet"]
        assert h["fallbacks"] == 1
        assert h["software_calls"] == 1
        assert h["hardware_calls"] == 0
        assert len(seen) == 1
        assert isinstance(seen[0], DegradationRecord)
        assert "service down" in seen[0].reason
        assert r.health()["recent_degradations"][0]["op"] == "greet"

    def test_hardware_only_raises_instead_of_degrading(self):
        r = AccelRouter(policy=RoutePolicy.HARDWARE_ONLY)
        r.register("greet", software=_sw_upper, hardware=_hw_broken)
        with pytest.raises(HardwareUnavailable, match="service down"):
            r.run("greet", "hi")
        assert r.health()["ops"]["greet"]["fallbacks"] == 0

    def test_no_software_fallback_raises(self):
        r = AccelRouter()
        r.register("greet", hardware=_hw_broken)
        with pytest.raises(HardwareUnavailable, match="no software fallback"):
            r.run("greet", "hi")

    def test_unavailable_op_raises_on_run(self):
        r = AccelRouter()
        r.register("greet")
        with pytest.raises(HardwareUnavailable):
            r.run("greet", "hi")

    def test_on_degradation_failure_is_swallowed(self):
        r = AccelRouter()

        def boom(record):
            raise RuntimeError("health sink down")

        r.register("greet", software=_sw_upper, hardware=_hw_broken)
        r.on_degradation = boom
        assert r.run("greet", "hi") == "HI"

    def test_software_only_policy_executes_software(self):
        r = AccelRouter(policy=RoutePolicy.SOFTWARE_ONLY)
        r.register("greet", software=_sw_upper, hardware=_hw_upper)
        assert r.run("greet", "hi") == "HI"


class TestUseHardware:
    def test_attach_and_replace(self):
        r = AccelRouter()
        r.register("greet", software=_sw_upper)
        assert r.route("greet").backend == "software"
        r.use_hardware("greet", _hw_upper)
        assert r.route("greet").backend == "hardware"

    def test_unknown_op_raises(self):
        r = AccelRouter()
        with pytest.raises(KeyError, match="not registered"):
            r.use_hardware("unknown", _hw_upper)


class TestProbes:
    def test_probe_results(self):
        r = AccelRouter()
        r.add_probe("ok", lambda: True)
        r.add_probe("bad", lambda: 1 / 0)
        assert r.probe() == {"ok": True, "bad": False}


class TestHealth:
    def test_snapshot_shape(self):
        r = AccelRouter()
        r.register("greet", software=_sw_upper, hardware=_hw_upper)
        r.run("greet", "x")
        h = r.health()
        assert h["policy"] == "prefer_hardware"
        assert set(h["ops"]["greet"]) == {
            "hardware_calls",
            "software_calls",
            "fallbacks",
            "backend",
        }
        assert h["ops"]["greet"]["backend"] == "hardware"
        assert h["recent_degradations"] == []


class TestDefaultRouter:
    def test_singleton_identity(self):
        assert get_default_router() is get_default_router()

    def test_expected_ops_registered(self):
        ops = get_default_router().health()["ops"]
        for op in ("resize_nv12", "rgb_to_nv12", "nms"):
            assert op in ops

    def test_resize_nv12_routes_hardware_when_leg_present(self):
        # pure route() decision — no socket is touched
        d = get_default_router().route("resize_nv12")
        assert d.backend == "hardware"

    def test_nms_runs_software(self):
        r = get_default_router()
        assert r.run("nms", np.zeros((0, 4)), np.zeros((0,))) == []
        assert r.health()["ops"]["nms"]["backend"] == "software"


class TestSetRoutePolicy:
    def test_string_and_enum_set_the_default_policy(self):
        from neoruntime_ipc_sdk import set_route_policy as top_level
        from neoruntime_ipc_sdk.accel import set_route_policy

        router = get_default_router()
        original = router._policy
        try:
            set_route_policy("software_only")
            assert router._policy is RoutePolicy.SOFTWARE_ONLY
            top_level(RoutePolicy.HARDWARE_ONLY)  # exported at package level too
            assert router._policy is RoutePolicy.HARDWARE_ONLY
        finally:
            set_route_policy(original)  # don't leak policy into other tests

    def test_policy_change_is_visible_in_health(self):
        from neoruntime_ipc_sdk.accel import set_route_policy

        router = get_default_router()
        original = router._policy
        try:
            set_route_policy(RoutePolicy.SOFTWARE_ONLY)
            assert router.health()["policy"] == "software_only"
        finally:
            set_route_policy(original)
