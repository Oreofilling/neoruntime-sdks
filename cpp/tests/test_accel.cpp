// test_accel.cpp — hardware/software routing parity (port of python
// tests/test_accel.py). The route/run/health/probe surface maps 1:1; the
// python suite's TestSharedResident monkeypatches accel._lazy_dsp_client —
// the C++ seam is detail::install_dsp_dispatch() with a FakeDispatch.
#include <gtest/gtest.h>

#include <opencv2/core.hpp>

#include <condition_variable>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "neoruntime_ipc_sdk/accel.hpp"
#include "neoruntime_ipc_sdk/dsp.hpp"

#include "detail/dsp_dispatch.hpp"

using namespace neoruntime_ipc_sdk;

namespace {

// python's _sw_upper / _hw_upper / _hw_broken.
AccelRouter::Fn sw_upper = [](std::vector<std::any> args) -> std::any {
    std::string s = std::any_cast<std::string>(args.at(0));
    for (char& c : s) c = static_cast<char>(::toupper(static_cast<unsigned char>(c)));
    return s;
};
AccelRouter::Fn hw_upper = [](std::vector<std::any> args) -> std::any {
    std::string s = std::any_cast<std::string>(args.at(0));
    for (char& c : s) c = static_cast<char>(::toupper(static_cast<unsigned char>(c)));
    return s + "!";
};
AccelRouter::Fn hw_broken = [](std::vector<std::any>) -> std::any {
    throw HardwareUnavailable("service down");
};

AccelRouter::Leg leg(const char* name, AccelRouter::Fn fn) {
    return AccelRouter::Leg{name, std::move(fn)};
}

std::string upper_result(AccelRouter& r) {
    return std::any_cast<std::string>(r.run("greet", {std::string("hi")}));
}

// ---- TestSharedResident support ---------------------------------------------------

// threading.Event pair: the fake marks itself entered, then parks until
// the test releases it (python's block=("blend_hw", entered, release)).
struct Gate {
    void mark_entered() {
        {
            std::lock_guard<std::mutex> lock(mutex);
            entered = true;
        }
        cv.notify_all();
    }
    bool wait_entered() {
        std::unique_lock<std::mutex> lock(mutex);
        return cv.wait_for(lock, std::chrono::seconds(2), [&] { return entered; });
    }
    void signal_release() {
        {
            std::lock_guard<std::mutex> lock(mutex);
            released = true;
        }
        cv.notify_all();
    }
    void wait_release() {
        std::unique_lock<std::mutex> lock(mutex);
        cv.wait(lock, [&] { return released; });
    }
    std::mutex mutex;
    std::condition_variable cv;
    bool entered = false;
    bool released = false;
};

// python's TestSharedResident._Fake — fails named methods with DspError,
// can park one method mid-flight, and models pool refs via live flags.
// live_pools is only mutated by the test thread while no call is in
// flight (same discipline as the python list swap).
class FakeDispatch final : public detail::DspDispatch {
public:
    bool has_live_pools() override {
        for (bool alive : live_pools)
            if (alive) return true;
        return false;
    }
    void close() override { closed = true; }
    std::any call(const std::string& method, std::vector<std::any>) override {
        if (gate != nullptr && method == block_method) {
            gate->mark_entered();
            gate->wait_release();
        }
        if (fail.count(method) != 0) throw DspError(method + " failed");
        return std::make_pair(std::string("ok"), method);
    }

    std::atomic<bool> closed{false};
    std::vector<bool> live_pools;
    std::set<std::string> fail;
    std::string block_method;
    Gate* gate = nullptr;
};

// Restore the default factory when the test ends (python's monkeypatch
// undo) — the fake is stack/state owned by the test, never deleted.
struct DispatchReset {
    ~DispatchReset() { detail::install_dsp_dispatch(nullptr); }
};

}  // namespace

// ---- TestRoute ----------------------------------------------------------------------

TEST(Route, PreferHardwareWhenRegistered) {
    AccelRouter r;
    r.register_op("greet", leg("_sw_upper", sw_upper), leg("_hw_upper", hw_upper));
    const RouteDecision d = r.route("greet");
    EXPECT_EQ(d.op, "greet");
    EXPECT_EQ(d.backend, "hardware");
    EXPECT_EQ(d.provider, "_hw_upper");
}

TEST(Route, SoftwareOnlyPolicyForcesSoftware) {
    AccelRouter r(RoutePolicy::SoftwareOnly);
    r.register_op("greet", leg("_sw_upper", sw_upper), leg("_hw_upper", hw_upper));
    const RouteDecision d = r.route("greet");
    EXPECT_EQ(d.backend, "software");
    EXPECT_NE(d.reason.find("software-only"), std::string::npos);
}

TEST(Route, MissingHardwareLegReportsNote) {
    AccelRouter r;
    r.register_op("greet", leg("_sw_upper", sw_upper), {}, "pending platform exposure");
    const RouteDecision d = r.route("greet");
    EXPECT_EQ(d.backend, "software");
    EXPECT_EQ(d.reason, "pending platform exposure");
}

TEST(Route, NoProvidersIsUnavailable) {
    AccelRouter r;
    r.register_op("greet");
    EXPECT_EQ(r.route("greet").backend, "unavailable");
}

TEST(Route, SoftwareOnlyWithoutSoftwareLegIsUnavailable) {
    AccelRouter r(RoutePolicy::SoftwareOnly);
    r.register_op("bare", {}, leg("_hw_upper", hw_upper));
    EXPECT_EQ(r.route("bare").backend, "unavailable");
}

TEST(Route, UnregisteredOpRaises) {
    AccelRouter r;
    try {
        r.route("nope");
        FAIL() << "expected std::out_of_range";
    } catch (const std::out_of_range& exc) {
        EXPECT_NE(std::string(exc.what()).find("not registered"), std::string::npos);
    }
}

// ---- TestRun --------------------------------------------------------------------------

TEST(Run, HardwareSuccess) {
    AccelRouter r;
    r.register_op("greet", leg("_sw_upper", sw_upper), leg("_hw_upper", hw_upper));
    EXPECT_EQ(upper_result(r), "HI!");
    EXPECT_EQ(r.health().ops.at("greet").hardware_calls, 1);
}

TEST(Run, DegradesToSoftware) {
    AccelRouter r;
    std::vector<DegradationRecord> seen;
    r.register_op("greet", leg("_sw_upper", sw_upper), leg("_hw_broken", hw_broken));
    r.on_degradation = [&](const DegradationRecord& record) { seen.push_back(record); };
    EXPECT_EQ(upper_result(r), "HI");
    const OpHealth h = r.health().ops.at("greet");
    EXPECT_EQ(h.fallbacks, 1);
    EXPECT_EQ(h.software_calls, 1);
    EXPECT_EQ(h.hardware_calls, 0);
    ASSERT_EQ(seen.size(), 1u);
    EXPECT_NE(std::string(seen[0].reason).find("service down"), std::string::npos);
    const RouterHealth health = r.health();
    ASSERT_FALSE(health.recent_degradations.empty());
    EXPECT_EQ(health.recent_degradations.back().op, "greet");
}

TEST(Run, HardwareOnlyRaisesInsteadOfDegrading) {
    AccelRouter r(RoutePolicy::HardwareOnly);
    r.register_op("greet", leg("_sw_upper", sw_upper), leg("_hw_broken", hw_broken));
    try {
        upper_result(r);
        FAIL() << "expected HardwareUnavailable";
    } catch (const HardwareUnavailable& exc) {
        EXPECT_NE(std::string(exc.what()).find("service down"), std::string::npos);
    }
    EXPECT_EQ(r.health().ops.at("greet").fallbacks, 0);
}

TEST(Run, NoSoftwareFallbackRaises) {
    AccelRouter r;
    r.register_op("greet", {}, leg("_hw_broken", hw_broken));
    try {
        upper_result(r);
        FAIL() << "expected HardwareUnavailable";
    } catch (const HardwareUnavailable& exc) {
        EXPECT_NE(std::string(exc.what()).find("no software fallback"), std::string::npos);
    }
}

TEST(Run, UnavailableOpRaisesOnRun) {
    AccelRouter r;
    r.register_op("greet");
    EXPECT_THROW(upper_result(r), HardwareUnavailable);
}

TEST(Run, OnDegradationFailureIsSwallowed) {
    AccelRouter r;
    r.register_op("greet", leg("_sw_upper", sw_upper), leg("_hw_broken", hw_broken));
    r.on_degradation = [](const DegradationRecord&) {
        throw std::runtime_error("health sink down");
    };
    EXPECT_EQ(upper_result(r), "HI");
}

TEST(Run, SoftwareOnlyPolicyExecutesSoftware) {
    AccelRouter r(RoutePolicy::SoftwareOnly);
    r.register_op("greet", leg("_sw_upper", sw_upper), leg("_hw_upper", hw_upper));
    EXPECT_EQ(upper_result(r), "HI");
}

// ---- TestUseHardware -------------------------------------------------------------------

TEST(UseHardware, AttachAndReplace) {
    AccelRouter r;
    r.register_op("greet", leg("_sw_upper", sw_upper));
    EXPECT_EQ(r.route("greet").backend, "software");
    r.use_hardware("greet", leg("_hw_upper", hw_upper));
    EXPECT_EQ(r.route("greet").backend, "hardware");
}

TEST(UseHardware, UnknownOpRaises) {
    AccelRouter r;
    try {
        r.use_hardware("unknown", leg("_hw_upper", hw_upper));
        FAIL() << "expected std::out_of_range";
    } catch (const std::out_of_range& exc) {
        EXPECT_NE(std::string(exc.what()).find("not registered"), std::string::npos);
    }
}

// ---- TestProbes --------------------------------------------------------------------------

TEST(Probes, ProbeResults) {
    AccelRouter r;
    r.add_probe("ok", [] { return true; });
    r.add_probe("bad", []() -> bool { throw std::runtime_error("1/0"); });
    const std::map<std::string, bool> results = r.probe();
    EXPECT_EQ(results.size(), 2u);
    EXPECT_TRUE(results.at("ok"));
    EXPECT_FALSE(results.at("bad"));
}

// ---- TestHealth ---------------------------------------------------------------------------

TEST(Health, SnapshotShape) {
    AccelRouter r;
    r.register_op("greet", leg("_sw_upper", sw_upper), leg("_hw_upper", hw_upper));
    r.run("greet", {std::string("x")});
    const RouterHealth h = r.health();
    EXPECT_EQ(h.policy, "prefer_hardware");
    ASSERT_NE(h.ops.find("greet"), h.ops.end());
    const OpHealth& row = h.ops.at("greet");
    EXPECT_EQ(row.hardware_calls, 1);
    EXPECT_EQ(row.software_calls, 0);
    EXPECT_EQ(row.fallbacks, 0);
    EXPECT_EQ(row.backend, "hardware");
    EXPECT_TRUE(h.recent_degradations.empty());
}

// ---- TestDefaultRouter ----------------------------------------------------------------------

TEST(DefaultRouter, SingletonIdentity) {
    EXPECT_EQ(&get_default_router(), &get_default_router());
}

TEST(DefaultRouter, ExpectedOpsRegistered) {
    const auto ops = get_default_router().health().ops;
    for (const char* op : {"resize_nv12", "rgb_to_nv12", "nms"}) {
        EXPECT_NE(ops.find(op), ops.end()) << op;
    }
}

TEST(DefaultRouter, ResizeNv12RoutesHardwareWhenLegPresent) {
    // pure route() decision — no socket is touched
    EXPECT_EQ(get_default_router().route("resize_nv12").backend, "hardware");
}

TEST(DefaultRouter, NmsRunsSoftware) {
    AccelRouter& r = get_default_router();
    const std::vector<int> keep = std::any_cast<std::vector<int>>(
        r.run("nms", {std::vector<std::array<float, 4>>{}, std::vector<float>{}}));
    EXPECT_TRUE(keep.empty());
    EXPECT_EQ(r.health().ops.at("nms").backend, "software");
}

// ---- TestSetRoutePolicy ------------------------------------------------------------------------

TEST(SetRoutePolicy, StringAndEnumSetTheDefaultPolicy) {
    AccelRouter& router = get_default_router();
    const RoutePolicy original = router.policy();
    try {
        set_route_policy("software_only");
        EXPECT_EQ(router.policy(), RoutePolicy::SoftwareOnly);
        set_route_policy(RoutePolicy::HardwareOnly);
        EXPECT_EQ(router.policy(), RoutePolicy::HardwareOnly);
    } catch (...) {
        set_route_policy(original);
        throw;
    }
    set_route_policy(original);  // don't leak policy into other tests
}

TEST(SetRoutePolicy, PolicyChangeIsVisibleInHealth) {
    AccelRouter& router = get_default_router();
    const RoutePolicy original = router.policy();
    try {
        set_route_policy(RoutePolicy::SoftwareOnly);
        EXPECT_EQ(router.health().policy, "software_only");
    } catch (...) {
        set_route_policy(original);
        throw;
    }
    set_route_policy(original);
}

// ---- TestSharedResident -------------------------------------------------------------------------

TEST(SharedResident, FailureRetiresAndDrainsToClose) {
    DispatchReset reset;
    FakeDispatch fake;
    fake.fail = {"resize_hw"};
    detail::install_dsp_dispatch(&fake);
    try {
        shared_dsp_call("resize_hw", {});
        FAIL() << "expected DspError";
    } catch (const DspError& exc) {
        EXPECT_NE(std::string(exc.what()).find("resize_hw failed"), std::string::npos);
    }
    EXPECT_TRUE(fake.closed);                       // nothing in flight, no pools — closed at drain
    EXPECT_EQ(detail::current_resident(), nullptr);  // ...and the slot is free again
}

TEST(SharedResident, RetirementWaitsForAConcurrentCall) {
    DispatchReset reset;
    Gate gate;
    FakeDispatch fake;
    fake.fail = {"resize_hw"};
    fake.block_method = "blend_hw";
    fake.gate = &gate;
    detail::install_dsp_dispatch(&fake);

    std::pair<std::string, std::string> done;
    std::thread worker([&] {
        done = std::any_cast<std::pair<std::string, std::string>>(
            shared_dsp_call("blend_hw", {}));
    });
    ASSERT_TRUE(gate.wait_entered());  // the other call is mid-flight on the resident
    EXPECT_THROW(shared_dsp_call("resize_hw", {}), DspError);  // a sibling fails
    EXPECT_FALSE(fake.closed);                       // retirement defers the close under it
    EXPECT_EQ(detail::current_resident(), nullptr);  // ...but freed the slot at once
    gate.signal_release();
    worker.join();
    EXPECT_EQ(done.first, "ok");  // mid-flight call finished fine
    EXPECT_EQ(done.second, "blend_hw");
    EXPECT_TRUE(fake.closed);  // and the drain closed once it ended
}

TEST(SharedResident, RetirementWaitsForLivePools) {
    DispatchReset reset;
    FakeDispatch fake;
    fake.fail = {"resize_hw"};
    fake.live_pools = {true};  // a ref from before the failure
    detail::install_dsp_dispatch(&fake);
    EXPECT_THROW(shared_dsp_call("resize_hw", {}), DspError);
    EXPECT_FALSE(fake.closed);  // the ref's pool still anchors the client
    fake.live_pools = {};       // holder released it
    // any call's end re-drains. python's fixture reuses the fake here; the
    // C++ seam lazily builds the real dispatch instead, which fails fast
    // offline — either way the leave-hook drains and the retired fake
    // closes, which is the assertion. (Args must be well-formed: the real
    // dispatch any_casts them before it ever reaches the socket.)
    try {
        shared_dsp_call("blend_hw", {cv::Mat(6, 4, CV_8UC1, cv::Scalar(0)),
                                     std::vector<DspOverlay>{}, std::string("nv12")});
    } catch (...) {  // no daemon on the test host — expected
    }
    EXPECT_TRUE(fake.closed);
}
