// accel.hpp — hardware-first capability router (port of accel.py, 0.8.1).
//
// The SDK ships loose components, not a pipeline; most have two
// implementations — a hardware leg executed by a daemon (DSP convert /
// resize / blend / JPEG) and an OpenCV software leg that always works.
// This router makes the hardware/software choice explicit and uniform:
// one policy per app, one route table, and degradation tracking surfaced
// through health().
//
//   AccelRouter& router = get_default_router();
//   cv::Mat small = std::any_cast<cv::Mat>(
//       router.run("resize_nv12", {nv12, std::make_pair(1920, 1080),
//                                  std::make_pair(640, 384)}));
//   if (router.health().ops["resize_nv12"].backend == "software") { ... }
//
// C++ note on the DSP resident: legs and SDK fast paths run DSP jobs
// through shared_dsp_call() — the process-resident client machinery in
// src/detail/dsp_dispatch.hpp (retire-on-failure, deferred drain). Tests
// substitute fakes via detail::install_dsp_dispatch().
#pragma once

#include <any>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace neoruntime_ipc_sdk {

// A hardware provider could not run (service down, op not exposed).
class HardwareUnavailable : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

// How the router picks a leg for each operation. PreferHardware uses the
// hardware leg when registered, falling back to software with a recorded
// degradation; SoftwareOnly never touches hardware (benchmarks,
// reproducible tests); HardwareOnly raises instead of degrading — when the
// zero-CPU guarantee matters more than uptime.
enum class RoutePolicy {
    PreferHardware,
    SoftwareOnly,
    HardwareOnly,
};

// RoutePolicy <-> value name ("prefer_hardware" / "software_only" /
// "hardware_only"); from_string throws std::invalid_argument on anything
// else (python RoutePolicy(value)).
std::string route_policy_name(RoutePolicy policy);
RoutePolicy route_policy_from_string(const std::string& name);

// Why a given backend serves an operation right now.
struct RouteDecision {
    std::string op;
    std::string backend;  // "hardware" | "software" | "unavailable"
    std::string provider;
    std::string reason;
};

// One hardware->software fallback, kept for health reporting.
struct DegradationRecord {
    std::string op;
    std::string reason;
    double timestamp = 0.0;  // unix seconds, python time.time() parity
};

// Per-op row of RouterHealth (python's dict: counters + backend).
struct OpHealth {
    long hardware_calls = 0;
    long software_calls = 0;
    long fallbacks = 0;
    std::string backend;
};

struct RouterHealth {
    std::string policy;  // value name
    std::map<std::string, OpHealth> ops;
    std::vector<DegradationRecord> recent_degradations;  // newest last, cap 64
};

// Route-table dispatcher between hardware and software providers.
// Operations are registered with register_op(); run() executes through
// the leg chosen by the policy, recording any fallback.
// get_default_router() returns a singleton pre-registered with the
// operations the SDK serves today.
class AccelRouter {
public:
    // Type-erased leg: takes the op's positional args, returns its result.
    // `name` replaces python's fn.__name__ (route decisions and health
    // report it; tests assert provider == "_hw_upper").
    using Fn = std::function<std::any(std::vector<std::any>)>;

    struct Leg {
        std::string name;
        Fn fn;
        // python `if leg is None` / `if leg:` — an empty-named Leg is "not
        // registered"; any other Leg counts as attached.
        explicit operator bool() const noexcept { return !name.empty(); }
    };

    explicit AccelRouter(RoutePolicy policy = RoutePolicy::PreferHardware);

    // Switch the routing policy at runtime.
    void set_policy(RoutePolicy policy);
    RoutePolicy policy() const noexcept;

    // Map an operation name to its software/hardware providers. Either leg
    // may be unset: a missing hardware leg means the capability awaits
    // platform exposure (state it in `note`), a missing software leg means
    // there is no fallback. (`register` is a C++ keyword.)
    void register_op(const std::string& op, Leg software = {}, Leg hardware = {},
                     std::string note = "");

    // Attach (or replace) the hardware leg of an operation.
    // Throws std::out_of_range (python KeyError) for an unregistered op.
    void use_hardware(const std::string& op, Leg hardware);

    // Register a cheap availability probe (e.g. service reachable).
    void add_probe(const std::string& name, std::function<bool()> probe);

    // Decide which backend serves `op` under the current policy.
    // Throws std::out_of_range for an unregistered op.
    RouteDecision route(const std::string& op) const;

    // Execute `op` through the leg chosen by route().
    std::any run(const std::string& op, std::vector<std::any> args);

    // Record a fallback that happened outside run() — Frame::resize keeps
    // its own direct DSP fast path; when that attempt fails under
    // PreferHardware it reports here so health() stays honest. Unregistered
    // ops still count (counter-only health rows).
    void note_degradation(const std::string& op, const std::string& reason);

    // Run all registered probes -> name -> available (never throws).
    std::map<std::string, bool> probe() const;

    // Snapshot of routing state, per-op counters and recent fallbacks.
    RouterHealth health() const;

    // Optional app hook fired on every degradation; exceptions from it are
    // swallowed (a health-sink failure must not break the app).
    std::function<void(const DegradationRecord&)> on_degradation;

private:
    struct OpEntry {
        Leg software;
        Leg hardware;
        std::string note;
    };
    struct Counters {
        long hardware_calls = 0;
        long software_calls = 0;
        long fallbacks = 0;
    };

    // Copy a route under the lock (use_hardware may swap concurrently);
    // the copy is what run() executes. Throws for an unregistered op.
    OpEntry snapshot(const std::string& op) const;
    RouteDecision decide(const std::string& op, const OpEntry& entry) const;
    void record_degradation(const std::string& op, const std::string& reason);
    Counters& counters_for(const std::string& op);

    mutable std::mutex mutex_;
    RoutePolicy policy_ = RoutePolicy::PreferHardware;
    std::map<std::string, OpEntry> routes_;
    std::map<std::string, std::function<bool()>> probes_;
    std::map<std::string, Counters> counters_;
    std::vector<DegradationRecord> degradations_;  // ring, newest last
    static constexpr std::size_t kMaxDegradations = 64;
};

// The pre-registered router singleton (resize_nv12 / rgb_to_nv12 /
// nv12_to_rgb / encode_jpeg / nms / draw_detections + cv2 & dsp probes).
AccelRouter& get_default_router();

// App-wide hardware/software routing policy on the default router — one
// obvious knob instead of reaching for the singleton.
void set_route_policy(RoutePolicy policy);
inline void set_route_policy(const std::string& name) {
    set_route_policy(route_policy_from_string(name));
}

// Run one DspClient method on the process-resident client. The same
// resident machinery the router legs use, exposed for SDK fast paths that
// live outside the route table (Frame::resize's DSP arm). On any error the
// resident is retired — the next call reconnects — and the original
// exception propagates so the caller keeps its own fallback policy.
// Args convention (see accel.cpp's DefaultDspDispatch):
//   resize_hw:      {Mat|Frame, dst_w, dst_h, RouterDspOpts}
//   convert_hw:     {Mat|Frame, dst_fmt, RouterDspOpts}
//   encode_jpeg_hw: {Mat|Frame, quality, fmt ("" = frame-inferred)}
//   blend_hw:       {Mat base, std::vector<DspOverlay>, fmt}
//   ensure_sock:    {}  (reachability probe)
std::any shared_dsp_call(const std::string& method, std::vector<std::any> args);

// Type-erased job options carried inside shared_dsp_call arg vectors
// (python's kwargs). fmt "" = infer from the source; out_ref asks
// resize_hw to land in a daemon buffer and return a DspBufferRef.
struct RouterDspOpts {
    std::string fmt;
    std::string scaling = "stretch";
    bool cpu_fallback = true;
    bool out_ref = false;
};

}  // namespace neoruntime_ipc_sdk
