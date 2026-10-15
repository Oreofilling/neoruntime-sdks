// diagnostics.hpp — one-shot environment diagnostics for SDK apps
// ("why is my app slow?") (port of diagnostics.py, 0.8.1).
//
// Collects, without ever raising: dependency availability (OpenCV is the
// performance-critical one — a hard link dependency in C++, reported for
// parity), platform socket reachability, and the accel router's view of
// which backend is actually serving each operation.
//
//   DiagnosticsReport report = diagnostics();
//   if (!report.optional_deps.at("cv2").available) { ... }
//   if (report.accel_router.ops.at("nv12_to_rgb").backend == "software") {
//       ...  // DSP unreachable or refused — check camera-daemon
//   }
//
// check() grades a snapshot against named requirements and either hands
// back actionable problem strings or raises naming every failure:
//
//   check({"inference", "camera_sock"}).raise_if_unhealthy();
#pragma once

#include <map>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "neoruntime_ipc_sdk/accel.hpp"
#include "neoruntime_ipc_sdk/camera.hpp"
#include "neoruntime_ipc_sdk/inference.hpp"

namespace neoruntime_ipc_sdk {

// Existence + connectability of a Unix domain socket.
struct SocketProbe {
    std::string path;
    bool exists = false;
    bool connectable = false;
};

// python's optional-deps rows ({"available": bool, "version": str}).
struct DepInfo {
    bool available = false;
    std::string version;
};

// The five daemon sockets (python services{}).
struct ServiceProbes {
    SocketProbe ai_runtime;
    SocketProbe event_bus;
    SocketProbe device_control;
    SocketProbe camera_control;
    SocketProbe app_manager;
};

// python media{}: camera.sock fd publisher + encoded-stream directory.
struct MediaDiagnostics {
    SocketProbe camera_sock;
    std::string encoded_dir;
    std::vector<std::string> streams;  // sorted stream names, ".sock" stripped
};

// python's "python" section analog: what compiled this binary.
struct ToolchainInfo {
    std::string standard;        // e.g. "C++17"
    std::string implementation;  // e.g. "gcc 12.3.0"
};

// Snapshot of the SDK's execution environment.
struct DiagnosticsReport {
    std::string sdk_version;
    ToolchainInfo toolchain;
    std::map<std::string, DepInfo> optional_deps;  // "cv2" key kept for parity
    ServiceProbes services;
    MediaDiagnostics media;
    RouterHealth accel_router;
    std::map<std::string, bool> accel_probes;
};

// A diagnostics() snapshot plus a health verdict: checks maps each
// available check name to a bool, healthy is true when no required check
// failed, and problems lists actionable messages for the failures.
struct CheckReport {
    DiagnosticsReport report;
    std::map<std::string, bool> checks;
    bool healthy = true;
    std::vector<std::string> problems;

    // Live-probe extras (python's report["model_lookup_error"] etc.).
    std::optional<std::string> model_lookup_error;
    std::optional<std::string> stream_lookup_error;
    std::optional<std::string> stream_status;

    // Throw RuntimeError naming every failed requirement, if any.
    void raise_if_unhealthy() const;
};

// Strip the "unix://" scheme from an endpoint (identity for plain paths).
std::string uds_path(const std::string& endpoint);

// Snapshot the SDK's execution environment. Never throws.
DiagnosticsReport diagnostics();

// Pre-flight an app: run diagnostics() and grade it. `required` names
// checks from: inference, events, device, camera, app_manager,
// camera_sock, cv2, dsp — plus model_registered / stream_active when the
// corresponding live client + id are passed. Each failed requirement
// lands in problems with an actionable hint. Throws std::invalid_argument
// for an unknown check name. The live probes are opt-in so the base
// check stays offline-safe.
CheckReport check(const std::vector<std::string>& required = {},
                  InferenceClient* inference = nullptr,
                  const std::string& model_id = "", CameraClient* camera = nullptr,
                  const std::string& stream_id = "");

}  // namespace neoruntime_ipc_sdk
