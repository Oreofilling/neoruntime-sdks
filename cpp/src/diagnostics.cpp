// diagnostics.cpp — port of diagnostics.py (0.8.1): UDS reachability
// probes, dependency/toolchain report, accel-router health pass-through,
// and the named pre-flight checks with actionable problem hints.
#include "neoruntime_ipc_sdk/diagnostics.hpp"

#include <poll.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <filesystem>
#include <map>
#include <system_error>
#include <utility>

#include <opencv2/core.hpp>  // CV_VERSION

#include "neoruntime_ipc_sdk/config.hpp"
#include "neoruntime_ipc_sdk/version.hpp"

namespace neoruntime_ipc_sdk {

namespace {

constexpr int kSocketTimeoutMs = 1000;  // python _SOCKET_TIMEOUT = 1.0s


// Existence + connectability of a Unix domain socket; never throws (any
// failure leaves connectable = false).
SocketProbe probe_uds(const std::string& path) {
    SocketProbe out;
    out.path = path;
    std::error_code ec;
    out.exists = std::filesystem::exists(path, ec);
    if (!out.exists) return out;

    const int fd = ::socket(AF_UNIX, SOCK_STREAM | SOCK_NONBLOCK, 0);
    if (fd < 0) return out;
    sockaddr_un addr{};
    addr.sun_family = AF_UNIX;
    if (path.size() >= sizeof(addr.sun_path)) {
        ::close(fd);
        return out;
    }
    std::copy(path.begin(), path.end(), addr.sun_path);
    addr.sun_path[path.size()] = '\0';
    if (::connect(fd, reinterpret_cast<const sockaddr*>(&addr), sizeof(addr)) == 0) {
        out.connectable = true;
    } else if (errno == EINPROGRESS) {  // backlog-full UDS: poll it out
        pollfd pfd{};
        pfd.fd = fd;
        pfd.events = POLLOUT;
        if (::poll(&pfd, 1, kSocketTimeoutMs) > 0) {
            int so_error = 0;
            socklen_t len = sizeof(so_error);
            if (::getsockopt(fd, SOL_SOCKET, SO_ERROR, &so_error, &len) == 0 &&
                so_error == 0)
                out.connectable = true;
        }
    }
    ::close(fd);
    return out;
}

ToolchainInfo toolchain_info() {
    ToolchainInfo out;
#if defined(_MSC_VER)
    out.implementation = "msvc " + std::to_string(_MSC_VER);
#elif defined(__clang__)
    out.implementation = "clang " + std::to_string(__clang_major__) + "." +
                         std::to_string(__clang_minor__);
#elif defined(__GNUC__)
    out.implementation = "gcc " + std::to_string(__GNUC__) + "." +
                         std::to_string(__GNUC_MINOR__) + "." +
                         std::to_string(__GNUC_PATCHLEVEL__);
#else
    out.implementation = "unknown";
#endif
#if __cplusplus >= 202002L
    out.standard = "C++20";
#elif __cplusplus >= 201703L
    out.standard = "C++17";
#else
    out.standard = "C++";
#endif
    return out;
}

std::vector<std::string> list_encoded_streams(const std::string& dir) {
    std::vector<std::string> streams;
    try {
        std::error_code ec;
        if (!std::filesystem::is_directory(dir, ec)) return streams;
        const std::string suffix = ".sock";
        for (const auto& entry : std::filesystem::directory_iterator(dir)) {
            std::string name = entry.path().filename().string();
            if (name.size() > suffix.size() &&
                name.compare(name.size() - suffix.size(), suffix.size(), suffix) == 0)
                streams.push_back(name.substr(0, name.size() - suffix.size()));
        }
        std::sort(streams.begin(), streams.end());
    } catch (...) {  // permissions etc. — diagnostics never raises
    }
    return streams;
}

// python _PROBLEM_HINTS, verbatim.
const std::map<std::string, std::string>& problem_hints() {
    static const std::map<std::string, std::string> hints = {
        {"inference", "ai-runtime socket not reachable — is the inference daemon running?"},
        {"events", "event-bus socket not reachable — is the event bus running?"},
        {"device", "device-control socket not reachable — is the device daemon running?"},
        {"camera", "camera-control socket not reachable — is camera-daemon running?"},
        {"app_manager", "app-manager socket not reachable — app management unavailable"},
        {"camera_sock", "camera.sock fd publisher not reachable — video frames unavailable"},
        {"cv2", "cv2 not installed — conversions/resize/JPEG run 10-50x slower (pure numpy)"},
        {"dsp", "DSP service not reachable — pixel operations fall back to CPU"},
    };
    return hints;
}

}  // namespace

std::string uds_path(const std::string& endpoint) {
    const std::string prefix = "unix://";
    if (endpoint.rfind(prefix, 0) == 0) return endpoint.substr(prefix.size());
    return endpoint;
}

DiagnosticsReport diagnostics() {
    DiagnosticsReport out;
    out.sdk_version = kVersion;
    out.toolchain = toolchain_info();
    DepInfo cv;
    cv.available = true;  // hard link dependency in C++ (python probes the import)
    cv.version = CV_VERSION;
    out.optional_deps.emplace("cv2", std::move(cv));

    out.services.ai_runtime = probe_uds(uds_path(Config::get_inference_endpoint()));
    out.services.event_bus = probe_uds(uds_path(Config::get_event_bus_endpoint()));
    out.services.device_control = probe_uds(uds_path(Config::get_device_control_endpoint()));
    out.services.camera_control = probe_uds(uds_path(Config::get_camera_control_endpoint()));
    out.services.app_manager = probe_uds(uds_path(Config::get_app_manager_endpoint()));

    const char* camera_env = std::getenv("CAMERA_SOCK_PATH");
    out.media.camera_sock = probe_uds(camera_env != nullptr ? camera_env
                                                            : "/run/aipc/camera.sock");
    out.media.encoded_dir = Config::get_encoded_socket_dir();
    out.media.streams = list_encoded_streams(out.media.encoded_dir);

    try {
        AccelRouter& router = get_default_router();
        out.accel_router = router.health();
        out.accel_probes = router.probe();
    } catch (...) {  // never raises — rows stay empty
    }
    return out;
}

CheckReport check(const std::vector<std::string>& required, InferenceClient* inference,
                  const std::string& model_id, CameraClient* camera,
                  const std::string& stream_id) {
    CheckReport out;
    out.report = diagnostics();
    const DiagnosticsReport& r = out.report;

    out.checks["inference"] = r.services.ai_runtime.connectable;
    out.checks["events"] = r.services.event_bus.connectable;
    out.checks["device"] = r.services.device_control.connectable;
    out.checks["camera"] = r.services.camera_control.connectable;
    out.checks["app_manager"] = r.services.app_manager.connectable;
    out.checks["camera_sock"] = r.media.camera_sock.connectable;
    const auto cv2 = r.optional_deps.find("cv2");
    out.checks["cv2"] = cv2 != r.optional_deps.end() && cv2->second.available;
    const auto dsp = r.accel_probes.find("dsp");
    out.checks["dsp"] = dsp != r.accel_probes.end() && dsp->second;

    if (inference != nullptr && !model_id.empty()) {
        try {
            out.checks["model_registered"] =
                inference->get_model_info(model_id).has_value();
        } catch (const std::exception& exc) {  // service down → not registered
            out.checks["model_registered"] = false;
            out.model_lookup_error = exc.what();
        } catch (...) {
            out.checks["model_registered"] = false;
            out.model_lookup_error = "unknown error";
        }
    }
    if (camera != nullptr && !stream_id.empty()) {
        try {
            std::map<std::string, std::string> statuses;
            for (const StreamStatus& s : camera->get_stream_status())
                statuses[s.stream_id] = s.status;
            const auto it = statuses.find(stream_id);
            out.checks["stream_active"] =
                it != statuses.end() && it->second == "active";
            out.stream_status = it != statuses.end() ? it->second : std::string("absent");
        } catch (const std::exception& exc) {
            out.checks["stream_active"] = false;
            out.stream_lookup_error = exc.what();
        } catch (...) {
            out.checks["stream_active"] = false;
            out.stream_lookup_error = "unknown error";
        }
    }

    for (const std::string& key : required) {
        const auto it = out.checks.find(key);
        if (it == out.checks.end()) {
            std::vector<std::string> names;
            for (const auto& entry : out.checks) names.push_back(entry.first);
            std::sort(names.begin(), names.end());
            std::string joined;
            for (std::size_t i = 0; i < names.size(); ++i) {
                if (i > 0) joined += ", ";
                joined += names[i];
            }
            throw std::invalid_argument("unknown check '" + key +
                                        "' (available: " + joined + ")");
        }
        if (it->second) continue;
        if (key == "model_registered") {
            out.problems.push_back("model '" + model_id +
                                   "' is not registered with the inference service");
        } else if (key == "stream_active") {
            out.problems.push_back("stream '" + stream_id + "' is not active (status: " +
                                   out.stream_status.value_or("unknown") + ")");
        } else {
            const auto& hints = problem_hints();
            const auto hint = hints.find(key);
            out.problems.push_back(hint != hints.end() ? hint->second
                                                       : key + ": check failed");
        }
    }
    out.healthy = out.problems.empty();
    return out;
}

void CheckReport::raise_if_unhealthy() const {
    if (problems.empty()) return;
    std::string text = "SDK health check failed:";
    for (const std::string& problem : problems) text += "\n- " + problem;
    throw std::runtime_error(text);
}

}  // namespace neoruntime_ipc_sdk
