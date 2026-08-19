// config.cpp — implementation of Config. See config.hpp.
#include "hailo_ipc_sdk/config.hpp"

#include <cstdlib>
#include <string>
#include <string_view>

namespace hailo_ipc_sdk {

std::string Config::env(const char* var, std::string_view fallback) {
    const char* val = std::getenv(var);
    if (val == nullptr || val[0] == '\0') {
        return std::string(fallback);
    }
    return std::string(val);
}

std::string Config::get_app_id() {
    return env("APP_ID", "unknown");
}

std::string Config::get_inference_endpoint() {
    return env("AI_RUNTIME_ENDPOINT", "unix:///run/aipc/ai-runtime.sock");
}

std::string Config::get_event_bus_endpoint() {
    return env("EVENT_BUS_ENDPOINT", "unix:///run/aipc/event-bus.sock");
}

std::string Config::get_device_control_endpoint() {
    return env("DEVICE_CONTROL_ENDPOINT", "unix:///run/aipc/device-control.sock");
}

std::string Config::get_camera_control_endpoint() {
    return env("CAMERA_CONTROL_ENDPOINT", "unix:///run/aipc/camera-control.sock");
}

std::string Config::get_app_manager_endpoint() {
    return env("APP_MANAGER_ENDPOINT", "unix:///run/aipc/app-manager.sock");
}

std::string Config::get_shm_base_path() {
    return env("SHM_BASE_PATH", "/run/aipc/shm");
}

std::string Config::get_encoded_socket_dir() {
    return env("ENCODED_SOCKET_DIR", "/run/aipc/encoded");
}

std::string Config::get_host_prefix() {
    return env("AIPC_HOST_PREFIX", "/data/aipc");
}

std::string Config::translate_path_to_host(std::string_view container_path) {
    const std::string host_prefix = get_host_prefix();
    if (host_prefix.empty()) {
        return std::string(container_path);
    }
    constexpr std::string_view kOptAipc = "/opt/aipc";
    constexpr std::string_view kDataAipc = "/data/aipc";
    auto replace = [&](std::string_view prefix) {
        std::string out;
        out.reserve(host_prefix.size() + (container_path.size() - prefix.size()));
        out.append(host_prefix);
        out.append(container_path.substr(prefix.size()));
        return out;
    };
    if (container_path.rfind(kOptAipc, 0) == 0) {
        return replace(kOptAipc);
    }
    if (container_path.rfind(kDataAipc, 0) == 0) {
        return replace(kDataAipc);
    }
    return std::string(container_path);
}

bool Config::is_debug() {
    return env("DEBUG", "0") == "1";
}

std::string Config::get_log_level() {
    return env("LOG_LEVEL", "INFO");
}

}  // namespace hailo_ipc_sdk
