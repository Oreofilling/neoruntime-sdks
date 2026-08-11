// config.hpp — SDK configuration from environment variables.
//
// 1:1 port of python/hailo_ipc_sdk/config.py. All accessors are static and
// read the same environment variables as the Python SDK, so a deployment can
// configure both runtimes identically. Endpoint strings keep the gRPC
// `unix:///path` (triple-slash) form that the C-core resolver accepts and
// that Python already uses — no translation between SDKs.
#pragma once

#include <string>
#include <string_view>

namespace hailo_ipc_sdk {

class Config {
public:
    Config() = delete;  // static-only utility

    static std::string get_app_id();

    // gRPC UDS endpoints (triple-slash unix:/// form).
    static std::string get_inference_endpoint();
    static std::string get_event_bus_endpoint();
    static std::string get_device_control_endpoint();
    static std::string get_camera_control_endpoint();
    static std::string get_app_manager_endpoint();

    // Filesystem paths (no unix: scheme) for the raw-UDS clients.
    static std::string get_shm_base_path();
    static std::string get_encoded_socket_dir();

    // Container→host path translation for the media/inference daemons.
    static std::string get_host_prefix();
    static std::string translate_path_to_host(std::string_view container_path);

    static bool is_debug();
    static std::string get_log_level();

private:
    // Returns env[var], or fallback when unset/empty. Single source for all
    // getenv reads; empty-string env values are treated as unset (matches the
    // Python os.getenv-with-default intent for unset vars).
    static std::string env(const char* var, std::string_view fallback);
};

}  // namespace hailo_ipc_sdk
