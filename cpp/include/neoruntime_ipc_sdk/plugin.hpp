// plugin.hpp — plugin discovery + server helpers. 1:1 port of plugin.py.
//
// NOTE: unlike the other clients, this header deliberately exposes gRPC types
// (grpc::Channel / grpc::Server). plugin.py is itself a gRPC plumbing helper:
// PluginEndpoint.connect() returns a grpc.Channel and PluginServer wraps a
// grpc.Server, so the C++ port mirrors that rather than hiding grpc behind
// PImpl.
//
// Discovery is NOT gRPC — it reads /run/aipc/plugins/discovery.json (nlohmann/json).
#pragma once

#include <grpcpp/grpcpp.h>
#include <grpcpp/server_builder.h>
#include <nlohmann/json.hpp>

#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <vector>

namespace neoruntime_ipc_sdk {

inline constexpr const char* kPluginDiscoveryDir = "/run/aipc/plugins";
inline constexpr const char* kPluginDiscoveryFile = "discovery.json";

// One resolved plugin capability, parsed out of discovery.json.
struct PluginEndpoint {
    std::string app_id;
    std::string capability_id;
    std::string version;
    std::string transport;
    std::string socket_path;   // empty if the capability has no gRPC endpoint
    std::string grpc_service;
    std::vector<std::string> event_publish;
    std::vector<std::string> event_subscribe;
    std::string state = "unknown";

    bool is_available() const noexcept { return state == "running"; }
    // Build a gRPC channel to this plugin's UDS. Throws if no socket_path.
    std::shared_ptr<grpc::Channel> connect() const;
};

// Client-side discovery: parse discovery.json and find capabilities.
class PluginDiscovery {
public:
    explicit PluginDiscovery(std::string discovery_dir = kPluginDiscoveryDir);
    ~PluginDiscovery();
    PluginDiscovery(const PluginDiscovery&) = delete;
    PluginDiscovery& operator=(const PluginDiscovery&) = delete;
    PluginDiscovery(PluginDiscovery&&) noexcept;
    PluginDiscovery& operator=(PluginDiscovery&&) noexcept;

    // Re-read discovery.json from disk.
    void reload();

    // First plugin providing `capability_id` (regardless of state).
    std::optional<PluginEndpoint> get(const std::string& capability_id) const;

    // Block until a running plugin provides `capability_id`. Throws
    // std::runtime_error on timeout (mirrors Python's TimeoutError).
    PluginEndpoint require(const std::string& capability_id, double timeout_seconds = 30.0);

    // Raw discovery document: { "plugins": { app_id: { ... } } }.
    nlohmann::json list_plugins() const;

    // All known capability IDs (de-duplicated, insertion-ordered).
    std::vector<std::string> list_capabilities() const;

    // Invoke `callback` whenever discovery.json changes (polls mtime every 2s,
    // like plugin.py). Managed internally — stopped by close()/destruction.
    void watch(std::function<void()> callback);

    void close();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// Server-side helper: bind a gRPC service on the standard plugin socket path.
class PluginServer {
public:
    PluginServer(std::string plugin_id, std::string socket_dir = kPluginDiscoveryDir);

    // Standard socket: {socket_dir}/{plugin_id}.sock
    const std::string& socket_path() const noexcept;

    // Convenience: register `service`, bind the UDS, BuildAndStart(). Returns
    // the running server (caller owns it). For multi-service servers, build the
    // grpc::ServerBuilder yourself using socket_path().
    std::unique_ptr<grpc::Server> build_and_start(grpc::Service* service);

    // Remove a leftover socket file (best-effort; ignores a missing file).
    static void cleanup(const std::string& socket_path);

private:
    std::string plugin_id_;
    std::string socket_path_;
};

}  // namespace neoruntime_ipc_sdk
