// app.hpp — application container management client. 1:1 port of app.py.
//
// Install/start/stop/uninstall apps, query info/stats, and stream logs over the
// app-manager gRPC service (package `appmanager`). Transport: sync gRPC stubs
// over UDS (Config::get_app_manager_endpoint()). GetAppLogs is a server-streaming
// RPC, exposed here as the pull-stream LogStream::next().
#pragma once

#include <cstdint>
#include <ctime>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace hailo_ipc_sdk {

// Application information (mirrors app.py AppInfo / proto AppInfo).
struct AppInfo {
    std::string id;
    std::string name;
    std::string version;
    std::string state;     // installed, running, stopped, failed
    std::string container_id;
    int pid = 0;
    std::int64_t installed_at = 0;  // Unix timestamp, seconds
    std::int64_t started_at = 0;
    std::int64_t stopped_at = 0;
    int restart_count = 0;
    std::string manifest_path;
    std::string instance_path;
};

// Application runtime statistics (mirrors app.py AppStats / proto AppStats).
struct AppStats {
    std::string app_id;
    double cpu_usage_percent = 0.0;
    std::int64_t memory_usage_bytes = 0;
    std::int64_t memory_limit_bytes = 0;
    int thread_count = 0;
    std::int64_t uptime_seconds = 0;
};

// One log line (mirrors app.py LogLine). timestamp is Unix nanoseconds.
struct LogLine {
    std::int64_t timestamp_ns = 0;
    std::string level;    // info, warn, error, debug
    std::string message;

    // "[YYYY-MM-DD HH:MM:SS.mmm] [LEVEL] message" — matches app.py LogLine.__str__.
    std::string to_string() const;
};

// Server-streaming log cursor over GetAppLogs. next() returns std::nullopt when
// the stream ends. Hold one at a time per AppClient (a fresh RPC per stream).
class LogStream {
public:
    LogStream();
    ~LogStream();
    LogStream(const LogStream&) = delete;
    LogStream& operator=(const LogStream&) = delete;
    LogStream(LogStream&&) noexcept;
    LogStream& operator=(LogStream&&) noexcept;

    std::optional<LogLine> next();

private:
    friend class AppClient;
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

class AppClient {
public:
    // Empty endpoint => Config::get_app_manager_endpoint().
    explicit AppClient(std::string endpoint = "");
    ~AppClient();
    AppClient(const AppClient&) = delete;
    AppClient& operator=(const AppClient&) = delete;
    AppClient(AppClient&&) noexcept;
    AppClient& operator=(AppClient&&) noexcept;

    void connect();
    void close();
    bool connected() const noexcept;

    // Register a web access path for this app (APP_ID env must be set).
    void register_web_url(const std::string& path = "/");

    // Install from manifest + image tar; returns the new app_id.
    std::string install_app(const std::string& manifest_path, const std::string& image_path);

    void start_app(const std::string& app_id);
    void stop_app(const std::string& app_id, int timeout_seconds = 30);
    void uninstall_app(const std::string& app_id, bool keep_logs = true);
    // Composite: stop then start (no dedicated RestartApp RPC).
    void restart_app(const std::string& app_id, int timeout_seconds = 30);

    std::vector<AppInfo> list_apps();
    AppInfo get_app(const std::string& app_id);
    AppStats get_app_stats(const std::string& app_id);

    // Open a GetAppLogs stream. follow=true streams indefinitely until close.
    LogStream get_logs(const std::string& app_id, int max_lines = 100, bool follow = false);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace hailo_ipc_sdk
