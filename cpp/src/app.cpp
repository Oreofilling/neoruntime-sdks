// app.cpp — AppClient + LogStream. See app.hpp.
//
// Port of app.py over the `appmanager` gRPC service. Two Status shapes to mind:
// InstallApp returns InstallResponse{ Status status; string app_id; ... } while
// Start/Stop/Uninstall return Status directly. GetAppLogs is server-streaming.
#include "neoruntime_ipc_sdk/app.hpp"

#include <grpcpp/grpcpp.h>
#include <google/protobuf/empty.pb.h>

#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "neoruntime_ipc_sdk/config.hpp"

#include "detail/grpc_channel.hpp"
#include "app-manager/app.grpc.pb.h"
#include "app-manager/app.pb.h"

namespace neoruntime_ipc_sdk {

namespace pb = appmanager;

namespace {

AppInfo parse_app_info(const pb::AppInfo& a) {
    AppInfo info;
    info.id = a.id();
    info.name = a.name();
    info.version = a.version();
    info.state = a.state();
    info.container_id = a.container_id();
    info.pid = a.pid();
    info.installed_at = a.installed_at();
    info.started_at = a.started_at();
    info.stopped_at = a.stopped_at();
    info.restart_count = a.restart_count();
    info.manifest_path = a.manifest_path();
    info.instance_path = a.instance_path();
    return info;
}

AppStats parse_app_stats(const pb::AppStats& s) {
    AppStats stats;
    stats.app_id = s.app_id();
    stats.cpu_usage_percent = s.cpu_usage_percent();
    stats.memory_usage_bytes = s.memory_usage_bytes();
    stats.memory_limit_bytes = s.memory_limit_bytes();
    stats.thread_count = s.thread_count();
    stats.uptime_seconds = s.uptime_seconds();
    return stats;
}

LogLine parse_log_line(const pb::LogLine& l) {
    return LogLine{l.timestamp(), l.level(), l.message()};
}

}  // namespace

// ---- LogLine::to_string ----------------------------------------------------
std::string LogLine::to_string() const {
    constexpr std::int64_t kNsPerS = 1'000'000'000LL;
    std::time_t sec = static_cast<std::time_t>(timestamp_ns / kNsPerS);
    long ms = static_cast<long>((timestamp_ns % kNsPerS) / 1'000'000L);
    std::tm tmv{};
    ::localtime_r(&sec, &tmv);
    char ts[24];
    std::strftime(ts, sizeof(ts), "%Y-%m-%d %H:%M:%S", &tmv);

    std::string lvl = level;
    for (auto& c : lvl) c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
    if (lvl.size() < 5) lvl.append(5 - lvl.size(), ' ');

    char head[64];
    std::snprintf(head, sizeof(head), "[%s.%03ld] [%s] ", ts, ms, lvl.c_str());
    return std::string(head) + message;
}

// ============================================================================
// LogStream — wraps the GetAppLogs ClientReader.
// ============================================================================
struct LogStream::Impl {
    grpc::ClientContext ctx;
    std::unique_ptr<grpc::ClientReader<pb::LogLine>> reader;
};

LogStream::LogStream() : impl_(std::make_unique<Impl>()) {}
LogStream::~LogStream() = default;
LogStream::LogStream(LogStream&&) noexcept = default;
LogStream& LogStream::operator=(LogStream&&) noexcept = default;

std::optional<LogLine> LogStream::next() {
    if (!impl_ || !impl_->reader) return std::nullopt;
    pb::LogLine line;
    if (!impl_->reader->Read(&line)) return std::nullopt;  // stream ended
    return parse_log_line(line);
}

// ============================================================================
// AppClient
// ============================================================================
struct AppClient::Impl {
    std::string endpoint;
    std::shared_ptr<grpc::Channel> channel;
    std::unique_ptr<pb::AppManager::Stub> stub;

    void ensure_connected() {
        if (!stub) {
            channel = detail::make_channel(endpoint);
            stub = pb::AppManager::NewStub(channel);
        }
    }
};

AppClient::AppClient(std::string endpoint) : impl_(std::make_unique<Impl>()) {
    impl_->endpoint = endpoint.empty() ? Config::get_app_manager_endpoint()
                                       : std::move(endpoint);
}

AppClient::~AppClient() = default;
AppClient::AppClient(AppClient&&) noexcept = default;
AppClient& AppClient::operator=(AppClient&&) noexcept = default;

void AppClient::connect() { impl_->ensure_connected(); }

void AppClient::close() {
    impl_->stub.reset();
    impl_->channel.reset();
}

bool AppClient::connected() const noexcept { return impl_->stub != nullptr; }

void AppClient::register_web_url(const std::string& path) {
    impl_->ensure_connected();
    const char* app_id_env = std::getenv("APP_ID");
    if (!app_id_env || !*app_id_env) {
        throw std::runtime_error(
            "AppClient::register_web_url(): APP_ID env var not set "
            "— this method must be called inside an app container");
    }
    pb::RegisterWebUrlRequest req;
    req.set_app_id(app_id_env);
    req.set_path(path);
    pb::RegisterWebUrlResponse resp;
    grpc::ClientContext ctx;
    detail::check_grpc(impl_->stub->RegisterWebUrl(&ctx, req, &resp), "RegisterWebUrl");
    detail::require_success(resp.success(), resp.message(), "RegisterWebUrl");
}

std::string AppClient::install_app(const std::string& manifest_path,
                                   const std::string& image_path) {
    impl_->ensure_connected();
    pb::InstallRequest req;
    req.set_manifest_path(manifest_path);
    req.set_image_path(image_path);
    pb::InstallResponse resp;
    grpc::ClientContext ctx;
    detail::check_grpc(impl_->stub->InstallApp(&ctx, req, &resp), "InstallApp");
    detail::require_success(resp.status().success(), resp.status().message(), "InstallApp");
    return resp.app_id();
}

void AppClient::start_app(const std::string& app_id) {
    impl_->ensure_connected();
    pb::StartRequest req;
    req.set_app_id(app_id);
    pb::Status resp;
    grpc::ClientContext ctx;
    detail::check_grpc(impl_->stub->StartApp(&ctx, req, &resp), "StartApp");
    detail::require_success(resp.success(), resp.message(), "StartApp");
}

void AppClient::stop_app(const std::string& app_id, int timeout_seconds) {
    impl_->ensure_connected();
    pb::StopRequest req;
    req.set_app_id(app_id);
    req.set_timeout_seconds(timeout_seconds);
    pb::Status resp;
    grpc::ClientContext ctx;
    detail::check_grpc(impl_->stub->StopApp(&ctx, req, &resp), "StopApp");
    detail::require_success(resp.success(), resp.message(), "StopApp");
}

void AppClient::uninstall_app(const std::string& app_id, bool keep_logs) {
    impl_->ensure_connected();
    pb::UninstallRequest req;
    req.set_app_id(app_id);
    req.set_keep_logs(keep_logs);
    pb::Status resp;
    grpc::ClientContext ctx;
    detail::check_grpc(impl_->stub->UninstallApp(&ctx, req, &resp), "UninstallApp");
    detail::require_success(resp.success(), resp.message(), "UninstallApp");
}

void AppClient::restart_app(const std::string& app_id, int timeout_seconds) {
    // Composite: no dedicated RestartApp RPC (matches app.py).
    stop_app(app_id, timeout_seconds);
    start_app(app_id);
}

std::vector<AppInfo> AppClient::list_apps() {
    impl_->ensure_connected();
    google::protobuf::Empty req;
    pb::AppList resp;
    grpc::ClientContext ctx;
    detail::check_grpc(impl_->stub->ListApps(&ctx, req, &resp), "ListApps");
    std::vector<AppInfo> out;
    out.reserve(resp.apps_size());
    for (const auto& a : resp.apps()) out.push_back(parse_app_info(a));
    return out;
}

AppInfo AppClient::get_app(const std::string& app_id) {
    impl_->ensure_connected();
    pb::GetAppRequest req;
    req.set_app_id(app_id);
    pb::AppInfo resp;
    grpc::ClientContext ctx;
    detail::check_grpc(impl_->stub->GetApp(&ctx, req, &resp), "GetApp");
    return parse_app_info(resp);
}

AppStats AppClient::get_app_stats(const std::string& app_id) {
    impl_->ensure_connected();
    pb::GetAppRequest req;
    req.set_app_id(app_id);
    pb::AppStats resp;
    grpc::ClientContext ctx;
    detail::check_grpc(impl_->stub->GetAppStats(&ctx, req, &resp), "GetAppStats");
    return parse_app_stats(resp);
}

LogStream AppClient::get_logs(const std::string& app_id, int max_lines, bool follow) {
    impl_->ensure_connected();
    pb::GetLogsRequest req;
    req.set_app_id(app_id);
    req.set_max_lines(max_lines);
    req.set_follow(follow);
    LogStream stream;
    stream.impl_->reader = impl_->stub->GetAppLogs(&stream.impl_->ctx, req);
    return stream;
}

}  // namespace neoruntime_ipc_sdk
