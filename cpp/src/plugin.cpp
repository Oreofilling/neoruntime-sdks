// plugin.cpp — PluginDiscovery + PluginEndpoint + PluginServer. See plugin.hpp.
//
// Port of plugin.py. Discovery is pure JSON (no gRPC): /run/aipc/plugins/
// discovery.json is re-parsed on demand. The only gRPC touchpoints are
// PluginEndpoint::connect() (builds a Channel to a plugin's UDS) and
// PluginServer::build_and_start() (binds a Server on the plugin's UDS).
#include "neoruntime_ipc_sdk/plugin.hpp"

#include <grpcpp/grpcpp.h>
#include <grpcpp/server_builder.h>
#include <nlohmann/json.hpp>

#include <atomic>
#include <chrono>
#include <ctime>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_set>
#include <utility>
#include <vector>

#include <sys/stat.h>
#include <unistd.h>

#include "detail/endpoint.hpp"
#include "detail/grpc_channel.hpp"

namespace neoruntime_ipc_sdk {

namespace {

constexpr int kPluginWatchIntervalMs = 2000;
constexpr int kPluginRequirePollMs = 1000;

// Open `path`, parse it as JSON, return the document or an empty object on any
// read/parse error (plugin.py keeps the previous doc on failure; callers here
// treat an empty doc the same as "no plugins yet").
nlohmann::json load_discovery(const std::string& path) {
    std::ifstream in(path);
    if (!in) return nlohmann::json::object();
    try {
        nlohmann::json doc;
        in >> doc;
        return doc;
    } catch (...) {
        return nlohmann::json::object();
    }
}

// Build a PluginEndpoint from one capability entry, matching plugin.py's mapping.
PluginEndpoint make_endpoint(const std::string& app_id, const std::string& state,
                             const nlohmann::json& cap) {
    PluginEndpoint ep;
    ep.app_id = app_id;
    ep.capability_id = cap.value("id", std::string{});
    ep.version = cap.value("version", std::string{"1.0"});
    ep.transport = cap.value("transport", std::string{"grpc"});
    const auto& grpc = cap.value("grpc", nlohmann::json::object());
    ep.socket_path = grpc.value("socket_path", std::string{});
    ep.grpc_service = grpc.value("service", std::string{});
    const auto& ev = cap.value("event", nlohmann::json::object());
    if (ev.contains("publish") && ev["publish"].is_array()) {
        for (const auto& t : ev["publish"]) {
            if (t.is_string()) ep.event_publish.push_back(t.get<std::string>());
        }
    }
    if (ev.contains("subscribe") && ev["subscribe"].is_array()) {
        for (const auto& t : ev["subscribe"]) {
            if (t.is_string()) ep.event_subscribe.push_back(t.get<std::string>());
        }
    }
    ep.state = state;
    return ep;
}

}  // namespace

// ============================================================================
// PluginEndpoint
// ============================================================================
std::shared_ptr<grpc::Channel> PluginEndpoint::connect() const {
    if (socket_path.empty()) {
        throw std::runtime_error("PluginEndpoint::connect(): no socket_path for '" +
                                 capability_id + "'");
    }
    return detail::make_channel(socket_path);
}

// ============================================================================
// PluginDiscovery
// ============================================================================
struct PluginDiscovery::Impl {
    std::string discovery_dir;
    std::string discovery_path;
    nlohmann::json doc = nlohmann::json::object();
    mutable std::mutex mtx;
    std::vector<std::function<void()>> callbacks;
    std::atomic<bool> running{false};
    std::thread watcher;
    std::time_t last_mtime = 0;

    explicit Impl(std::string dir) : discovery_dir(std::move(dir)) {
        discovery_path = discovery_dir;
        if (!discovery_path.empty() && discovery_path.back() != '/') discovery_path.push_back('/');
        discovery_path += kPluginDiscoveryFile;
        reload_locked();
    }

    ~Impl() { stop(); }

    void reload_locked() {
        doc = load_discovery(discovery_path);
        struct stat st;
        last_mtime = (::stat(discovery_path.c_str(), &st) == 0) ? st.st_mtime : 0;
    }

    std::optional<PluginEndpoint> get_locked(const std::string& capability_id) const {
        const auto& plugins = doc.value("plugins", nlohmann::json::object());
        if (!plugins.is_object()) return std::nullopt;
        for (auto it = plugins.begin(); it != plugins.end(); ++it) {
            const auto& info = it.value();
            if (!info.is_object()) continue;
            std::string app_id = info.value("app_id", it.key());
            std::string state = info.value("state", std::string{"unknown"});
            const auto& caps = info.value("capabilities", nlohmann::json::array());
            if (!caps.is_array()) continue;
            for (const auto& cap : caps) {
                if (cap.is_object() && cap.value("id", std::string{}) == capability_id) {
                    return make_endpoint(app_id, state, cap);
                }
            }
        }
        return std::nullopt;
    }

    void ensure_watcher() {
        if (running.exchange(true)) return;  // already running
        watcher = std::thread([this]() {
            while (running.load()) {
                std::this_thread::sleep_for(std::chrono::milliseconds(kPluginWatchIntervalMs));
                if (!running.load()) break;
                struct stat st;
                if (::stat(discovery_path.c_str(), &st) != 0) continue;
                if (st.st_mtime == last_mtime) continue;
                {
                    std::lock_guard<std::mutex> lk(mtx);
                    reload_locked();
                }
                std::vector<std::function<void()>> cbs;
                {
                    std::lock_guard<std::mutex> lk(mtx);
                    cbs = callbacks;
                }
                for (const auto& cb : cbs) {
                    try { cb(); } catch (...) {}
                }
            }
        });
    }

    void stop() {
        if (!running.exchange(false)) return;
        if (watcher.joinable()) watcher.join();
    }
};

PluginDiscovery::PluginDiscovery(std::string discovery_dir)
    : impl_(std::make_unique<Impl>(std::move(discovery_dir))) {}

PluginDiscovery::~PluginDiscovery() = default;
PluginDiscovery::PluginDiscovery(PluginDiscovery&&) noexcept = default;
PluginDiscovery& PluginDiscovery::operator=(PluginDiscovery&&) noexcept = default;

void PluginDiscovery::reload() {
    std::lock_guard<std::mutex> lk(impl_->mtx);
    impl_->reload_locked();
}

std::optional<PluginEndpoint> PluginDiscovery::get(const std::string& capability_id) const {
    std::lock_guard<std::mutex> lk(impl_->mtx);
    return impl_->get_locked(capability_id);
}

PluginEndpoint PluginDiscovery::require(const std::string& capability_id, double timeout_seconds) {
    auto deadline = std::chrono::steady_clock::now() +
                    std::chrono::milliseconds(static_cast<int>(timeout_seconds * 1000));
    while (true) {
        {
            std::lock_guard<std::mutex> lk(impl_->mtx);
            impl_->reload_locked();
        }
        auto ep = get(capability_id);
        if (ep && ep->is_available()) return *ep;
        if (std::chrono::steady_clock::now() >= deadline) {
            throw std::runtime_error("PluginDiscovery::require(): timed out waiting for '" +
                                     capability_id + "'");
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(kPluginRequirePollMs));
    }
}

nlohmann::json PluginDiscovery::list_plugins() const {
    std::lock_guard<std::mutex> lk(impl_->mtx);
    return impl_->doc.value("plugins", nlohmann::json::object());
}

std::vector<std::string> PluginDiscovery::list_capabilities() const {
    std::lock_guard<std::mutex> lk(impl_->mtx);
    std::vector<std::string> out;
    std::unordered_set<std::string> seen;
    const auto& plugins = impl_->doc.value("plugins", nlohmann::json::object());
    if (!plugins.is_object()) return out;
    for (auto it = plugins.begin(); it != plugins.end(); ++it) {
        const auto& info = it.value();
        if (!info.is_object()) continue;
        const auto& caps = info.value("capabilities", nlohmann::json::array());
        if (!caps.is_array()) continue;
        for (const auto& cap : caps) {
            if (!cap.is_object()) continue;
            std::string id = cap.value("id", std::string{});
            if (!id.empty() && seen.insert(id).second) out.push_back(id);
        }
    }
    return out;
}

void PluginDiscovery::watch(std::function<void()> callback) {
    std::lock_guard<std::mutex> lk(impl_->mtx);
    impl_->callbacks.push_back(std::move(callback));
    impl_->ensure_watcher();
}

void PluginDiscovery::close() { impl_->stop(); }

// ============================================================================
// PluginServer
// ============================================================================
PluginServer::PluginServer(std::string plugin_id, std::string socket_dir)
    : plugin_id_(std::move(plugin_id)) {
    socket_path_ = socket_dir;
    if (!socket_path_.empty() && socket_path_.back() != '/') socket_path_.push_back('/');
    socket_path_ += plugin_id_ + ".sock";
}

const std::string& PluginServer::socket_path() const noexcept { return socket_path_; }

std::unique_ptr<grpc::Server> PluginServer::build_and_start(grpc::Service* service) {
    grpc::ServerBuilder builder;
    builder.RegisterService(service);
    builder.AddListeningPort(detail::grpc_endpoint(socket_path_),
                             grpc::InsecureServerCredentials());
    std::unique_ptr<grpc::Server> server(builder.BuildAndStart());
    if (!server) {
        throw std::runtime_error("PluginServer: failed to build server on '" + socket_path_ + "'");
    }
    return server;
}

void PluginServer::cleanup(const std::string& socket_path) {
    // Best-effort: ignore missing file. Matches plugin.py's os.unlink try/except.
    ::unlink(socket_path.c_str());
}

}  // namespace neoruntime_ipc_sdk
