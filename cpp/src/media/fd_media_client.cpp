// fd_media_client.cpp — FdMediaClient + FdMediaStream.
// Port of fd_client.py: raw UDS + SCM_RIGHTS fd-passing. Each frame arrives
// as an 80-byte FdPubFrameMsg carrying up to 3 DMA-BUF fds as ancillary data.
// The copy path materializes the planes (fenced per HAL-3, shared helpers in
// detail/frame_pixels.hpp) and RELEASEs immediately; keep_fd defers the
// RELEASE to the last FrameHandle copy via a weak-entry retainer.
#include "neoruntime_ipc_sdk/fd_client.hpp"

#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <unistd.h>

#include "neoruntime_ipc_sdk/camera.hpp"
#include "neoruntime_ipc_sdk/config.hpp"

#include "detail/fd_protocol.hpp"
#include "detail/frame_pixels.hpp"
#include "detail/raw_socket.hpp"

namespace neoruntime_ipc_sdk {

using detail::FdPubFrameMsg;
using detail::FdPubMsgHeader;
using detail::FdPubReleaseMsg;
using detail::FdPubResponseMsg;
using detail::FdPubSubscribeMsg;
using detail::connect_unix;
using detail::recv_exact;
using detail::recvmsg_with_fds;
using detail::send_all;
using detail::set_recv_timeout;

namespace {

constexpr int kFdFrameSize = sizeof(FdPubFrameMsg);       // 80
constexpr int kFdRespSize = sizeof(FdPubResponseMsg);     // 12
constexpr int kFdSubSize = sizeof(FdPubSubscribeMsg);     // 76
constexpr int kFdRelSize = sizeof(FdPubReleaseMsg);       // 16
constexpr int kFdMaxAttempts = 32;
constexpr int kFdReconnectAttempts = 5;
constexpr int kFdRecvTimeoutMs = 5000;                     // subscribe_raw 5s
constexpr int kFdReconnectSleepMs = 500;                   // Python time.sleep(0.5)
// list_streams probe results cached this long (fd_client.py _LIST_STREAM_CACHE_S).
constexpr auto kListStreamsCacheS = std::chrono::seconds(5);
// fd_client.py _LIST_STREAM_TIMEOUT_S: bound the GetStreamStatus probe.
constexpr double kListStreamsTimeoutS = 1.0;

std::string format_code_to_name(std::uint32_t code) {
    switch (static_cast<PixelFormat>(code)) {
        case PixelFormat::NV12: return "NV12";
        case PixelFormat::NV21: return "NV21";
        case PixelFormat::RGB: return "RGB";
        case PixelFormat::BGR: return "BGR";
        case PixelFormat::RGBA: return "RGBA";
        case PixelFormat::BGRA: return "BGRA";
        case PixelFormat::GRAY8: return "GRAY8";
        case PixelFormat::YUYV: return "YUYV";
        default: return "UNKNOWN(" + std::to_string(code) + ")";
    }
}

// One subscribed connection. The mutex serializes RELEASE/UNSUBSCRIBE sends
// against close(), so a deferred release fired after close() sees alive=false
// instead of writing into a closed (possibly reused) fd number — the hazard
// Python avoids implicitly because closed socket objects invalidate fileno.
struct StreamSock {
    std::mutex mu;
    int fd;
    bool alive = true;

    explicit StreamSock(int f) : fd(f) {}
};

// Weak tracking of retained keep-fd handles (Python weakref.WeakSet): entries
// vanish when the last Frame copy drops, so a forgotten frame is released by
// destruction rather than held forever.
class FrameRetainer {
public:
    void add(const std::shared_ptr<FrameHandle>& h) {
        prune();
        items_.push_back(h);
    }

    std::size_t size() {
        prune();
        return items_.size();
    }

    std::vector<std::shared_ptr<FrameHandle>> live() {
        prune();
        std::vector<std::shared_ptr<FrameHandle>> out;
        out.reserve(items_.size());
        for (auto& w : items_) {
            if (auto s = w.lock()) out.push_back(std::move(s));
        }
        return out;
    }

    void discard(FrameHandle* h) {
        for (auto it = items_.begin(); it != items_.end(); ++it) {
            if (it->lock().get() == h) {
                items_.erase(it);
                return;
            }
        }
        prune();
    }

private:
    void prune() {
        items_.erase(std::remove_if(items_.begin(), items_.end(),
                                    [](const std::weak_ptr<FrameHandle>& w) {
                                        return w.expired();
                                    }),
                     items_.end());
    }
    std::vector<std::weak_ptr<FrameHandle>> items_;
};

// Connect to the FdPublisher and SUBSCRIBE to `stream_id`. Returns the fd or
// throws std::runtime_error if the server rejects the subscription.
int connect_stream_fd(const std::string& socket_path, const std::string& stream_id) {
    int fd = connect_unix(socket_path);

    FdPubSubscribeMsg sub{};
    sub.header.type = detail::FD_PUB_MSG_SUBSCRIBE;
    sub.header.size = static_cast<std::uint32_t>(kFdSubSize);
    sub.version = detail::FD_PUB_PROTOCOL_VERSION;
    auto name = stream_id.substr(0, detail::FD_PUB_MAX_STREAM_NAME - 1);
    std::memcpy(sub.stream_name, name.data(), name.size());  // rest stays '\0'

    if (!send_all(fd, &sub, kFdSubSize)) {
        ::close(fd);
        throw std::runtime_error("FdMediaClient: failed to send SUBSCRIBE for '" + stream_id + "'");
    }

    FdPubResponseMsg resp{};
    if (!recv_exact(fd, &resp, kFdRespSize)) {
        ::close(fd);
        throw std::runtime_error("FdMediaClient: no response for stream '" + stream_id + "'");
    }
    if (resp.header.type != detail::FD_PUB_MSG_OK) {
        ::close(fd);
        throw std::runtime_error("FdMediaClient: subscribe rejected for '" + stream_id +
                                 "' (code=" + std::to_string(resp.code) + ")");
    }
    return fd;
}

void release_frame_fd(int fd, std::uint64_t frame_id) {
    FdPubReleaseMsg rel{};
    rel.header.type = detail::FD_PUB_MSG_RELEASE;
    rel.header.size = static_cast<std::uint32_t>(kFdRelSize);
    rel.frame_id = frame_id;
    send_all(fd, &rel, kFdRelSize);  // best-effort; ignore failure
}

void unsubscribe_fd(int fd) {
    FdPubMsgHeader hdr{};
    hdr.type = detail::FD_PUB_MSG_UNSUBSCRIBE;
    hdr.size = static_cast<std::uint32_t>(sizeof(FdPubMsgHeader));
    send_all(fd, &hdr, sizeof(hdr));  // best-effort
}

void close_all(std::vector<int>& fds) {
    for (int f : fds) {
        if (f >= 0) ::close(f);
    }
    fds.clear();
}

}  // namespace

// ============================================================================
// FdMediaClient::Impl — per-stream cached sockets + the shared retainer.
// ============================================================================
struct FdMediaClient::Impl : std::enable_shared_from_this<FdMediaClient::Impl> {
    std::string socket_path;
    mutable std::mutex mu;  // streams_ + list_cache
    std::map<std::string, std::shared_ptr<StreamSock>> streams_;
    mutable std::mutex retained_mu;  // retained_
    FrameRetainer retained_;
    mutable std::optional<std::pair<std::chrono::steady_clock::time_point,
                                    std::vector<std::string>>> list_cache;

    explicit Impl(std::string path) : socket_path(std::move(path)) {
        if (socket_path.empty()) {
            // fd_client.py: os.getenv("CAMERA_SOCK_PATH", "/run/aipc/camera.sock").
            // This is the raw FdPublisher UDS, NOT the camera-control gRPC endpoint.
            const char* env = std::getenv("CAMERA_SOCK_PATH");
            socket_path = env && *env ? std::string(env) : "/run/aipc/camera.sock";
        }
    }
    ~Impl() { close(); }
    Impl(const Impl&) = delete;
    Impl& operator=(const Impl&) = delete;

    // Cached per-stream socket (connects lazily). Throws on subscribe
    // failure. Like Python's _get_sock, the blocking handshake runs under the
    // client lock.
    std::shared_ptr<StreamSock> get_sock(const std::string& stream_id) {
        std::lock_guard<std::mutex> lk(mu);
        auto it = streams_.find(stream_id);
        if (it != streams_.end()) {
            std::lock_guard<std::mutex> sk(it->second->mu);
            if (it->second->alive) return it->second;
            streams_.erase(it);
        }
        auto sock = std::make_shared<StreamSock>(connect_stream_fd(socket_path, stream_id));
        streams_[stream_id] = sock;
        return sock;
    }

    // Drop a stale cached socket so the next read reconnects (plain close, no
    // UNSUBSCRIBE — mirrors the Python error paths).
    void drop_sock(const std::string& stream_id) {
        std::shared_ptr<StreamSock> sock;
        {
            std::lock_guard<std::mutex> lk(mu);
            auto it = streams_.find(stream_id);
            if (it == streams_.end()) return;
            sock = std::move(it->second);
            streams_.erase(it);
        }
        std::lock_guard<std::mutex> sk(sock->mu);
        if (sock->alive) {
            sock->alive = false;
            ::close(sock->fd);
            sock->fd = -1;
        }
    }

    void register_retained(std::shared_ptr<FrameHandle> h) {
        std::lock_guard<std::mutex> lk(retained_mu);
        retained_.add(h);
        // Python warns at log-spaced thresholds (8, 16, 32, ...) that retained
        // frames may starve the daemon buffer pool; there is no logger here —
        // the count is surfaced via FdMediaClient::retained_frames().
    }

    void discard_retained(FrameHandle* h) {
        std::lock_guard<std::mutex> lk(retained_mu);
        retained_.discard(h);
    }

    // Deferred RELEASE from a FrameHandle closure: only while the socket is
    // still alive (the fd may have been closed by drop_sock/close()).
    void release_on_sock(const std::shared_ptr<StreamSock>& sock, std::uint64_t frame_id) {
        std::lock_guard<std::mutex> sk(sock->mu);
        if (sock->alive) release_frame_fd(sock->fd, frame_id);
    }

    // Python close() order: release retained frames first so the daemon
    // recycles their buffers before the subscriptions and sockets go away.
    void close() {
        std::vector<std::shared_ptr<FrameHandle>> held;
        {
            std::lock_guard<std::mutex> lk(retained_mu);
            held = retained_.live();
        }
        for (auto& h : held) h->close();  // idempotent; fires RELEASE closures

        std::vector<std::shared_ptr<StreamSock>> socks;
        {
            std::lock_guard<std::mutex> lk(mu);
            socks.reserve(streams_.size());
            for (auto& [id, s] : streams_) socks.push_back(s);
            streams_.clear();
        }
        for (auto& s : socks) {
            std::lock_guard<std::mutex> sk(s->mu);
            if (s->alive) {
                unsubscribe_fd(s->fd);
                s->alive = false;
                ::close(s->fd);
                s->fd = -1;
            }
        }
    }

    std::vector<std::string> list_streams() const {
        auto now = std::chrono::steady_clock::now();
        {
            std::lock_guard<std::mutex> lk(mu);
            if (list_cache.has_value() && now - list_cache->first < kListStreamsCacheS) {
                return list_cache->second;
            }
        }
        std::vector<std::string> ids;
        try {
            // fd_client.py probes with a 1 s deadline (_LIST_STREAM_TIMEOUT_S).
            CameraClient cam;
            cam.connect();
            for (const auto& s : cam.get_stream_status(kListStreamsTimeoutS)) {
                if (s.status == "active") ids.push_back(s.stream_id);
            }
        } catch (...) {
            ids.clear();
        }
        if (ids.empty()) ids = {"main", "sub"};  // guaranteed streams
        std::lock_guard<std::mutex> lk(mu);
        list_cache = {now, ids};
        return ids;
    }

    // Read one Frame off a connected, subscribed socket (Python _recv_frame).
    // Returns std::nullopt on timeout / no frame this cycle; throws
    // std::runtime_error on hard connection loss (3 EOFs / recv error / plane
    // mapping failure) — callers reconnect.
    std::optional<Frame> recv_frame(const std::shared_ptr<StreamSock>& sock,
                                    bool keep_fd) {
        int skipped = 0;
        int eof_count = 0;
        bool got_frame = false;
        FdPubFrameMsg msg{};
        std::vector<int> fds;

        for (int attempt = 0; attempt < kFdMaxAttempts && !got_frame; ++attempt) {
            fds.clear();
            ssize_t n = recvmsg_with_fds(sock->fd, &msg, kFdFrameSize, fds);

            if (n == 0) {  // EOF
                close_all(fds);
                if (++eof_count >= 3) {
                    throw std::runtime_error("FdMediaClient: socket EOF (server closed)");
                }
                continue;
            }
            if (n < 0) {  // SO_RCVTIMEO expiry (timeout) or hard error
                close_all(fds);
                if (errno == EAGAIN || errno == EWOULDBLOCK) return std::nullopt;
                throw std::runtime_error("FdMediaClient: socket read failed");
            }
            if (static_cast<std::size_t>(n) < kFdFrameSize) {  // short read
                close_all(fds);
                ++skipped;
                continue;
            }
            if (msg.header.type != detail::FD_PUB_MSG_FRAME) {  // stray message
                close_all(fds);
                ++skipped;
                continue;
            }
            got_frame = true;
        }
        if (!got_frame) {
            // 32 non-frame messages / transient EOFs without a frame — give up
            // this cycle (Python returns None; callers retry).
            return std::nullopt;
        }

        const std::string fmt = format_code_to_name(msg.format);
        const int width = static_cast<int>(msg.width);
        const int height = static_cast<int>(msg.height);
        const std::uint32_t flags = msg.flags;

        if (fds.empty()) {
            // Frame message without planes — release and skip (Python returns None).
            release_on_sock(sock, msg.frame_id);
            return std::nullopt;
        }

        std::array<std::uint32_t, 3> strides{msg.strides[0], msg.strides[1], msg.strides[2]};
        std::array<std::uint32_t, 3> sizes{msg.sizes[0], msg.sizes[1], msg.sizes[2]};

        if (keep_fd) {
            // Retained path: keep the dma-bufs; RELEASE is deferred to the
            // last FrameHandle copy. The closure holds this Impl strongly
            // (like Python's bound `self`) so the sockets outlive the frames.
            FrameHandle::ReleaseFn on_release = [self = shared_from_this(), sock](FrameHandle& h) {
                self->release_on_sock(sock, h.frame_id);
                self->discard_retained(&h);
            };
            auto handle = std::make_shared<FrameHandle>(
                std::move(fds), strides, sizes, msg.frame_id, std::move(on_release),
                width, height, fmt, flags);
            register_retained(handle);

            Frame frame;
            frame.sequence = msg.sequence;
            frame.timestamp_ns = msg.timestamp_ns;
            frame.width = width;
            frame.height = height;
            frame.format = fmt;
            frame.handle = std::move(handle);
            frame.flags = flags;
            return frame;
        }

        // Copy path: materialize the planes through the shared (fenced)
        // helper — throws propagate (Python lets OSError out of
        // _recv_frame). The handle owns the fds, so they are closed even on
        // the throwing path.
        auto temp = std::make_shared<FrameHandle>(std::move(fds), strides, sizes,
                                                  msg.frame_id, nullptr, width, height,
                                                  fmt, flags);
        std::vector<std::uint8_t> raw = detail::materialize_frame_handle(*temp);
        cv::Mat image = detail::decode_frame_raw(raw, width, height, fmt);
        temp.reset();  // close the fds now (Python closes them in `finally`)
        release_on_sock(sock, msg.frame_id);

        Frame frame;
        frame.sequence = msg.sequence;
        frame.timestamp_ns = msg.timestamp_ns;
        frame.width = width;
        frame.height = height;
        frame.format = fmt;
        frame.image = std::move(image);
        frame.flags = flags;
        return frame;
    }
};

// ============================================================================
// FdMediaStream — shares the client's cached socket, bounded auto-reconnect.
// ============================================================================
struct FdMediaStream::Impl {
    std::shared_ptr<FdMediaClient::Impl> owner;  // keeps the client state alive
    std::string stream_id;
    int interval = 1;   // decimation: yield every Nth frame
    bool keep_fd = false;
    bool reconnect = true;
    bool exhausted = false;
    std::uint64_t seen = 0;
    std::shared_ptr<StreamSock> sock;

    Impl(std::shared_ptr<FdMediaClient::Impl> o, std::string sid, int skip_frames,
         bool keep, bool rec)
        : owner(std::move(o)),
          stream_id(std::move(sid)),
          interval(std::max(1, skip_frames)),
          keep_fd(keep),
          reconnect(rec) {}
};

FdMediaStream::FdMediaStream() : impl_(std::make_unique<Impl>(nullptr, "", 1, false, false)) {}
FdMediaStream::~FdMediaStream() = default;
FdMediaStream::FdMediaStream(FdMediaStream&&) noexcept = default;
FdMediaStream& FdMediaStream::operator=(FdMediaStream&&) noexcept = default;

std::optional<Frame> FdMediaStream::next() {
    if (!impl_ || !impl_->owner || impl_->exhausted) return std::nullopt;

    int attempts_left = impl_->reconnect ? kFdReconnectAttempts : 1;
    while (true) {
        if (!impl_->sock) {
            try {
                impl_->sock = impl_->owner->get_sock(impl_->stream_id);
            } catch (...) {
                if (--attempts_left <= 0) {
                    impl_->exhausted = true;
                    return std::nullopt;
                }
                std::this_thread::sleep_for(
                    std::chrono::milliseconds(kFdReconnectSleepMs));
                continue;
            }
        }

        set_recv_timeout(impl_->sock->fd, kFdRecvTimeoutMs);
        try {
            auto frame = impl_->owner->recv_frame(impl_->sock, impl_->keep_fd);
            if (frame.has_value()) {
                ++impl_->seen;
                if (impl_->interval > 1 &&
                    ((impl_->seen - 1) % impl_->interval) != 0) {
                    // Decimated away: return the buffer immediately so its
                    // dma-buf is not retained (Python frame.release()).
                    frame->release();
                    continue;
                }
                return frame;
            }
            continue;  // timeout — keep the socket (Python subscribe_raw)
        } catch (...) {
            // Connection lost: drop the cached socket, reconnect if budgeted.
            impl_->owner->drop_sock(impl_->stream_id);
            impl_->sock.reset();
            if (--attempts_left <= 0) {
                impl_->exhausted = true;
                return std::nullopt;
            }
            std::this_thread::sleep_for(
                std::chrono::milliseconds(kFdReconnectSleepMs));
        }
    }
}

// ============================================================================
// FdMediaClient
// ============================================================================
FdMediaClient::FdMediaClient(std::string socket_path)
    : impl_(std::make_shared<Impl>(std::move(socket_path))) {}
FdMediaClient::~FdMediaClient() = default;
FdMediaClient::FdMediaClient(FdMediaClient&&) noexcept = default;
FdMediaClient& FdMediaClient::operator=(FdMediaClient&&) noexcept = default;

void FdMediaClient::close() { impl_->close(); }

std::optional<Frame> FdMediaClient::get_frame(const std::string& stream_id,
                                              int timeout_ms, bool keep_fd) {
    std::shared_ptr<StreamSock> sock = impl_->get_sock(stream_id);  // throws (parity)
    set_recv_timeout(sock->fd, timeout_ms > 0 ? timeout_ms : 5000);
    try {
        return impl_->recv_frame(sock, keep_fd);
    } catch (...) {
        // Stale socket — clear the cache so the next call reconnects, then
        // re-raise (Python get_frame re-raises ConnectionError).
        impl_->drop_sock(stream_id);
        throw;
    }
}

FdMediaStream FdMediaClient::subscribe(const std::string& stream_id, int skip_frames,
                                       bool keep_fd, bool reconnect) {
    FdMediaStream s;
    s.impl_ = std::make_unique<FdMediaStream::Impl>(impl_, stream_id, skip_frames,
                                                    keep_fd, reconnect);
    return s;
}

std::thread FdMediaClient::on_frame(const std::string& stream_id,
                                    std::function<void(const Frame&)> cb,
                                    int skip_frames, bool keep_fd) {
    auto owner = impl_;
    return std::thread([owner, stream_id, skip_frames, keep_fd,
                        cb = std::move(cb)]() {
        // Python's generator reconnects forever; each FdMediaStream carries a
        // bounded budget, so take a fresh one whenever the budget is spent.
        while (true) {
            FdMediaStream stream;
            stream.impl_ = std::make_unique<FdMediaStream::Impl>(
                owner, stream_id, skip_frames, keep_fd, true);
            while (auto frame = stream.next()) {
                try {
                    cb(*frame);
                } catch (...) {
                    // Callback errors must not kill the pump (Python logs and
                    // continues).
                }
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(kFdReconnectSleepMs));
        }
    });
}

int FdMediaClient::retained_frames() const {
    std::lock_guard<std::mutex> lk(impl_->retained_mu);
    return static_cast<int>(impl_->retained_.size());
}

EncodedStreamClient FdMediaClient::get_encoded_stream(const std::string& stream_id,
                                                      std::string socket_dir) {
    return EncodedStreamClient("", stream_id, std::move(socket_dir));
}

std::vector<std::string> FdMediaClient::list_streams() const {
    return impl_->list_streams();
}

std::string FdMediaClient::get_rtsp_url(const std::string& stream_id,
                                        const std::string& host, int port) const {
    return "rtsp://" + host + ":" + std::to_string(port) + "/" + stream_id;
}

}  // namespace neoruntime_ipc_sdk
