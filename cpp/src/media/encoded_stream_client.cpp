// encoded_stream_client.cpp — EncodedStreamClient + EncodedStream.
// Port of media.py EncodedStreamClient. Raw UDS, 30-byte little-endian header +
// length-prefixed NALU payload (H.264/H.265).
#include "hailo_ipc_sdk/media.hpp"

#include <chrono>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>

#include "detail/fd_protocol.hpp"
#include "detail/raw_socket.hpp"

namespace hailo_ipc_sdk {

using detail::EncHeader;
using detail::connect_unix;
using detail::recv_exact;
using detail::send_all;
using detail::set_recv_timeout;

namespace {
constexpr int kEncHeaderSize = sizeof(EncHeader);          // 30
constexpr std::size_t kEncMaxPayload = 50u * 1024u * 1024u;  // 50 MB sanity cap
constexpr int kEncReconnectAttempts = 3;                     // bounded reconnect
constexpr int kEncDefaultTimeoutMs = 5000;

// Read one complete encoded frame from a connected socket. Returns false on
// EOF/timeout/short-read (caller decides whether to reconnect).
bool read_one_frame(int fd, EncodedFrame& out) {
    EncHeader hdr{};
    if (!recv_exact(fd, &hdr, kEncHeaderSize)) return false;

    // total_size counts header + payload; payload may be zero (keepalive).
    std::uint32_t total = hdr.total_size;
    if (total < static_cast<std::uint32_t>(kEncHeaderSize)) return false;
    std::size_t payload = total - static_cast<std::size_t>(kEncHeaderSize);
    if (payload > kEncMaxPayload) return false;

    out.codec = hdr.codec;
    out.flags = hdr.flags;
    out.pts_ns = hdr.pts_ns;
    out.width = static_cast<int>(hdr.width);
    out.height = static_cast<int>(hdr.height);
    out.dts_ns = hdr.dts_ns;

    out.data.resize(payload);
    if (payload > 0 && !recv_exact(fd, out.data.data(), payload)) return false;
    return true;
}
}  // namespace

// ============================================================================
// EncodedStream::Impl — owns its own connection, bounded auto-reconnect.
// ============================================================================
struct EncodedStream::Impl {
    std::string socket_path;
    bool reconnect_enabled = true;
    int fd = -1;
    bool exhausted = false;  // reconnections spent; next() will keep returning nullopt

    explicit Impl(std::string path, bool reconnect)
        : socket_path(std::move(path)), reconnect_enabled(reconnect) {}

    ~Impl() { close_fd(); }

    void close_fd() {
        if (fd >= 0) {
            ::close(fd);
            fd = -1;
        }
    }

    bool ensure_connected() {
        if (fd >= 0) return true;
        if (!reconnect_enabled) {
            // First connect only; no retry budget.
            try {
                fd = connect_unix(socket_path);
            } catch (...) {
                return false;
            }
            return fd >= 0;
        }
        for (int i = 0; i < kEncReconnectAttempts; ++i) {
            try {
                fd = connect_unix(socket_path);
                if (fd >= 0) return true;
            } catch (...) {
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(100 * (i + 1)));
        }
        return false;
    }
};

EncodedStream::EncodedStream() : impl_(std::make_unique<Impl>("", false)) {}
EncodedStream::~EncodedStream() = default;
EncodedStream::EncodedStream(EncodedStream&&) noexcept = default;
EncodedStream& EncodedStream::operator=(EncodedStream&&) noexcept = default;

std::optional<EncodedFrame> EncodedStream::next() {
    if (!impl_ || impl_->exhausted) return std::nullopt;

    EncodedFrame frame;
    // Keep (re)connecting and reading until we get a frame or exhaust retries.
    while (true) {
        if (!impl_->ensure_connected()) {
            impl_->exhausted = true;
            return std::nullopt;
        }
        if (read_one_frame(impl_->fd, frame)) {
            return frame;
        }
        // Lost the stream. Drop the fd; loop will reconnect (if allowed).
        impl_->close_fd();
        if (!impl_->reconnect_enabled) {
            impl_->exhausted = true;
            return std::nullopt;
        }
        // ensure_connected() will retry; if it fails, exhausted is set there.
        impl_->fd = -1;
    }
}

// ============================================================================
// EncodedStreamClient::Impl
// ============================================================================
struct EncodedStreamClient::Impl {
    std::string socket_path;
    int fd = -1;  // cached socket for get_frame()

    explicit Impl(std::string path) : socket_path(std::move(path)) {}
    ~Impl() { close(); }
    Impl(const Impl&) = delete;
    Impl& operator=(const Impl&) = delete;

    void close() {
        if (fd >= 0) {
            ::close(fd);
            fd = -1;
        }
    }

    bool ensure_connected() {
        if (fd >= 0) return true;
        fd = connect_unix(socket_path);
        return fd >= 0;
    }
};

EncodedStreamClient::EncodedStreamClient(std::string socket_path)
    : impl_(std::make_unique<Impl>(std::move(socket_path))) {}
EncodedStreamClient::~EncodedStreamClient() = default;
EncodedStreamClient::EncodedStreamClient(EncodedStreamClient&&) noexcept = default;
EncodedStreamClient& EncodedStreamClient::operator=(EncodedStreamClient&&) noexcept = default;

void EncodedStreamClient::connect() {
    if (impl_->fd < 0) impl_->fd = connect_unix(impl_->socket_path);
}

void EncodedStreamClient::close() { impl_->close(); }

bool EncodedStreamClient::connected() const noexcept { return impl_->fd >= 0; }

std::optional<EncodedFrame> EncodedStreamClient::get_frame(int timeout_ms) {
    if (!impl_->ensure_connected()) return std::nullopt;
    set_recv_timeout(impl_->fd, timeout_ms > 0 ? timeout_ms : kEncDefaultTimeoutMs);

    EncodedFrame frame;
    if (read_one_frame(impl_->fd, frame)) return frame;

    // Timeout/EOF — drop the connection so the next call reconnects.
    impl_->close();
    return std::nullopt;
}

EncodedStream EncodedStreamClient::subscribe(bool reconnect) {
    EncodedStream s;
    s.impl_->socket_path = impl_->socket_path;
    s.impl_->reconnect_enabled = reconnect;
    s.impl_->fd = -1;
    s.impl_->exhausted = false;
    return s;
}

std::thread EncodedStreamClient::on_frame(std::function<void(const EncodedFrame&)> cb,
                                          bool reconnect) {
    auto path = impl_->socket_path;
    return std::thread([path, reconnect, cb = std::move(cb)]() {
        EncodedStream stream;
        stream.impl_->socket_path = path;
        stream.impl_->reconnect_enabled = reconnect;
        stream.impl_->fd = -1;
        stream.impl_->exhausted = false;
        while (auto frame = stream.next()) {
            cb(*frame);
        }
    });
}

}  // namespace hailo_ipc_sdk
