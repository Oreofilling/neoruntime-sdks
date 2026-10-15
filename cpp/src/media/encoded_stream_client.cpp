// encoded_stream_client.cpp — EncodedStreamClient + EncodedStream.
// Port of encoded.py: raw UDS, 30-byte little-endian EncHeader +
// length-prefixed NALU payload (H.264/H.265), optional V3 packet-seq
// extension (flags bit1 → 8 more bytes) feeding the shared observability
// counters (EncodedSeqStats).
#include "neoruntime_ipc_sdk/encoded.hpp"

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>

#include "detail/fd_protocol.hpp"
#include "detail/raw_socket.hpp"
#include "detail/transport.hpp"

namespace neoruntime_ipc_sdk {

using detail::EncHeader;
using detail::connect_unix;
using detail::recv_exact;
using detail::set_recv_timeout;

namespace {
constexpr int kEncHeaderSize = sizeof(EncHeader);          // 30
constexpr std::size_t kEncMaxPayload = 50u * 1024u * 1024u;  // 50 MB sanity cap
constexpr int kEncReconnectAttempts = 3;                     // bounded reconnect
constexpr int kEncBackoffBaseMs = 100;                        // linear backoff step
constexpr int kEncDefaultTimeoutMs = 5000;

// Packet-seq observability shared by the client and every stream/thread it
// spawns (Python keeps plain attributes on the client instance; atomics here
// because on_frame reads run on a background thread). Relaxed ordering: the
// counters are advisory, and each socket has a single reader at a time.
struct SeqTracker {
    std::atomic<std::uint64_t> seq_packets{0};
    std::atomic<std::uint64_t> seq_gap_events{0};
    std::atomic<std::uint64_t> seq_missing{0};
    std::atomic<std::uint64_t> last_seq{0};
    std::atomic<bool> has_last{false};

    void observe(std::uint64_t seq) {
        if (has_last.load(std::memory_order_relaxed)) {
            const std::uint64_t prev = last_seq.load(std::memory_order_relaxed);
            if (seq > prev + 1) {
                seq_gap_events.fetch_add(1, std::memory_order_relaxed);
                seq_missing.fetch_add(seq - prev - 1, std::memory_order_relaxed);
            }
            // seq <= prev: publisher restarted (seq rebased to 1). Not a
            // loss — the next forward jump from the new baseline is what
            // counts.
        }
        last_seq.store(seq, std::memory_order_relaxed);
        has_last.store(true, std::memory_order_relaxed);
        seq_packets.fetch_add(1, std::memory_order_relaxed);
    }
};

// Read one complete encoded frame from a connected socket. Returns false on
// EOF/timeout/short-read/bogus length (caller decides whether to reconnect).
bool read_one_frame(int fd, EncodedFrame& out, SeqTracker& seq) {
    EncHeader hdr{};
    if (!recv_exact(fd, &hdr, kEncHeaderSize)) return false;

    std::optional<std::uint64_t> packet_seq;
    int header_size = kEncHeaderSize;
    if (hdr.flags & detail::ENC_FLAG_SEQ_PRESENT) {
        std::uint8_t seq_bytes[detail::ENC_SEQ_SIZE];
        if (!recv_exact(fd, seq_bytes, detail::ENC_SEQ_SIZE)) return false;
        std::uint64_t v = 0;
        for (int i = detail::ENC_SEQ_SIZE - 1; i >= 0; --i) {
            v = (v << 8) | seq_bytes[i];  // little-endian wire order
        }
        packet_seq = v;
        header_size += detail::ENC_SEQ_SIZE;
    }

    // total_size counts header + payload; payload may be zero (keepalive).
    const std::uint32_t total = hdr.total_size;
    if (total < static_cast<std::uint32_t>(header_size)) return false;
    const std::size_t payload = total - static_cast<std::size_t>(header_size);
    if (payload > kEncMaxPayload) return false;

    out.codec = hdr.codec;
    out.flags = hdr.flags;
    out.pts_ns = hdr.pts_ns;
    out.width = static_cast<int>(hdr.width);
    out.height = static_cast<int>(hdr.height);
    out.dts_ns = hdr.dts_ns;
    out.seq = packet_seq;

    out.data.resize(payload);
    if (payload > 0 && !recv_exact(fd, out.data.data(), payload)) return false;

    if (packet_seq.has_value()) seq.observe(*packet_seq);
    return true;
}

// encoded.py socket resolution: ENCODED_SOCK_DIR, then /run/aipc/encoded.
// (config.py spells the env var ENCODED_SOCKET_DIR; parity is with encoded.py.)
std::string default_encoded_socket_dir() {
    const char* env = std::getenv("ENCODED_SOCK_DIR");
    if (env && *env) return env;
    return "/run/aipc/encoded";
}

}  // namespace

std::string EncodedFrame::codec_name() const {
    switch (codec) {
        case 0: return "h264";
        case 1: return "h265";
        default: return "unknown(" + std::to_string(codec) + ")";
    }
}

// ============================================================================
// EncodedStream::Impl — owns its own connection, bounded auto-reconnect.
// ============================================================================
struct EncodedStream::Impl : detail::UdsStreamClientBase {
    bool exhausted = false;  // reconnections spent; next() will keep returning nullopt
    std::shared_ptr<SeqTracker> seq;  // shared with the owning client

    Impl(std::string path, bool reconnect, std::shared_ptr<SeqTracker> tracker)
        : UdsStreamClientBase(std::move(path), reconnect,
                              kEncReconnectAttempts, kEncBackoffBaseMs),
          seq(std::move(tracker)) {}

    // close_fd() here (not only the base dtor): on_disconnect must dispatch
    // while the derived object is still alive.
    ~Impl() { close_fd(); }

    int do_connect() override { return connect_unix(socket_path_); }
};

EncodedStream::EncodedStream() : impl_(std::make_unique<Impl>("", false, std::make_shared<SeqTracker>())) {}
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
        if (read_one_frame(impl_->fd(), frame, *impl_->seq)) {
            return frame;
        }
        // Lost the stream. Drop the fd; loop will reconnect (if allowed).
        impl_->close_fd();
        if (!impl_->reconnect_enabled()) {
            impl_->exhausted = true;
            return std::nullopt;
        }
        // ensure_connected() will retry; if it fails, exhausted is set there.
    }
}

// ============================================================================
// EncodedStreamClient::Impl — lazy cached connection for get_frame().
// Unlike the streams, connect failures PROPAGATE (Python get_frame lets
// ConnectionError from the lazy connect reach the caller); only the pull
// streams swallow-and-retry.
// ============================================================================
struct EncodedStreamClient::Impl {
    std::string socket_path;
    int fd = -1;  // cached socket for get_frame()
    std::shared_ptr<SeqTracker> seq;  // shared with every stream/thread spawned here

    explicit Impl(std::string path)
        : socket_path(std::move(path)), seq(std::make_shared<SeqTracker>()) {}
    ~Impl() { close(); }
    Impl(const Impl&) = delete;
    Impl& operator=(const Impl&) = delete;

    void close() {
        if (fd >= 0) {
            ::close(fd);
            fd = -1;
        }
    }

    // Throws std::runtime_error when the socket is unreachable.
    void ensure_connected() {
        if (fd >= 0) return;
        fd = connect_unix(socket_path);
    }
};

EncodedStreamClient::EncodedStreamClient(std::string socket_path,
                                         const std::string& stream_id,
                                         std::string socket_dir) {
    if (socket_path.empty()) {
        std::string base = std::move(socket_dir);
        if (base.empty()) base = default_encoded_socket_dir();
        while (!base.empty() && base.back() == '/') base.pop_back();
        socket_path = base + "/" + stream_id + ".sock";
    }
    impl_ = std::make_unique<Impl>(std::move(socket_path));
}
EncodedStreamClient::~EncodedStreamClient() = default;
EncodedStreamClient::EncodedStreamClient(EncodedStreamClient&&) noexcept = default;
EncodedStreamClient& EncodedStreamClient::operator=(EncodedStreamClient&&) noexcept = default;

void EncodedStreamClient::connect() { impl_->ensure_connected(); }

void EncodedStreamClient::close() { impl_->close(); }

bool EncodedStreamClient::connected() const noexcept { return impl_->fd >= 0; }

std::optional<EncodedFrame> EncodedStreamClient::get_frame(int timeout_ms) {
    impl_->ensure_connected();  // throws on unreachable socket (Python parity)
    set_recv_timeout(impl_->fd, timeout_ms > 0 ? timeout_ms : kEncDefaultTimeoutMs);

    EncodedFrame frame;
    if (read_one_frame(impl_->fd, frame, *impl_->seq)) return frame;

    // Timeout/EOF — drop the connection so the next call reconnects.
    impl_->close();
    return std::nullopt;
}

EncodedStream EncodedStreamClient::subscribe(bool reconnect) {
    EncodedStream s;
    s.impl_ = std::make_unique<EncodedStream::Impl>(impl_->socket_path, reconnect,
                                                    impl_->seq);
    return s;
}

std::thread EncodedStreamClient::on_frame(std::function<void(const EncodedFrame&)> cb,
                                          bool reconnect) {
    auto path = impl_->socket_path;
    auto tracker = impl_->seq;
    return std::thread([path, reconnect, tracker, cb = std::move(cb)]() {
        // Python's generator reconnects forever; each EncodedStream carries a
        // bounded budget, so take a fresh one whenever the budget is spent.
        while (true) {
            EncodedStream stream;
            stream.impl_ = std::make_unique<EncodedStream::Impl>(path, reconnect,
                                                                 tracker);
            while (auto frame = stream.next()) {
                try {
                    cb(*frame);
                } catch (...) {
                    // Callback errors must not kill the pump (Python logs and
                    // continues).
                }
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
        }
    });
}

EncodedSeqStats EncodedStreamClient::seq_stats() const {
    EncodedSeqStats out;
    out.seq_packets = impl_->seq->seq_packets.load(std::memory_order_relaxed);
    out.seq_gap_events = impl_->seq->seq_gap_events.load(std::memory_order_relaxed);
    out.seq_missing = impl_->seq->seq_missing.load(std::memory_order_relaxed);
    if (impl_->seq->has_last.load(std::memory_order_relaxed)) {
        out.last_seq = impl_->seq->last_seq.load(std::memory_order_relaxed);
    }
    return out;
}

}  // namespace neoruntime_ipc_sdk
