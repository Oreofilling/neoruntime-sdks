// encoded.hpp — encoded video stream client over the EncodedPublisher UDS
// socket (port of encoded.py).
//
// Reads length-prefixed H.264/H.265 NAL units from sockets like
// /run/aipc/encoded/main.sock. Wire framing: a 30-byte little-endian header
// (see detail::EncHeader) plus payload; when flags bit1 is set the header
// carries the V3 extension — 8 more bytes of packet sequence number, first
// packet = 1, assigned at publisher enqueue.
//
// Streaming API (shared with fd_client.hpp): subscribe() returns a pull
// stream whose next() blocks and reconnects a bounded number of times before
// yielding std::nullopt; on_frame() runs a background std::thread that
// reconnects for as long as it lives (matches the unbounded Python
// generator). Callers own the returned thread (detach or join at shutdown).
#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <thread>
#include <vector>

namespace neoruntime_ipc_sdk {

// One encoded (H.264/H.265) frame from an EncodedPublisher.
struct EncodedFrame {
    int codec = 0;            // 0=h264, 1=h265
    std::uint32_t flags = 0;  // bit0 = keyframe, bit1 = V3 seq present
    std::uint64_t pts_ns = 0;  // presentation timestamp
    int width = 0;
    int height = 0;
    std::uint64_t dts_ns = 0;  // decode timestamp
    std::string data;          // encoded NALU payload
    // Publisher packet sequence number (wire V3, flags bit1). Monotonically
    // increasing per stream; std::nullopt when the server predates V3
    // (30-byte header).
    std::optional<std::uint64_t> seq;

    bool is_keyframe() const noexcept { return (flags & 0x01u) != 0; }
    // "h264" / "h265" / "unknown(N)"
    std::string codec_name() const;
};

// Snapshot of a client's packet-seq observability counters. Deliberately NOT
// reset on reconnect: packets enqueued while this client was away were
// publisher-accepted but never received here, and that hole is the thing
// worth seeing. A seq that goes backwards is a publisher restart (rebased to
// 1), not a loss — only forward jumps count as gaps. Counters accumulate
// across every read path sharing the client (get_frame, subscribe streams,
// on_frame threads), mirroring the Python client-instance attributes.
struct EncodedSeqStats {
    std::uint64_t seq_packets = 0;    // frames that carried a seq
    std::uint64_t seq_gap_events = 0; // discontinuities seen (count, not size)
    std::uint64_t seq_missing = 0;    // total packets inside those holes
    std::optional<std::uint64_t> last_seq;
};

// ============================================================================
// EncodedStream — pull-based frame stream with bounded auto-reconnect.
// ============================================================================
class EncodedStream {
public:
    ~EncodedStream();
    EncodedStream(EncodedStream&&) noexcept;
    EncodedStream& operator=(EncodedStream&&) noexcept;
    EncodedStream(const EncodedStream&) = delete;
    EncodedStream& operator=(const EncodedStream&) = delete;

    // Pull the next encoded frame. Blocks until one arrives. Returns
    // std::nullopt when the server closes and bounded reconnect attempts are
    // exhausted (get a fresh stream from subscribe() to continue).
    std::optional<EncodedFrame> next();

private:
    EncodedStream();
    struct Impl;
    std::unique_ptr<Impl> impl_;
    friend class EncodedStreamClient;
};

// ============================================================================
// EncodedStreamClient — length-prefixed H.264/H.265 over UDS.
// ============================================================================
class EncodedStreamClient {
public:
    // `socket_path` (explicit) wins; otherwise the path is derived as
    // `{socket_dir}/{stream_id}.sock` with `socket_dir` defaulting to the
    // ENCODED_SOCK_DIR env var, then /run/aipc/encoded (encoded.py reads this
    // exact name; note config.py's get_encoded_socket_dir spells it
    // ENCODED_SOCKET_DIR — parity here is with encoded.py).
    explicit EncodedStreamClient(std::string socket_path = "",
                                 const std::string& stream_id = "main",
                                 std::string socket_dir = "");
    ~EncodedStreamClient();
    EncodedStreamClient(const EncodedStreamClient&) = delete;
    EncodedStreamClient& operator=(const EncodedStreamClient&) = delete;
    EncodedStreamClient(EncodedStreamClient&&) noexcept;
    EncodedStreamClient& operator=(EncodedStreamClient&&) noexcept;

    // Connect eagerly (otherwise the first get_frame()/subscribe() connects).
    // Throws std::runtime_error when the socket is unreachable.
    void connect();
    void close();
    bool connected() const noexcept;

    // One frame with a timeout (lazy connect; connect failures propagate as
    // std::runtime_error, matching Python's ConnectionError). std::nullopt on
    // timeout or EOF.
    std::optional<EncodedFrame> get_frame(int timeout_ms = 5000);

    // Pull-based stream; `reconnect` enables bounded auto-reconnect in next().
    EncodedStream subscribe(bool reconnect = true);

    // Background thread invoking cb per frame (reconnects for as long as it
    // runs; cb exceptions are swallowed). Detach or join at shutdown.
    std::thread on_frame(std::function<void(const EncodedFrame&)> cb,
                         bool reconnect = true);

    // Packet-seq observability counters shared by every read path of this
    // client (see EncodedSeqStats).
    EncodedSeqStats seq_stats() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace neoruntime_ipc_sdk
