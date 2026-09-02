// media.hpp — zero-copy video via DMA-BUF fd-passing (FdMediaClient) and encoded
// H.264/H.265 streams (EncodedStreamClient). 1:1 port of media.py.
//
// These two clients do NOT use gRPC. They speak two raw UDS protocols:
//   * FdMediaClient      — FdPublisher: recvmsg + SCM_RIGHTS fd-passing, then
//                          mmap each DMA-BUF plane into a cv::Mat and RELEASE it.
//   * EncodedStreamClient— EncodedPublisher: 30-byte little-endian header +
//                          length-prefixed NALU payload.
//
// Streaming API: Python's infinite reconnecting `subscribe()` generator maps to
//   * subscribe()  -> a pull stream whose next() blocks and reconnects a bounded
//                     number of times before yielding std::nullopt;
//   * on_frame()   -> a background std::thread that loops forever (matches the
//                     unbounded Python generator).
#pragma once

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#include "neoruntime_ipc_sdk/types.hpp"

namespace neoruntime_ipc_sdk {

// One decoded video frame. `image` is a cv::Mat whose layout depends on `format`
// (e.g. NV12 is H*3/2 x W single-channel; RGB is H x W x 3). to_rgb() converts to
// a standard 3-channel RGB Mat for display; save() writes a PNG/JPG.
struct Frame {
    std::uint64_t sequence = 0;
    std::uint64_t timestamp_ns = 0;
    int width = 0;
    int height = 0;
    std::string format;  // "RGB", "BGR", "NV12", "NV21", "RGBA", "BGRA", "GRAY8", "YUYV"
    cv::Mat image;
    std::map<std::string, std::string> metadata;  // reserved (currently unused)

    // Convert to HxWx3 RGB. Throws std::invalid_argument for unsupported formats.
    cv::Mat to_rgb() const;
    // Write the frame (as RGB) to disk. Extension selects the codec (cv::imwrite).
    void save(const std::string& path) const;
    // Flat uint8 view of the raw pixel bytes (API parity with Frame.data).
    cv::Mat data() const;
};

// One encoded (H.264/H.265) frame from an EncodedPublisher.
struct EncodedFrame {
    int codec = 0;            // 0=h264, 1=h265
    std::uint32_t flags = 0;  // bit0 = keyframe
    std::uint64_t pts_ns = 0;
    int width = 0;
    int height = 0;
    std::uint64_t dts_ns = 0;
    std::string data;         // raw NALU payload

    bool is_keyframe() const noexcept { return (flags & 0x01u) != 0; }
    std::string codec_name() const;  // "h264" / "h265"
};

// ============================================================================
// EncodedStreamClient — length-prefixed H.264/H.265 over UDS.
// ============================================================================
class EncodedStream {
public:
    ~EncodedStream();
    EncodedStream(EncodedStream&&) noexcept;
    EncodedStream& operator=(EncodedStream&&) noexcept;
    EncodedStream(const EncodedStream&) = delete;
    EncodedStream& operator=(const EncodedStream&) = delete;

    // Pull the next encoded frame. Blocks until one arrives. Returns std::nullopt
    // when the server closes and bounded reconnect attempts are exhausted.
    std::optional<EncodedFrame> next();

private:
    EncodedStream();
    struct Impl;
    std::unique_ptr<Impl> impl_;
    friend class EncodedStreamClient;
};

class EncodedStreamClient {
public:
    explicit EncodedStreamClient(std::string socket_path);
    ~EncodedStreamClient();
    EncodedStreamClient(const EncodedStreamClient&) = delete;
    EncodedStreamClient& operator=(const EncodedStreamClient&) = delete;
    EncodedStreamClient(EncodedStreamClient&&) noexcept;
    EncodedStreamClient& operator=(EncodedStreamClient&&) noexcept;

    void connect();
    void close();
    bool connected() const noexcept;

    // One frame with a timeout. std::nullopt on timeout or EOF.
    std::optional<EncodedFrame> get_frame(int timeout_ms = 5000);

    // Pull-based stream. `reconnect` enables bounded auto-reconnect inside next().
    EncodedStream subscribe(bool reconnect = true);

    // Background thread invoking cb per frame (unbounded reconnect). Detach or
    // join the returned thread; close() does not stop it — return from cb to end.
    std::thread on_frame(std::function<void(const EncodedFrame&)> cb, bool reconnect = true);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// ============================================================================
// FdMediaClient — zero-copy DMA-BUF video via fd-passing + mmap.
// ============================================================================
class FdMediaStream {
public:
    ~FdMediaStream();
    FdMediaStream(FdMediaStream&&) noexcept;
    FdMediaStream& operator=(FdMediaStream&&) noexcept;
    FdMediaStream(const FdMediaStream&) = delete;
    FdMediaStream& operator=(const FdMediaStream&) = delete;

    // Pull the next decoded frame. Blocks. std::nullopt when the server closes
    // and bounded reconnect attempts are exhausted.
    std::optional<Frame> next();

private:
    FdMediaStream();
    struct Impl;
    std::unique_ptr<Impl> impl_;
    friend class FdMediaClient;
};

class FdMediaClient {
public:
    explicit FdMediaClient(std::string socket_path = "");
    ~FdMediaClient();
    FdMediaClient(const FdMediaClient&) = delete;
    FdMediaClient& operator=(const FdMediaClient&) = delete;
    FdMediaClient(FdMediaClient&&) noexcept;
    FdMediaClient& operator=(FdMediaClient&&) noexcept;

    void close();

    // One decoded frame for `stream_id` with a timeout. std::nullopt on timeout.
    std::optional<Frame> get_frame(const std::string& stream_id, int timeout_ms = 5000);

    // Pull-based stream for `stream_id` with bounded auto-reconnect.
    FdMediaStream subscribe(const std::string& stream_id, bool reconnect = true);

    // Background thread invoking cb per frame (unbounded reconnect).
    std::thread on_frame(const std::string& stream_id,
                         std::function<void(const Frame&)> cb,
                         bool reconnect = true);

    // ---- Encoded-stream convenience (delegates to EncodedStreamClient) -------
    // socket_dir defaults to Config::get_encoded_socket_dir() when empty.
    EncodedStreamClient get_encoded_stream(const std::string& stream_id = "main",
                                           std::string socket_dir = "");

    // Common raw stream IDs (use CameraClient::get_stream_status for detail).
    std::vector<std::string> list_streams() const;

    // RTSP must be enabled on the device first (CameraClient or REST API).
    std::string get_rtsp_url(const std::string& stream_id = "main",
                             const std::string& host = "192.0.2.72",
                             int port = 8554) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace neoruntime_ipc_sdk
