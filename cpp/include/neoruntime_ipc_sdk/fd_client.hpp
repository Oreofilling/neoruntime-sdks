// fd_client.hpp — zero-copy media client over DMA-BUF fd passing (port of
// fd_client.py).
//
// Speaks the FdPublisher UDS protocol: each frame arrives as an 80-byte
// FdPubFrameMsg carrying up to 3 DMA-BUF fds as SCM_RIGHTS ancillary data.
// Two receive paths:
//   * copy (default)     — planes are mmapped (fenced per HAL-3), copied into
//                          a cv::Mat, fds closed and the frame RELEASEd
//                          immediately;
//   * keep_fd=true       — the dma-buf fds are retained in a FrameHandle and
//                          the daemon-side RELEASE is deferred until the last
//                          Frame copy goes away (see frame.hpp).
//
// Streaming API (shared with encoded.hpp): subscribe() returns a pull stream
// with bounded auto-reconnect; on_frame() runs a background std::thread that
// reconnects for as long as it lives. Callers own the returned thread.
#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#include "neoruntime_ipc_sdk/encoded.hpp"
#include "neoruntime_ipc_sdk/frame.hpp"

namespace neoruntime_ipc_sdk {

class FdMediaClient;

// ============================================================================
// FdMediaStream — pull-based frame stream with bounded auto-reconnect.
// ============================================================================
class FdMediaStream {
public:
    ~FdMediaStream();
    FdMediaStream(FdMediaStream&&) noexcept;
    FdMediaStream& operator=(FdMediaStream&&) noexcept;
    FdMediaStream(const FdMediaStream&) = delete;
    FdMediaStream& operator=(const FdMediaStream&) = delete;

    // Pull the next decoded frame. Blocks. std::nullopt when the server
    // closes and bounded reconnect attempts are exhausted (get a fresh stream
    // from subscribe() to continue).
    std::optional<Frame> next();

private:
    FdMediaStream();
    struct Impl;
    std::unique_ptr<Impl> impl_;
    friend class FdMediaClient;
};

// ============================================================================
// FdMediaClient
// ============================================================================
class FdMediaClient {
public:
    // Empty socket_path resolves via the CAMERA_SOCK_PATH env var, then
    // /run/aipc/camera.sock (the raw FdPublisher UDS, NOT the camera-control
    // gRPC endpoint).
    explicit FdMediaClient(std::string socket_path = "");
    ~FdMediaClient();
    FdMediaClient(const FdMediaClient&) = delete;
    FdMediaClient& operator=(const FdMediaClient&) = delete;
    FdMediaClient(FdMediaClient&&) noexcept;
    FdMediaClient& operator=(FdMediaClient&&) noexcept;

    // Release retained keep-fd frames back to the daemon (deferred RELEASE
    // messages), send UNSUBSCRIBE and close every stream socket. Streams
    // still alive transparently reconnect on their next read. Idempotent.
    void close();

    // One frame for `stream_id` with a timeout. std::nullopt on timeout.
    // Connect/subscribe failures propagate as std::runtime_error (Python
    // get_frame re-raises ConnectionError). With keep_fd=true the frame's
    // dma-buf fds are retained (zero-copy handoff; see FrameHandle) and the
    // daemon-side release is deferred until frame.release() / last-copy
    // destruction / close().
    std::optional<Frame> get_frame(const std::string& stream_id,
                                   int timeout_ms = 5000, bool keep_fd = false);

    // Pull-based stream for `stream_id`. `skip_frames` is a decimation
    // interval: 1 (default) yields every frame, N > 1 yields every Nth frame
    // (decimated keep-fd frames are released back to the daemon immediately
    // so their dma-bufs are not retained). `reconnect` enables bounded
    // auto-reconnect inside next().
    FdMediaStream subscribe(const std::string& stream_id, int skip_frames = 1,
                            bool keep_fd = false, bool reconnect = true);

    // Background thread invoking cb per frame (reconnects for as long as it
    // runs; cb exceptions are swallowed). Detach or join at shutdown.
    std::thread on_frame(const std::string& stream_id,
                         std::function<void(const Frame&)> cb,
                         int skip_frames = 1, bool keep_fd = false);

    // Number of keep-fd frames currently held (not yet RELEASEd). A camera
    // daemon buffer pool is small (single digits); retaining close to that
    // many frames starves the live stream — release frames you no longer
    // need.
    int retained_frames() const;

    // ---- Encoded-stream convenience (delegates to EncodedStreamClient) -------
    // socket_dir "" resolves via ENCODED_SOCK_DIR, then /run/aipc/encoded.
    EncodedStreamClient get_encoded_stream(const std::string& stream_id = "main",
                                           std::string socket_dir = "");

    // Raw stream IDs by querying the camera daemon (streams whose pipeline is
    // active). Falls back to the guaranteed {"main", "sub"} when the camera
    // control service is unreachable or reports nothing; results are cached
    // for 5 seconds. For detailed status use CameraClient::get_stream_status.
    std::vector<std::string> list_streams() const;

    // RTSP must be enabled on the device first (CameraClient or REST API).
    std::string get_rtsp_url(const std::string& stream_id = "main",
                             const std::string& host = "192.0.2.72",
                             int port = 8554) const;

private:
    friend class FdMediaStream;  // streams share this client's cached sockets
    struct Impl;
    std::shared_ptr<Impl> impl_;  // shared: streams keep the client state alive
};

}  // namespace neoruntime_ipc_sdk
