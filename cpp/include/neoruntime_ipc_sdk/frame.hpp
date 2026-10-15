// frame.hpp — Frame types and pixel-format helpers (port of frame.py).
//
// Frame (a decoded or fd-backed video frame), FrameHandle (retained dma-buf
// planes) and the affine transform metadata recorded by crop/resize. The
// media clients (fd_client.hpp / encoded.hpp) produce these; media.hpp
// re-exports everything for backwards compatibility.
#pragma once

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <array>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "neoruntime_ipc_sdk/types.hpp"

namespace neoruntime_ipc_sdk {

// Frame metadata flags — mirror of FD_PUB_FRAME_FLAG_* in the daemon's
// fd_protocol.h. Coarse "bake active" truth per stream at dispatch time: a
// set bit means the corresponding pass ran on the dispatching path, so the
// pixels may be modified. Clean-capture apps should subscribe a stream
// outside the bake set (ai_overlay.stream_map) rather than rely on the bit
// being clear while the pass is enabled.
inline constexpr std::uint32_t FRAME_FLAG_OVERLAY_BAKED = 0x1;
inline constexpr std::uint32_t FRAME_FLAG_DPM_BAKED = 0x2;

// Affine mapping recorded by Frame::crop / Frame::resize — the strong-typed
// form of Python's metadata["transform"] dict. Semantics (per axis):
//   dst = src * scale + origin
// so the inverse — what postprocessing needs to map model-input coordinates
// back to the source frame — is src = (dst - origin) / scale.
struct AffineTransform {
    std::string op;    // "crop" | "resize"
    int src_width = 0;
    int src_height = 0;
    int dst_width = 0;
    int dst_height = 0;
    std::string mode;  // "crop" | "letterbox" | "stretch"
    double scale_x = 1.0;
    double scale_y = 1.0;
    double origin_x = 0.0;
    double origin_y = 0.0;
};

// Compose chained transforms: prev maps src→parent, cur maps parent→dst.
AffineTransform compose_transform(const AffineTransform& prev,
                                  const AffineTransform& cur);

// Retained dma-buf backing store for one received frame (SDK-1).
//
// In keep-fd mode the per-plane dma-buf fds the daemon passed with the FRAME
// message are kept open here and the RELEASE message — the daemon's
// buffer-recycling ticket — is deferred until close(). CPU access to the
// pixels must go through Frame::to_array(), which applies the required
// DMA_BUF_IOCTL_SYNC read fences. The fds can also be handed to a DspClient
// job as-is for a zero-copy hardware path; the geometry carried here is what
// the daemon needs to import them.
//
// Ownership: Frame holds a shared_ptr<FrameHandle> so frames copy cheaply;
// the LAST copy runs the deferred release (Python relies on GC the same
// way). close() is idempotent.
class FrameHandle {
public:
    // Sends the deferred RELEASE for this frame (best-effort, must not throw
    // out; receives the handle so the closure can capture the socket fd).
    using ReleaseFn = std::function<void(FrameHandle&)>;

    FrameHandle(std::vector<int> fds,
                std::array<std::uint32_t, 3> strides,
                std::array<std::uint32_t, 3> plane_sizes,
                std::uint64_t frame_id,
                ReleaseFn on_release,
                int width,
                int height,
                std::string format,
                std::uint32_t flags);
    ~FrameHandle();

    FrameHandle(FrameHandle&&) noexcept;
    FrameHandle& operator=(FrameHandle&&) noexcept;
    FrameHandle(const FrameHandle&) = delete;
    FrameHandle& operator=(const FrameHandle&) = delete;

    // Close the fds and send the deferred RELEASE. Idempotent.
    void close() noexcept;

    bool closed() const noexcept { return closed_; }
    // Convenience accessor for the first plane's fd (-1 when empty).
    int fd() const noexcept { return fds.empty() ? -1 : fds[0]; }

    // Geometry the daemon needs to import the planes (public data, Python
    // attribute parity).
    std::vector<int> fds;
    std::array<std::uint32_t, 3> strides{};
    std::array<std::uint32_t, 3> plane_sizes{};
    std::uint64_t frame_id = 0;
    int width = 0;
    int height = 0;
    std::string format;
    std::uint32_t flags = 0;

private:
    ReleaseFn on_release_;
    bool closed_ = false;
};

// One decoded or fd-backed video frame. `image` layout depends on `format`
// (NV12 is H*3/2 x W single-channel; RGB is H x W x 3). When the frame was
// received with keep_fd=true, `image` is empty until to_array() materializes
// the retained handle (mapped once, then cached).
struct Frame {
    std::uint64_t sequence = 0;
    std::uint64_t timestamp_ns = 0;
    int width = 0;
    int height = 0;
    std::string format;  // "RGB", "BGR", "NV12", "NV21", "RGBA", "BGRA", "GRAY8", "YUYV"
    cv::Mat image;
    std::map<std::string, std::string> metadata;  // reserved (app-defined)
    std::shared_ptr<FrameHandle> handle;          // retained fd backing (keep-fd)
    std::uint32_t flags = 0;                      // FRAME_FLAG_*
    std::optional<AffineTransform> transform;     // chained crop/resize mapping

    bool baked_overlay() const noexcept { return (flags & FRAME_FLAG_OVERLAY_BAKED) != 0; }
    bool baked_dpm() const noexcept { return (flags & FRAME_FLAG_DPM_BAKED) != 0; }

    // Raw pixel buffer as a flat single-row uint8 view; materializes a
    // retained fd (cached in image) exactly like to_array().
    cv::Mat data();
    // Return the raw pixel buffer, materializing a retained fd. Throws
    // std::invalid_argument when there is neither image data nor a handle.
    cv::Mat to_array();
    // Release a retained fd frame back to the daemon (idempotent; no-op for
    // frames that were copied on receive).
    void release() noexcept;

    // Convert to HxWx3 RGB. Throws std::invalid_argument for unsupported
    // formats (NV21, RGBA, BGRA, YUYV).
    cv::Mat to_rgb();

    // New frame cropped to the pixel rectangle. NV12/NV21 require even
    // x, y, width, height (chroma subsampling). The crop geometry is
    // recorded in the new frame's `transform` (scale 1:1, origin (-x, -y)).
    Frame crop(int x, int y, int crop_width, int crop_height) const;

    // New frame resized to width x height. mode: "letterbox" (default; fit
    // inside, preserve aspect, pad with pad_value — NV12 pads luma with
    // pad_value and chroma with neutral 128), "stretch" (fill exactly),
    // "crop" (scale to cover, center-crop the overflow). NV12/NV21 require
    // even target dimensions. The geometry is recorded in `transform` as the
    // per-axis affine dst = src * scale + origin, composed across chained
    // crop/resize calls — postprocessing maps model-input coordinates back
    // with src = (dst - origin) / scale.
    //
    // Phase note: the DSP zero-copy fast path for keep-fd frames lands with
    // the accel layer (Phase 6); this build always takes the CPU path.
    Frame resize(int resize_width, int resize_height,
                 std::string_view mode = "letterbox", int pad_value = 114) const;

    // Encode the frame as JPEG bytes (cv::imencode; hardware encode lands
    // with the accel layer).
    std::vector<std::uint8_t> to_jpeg_bytes(int quality = 85) const;

    // Write the frame (as RGB) to disk. Extension selects the codec
    // (cv::imwrite; .jpg/.jpeg go through the JPEG encoder).
    void save(const std::string& path) const;
};

// Static description of one raw stream (camera daemon pipeline).
struct StreamInfo {
    std::string stream_id;
    int width = 0;
    int height = 0;
    std::string format;
    double fps = 0.0;
    int buffer_count = 0;
};

// Format classification shared by crop/resize (Python _PACKED_FORMATS etc.).
bool is_packed_format(std::string_view format);   // RGB/BGR/RGBA/BGRA/GRAY8
bool is_yuv_format(std::string_view format);      // NV12/NV21
// Formats the DSP can scale zero-copy (must stay in sync with the DSP layer).
bool is_dsp_resize_format(std::string_view format);  // NV12/RGB/BGR/GRAY8

}  // namespace neoruntime_ipc_sdk
