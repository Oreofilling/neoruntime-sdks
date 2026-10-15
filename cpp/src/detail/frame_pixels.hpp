// frame_pixels.hpp — pixel materialization shared by Frame's accessors and
// the FdMediaClient copy path. Lives next to frame.cpp so both read paths
// fence DMA-BUF access and reshape planes identically (single source of
// truth; fd_media_client.cpp previously carried a divergent copy).
//
// INTERNAL (PRIVATE include path). No public API depends on this header.
#pragma once

#include <opencv2/core.hpp>

#include <cstdint>
#include <string>
#include <vector>

#include "neoruntime_ipc_sdk/frame.hpp"

namespace neoruntime_ipc_sdk::detail {

// Map each plane of a retained frame (DMA_BUF_IOCTL_SYNC READ fences, fstat
// mapping size, protocol plane size copied) and return them as one flat
// buffer. Throws std::runtime_error when a plane cannot be mapped or there
// are no planes.
std::vector<std::uint8_t> materialize_frame_handle(const FrameHandle& handle);

// Reshape a flat frame buffer into its per-format cv::Mat layout (NV12 is
// H*3/2 x W single-channel; RGB is H x W x 3; unknown codes fall back to
// 3-channel). The returned Mat owns its pixels (clone).
cv::Mat decode_frame_raw(const std::vector<std::uint8_t>& raw, int width,
                         int height, const std::string& format);

}  // namespace neoruntime_ipc_sdk::detail
