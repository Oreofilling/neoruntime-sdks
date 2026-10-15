// jpeg_encode.hpp — the cv2-backed JPEG encode shared by Frame's save
// paths and the accel router's "encode_jpeg" software leg (python
// frame._encode_jpeg, cv2 leg — the C++ SDK has no Pillow fallback).
#pragma once

#include <opencv2/core.hpp>

#include <cstdint>
#include <vector>

namespace neoruntime_ipc_sdk::detail {

// Encode an RGB (h, w, 3) uint8 image to JPEG bytes at `quality` (1..100).
std::vector<std::uint8_t> encode_jpeg_rgb(const cv::Mat& rgb, int quality);

}  // namespace neoruntime_ipc_sdk::detail
