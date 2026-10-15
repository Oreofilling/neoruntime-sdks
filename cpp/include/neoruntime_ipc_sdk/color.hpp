// color.hpp — NV12 <-> packed RGB/BGR plane conversions (port of color.py,
// 0.8.1).
//
// OpenCV only: the C++ SDK's hard dependency replaces Python's numpy
// fallback legs, so every conversion is cv::cvtColor (BT.601 limited range,
// same family cv2 uses) and the *_impl functions are plain cvtColor — there
// is no warn_numpy_fallback machinery to port. nv12_to_rgb / rgb_to_nv12
// ride the accel router's "nv12_to_rgb" / "rgb_to_nv12" ops (DSP convert
// when the daemon is reachable, the cv leg otherwise); nv12_to_bgr /
// bgr_to_nv12 / nv12_resize stay direct.
#pragma once

#include <opencv2/core.hpp>

#include <string>
#include <utility>

namespace neoruntime_ipc_sdk {

// Validation helpers shared with the accel hardware legs (python
// _check_dims / _check_even — same messages). `shape_text` renders a Mat's
// shape numpy-style: "(rows, cols)" for single-channel, "(rows, cols, ch)"
// otherwise.
std::string shape_text(const cv::Mat& mat);
void check_nv12_dims(int width, int height, const cv::Mat& nv12);
void check_nv12_even(int height, int width, const cv::Mat& image);

// NV12 -> RGB (h, w, 3). Rides the accel router's "nv12_to_rgb" op.
cv::Mat nv12_to_rgb(const cv::Mat& nv12, int width, int height);
// CPU leg of nv12_to_rgb (what the router serves when the DSP is not
// reachable) — also the direct implementation for callers that want no
// routing.
cv::Mat nv12_to_rgb_impl(const cv::Mat& nv12, int width, int height);

// NV12 -> BGR. Direct cvtColor, no routing (python parity).
cv::Mat nv12_to_bgr(const cv::Mat& nv12, int width, int height);

// RGB (h, w, 3) -> NV12 (h*3/2, w). Rides the accel router's
// "rgb_to_nv12" op.
cv::Mat rgb_to_nv12(const cv::Mat& rgb);
// CPU leg of rgb_to_nv12.
cv::Mat rgb_to_nv12_impl(const cv::Mat& rgb);

// BGR (h, w, 3) -> NV12. Direct cvtColor, no routing.
cv::Mat bgr_to_nv12(const cv::Mat& bgr);

// Resize an NV12 buffer (Y and UV planes separately, bilinear). Downscale
// only — upscaling raises (model inputs are smaller than the source frame).
cv::Mat nv12_resize(const cv::Mat& nv12, std::pair<int, int> src_size,
                    std::pair<int, int> dst_size);

}  // namespace neoruntime_ipc_sdk
