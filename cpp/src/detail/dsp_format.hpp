// dsp_format.hpp — DSP job format inference, geometry validation and CPU
// fallback (port of python dsp_format.py, 0.8.1).
//
// The pure-Mat side of the DSP module: mapping SDK frame formats onto DSP
// formats, validating crop/dest geometry against daemon caps, and the
// software path used when the daemon lacks the DSP surface (or for tests).
// Byte-level parity with the numpy legs is asserted by test_dsp_fallback.
//
// Two deliberate divergences from python 0.8.1's nv12 legs (both are
// defects there, contradicting their own intent comments; the daemon's
// hardware path does what this port does):
//   * stretch keeps (U,V) pairs under nearest sampling when dw != sw —
//     python resizes the byte plane and un-pairs chroma.
//   * letterbox _place writes the full [ox, ox+cw) chroma byte range —
//     python's ox//2:(ox+cw)//2 slice is half-width and raises a numpy
//     broadcast ValueError for every nv12 letterbox.
#pragma once

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <string>
#include <vector>

#include "neoruntime_ipc_sdk/dsp.hpp"

namespace neoruntime_ipc_sdk::detail::dsp_format {

// Frame.format names (PIXEL_FORMAT_NAMES) a handle may carry into an
// import. RGB and BGR both map to rgb24: these ops are byte-order
// agnostic per-pixel geometry transforms, so the plane imports verbatim.
// Returns "" when the frame format has no DSP mapping.
inline std::string frame_fmt_to_dsp(const std::string& frame_format) {
    if (frame_format == "NV12") return "nv12";
    if (frame_format == "RGB") return "rgb24";
    if (frame_format == "BGR") return "rgb24";
    if (frame_format == "GRAY8") return "gray8";
    return "";
}

// Infer the DSP format of a pixel source. `fmt` empty = infer from the
// Mat shape: 3-channel -> rgb24, 2D -> gray8 (with a deprecation warning,
// ambiguous with nv12 — same contract as python).
std::string infer_fmt(const cv::Mat& src, const std::string& fmt);

// (width, height) of a source in a given format; validates the Mat shape.
cv::Size src_dims(const cv::Mat& src, const std::string& fmt);

struct CropRect {
    int x = 0, y = 0, w = 0, h = 0, dw = 0, dh = 0;
};

// Validate one crop rect against source dims sw x sh (frame handles arrive
// as geometry, not pixels — dims are resolved by the caller).
CropRect validated_rect(int sw, int sh, const std::string& fmt, int x, int y, int w,
                        int h, int dw, int dh);

// OpenCV interpolation flag for nearest/bilinear/area/bicubic.
int cv_interp(const std::string& interpolation);

// ---- CPU fallback (parity: same math as the numpy legs) ----------------------
// `scaling` = stretch | letterbox | letterbox_middle | letterbox_up_left |
//             scale_crop; nv12 resizes Y with `interpolation` and UV with
//             nearest (interleaved UV rows must stay paired).
cv::Mat cpu_resize(const cv::Mat& src, const std::string& fmt, int dw, int dh,
                   const std::string& scaling, const std::string& interpolation);

// Paste `content` onto a black canvas at (ox, oy); nv12 pads UV with 128.
cv::Mat place(const cv::Mat& content, const std::string& fmt, int dw, int dh, int ox,
              int oy);

// Crop w x h at (x, y) honoring the format's plane layout.
cv::Mat cpu_crop(const cv::Mat& src, const std::string& fmt, int x, int y, int w, int h);

// Crop then stretch-resize to (dw, dh) when they differ (bilinear, as the
// daemon does).
cv::Mat cpu_crop_resize(const cv::Mat& src, const std::string& fmt, const CropRect& rect);

// Format conversion with same dimensions, differing formats. RGB<->NV12
// rides the BT.601 legs (cv::cvtColor, the same family the daemon uses);
// gray8 pairs are trivial (replicate / luma slice / neutral chroma).
cv::Mat cpu_convert(const cv::Mat& src, const std::string& fmt, const std::string& dst_fmt);

struct RgbaOverlay {
    cv::Mat rgba;  // uint8 (h, w, 4); channels 0..2 = RGB, 3 = alpha
    int x = 0;
    int y = 0;
};

// Straight alpha composite: out = a*C + (255-a)*B over 255, overlays
// pasted 1:1 in order at their (x, y). The base is converted to rgb24 for
// the math and back; the input is never modified.
cv::Mat cpu_blend(const cv::Mat& base, const std::string& fmt,
                  const std::vector<RgbaOverlay>& overlays);

}  // namespace neoruntime_ipc_sdk::detail::dsp_format
