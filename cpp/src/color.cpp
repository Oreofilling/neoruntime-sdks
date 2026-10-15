// color.cpp — port of color.py (0.8.1), cv2 legs only (OpenCV replaces
// python's numpy fallbacks; no warn_numpy_fallback machinery). Messages
// match the python f-strings byte for byte.
#include "neoruntime_ipc_sdk/color.hpp"

#include <opencv2/imgproc.hpp>

#include <stdexcept>
#include <string>
#include <utility>

#include "neoruntime_ipc_sdk/accel.hpp"

namespace neoruntime_ipc_sdk {

namespace {

// numpy-style shape tuple: "(rows, cols)" single-channel, "(rows, cols,
// ch)" multi-channel, "(d0, d1, ...)" beyond 2 dims.
std::string shape_text_impl(const cv::Mat& m) {
    std::string out = "(";
    if (m.dims <= 2) {
        out += std::to_string(m.rows) + ", " + std::to_string(m.cols);
        if (m.channels() > 1) out += ", " + std::to_string(m.channels());
    } else {
        for (int i = 0; i < m.dims; ++i) {
            if (i > 0) out += ", ";
            out += std::to_string(m.size.p[i]);
        }
    }
    return out + ")";
}

// python _packed_to_nv12 (cv2 leg): RGB/BGR -> I420 -> repack the U/V
// planes into the NV12 UV plane (U even columns, V odd).
cv::Mat packed_to_nv12(const cv::Mat& image, bool bgr_order) {
    const int height = image.rows;
    const int width = image.cols;
    check_nv12_even(height, width, image);
    const int code = bgr_order ? cv::COLOR_BGR2YUV_I420 : cv::COLOR_RGB2YUV_I420;
    cv::Mat i420;
    cv::cvtColor(image, i420, code);  // (h*3/2, w) 8UC1, I420 plane layout
    const cv::Mat y_plane = i420(cv::Rect(0, 0, width, height));
    const cv::Mat u_plane =
        i420(cv::Rect(0, height, width, height / 4)).clone().reshape(1, height / 2);
    const cv::Mat v_plane =
        i420(cv::Rect(0, height + height / 4, width, height / 4)).clone().reshape(1, height / 2);
    cv::Mat uv_plane(height / 2, width, CV_8UC1);
    for (int y = 0; y < height / 2; ++y) {
        const uchar* urow = u_plane.ptr<uchar>(y);
        const uchar* vrow = v_plane.ptr<uchar>(y);
        uchar* dst = uv_plane.ptr<uchar>(y);
        for (int x = 0; x < width / 2; ++x) {
            dst[2 * x] = urow[x];
            dst[2 * x + 1] = vrow[x];
        }
    }
    cv::Mat out;
    cv::vconcat(y_plane, uv_plane, out);
    return out;
}

}  // namespace

std::string shape_text(const cv::Mat& mat) { return shape_text_impl(mat); }

void check_nv12_dims(int width, int height, const cv::Mat& nv12) {
    const int expected = height * 3 / 2;
    if (nv12.rows != expected || nv12.cols != width)
        throw std::invalid_argument("NV12 buffer is " + shape_text_impl(nv12) + ", expected (" +
                                    std::to_string(expected) + ", " + std::to_string(width) +
                                    ") for " + std::to_string(width) + "x" +
                                    std::to_string(height));
}

void check_nv12_even(int height, int width, const cv::Mat& image) {
    if (height % 2 != 0 || width % 2 != 0)
        throw std::invalid_argument("NV12 needs even dimensions, got " + std::to_string(width) +
                                    "x" + std::to_string(height) + " (shape " +
                                    shape_text_impl(image) + ")");
}

cv::Mat nv12_to_rgb(const cv::Mat& nv12, int width, int height) {
    return std::any_cast<cv::Mat>(get_default_router().run(
        "nv12_to_rgb", {std::any(nv12), std::any(width), std::any(height)}));
}

cv::Mat nv12_to_rgb_impl(const cv::Mat& nv12, int width, int height) {
    if (nv12.dims != 2 || nv12.channels() != 1)
        // same contract as the hardware leg — a 3D NV12-shaped array is a
        // caller error, not something cv2 should fail on deep inside
        throw std::invalid_argument("nv12 source must be 2D, got shape " +
                                    shape_text_impl(nv12));
    check_nv12_dims(width, height, nv12);
    cv::Mat out;
    cv::cvtColor(nv12, out, cv::COLOR_YUV2RGB_NV12);
    return out;
}

cv::Mat nv12_to_bgr(const cv::Mat& nv12, int width, int height) {
    check_nv12_dims(width, height, nv12);
    cv::Mat out;
    cv::cvtColor(nv12, out, cv::COLOR_YUV2BGR_NV12);
    return out;
}

cv::Mat rgb_to_nv12(const cv::Mat& rgb) {
    return std::any_cast<cv::Mat>(get_default_router().run("rgb_to_nv12", {std::any(rgb)}));
}

cv::Mat rgb_to_nv12_impl(const cv::Mat& rgb) { return packed_to_nv12(rgb, false); }

cv::Mat bgr_to_nv12(const cv::Mat& bgr) { return packed_to_nv12(bgr, true); }

cv::Mat nv12_resize(const cv::Mat& nv12, std::pair<int, int> src_size,
                    std::pair<int, int> dst_size) {
    const int src_w = src_size.first;
    const int src_h = src_size.second;
    const int dst_w = dst_size.first;
    const int dst_h = dst_size.second;
    check_nv12_dims(src_w, src_h, nv12);
    if (dst_w % 2 != 0 || dst_h % 2 != 0)
        throw std::invalid_argument("destination must be even-sized, got " + std::to_string(dst_w) +
                                    "x" + std::to_string(dst_h));
    if (dst_w > src_w || dst_h > src_h)
        throw std::invalid_argument("upscaling " + std::to_string(src_w) + "x" +
                                    std::to_string(src_h) + " -> " + std::to_string(dst_w) + "x" +
                                    std::to_string(dst_h) +
                                    " not supported (model inputs are smaller than the source "
                                    "frame)");

    // resize the Y and UV planes separately — avoids the
    // NV12->BGR->resize->BGR->NV12 round-trip
    const cv::Mat y_plane = nv12(cv::Rect(0, 0, src_w, src_h));
    const cv::Mat uv_plane = nv12(cv::Rect(0, src_h, src_w, src_h / 2));
    cv::Mat y_out, uv_out;
    cv::resize(y_plane, y_out, cv::Size(dst_w, dst_h), 0, 0, cv::INTER_LINEAR);
    cv::resize(uv_plane, uv_out, cv::Size(dst_w, dst_h / 2), 0, 0, cv::INTER_LINEAR);
    cv::Mat out;
    cv::vconcat(y_out, uv_out, out);
    return out;
}

}  // namespace neoruntime_ipc_sdk
