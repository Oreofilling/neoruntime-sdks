// dsp_format.cpp — see detail/dsp_format.hpp (port of python dsp_format.py).
#include "detail/dsp_format.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <vector>

#include "detail/dsp_wire.hpp"

namespace neoruntime_ipc_sdk::detail::dsp_format {
namespace {

using neoruntime_ipc_sdk::DspError;

// Python np.round / cvRound: half-to-even, not half-away-from-zero.
inline int half_even(double v) { return static_cast<int>(std::lrint(v)); }

// ---- BT.601 legs (cv2 family, same as python color.py's cv legs) -------------
// RGB (h, w, 3) -> NV12 (h*3/2, w): I420 planar, then interleave U/V rows.
cv::Mat rgb_to_nv12(const cv::Mat& rgb) {
    const int h = rgb.rows, w = rgb.cols;
    if (h % 2 || w % 2)
        throw std::invalid_argument("NV12 needs even dimensions, got " + std::to_string(w) +
                                    "x" + std::to_string(h));
    cv::Mat i420;
    cv::cvtColor(rgb, i420, cv::COLOR_RGB2YUV_I420);
    cv::Mat y = i420.rowRange(0, h).clone();
    const cv::Mat u = i420.rowRange(h, h + h / 4).reshape(1, h / 2);  // (h/2, w/2)
    const cv::Mat v = i420.rowRange(h + h / 4, h * 3 / 2).reshape(1, h / 2);
    cv::Mat uv(h / 2, w, CV_8UC1);
    for (int r = 0; r < h / 2; ++r) {
        const std::uint8_t* ur = u.ptr<std::uint8_t>(r);
        const std::uint8_t* vr = v.ptr<std::uint8_t>(r);
        std::uint8_t* out = uv.ptr<std::uint8_t>(r);
        for (int c = 0; c < w / 2; ++c) {
            out[2 * c] = ur[c];
            out[2 * c + 1] = vr[c];
        }
    }
    cv::Mat nv12(h * 3 / 2, w, CV_8UC1);
    y.copyTo(nv12.rowRange(0, h));
    uv.copyTo(nv12.rowRange(h, h * 3 / 2));
    return nv12;
}

// NV12 (h*3/2, w) -> RGB (h, w, 3).
cv::Mat nv12_to_rgb(const cv::Mat& nv12, int width, int height) {
    const int expected = height * 3 / 2;
    if (nv12.rows != expected || nv12.cols != width)
        throw std::invalid_argument("NV12 buffer is (" + std::to_string(nv12.rows) + ", " +
                                    std::to_string(nv12.cols) + "), expected (" +
                                    std::to_string(expected) + ", " + std::to_string(width) +
                                    ") for " + std::to_string(width) + "x" +
                                    std::to_string(height));
    cv::Mat rgb;
    cv::cvtColor(nv12, rgb, cv::COLOR_YUV2RGB_NV12);
    return rgb;
}

cv::Mat resize_plane(const cv::Mat& plane, int out_h, int out_w, int interp) {
    cv::Mat out;
    cv::resize(plane, out, cv::Size(out_w, out_h), 0, 0, interp);
    return out;
}

}  // namespace

std::string infer_fmt(const cv::Mat& src, const std::string& fmt) {
    if (!fmt.empty()) {
        if (!dsp_wire::is_dsp_format(fmt))
            throw DspError("unsupported format '" + fmt + "' (nv12/rgb24/gray8)");
        return fmt;
    }
    if (src.channels() == 3) return "rgb24";
    if (src.channels() == 1 && src.dims == 2) {
        static bool warned = false;  // once per process, like warnings' default filter
        if (!warned) {
            warned = true;
            std::fprintf(stderr,
                         "neoruntime_ipc_sdk DeprecationWarning: inferring gray8 from a "
                         "2D array is ambiguous with nv12; pass fmt= explicitly (this "
                         "inference will be removed)\n");
        }
        return "gray8";
    }
    throw DspError("cannot infer format from shape (" + std::to_string(src.rows) + ", " +
                   std::to_string(src.cols) + ", " + std::to_string(src.channels()) + ")");
}

cv::Size src_dims(const cv::Mat& src, const std::string& fmt) {
    if (fmt == "nv12") {
        if (src.channels() != 1 || src.dims != 2)
            throw DspError("nv12 src must be (h*3//2, w)");
        return {src.cols, src.rows * 2 / 3};
    }
    if (fmt == "rgb24") {
        if (src.channels() != 3) throw DspError("rgb24 src must be (h, w, 3)");
        return {src.cols, src.rows};
    }
    if (src.channels() != 1 || src.dims != 2) throw DspError("gray8 src must be (h, w)");
    return {src.cols, src.rows};
}

CropRect validated_rect(int sw, int sh, const std::string& fmt, int x, int y, int w, int h,
                        int dw, int dh) {
    if (w < 1 || h < 1)
        throw DspError("crop size must be positive (got " + std::to_string(w) + "x" +
                       std::to_string(h) + ")");
    if (x < 0 || y < 0 || x + w > sw || y + h > sh)
        throw DspError("crop (" + std::to_string(x) + "," + std::to_string(y) + "," +
                       std::to_string(w) + "x" + std::to_string(h) + ") outside source " +
                       std::to_string(sw) + "x" + std::to_string(sh));
    if (fmt == "nv12" && ((x | y | w | h | dw | dh) & 1))
        throw DspError("nv12 crop/dest coords and sizes must be even");
    if (!(dsp_wire::kMinDim <= dw && dw <= dsp_wire::kMaxDim &&
          dsp_wire::kMinDim <= dh && dh <= dsp_wire::kMaxDim))
        throw DspError("destination dims " + std::to_string(dw) + "x" + std::to_string(dh) +
                       " outside daemon range [" + std::to_string(dsp_wire::kMinDim) + ", " +
                       std::to_string(dsp_wire::kMaxDim) + "]");
    return CropRect{x, y, w, h, dw, dh};
}

int cv_interp(const std::string& interpolation) {
    if (interpolation == "nearest") return cv::INTER_NEAREST;
    if (interpolation == "bilinear") return cv::INTER_LINEAR;
    if (interpolation == "area") return cv::INTER_AREA;
    if (interpolation == "bicubic") return cv::INTER_CUBIC;
    throw DspError("unsupported interpolation '" + interpolation + "'");
}

cv::Mat cpu_resize(const cv::Mat& src, const std::string& fmt, int dw, int dh,
                   const std::string& scaling, const std::string& interpolation) {
    const cv::Size dims = src_dims(src, fmt);
    const int sw = dims.width, sh = dims.height;
    const int interp = cv_interp(interpolation);

    if (scaling == "letterbox" || scaling == "letterbox_middle" ||
        scaling == "letterbox_up_left") {
        const double scale =
            std::min(static_cast<double>(dw) / sw, static_cast<double>(dh) / sh);
        int cw = std::max(1, half_even(sw * scale));
        int ch = std::max(1, half_even(sh * scale));
        if (fmt == "nv12") {
            cw &= ~1;
            ch &= ~1;
        }
        cv::Mat content = cpu_resize(src, fmt, cw, ch, "stretch", interpolation);
        int ox = scaling == "letterbox_up_left" ? 0 : (dw - cw) / 2;
        int oy = scaling == "letterbox_up_left" ? 0 : (dh - ch) / 2;
        if (fmt == "nv12") {  // keep UV half-sample alignment
            ox &= ~1;
            oy &= ~1;
        }
        return place(content, fmt, dw, dh, ox, oy);
    }
    if (scaling == "scale_crop") {
        const double scale =
            std::max(static_cast<double>(dw) / sw, static_cast<double>(dh) / sh);
        int cw = std::max(1, half_even(sw * scale));
        int ch = std::max(1, half_even(sh * scale));
        if (fmt == "nv12") {
            cw &= ~1;
            ch &= ~1;
        }
        cv::Mat big = cpu_resize(src, fmt, cw, ch, "stretch", interpolation);
        return cpu_crop(big, fmt, (cw - dw) / 2 & ~1, (ch - dh) / 2 & ~1, dw, dh);
    }
    // stretch
    if (fmt == "nv12") {
        cv::Mat y = resize_plane(src.rowRange(0, sh), dh, dw, interp);
        // UV rows are interleaved (U,V) pairs: nearest-sample PAIRS so they
        // stay paired across width changes. python 0.8.1 resizes the byte
        // plane instead, which un-pairs chroma when dw != sw (deliberate
        // divergence — see header note).
        cv::Mat pairs = src.rowRange(sh, src.rows).reshape(2, src.rows - sh);
        cv::Mat uv_pairs = resize_plane(pairs, dh / 2, dw / 2, cv::INTER_NEAREST);
        cv::Mat out(dh * 3 / 2, dw, CV_8UC1);
        y.copyTo(out.rowRange(0, dh));
        uv_pairs.reshape(1, dh / 2).copyTo(out.rowRange(dh, dh * 3 / 2));
        return out;
    }
    return resize_plane(src, dh, dw, interp);
}

cv::Mat place(const cv::Mat& content, const std::string& fmt, int dw, int dh, int ox,
              int oy) {
    if (fmt == "nv12") {
        const int ch = content.rows * 2 / 3;
        const int cw = content.cols;
        cv::Mat canvas(dh * 3 / 2, dw, CV_8UC1, cv::Scalar(0));
        canvas.rowRange(dh, dh * 3 / 2).setTo(128);  // neutral chroma
        content.rowRange(0, ch)
            .colRange(0, cw)
            .copyTo(canvas.rowRange(oy, oy + ch).colRange(ox, ox + cw));
        // chroma rows: content pairs land at chroma row offset oy/2, byte
        // range [ox, ox+cw) — a chroma row is w bytes (w/2 pairs). python
        // 0.8.1 slices ox//2:(ox+cw)//2, half the bytes, and throws a numpy
        // broadcast ValueError for every nv12 letterbox (deliberate
        // divergence — see header note).
        const int c_y0 = dh + oy / 2, c_y1 = dh + (oy + ch) / 2;
        content.rowRange(ch, content.rows)
            .copyTo(canvas.rowRange(c_y0, c_y1).colRange(ox, ox + cw));
        return canvas;
    }
    cv::Mat canvas(dh, dw, fmt == "rgb24" ? CV_8UC3 : CV_8UC1, cv::Scalar(0, 0, 0));
    content.copyTo(canvas.rowRange(oy, oy + content.rows).colRange(ox, ox + content.cols));
    return canvas;
}

cv::Mat cpu_crop(const cv::Mat& src, const std::string& fmt, int x, int y, int w, int h) {
    if (fmt == "nv12") {
        const int sh = src.rows * 2 / 3;
        // interleaved UV: a w-pixel crop spans w bytes of chroma rows
        cv::Mat out(h + h / 2, w, CV_8UC1);
        src(cv::Rect(x, y, w, h)).copyTo(out.rowRange(0, h));
        const int c_rows = (y + h) / 2 - y / 2;
        src(cv::Rect(x, sh + y / 2, w, c_rows)).copyTo(out.rowRange(h, h + c_rows));
        return out;
    }
    return src(cv::Rect(x, y, w, h)).clone();
}

cv::Mat cpu_crop_resize(const cv::Mat& src, const std::string& fmt, const CropRect& rect) {
    cv::Mat out = cpu_crop(src, fmt, rect.x, rect.y, rect.w, rect.h);
    if (rect.dw != rect.w || rect.dh != rect.h)
        out = cpu_resize(out, fmt, rect.dw, rect.dh, "stretch", "bilinear");
    return out;
}

cv::Mat cpu_convert(const cv::Mat& src, const std::string& fmt, const std::string& dst_fmt) {
    if (fmt == dst_fmt)
        throw DspError("convert needs differing formats (dimensions stay equal)");
    if (fmt == "rgb24") {
        if (dst_fmt == "nv12") return rgb_to_nv12(src);
        // gray8: BT.601 luma, studio swing — the same coefficients the
        // numpy leg uses (16 + (65.481R+128.553G+24.966B)/255)
        cv::Mat out(src.rows, src.cols, CV_8UC1);
        for (int r = 0; r < src.rows; ++r) {
            const cv::Vec3b* in = src.ptr<cv::Vec3b>(r);
            std::uint8_t* o = out.ptr<std::uint8_t>(r);
            for (int c = 0; c < src.cols; ++c) {
                const double y = 16.0 + (65.481 * in[c][0] + 128.553 * in[c][1] +
                                         24.966 * in[c][2]) / 255.0;
                o[c] = static_cast<std::uint8_t>(
                    std::min(255.0, std::max(0.0, static_cast<double>(half_even(y)))));
            }
        }
        return out;
    }
    if (fmt == "nv12") {
        const int h = src.rows * 2 / 3;
        if (dst_fmt == "rgb24") return nv12_to_rgb(src, src.cols, h);
        return src.rowRange(0, h).clone();  // gray8: luma plane verbatim
    }
    // gray8 source
    if (dst_fmt == "rgb24") {
        cv::Mat out;
        std::vector<cv::Mat> ch{src, src, src};
        cv::merge(ch, out);
        return out;
    }
    cv::Mat out(src.rows * 3 / 2, src.cols, CV_8UC1);
    src.copyTo(out.rowRange(0, src.rows));
    out.rowRange(src.rows, src.rows * 3 / 2).setTo(128);  // neutral chroma
    return out;
}

cv::Mat cpu_blend(const cv::Mat& base, const std::string& fmt,
                  const std::vector<RgbaOverlay>& overlays) {
    cv::Mat work = (fmt == "rgb24") ? base.clone() : cpu_convert(base, fmt, "rgb24").clone();
    for (const RgbaOverlay& ov : overlays) {
        const int h = ov.rgba.rows, w = ov.rgba.cols;
        if (ov.y >= work.rows || ov.x >= work.cols || ov.y < 0 || ov.x < 0) continue;
        const int rh = std::min(h, work.rows - ov.y);  // clip overhang
        const int rw = std::min(w, work.cols - ov.x);
        for (int r = 0; r < rh; ++r) {
            const cv::Vec4b* src = ov.rgba.ptr<cv::Vec4b>(r);
            cv::Vec3b* dst = work.ptr<cv::Vec3b>(ov.y + r);
            for (int c = 0; c < rw; ++c) {
                const std::uint32_t a = src[c][3];
                for (int k = 0; k < 3; ++k) {
                    const std::uint32_t bg = dst[ov.x + c][k];
                    dst[ov.x + c][k] =
                        static_cast<std::uint8_t>((a * src[c][k] + (255 - a) * bg) / 255);
                }
            }
        }
    }
    return fmt == "rgb24" ? work : cpu_convert(work, "rgb24", fmt);
}

}  // namespace neoruntime_ipc_sdk::detail::dsp_format
