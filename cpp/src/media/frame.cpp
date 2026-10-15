// frame.cpp — Frame/FrameHandle implementation. See frame.hpp.
//
// Port of frame.py. Geometry comes from detail/geometry.hpp — the single
// placement truth shared with the DSP resize fast path and preprocessing.
#include "neoruntime_ipc_sdk/accel.hpp"
#include "neoruntime_ipc_sdk/dsp.hpp"
#include "neoruntime_ipc_sdk/frame.hpp"

#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <utility>

#include <fcntl.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include "detail/frame_pixels.hpp"
#include "detail/geometry.hpp"
#include "detail/jpeg_encode.hpp"

namespace neoruntime_ipc_sdk {

namespace {

// DMA_BUF_IOCTL_SYNC (linux/dma-buf.h): _IOW('b', 0, u64) on 64-bit.
constexpr unsigned long kDmaBufIoctlSync = 0x40086200;
constexpr std::uint64_t kDmaBufSyncRead = 1 << 0;
constexpr std::uint64_t kDmaBufSyncStart = 1 << 2;
constexpr std::uint64_t kDmaBufSyncEnd = 2 << 2;

// Fence CPU access to a dma-buf (HAL-3 discipline). Device-written dma-bufs
// must be fenced READ|START before a CPU read and READ|END after it, or the
// read can serve stale cache lines. On non-dma-buf fds (memfd in tests,
// plain anon memory) the ioctl raises ENOTTY — nothing to fence, so errors
// are swallowed.
void dma_buf_sync(int fd, std::uint64_t flags) noexcept {
    ::ioctl(fd, kDmaBufIoctlSync, &flags);
}

// cv2-backed resize (Python _resize_array): INTER_AREA down / INTER_LINEAR
// up; no numpy fallback leg — OpenCV is a hard dependency of the C++ SDK.
cv::Mat resize_array(const cv::Mat& img, int width, int height) {
    if (img.rows == height && img.cols == width) return img;
    const int interp = (height < img.rows && width < img.cols)
                           ? cv::INTER_AREA
                           : cv::INTER_LINEAR;
    cv::Mat out;
    cv::resize(img, out, cv::Size(width, height), 0, 0, interp);
    return out;
}

}  // namespace

namespace detail {

// Map a retained frame's planes and return them as one flat copy (fenced per
// HAL-3; plane sizes from the protocol, mapping sizes from fstat). Shared
// with the FdMediaClient copy path via frame_pixels.hpp.
std::vector<std::uint8_t> materialize_frame_handle(const FrameHandle& h) {
    std::vector<std::uint8_t> raw;
    for (std::size_t i = 0; i < h.fds.size(); ++i) {
        const std::uint32_t size = i < h.plane_sizes.size() ? h.plane_sizes[i] : 0;
        const int fd = h.fds[i];
        dma_buf_sync(fd, kDmaBufSyncRead | kDmaBufSyncStart);

        struct stat st{};
        void* mapped = MAP_FAILED;
        if (::fstat(fd, &st) == 0 && st.st_size > 0) {
            mapped = ::mmap(nullptr, static_cast<std::size_t>(st.st_size),
                            PROT_READ, MAP_SHARED, fd, 0);
        }
        if (mapped == MAP_FAILED) {
            dma_buf_sync(fd, kDmaBufSyncRead | kDmaBufSyncEnd);
            throw std::runtime_error("FrameHandle: cannot map dma-buf plane " +
                                     std::to_string(i));
        }
        auto base = static_cast<const std::uint8_t*>(mapped);
        std::size_t copy_bytes = std::min<std::size_t>(size, st.st_size);
        raw.insert(raw.end(), base, base + copy_bytes);
        ::munmap(mapped, static_cast<std::size_t>(st.st_size));
        dma_buf_sync(fd, kDmaBufSyncRead | kDmaBufSyncEnd);
    }
    if (raw.empty()) throw std::runtime_error("FrameHandle: no planes to map");
    return raw;
}

// Reshape a flat frame buffer into its per-format image layout.
cv::Mat decode_frame_raw(const std::vector<std::uint8_t>& raw, int w, int h,
                         const std::string& fmt) {
    auto* p = const_cast<std::uint8_t*>(raw.data());
    if (fmt == "NV12" || fmt == "NV21") {
        return cv::Mat(h * 3 / 2, w, CV_8UC1, p).clone();
    }
    if (fmt == "RGB" || fmt == "BGR") {
        return cv::Mat(h, w, CV_8UC3, p).clone();
    }
    if (fmt == "RGBA" || fmt == "BGRA") {
        return cv::Mat(h, w, CV_8UC4, p).clone();
    }
    if (fmt == "GRAY8") {
        return cv::Mat(h, w, CV_8UC1, p).clone();
    }
    if (fmt == "YUYV") {
        return cv::Mat(h, w, CV_8UC2, p).clone();
    }
    return cv::Mat(h, w, CV_8UC3, p).clone();
}

}  // namespace detail

// ---- format classification --------------------------------------------------
bool is_packed_format(std::string_view format) {
    return format == "RGB" || format == "BGR" || format == "RGBA" ||
           format == "BGRA" || format == "GRAY8";
}

bool is_yuv_format(std::string_view format) {
    return format == "NV12" || format == "NV21";
}

bool is_dsp_resize_format(std::string_view format) {
    return format == "NV12" || format == "RGB" || format == "BGR" ||
           format == "GRAY8";
}

// ---- transform composition ---------------------------------------------------
AffineTransform compose_transform(const AffineTransform& prev,
                                  const AffineTransform& cur) {
    AffineTransform composed = cur;
    composed.src_width = prev.src_width;
    composed.src_height = prev.src_height;
    composed.scale_x = prev.scale_x * cur.scale_x;
    composed.scale_y = prev.scale_y * cur.scale_y;
    composed.origin_x = prev.origin_x * cur.scale_x + cur.origin_x;
    composed.origin_y = prev.origin_y * cur.scale_y + cur.origin_y;
    return composed;
}

namespace {

// Set/chain the affine transform on a copy of the parent metadata (Python
// _frame_transform_metadata).
std::optional<AffineTransform> chained_transform(
    const std::optional<AffineTransform>& parent,
    const AffineTransform& transform) {
    if (parent.has_value()) return compose_transform(*parent, transform);
    return transform;
}

}  // namespace

// ---- FrameHandle --------------------------------------------------------------
FrameHandle::FrameHandle(std::vector<int> fds_in,
                         std::array<std::uint32_t, 3> strides_in,
                         std::array<std::uint32_t, 3> plane_sizes_in,
                         std::uint64_t frame_id_in,
                         ReleaseFn on_release_in,
                         int width_in,
                         int height_in,
                         std::string format_in,
                         std::uint32_t flags_in)
    : fds(std::move(fds_in)),
      strides(strides_in),
      plane_sizes(plane_sizes_in),
      frame_id(frame_id_in),
      width(width_in),
      height(height_in),
      format(std::move(format_in)),
      flags(flags_in),
      on_release_(std::move(on_release_in)) {}

FrameHandle::~FrameHandle() { close(); }

FrameHandle::FrameHandle(FrameHandle&&) noexcept = default;
FrameHandle& FrameHandle::operator=(FrameHandle&&) noexcept = default;

void FrameHandle::close() noexcept {
    if (closed_) return;
    closed_ = true;
    for (int fd : fds) {
        ::close(fd);
    }
    if (on_release_) {
        try {
            on_release_(*this);
        } catch (...) {
            // Release callback failure must not propagate (Python logs and
            // moves on; there is no logger here, so swallow).
        }
    }
}

// ---- Frame --------------------------------------------------------------------
cv::Mat Frame::to_array() {
    if (!image.empty()) return image;
    if (!handle) {
        throw std::invalid_argument("Frame has no image data and no retained fd");
    }
    image = detail::decode_frame_raw(detail::materialize_frame_handle(*handle),
                                     width, height, format);
    return image;
}

cv::Mat Frame::data() {
    cv::Mat arr = to_array();
    if (!arr.isContinuous()) arr = arr.clone();
    return arr.reshape(1, 1);
}

void Frame::release() noexcept {
    if (handle) handle->close();
}

cv::Mat Frame::to_rgb() {
    cv::Mat arr = to_array();
    if (format == "RGB") {
        return arr;
    }
    cv::Mat out;
    if (format == "BGR") {
        cv::cvtColor(arr, out, cv::COLOR_BGR2RGB);
        return out;
    }
    if (format == "NV12") {
        cv::cvtColor(arr, out, cv::COLOR_YUV2RGB_NV12);
        return out;
    }
    if (format == "GRAY8") {
        cv::cvtColor(arr, out, cv::COLOR_GRAY2RGB);
        return out;
    }
    throw std::invalid_argument("Unsupported format: " + format);
}

namespace {

// Pixels for the const paths (crop/resize/to_jpeg_bytes/save): the cached
// image when present, otherwise a fresh mapping of a retained handle (NOT
// cached — `this` is const; repeated calls on an unmaterialized keep-fd
// frame re-map).
cv::Mat source_pixels(const Frame& f) {
    if (!f.image.empty()) return f.image;
    if (!f.handle) {
        throw std::invalid_argument("Frame has no image data and no retained fd");
    }
    return detail::decode_frame_raw(detail::materialize_frame_handle(*f.handle),
                                    f.width, f.height, f.format);
}

}  // namespace

Frame Frame::crop(int x, int y, int crop_width, int crop_height) const {
    if (crop_width <= 0 || crop_height <= 0) {
        throw std::invalid_argument("crop width/height must be positive");
    }
    if (x < 0 || y < 0 || x + crop_width > width || y + crop_height > height) {
        throw std::invalid_argument("crop (" + std::to_string(x) + "," +
                                    std::to_string(y) + "," +
                                    std::to_string(crop_width) + "x" +
                                    std::to_string(crop_height) +
                                    ") out of bounds for " + std::to_string(width) +
                                    "x" + std::to_string(height) + " frame");
    }
    cv::Mat sub;
    if (is_packed_format(format)) {
        sub = source_pixels(*this)(cv::Rect(x, y, crop_width, crop_height)).clone();
    } else if (is_yuv_format(format)) {
        if (x % 2 || y % 2 || crop_width % 2 || crop_height % 2) {
            throw std::invalid_argument(format +
                                        " crop requires even x, y, width, height");
        }
        cv::Mat arr = source_pixels(*this);
        cv::Mat y_plane = arr(cv::Rect(x, y, crop_width, crop_height));
        // Interleaved chroma: byte columns x..x+crop_width of rows y/2.. on
        // the UV plane (each chroma pair spans 2 bytes).
        cv::Mat uv_plane =
            arr(cv::Rect(x, height + y / 2, crop_width, crop_height / 2));
        cv::vconcat(y_plane, uv_plane, sub);
    } else {
        throw std::invalid_argument("crop not supported for format: " + format);
    }

    AffineTransform step;
    step.op = "crop";
    step.src_width = width;
    step.src_height = height;
    step.dst_width = crop_width;
    step.dst_height = crop_height;
    step.mode = "crop";
    step.scale_x = 1.0;
    step.scale_y = 1.0;
    step.origin_x = -static_cast<double>(x);
    step.origin_y = -static_cast<double>(y);

    Frame out;
    out.sequence = sequence;
    out.timestamp_ns = timestamp_ns;
    out.width = crop_width;
    out.height = crop_height;
    out.format = format;
    out.image = std::move(sub);
    out.metadata = metadata;
    out.flags = flags;
    out.transform = chained_transform(transform, step);
    return out;
}

namespace {

// Packed (RGB/BGR/RGBA/BGRA/GRAY8) resize legs.
cv::Mat resize_packed(const cv::Mat& src, const std::string& fmt, int dw, int dh,
                      std::string_view mode, int pad) {
    if (mode == "stretch") return resize_array(src, dw, dh);
    const detail::ResizeBox box =
        detail::resize_geometry(src.cols, src.rows, dw, dh, mode, /*yuv=*/false);
    if (mode == "letterbox") {
        cv::Mat content = resize_array(src, box.rw, box.rh);
        cv::Mat canvas;
        if (fmt == "RGBA" || fmt == "BGRA") {
            canvas = cv::Mat(dh, dw, CV_8UC4, cv::Scalar(pad, pad, pad, 255));
        } else if (src.channels() == 3) {
            canvas = cv::Mat(dh, dw, CV_8UC3, cv::Scalar(pad, pad, pad));
        } else {
            canvas = cv::Mat(dh, dw, CV_8UC1, cv::Scalar(pad));
        }
        content.copyTo(canvas(cv::Rect(box.ox, box.oy, box.rw, box.rh)));
        return canvas;
    }
    // mode == "crop": scale to cover, then center-crop
    cv::Mat tmp = resize_array(src, box.rw, box.rh);
    return tmp(cv::Rect(box.ox, box.oy, dw, dh)).clone();
}

// Interleaved-UV chroma resize: treat the UV plane as (h/2, w/2, 2) so U and
// V stay on separate channels, resize, flatten back to (h/2, w).
cv::Mat uv_resize(const cv::Mat& uv, int w, int h) {
    cv::Mat paired = uv.reshape(2, uv.rows);
    cv::Mat out = resize_array(paired, w / 2, h / 2);
    return out.reshape(1, out.rows);
}

// NV12/NV21 resize legs.
cv::Mat resize_yuv(const cv::Mat& image, int sw, int sh, int dw, int dh,
                   std::string_view mode, int pad,
                   const detail::ResizeBox& box) {
    cv::Mat y_plane = image.rowRange(0, sh);
    cv::Mat uv_plane = image.rowRange(sh, image.rows).clone();
    if (uv_plane.rows * 2 != sh || uv_plane.cols != sw) {
        throw std::invalid_argument(
            "NV12/NV21 buffer does not match the frame geometry");
    }
    if (mode == "stretch") {
        cv::Mat y = resize_array(y_plane, dw, dh);
        cv::Mat uv = uv_resize(uv_plane, dw, dh);
        cv::Mat out;
        cv::vconcat(y, uv, out);
        return out;
    }
    cv::Mat content_y = resize_array(y_plane, box.rw, box.rh);
    cv::Mat content_uv = uv_resize(uv_plane, box.rw, box.rh);
    if (mode == "letterbox") {
        cv::Mat canvas_y(dh, dw, CV_8UC1, cv::Scalar(pad));
        cv::Mat canvas_uv(dh / 2, dw, CV_8UC1, cv::Scalar(128));
        content_y.copyTo(canvas_y(cv::Rect(box.ox, box.oy, box.rw, box.rh)));
        content_uv.copyTo(
            canvas_uv(cv::Rect(box.ox, box.oy / 2, box.rw, box.rh / 2)));
        cv::Mat out;
        cv::vconcat(canvas_y, canvas_uv, out);
        return out;
    }
    // mode == "crop": scale to cover, then center-crop both planes
    cv::Mat y = content_y(cv::Rect(box.ox, box.oy, dw, dh));
    cv::Mat uv = content_uv(cv::Rect(box.ox, box.oy / 2, dw, dh / 2));
    cv::Mat out;
    cv::vconcat(y, uv, out);
    return out;
}

// python _hw_resize: zero-copy DSP scale for keep-fd frames. Returns an
// empty Mat meaning "use the CPU path" (no handle, closed handle, format
// the DSP cannot take, service down, software-only policy).
//
// Every mode scales on the DSP into the exact box the CPU path computes,
// then pads/crops on the CPU, so both paths agree on content placement:
// the daemon's own letterbox pads Y=U=V=0 (green in YUV) and its
// SCALE_AND_CROP disagreed with CPU placement by ~21 luma levels on
// device, so neither is used here.
cv::Mat hw_resize(const Frame& f, int dw, int dh, std::string_view mode, int pad) {
    if (!f.handle || f.handle->closed()) return {};
    if (!is_dsp_resize_format(f.format)) return {};
    const bool yuv = is_yuv_format(f.format);
    const detail::ResizeBox box =
        detail::resize_geometry(f.width, f.height, dw, dh, mode, yuv);

    AccelRouter& router = get_default_router();
    bool wants_hw = true;
    try {
        wants_hw = router.route("resize_nv12").backend == "hardware";
    } catch (const std::out_of_range&) {
        wants_hw = true;  // unregistered op: keep the historical attempt
    }
    if (!wants_hw) return {};

    cv::Mat content;
    try {
        // fmt "" (frame-inferred), scaling "stretch" — the daemon resize
        // fills the fitted box; letterboxing/cropping stays local so the
        // pad values match the CPU path exactly
        RouterDspOpts opts;
        content = std::any_cast<cv::Mat>(shared_dsp_call(
            "resize_hw",
            {std::any(f), std::any(box.rw), std::any(box.rh), std::any(opts)}));
    } catch (const DspError& exc) {
        if (router.policy() == RoutePolicy::HardwareOnly)
            throw HardwareUnavailable(std::string("resize: DSP fast path failed (") +
                                      exc.what() +
                                      ") and the policy is hardware-only — refusing the "
                                      "silent CPU fallback");
        // PREFER_HARDWARE degrades, but honestly: the router's
        // health()/counters must see the fast path it did not take
        router.note_degradation("resize_nv12",
                                std::string("DSP resize fast path unavailable: ") + exc.what());
        return {};
    }

    if (mode == "stretch") return content;
    if (!yuv) {
        if (mode == "crop")
            return content(cv::Rect(box.ox, box.oy, dw, dh)).clone();
        cv::Mat canvas;
        if (content.channels() == 4)
            canvas = cv::Mat(dh, dw, CV_8UC4, cv::Scalar(pad, pad, pad, 255));
        else if (content.channels() == 3)
            canvas = cv::Mat(dh, dw, CV_8UC3, cv::Scalar(pad, pad, pad));
        else
            canvas = cv::Mat(dh, dw, CV_8UC1, cv::Scalar(pad));
        content.copyTo(canvas(cv::Rect(box.ox, box.oy, box.rw, box.rh)));
        return canvas;
    }
    // NV12/NV21 letterbox/crop: content is (rh*3/2, rw)
    const cv::Mat y_part = content(cv::Rect(0, 0, box.rw, box.rh));
    const cv::Mat uv_part = content(cv::Rect(0, box.rh, box.rw, box.rh / 2));
    if (mode == "crop") {
        cv::Mat out;
        cv::vconcat(y_part(cv::Rect(box.ox, box.oy, dw, dh)),
                    uv_part(cv::Rect(box.ox, box.oy / 2, dw, dh / 2)), out);
        return out;
    }
    cv::Mat canvas_y(dh, dw, CV_8UC1, cv::Scalar(pad));
    cv::Mat canvas_uv(dh / 2, dw, CV_8UC1, cv::Scalar(128));
    y_part.copyTo(canvas_y(cv::Rect(box.ox, box.oy, box.rw, box.rh)));
    uv_part.copyTo(canvas_uv(cv::Rect(box.ox, box.oy / 2, box.rw, box.rh / 2)));
    cv::Mat out;
    cv::vconcat(canvas_y, canvas_uv, out);
    return out;
}

}  // namespace

Frame Frame::resize(int resize_width, int resize_height, std::string_view mode,
                    int pad_value) const {
    if (resize_width <= 0 || resize_height <= 0) {
        throw std::invalid_argument("resize width/height must be positive");
    }
    if (mode != "letterbox" && mode != "stretch" && mode != "crop") {
        throw std::invalid_argument("unsupported resize mode: " + std::string(mode));
    }
    if (is_yuv_format(format) && (resize_width % 2 || resize_height % 2)) {
        throw std::invalid_argument(format +
                                    " resize requires even width/height");
    }
    const bool yuv = is_yuv_format(format);
    const detail::ResizeBox box = detail::resize_geometry(
        width, height, resize_width, resize_height, mode, yuv);

    cv::Mat out_image = hw_resize(*this, resize_width, resize_height, mode, pad_value);
    if (out_image.empty()) {
        // CPU path: materialize a retained fd before slicing planes
        cv::Mat src = source_pixels(*this);
        if (yuv) {
            out_image = resize_yuv(src, width, height, resize_width, resize_height,
                                   mode, pad_value, box);
        } else if (is_packed_format(format)) {
            out_image = resize_packed(src, format, resize_width, resize_height, mode,
                                      pad_value);
        } else {
            throw std::invalid_argument("resize not supported for format: " + format);
        }
    }

    AffineTransform step;
    step.op = "resize";
    step.src_width = width;
    step.src_height = height;
    step.dst_width = resize_width;
    step.dst_height = resize_height;
    step.mode = std::string(mode);
    if (mode == "stretch") {
        step.scale_x = static_cast<double>(resize_width) / width;
        step.scale_y = static_cast<double>(resize_height) / height;
        step.origin_x = 0.0;
        step.origin_y = 0.0;
    } else if (mode == "letterbox") {
        step.scale_x = static_cast<double>(box.rw) / width;
        step.scale_y = static_cast<double>(box.rh) / height;
        step.origin_x = box.ox;
        step.origin_y = box.oy;
    } else {  // crop: dst pixel (x, y) shows scaled content at (x + ox, y + oy)
        step.scale_x = static_cast<double>(box.rw) / width;
        step.scale_y = static_cast<double>(box.rh) / height;
        step.origin_x = -static_cast<double>(box.ox);
        step.origin_y = -static_cast<double>(box.oy);
    }

    Frame out;
    out.sequence = sequence;
    out.timestamp_ns = timestamp_ns;
    out.width = resize_width;
    out.height = resize_height;
    out.format = format;
    out.image = std::move(out_image);
    out.metadata = metadata;
    out.flags = flags;
    out.transform = chained_transform(transform, step);
    return out;
}

namespace detail {

// cv2-backed JPEG encode of an RGB image (Python _encode_jpeg, cv2 leg).
// Declared in detail/jpeg_encode.hpp — shared with the accel router's
// "encode_jpeg" software leg.
std::vector<std::uint8_t> encode_jpeg_rgb(const cv::Mat& rgb, int quality) {
    cv::Mat bgr;
    cv::cvtColor(rgb, bgr, cv::COLOR_RGB2BGR);
    std::vector<uchar> buf;
    std::vector<int> params{cv::IMWRITE_JPEG_QUALITY, quality};
    if (!cv::imencode(".jpg", bgr, buf, params)) {
        throw std::runtime_error("cv::imencode failed to encode JPEG");
    }
    return std::vector<std::uint8_t>(buf.begin(), buf.end());
}

}  // namespace detail

namespace {

std::string lower_tail(const std::string& path, std::size_t n) {
    if (path.size() < n) return {};
    std::string tail = path.substr(path.size() - n);
    for (auto& c : tail) {
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }
    return tail;
}

}  // namespace

std::vector<std::uint8_t> Frame::to_jpeg_bytes(int quality) const {
    if (!handle || handle->closed()) {
        // no live handle -> CPU encode outright (the daemon encoder's win
        // is the zero-copy import, not raw speed; S-3 record)
        Frame copy = *this;  // to_rgb may materialize a retained handle
        return detail::encode_jpeg_rgb(copy.to_rgb(), quality);
    }
    // live keep-fd handle: the router's zero-copy EncodeImage leg (the
    // daemon imports the buffer — no RGB materialization) with its CPU
    // degrade when the daemon cannot serve it
    return std::any_cast<std::vector<std::uint8_t>>(
        get_default_router().run("encode_jpeg", {std::any(*this), std::any(quality)}));
}

void Frame::save(const std::string& path) const {
    std::string tail4 = lower_tail(path, 4);
    if (tail4 == ".jpg" || lower_tail(path, 5) == ".jpeg") {
        std::vector<std::uint8_t> jpeg = to_jpeg_bytes();
        std::FILE* f = std::fopen(path.c_str(), "wb");
        if (!f) {
            throw std::runtime_error("Frame::save(): cannot open " + path);
        }
        std::size_t n = std::fwrite(jpeg.data(), 1, jpeg.size(), f);
        std::fclose(f);
        if (n != jpeg.size()) {
            throw std::runtime_error("Frame::save(): short write to " + path);
        }
        return;
    }
    Frame copy = *this;
    cv::Mat bgr;
    cv::cvtColor(copy.to_rgb(), bgr, cv::COLOR_RGB2BGR);
    if (!cv::imwrite(path, bgr)) {
        throw std::runtime_error("Frame::save(): cv::imwrite failed for " + path);
    }
}

}  // namespace neoruntime_ipc_sdk
