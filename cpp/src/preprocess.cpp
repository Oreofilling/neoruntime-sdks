// preprocess.cpp — port of preprocess.py (0.8.1). Validation order and
// error strings match the python f-strings byte for byte; the transform
// bookkeeping rides Frame's strong-typed AffineTransform instead of the
// python metadata dict.
#include "neoruntime_ipc_sdk/preprocess.hpp"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cctype>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "neoruntime_ipc_sdk/accel.hpp"
#include "neoruntime_ipc_sdk/dsp.hpp"

namespace neoruntime_ipc_sdk {

namespace {

std::string to_upper(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned char c) { return static_cast<char>(std::toupper(c)); });
    return s;
}

std::string to_lower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return s;
}

// python f"{shape}" of a list: "[640, 640, 3]"
std::string shape_list_repr(const std::vector<int>& shape) {
    std::string out = "[";
    for (std::size_t i = 0; i < shape.size(); ++i) {
        if (i > 0) out += ", ";
        out += std::to_string(shape[i]);
    }
    return out + "]";
}

// python f"{arr.shape}" for a cv::Mat: dims/channels as a list repr.
std::string shape_text(const cv::Mat& arr) {
    std::vector<int> shape;
    for (int i = 0; i < arr.dims; ++i) shape.push_back(arr.size.p[i]);
    return shape_list_repr(shape);
}

// python _infer_source_format: hint bypasses every check (already
// validated at construction); the ndarray shape rules map to cv::Mat as
// dims/channels.
std::string infer_source_format(const cv::Mat& arr, const std::optional<std::string>& hint) {
    if (hint.has_value()) return *hint;
    if (arr.dims == 2) {
        const int ch = arr.channels();
        if (ch == 3) return "RGB";  // honest default for packed 3-channel input
        if (ch == 1)
            throw std::invalid_argument(
                "2D arrays are ambiguous (NV12 vs GRAY8) — pass "
                "source_format='NV12' or 'GRAY8'");
        if (ch == 4)
            throw std::invalid_argument(
                "4-channel (RGBA) arrays are not supported — strip alpha "
                "first (e.g. arr[:, :, :3]) and pass the RGB array");
        throw std::invalid_argument("unsupported source array shape " + shape_text(arr));
    }
    if (arr.dims == 3) {
        const int ch = arr.size.p[2];
        if (ch == 3) return "RGB";
        if (ch == 1) return "GRAY8";
        if (ch == 4)
            throw std::invalid_argument(
                "4-channel (RGBA) arrays are not supported — strip alpha "
                "first (e.g. arr[:, :, :3]) and pass the RGB array");
    }
    throw std::invalid_argument("unsupported source array shape " + shape_text(arr));
}

// python _derive_layout_size. The TensorSpec layout string wins when
// present; otherwise channels are located by size (1/3/4) with HWC
// preferred.
std::pair<std::string, std::pair<int, int>> derive_layout_size(
    const std::vector<int>& shape, const std::string& layout_hint_raw) {
    if (shape.size() != 3)
        throw std::invalid_argument(
            "cannot derive spatial size from input shape " + shape_list_repr(shape) +
            " — expected [N,]H,W,C or [N,]C,H,W");
    const std::string layout_hint = to_upper(layout_hint_raw);
    const int a = shape[0];
    const int b = shape[1];
    const int c = shape[2];
    int h = 0, w = 0, ch = 0;
    std::string layout;
    if (layout_hint == "HWC" || layout_hint == "NHWC") {
        h = a; w = b; ch = c; layout = "NHWC";
    } else if (layout_hint == "CHW" || layout_hint == "NCHW") {
        ch = a; h = b; w = c; layout = "NCHW";
    } else if (c == 1 || c == 3 || c == 4) {
        h = a; w = b; ch = c; layout = "NHWC";
    } else if (a == 1 || a == 3 || a == 4) {
        ch = a; h = b; w = c; layout = "NCHW";
    } else {
        throw std::invalid_argument("ambiguous input shape " + shape_list_repr(shape) +
                                    " — no channels dimension");
    }
    if (ch != 1 && ch != 3 && ch != 4)
        throw std::invalid_argument("unsupported channel count " + std::to_string(ch) +
                                    " in input shape " + shape_list_repr(shape));
    return {layout, {w, h}};
}

// python (x - origin) / scale read off the strong-typed transform.
std::pair<int, int> transform_origin(const AffineTransform& t) {
    return {static_cast<int>(t.origin_x), static_cast<int>(t.origin_y)};
}

}  // namespace

// ---- PreprocessMeta -------------------------------------------------------------

std::pair<double, double> PreprocessMeta::to_source(double x, double y) const {
    return {(x - origin.first) / scale.first, (y - origin.second) / scale.second};
}

std::array<double, 4> PreprocessMeta::to_source_box(const std::array<double, 4>& box) const {
    const auto a = to_source(box[0], box[1]);
    const auto b = to_source(box[2], box[3]);
    return {a.first, a.second, b.first, b.second};
}

// ---- construction ----------------------------------------------------------------

bool is_supported_source_format(std::string_view format) {
    return format == "NV12" || format == "RGB" || format == "BGR" || format == "GRAY8";
}

Preprocessor::Preprocessor(std::optional<std::pair<int, int>> size_in, std::string color_in,
                           std::string resize_mode_in, int pad_value_in, bool normalize_in,
                           std::string layout_in, std::optional<std::string> source_format_in,
                           bool nv12_passthrough_in) {
    // Normalise first, validate after: "rgb"/"Nchw"/"nv12" are all fine.
    color_in = to_upper(std::move(color_in));
    layout_in = to_upper(std::move(layout_in));
    resize_mode_in = to_lower(std::move(resize_mode_in));
    if (source_format_in.has_value()) source_format_in = to_upper(*source_format_in);
    if (color_in != "RGB" && color_in != "BGR")
        throw std::invalid_argument("color must be 'RGB' or 'BGR', got '" + color_in + "'");
    if (resize_mode_in != "letterbox" && resize_mode_in != "stretch" &&
        resize_mode_in != "crop")
        throw std::invalid_argument("unsupported resize_mode: '" + resize_mode_in + "'");
    if (layout_in != "NHWC" && layout_in != "NCHW")
        throw std::invalid_argument("layout must be 'NHWC' or 'NCHW', got '" + layout_in + "'");
    if (source_format_in.has_value() && !is_supported_source_format(*source_format_in))
        throw std::invalid_argument("source_format must be one of ['BGR', 'GRAY8', 'NV12', "
                                    "'RGB'], got '" +
                                    *source_format_in + "'");
    if (nv12_passthrough_in) {
        // The daemon's DSP letterbox pads Y=U=V=0 (green in YUV, not
        // neutral) and its SCALE_AND_CROP placement disagrees with the CPU
        // path by ~21 luma levels — stretch is the only mode both legs
        // place identically, so passthrough refuses to guess at the others.
        if (resize_mode_in != "stretch")
            throw std::invalid_argument(
                "nv12_passthrough supports resize_mode='stretch' only (got '" +
                resize_mode_in +
                "'): the DSP letterbox pads green and its crop placement differs "
                "from the CPU path");
        if (normalize_in)
            throw std::invalid_argument(
                "nv12_passthrough cannot normalize — NV12 HEF models quantise "
                "internally; pass normalize=False");
        if (layout_in != "NHWC")
            throw std::invalid_argument(
                "nv12_passthrough emits packed NV12 planes (got layout='" + layout_in +
                "') — there is no channel dimension to transpose");
    }
    size = size_in;
    color = std::move(color_in);
    resize_mode = std::move(resize_mode_in);
    pad_value = pad_value_in;
    normalize = normalize_in;
    layout = std::move(layout_in);
    source_format = std::move(source_format_in);
    nv12_passthrough = nv12_passthrough_in;
}

Preprocessor Preprocessor::from_model(InferenceClient& client, const std::string& model_id) {
    return from_model(client, model_id, Overrides{});
}

Preprocessor Preprocessor::from_model(InferenceClient& client, const std::string& model_id,
                                      const Overrides& overrides) {
    std::optional<ModelInfo> info = client.get_model_info(model_id);
    if (!info.has_value())
        throw std::invalid_argument("model '" + model_id +
                                    "' is not registered with the service");
    if (info->inputs.empty())
        throw std::invalid_argument("model '" + model_id + "' exposes no input specs");

    const TensorSpec& inp = info->inputs.front();
    std::vector<int> shape = inp.shape;
    if (!shape.empty() && shape.front() == 1 && shape.size() >= 3)
        shape.erase(shape.begin());  // strip the batch dimension
    const auto [layout, size] = derive_layout_size(shape, inp.layout);
    const bool normalize = std::string(data_type_name(inp.dtype)).rfind("float", 0) == 0;

    // python kwargs start as the derived values; keyword overrides replace
    // any of them. The NV12 auto-routing only fires when neither
    // nv12_passthrough nor resize_mode was pinned.
    std::string resize_mode = "letterbox";
    bool nv12_passthrough = false;
    if (to_upper(inp.layout) == "NV12" && !normalize && !overrides.nv12_passthrough.has_value() &&
        !overrides.resize_mode.has_value()) {
        // NV12-layout model: the RGB path cannot serve it (3-channel
        // arrays fail the model's byte_size) — route through the
        // device-resident NV12 leg instead. Callers who pinned
        // resize_mode keep their choice (and the old behaviour).
        nv12_passthrough = true;
        resize_mode = "stretch";
    }
    if (overrides.resize_mode.has_value()) resize_mode = *overrides.resize_mode;
    if (overrides.nv12_passthrough.has_value()) nv12_passthrough = *overrides.nv12_passthrough;

    std::optional<std::pair<int, int>> out_size = size;
    std::string out_color = "RGB";
    int out_pad = 114;
    bool out_normalize = normalize;
    std::string out_layout = layout;
    std::optional<std::string> out_source_format;
    if (overrides.size.has_value()) out_size = overrides.size;
    if (overrides.color.has_value()) out_color = *overrides.color;
    if (overrides.pad_value.has_value()) out_pad = *overrides.pad_value;
    if (overrides.normalize.has_value()) out_normalize = *overrides.normalize;
    if (overrides.layout.has_value()) out_layout = *overrides.layout;
    if (overrides.source_format.has_value()) out_source_format = overrides.source_format;
    return Preprocessor(out_size, std::move(out_color), std::move(resize_mode), out_pad,
                        out_normalize, std::move(out_layout), std::move(out_source_format),
                        nv12_passthrough);
}

// ---- execution --------------------------------------------------------------------

Frame Preprocessor::as_frame(const Frame& source) const { return source; }

Frame Preprocessor::as_frame(const cv::Mat& source) const {
    const std::string fmt = infer_source_format(source, source_format);
    int height;
    int width;
    if (fmt == "NV12" || fmt == "NV21") {
        height = source.rows * 2 / 3;
        width = source.cols;
    } else {
        height = source.rows;
        width = source.cols;
    }
    Frame frame;
    frame.sequence = 0;
    frame.timestamp_ns = 0;
    frame.width = width;
    frame.height = height;
    frame.format = fmt;
    frame.image = source;
    return frame;
}

Preprocessor::Output Preprocessor::operator()(const Frame& source) const {
    return run(as_frame(source));
}

Preprocessor::Output Preprocessor::operator()(const cv::Mat& source) const {
    return run(as_frame(source));
}

Preprocessor::Output Preprocessor::run(const Frame& frame) const {
    if (!size.has_value())
        throw std::invalid_argument("no target size — construct via from_model() or pass size=");

    // python __call__: the NV12 device-resident leg bypasses the RGB path
    // entirely (frame stays packed NV12 through identity/resize).
    if (nv12_passthrough) return passthrough(frame);

    std::pair<int, int> original_size;
    std::pair<double, double> scale;
    std::pair<int, int> origin;
    Frame resized;
    if (std::make_pair(frame.width, frame.height) == *size) {
        resized = frame;
        // No further geometry — but a frame that was already cropped or
        // resized carries its transform; inherit it so postprocessing
        // still maps back to the ORIGINAL frame, not this intermediate.
        if (frame.transform.has_value()) {
            original_size = {frame.transform->src_width, frame.transform->src_height};
            scale = {frame.transform->scale_x, frame.transform->scale_y};
            origin = transform_origin(*frame.transform);
        } else {
            original_size = *size;
            scale = {1.0, 1.0};
            origin = {0, 0};
        }
    } else {
        resized = frame.resize(size->first, size->second, resize_mode, pad_value);
        const AffineTransform& t = *resized.transform;
        original_size = {t.src_width, t.src_height};
        scale = {t.scale_x, t.scale_y};
        origin = transform_origin(t);
    }

    cv::Mat arr = to_color(resized);
    if (normalize) {
        cv::Mat float_arr;
        arr.convertTo(float_arr, CV_32F, 1.0 / 255.0);
        arr = float_arr;
    }
    if (layout == "NCHW") {
        // HWC (h, w, c) -> CHW (c, h, w) as one continuous 3D Mat
        std::vector<cv::Mat> planes;
        cv::split(arr, planes);
        const int dims[] = {static_cast<int>(planes.size()), arr.rows, arr.cols};
        cv::Mat chw(3, dims, planes.front().type());
        for (std::size_t i = 0; i < planes.size(); ++i) {
            const int sub[] = {dims[1], dims[2]};
            cv::Mat view(2, sub, planes[i].type(), chw.ptr(static_cast<int>(i)));
            planes[i].copyTo(view);
        }
        arr = chw;
    }

    PreprocessMeta meta;
    meta.original_size = original_size;
    meta.input_size = *size;
    meta.scale = scale;
    meta.origin = origin;
    meta.color = color;
    meta.layout = layout;
    return {Tensor(arr), meta};
}

// ---- device-resident NV12 leg -------------------------------------------------------

Preprocessor::Output Preprocessor::passthrough(const Frame& frame) const {
    if (frame.format != "NV12")
        throw std::invalid_argument("nv12_passthrough needs NV12 sources, got '" +
                                    frame.format +
                                    "' — drop nv12_passthrough to convert through RGB");
    const int tw = size->first;
    const int th = size->second;
    const int sw = frame.width;
    const int sh = frame.height;

    if (sw == tw && sh == th) {
        // Identity: the source already IS the model input. A keep-fd frame
        // feeds infer's zero-copy path as-is; an array source must come
        // back as the array (a handle-less Frame is not a valid infer
        // input). Either way inherit any prior transform so postprocessing
        // still maps to the ORIGINAL frame.
        std::pair<int, int> original_size;
        std::pair<double, double> scale;
        std::pair<int, int> origin;
        if (frame.transform.has_value()) {
            original_size = {frame.transform->src_width, frame.transform->src_height};
            scale = {frame.transform->scale_x, frame.transform->scale_y};
            origin = transform_origin(*frame.transform);
        } else {
            original_size = {tw, th};
            scale = {1.0, 1.0};
            origin = {0, 0};
        }
        Tensor tensor = frame.handle
                            ? Tensor(frame)
                            : [&] {
                                  Frame copy = frame;  // to_array may cache pixels
                                  return Tensor(copy.to_array());
                              }();
        return {std::move(tensor), passthrough_meta(original_size, scale, origin)};
    }

    AccelRouter& router = get_default_router();
    bool wants_hw = true;
    try {
        wants_hw = router.route("resize_nv12").backend == "hardware";
    } catch (const std::out_of_range&) {
        wants_hw = true;  // unregistered op: keep the historical attempt
    }

    std::shared_ptr<DspBufferRef> ref;
    if (wants_hw) {
        try {
            // out_ref: the scaled NV12 lands in a daemon buffer and never
            // leaves the device — the ref rides the Tensor variant; infer
            // sends its buffer_id
            RouterDspOpts opts;  // scaling "stretch" default
            opts.fmt = "nv12";
            opts.cpu_fallback = false;
            opts.out_ref = true;
            std::any out = shared_dsp_call(
                "resize_hw", {std::any(frame), std::any(tw), std::any(th), std::any(opts)});
            if (auto* held = std::any_cast<std::shared_ptr<DspBufferRef>>(&out)) ref = *held;
        } catch (const DspError& exc) {
            if (router.policy() == RoutePolicy::HardwareOnly)
                throw HardwareUnavailable(
                    std::string("nv12_passthrough: DSP resize failed (") + exc.what() +
                    ") and the policy is hardware-only — refusing the silent CPU fallback");
            router.note_degradation(
                "resize_nv12", std::string("nv12_passthrough DSP leg unavailable: ") + exc.what());
        }
    }
    if (ref) {
        AffineTransform step;
        step.op = "resize";
        step.src_width = sw;
        step.src_height = sh;
        step.dst_width = tw;
        step.dst_height = th;
        step.mode = "stretch";
        step.scale_x = static_cast<double>(tw) / sw;
        step.scale_y = static_cast<double>(th) / sh;
        step.origin_x = 0.0;
        step.origin_y = 0.0;
        const AffineTransform transform =
            frame.transform.has_value() ? compose_transform(*frame.transform, step) : step;
        return {Tensor(std::move(*ref)),
                passthrough_meta({transform.src_width, transform.src_height},
                                 {transform.scale_x, transform.scale_y},
                                 transform_origin(transform))};
    }

    // CPU fallback: same stretch geometry, packed NV12 array out. (A
    // keep-fd frame still scales on the DSP inside Frame::resize — this
    // leg only gives up the device-resident output, not the hardware
    // scale.)
    Frame resized = frame.resize(tw, th, "stretch", 114);
    const AffineTransform& t = *resized.transform;
    Frame copy = resized;
    return {Tensor(copy.to_array()),
            passthrough_meta({t.src_width, t.src_height}, {t.scale_x, t.scale_y},
                             transform_origin(t))};
}

PreprocessMeta Preprocessor::passthrough_meta(std::pair<int, int> original_size,
                                              std::pair<double, double> scale,
                                              std::pair<int, int> origin) const {
    PreprocessMeta meta;
    meta.original_size = original_size;
    meta.input_size = *size;
    meta.scale = scale;
    meta.origin = origin;
    meta.color = "NV12";
    meta.layout = "NHWC";
    return meta;
}

// ---- colour conversion ---------------------------------------------------------------

cv::Mat Preprocessor::to_color(const Frame& frame) const {
    const std::string& fmt = frame.format;
    cv::Mat arr;
    bool want_swap = false;
    if (fmt == "NV12") {
        Frame copy = frame;
        arr = copy.to_rgb();  // convert-after-shrink: the small image
        want_swap = color == "BGR";
    } else if (fmt == "RGB" || fmt == "BGR" || fmt == "GRAY8") {
        Frame copy = frame;
        arr = copy.to_array();
        if (fmt == "GRAY8") {
            cv::Mat expanded;  // expand to 3 channels (python stacks equals)
            cv::cvtColor(arr, expanded, cv::COLOR_GRAY2RGB);
            arr = expanded;
        }
        want_swap = (fmt == "RGB" && color == "BGR") || (fmt == "BGR" && color == "RGB");
    } else {
        throw std::invalid_argument("unsupported source format for preprocessing: " + fmt);
    }
    if (want_swap) {
        cv::Mat swapped;  // RGB<->BGR is a pure channel reversal either way
        cv::cvtColor(arr, swapped, cv::COLOR_RGB2BGR);
        arr = swapped;
    }
    if (!arr.isContinuous()) arr = arr.clone();
    return arr;
}

}  // namespace neoruntime_ipc_sdk
