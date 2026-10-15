// preprocess.hpp — client-side preprocessing for model inference (port of
// preprocess.py, 0.8.1).
//
// Composes the SDK's existing pieces into one call and returns the affine
// PreprocessMeta alongside the model-ready tensor, so postprocessing can
// map model-input coordinates back to the source frame without hand-
// derived letterbox maths:
//
//   * geometry rides Frame::resize — keep-fd frames scale on the DSP
//     (zero-copy import) and the transform comes straight from the shared
//     geometry helper, so the pixels and the meta cannot drift;
//   * colour conversion runs on the *resized* image (convert-after-shrink:
//     a 640x640 convert is ~9x cheaper than a 1080p one);
//   * normalize is opt-in and off by default — HEF models on this platform
//     take uint8 inputs with the quantisation scales compiled into the
//     HEF; pre-normalising would corrupt them. from_model enables it
//     automatically for float-dtype inputs only.
//
//   Preprocessor pre = Preprocessor::from_model(client, "person_v1");
//   auto [tensor, meta] = pre(frame);      // Frame or cv::Mat source
//   InferenceResult r = client.infer(tensor, "person_v1");
//   auto [x_src, y_src] = meta.to_source(x_model, y_model);
#pragma once

#include <opencv2/core.hpp>

#include <array>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#include "neoruntime_ipc_sdk/dsp.hpp"
#include "neoruntime_ipc_sdk/frame.hpp"
#include "neoruntime_ipc_sdk/inference.hpp"

namespace neoruntime_ipc_sdk {

// Affine geometry of a preprocessing pass. Semantics per axis:
// dst = src * scale + origin, where dst is the model-input image and src
// the original frame — so the inverse (what postprocessing needs) is
// src = (dst - origin) / scale.
struct PreprocessMeta {
    std::pair<int, int> original_size{0, 0};      // (w, h) of the source frame
    std::pair<int, int> input_size{0, 0};         // (w, h) of the model input
    std::pair<double, double> scale{1.0, 1.0};    // (sx, sy)
    std::pair<int, int> origin{0, 0};             // (ox, oy)
    std::string color = "RGB";
    std::string layout = "NHWC";

    // Map a model-input point back to source-frame coordinates.
    std::pair<double, double> to_source(double x, double y) const;
    // Map an (x1, y1, x2, y2) box back to source-frame coordinates.
    std::array<double, 4> to_source_box(const std::array<double, 4>& box) const;
};

// Resize + colour-convert (+ optional normalise/layout) in one call.
//
//   size           target (width, height); nullopt until from_model derives
//                  it (calling before then raises)
//   color          target channel order, "RGB" or "BGR" (case-insensitive)
//   resize_mode    "letterbox" (default) / "stretch" / "crop", forwarded
//                  to Frame::resize
//   pad_value      letterbox pad value (114 is the YOLO convention)
//   normalize      divide by 255 into float32. Off by default — HEF models
//                  quantise internally and expect uint8
//   layout         "NHWC" (default) or "NCHW" output arrangement
//   source_format  format hint for cv::Mat inputs ("NV12" / "GRAY8" /
//                  "RGB" / "BGR"); single-channel Mats require it
//   nv12_passthrough  keep the frame in NV12 end to end instead of
//                  converting to RGB. The resize rides the DSP and its
//                  result stays device-side: the returned tensor is a
//                  DspBufferRef (infer sends its buffer_id — no read-back,
//                  no 3-channel copy over the wire), or the source itself
//                  at identity geometry. Requires resize_mode="stretch",
//                  normalize=false and layout="NHWC" (the daemon's
//                  letterbox pads green and its crop placement disagrees
//                  with the CPU path, so stretch is the only mode both
//                  legs place identically); color is ignored —
//                  PreprocessMeta.color reports "NV12". Sources must
//                  already be NV12. from_model enables it automatically
//                  for NV12-layout models.
class Preprocessor {
public:
    // What a pass produces: a pixel Mat (the common RGB path), a
    // device-resident DspBufferRef (nv12_passthrough resize), or the
    // source Frame itself (passthrough at identity geometry on a keep-fd
    // source — infer's zero-copy input).
    using Tensor = std::variant<cv::Mat, DspBufferRef, Frame>;

    struct Output {
        Tensor tensor;
        PreprocessMeta meta;
    };

    // Explicit-field overrides for from_model (python's **overrides).
    struct Overrides {
        std::optional<std::pair<int, int>> size;
        std::optional<std::string> color;
        std::optional<std::string> resize_mode;
        std::optional<int> pad_value;
        std::optional<bool> normalize;
        std::optional<std::string> layout;
        std::optional<std::string> source_format;
        std::optional<bool> nv12_passthrough;
    };

    Preprocessor() = default;
    Preprocessor(std::optional<std::pair<int, int>> size, std::string color = "RGB",
                 std::string resize_mode = "letterbox", int pad_value = 114,
                 bool normalize = false, std::string layout = "NHWC",
                 std::optional<std::string> source_format = std::nullopt,
                 bool nv12_passthrough = false);

    // Derive size/layout/normalisation from the model's input spec.
    // Keyword overrides replace any derived value.
    static Preprocessor from_model(InferenceClient& client, const std::string& model_id);
    static Preprocessor from_model(InferenceClient& client, const std::string& model_id,
                                   const Overrides& overrides);

    Output operator()(const Frame& source) const;
    Output operator()(const cv::Mat& source) const;

    // Public attribute parity (python exposes these directly).
    std::optional<std::pair<int, int>> size;
    std::string color = "RGB";
    std::string resize_mode = "letterbox";
    int pad_value = 114;
    bool normalize = false;
    std::string layout = "NHWC";
    std::optional<std::string> source_format;
    bool nv12_passthrough = false;

private:
    Frame as_frame(const Frame& source) const;
    Frame as_frame(const cv::Mat& source) const;  // infers format + geometry
    Output passthrough(const Frame& frame) const;
    PreprocessMeta passthrough_meta(std::pair<int, int> original_size,
                                    std::pair<double, double> scale,
                                    std::pair<int, int> origin) const;
    cv::Mat to_color(const Frame& frame) const;
    Output run(const Frame& frame) const;
};

// Formats to_color handles; the constructor validates a hint against this
// set (python _SOURCE_FORMATS, sorted repr in the error:
// ['BGR', 'GRAY8', 'NV12', 'RGB']).
bool is_supported_source_format(std::string_view format);

}  // namespace neoruntime_ipc_sdk
