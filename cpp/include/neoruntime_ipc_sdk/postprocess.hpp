// postprocess.hpp — post-processing helpers for raw model outputs (port
// of postprocess.py, 0.8.1): greedy NMS plus decode adapters for the
// common YOLO export shapes.
//
// Streaming inference applies NMS server-side when the model is registered
// with NMS parameters, but single-shot infer() results, custom decode
// heads and app-side re-filtering still need a local implementation.
//
// The YoloV8Postprocessor / YoloV5Postprocessor adapters decode
// InferenceResult.raw_outputs from models whose postprocess is *not*
// handled server-side (generic plugin exports — the ones whose thresholds
// are baked in at export time and cannot be tuned via
// update_postprocess_config). Thresholds here are constructor parameters,
// i.e. tunable at runtime.
//
//   std::vector<int> keep = nms(boxes_xyxy, scores, 0.45, &cls);
//   YoloV8Postprocessor post({"person", "car"}, /*score_threshold=*/0.3);
//   std::vector<DetectedObject> objects = post(result.raw_outputs, &meta);
#pragma once

#include <opencv2/core.hpp>

#include <array>
#include <optional>
#include <string>
#include <vector>

#include "neoruntime_ipc_sdk/inference_types.hpp"
#include "neoruntime_ipc_sdk/preprocess.hpp"

namespace neoruntime_ipc_sdk {

// Greedy non-maximum suppression.
//   boxes         (N, 4) xyxy boxes [x1, y1, x2, y2]
//   scores        (N,) confidence scores
//   iou_threshold boxes overlapping a kept box by more than this are
//                 suppressed (must be in [0, 1])
//   class_ids     optional (N,) class labels — boxes of different classes
//                 never suppress each other
// Returns the indices of the kept boxes, in descending-score order.
// Throws std::invalid_argument (python ValueError) on length mismatches
// or an out-of-range threshold, with python's exact messages.
std::vector<int> nms(const std::vector<std::array<float, 4>>& boxes,
                     const std::vector<float>& scores, double iou_threshold = 0.5,
                     const std::vector<int>* class_ids = nullptr);

// Base class for client-side raw_outputs decoders.
//
// Subclasses take the raw output tensors plus the optional PreprocessMeta
// from the matching Preprocessor call and return domain objects. Boxes
// come back in *source-frame* pixel coordinates whenever a meta is
// provided (nullptr = python None: input coordinates, no clipping).
class Postprocessor {
public:
    virtual ~Postprocessor() = default;
    virtual std::vector<DetectedObject> operator()(
        const std::optional<std::vector<cv::Mat>>& raw_outputs,
        const PreprocessMeta* meta = nullptr) const = 0;
};

// Shared plumbing: thresholds, labels, NMS hand-off, object build.
//
// When a meta is passed, boxes come back in source-frame coordinates and
// (unless clip=false) are clipped to the source frame — boxes detected in
// the letterbox pad would otherwise carry negative or beyond-frame
// coordinates. Boxes with nothing left after clipping are dropped.
class YoloPostprocessorBase : public Postprocessor {
public:
    YoloPostprocessorBase(std::vector<std::string> labels = {},
                          double score_threshold = 0.25, double iou_threshold = 0.45,
                          bool clip = true);

    std::vector<std::string> labels;
    double score_threshold = 0.25;
    double iou_threshold = 0.45;
    bool clip = true;

    // python _label: the class name when known, else the id as a string.
    std::string label_for(int class_id) const;

protected:
    // python _finish: threshold filter, NMS, meta mapping + clipping.
    std::vector<DetectedObject> finish(const std::vector<std::array<float, 4>>& boxes_xyxy,
                                       const std::vector<float>& scores,
                                       const std::vector<int>& class_ids,
                                       const PreprocessMeta* meta) const;
};

// Decode YOLOv8-family heads (anchor-free, ONNX-standard export).
//
// Expects raw_outputs[0] shaped (1, 4+nc, N) / (4+nc, N) (or the
// transposed variant): boxes as cx, cy, w, h in input-pixel coordinates
// plus one score column per class, no objectness. Raw anchor-grid heads
// (pre-decode exports) are out of scope, as are non-dequantised int8
// outputs — standard exports dequantise to float.
class YoloV8Postprocessor : public YoloPostprocessorBase {
public:
    using YoloPostprocessorBase::YoloPostprocessorBase;
    std::vector<DetectedObject> operator()(
        const std::optional<std::vector<cv::Mat>>& raw_outputs,
        const PreprocessMeta* meta = nullptr) const override;
};

// Decode YOLOv5-family heads (anchor-based, ONNX-standard export).
//
// Expects raw_outputs[0] shaped (1, N, 5+nc) / (N, 5+nc) (or the
// transposed variant): cx, cy, w, h, objectness plus one class score
// column. The final score is objectness * class_score. Same scope limits
// as YoloV8Postprocessor: decoded-grid exports only, float outputs
// assumed.
class YoloV5Postprocessor : public YoloPostprocessorBase {
public:
    using YoloPostprocessorBase::YoloPostprocessorBase;
    std::vector<DetectedObject> operator()(
        const std::optional<std::vector<cv::Mat>>& raw_outputs,
        const PreprocessMeta* meta = nullptr) const override;
};

}  // namespace neoruntime_ipc_sdk
