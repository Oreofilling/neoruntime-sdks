// inference_types.hpp — inference value types, 1:1 port of inference_types.py
// (0.8.1) plus the inference-facing types that lived in inference.hpp.
//
// Split from inference.hpp so the data model has its own header (mirroring
// the Python inference_types.py module); inference.hpp remains the facade
// that re-exports everything for existing includes.
#pragma once

#include <opencv2/core.hpp>

#include <cstdint>
#include <optional>
#include <string>
#include <variant>
#include <vector>

#include "neoruntime_ipc_sdk/frame.hpp"
#include "neoruntime_ipc_sdk/types.hpp"

namespace neoruntime_ipc_sdk {

// A detected object. Bounding box is in normalized [0,1] coordinates.
struct DetectedObject {
    std::string label;
    float score = 0.0f;
    BoundingBox bbox{};
    std::int32_t class_id = 0;
    std::optional<std::int64_t> track_id;  // not populated from proto; reserved
};

struct LandmarkPoint {
    float x = 0.0f;
    float y = 0.0f;
    float confidence = 1.0f;
};

struct LandmarkSet {
    std::string type;
    std::vector<LandmarkPoint> points;
};

struct Classification {
    std::string type;
    std::int32_t class_id = 0;
    std::string label;
    float confidence = 0.0f;
};

// Segmentation mask. mask_rle is varint RLE bytes (run start, run length, ...).
// to_mask() decodes it to an HxW CV_8U Mat (foreground=255).
struct SegmentationMask {
    std::int32_t class_id = 0;
    std::string label;
    float confidence = 0.0f;
    BoundingBox bbox{};
    std::string mask_rle;
    std::int32_t mask_width = 0;
    std::int32_t mask_height = 0;

    cv::Mat to_mask() const;
};

struct OcrLine {
    std::string text;
    float confidence = 0.0f;
    BoundingBox bbox{};
};

struct Embedding {
    std::uint32_t dim = 0;
    std::vector<float> data;
};

struct DepthMap {
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    cv::Mat data;  // CV_32F, height x width (meters)
};

// One inference result. `raw_outputs` is populated when the server returns raw
// tensors (or raw_output_only); otherwise it is std::nullopt.
struct InferenceResult {
    std::uint64_t frame_sequence = 0;
    std::uint64_t timestamp_ns = 0;
    std::vector<DetectedObject> objects;
    std::vector<Classification> classifications;
    std::vector<LandmarkSet> landmarks;
    std::vector<SegmentationMask> masks;
    std::vector<OcrLine> ocr_lines;
    std::vector<Embedding> embeddings;
    std::vector<DepthMap> depth_maps;
    std::optional<std::vector<cv::Mat>> raw_outputs;
    std::uint64_t infer_time_us = 0;
    std::uint64_t queue_time_us = 0;
    std::uint64_t hw_infer_time_us = 0;   // NPU hw latency, 0 if unavailable
    std::string status_message;           // diagnostic; "simulation" if no source
    // Result skew: result-ready minus frame capture, in microseconds —
    // server-computed on the device clock (same domain as timestamp_ns).
    // 0 means "not measured" (failed inference or an older server); the
    // wire field is int64, so keep it signed like Python's int.
    std::int64_t skew_us = 0;

    bool has_person() const {
        for (const auto& o : objects)
            if (o.label == "person") return true;
        return false;
    }
    std::size_t count_by_label(const std::string& label) const {
        std::size_t n = 0;
        for (const auto& o : objects)
            if (o.label == label) ++n;
        return n;
    }
    std::vector<DetectedObject> get_objects_by_label(const std::string& label) const {
        std::vector<DetectedObject> out;
        for (const auto& o : objects)
            if (o.label == label) out.push_back(o);
        return out;
    }
};

// A named, shaped, typed tensor spec — used both to register models and to
// describe a loaded model's inputs/outputs.
struct TensorSpec {
    std::vector<int> shape;
    DataType dtype = DataType::Float32;
    std::string name;
    std::string layout;  // "NHWC"/"NCHW"/"HWC"/"CHW"/"NC"/"HW"/"NV12"; "" unset
};

struct ModelInfo {
    std::string model_id;
    std::string model_path;
    std::string version;
    std::vector<TensorSpec> inputs;
    std::vector<TensorSpec> outputs;
    float estimated_tops = 0.0f;
    std::uint32_t estimated_memory = 0;
    std::uint64_t load_timestamp = 0;
    std::string owner_id;               // app owning the registration ("" = global)
    bool transient = false;             // app-bundled model, hidden from model page
    std::uint32_t batch_size = 1;       // NPU batch the session is configured with
};

// One item within an infer_batch() call.
struct BatchInferItem {
    cv::Mat image;
    std::string model_id;
    std::uint32_t timeout_ms = 5000;
    std::uint32_t priority = 4;
};

// Zero-copy infer() input: a buffer id in camera-daemon's DSP registry
// (DSP_ALLOC or DSP_IMPORT over camera.sock). NV12 geometry only —
// ai-runtime's registry-side repack knows tight NV12. The id is BORROWED:
// keep the owning buffer alive until the RPC settles (infer() blocks, so
// simply holding the ref across the call is enough).
struct BufferRefInput {
    std::uint64_t buffer_id = 0;
    std::uint32_t width = 0;   // NV12 luma width
    std::uint32_t height = 0;  // NV12 luma height (tensor shape is H*3/2 x W)
};

// infer() input. cv::Mat ships pixels as bytes; the fd-backed members take
// the zero-copy path (Phase 5: Frame/FrameHandle dma-bufs are imported via
// DspClient and referenced by buffer id — until then they are materialized
// to a Mat through the handle, same pixels, one copy); BufferRefInput sends
// its daemon-registry id directly, no import, no copy.
using InferInput =
    std::variant<cv::Mat, Frame, std::shared_ptr<FrameHandle>, BufferRefInput>;

struct ModelStats {
    std::string model_id;
    std::uint64_t total_inferences = 0;
    std::uint64_t total_errors = 0;
    std::uint64_t avg_latency_us = 0;
    float current_qps = 0.0f;
    std::uint32_t queue_depth = 0;
    float hw_fps = 0.0f;  // 0 if unavailable
    // Stream-infer skew aggregates (µs); 0 when the model has never run on a
    // stream or the server predates the fields.
    std::uint64_t avg_skew_us = 0;
    std::uint64_t max_skew_us = 0;
    std::uint64_t skew_samples = 0;
};

struct InferenceSystemStats {
    std::vector<ModelStats> model_stats;
    float device_utilization = 0.0f;   // 0..1
    float device_temperature = 0.0f;
    std::uint64_t total_memory_bytes = 0;
    std::uint64_t used_memory_bytes = 0;
    float cpu_utilization = 0.0f;      // 0..1, -1 if unknown
    float dsp_utilization = 0.0f;      // 0..1, -1 if N/A
    std::int64_t ram_total_kib = -1;
    std::int64_t ram_used_kib = -1;
};

}  // namespace neoruntime_ipc_sdk
