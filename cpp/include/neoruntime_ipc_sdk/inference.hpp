// inference.hpp — AI InferenceService gRPC client. 1:1 port of inference.py.
//
// Transport: sync gRPC stubs over UDS. The Python client runs grpc.aio on a
// background asyncio loop to dodge the sync completion-queue spin; C++ sync
// stubs block on a futex, so that entire asyncio/_invoke/_loop_thread layer is
// NOT ported. infer_async()/infer_batch_async() use std::async (std::future)
// instead. Streaming RPCs (StreamInfer, GenaiGenerate) use ClientReader loops.
//
// Images and tensors are cv::Mat (the Python SDK's role is played by numpy);
// cv::Mat <-> proto Tensor conversion is in src/detail/tensor_io.hpp.
#pragma once

#include <opencv2/core.hpp>

#include <cstdint>
#include <functional>
#include <future>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

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
};

// One item within an infer_batch() call.
struct BatchInferItem {
    cv::Mat image;
    std::string model_id;
    std::uint32_t timeout_ms = 5000;
    std::uint32_t priority = 4;
};

struct ModelStats {
    std::string model_id;
    std::uint64_t total_inferences = 0;
    std::uint64_t total_errors = 0;
    std::uint64_t avg_latency_us = 0;
    float current_qps = 0.0f;
    std::uint32_t queue_depth = 0;
    float hw_fps = 0.0f;  // 0 if unavailable
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

enum class GenaiKind : int { Llm = 0, Vlm = 1 };
enum class GenaiFinishReason : int { Done = 0, Aborted = 1, MaxTokens = 2, Error = 3 };

// Pull-based stream over StreamInfer. Each next() is (frame_sequence, result),
// or std::nullopt when the server closes/breaks the stream. Must not outlive the
// InferenceClient that produced it (borrows the channel).
class InferenceStream {
public:
    ~InferenceStream();
    InferenceStream(InferenceStream&&) noexcept;
    InferenceStream& operator=(InferenceStream&&) noexcept;
    InferenceStream(const InferenceStream&) = delete;
    InferenceStream& operator=(const InferenceStream&) = delete;

    std::optional<std::pair<std::uint64_t, InferenceResult>> next();

private:
    InferenceStream();
    struct Impl;
    std::unique_ptr<Impl> impl_;
    friend class InferenceClient;
};

// Pull-based stream over GenaiGenerate. next() yields each generated token;
// std::nullopt marks the end (finish/abort/error/broken). After the stream
// ends, finish_reason() reports why.
class GenaiStream {
public:
    ~GenaiStream();
    GenaiStream(GenaiStream&&) noexcept;
    GenaiStream& operator=(GenaiStream&&) noexcept;
    GenaiStream(const GenaiStream&) = delete;
    GenaiStream& operator=(const GenaiStream&) = delete;

    std::optional<std::string> next();
    GenaiFinishReason finish_reason() const;

private:
    GenaiStream();
    struct Impl;
    std::unique_ptr<Impl> impl_;
    friend class InferenceClient;
};

class InferenceClient {
public:
    explicit InferenceClient(std::string endpoint = "");
    ~InferenceClient();
    InferenceClient(const InferenceClient&) = delete;
    InferenceClient& operator=(const InferenceClient&) = delete;
    InferenceClient(InferenceClient&&) noexcept;
    InferenceClient& operator=(InferenceClient&&) noexcept;

    void connect();
    void close();
    bool connected() const noexcept;

    // ---- Single inference ----
    InferenceResult infer(const cv::Mat& image,
                          const std::string& model_id,
                          std::uint32_t timeout_ms = 5000,
                          std::uint32_t priority = 4,
                          const std::string& session_id = "");

    // Non-blocking infer; resolves to a parsed InferenceResult. Caller MUST
    // wait on the future. Safe to call after the client is destroyed — the
    // task captures its own channel/stub.
    std::future<InferenceResult> infer_async(const cv::Mat& image,
                                             const std::string& model_id,
                                             std::uint32_t timeout_ms = 5000,
                                             std::uint32_t priority = 4,
                                             const std::string& session_id = "");

    // ---- Batch inference ----
    std::vector<InferenceResult> infer_batch(const std::vector<BatchInferItem>& items,
                                             std::uint32_t timeout_ms = 10000);
    std::future<std::vector<InferenceResult>>
    infer_batch_async(const std::vector<BatchInferItem>& items,
                      std::uint32_t timeout_ms = 10000);

    // ---- Multi-tensor inference (named inputs) ----
    // Returns the raw output tensors as cv::Mats. input_names default to
    // "input_0", "input_1", ... when empty (mirrors inference.py).
    std::vector<cv::Mat> infer_with_tensors(const std::string& model_id,
                                            const std::vector<cv::Mat>& inputs,
                                            const std::vector<std::string>& input_names = {},
                                            std::uint32_t timeout_ms = 5000);

    // ---- Stream inference ----
    InferenceStream subscribe(const std::string& stream,
                              const std::string& model,
                              std::uint32_t fps = 10,
                              const std::string& session_id = "",
                              bool raw_output_only = false);

    // ---- Model management ----
    std::string register_model(const std::string& model_path,
                               const std::string& model_id = "",
                               const std::string& owner_id = "",
                               const std::string& model_type = "",
                               const std::string& model_variant = "",
                               const std::vector<TensorSpec>& inputs = {},
                               const std::vector<TensorSpec>& outputs = {});
    void unregister_model(const std::string& model_id);
    std::vector<ModelInfo> list_models();
    std::optional<ModelInfo> get_model_info(const std::string& model_id);

    // ---- Statistics ----
    InferenceSystemStats get_stats();

    // ---- Sessions ----
    std::string create_session(const std::string& session_id,
                               const std::string& app_id = "",
                               const std::vector<std::string>& allowed_models = {},
                               std::uint32_t max_qps = 0,
                               std::uint32_t max_concurrent = 0,
                               std::uint32_t priority = 4);
    void destroy_session(const std::string& session_id);

    // ---- Dynamic postprocess config ----
    bool update_postprocess_config(const std::string& model_id,
                                   const std::string& config_json);

    // ---- CLIP text encoding ----
    std::vector<float> encode_text(const std::string& text,
                                   std::uint32_t timeout_ms = 5000);

    // ---- GenAI (LLM/VLM) ----
    std::string genai_create_session(const std::string& hef_path,
                                     GenaiKind kind = GenaiKind::Llm,
                                     const std::string& lora_name = "",
                                     bool optimize_memory = false);
    void genai_destroy_session(const std::string& session_id);

    // Stream tokens. `images` entries are raw RGB pixel bytes per frame.
    GenaiStream genai_generate(const std::string& session_id,
                               const std::vector<std::string>& messages,
                               const std::vector<std::string>& images = {},
                               const std::vector<std::string>& stop_tokens = {},
                               float temperature = 0.0f,
                               float top_p = 1.0f,
                               std::uint32_t top_k = 0,
                               std::uint32_t max_tokens = 512,
                               bool do_sample = false);
    void genai_abort(const std::string& session_id);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace neoruntime_ipc_sdk
