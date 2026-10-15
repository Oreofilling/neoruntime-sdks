// inference.hpp — AI InferenceService gRPC client. 1:1 port of inference.py
// (0.8.1, GenaiMixin folded in).
//
// Transport: sync gRPC stubs over UDS. The Python client runs grpc.aio on a
// background asyncio loop to dodge the sync completion-queue spin; C++ sync
// stubs block on a futex, so that entire asyncio/_invoke/_loop_thread layer is
// NOT ported. infer_async()/infer_batch_async() use std::async (std::future)
// instead. GenaiGenerate keeps a plain ClientReader loop; StreamInfer bridges
// to a pull iterator via a pump thread + bounded drop-oldest queue (see
// InferenceStream).
//
// Value types live in inference_types.hpp (mirrors the Python
// inference_types.py split); this header re-exports them.
//
// Images and tensors are cv::Mat (the Python SDK's role is played by numpy);
// cv::Mat <-> proto Tensor conversion is in src/detail/tensor_io.hpp.
#pragma once

#include <cstdint>
#include <future>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "neoruntime_ipc_sdk/inference_types.hpp"

namespace neoruntime_ipc_sdk {

enum class GenaiKind : int { Llm = 0, Vlm = 1 };
enum class GenaiFinishReason : int { Done = 0, Aborted = 1, MaxTokens = 2, Error = 3 };

// Pull-based stream over StreamInfer — the C++ shape of inference.py's
// _SubscribeIterator. A background pump thread reads the server stream into
// a bounded drop-oldest queue, so a slow consumer drops its OLDEST queued
// results (counted in dropped()) instead of growing memory without bound.
//
// next() returns (frame_sequence, InferenceResult), skipping failed frames,
// or std::nullopt at clean end. A broken stream raises std::runtime_error
// from next(). After max_consecutive_failures failed frames in a row
// (subscribe parameter, default 10), next() raises instead of skipping
// forever. Health telemetry: dropped(), last/avg_latency_ms() (EMA, alpha
// 0.1, sane 0..60 s window), last/avg_skew_us().
//
// Must not outlive the InferenceClient that produced it (borrows the
// channel). cancel() stops the pump from any thread; close() is the
// consumer-thread end (both idempotent; the destructor cancels).
class InferenceStream {
public:
    ~InferenceStream();
    InferenceStream(InferenceStream&&) noexcept;
    InferenceStream& operator=(InferenceStream&&) noexcept;
    InferenceStream(const InferenceStream&) = delete;
    InferenceStream& operator=(const InferenceStream&) = delete;

    // Next (frame_sequence, result), or std::nullopt at end of stream.
    // Throws std::runtime_error when the stream broke or the consecutive
    // failure limit tripped.
    std::optional<std::pair<std::uint64_t, InferenceResult>> next();

    // End the subscription from the consuming thread.
    void close();
    // Stop the subscription from ANY thread (a consumer blocked in next()
    // wakes and gets nullopt). Idempotent; safe before the first next().
    void cancel();

    // Health telemetry (Python _SubscribeIterator attributes).
    std::uint64_t dropped() const noexcept;
    double last_latency_ms() const noexcept;
    double avg_latency_ms() const noexcept;
    std::int64_t last_skew_us() const noexcept;
    double avg_skew_us() const noexcept;

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
    // image: cv::Mat pixels, or a zero-copy input (see InferInput) — a keep-fd
    // NV12 Frame/FrameHandle (materialized through the handle until the DSP
    // import path lands in Phase 5) or a daemon-registry BufferRefInput.
    InferenceResult infer(const InferInput& image,
                          const std::string& model_id,
                          std::uint32_t timeout_ms = 5000,
                          std::uint32_t priority = 4,
                          const std::string& session_id = "");
    InferenceResult infer(const cv::Mat& image,
                          const std::string& model_id,
                          std::uint32_t timeout_ms = 5000,
                          std::uint32_t priority = 4,
                          const std::string& session_id = "");

    // Non-blocking infer; resolves to a parsed InferenceResult. Caller MUST
    // wait on the future. Safe to call after the client is destroyed — the
    // task captures its own channel/stub.
    std::future<InferenceResult> infer_async(const InferInput& image,
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
    // Yields (frame_sequence, InferenceResult) for a camera stream
    // subscription. Failed frames are skipped; max_consecutive_failures
    // failed frames in a row (default 10; nullopt disables) raise from
    // next(). queue_size bounds the client-side result queue (default 100;
    // nullopt unbounded) — a slow consumer drops oldest, counted on the
    // stream's dropped().
    InferenceStream subscribe(const std::string& stream,
                              const std::string& model,
                              std::uint32_t fps = 10,
                              const std::string& session_id = "",
                              bool raw_output_only = false,
                              std::optional<std::uint32_t> max_consecutive_failures = 10,
                              std::optional<std::uint32_t> queue_size = 100);

    // ---- Model management ----
    // owner_id GOTCHA (register-owner-id): a registration carrying owner_id
    // does NOT enter the global model table — subscribe()/list_models() from
    // other clients then report "Model not found". Manual registrations must
    // leave owner_id empty.
    // batch_size: NPU batch for the session (>1 requires a batch-capable
    // HEF; registration fails otherwise). Batched models must use
    // infer_batch(); single-image infer() is rejected for batch>1 models.
    std::string register_model(const std::string& model_path,
                               const std::string& model_id = "",
                               const std::string& owner_id = "",
                               const std::string& model_type = "",
                               const std::string& model_variant = "",
                               const std::vector<TensorSpec>& inputs = {},
                               const std::vector<TensorSpec>& outputs = {},
                               std::uint32_t batch_size = 1);
    void unregister_model(const std::string& model_id);
    std::vector<ModelInfo> list_models();
    std::optional<ModelInfo> get_model_info(const std::string& model_id);

    // ---- Statistics ----
    InferenceSystemStats get_stats();
    // sampling_window_ms bounds the blocking HAL sampling window behind the
    // utilization figures. 0 = server-default request; explicit values are
    // clamped to [1, 5000] ms client-side (mirrors inference.py).
    InferenceSystemStats get_stats(std::uint32_t sampling_window_ms);

    // ---- Sessions ----
    std::string create_session(const std::string& session_id,
                               const std::string& app_id = "",
                               const std::vector<std::string>& allowed_models = {},
                               std::uint32_t max_qps = 0,
                               std::uint32_t max_concurrent = 0,
                               std::uint32_t priority = 4);
    void destroy_session(const std::string& session_id);

    // ---- Dynamic postprocess config ----
    // Runtime postprocess tuning for a model (e.g. detection thresholds for
    // family-function postprocess; unknown keys raise server-side).
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
