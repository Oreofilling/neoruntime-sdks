// pipeline.hpp — composed pre → infer → post pipeline over InferenceClient
// (port of pipeline.py, 0.8.1).
//
// The pipeline is for the **client-side processing path**: apps that need
// their own preprocessing (raw-tensor models, bandwidth-sensitive loops on
// the sub stream) and/or their own postprocessing (generic plugin exports
// whose thresholds cannot be tuned server-side). When a family model with
// server-side postprocess fits, InferenceClient::subscribe remains the
// cheaper path — no pixel ever reaches the app.
//
// For the whole capture → infer → overlay loop as one platform-scheduled
// call use StreamPipeline instead; this module pulls pixels into the app,
// that one never does. Choose by where the pixels should live.
//
// Result priority: when a postprocessor is set and the model returned
// raw_outputs, the client-side decode wins (that is the point — its
// thresholds are yours); otherwise the server-decoded
// InferenceResult::objects pass through untouched.
//
//   InferencePipeline pipe = InferencePipeline::from_model(
//       "custom_yolov8n", nullptr,
//       std::make_unique<YoloV8Postprocessor>(std::vector<std::string>{"person",
//       "car"}, 0.3));
//   for (std::optional<Frame> frame = media.subscribe("sub", true); frame;
//        frame = media.subscribe("sub", true)) {
//       PipelineResult out = pipe.run(*frame);  // .objects in source coords
//   }
#pragma once

#include <chrono>
#include <cstdint>
#include <future>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "neoruntime_ipc_sdk/frame.hpp"
#include "neoruntime_ipc_sdk/inference.hpp"
#include "neoruntime_ipc_sdk/inference_types.hpp"
#include "neoruntime_ipc_sdk/postprocess.hpp"
#include "neoruntime_ipc_sdk/preprocess.hpp"

namespace neoruntime_ipc_sdk {

// One pipeline pass: decoded objects plus everything underneath. Move-only
// (tensor may hold a device-resident DspBufferRef).
struct PipelineResult {
    std::vector<DetectedObject> objects;
    InferenceResult result;
    std::optional<PreprocessMeta> meta;
    double latency_ms = 0.0;
    // The preprocessed image actually sent to the model (when a
    // preprocessor ran) — handy for drawing previews at input size. On the
    // NV12 passthrough path this is a device-resident buffer ref, which
    // the pipeline releases once inference settles unless it was built
    // with retain_input=true (then release() hands it back when the chain
    // using it is done).
    std::optional<Preprocessor::Tensor> tensor;

    // Release a retained passthrough input ref, if any. Idempotent, and a
    // no-op for plain-array tensors — only a retain_input=true pipeline
    // leaves a live ref in tensor.
    void release();
};

// Bind a Preprocessor, an InferenceClient and a Postprocessor.
class InferencePipeline {
public:
    // What run() accepts: a keep-fd Frame or a plain cv::Mat (python's
    // duck-typed source).
    using Source = std::variant<Frame, cv::Mat>;

    // A default client is created when omitted / null (python client=None).
    InferencePipeline();
    InferencePipeline(std::unique_ptr<InferenceClient> client,
                      std::string model_id = "",
                      std::optional<Preprocessor> preprocessor = std::nullopt,
                      std::unique_ptr<Postprocessor> postprocessor = nullptr,
                      std::uint32_t infer_timeout_ms = 5000,
                      bool retain_input = false);

    // Derive the preprocessor from the model's input spec and bind it
    // (same overrides as Preprocessor::from_model).
    static InferencePipeline from_model(
        const std::string& model_id, std::unique_ptr<InferenceClient> client = {},
        std::unique_ptr<Postprocessor> postprocessor = nullptr,
        const Preprocessor::Overrides& preprocess_overrides = {});

    // Preprocess → infer → postprocess one frame or array, blocking.
    PipelineResult run(const Source& source, const std::string& model_id = "");

    // Non-blocking run(); the future resolves to PipelineResult.
    // Preprocessing runs inline (cheap — it is the resize/convert on the
    // small image); the NPU call rides InferenceClient::infer_async and
    // the postprocessor runs on the completing thread. latency_ms is not
    // measured on this path. As with infer_async, the returned future must
    // be resolved (.get()) or exceptions stay silently swallowed. The
    // pipeline must outlive the future (the task borrows `this` for the
    // postprocessor, exactly like python's closure over self).
    std::future<PipelineResult> run_async(const Source& source,
                                          const std::string& model_id = "");

    // python exposes .client directly; the pipeline owns it either way.
    InferenceClient& client() { return *owned_client_; }

    std::string model_id;
    std::optional<Preprocessor> preprocessor;
    std::unique_ptr<Postprocessor> postprocessor;
    std::uint32_t infer_timeout_ms = 5000;
    bool retain_input = false;

private:
    std::pair<Preprocessor::Tensor, std::optional<PreprocessMeta>> prepare(
        const Source& source) const;
    std::vector<DetectedObject> decode(
        const InferenceResult& result,
        const std::optional<PreprocessMeta>& meta) const;
    // Release a passthrough input ref once inference has settled. Returns
    // what PipelineResult::tensor should carry: the tensor when there is
    // nothing to release — plain arrays, or a retain_input=true pipeline
    // keeping its ref for a zero-copy chain — and nullopt behind a
    // released ref.
    std::optional<Preprocessor::Tensor> settle_input(Preprocessor::Tensor& tensor) const;
    const std::string& resolve_model(const std::string& call_model) const;

    std::unique_ptr<InferenceClient> owned_client_;
};

}  // namespace neoruntime_ipc_sdk
