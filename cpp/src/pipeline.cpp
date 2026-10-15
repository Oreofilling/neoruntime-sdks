// pipeline.cpp — port of pipeline.py (0.8.1): pre → infer → post with the
// device-resident passthrough ref settled exactly like python (released
// once inference settles unless retain_input, released-not-leaked on the
// failure path).
#include "neoruntime_ipc_sdk/pipeline.hpp"

#include <stdexcept>
#include <utility>

namespace neoruntime_ipc_sdk {

namespace {

// python _release_ref_like / the InferInput hand-off: a passthrough
// DspBufferRef rides infer as its daemon-registry buffer_id (zero read-
// back, no 3-channel copy); plain tensors ship their pixels.
InferInput to_infer_input(const Preprocessor::Tensor& tensor) {
    if (const auto* mat = std::get_if<cv::Mat>(&tensor)) return InferInput(*mat);
    if (const auto* frame = std::get_if<Frame>(&tensor)) return InferInput(*frame);
    const DspBufferRef& ref = std::get<DspBufferRef>(tensor);
    BufferRefInput input;
    input.buffer_id = ref.buffer_id();  // throws when released/dead
    input.width = static_cast<std::uint32_t>(ref.width());
    input.height = static_cast<std::uint32_t>(ref.height());
    return InferInput(input);
}

}  // namespace

// ---- PipelineResult ----------------------------------------------------------------

void PipelineResult::release() {
    if (!tensor.has_value()) return;
    if (auto* ref = std::get_if<DspBufferRef>(&*tensor)) {
        ref->release();  // noexcept + idempotent (python swallows DspError)
        tensor.reset();
    }
}

// ---- InferencePipeline ----------------------------------------------------------------

InferencePipeline::InferencePipeline()
    : InferencePipeline(std::unique_ptr<InferenceClient>()) {}

InferencePipeline::InferencePipeline(std::unique_ptr<InferenceClient> client,
                                     std::string model_id_in,
                                     std::optional<Preprocessor> preprocessor_in,
                                     std::unique_ptr<Postprocessor> postprocessor_in,
                                     std::uint32_t infer_timeout_ms_in, bool retain_input_in)
    : model_id(std::move(model_id_in)),
      preprocessor(std::move(preprocessor_in)),
      postprocessor(std::move(postprocessor_in)),
      infer_timeout_ms(infer_timeout_ms_in),
      retain_input(retain_input_in),
      owned_client_(client ? std::move(client) : std::make_unique<InferenceClient>()) {}

InferencePipeline InferencePipeline::from_model(
    const std::string& model_id, std::unique_ptr<InferenceClient> client,
    std::unique_ptr<Postprocessor> postprocessor,
    const Preprocessor::Overrides& preprocess_overrides) {
    if (!client) client = std::make_unique<InferenceClient>();
    Preprocessor pre = Preprocessor::from_model(*client, model_id, preprocess_overrides);
    return InferencePipeline(std::move(client), model_id, std::move(pre),
                             std::move(postprocessor));
}

std::pair<Preprocessor::Tensor, std::optional<PreprocessMeta>> InferencePipeline::prepare(
    const Source& source) const {
    if (!preprocessor.has_value()) {
        // pass sources through untouched
        if (const auto* frame = std::get_if<Frame>(&source))
            return {Preprocessor::Tensor(*frame), std::nullopt};
        return {Preprocessor::Tensor(std::get<cv::Mat>(source)), std::nullopt};
    }
    if (const auto* frame = std::get_if<Frame>(&source)) {
        Preprocessor::Output out = (*preprocessor)(*frame);
        return {std::move(out.tensor), std::move(out.meta)};
    }
    Preprocessor::Output out = (*preprocessor)(std::get<cv::Mat>(source));
    return {std::move(out.tensor), std::move(out.meta)};
}

std::vector<DetectedObject> InferencePipeline::decode(
    const InferenceResult& result, const std::optional<PreprocessMeta>& meta) const {
    if (postprocessor != nullptr && result.raw_outputs.has_value())
        return (*postprocessor)(result.raw_outputs, meta.has_value() ? &*meta : nullptr);
    return result.objects;
}

std::optional<Preprocessor::Tensor> InferencePipeline::settle_input(
    Preprocessor::Tensor& tensor) const {
    if (retain_input) return std::move(tensor);
    if (auto* ref = std::get_if<DspBufferRef>(&tensor)) {
        // a released ref is dead either way; a dropped transport reclaims
        // its buffers at disconnect (python swallows DspError the same way)
        ref->release();
        return std::nullopt;
    }
    return std::move(tensor);
}

const std::string& InferencePipeline::resolve_model(const std::string& call_model) const {
    if (!call_model.empty()) return call_model;
    if (!model_id.empty()) return model_id;
    throw std::invalid_argument("no model_id — pass one to run() or the constructor");
}

PipelineResult InferencePipeline::run(const Source& source, const std::string& call_model) {
    const std::string& model = resolve_model(call_model);
    const auto started = std::chrono::steady_clock::now();
    auto [tensor, meta] = prepare(source);
    InferenceResult result;
    try {
        result = owned_client_->infer(to_infer_input(tensor), model, infer_timeout_ms);
    } catch (...) {
        settle_input(tensor);  // the pass is dead — don't leak its ref
        throw;
    }
    PipelineResult out;
    out.objects = decode(result, meta);
    out.result = std::move(result);
    out.meta = std::move(meta);
    out.latency_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started)
            .count();
    out.tensor = settle_input(tensor);
    return out;
}

std::future<PipelineResult> InferencePipeline::run_async(const Source& source,
                                                         const std::string& call_model) {
    const std::string& model = resolve_model(call_model);
    auto [tensor, meta] = prepare(source);  // inline — the cheap small-image work
    std::future<InferenceResult> inner =
        owned_client_->infer_async(to_infer_input(tensor), model, infer_timeout_ms);
    // decode/settle on the completing thread (python add_done_callback);
    // latency_ms is not measured on this path, like python
    return std::async(std::launch::async,
                      [this, tensor = std::move(tensor), meta = std::move(meta),
                       inner = std::move(inner)]() mutable -> PipelineResult {
                          InferenceResult result;
                          try {
                              result = inner.get();
                          } catch (...) {
                              settle_input(tensor);  // don't leak the ref
                              throw;
                          }
                          PipelineResult out;
                          out.objects = decode(result, meta);
                          out.result = std::move(result);
                          out.meta = meta;
                          out.tensor = settle_input(tensor);
                          return out;
                      });
}

}  // namespace neoruntime_ipc_sdk
