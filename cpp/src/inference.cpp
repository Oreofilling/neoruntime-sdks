// inference.cpp — InferenceClient implementation. See inference.hpp.
//
// Async model: Python runs grpc.aio on a background asyncio loop and bridges
// sync callers via run_coroutine_threadsafe. C++ sync stubs already block on a
// futex (no CQ spin), so that machinery is replaced here by:
//   * sync stubs for every RPC;
//   * std::async for infer_async()/infer_batch_async() — the task captures its
//     own channel/stub so it is safe even if the client is destroyed first.
#include "hailo_ipc_sdk/inference.hpp"

#include <grpcpp/grpcpp.h>

#include <algorithm>
#include <chrono>
#include <cstring>
#include <stdexcept>
#include <utility>

#include "hailo_ipc_sdk/config.hpp"

#include "detail/grpc_channel.hpp"
#include "detail/tensor_io.hpp"
#include "ai-runtime/inference.grpc.pb.h"
#include "ai-runtime/inference.pb.h"

namespace hailo_ipc_sdk {

namespace pb = aipc::inference;

namespace {

// gRPC deadline = now + ms. Centralized so each call site is explicit about the
// slack it adds over the Python timeout.
void deadline_in(grpc::ClientContext& ctx, std::uint32_t ms) {
    ctx.set_deadline(std::chrono::system_clock::now() + std::chrono::milliseconds(ms));
}

BoundingBox bbox_from_proto(const pb::BBox& b) {
    BoundingBox bb;
    bb.x = b.x();
    bb.y = b.y();
    bb.width = b.w();
    bb.height = b.h();
    return bb;
}

TensorSpec tensor_spec_from_proto(const pb::TensorSpec& s) {
    TensorSpec out;
    out.shape.reserve(s.shape_size());
    for (int d : s.shape()) out.shape.push_back(d);
    out.dtype = detail::pb_to_data_type(s.dtype());
    out.name = s.name();
    return out;
}

ModelInfo model_info_from_proto(const pb::ModelInfo& m) {
    ModelInfo out;
    out.model_id = m.model_id();
    out.model_path = m.model_path();
    out.version = m.version();
    out.inputs.reserve(m.inputs_size());
    for (const auto& i : m.inputs()) out.inputs.push_back(tensor_spec_from_proto(i));
    out.outputs.reserve(m.outputs_size());
    for (const auto& o : m.outputs()) out.outputs.push_back(tensor_spec_from_proto(o));
    out.estimated_tops = m.estimated_tops();
    out.estimated_memory = m.estimated_memory();
    out.load_timestamp = m.load_timestamp();
    return out;
}

// Populate the structured-result fields of `out` from a PostResult. Mirrors
// inference.py::_parse_post_result. Best-effort: a malformed sub-message must
// not lose the whole result (Python wraps this in try/except).
void fill_post_result(const pb::PostResult& pr, InferenceResult& out) {
    out.objects.reserve(pr.detections_size());
    for (const auto& det : pr.detections()) {
        DetectedObject o;
        o.label = det.label();
        o.score = det.confidence();
        o.class_id = det.class_id();
        o.bbox = bbox_from_proto(det.bbox());
        out.objects.push_back(std::move(o));
    }
    out.classifications.reserve(pr.classifications_size());
    for (const auto& cls : pr.classifications()) {
        Classification c;
        c.type = cls.type();
        c.class_id = cls.class_id();
        c.label = cls.label();
        c.confidence = cls.confidence();
        out.classifications.push_back(std::move(c));
    }
    for (const auto& set : pr.landmarks()) {
        LandmarkSet ls;
        ls.type = set.type();
        ls.points.reserve(set.points_size());
        for (const auto& p : set.points())
            ls.points.push_back(LandmarkPoint{p.x(), p.y(), p.confidence()});
        out.landmarks.push_back(std::move(ls));
    }
    for (const auto& m : pr.masks()) {
        SegmentationMask sm;
        sm.class_id = m.class_id();
        sm.label = m.label();
        sm.confidence = m.confidence();
        sm.bbox = bbox_from_proto(m.bbox());
        sm.mask_rle = m.mask_rle();
        sm.mask_width = m.mask_width();
        sm.mask_height = m.mask_height();
        out.masks.push_back(std::move(sm));
    }
    for (const auto& line : pr.ocr_lines()) {
        OcrLine ol;
        ol.text = line.text();
        ol.confidence = line.confidence();
        ol.bbox = bbox_from_proto(line.bbox());
        out.ocr_lines.push_back(std::move(ol));
    }
    for (const auto& emb : pr.embeddings()) {
        Embedding e;
        e.dim = emb.dim();
        e.data.assign(emb.data().begin(), emb.data().end());
        out.embeddings.push_back(std::move(e));
    }
    for (const auto& dm : pr.depth_maps()) {
        DepthMap d;
        d.width = dm.width();
        d.height = dm.height();
        cv::Mat mat(static_cast<int>(dm.height()), static_cast<int>(dm.width()), CV_32F);
        const auto& bytes = dm.depth_data();
        const std::size_t want = mat.total() * mat.elemSize();
        std::memcpy(mat.data, bytes.data(), std::min(bytes.size(), want));
        d.data = mat;
        out.depth_maps.push_back(std::move(d));
    }
}

// Decode an InferResponse into an InferenceResult. Shared by infer() /
// infer_batch() / infer_async(). frame_sequence/timestamp_ns stay 0 here
// (only StreamInferResponse carries them).
InferenceResult parse_infer_response(const pb::InferResponse& resp) {
    InferenceResult out;
    if (resp.has_post_result()) {
        try {
            fill_post_result(resp.post_result(), out);
        } catch (...) {
            // Mirrors inference.py's (ValueError, AttributeError) swallow.
        }
    }
    if (resp.outputs_size() > 0) {
        std::vector<cv::Mat> raw;
        raw.reserve(resp.outputs_size());
        for (const auto& t : resp.outputs()) raw.push_back(detail::tensor_to_mat(t));
        out.raw_outputs = std::move(raw);
    }
    out.infer_time_us = resp.infer_time_us();
    out.queue_time_us = resp.queue_time_us();
    out.hw_infer_time_us = resp.hw_infer_time_us();
    if (!resp.status().success()) out.status_message = resp.status().message();
    return out;
}

}  // namespace

// ---------------------------------------------------------------------------
// SegmentationMask
// ---------------------------------------------------------------------------
cv::Mat SegmentationMask::to_mask() const {
    cv::Mat m = cv::Mat::zeros(mask_height, mask_width, CV_8U);
    const auto& data = mask_rle;
    std::size_t pos = 0;
    const auto read_varint = [&]() -> std::uint64_t {
        std::uint64_t val = 0;
        int shift = 0;
        while (pos < data.size()) {
            unsigned char b = static_cast<unsigned char>(data[pos]);
            val |= static_cast<std::uint64_t>(b & 0x7F) << shift;
            ++pos;
            if (!(b & 0x80)) break;
            shift += 7;
        }
        return val;
    };
    const std::size_t total = static_cast<std::size_t>(mask_width) * mask_height;
    while (pos < data.size()) {
        const std::uint64_t start = read_varint();
        const std::uint64_t length = read_varint();
        for (std::uint64_t k = start; k < start + length; ++k) {
            if (k < total) {
                m.at<unsigned char>(static_cast<int>(k / mask_width),
                                    static_cast<int>(k % mask_width)) = 255;
            }
        }
    }
    return m;
}

// ---------------------------------------------------------------------------
// InferenceStream
// ---------------------------------------------------------------------------
struct InferenceStream::Impl {
    grpc::ClientContext ctx;
    std::unique_ptr<grpc::ClientReader<pb::StreamInferResponse>> reader;
};

InferenceStream::InferenceStream() : impl_(std::make_unique<Impl>()) {}
InferenceStream::~InferenceStream() = default;
InferenceStream::InferenceStream(InferenceStream&&) noexcept = default;
InferenceStream& InferenceStream::operator=(InferenceStream&&) noexcept = default;

std::optional<std::pair<std::uint64_t, InferenceResult>> InferenceStream::next() {
    if (!impl_->reader) return std::nullopt;
    pb::StreamInferResponse resp;
    while (impl_->reader->Read(&resp)) {
        if (!resp.status().success()) continue;  // skip failed frames (matches Python)
        InferenceResult out;
        out.frame_sequence = resp.frame_sequence();
        out.timestamp_ns = resp.timestamp_ns();
        if (resp.has_post_result()) {
            try {
                fill_post_result(resp.post_result(), out);
            } catch (...) {
            }
        }
        if (resp.outputs_size() > 0) {
            std::vector<cv::Mat> raw;
            raw.reserve(resp.outputs_size());
            for (const auto& t : resp.outputs()) raw.push_back(detail::tensor_to_mat(t));
            out.raw_outputs = std::move(raw);
        }
        out.status_message = resp.status().message();
        return std::make_pair(resp.frame_sequence(), std::move(out));
    }
    return std::nullopt;
}

// ---------------------------------------------------------------------------
// GenaiStream
// ---------------------------------------------------------------------------
struct GenaiStream::Impl {
    grpc::ClientContext ctx;
    std::unique_ptr<grpc::ClientReader<pb::GenaiGenerateResponse>> reader;
    GenaiFinishReason finish = GenaiFinishReason::Done;
};

GenaiStream::GenaiStream() : impl_(std::make_unique<Impl>()) {}
GenaiStream::~GenaiStream() = default;
GenaiStream::GenaiStream(GenaiStream&&) noexcept = default;
GenaiStream& GenaiStream::operator=(GenaiStream&&) noexcept = default;

std::optional<std::string> GenaiStream::next() {
    if (!impl_->reader) return std::nullopt;
    pb::GenaiGenerateResponse resp;
    while (impl_->reader->Read(&resp)) {
        switch (resp.event_case()) {
            case pb::GenaiGenerateResponse::kToken:
                return std::optional<std::string>(std::string(resp.token()));
            case pb::GenaiGenerateResponse::kFinish:
                impl_->finish = static_cast<GenaiFinishReason>(resp.finish());
                return std::nullopt;
            default:
                break;
        }
    }
    return std::nullopt;
}

GenaiFinishReason GenaiStream::finish_reason() const {
    return impl_->finish;
}

// ---------------------------------------------------------------------------
// InferenceClient
// ---------------------------------------------------------------------------
struct InferenceClient::Impl {
    std::string endpoint;
    std::shared_ptr<grpc::Channel> channel;
    std::unique_ptr<pb::InferenceService::Stub> stub;

    void ensure_connected() {
        if (!stub) {
            channel = detail::make_channel(endpoint);
            stub = pb::InferenceService::NewStub(channel);
        }
    }
};

InferenceClient::InferenceClient(std::string endpoint)
    : impl_(std::make_unique<Impl>()) {
    impl_->endpoint =
        endpoint.empty() ? Config::get_inference_endpoint() : std::move(endpoint);
}

InferenceClient::~InferenceClient() = default;
InferenceClient::InferenceClient(InferenceClient&&) noexcept = default;
InferenceClient& InferenceClient::operator=(InferenceClient&&) noexcept = default;

void InferenceClient::connect() { impl_->ensure_connected(); }

void InferenceClient::close() {
    impl_->stub.reset();
    impl_->channel.reset();
}

bool InferenceClient::connected() const noexcept { return impl_->stub != nullptr; }

// ---- Single / async / batch ----
InferenceResult InferenceClient::infer(const cv::Mat& image,
                                       const std::string& model_id,
                                       std::uint32_t timeout_ms,
                                       std::uint32_t priority,
                                       const std::string& session_id) {
    impl_->ensure_connected();
    pb::InferRequest req;
    req.set_model_id(model_id);
    *req.add_inputs() = detail::mat_to_tensor(image);
    req.set_timeout_ms(timeout_ms);
    req.set_priority(priority);
    req.set_session_id(session_id);

    grpc::ClientContext ctx;
    deadline_in(ctx, timeout_ms + 5000);  // +5s slack for NPU cold start (matches Python)
    pb::InferResponse resp;
    detail::check_grpc(impl_->stub->Infer(&ctx, req, &resp), "Infer");
    if (!resp.status().success())
        throw std::runtime_error("Inference failed: " + resp.status().message());
    return parse_infer_response(resp);
}

std::future<InferenceResult> InferenceClient::infer_async(const cv::Mat& image,
                                                          const std::string& model_id,
                                                          std::uint32_t timeout_ms,
                                                          std::uint32_t priority,
                                                          const std::string& session_id) {
    impl_->ensure_connected();
    pb::InferRequest req;
    req.set_model_id(model_id);
    *req.add_inputs() = detail::mat_to_tensor(image);
    req.set_timeout_ms(timeout_ms);
    req.set_priority(priority);
    req.set_session_id(session_id);

    // Capture a shared channel and build a private stub so the task is safe
    // even if this client is destroyed before the future is consumed.
    auto channel = impl_->channel;
    return std::async(std::launch::async,
                      [channel, req = std::move(req), timeout_ms]() -> InferenceResult {
                          auto stub = pb::InferenceService::NewStub(channel);
                          grpc::ClientContext ctx;
                          deadline_in(ctx, timeout_ms + 5000);
                          pb::InferResponse resp;
                          detail::check_grpc(stub->Infer(&ctx, req, &resp), "Infer");
                          if (!resp.status().success())
                              throw std::runtime_error("Inference failed: " +
                                                       resp.status().message());
                          return parse_infer_response(resp);
                      });
}

std::vector<InferenceResult> InferenceClient::infer_batch(
    const std::vector<BatchInferItem>& items, std::uint32_t timeout_ms) {
    impl_->ensure_connected();
    pb::InferBatchRequest batch;
    for (const auto& it : items) {
        auto* r = batch.add_requests();
        r->set_model_id(it.model_id);
        *r->add_inputs() = detail::mat_to_tensor(it.image);
        r->set_timeout_ms(it.timeout_ms);
        r->set_priority(it.priority);
    }
    batch.set_timeout_ms(timeout_ms);

    grpc::ClientContext ctx;
    deadline_in(ctx, timeout_ms + 5000);
    pb::InferBatchResponse resp;
    detail::check_grpc(impl_->stub->InferBatch(&ctx, batch, &resp), "InferBatch");
    // Partial failure: still return per-item results, in order (matches Python).
    std::vector<InferenceResult> out;
    out.reserve(resp.responses_size());
    for (const auto& r : resp.responses()) out.push_back(parse_infer_response(r));
    return out;
}

std::future<std::vector<InferenceResult>> InferenceClient::infer_batch_async(
    const std::vector<BatchInferItem>& items, std::uint32_t timeout_ms) {
    impl_->ensure_connected();
    pb::InferBatchRequest batch;
    for (const auto& it : items) {
        auto* r = batch.add_requests();
        r->set_model_id(it.model_id);
        *r->add_inputs() = detail::mat_to_tensor(it.image);
        r->set_timeout_ms(it.timeout_ms);
        r->set_priority(it.priority);
    }
    batch.set_timeout_ms(timeout_ms);

    auto channel = impl_->channel;
    return std::async(std::launch::async,
                      [channel, batch = std::move(batch), timeout_ms]()
                          -> std::vector<InferenceResult> {
                          auto stub = pb::InferenceService::NewStub(channel);
                          grpc::ClientContext ctx;
                          deadline_in(ctx, timeout_ms + 5000);
                          pb::InferBatchResponse resp;
                          detail::check_grpc(stub->InferBatch(&ctx, batch, &resp),
                                             "InferBatch");
                          std::vector<InferenceResult> out;
                          out.reserve(resp.responses_size());
                          for (const auto& r : resp.responses())
                              out.push_back(parse_infer_response(r));
                          return out;
                      });
}

std::vector<cv::Mat> InferenceClient::infer_with_tensors(
    const std::string& model_id,
    const std::vector<cv::Mat>& inputs,
    const std::vector<std::string>& input_names,
    std::uint32_t timeout_ms) {
    impl_->ensure_connected();
    // proto Tensor has no name field; Python's per-tensor name is vestigial
    // (_numpy_to_tensor ignores it). input_names is accepted for API parity only.
    (void)input_names;

    pb::InferRequest req;
    req.set_model_id(model_id);
    for (const auto& in : inputs) *req.add_inputs() = detail::mat_to_tensor(in);
    req.set_timeout_ms(timeout_ms);

    grpc::ClientContext ctx;
    deadline_in(ctx, timeout_ms + 5000);
    pb::InferResponse resp;
    detail::check_grpc(impl_->stub->Infer(&ctx, req, &resp), "Infer");
    if (!resp.status().success())
        throw std::runtime_error("Inference failed: " + resp.status().message());

    std::vector<cv::Mat> out;
    out.reserve(resp.outputs_size());
    for (const auto& t : resp.outputs()) out.push_back(detail::tensor_to_mat(t));
    return out;
}

// ---- Stream ----
InferenceStream InferenceClient::subscribe(const std::string& stream,
                                           const std::string& model,
                                           std::uint32_t fps,
                                           const std::string& session_id,
                                           bool raw_output_only) {
    impl_->ensure_connected();
    pb::StreamInferRequest req;
    req.set_model_id(model);
    req.set_stream_id(stream);
    req.set_fps_limit(fps);
    req.set_session_id(session_id);
    req.set_raw_output_only(raw_output_only);

    InferenceStream s;
    s.impl_->reader = impl_->stub->StreamInfer(&s.impl_->ctx, req);
    return s;
}

// ---- Model management ----
std::string InferenceClient::register_model(const std::string& model_path,
                                            const std::string& model_id,
                                            const std::string& owner_id,
                                            const std::string& model_type,
                                            const std::string& model_variant,
                                            const std::vector<TensorSpec>& inputs,
                                            const std::vector<TensorSpec>& outputs) {
    impl_->ensure_connected();
    pb::ModelRegisterRequest req;
    req.set_model_path(Config::translate_path_to_host(model_path));
    req.set_model_id(model_id);
    if (!owner_id.empty()) req.set_owner_id(owner_id);
    if (!model_type.empty()) req.set_model_type(model_type);
    if (!model_variant.empty()) req.set_model_variant(model_variant);
    for (const auto& in : inputs) {
        auto* s = req.add_inputs();
        for (int d : in.shape) s->add_shape(d);
        s->set_dtype(detail::data_type_to_pb(in.dtype));
        s->set_name(in.name);
    }
    for (const auto& out : outputs) {
        auto* s = req.add_outputs();
        for (int d : out.shape) s->add_shape(d);
        s->set_dtype(detail::data_type_to_pb(out.dtype));
        s->set_name(out.name);
    }

    grpc::ClientContext ctx;
    deadline_in(ctx, 125000);  // model load can be slow (Python result_timeout=120)
    pb::ModelRegisterResponse resp;
    detail::check_grpc(impl_->stub->RegisterModel(&ctx, req, &resp), "RegisterModel");
    if (!resp.status().success())
        throw std::runtime_error("Model registration failed: " + resp.status().message());
    return resp.model_id();
}

void InferenceClient::unregister_model(const std::string& model_id) {
    impl_->ensure_connected();
    pb::ModelInfo req;
    req.set_model_id(model_id);
    grpc::ClientContext ctx;
    deadline_in(ctx, 35000);
    pb::Status resp;
    detail::check_grpc(impl_->stub->UnregisterModel(&ctx, req, &resp), "UnregisterModel");
    detail::require_success(resp.success(), resp.message(), "UnregisterModel");
}

std::vector<ModelInfo> InferenceClient::list_models() {
    impl_->ensure_connected();
    grpc::ClientContext ctx;
    deadline_in(ctx, 35000);
    pb::Empty req;
    pb::ModelListResponse resp;
    detail::check_grpc(impl_->stub->ListModels(&ctx, req, &resp), "ListModels");
    std::vector<ModelInfo> out;
    out.reserve(resp.models_size());
    for (const auto& m : resp.models()) out.push_back(model_info_from_proto(m));
    return out;
}

std::optional<ModelInfo> InferenceClient::get_model_info(const std::string& model_id) {
    impl_->ensure_connected();
    pb::ModelInfo req;
    req.set_model_id(model_id);
    grpc::ClientContext ctx;
    deadline_in(ctx, 35000);
    pb::ModelInfo resp;
    detail::check_grpc(impl_->stub->GetModelInfo(&ctx, req, &resp), "GetModelInfo");
    if (resp.model_id().empty()) return std::nullopt;
    // Lite info only — matches inference.py::get_model_info, which deliberately
    // populates just the id/path/version (the full spec comes from list_models).
    ModelInfo info;
    info.model_id = resp.model_id();
    info.model_path = resp.model_path();
    info.version = resp.version();
    return info;
}

// ---- Statistics ----
InferenceSystemStats InferenceClient::get_stats() {
    impl_->ensure_connected();
    grpc::ClientContext ctx;
    deadline_in(ctx, 35000);
    pb::Empty req;
    pb::SystemStats resp;
    detail::check_grpc(impl_->stub->GetStats(&ctx, req, &resp), "GetStats");

    InferenceSystemStats out;
    out.device_utilization = resp.device_utilization();
    out.device_temperature = resp.device_temperature();
    out.total_memory_bytes = resp.total_memory_bytes();
    out.used_memory_bytes = resp.used_memory_bytes();
    out.cpu_utilization = resp.cpu_utilization();
    out.dsp_utilization = resp.dsp_utilization();
    out.ram_total_kib = resp.ram_total_kib();
    out.ram_used_kib = resp.ram_used_kib();
    out.model_stats.reserve(resp.model_stats_size());
    for (const auto& s : resp.model_stats()) {
        ModelStats ms;
        ms.model_id = s.model_id();
        ms.total_inferences = s.total_inferences();
        ms.total_errors = s.total_errors();
        ms.avg_latency_us = s.avg_latency_us();
        ms.current_qps = s.current_qps();
        ms.queue_depth = s.queue_depth();
        ms.hw_fps = s.hw_fps();
        out.model_stats.push_back(std::move(ms));
    }
    return out;
}

// ---- Sessions ----
std::string InferenceClient::create_session(const std::string& session_id,
                                            const std::string& app_id,
                                            const std::vector<std::string>& allowed_models,
                                            std::uint32_t max_qps,
                                            std::uint32_t max_concurrent,
                                            std::uint32_t priority) {
    impl_->ensure_connected();
    pb::SessionConfig req;
    req.set_session_id(session_id);
    req.set_app_id(app_id);
    req.set_max_qps(max_qps);
    req.set_max_concurrent(max_concurrent);
    req.set_priority(priority);
    for (const auto& m : allowed_models) req.add_allowed_models(m);

    grpc::ClientContext ctx;
    deadline_in(ctx, 35000);
    pb::SessionCreateResponse resp;
    detail::check_grpc(impl_->stub->CreateSession(&ctx, req, &resp), "CreateSession");
    if (!resp.status().success())
        throw std::runtime_error("Session creation failed: " + resp.status().message());
    return resp.session_id();
}

void InferenceClient::destroy_session(const std::string& session_id) {
    impl_->ensure_connected();
    pb::SessionConfig req;
    req.set_session_id(session_id);
    grpc::ClientContext ctx;
    deadline_in(ctx, 35000);
    pb::Status resp;
    detail::check_grpc(impl_->stub->DestroySession(&ctx, req, &resp), "DestroySession");
    detail::require_success(resp.success(), resp.message(), "DestroySession");
}

// ---- Postprocess config ----
bool InferenceClient::update_postprocess_config(const std::string& model_id,
                                                const std::string& config_json) {
    impl_->ensure_connected();
    pb::UpdatePostprocessConfigRequest req;
    req.set_model_id(model_id);
    req.set_config_json(config_json);

    grpc::ClientContext ctx;
    deadline_in(ctx, 35000);
    pb::UpdatePostprocessConfigResponse resp;
    detail::check_grpc(impl_->stub->UpdatePostprocessConfig(&ctx, req, &resp),
                       "UpdatePostprocessConfig");
    if (!resp.status().success())
        throw std::runtime_error("UpdatePostprocessConfig failed: " +
                                 resp.status().message());
    return true;
}

// ---- CLIP text encoding ----
std::vector<float> InferenceClient::encode_text(const std::string& text,
                                                std::uint32_t timeout_ms) {
    impl_->ensure_connected();
    pb::EncodeTextRequest req;
    req.set_text(text);
    grpc::ClientContext ctx;
    deadline_in(ctx, timeout_ms + 5000);
    pb::EncodeTextResponse resp;
    detail::check_grpc(impl_->stub->EncodeText(&ctx, req, &resp), "EncodeText");
    if (resp.status().code() != 0)
        throw std::runtime_error("EncodeText failed: " + resp.status().message());
    const auto& d = resp.embedding().data();
    return std::vector<float>(d.begin(), d.end());
}

// ---- GenAI ----
std::string InferenceClient::genai_create_session(const std::string& hef_path,
                                                  GenaiKind kind,
                                                  const std::string& lora_name,
                                                  bool optimize_memory) {
    impl_->ensure_connected();
    pb::GenaiCreateSessionRequest req;
    req.set_hef_path(Config::translate_path_to_host(hef_path));
    req.set_kind(static_cast<pb::GenaiKind>(kind));
    req.set_lora_name(lora_name);
    req.set_optimize_memory(optimize_memory);

    grpc::ClientContext ctx;
    deadline_in(ctx, 305000);  // HEF load is slow (Python result_timeout=300)
    pb::GenaiCreateSessionResponse resp;
    detail::check_grpc(impl_->stub->GenaiCreateSession(&ctx, req, &resp),
                       "GenaiCreateSession");
    if (resp.status().code() != 0)
        throw std::runtime_error("GenAI create session failed: " +
                                 resp.status().message());
    return resp.session_id();
}

void InferenceClient::genai_destroy_session(const std::string& session_id) {
    impl_->ensure_connected();
    // hef_path is reused as the session_id carrier for destroy (matches Python).
    pb::GenaiCreateSessionRequest req;
    req.set_hef_path(session_id);
    grpc::ClientContext ctx;
    deadline_in(ctx, 15000);
    pb::Status resp;
    detail::check_grpc(impl_->stub->GenaiDestroySession(&ctx, req, &resp),
                       "GenaiDestroySession");
    if (resp.code() != 0)
        throw std::runtime_error("GenAI destroy session failed: " + resp.message());
}

GenaiStream InferenceClient::genai_generate(const std::string& session_id,
                                            const std::vector<std::string>& messages,
                                            const std::vector<std::string>& images,
                                            const std::vector<std::string>& stop_tokens,
                                            float temperature,
                                            float top_p,
                                            std::uint32_t top_k,
                                            std::uint32_t max_tokens,
                                            bool do_sample) {
    impl_->ensure_connected();
    pb::GenaiGenerateRequest req;
    req.set_session_id(session_id);
    for (const auto& m : messages) req.add_messages_json(m);
    for (const auto& img : images) req.add_image_frames(img);
    for (const auto& st : stop_tokens) req.add_stop_tokens(st);
    if (do_sample || temperature > 0.0f || max_tokens != 512) {
        auto* p = req.mutable_params();
        p->set_temperature(temperature);
        p->set_top_p(top_p);
        p->set_top_k(top_k);
        p->set_max_generated_tokens(max_tokens);
        p->set_do_sample(do_sample);
    }

    GenaiStream s;
    s.impl_->reader = impl_->stub->GenaiGenerate(&s.impl_->ctx, req);
    return s;
}

void InferenceClient::genai_abort(const std::string& session_id) {
    impl_->ensure_connected();
    pb::GenaiAbortRequest req;
    req.set_session_id(session_id);
    grpc::ClientContext ctx;
    deadline_in(ctx, 10000);
    pb::Status resp;
    // Best-effort: Python does not check the abort status, so neither do we.
    detail::check_grpc(impl_->stub->GenaiAbort(&ctx, req, &resp), "GenaiAbort");
}

}  // namespace hailo_ipc_sdk
