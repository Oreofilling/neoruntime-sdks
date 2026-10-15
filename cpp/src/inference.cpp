// inference.cpp — InferenceClient implementation. See inference.hpp.
//
// Async model: Python runs grpc.aio on a background asyncio loop and bridges
// sync callers via run_coroutine_threadsafe. C++ sync stubs already block on a
// futex (no CQ spin), so that machinery is replaced here by:
//   * sync stubs for every RPC;
//   * std::async for infer_async()/infer_batch_async() — the task captures its
//     own channel/stub so it is safe even if the client is destroyed first.
//
// subscribe(): a pump thread bridges the server stream into a
// BoundedDropOldestQueue (drop-oldest overflow, terminal item guaranteed
// delivered); the consumer applies the health policy (consecutive-failure
// limit, latency/skew EMA) exactly as inference.py's generator does.
#include "neoruntime_ipc_sdk/inference.hpp"

#include <grpcpp/grpcpp.h>

#include <algorithm>
#include <chrono>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <utility>
#include <variant>

#include "neoruntime_ipc_sdk/config.hpp"
#include "neoruntime_ipc_sdk/frame.hpp"

#include "detail/grpc_channel.hpp"
#include "detail/inference_codec.hpp"
#include "detail/subscribe_health.hpp"
#include "detail/transport.hpp"
#include "detail/tensor_io.hpp"
#include "ai-runtime/inference.grpc.pb.h"
#include "ai-runtime/inference.pb.h"

namespace neoruntime_ipc_sdk {

namespace pb = aipc::inference;

namespace {

// gRPC deadline = now + ms. Centralized so each call site is explicit about the
// slack it adds over the Python timeout.
void deadline_in(grpc::ClientContext& ctx, std::uint32_t ms) {
    ctx.set_deadline(std::chrono::system_clock::now() + std::chrono::milliseconds(ms));
}

// One queued item on the subscribe bridge: a server response, the terminal
// stream error, or the terminal end marker (Python: response | Exception |
// SENTINEL on the same queue).
struct StreamItem {
    enum class Kind { kResponse, kError, kEnd };
    Kind kind = Kind::kEnd;
    pb::StreamInferResponse response;
    std::string error;
};

// Wall-clock nanoseconds — the C++ stand-in for time.time_ns() in the
// latency EMA (both stamps are only compared inside the sane window gate).
std::int64_t wall_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::system_clock::now().time_since_epoch())
        .count();
}

pb::InferRequest infer_request(const pb::Tensor& tensor,
                               const std::string& model_id,
                               std::uint32_t timeout_ms,
                               std::uint32_t priority,
                               const std::string& session_id) {
    pb::InferRequest req;
    req.set_model_id(model_id);
    *req.add_inputs() = tensor;
    req.set_timeout_ms(timeout_ms);
    req.set_priority(priority);
    req.set_session_id(session_id);
    return req;
}

InferenceResult run_infer(pb::InferenceService::Stub& stub,
                          const pb::InferRequest& req,
                          std::uint32_t timeout_ms) {
    grpc::ClientContext ctx;
    deadline_in(ctx, timeout_ms + 5000);  // +5s slack for NPU cold start (matches Python)
    pb::InferResponse resp;
    detail::check_grpc(stub.Infer(&ctx, req, &resp), "Infer");
    if (!resp.status().success())
        throw std::runtime_error("Inference failed: " + resp.status().message());
    return detail::inference_codec::parse_infer_response(resp);
}

// Resolve the InferInput variant to the wire tensor. The fd-backed members
// materialize through the handle here (mapped once, cached in the Frame) —
// same pixels, one copy; the zero-copy DSP import lands with DspClient in
// Phase 5. BufferRefInput references the daemon-side registry by id only.
pb::Tensor tensor_from_input(const InferInput& in) {
    if (const auto* m = std::get_if<cv::Mat>(&in)) return detail::mat_to_tensor(*m);
    if (const auto* f = std::get_if<Frame>(&in)) {
        // Copy first: to_array() materializes+ caches, so it is non-const.
        // The copy is shallow (Mat header + shared handle ref).
        Frame fr = *f;
        return detail::mat_to_tensor(fr.image.empty() ? fr.to_array() : fr.image);
    }
    if (const auto* h = std::get_if<std::shared_ptr<FrameHandle>>(&in)) {
        Frame tmp;
        tmp.handle = *h;
        tmp.width = (*h)->width;
        tmp.height = (*h)->height;
        tmp.format = (*h)->format;
        return detail::mat_to_tensor(tmp.to_array());
    }
    const auto& b = std::get<BufferRefInput>(in);
    return detail::inference_codec::buffer_ref_tensor(b.buffer_id, b.width, b.height);
}

InferenceSystemStats system_stats_from(const pb::SystemStats& resp) {
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
    for (const auto& s : resp.model_stats())
        out.model_stats.push_back(detail::inference_codec::inference_stats_from_proto(s));
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
    Impl() = default;
    Impl(std::size_t queue_capacity,
         std::uint32_t max_failures,
         std::string stream_name,
         std::string model_name)
        : queue(queue_capacity),
          limiter(max_failures),
          stream(std::move(stream_name)),
          model(std::move(model_name)) {}

    grpc::ClientContext ctx;
    std::unique_ptr<grpc::ClientReader<pb::StreamInferResponse>> reader;
    std::thread pump;
    detail::BoundedDropOldestQueue<StreamItem> queue{0};

    detail::ConsecutiveFailureLimiter limiter{0};
    const std::string stream;
    const std::string model;

    // Telemetry (written by next(), read by the accessors from any thread).
    std::mutex telemetry_mutex;
    detail::Ema latency_ema;
    detail::Ema skew_ema;
    double last_latency_ms_ = 0.0;
    std::int64_t last_skew_us_ = 0;

    // Latched by the terminal item / close(); once ended, next() mirrors an
    // exhausted Python generator: the first error was raised once, then
    // nullopt forever.
    std::mutex end_mutex;
    bool ended_ = false;

    std::once_flag cancel_once;

    // Cross-thread stop: TryCancel unblocks a pump stuck in Read; the extra
    // kEnd wakes a consumer blocked in take(). Idempotent (once_flag); the
    // unread duplicate terminal is harmless — the queue dies with the stream.
    void cancel() {
        std::call_once(cancel_once, [this] {
            ctx.TryCancel();
            queue.offer(StreamItem{});
        });
    }

    ~Impl() {
        cancel();
        if (pump.joinable()) pump.join();
    }
};

InferenceStream::InferenceStream() : impl_(std::make_unique<Impl>()) {}
InferenceStream::~InferenceStream() = default;
InferenceStream::InferenceStream(InferenceStream&&) noexcept = default;
InferenceStream& InferenceStream::operator=(InferenceStream&&) noexcept = default;

std::optional<std::pair<std::uint64_t, InferenceResult>> InferenceStream::next() {
    for (;;) {
        {
            std::lock_guard<std::mutex> lock(impl_->end_mutex);
            if (impl_->ended_) return std::nullopt;
        }

        StreamItem item = impl_->queue.take();
        if (item.kind == StreamItem::Kind::kEnd) {
            std::lock_guard<std::mutex> lock(impl_->end_mutex);
            impl_->ended_ = true;
            return std::nullopt;
        }
        if (item.kind == StreamItem::Kind::kError) {
            {
                std::lock_guard<std::mutex> lock(impl_->end_mutex);
                impl_->ended_ = true;
            }
            throw std::runtime_error(item.error);
        }

        const pb::StreamInferResponse& resp = item.response;

        // Telemetry before the failure check — inference.py's order.
        const double latency_ms =
            static_cast<double>(wall_ns() - static_cast<std::int64_t>(resp.timestamp_ns())) / 1e6;
        {
            std::lock_guard<std::mutex> lock(impl_->telemetry_mutex);
            if (detail::sane_latency_ms(latency_ms)) {
                impl_->last_latency_ms_ = latency_ms;
                impl_->latency_ema.update(latency_ms);
            }
            // Skew is server-computed on the device clock — valid regardless
            // of any host/device offset that limits latency above; only
            // measured (>0) samples feed the EMA (inference.py:683-689).
            if (resp.skew_us() > 0) {
                impl_->last_skew_us_ = resp.skew_us();
                impl_->skew_ema.update(static_cast<double>(resp.skew_us()));
            }
        }

        if (!resp.status().success()) {
            if (impl_->limiter.note_failure()) {
                const std::string msg = detail::consecutive_failure_message(
                    impl_->limiter.count(), impl_->stream, impl_->model,
                    resp.frame_sequence(), resp.status().message());
                {
                    std::lock_guard<std::mutex> lock(impl_->end_mutex);
                    impl_->ended_ = true;
                }
                throw std::runtime_error(msg);
            }
            continue;  // skip failed frame (matches Python)
        }
        impl_->limiter.note_success();

        return std::make_pair(resp.frame_sequence(),
                              detail::inference_codec::parse_stream_response(resp));
    }
}

void InferenceStream::close() {
    {
        std::lock_guard<std::mutex> lock(impl_->end_mutex);
        impl_->ended_ = true;
    }
    impl_->cancel();
}

void InferenceStream::cancel() { impl_->cancel(); }

std::uint64_t InferenceStream::dropped() const noexcept { return impl_->queue.dropped(); }

double InferenceStream::last_latency_ms() const noexcept {
    std::lock_guard<std::mutex> lock(impl_->telemetry_mutex);
    return impl_->last_latency_ms_;
}

double InferenceStream::avg_latency_ms() const noexcept {
    std::lock_guard<std::mutex> lock(impl_->telemetry_mutex);
    return impl_->latency_ema.value();
}

std::int64_t InferenceStream::last_skew_us() const noexcept {
    std::lock_guard<std::mutex> lock(impl_->telemetry_mutex);
    return impl_->last_skew_us_;
}

double InferenceStream::avg_skew_us() const noexcept {
    std::lock_guard<std::mutex> lock(impl_->telemetry_mutex);
    return impl_->skew_ema.value();
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
struct InferenceClient::Impl : detail::GrpcClientBase<pb::InferenceService> {
    explicit Impl(std::string endpoint)
        : GrpcClientBase(std::move(endpoint)) {}
};

InferenceClient::InferenceClient(std::string endpoint)
    : impl_(std::make_unique<Impl>(
          endpoint.empty() ? Config::get_inference_endpoint() : std::move(endpoint))) {}

InferenceClient::~InferenceClient() = default;
InferenceClient::InferenceClient(InferenceClient&&) noexcept = default;
InferenceClient& InferenceClient::operator=(InferenceClient&&) noexcept = default;

void InferenceClient::connect() { impl_->connect(); }

void InferenceClient::close() { impl_->close(); }

bool InferenceClient::connected() const noexcept { return impl_->connected(); }

// ---- Single / async / batch ----
InferenceResult InferenceClient::infer(const InferInput& image,
                                       const std::string& model_id,
                                       std::uint32_t timeout_ms,
                                       std::uint32_t priority,
                                       const std::string& session_id) {
    const pb::InferRequest req =
        infer_request(tensor_from_input(image), model_id, timeout_ms, priority, session_id);
    return run_infer(impl_->stub(), req, timeout_ms);
}

InferenceResult InferenceClient::infer(const cv::Mat& image,
                                       const std::string& model_id,
                                       std::uint32_t timeout_ms,
                                       std::uint32_t priority,
                                       const std::string& session_id) {
    const pb::InferRequest req =
        infer_request(detail::mat_to_tensor(image), model_id, timeout_ms, priority, session_id);
    return run_infer(impl_->stub(), req, timeout_ms);
}

std::future<InferenceResult> InferenceClient::infer_async(const InferInput& image,
                                                          const std::string& model_id,
                                                          std::uint32_t timeout_ms,
                                                          std::uint32_t priority,
                                                          const std::string& session_id) {
    pb::InferRequest req =
        infer_request(tensor_from_input(image), model_id, timeout_ms, priority, session_id);

    // Capture a shared channel and build a private stub so the task is safe
    // even if this client is destroyed before the future is consumed.
    auto channel = impl_->channel();
    return std::async(std::launch::async,
                      [channel, req = std::move(req), timeout_ms]() -> InferenceResult {
                          auto stub = pb::InferenceService::NewStub(channel);
                          return run_infer(*stub, req, timeout_ms);
                      });
}

std::vector<InferenceResult> InferenceClient::infer_batch(
    const std::vector<BatchInferItem>& items, std::uint32_t timeout_ms) {
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
    detail::check_grpc(impl_->stub().InferBatch(&ctx, batch, &resp), "InferBatch");
    // Partial failure: still return per-item results, in order (matches Python).
    std::vector<InferenceResult> out;
    out.reserve(resp.responses_size());
    for (const auto& r : resp.responses())
        out.push_back(detail::inference_codec::parse_infer_response(r));
    return out;
}

std::future<std::vector<InferenceResult>> InferenceClient::infer_batch_async(
    const std::vector<BatchInferItem>& items, std::uint32_t timeout_ms) {
    pb::InferBatchRequest batch;
    for (const auto& it : items) {
        auto* r = batch.add_requests();
        r->set_model_id(it.model_id);
        *r->add_inputs() = detail::mat_to_tensor(it.image);
        r->set_timeout_ms(it.timeout_ms);
        r->set_priority(it.priority);
    }
    batch.set_timeout_ms(timeout_ms);

    auto channel = impl_->channel();
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
                              out.push_back(detail::inference_codec::parse_infer_response(r));
                          return out;
                      });
}

std::vector<cv::Mat> InferenceClient::infer_with_tensors(
    const std::string& model_id,
    const std::vector<cv::Mat>& inputs,
    const std::vector<std::string>& input_names,
    std::uint32_t timeout_ms) {
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
    detail::check_grpc(impl_->stub().Infer(&ctx, req, &resp), "Infer");
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
                                           bool raw_output_only,
                                           std::optional<std::uint32_t> max_consecutive_failures,
                                           std::optional<std::uint32_t> queue_size) {
    pb::StreamInferRequest req = detail::inference_codec::stream_infer_request(
        stream, model, fps, session_id, raw_output_only);

    InferenceStream s;
    // 0/nullopt disables either bound (Python truthiness of the params).
    s.impl_ = std::make_unique<InferenceStream::Impl>(
        queue_size ? static_cast<std::size_t>(*queue_size) : 0,
        max_consecutive_failures ? *max_consecutive_failures : 0u,
        stream, model);
    InferenceStream::Impl* impl = s.impl_.get();
    impl->reader = impl_->stub().StreamInfer(&impl->ctx, req);

    // Pump: bridge the server stream into the bounded queue; the terminal
    // item (error, or end on clean close/cancel) is offered exactly once,
    // last — backpressure can delay it, never drop it.
    impl->pump = std::thread([impl] {
        pb::StreamInferResponse resp;
        while (impl->reader->Read(&resp))
            impl->queue.offer(StreamItem{StreamItem::Kind::kResponse, std::move(resp), {}});
        const grpc::Status st = impl->reader->Finish();
        if (!st.ok() && st.error_code() != grpc::StatusCode::CANCELLED)
            impl->queue.offer(
                StreamItem{StreamItem::Kind::kError, {}, st.error_message()});
        else
            impl->queue.offer(StreamItem{});
    });
    return s;
}

// ---- Model management ----
std::string InferenceClient::register_model(const std::string& model_path,
                                            const std::string& model_id,
                                            const std::string& owner_id,
                                            const std::string& model_type,
                                            const std::string& model_variant,
                                            const std::vector<TensorSpec>& inputs,
                                            const std::vector<TensorSpec>& outputs,
                                            std::uint32_t batch_size) {
    pb::ModelRegisterRequest req;
    req.set_model_path(Config::translate_path_to_host(model_path));
    req.set_model_id(model_id);
    if (!owner_id.empty()) req.set_owner_id(owner_id);
    if (!model_type.empty()) req.set_model_type(model_type);
    if (!model_variant.empty()) req.set_model_variant(model_variant);
    for (const auto& in : inputs)
        detail::inference_codec::tensor_spec_to_proto(in, req.add_inputs());
    for (const auto& out : outputs)
        detail::inference_codec::tensor_spec_to_proto(out, req.add_outputs());
    // 1 is the proto default — only an explicit batched session carries it.
    if (batch_size > 1) req.set_batch_size(batch_size);

    grpc::ClientContext ctx;
    deadline_in(ctx, 125000);  // model load can be slow (Python result_timeout=120)
    pb::ModelRegisterResponse resp;
    detail::check_grpc(impl_->stub().RegisterModel(&ctx, req, &resp), "RegisterModel");
    if (!resp.status().success())
        throw std::runtime_error("Model registration failed: " + resp.status().message());
    return resp.model_id();
}

void InferenceClient::unregister_model(const std::string& model_id) {
    pb::ModelInfo req;
    req.set_model_id(model_id);
    grpc::ClientContext ctx;
    deadline_in(ctx, 35000);
    pb::Status resp;
    detail::check_grpc(impl_->stub().UnregisterModel(&ctx, req, &resp), "UnregisterModel");
    detail::require_success(resp.success(), resp.message(), "UnregisterModel");
}

std::vector<ModelInfo> InferenceClient::list_models() {
    grpc::ClientContext ctx;
    deadline_in(ctx, 35000);
    pb::Empty req;
    pb::ModelListResponse resp;
    detail::check_grpc(impl_->stub().ListModels(&ctx, req, &resp), "ListModels");
    std::vector<ModelInfo> out;
    out.reserve(resp.models_size());
    for (const auto& m : resp.models())
        out.push_back(detail::inference_codec::model_info_from_proto(m));
    return out;
}

std::optional<ModelInfo> InferenceClient::get_model_info(const std::string& model_id) {
    pb::ModelInfo req;
    req.set_model_id(model_id);
    grpc::ClientContext ctx;
    deadline_in(ctx, 35000);
    pb::ModelInfo resp;
    detail::check_grpc(impl_->stub().GetModelInfo(&ctx, req, &resp), "GetModelInfo");
    if (resp.model_id().empty()) return std::nullopt;
    // Full parse — matches inference.py 0.8.1 (specs + batch included).
    return detail::inference_codec::model_info_from_proto(resp);
}

// ---- Statistics ----
InferenceSystemStats InferenceClient::get_stats() {
    // Python None branch sends Empty, which the server parses into
    // GetStatsRequest as all-defaults (its own default window). The C++ stub
    // is typed, so send the default-constructed request — identical bytes.
    grpc::ClientContext ctx;
    deadline_in(ctx, 35000);
    pb::GetStatsRequest req;
    pb::SystemStats resp;
    detail::check_grpc(impl_->stub().GetStats(&ctx, req, &resp), "GetStats");
    return system_stats_from(resp);
}

InferenceSystemStats InferenceClient::get_stats(std::uint32_t sampling_window_ms) {
    grpc::ClientContext ctx;
    deadline_in(ctx, 35000);
    pb::GetStatsRequest req;
    // Client-side clamp to [1, 5000] (inference.py:942) — the blocking HAL
    // window must stay bounded no matter what the caller passes.
    req.set_sampling_window_ms(std::max(1u, std::min(5000u, sampling_window_ms)));
    pb::SystemStats resp;
    detail::check_grpc(impl_->stub().GetStats(&ctx, req, &resp), "GetStats");
    return system_stats_from(resp);
}

// ---- Sessions ----
std::string InferenceClient::create_session(const std::string& session_id,
                                            const std::string& app_id,
                                            const std::vector<std::string>& allowed_models,
                                            std::uint32_t max_qps,
                                            std::uint32_t max_concurrent,
                                            std::uint32_t priority) {
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
    detail::check_grpc(impl_->stub().CreateSession(&ctx, req, &resp), "CreateSession");
    if (!resp.status().success())
        throw std::runtime_error("Session creation failed: " + resp.status().message());
    return resp.session_id();
}

void InferenceClient::destroy_session(const std::string& session_id) {
    pb::SessionConfig req;
    req.set_session_id(session_id);
    grpc::ClientContext ctx;
    deadline_in(ctx, 35000);
    pb::Status resp;
    detail::check_grpc(impl_->stub().DestroySession(&ctx, req, &resp), "DestroySession");
    detail::require_success(resp.success(), resp.message(), "DestroySession");
}

// ---- Postprocess config ----
bool InferenceClient::update_postprocess_config(const std::string& model_id,
                                                const std::string& config_json) {
    pb::UpdatePostprocessConfigRequest req;
    req.set_model_id(model_id);
    req.set_config_json(config_json);

    grpc::ClientContext ctx;
    deadline_in(ctx, 35000);
    pb::UpdatePostprocessConfigResponse resp;
    detail::check_grpc(impl_->stub().UpdatePostprocessConfig(&ctx, req, &resp),
                       "UpdatePostprocessConfig");
    if (!resp.status().success())
        throw std::runtime_error("UpdatePostprocessConfig failed: " +
                                 resp.status().message());
    return true;
}

// ---- CLIP text encoding ----
std::vector<float> InferenceClient::encode_text(const std::string& text,
                                                std::uint32_t timeout_ms) {
    pb::EncodeTextRequest req;
    req.set_text(text);
    grpc::ClientContext ctx;
    deadline_in(ctx, timeout_ms + 5000);
    pb::EncodeTextResponse resp;
    detail::check_grpc(impl_->stub().EncodeText(&ctx, req, &resp), "EncodeText");
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
    pb::GenaiCreateSessionRequest req;
    req.set_hef_path(Config::translate_path_to_host(hef_path));
    req.set_kind(static_cast<pb::GenaiKind>(kind));
    req.set_lora_name(lora_name);
    req.set_optimize_memory(optimize_memory);

    grpc::ClientContext ctx;
    deadline_in(ctx, 305000);  // HEF load is slow (Python result_timeout=300)
    pb::GenaiCreateSessionResponse resp;
    detail::check_grpc(impl_->stub().GenaiCreateSession(&ctx, req, &resp),
                       "GenaiCreateSession");
    if (resp.status().code() != 0)
        throw std::runtime_error("GenAI create session failed: " +
                                 resp.status().message());
    return resp.session_id();
}

void InferenceClient::genai_destroy_session(const std::string& session_id) {
    // hef_path is reused as the session_id carrier for destroy (matches Python).
    pb::GenaiCreateSessionRequest req;
    req.set_hef_path(session_id);
    grpc::ClientContext ctx;
    deadline_in(ctx, 15000);
    pb::Status resp;
    detail::check_grpc(impl_->stub().GenaiDestroySession(&ctx, req, &resp),
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
    s.impl_->reader = impl_->stub().GenaiGenerate(&s.impl_->ctx, req);
    return s;
}

void InferenceClient::genai_abort(const std::string& session_id) {
    pb::GenaiAbortRequest req;
    req.set_session_id(session_id);
    grpc::ClientContext ctx;
    deadline_in(ctx, 10000);
    pb::Status resp;
    // Best-effort: Python does not check the abort status, so neither do we.
    detail::check_grpc(impl_->stub().GenaiAbort(&ctx, req, &resp), "GenaiAbort");
}

}  // namespace neoruntime_ipc_sdk
