// stream_pipeline.cpp — port of stream_pipeline.py (0.8.1): subscribe →
// (optional on_result hook) → hardware-overlay annotate → app-facing
// drop-oldest result queue, single-shot start/stop with box/zone
// clearing at teardown.
#include "neoruntime_ipc_sdk/stream_pipeline.hpp"

#include <algorithm>
#include <chrono>
#include <stdexcept>
#include <utility>

namespace neoruntime_ipc_sdk {

namespace {
constexpr auto kJoinTimeout = std::chrono::seconds(5);
}

// ---- ResultQueue -------------------------------------------------------------------

void StreamPipeline::ResultQueue::evict_until_fits_locked() {
    while (slots_.size() >= capacity_) {
        slots_.pop_front();
        ++drops_;
    }
}

void StreamPipeline::ResultQueue::offer(std::uint64_t seq, InferenceResult result) {
    std::lock_guard<std::mutex> lock(mutex_);
    evict_until_fits_locked();
    Slot slot;
    slot.seq = seq;
    slot.result = std::move(result);
    slots_.push_back(std::move(slot));
    can_get_.notify_one();
}

void StreamPipeline::ResultQueue::offer_stop() {
    std::lock_guard<std::mutex> lock(mutex_);
    // the one terminal item, offered last: evicts but is never evicted
    evict_until_fits_locked();
    Slot slot;
    slot.stop = true;
    slots_.push_back(std::move(slot));
    can_get_.notify_all();
}

std::optional<std::pair<std::uint64_t, InferenceResult>>
StreamPipeline::ResultQueue::get() {
    std::unique_lock<std::mutex> lock(mutex_);
    can_get_.wait(lock, [&] { return !slots_.empty() || ended_; });
    if (slots_.empty()) return std::nullopt;  // already ended and drained
    Slot slot = std::move(slots_.front());
    slots_.pop_front();
    if (slot.stop) {
        ended_ = true;  // later get() calls return nullopt immediately
        return std::nullopt;
    }
    return std::make_pair(slot.seq, std::move(slot.result));
}

std::uint64_t StreamPipeline::ResultQueue::drops() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return drops_;
}

// ---- lifecycle -----------------------------------------------------------------------

StreamPipeline::StreamPipeline(
    std::string stream_in, std::string model_in, std::uint32_t fps_in,
    std::string session_id_in, bool raw_output_only_in,
    std::optional<std::uint32_t> max_consecutive_failures_in,
    std::optional<std::uint32_t> queue_size_in, bool draw_in,
    std::optional<OverlayConfig> overlay_config_in,
    std::optional<std::uint32_t> ttl_ms_in, double min_score_in,
    std::optional<std::vector<std::string>> labels_in,
    std::optional<std::vector<OverlayPolygon>> polygons_in, OnResult on_result_in,
    std::size_t result_queue_size_in, InferenceClient* inference_in,
    OverlayClient* overlay_in)
    : stream(std::move(stream_in)),
      model(std::move(model_in)),
      fps(fps_in),
      session_id(std::move(session_id_in)),
      raw_output_only(raw_output_only_in),
      max_consecutive_failures(max_consecutive_failures_in),
      queue_size(queue_size_in),
      draw(draw_in),
      overlay_config(overlay_config_in),
      ttl_ms(ttl_ms_in),
      min_score(min_score_in),
      labels(std::move(labels_in)),
      polygons(std::move(polygons_in)),
      on_result(std::move(on_result_in)),
      result_queue_size_(result_queue_size_in),
      inference_(inference_in),
      overlay_(overlay_in) {
    if (!(min_score >= 0.0 && min_score <= 1.0))
        throw std::invalid_argument("min_score must be in [0, 1], got " +
                                    std::to_string(min_score));
    if (result_queue_size_ < 1)
        throw std::invalid_argument("result_queue_size must be >= 1, got " +
                                    std::to_string(result_queue_size_));
    if (ttl_ms.has_value() && *ttl_ms == 0)
        throw std::invalid_argument("ttl_ms must be a positive int, got " +
                                    std::to_string(*ttl_ms));
    filter_active_ = min_score > 0.0 || labels.has_value();
}

StreamPipeline::~StreamPipeline() { stop(); }

StreamPipeline& StreamPipeline::start() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (started_)
            throw std::runtime_error(
                "this StreamPipeline was already started; "
                "create a new instance for another run");
        started_ = true;
        running_ = true;
    }
    if (inference_ == nullptr) {
        owned_inference_ = std::make_unique<InferenceClient>();
        inference_ = owned_inference_.get();
    }
    if (draw) {
        if (overlay_ == nullptr) {
            owned_overlay_ = std::make_unique<OverlayClient>();
            overlay_ = owned_overlay_.get();
        }
        if (overlay_config.has_value()) overlay_->apply(*overlay_config);
        if (polygons.has_value())
            overlay_->annotate(stream, std::nullopt, *polygons);  // static zones
    }
    results_ = std::make_unique<ResultQueue>(result_queue_size_);
    worker_ = std::thread([this] { run(); });
    return *this;
}

void StreamPipeline::stop() {
    bool first = false;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!started_) return;
        first = !stopped_;
        stopped_ = true;  // cancel + clears + close run exactly once
    }
    if (first) {
        // best-effort wake-up so a worker blocked in next() can exit
        if (iterator_.has_value()) {
            try {
                iterator_->cancel();
            } catch (...) {
            }
        }
    }
    // The join is deliberately NOT part of the once-only teardown. A worker
    // that stopped itself from on_result cannot join itself, so its thread
    // object stays joinable after run() exits — and destroying a joinable
    // std::thread (even a finished one) terminates the process. Whoever
    // calls stop() next from another thread — typically the destructor —
    // joins it here.
    if (worker_.joinable() && worker_.get_id() != std::this_thread::get_id()) {
        if (first) {
            std::unique_lock<std::mutex> lock(mutex_);
            done_cv_.wait_for(lock, kJoinTimeout, [this] { return worker_done_; });
        }
        // bounded in practice: the cancel() above woke the worker
        worker_.join();
    }
    if (!first) return;  // teardown already ran (on the worker or earlier)
    // (self-stop from on_result: teardown below runs on the worker thread)

    if (overlay_ != nullptr) {
        try {
            if (draw)
                overlay_->annotate(stream, std::vector<DetectedObject>{});  // clear boxes
            if (draw && polygons.has_value())
                overlay_->annotate(stream, std::nullopt,
                                   std::vector<OverlayPolygon>{});  // clear zones
        } catch (...) {  // teardown must not throw
        }
    }
    if (owned_inference_) {
        try {
            owned_inference_->close();
        } catch (...) {
        }
    }
    if (owned_overlay_) {
        try {
            owned_overlay_->close();
        } catch (...) {
        }
    }
}

std::optional<std::pair<std::uint64_t, InferenceResult>> StreamPipeline::next_result() {
    if (results_ == nullptr)
        throw std::runtime_error("start() the StreamPipeline before reading results");
    return results_->get();
}

StreamPipelineStatus StreamPipeline::status() const {
    StreamPipelineStatus out;
    const InferenceStream* it = nullptr;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        out.running = running_;
        out.results_seen = results_seen_;
        out.results_annotated = results_annotated_;
        out.annotate_errors = annotate_errors_;
        out.last_error = last_error_;
        it = iterator_.has_value() ? &*iterator_ : nullptr;
    }
    if (it != nullptr) {  // pass-through subscribe telemetry (atomics inside)
        out.subscribe_dropped = it->dropped();
        out.last_latency_ms = it->last_latency_ms();
        out.avg_latency_ms = it->avg_latency_ms();
        out.last_skew_us = it->last_skew_us();
        out.avg_skew_us = it->avg_skew_us();
    }
    if (results_ != nullptr) out.result_queue_drops = results_->drops();
    return out;
}

// ---- worker --------------------------------------------------------------------------

void StreamPipeline::run() {
    try {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            iterator_ = inference_->subscribe(stream, model, fps, session_id,
                                              raw_output_only, max_consecutive_failures,
                                              queue_size);
        }
        for (;;) {
            // iterator_ is assign-once; reading it here without the lock is
            // safe (never reassigned until destruction)
            std::optional<std::pair<std::uint64_t, InferenceResult>> item =
                iterator_->next();
            if (!item.has_value()) break;  // clean end of stream
            handle(item->first, std::move(item->second));
        }
    } catch (const std::exception& exc) {  // surfaced via status().last_error
        std::lock_guard<std::mutex> lock(mutex_);
        last_error_ = exc.what();
    } catch (...) {
        std::lock_guard<std::mutex> lock(mutex_);
        last_error_ = "unknown error";
    }
    {
        std::lock_guard<std::mutex> lock(mutex_);
        running_ = false;
        worker_done_ = true;
    }
    done_cv_.notify_all();
    if (results_) results_->offer_stop();  // terminal item, offered exactly once
}

void StreamPipeline::handle(std::uint64_t seq, InferenceResult result) {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        ++results_seen_;  // counted before the hook runs, like python
    }
    if (on_result) {
        std::optional<InferenceResult> replaced;
        try {
            replaced = on_result(result);
        } catch (const std::exception& exc) {  // app hook must not kill us
            set_last_error(std::string("on_result raised: ") + exc.what());
            return;  // result dropped: neither drawn nor queued
        } catch (...) {
            set_last_error("on_result raised: unknown error");
            return;
        }
        if (!replaced.has_value()) return;  // hook says drop
        result = std::move(*replaced);
    }
    if (draw) annotate(result);
    results_->offer(seq, std::move(result));
}

void StreamPipeline::annotate(const InferenceResult& result) {
    try {
        if (filter_active_) {
            // Filtered detections go out directly: an empty kept-list must
            // publish (clearing stale boxes) instead of falling through
            // annotate_result's classifications precedence.
            std::vector<DetectedObject> kept;
            for (const DetectedObject& obj : result.objects) {
                if (obj.score < min_score) continue;
                if (labels.has_value() &&
                    std::find(labels->begin(), labels->end(), obj.label) ==
                        labels->end())
                    continue;
                kept.push_back(obj);
            }
            overlay_->annotate(stream, kept, std::nullopt, ttl_ms, session_id);
        } else {
            overlay_->annotate_result(stream, result, ttl_ms, session_id);
        }
    } catch (const std::exception& exc) {
        std::lock_guard<std::mutex> lock(mutex_);
        ++annotate_errors_;
        last_error_ = std::string("annotate failed: ") + exc.what();
        return;  // drawing is never fatal
    } catch (...) {
        std::lock_guard<std::mutex> lock(mutex_);
        ++annotate_errors_;
        last_error_ = "annotate failed: unknown error";
        return;
    }
    {
        std::lock_guard<std::mutex> lock(mutex_);
        ++results_annotated_;
    }
}

void StreamPipeline::set_last_error(const std::string& text) {
    std::lock_guard<std::mutex> lock(mutex_);
    last_error_ = text;
}

}  // namespace neoruntime_ipc_sdk
