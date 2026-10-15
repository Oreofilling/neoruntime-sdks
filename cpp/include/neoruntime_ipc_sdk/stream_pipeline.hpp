// stream_pipeline.hpp — whole-pipeline facade: one call wiring
// capture -> inference -> overlay (port of stream_pipeline.py, 0.8.1).
//
// This is the **platform-scheduled** pipeline: the camera daemon feeds
// frames to the model (InferenceClient::subscribe), so no pixels ever
// cross into the app process; results come back over the stream, get
// pushed to the hardware overlay via the event bus (OverlayClient), and
// the annotated video ships inside the encoded stream — the app only
// ever sees lightweight InferenceResult structs. InferencePipeline, by
// contrast, is the **client-side** convenience: it pulls frames into the
// app and is the right tool when the app needs the pixels (custom
// preprocessing, multi-model chains, its own rasterizer).
//
// Choose by where the pixels should live: keep them on the device and
// take results — StreamPipeline; need them in your process —
// InferencePipeline.
//
// One instance is single-shot: start() once, stop() once; build a new
// instance for another run. stop() clears boxes + zones on the stream.
//
//   std::vector<OverlayPolygon> zones{...};
//   StreamPipeline pipe("main", "yolov5n", /*fps=*/10);
//   pipe.start();
//   while (auto item = pipe.next_result()) {        // optional, app side
//       auto [seq, result] = std::move(*item);
//   }
//   pipe.stop();
#pragma once

#include <condition_variable>
#include <cstdint>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "neoruntime_ipc_sdk/inference.hpp"
#include "neoruntime_ipc_sdk/overlay.hpp"

namespace neoruntime_ipc_sdk {

// Point-in-time health snapshot from StreamPipeline::status().
// results_seen counts every result the worker accepted from the
// subscription, *before* the on_result hook runs — results the hook
// dropped or replaced are included. last/avg_latency_ms likewise time
// every accepted result, including ones later dropped by the hook.
struct StreamPipelineStatus {
    bool running = false;
    std::uint64_t results_seen = 0;
    std::uint64_t results_annotated = 0;
    std::uint64_t annotate_errors = 0;
    std::uint64_t result_queue_drops = 0;
    std::uint64_t subscribe_dropped = 0;  // pass-through subscribe health
    std::string last_error;               // "" when none, like python
    double last_latency_ms = 0.0;
    double avg_latency_ms = 0.0;
    std::int64_t last_skew_us = 0;
    double avg_skew_us = 0.0;
};

// One-call pipeline: subscribe, draw on the hardware overlay, observe.
class StreamPipeline {
public:
    // Per-result hook on the worker thread. Return a (possibly replaced)
    // result to flow onward, or nullopt to drop it (not drawn, not
    // queued) — put tracking/filtering logic here. A thrown exception
    // drops that result, is recorded in status().last_error and is
    // non-fatal.
    using OnResult = std::function<std::optional<InferenceResult>(const InferenceResult&)>;

    // draw: push results to the hardware overlay. false also suppresses
    // static polygons — the encoded stream stays clean; results still
    // flow to next_result(). polygons: static zone polygons pushed once
    // at start and cleared at stop (nullopt = none; an empty vector
    // clears zones). inference/overlay: inject clients (tests / reuse);
    // injected clients are NOT closed by stop().
    StreamPipeline(std::string stream, std::string model, std::uint32_t fps = 10,
                   std::string session_id = "", bool raw_output_only = false,
                   std::optional<std::uint32_t> max_consecutive_failures = 10,
                   std::optional<std::uint32_t> queue_size = 100, bool draw = true,
                   std::optional<OverlayConfig> overlay_config = std::nullopt,
                   std::optional<std::uint32_t> ttl_ms = std::nullopt,
                   double min_score = 0.0,
                   std::optional<std::vector<std::string>> labels = std::nullopt,
                   std::optional<std::vector<OverlayPolygon>> polygons = std::nullopt,
                   OnResult on_result = nullptr, std::size_t result_queue_size = 100,
                   InferenceClient* inference = nullptr, OverlayClient* overlay = nullptr);
    ~StreamPipeline();
    StreamPipeline(const StreamPipeline&) = delete;
    StreamPipeline& operator=(const StreamPipeline&) = delete;

    // Subscribe, apply overlay config / static zones, start drawing.
    // Single-shot: throws when already started.
    StreamPipeline& start();

    // Stop the worker, clear what was drawn, close owned clients.
    // Idempotent, and a no-op before start(). The worker is joined
    // first so the box-clearing annotate cannot race a final result.
    // Safe to call from on_result (i.e. from the worker thread itself):
    // the self-join is skipped and the teardown runs on the caller's
    // thread — a later stop() or the destructor joins the self-stopped
    // worker (a joinable std::thread must be joined before destruction).
    void stop();

    // Blocking pull of the next (frame_sequence, result), post-callback.
    // nullopt = the worker stopped (stop() or a fatal stream error).
    // Throws at call time (not on the first pull) when not started. The
    // queue is bounded by result_queue_size with drop-oldest overflow —
    // a slow consumer loses its oldest results, counted in
    // status().result_queue_drops.
    std::optional<std::pair<std::uint64_t, InferenceResult>> next_result();

    // Snapshot of pipeline health and pass-through subscribe stats.
    StreamPipelineStatus status() const;

    // configuration (python attributes; set before start())
    std::string stream;
    std::string model;
    std::uint32_t fps = 10;
    std::string session_id;
    bool raw_output_only = false;
    std::optional<std::uint32_t> max_consecutive_failures = 10;
    std::optional<std::uint32_t> queue_size = 100;
    bool draw = true;
    std::optional<OverlayConfig> overlay_config;
    std::optional<std::uint32_t> ttl_ms;
    double min_score = 0.0;
    std::optional<std::vector<std::string>> labels;
    std::optional<std::vector<OverlayPolygon>> polygons;
    OnResult on_result;

private:
    // ---- app-facing result queue (python queue.Queue + _STOP) ----
    class ResultQueue {
    public:
        explicit ResultQueue(std::size_t capacity) : capacity_(capacity) {}
        // Drop-oldest offer, never blocks: evicts queued results until
        // the item fits. The stop marker, offered last, evicts but is
        // never evicted (nothing is offered after it).
        void offer(std::uint64_t seq, InferenceResult result);
        void offer_stop();
        // Blocking; nullopt = stop marker seen (or already ended).
        std::optional<std::pair<std::uint64_t, InferenceResult>> get();
        std::uint64_t drops() const;

    private:
        struct Slot {
            bool stop = false;
            std::uint64_t seq = 0;
            InferenceResult result;
        };
        void evict_until_fits_locked();

        mutable std::mutex mutex_;
        std::condition_variable can_get_;
        std::deque<Slot> slots_;
        std::size_t capacity_;
        bool ended_ = false;
        std::uint64_t drops_ = 0;
    };

    void run();
    void handle(std::uint64_t seq, InferenceResult result);
    void annotate(const InferenceResult& result);
    void set_last_error(const std::string& text);

    std::size_t result_queue_size_;
    bool filter_active_ = false;  // min_score > 0 || labels set

    std::unique_ptr<InferenceClient> owned_inference_;
    std::unique_ptr<OverlayClient> owned_overlay_;
    InferenceClient* inference_ = nullptr;  // injected or owned
    OverlayClient* overlay_ = nullptr;      // injected or owned (draw only)

    std::thread worker_;
    std::optional<InferenceStream> iterator_;  // assigned once by worker
    std::unique_ptr<ResultQueue> results_;
    mutable std::mutex mutex_;  // counters/flags/iterator_ assignment
    std::condition_variable done_cv_;
    bool started_ = false;  // single-shot: set once, never cleared
    bool stopped_ = false;  // stop() idempotency
    bool running_ = false;
    bool worker_done_ = false;  // worker signalled exit (join handshake)
    std::uint64_t results_seen_ = 0;
    std::uint64_t results_annotated_ = 0;
    std::uint64_t annotate_errors_ = 0;
    std::string last_error_;
};

}  // namespace neoruntime_ipc_sdk
