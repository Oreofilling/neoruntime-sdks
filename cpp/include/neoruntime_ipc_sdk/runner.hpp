// runner.hpp — long-running application skeleton around an
// InferencePipeline (port of runner.py, 0.8.1).
//
// Productises the B-form loop: a reader thread pulls frames from any
// source — typically FdMediaClient's keep-fd subscribe — into a tiny
// latest-wins queue, and a worker thread runs the pipeline and hands each
// result to a sink. When processing is slower than the camera, old frames
// are dropped instead of ballooning memory, keep-fd handles are released
// the moment a frame is consumed, and everything is visible through
// stats().
//
// Frame ownership: frames yielded by source belong to the runner for the
// duration of one pass. The runner releases keep-fd handles after the
// sink returns (evicted frames are released immediately), so **a sink
// must not retain the frame beyond the call** — copy what you need
// (frame.to_array() already copies).
//
//   PipelineRunner runner(
//       [&media] { return media.subscribe_next("sub", /*keep_fd=*/true); },
//       [&pipe](const Frame& f) { return pipe.run(f); },
//       [](PipelineResult& out, const Frame&) { /* use out.objects */ },
//       /*queue_size=*/1);  // latest-wins: always infer the newest frame
//   runner.start();
//   ...
//   runner.stop();
//
// C++ note on stop(): python's threads are daemons; C++ threads must be
// joined, so the destructor stops and joins unconditionally. A source
// callback that blocks forever therefore blocks destruction — give it a
// receive timeout (the SDK's subscribe pulls already reconnect with
// bounded retries and raise, which ends the reader cleanly).
#pragma once

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <exception>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#include "neoruntime_ipc_sdk/frame.hpp"
#include "neoruntime_ipc_sdk/pipeline.hpp"

namespace neoruntime_ipc_sdk {

// stats() snapshot: throughput, drops, latency and error counters.
struct RunnerStats {
    std::string name;
    bool running = false;
    long processed = 0;
    long dropped = 0;
    double fps = 0.0;
    double avg_latency_ms = 0.0;
    double recent_latency_ms = 0.0;  // EMA (alpha 0.2), rounded to 2 decimals
    long errors = 0;
    long consecutive_errors = 0;
    std::optional<std::string> last_error;
    double uptime_s = 0.0;  // rounded to 1 decimal
};

// Run source → pipeline → sink continuously on two threads.
class PipelineRunner {
public:
    // Pull-style frame source: return the next frame, nullopt when
    // exhausted (python's iterator protocol); throw to end the reader
    // with a recorded "source" error.
    using SourceFn = std::function<std::optional<Frame>()>;
    // python "anything with a .run(frame)" — InferencePipeline::run bound,
    // or a test fake.
    using PipelineFn = std::function<PipelineResult(const Frame&)>;
    // Per-pass consumer on the worker thread. Receives the result by
    // reference (it is move-only); the frame is released right after.
    using SinkFn = std::function<void(PipelineResult&, const Frame&)>;
    // Extra per-error hook (python on_error), in addition to the counters.
    using ErrorFn = std::function<void(const std::exception&)>;

    PipelineRunner(SourceFn source, PipelineFn pipeline, SinkFn sink = nullptr,
                   std::size_t queue_size = 1, std::string drop_policy = "latest",
                   ErrorFn on_error = nullptr,
                   std::optional<long> max_consecutive_errors = 10,
                   std::string name = "pipeline");
    ~PipelineRunner();
    PipelineRunner(const PipelineRunner&) = delete;
    PipelineRunner& operator=(const PipelineRunner&) = delete;

    // Start both threads (idempotent-guarded: throws when already
    // started, python semantics).
    PipelineRunner& start();

    // Signal stop, unblock the worker and join both threads. Returns true
    // when both threads exited in time; a reader blocked inside source()
    // may hold it past the deadline. The destructor joins unconditionally.
    bool stop(double timeout_s = 5.0);

    bool running() const;

    RunnerStats stats() const;

    // python's .last_results: the bounded sink-less capture (most recent 16
    // passes), moved out — PipelineResult is move-only (it may carry a
    // device-resident ref), so there is no live-view equivalent. Drains the
    // capture (non-const on purpose); a no-sink runner keeps collecting
    // after a call.
    std::vector<PipelineResult> take_last_results();

private:
    struct QueueItem {
        bool sentinel = false;
        Frame frame;
    };

    // ---- latest-wins queue (python queue.Queue) ----
    class FrameQueue {
    public:
        explicit FrameQueue(std::size_t capacity) : capacity_(capacity) {}
        // Offer with timeout; false = still full after `timeout_s` (the
        // caller evicts the oldest and retries). Takes frame by reference:
        // on failure the frame is left with the caller, exactly like
        // python's q.put(timeout=) — a by-value parameter would destroy a
        // moved-in frame on the timeout path (never queued, never
        // released).
        bool put(Frame& frame, double timeout_s);
        // Offer a terminal sentinel without waiting; false = full.
        bool try_put_sentinel();
        void put_sentinel();  // blocking offer (natural exhaustion)
        QueueItem get();      // blocking
        std::optional<QueueItem> try_get();

    private:
        bool full_locked() const { return items_.size() >= capacity_; }
        std::mutex mutex_;
        std::condition_variable can_put_;
        std::condition_variable can_get_;
        std::deque<QueueItem> items_;
        std::size_t capacity_;
    };

    void reader_loop();
    void worker_loop();
    void enqueue_terminal();
    void note_success(double latency_ms);
    void record_error(const std::exception& exc, const char* stage);
    void set_last_error(const std::string& text);
    std::optional<std::string> last_error_snapshot() const;
    void thread_exited();

    // configuration (python attributes)
    SourceFn source_;
    PipelineFn pipeline_;
    SinkFn sink_;
    std::size_t queue_size_;
    ErrorFn on_error_;
    std::optional<long> max_consecutive_errors_;
    std::string name_;

    // state
    FrameQueue queue_;
    std::atomic<bool> stop_flag_{false};
    std::atomic<bool> started_{false};
    std::atomic<bool> reader_done_{false};
    std::atomic<bool> worker_done_{false};
    std::thread reader_;
    std::thread worker_;
    mutable std::mutex done_mutex_;
    std::condition_variable done_cv_;

    // accounting (single-writer advisory snapshots, python parity)
    std::atomic<long> processed_{0};
    std::atomic<long> dropped_{0};
    std::atomic<long> errors_{0};
    std::atomic<long> consecutive_errors_{0};
    mutable std::mutex error_mutex_;
    std::optional<std::string> last_error_;
    double started_at_ = 0.0;  // steady-clock seconds, worker/reader only read
    double total_latency_ms_ = 0.0;  // worker-only
    double latency_ema_ = 0.0;       // worker-only
    std::mutex results_mutex_;
    std::vector<PipelineResult> last_results_;  // bounded sink-less capture
};

}  // namespace neoruntime_ipc_sdk
