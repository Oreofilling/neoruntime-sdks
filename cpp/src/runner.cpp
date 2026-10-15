// runner.cpp — port of runner.py (0.8.1): reader/worker threads over a
// latest-wins queue, EMA latency stats, error accounting and clean
// sentinel-based shutdown.
#include "neoruntime_ipc_sdk/runner.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace neoruntime_ipc_sdk {

namespace {

constexpr double kPutGraceS = 0.05;       // python _PUT_GRACE_S
constexpr double kEmaAlpha = 0.2;         // python _EMA_ALPHA
constexpr std::size_t kLastResultsKeep = 16;  // python last_results[:-16]

double steady_now_s() {
    return std::chrono::duration<double>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

double round_to(double value, int decimals) {
    const double scale = std::pow(10.0, decimals);
    return std::round(value * scale) / scale;
}

}  // namespace

// ---- FrameQueue -----------------------------------------------------------------------

bool PipelineRunner::FrameQueue::put(Frame& frame, double timeout_s) {
    std::unique_lock<std::mutex> lock(mutex_);
    if (!can_put_.wait_for(lock, std::chrono::duration<double>(timeout_s),
                           [this] { return !full_locked(); }))
        return false;  // frame untouched — the caller retries with it
    QueueItem item;
    item.frame = std::move(frame);
    items_.push_back(std::move(item));
    can_get_.notify_one();
    return true;
}

bool PipelineRunner::FrameQueue::try_put_sentinel() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (full_locked()) return false;
    QueueItem item;
    item.sentinel = true;
    items_.push_back(std::move(item));
    can_get_.notify_one();
    return true;
}

void PipelineRunner::FrameQueue::put_sentinel() {
    // natural exhaustion: the worker must see the end, so this offer
    // blocks until space appears (python's blocking q.put(SENTINEL))
    std::unique_lock<std::mutex> lock(mutex_);
    can_put_.wait(lock, [this] { return !full_locked(); });
    QueueItem item;
    item.sentinel = true;
    items_.push_back(std::move(item));
    can_get_.notify_one();
}

PipelineRunner::QueueItem PipelineRunner::FrameQueue::get() {
    std::unique_lock<std::mutex> lock(mutex_);
    can_get_.wait(lock, [this] { return !items_.empty(); });
    QueueItem item = std::move(items_.front());
    items_.pop_front();
    can_put_.notify_one();
    return item;
}

std::optional<PipelineRunner::QueueItem> PipelineRunner::FrameQueue::try_get() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (items_.empty()) return std::nullopt;
    std::optional<QueueItem> item(std::move(items_.front()));
    items_.pop_front();
    can_put_.notify_one();
    return item;
}

// ---- lifecycle -----------------------------------------------------------------------

PipelineRunner::PipelineRunner(SourceFn source, PipelineFn pipeline, SinkFn sink,
                               std::size_t queue_size, std::string drop_policy,
                               ErrorFn on_error,
                               std::optional<long> max_consecutive_errors, std::string name)
    : source_(std::move(source)),
      pipeline_(std::move(pipeline)),
      sink_(std::move(sink)),
      queue_size_(queue_size),
      on_error_(std::move(on_error)),
      max_consecutive_errors_(max_consecutive_errors),
      name_(std::move(name)),
      queue_(queue_size) {
    if (drop_policy != "latest")
        throw std::invalid_argument(
            "unsupported drop_policy '" + drop_policy +
            "' — only 'latest' (drop the oldest buffered frame, keep the newest) is supported");
    if (queue_size < 1)
        throw std::invalid_argument("queue_size must be >= 1, got " +
                                    std::to_string(queue_size));
}

PipelineRunner::~PipelineRunner() {
    // C++ threads must be joined (python's are daemons) — see header note.
    if (!started_.load()) return;
    stop_flag_.store(true);
    enqueue_terminal();
    if (reader_.joinable()) reader_.join();
    if (worker_.joinable()) worker_.join();
}

PipelineRunner& PipelineRunner::start() {
    bool expected = false;
    if (!started_.compare_exchange_strong(expected, true))
        throw std::runtime_error("PipelineRunner already started");
    stop_flag_.store(false);
    started_at_ = steady_now_s();
    reader_done_.store(false);
    worker_done_.store(false);
    worker_ = std::thread([this] { worker_loop(); });  // worker first, like python
    reader_ = std::thread([this] { reader_loop(); });
    return *this;
}

bool PipelineRunner::stop(double timeout_s) {
    if (!started_.load()) return true;  // never started → nothing running
    stop_flag_.store(true);
    enqueue_terminal();
    {
        std::unique_lock<std::mutex> lock(done_mutex_);
        done_cv_.wait_for(lock, std::chrono::duration<double>(timeout_s), [this] {
            return reader_done_.load() && worker_done_.load();
        });
    }
    if (reader_done_.load() && worker_done_.load()) {
        if (reader_.joinable()) reader_.join();
        if (worker_.joinable()) worker_.join();
    }
    // not finished in time: leave the threads running (a source blocked in
    // a receive is the user's contract) — the destructor joins them
    return !running();
}

bool PipelineRunner::running() const {
    if (!started_.load()) return false;
    return !(reader_done_.load() && worker_done_.load());
}

RunnerStats PipelineRunner::stats() const {
    RunnerStats out;
    out.name = name_;
    out.running = running();
    out.processed = processed_.load(std::memory_order_relaxed);
    out.dropped = dropped_.load(std::memory_order_relaxed);
    out.errors = errors_.load(std::memory_order_relaxed);
    out.consecutive_errors = consecutive_errors_.load(std::memory_order_relaxed);
    out.last_error = last_error_snapshot();
    const double uptime =
        started_.load() ? std::max(0.0, steady_now_s() - started_at_) : 0.0;
    out.fps = uptime > 0.0 ? static_cast<double>(out.processed) / uptime : 0.0;
    // total/EMA are worker-written doubles; reading them racily matches
    // python's advisory (non-locked) stats
    out.avg_latency_ms = out.processed > 0 ? total_latency_ms_ / out.processed : 0.0;
    out.recent_latency_ms = round_to(latency_ema_, 2);
    out.uptime_s = round_to(uptime, 1);
    return out;
}

std::vector<PipelineResult> PipelineRunner::take_last_results() {
    std::lock_guard<std::mutex> lock(results_mutex_);
    return std::move(last_results_);  // move-only results; the capture restarts
}

// ---- threads -----------------------------------------------------------------------

void PipelineRunner::reader_loop() {
    bool exhausted = false;
    while (!stop_flag_.load()) {
        std::optional<Frame> next;
        try {
            next = source_();  // nullopt = StopIteration, throw = source error
        } catch (const std::exception& exc) {
            record_error(exc, "source");
            break;
        } catch (...) {
            record_error(std::runtime_error("unknown error"), "source");
            break;
        }
        if (!next.has_value()) {
            exhausted = true;
            break;
        }
        Frame frame = std::move(*next);
        while (!stop_flag_.load()) {
            if (queue_.put(frame, kPutGraceS)) break;
            // queue stayed full through the grace window — evict the
            // oldest buffered frame and offer again (latest-wins)
            std::optional<QueueItem> evicted = queue_.try_get();
            if (evicted.has_value()) {
                evicted->frame.release();  // no-op for sentinels/empty frames
                dropped_.fetch_add(1, std::memory_order_relaxed);
            }
            // raced empty (another consumer took it) → loop retries the put
        }
    }
    if (exhausted && !stop_flag_.load())
        queue_.put_sentinel();  // natural end: nobody else will signal stop
    else
        enqueue_terminal();
    reader_done_.store(true);
    thread_exited();
}

void PipelineRunner::worker_loop() {
    auto error_stops = [&] {
        const long limit = max_consecutive_errors_.value_or(0);
        return limit > 0 &&
               consecutive_errors_.load(std::memory_order_relaxed) >= limit;
    };
    for (;;) {
        QueueItem item = queue_.get();
        if (item.sentinel) break;
        const double started = steady_now_s();
        bool ok = true;
        try {
            PipelineResult out = pipeline_(item.frame);
            if (sink_) {
                sink_(out, item.frame);
            } else {
                std::lock_guard<std::mutex> lock(results_mutex_);
                last_results_.push_back(std::move(out));
                if (last_results_.size() > kLastResultsKeep)
                    last_results_.erase(last_results_.begin(),
                                        last_results_.end() - kLastResultsKeep);
            }
        } catch (const std::exception& exc) {
            record_error(exc, "pipeline");
            ok = false;
        } catch (...) {
            record_error(std::runtime_error("unknown error"), "pipeline");
            ok = false;
        }
        if (ok) note_success((steady_now_s() - started) * 1000.0);
        item.frame.release();  // python's finally: hand the fd back
        if (!ok && error_stops()) {
            stop_flag_.store(true);  // failing forever — end the run
            break;
        }
    }
    // drain frames still buffered behind the sentinel
    while (std::optional<QueueItem> rest = queue_.try_get()) {
        if (!rest->sentinel) rest->frame.release();
    }
    worker_done_.store(true);
    thread_exited();
}

// ---- accounting -----------------------------------------------------------------------

void PipelineRunner::enqueue_terminal() {
    // python _enqueue_terminal: offer the sentinel without waiting; when
    // the queue is full, shed the oldest buffered frame (dropped++) and
    // retry. Capacity ≥ 1 ⇒ terminates.
    for (;;) {
        if (queue_.try_put_sentinel()) return;
        std::optional<QueueItem> evicted = queue_.try_get();
        if (evicted.has_value()) {
            evicted->frame.release();  // no-op for sentinels/empty frames
            dropped_.fetch_add(1, std::memory_order_relaxed);
        }
    }
}

void PipelineRunner::note_success(double latency_ms) {
    processed_.fetch_add(1, std::memory_order_relaxed);
    consecutive_errors_.store(0, std::memory_order_relaxed);
    total_latency_ms_ += latency_ms;
    if (latency_ema_ == 0.0)
        latency_ema_ = latency_ms;  // first sample seeds the EMA, like python
    else
        latency_ema_ += kEmaAlpha * (latency_ms - latency_ema_);
}

void PipelineRunner::record_error(const std::exception& exc, const char* stage) {
    errors_.fetch_add(1, std::memory_order_relaxed);
    consecutive_errors_.fetch_add(1, std::memory_order_relaxed);
    set_last_error(std::string(stage) + ": " + exc.what());
    if (on_error_) {
        try {
            on_error_(exc);
        } catch (...) {  // the hook must not kill the loop (python parity)
        }
    }
}

void PipelineRunner::set_last_error(const std::string& text) {
    std::lock_guard<std::mutex> lock(error_mutex_);
    last_error_ = text;
}

std::optional<std::string> PipelineRunner::last_error_snapshot() const {
    std::lock_guard<std::mutex> lock(error_mutex_);
    return last_error_;
}

void PipelineRunner::thread_exited() {
    {
        // keep the wakeup ordered with the cv predicate's mutex
        std::lock_guard<std::mutex> lock(done_mutex_);
    }
    done_cv_.notify_all();
}

}  // namespace neoruntime_ipc_sdk
