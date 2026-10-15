// subscribe_health.hpp — client-side stream health machinery for
// InferenceClient::subscribe: the port of inference.py's queue bridge +
// _SubscribeIterator telemetry (0.8.1, subscribe ~lines 560-761).
#pragma once

#include <condition_variable>
#include <cstdint>
#include <deque>
#include <mutex>
#include <string>
#include <utility>

namespace neoruntime_ipc_sdk::detail {

// Bounded handoff with drop-oldest overflow: a fast producer evicts its
// OLDEST queued item (counted) instead of blocking or growing memory — the
// newest always gets through. capacity 0 = unbounded. take() blocks on a
// condvar (a futex), the same shape as Python's queue.Queue.get().
//
// The one terminal item (end or error) is offered exactly once, last, by the
// producer — drop-oldest may evict a queued result to admit it, but since
// nothing follows it, the terminal can never be evicted itself: backpressure
// can delay the terminal, never drop it. The consumer always sees it.
template <typename T>
class BoundedDropOldestQueue {
public:
    explicit BoundedDropOldestQueue(std::size_t capacity = 0) : capacity_(capacity) {}

    BoundedDropOldestQueue(const BoundedDropOldestQueue&) = delete;
    BoundedDropOldestQueue& operator=(const BoundedDropOldestQueue&) = delete;

    // Enqueue with drop-oldest overflow (see class comment).
    void offer(T item) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (capacity_ != 0 && items_.size() >= capacity_) {
            items_.pop_front();
            ++dropped_;
        }
        items_.push_back(std::move(item));
        cv_.notify_one();
    }

    // Block until an item is available. The producer guarantees a terminal
    // item eventually arrives, so this cannot hang past stream end; cancel()
    // paths offer an extra terminal to wake a blocked consumer immediately.
    T take() {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait(lock, [this] { return !items_.empty(); });
        T item = std::move(items_.front());
        items_.pop_front();
        return item;
    }

    std::uint64_t dropped() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return dropped_;
    }

private:
    const std::size_t capacity_;  // 0 = unbounded
    mutable std::mutex mutex_;
    std::condition_variable cv_;
    std::deque<T> items_;
    std::uint64_t dropped_ = 0;
};

// Exponential moving average with Python-parity seeding: value 0.0 means "no
// sample yet" — the first sane sample seeds directly; later samples move the
// average by alpha = 0.1 (inference.py:675-678, 686-689).
class Ema {
public:
    void update(double sample) {
        if (value_ == 0.0)
            value_ = sample;
        else
            value_ += 0.1 * (sample - value_);
    }
    double value() const { return value_; }

private:
    double value_ = 0.0;
};

// Latency sanity gate: recv-minus-capture only means something when both
// stamps come from comparable clocks; outside (0, 60 s) the sample is
// discarded rather than poisoning the EMA (inference.py:673).
inline bool sane_latency_ms(double ms) { return ms > 0.0 && ms < 60000.0; }

// Consecutive-failure limiter for subscribe(): failed frames are skipped up
// to `max` in a row (0 = disabled); the limit tripping is the signal to
// raise instead of skipping forever (inference.py:691-714).
class ConsecutiveFailureLimiter {
public:
    explicit ConsecutiveFailureLimiter(std::uint32_t max_failures) : max_(max_failures) {}

    // Record a failed frame; true when the raise limit tripped.
    bool note_failure() { return max_ != 0 && ++count_ >= max_; }
    void note_success() { count_ = 0; }
    std::uint32_t count() const { return count_; }

private:
    std::uint32_t max_;
    std::uint32_t count_ = 0;
};

// Exact port of inference.py's RuntimeError f-string (with !r quoting) so
// Python-side tooling and behavior tests match messages verbatim:
//   Stream inference failed 10 consecutive times (stream='cam0_main',
//   model='yolo', last frame=42): 'npu busy'
inline std::string consecutive_failure_message(std::uint32_t count,
                                               const std::string& stream,
                                               const std::string& model,
                                               std::uint64_t frame_sequence,
                                               const std::string& message) {
    return "Stream inference failed " + std::to_string(count) +
           " consecutive times (stream='" + stream + "', model='" + model +
           "', last frame=" + std::to_string(frame_sequence) + "): '" + message + "'";
}

}  // namespace neoruntime_ipc_sdk::detail
