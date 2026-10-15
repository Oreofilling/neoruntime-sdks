// test_runner.cpp — PipelineRunner parity (port of python
// tests/test_runner.py). python's FakeFrame (seq + a released flag) maps to
// a Frame carrying a FrameHandle whose ReleaseFn bumps a shared counter —
// release observability with no wire. python's sink/last_results asserts
// run after the threads joined; same here (assert after stop()).
#include <gtest/gtest.h>

#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <functional>
#include <future>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "neoruntime_ipc_sdk/frame.hpp"
#include "neoruntime_ipc_sdk/pipeline.hpp"
#include "neoruntime_ipc_sdk/runner.hpp"

using namespace neoruntime_ipc_sdk;

namespace {

template <typename Pred>
bool wait_until(Pred pred, double timeout_s = 2.0) {
    const auto deadline = std::chrono::steady_clock::now() +
                          std::chrono::duration<double>(timeout_s);
    while (std::chrono::steady_clock::now() < deadline) {
        if (pred()) return true;
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    return pred();
}

// python FakeFrame: a numbered frame whose release is observable.
struct ReleaseCounter {
    std::atomic<int> releases{0};
};

Frame fake_frame(std::uint64_t seq, std::shared_ptr<ReleaseCounter> counter = {}) {
    Frame frame;
    frame.sequence = seq;
    frame.width = 4;
    frame.height = 4;
    frame.format = "GRAY8";
    if (counter)
        frame.handle = std::make_shared<FrameHandle>(
            std::vector<int>{}, std::array<std::uint32_t, 3>{},
            std::array<std::uint32_t, 3>{}, seq,
            [counter](FrameHandle&) { counter->releases.fetch_add(1); }, 4, 4,
            "GRAY8", 0);
    return frame;
}

// python FakePipeline: records seen seqs, optionally explodes.
struct FakePipeline {
    bool fail = false;
    std::vector<std::uint64_t> seen;

    PipelineResult operator()(const Frame& frame) {
        if (fail)
            throw std::runtime_error("infer exploded on " +
                                     std::to_string(frame.sequence));
        seen.push_back(frame.sequence);
        return PipelineResult{};
    }
};

// python iter(frames): a fixed list handed out once, then exhausted.
struct ListSource {
    std::vector<Frame> frames;
    std::size_t next = 0;

    std::optional<Frame> operator()() {
        if (next >= frames.size()) return std::nullopt;
        return frames[next++];  // copies share the handle, like python's refs
    }
};

}  // namespace

// ---- TestLifecycle -------------------------------------------------------------------

TEST(Lifecycle, RejectsBadConfig) {
    FakePipeline pipeline;
    auto pipe = [&pipeline](const Frame& frame) { return pipeline(frame); };
    try {
        PipelineRunner runner([]() -> std::optional<Frame> { return std::nullopt; },
                              pipe, nullptr, /*queue_size=*/1, /*drop_policy=*/"oldest");
        FAIL() << "expected std::invalid_argument for drop_policy";
    } catch (const std::invalid_argument& exc) {
        EXPECT_NE(std::string(exc.what()).find("drop_policy"), std::string::npos);
    }
    try {
        PipelineRunner runner([]() -> std::optional<Frame> { return std::nullopt; },
                              pipe, nullptr, /*queue_size=*/0);
        FAIL() << "expected std::invalid_argument for queue_size";
    } catch (const std::invalid_argument& exc) {
        EXPECT_NE(std::string(exc.what()).find("queue_size"), std::string::npos);
    }
}

TEST(Lifecycle, DoubleStartRejected) {
    ListSource src;  // empty: exhausts immediately, but started_ stays true
    FakePipeline pipeline;
    PipelineRunner runner([&src] { return src(); },
                          [&pipeline](const Frame& frame) { return pipeline(frame); });
    runner.start();
    try {
        runner.start();
        FAIL() << "expected std::runtime_error";
    } catch (const std::runtime_error& exc) {
        EXPECT_NE(std::string(exc.what()).find("already started"), std::string::npos);
    }
    runner.stop();
}

TEST(Lifecycle, SourceExhaustionStopsRunner) {
    std::vector<std::shared_ptr<ReleaseCounter>> counters;
    ListSource src;
    for (std::uint64_t i = 0; i < 5; ++i) {
        counters.push_back(std::make_shared<ReleaseCounter>());
        src.frames.push_back(fake_frame(i, counters.back()));
    }
    std::vector<std::uint64_t> got;
    FakePipeline pipeline;

    PipelineRunner runner(
        [&src] { return src(); },
        [&pipeline](const Frame& frame) { return pipeline(frame); },
        [&got](PipelineResult&, const Frame& frame) { got.push_back(frame.sequence); });
    runner.start();
    EXPECT_TRUE(wait_until([&] { return !runner.running(); }));
    runner.stop();

    EXPECT_EQ(runner.stats().processed, 5);
    EXPECT_EQ(got, (std::vector<std::uint64_t>{0, 1, 2, 3, 4}));
    for (const auto& counter : counters) EXPECT_EQ(counter->releases.load(), 1);
}

TEST(Lifecycle, StopIsGraceful) {
    // python's endless generator: yields every 10ms until the test says stop.
    // stop() must return while the source can still produce (the reader
    // re-checks the flag between sleeps) and leave both threads joined.
    std::atomic<bool> stop_source{false};
    PipelineRunner runner(
        [&stop_source]() -> std::optional<Frame> {
            if (stop_source.load()) return std::nullopt;
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            return fake_frame(0);
        },
        [](const Frame&) { return PipelineResult{}; });
    runner.start();
    EXPECT_TRUE(runner.running());
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    EXPECT_TRUE(runner.stop());
    stop_source.store(true);
    EXPECT_FALSE(runner.running());
}

// ---- TestLatestWins --------------------------------------------------------------------

TEST(LatestWins, SlowConsumerDropsOldFrames) {
    std::vector<std::shared_ptr<ReleaseCounter>> counters;
    ListSource src;
    for (std::uint64_t i = 0; i < 3; ++i) {
        counters.push_back(std::make_shared<ReleaseCounter>());
        src.frames.push_back(fake_frame(i, counters.back()));
    }
    std::promise<void> gate_promise;
    std::shared_future<void> gate = gate_promise.get_future().share();
    std::vector<std::uint64_t> got;
    FakePipeline pipeline;

    PipelineRunner runner(
        [&src] { return src(); },
        [&pipeline](const Frame& frame) { return pipeline(frame); },
        [&gate, &got](PipelineResult&, const Frame& frame) {
            gate.wait_for(std::chrono::seconds(2));  // worker parks → backlog builds
            got.push_back(frame.sequence);
        },
        /*queue_size=*/1);
    runner.start();
    // Frame 0 reached the worker before the backlog; frame 1 is shed after
    // the grace period; frame 2 (the newest) is queued.
    EXPECT_TRUE(wait_until([&] { return runner.stats().dropped == 1; }));
    gate_promise.set_value();
    EXPECT_TRUE(wait_until([&] { return !runner.running(); }));
    runner.stop();

    EXPECT_EQ(got, (std::vector<std::uint64_t>{0, 2}));  // newest after the shed
    EXPECT_EQ(runner.stats().dropped, 1);
    for (const auto& counter : counters) EXPECT_EQ(counter->releases.load(), 1);
}

TEST(LatestWins, FastConsumerNeverDrops) {
    // Grace period: a worker that keeps up must not lose frames to the
    // reader/worker race on a size-1 queue.
    ListSource src;
    for (std::uint64_t i = 0; i < 50; ++i) src.frames.push_back(fake_frame(i));
    FakePipeline pipeline;
    PipelineRunner runner([&src] { return src(); },
                          [&pipeline](const Frame& frame) { return pipeline(frame); });
    runner.start();
    EXPECT_TRUE(wait_until([&] { return !runner.running(); }));
    runner.stop();

    EXPECT_EQ(runner.stats().dropped, 0);
    EXPECT_EQ(runner.stats().processed, 50);
    EXPECT_EQ(pipeline.seen.size(), 50u);
}

TEST(LatestWins, SinklessModeCollectsBoundedResults) {
    ListSource src;
    for (std::uint64_t i = 0; i < 20; ++i) src.frames.push_back(fake_frame(i));
    FakePipeline pipeline;
    PipelineRunner runner([&src] { return src(); },
                          [&pipeline](const Frame& frame) { return pipeline(frame); },
                          nullptr, /*queue_size=*/4);
    runner.start();
    EXPECT_TRUE(wait_until([&] { return !runner.running(); }));
    runner.stop();

    std::vector<PipelineResult> results = runner.take_last_results();
    EXPECT_LE(results.size(), 16u);
    EXPECT_FALSE(results.empty());
    EXPECT_EQ(runner.stats().processed, 20);
}

// ---- TestErrorPolicy ---------------------------------------------------------------------

TEST(ErrorPolicy, ConsecutiveErrorsStopTheRunner) {
    FakePipeline pipeline;
    pipeline.fail = true;
    ListSource src;
    for (std::uint64_t i = 0; i < 50; ++i) src.frames.push_back(fake_frame(i));
    PipelineRunner runner([&src] { return src(); },
                          [&pipeline](const Frame& frame) { return pipeline(frame); },
                          nullptr, /*queue_size=*/1, "latest", nullptr,
                          /*max_consecutive_errors=*/3);
    runner.start();
    EXPECT_TRUE(wait_until([&] { return !runner.running(); }));
    runner.stop();

    const RunnerStats stats = runner.stats();
    EXPECT_EQ(stats.consecutive_errors, 3);
    ASSERT_TRUE(stats.last_error.has_value());
    EXPECT_NE(stats.last_error->find("infer exploded"), std::string::npos);
}

TEST(ErrorPolicy, SuccessResetsErrorCounter) {
    struct Flaky {
        int calls = 0;
        PipelineResult operator()(const Frame&) {
            if (++calls % 2) throw std::runtime_error("transient");  // every other
            return PipelineResult{};
        }
    } pipeline;
    std::vector<std::string> errors;
    ListSource src;
    for (std::uint64_t i = 0; i < 10; ++i) src.frames.push_back(fake_frame(i));
    PipelineRunner runner(
        [&src] { return src(); },
        [&pipeline](const Frame& frame) { return pipeline(frame); }, nullptr,
        /*queue_size=*/1, "latest",
        [&errors](const std::exception& exc) { errors.push_back(exc.what()); },
        /*max_consecutive_errors=*/3);
    runner.start();
    EXPECT_TRUE(wait_until([&] { return !runner.running(); }));
    runner.stop();

    EXPECT_EQ(runner.stats().processed, 5);
    EXPECT_EQ(runner.stats().consecutive_errors, 0);  // reset by successes
    EXPECT_EQ(errors.size(), 5u);
}
