// test_subscribe_health.cpp — subscribe() health machinery parity with
// inference.py (0.8.1): bounded drop-oldest queue, latency/skew EMA seeding,
// sane-window gate, consecutive-failure limiter, and the exact RuntimeError
// text of the failure raise.
#include <gtest/gtest.h>

#include <cstdint>
#include <string>
#include <thread>
#include <vector>

#include "detail/subscribe_health.hpp"

namespace n = neoruntime_ipc_sdk::detail;

// ---- BoundedDropOldestQueue ----------------------------------------------------
TEST(DropOldestQueue, FifoWhenUnderCapacity) {
    n::BoundedDropOldestQueue<int> q(4);
    q.offer(1);
    q.offer(2);
    q.offer(3);
    EXPECT_EQ(q.take(), 1);
    EXPECT_EQ(q.take(), 2);
    EXPECT_EQ(q.take(), 3);
    EXPECT_EQ(q.dropped(), 0u);
}

TEST(DropOldestQueue, OverflowDropsOldestAndCounts) {
    n::BoundedDropOldestQueue<int> q(3);
    for (int v : {1, 2, 3, 4, 5}) q.offer(v);  // 1 and 2 evicted
    EXPECT_EQ(q.dropped(), 2u);
    EXPECT_EQ(q.take(), 3);  // newest always survive
    EXPECT_EQ(q.take(), 4);
    EXPECT_EQ(q.take(), 5);
    EXPECT_EQ(q.dropped(), 2u);  // consuming does not reset the counter
}

TEST(DropOldestQueue, CapacityZeroIsUnbounded) {
    n::BoundedDropOldestQueue<int> q(0);
    for (int i = 0; i < 1000; ++i) q.offer(i);
    EXPECT_EQ(q.dropped(), 0u);
    EXPECT_EQ(q.take(), 0);
    EXPECT_EQ(q.take(), 1);
}

TEST(DropOldestQueue, TerminalItemIsNeverDropped) {
    // Full queue: the terminal (here: -1, offered last by a producer that
    // then stops) may evict a queued result but must itself land — the
    // consumer always observes the stream end.
    n::BoundedDropOldestQueue<int> q(3);
    q.offer(10);
    q.offer(11);
    q.offer(12);
    q.offer(-1);  // terminal; evicts 10

    EXPECT_EQ(q.dropped(), 1u);
    EXPECT_EQ(q.take(), 11);
    EXPECT_EQ(q.take(), 12);
    EXPECT_EQ(q.take(), -1);
}

TEST(DropOldestQueue, TwoThreadHandoff) {
    n::BoundedDropOldestQueue<int> q(8);
    constexpr int kItems = 2000;
    std::thread producer([&q] {
        for (int i = 0; i < kItems; ++i) q.offer(i);
        q.offer(-1);
    });
    int prev = -1;
    int taken = 0;
    for (;;) {
        const int v = q.take();
        if (v == -1) break;
        EXPECT_GT(v, prev);  // FIFO order kept; drop-oldest only punches gaps
        prev = v;
        ++taken;
    }
    producer.join();
    // Conservation: every produced item was either taken or counted dropped.
    EXPECT_EQ(q.dropped(), static_cast<std::uint64_t>(kItems - taken));
}

// ---- Ema (Python seeding + alpha 0.1) --------------------------------------------
TEST(Ema, SeedsThenConverges) {
    n::Ema e;
    EXPECT_DOUBLE_EQ(e.value(), 0.0);
    e.update(100.0);  // first sane sample seeds directly
    EXPECT_DOUBLE_EQ(e.value(), 100.0);
    e.update(200.0);  // 100 + 0.1 * (200 - 100)
    EXPECT_DOUBLE_EQ(e.value(), 110.0);
    e.update(200.0);  // 110 + 0.1 * (200 - 110)
    EXPECT_DOUBLE_EQ(e.value(), 119.0);
}

TEST(Ema, WorksForSkewScaleToo) {
    n::Ema e;
    e.update(18200.0);
    e.update(55100.0);  // 18200 + 0.1 * 36900
    EXPECT_DOUBLE_EQ(e.value(), 21890.0);
}

// ---- sane latency window -----------------------------------------------------------
TEST(SaneLatency, BoundsAreExclusive) {
    EXPECT_FALSE(n::sane_latency_ms(0.0));
    EXPECT_FALSE(n::sane_latency_ms(-1.0));
    EXPECT_FALSE(n::sane_latency_ms(60000.0));
    EXPECT_TRUE(n::sane_latency_ms(0.001));
    EXPECT_TRUE(n::sane_latency_ms(59999.9));
}

// ---- consecutive-failure limiter ----------------------------------------------------
TEST(FailureLimiter, TripsAtLimitAndResets) {
    n::ConsecutiveFailureLimiter lim(3);
    EXPECT_FALSE(lim.note_failure());  // 1
    EXPECT_FALSE(lim.note_failure());  // 2
    EXPECT_TRUE(lim.note_failure());   // 3 -> raise
    EXPECT_EQ(lim.count(), 3u);
    lim.note_success();
    EXPECT_EQ(lim.count(), 0u);
    EXPECT_FALSE(lim.note_failure());
    EXPECT_FALSE(lim.note_failure());
    EXPECT_TRUE(lim.note_failure());   // trips again after the reset
}

TEST(FailureLimiter, ZeroDisables) {
    n::ConsecutiveFailureLimiter lim(0);
    for (int i = 0; i < 1000; ++i) EXPECT_FALSE(lim.note_failure());
}

// ---- exact raise message ------------------------------------------------------------
TEST(FailureMessage, MatchesPythonFString) {
    // Python: f"Stream inference failed {n} consecutive times (stream={s!r},
    // model={m!r}, last frame={f}): {msg!r}" — !r on str renders '...'.
    EXPECT_EQ(
        n::consecutive_failure_message(10, "cam0_main", "yolo", 42, "npu busy"),
        "Stream inference failed 10 consecutive times (stream='cam0_main', "
        "model='yolo', last frame=42): 'npu busy'");
    EXPECT_EQ(
        n::consecutive_failure_message(1, "s", "m", 0, ""),
        "Stream inference failed 1 consecutive times (stream='s', model='m', "
        "last frame=0): ''");
}
