// test_overlay_strict.cpp — strict-mode wire-shape parity, ported from
// python/tests/test_overlay_strict.py. The Python suite asserts on
// msg.has_strict_frame_lock(); here the same assertions run against
// detail::overlay_codec::ai_overlay_config (the single choke point used by
// enable/disable/configure/apply).
#include <gtest/gtest.h>

#include <cstdint>
#include <optional>
#include <stdexcept>

#include "neoruntime_ipc_sdk/overlay.hpp"

#include "detail/overlay_codec.hpp"

using namespace neoruntime_ipc_sdk;
namespace codec = detail::overlay_codec;
namespace pb = aipc::camera;

TEST(OverlayStrict, PlainConfigSendsNeitherStrictField) {
    OverlayConfig c;  // defaults: enabled, labels on, strict fields nullopt
    pb::AiOverlayConfig cfg = codec::ai_overlay_config(c);
    EXPECT_FALSE(cfg.has_strict_frame_lock());
    EXPECT_FALSE(cfg.has_strict_wait_cap_ms());
    // Non-strict fields are plain scalars — always present on the wire.
    EXPECT_TRUE(cfg.enabled());
    EXPECT_TRUE(cfg.show_label());
}

TEST(OverlayStrict, OptInSetsLockOnly) {
    OverlayConfig c;
    c.strict_frame_lock = true;
    pb::AiOverlayConfig cfg = codec::ai_overlay_config(c);
    ASSERT_TRUE(cfg.has_strict_frame_lock());
    EXPECT_TRUE(cfg.strict_frame_lock());
    EXPECT_FALSE(cfg.has_strict_wait_cap_ms());
}

TEST(OverlayStrict, ExplicitFalseIsSent) {
    // Opt-OUT must go on the wire too — nullopt is the only "leave daemon
    // yaml alone" value.
    OverlayConfig c;
    c.strict_frame_lock = false;
    pb::AiOverlayConfig cfg = codec::ai_overlay_config(c);
    ASSERT_TRUE(cfg.has_strict_frame_lock());
    EXPECT_FALSE(cfg.strict_frame_lock());
}

TEST(OverlayStrict, CapZeroIsSent) {
    // 0 = derive cap from stream fps; a real value, not "unset".
    OverlayConfig c;
    c.strict_wait_cap_ms = 0;
    pb::AiOverlayConfig cfg = codec::ai_overlay_config(c);
    ASSERT_TRUE(cfg.has_strict_wait_cap_ms());
    EXPECT_EQ(cfg.strict_wait_cap_ms(), 0u);
    EXPECT_FALSE(cfg.has_strict_frame_lock());
}

TEST(OverlayStrict, BothStrictFieldsTogether) {
    OverlayConfig c;
    c.strict_frame_lock = true;
    c.strict_wait_cap_ms = 66;
    pb::AiOverlayConfig cfg = codec::ai_overlay_config(c);
    EXPECT_TRUE(cfg.has_strict_frame_lock());
    EXPECT_TRUE(cfg.has_strict_wait_cap_ms());
    EXPECT_EQ(cfg.strict_wait_cap_ms(), 66u);
}

TEST(OverlayStrict, CapValidationBoundaries) {
    // nullopt, 0 and 5000 all pass; >5000 throws client-side, before any
    // RPC. (Python also rejects -1/"66"/1.5/True — unrepresentable through
    // the typed optional<uint32_t>, so the type system carries those.)
    EXPECT_NO_THROW(codec::validate_strict_wait_cap(std::nullopt));
    EXPECT_NO_THROW(codec::validate_strict_wait_cap(0));
    EXPECT_NO_THROW(codec::validate_strict_wait_cap(5000));
    EXPECT_THROW(codec::validate_strict_wait_cap(5001), std::invalid_argument);
}
