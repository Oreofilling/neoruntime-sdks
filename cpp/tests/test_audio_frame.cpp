// test_audio_frame.cpp — AudioFrame decode logic (codec_name / is_keyframe /
// duration_ms). These run against the real audio_stream.cpp objects (linked via
// ne503::aipc_sdk) but need no socket — the AudioFrame is a plain value struct.
#include <string>

#include <gtest/gtest.h>

#include "hailo_ipc_sdk/audio_stream.hpp"

using hailo_ipc_sdk::AudioFrame;

// ---- codec_name ------------------------------------------------------------
TEST(AudioFrame, CodecNameKnown) {
    AudioFrame f;
    f.codec = 0; EXPECT_EQ(f.codec_name(), "pcm");
    f.codec = 1; EXPECT_EQ(f.codec_name(), "aac");
    f.codec = 2; EXPECT_EQ(f.codec_name(), "g711a");
    f.codec = 3; EXPECT_EQ(f.codec_name(), "g711u");
}

TEST(AudioFrame, CodecNameUnknownIncludesValue) {
    AudioFrame f;
    f.codec = 9;
    EXPECT_EQ(f.codec_name(), "unknown(9)");
}

// ---- is_keyframe -----------------------------------------------------------
TEST(AudioFrame, IsKeyframeFlagBit0) {
    AudioFrame f;
    f.flags = 0x00; EXPECT_FALSE(f.is_keyframe());
    f.flags = 0x01; EXPECT_TRUE(f.is_keyframe());
    f.flags = 0x02; EXPECT_FALSE(f.is_keyframe());  // only bit0 counts
    f.flags = 0x03; EXPECT_TRUE(f.is_keyframe());
}

// ---- duration_ms (PCM only; compressed/unknown => 0) -----------------------
TEST(AudioFrame, DurationMsPcmMonoS16) {
    // 48000 Hz, mono, 16-bit, 9600 bytes => 4800 samples => 100.0 ms.
    AudioFrame f;
    f.codec = 0;            // pcm
    f.sample_rate = 48000;
    f.channels = 1;
    f.bits_per_sample = 16;
    f.data.assign(9600, '\0');
    EXPECT_NEAR(f.duration_ms(), 100.0, 1e-6);
}

TEST(AudioFrame, DurationMsPcmStereoS16) {
    // 48000 Hz, stereo, 16-bit, 19200 bytes => 4800 samples => 100.0 ms.
    AudioFrame f;
    f.codec = 0;
    f.sample_rate = 48000;
    f.channels = 2;
    f.bits_per_sample = 16;
    f.data.assign(19200, '\0');
    EXPECT_NEAR(f.duration_ms(), 100.0, 1e-6);
}

TEST(AudioFrame, DurationMsCompressedIsZero) {
    AudioFrame f;
    f.codec = 1;            // aac — duration not derivable from byte count
    f.sample_rate = 48000;
    f.channels = 1;
    f.bits_per_sample = 16;
    f.data.assign(9600, '\0');
    EXPECT_DOUBLE_EQ(f.duration_ms(), 0.0);
}

TEST(AudioFrame, DurationMsZeroWhenSampleRateMissing) {
    AudioFrame f;
    f.codec = 0;
    f.sample_rate = 0;       // guards the formula's > 0 precondition
    f.channels = 1;
    f.bits_per_sample = 16;
    f.data.assign(9600, '\0');
    EXPECT_DOUBLE_EQ(f.duration_ms(), 0.0);
}

TEST(AudioFrame, DurationMsZeroWhenEmptyPayload) {
    AudioFrame f;
    f.codec = 0;
    f.sample_rate = 48000;
    f.channels = 1;
    f.bits_per_sample = 16;
    // data left empty -> 0 samples -> 0 ms
    EXPECT_DOUBLE_EQ(f.duration_ms(), 0.0);
}
