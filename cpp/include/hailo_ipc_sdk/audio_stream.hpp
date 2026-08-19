// audio_stream.hpp — captured-audio frame subscriber. 1:1 port of audio_stream.py.
//
// Reads encoded/raw audio frames from the camera-daemon EncodedPublisher audio
// socket over a raw Unix Domain Socket (the same 30-byte-header family as the
// video EncodedStreamClient, but with an audio-specific field layout). Default
// socket: /run/aipc/encoded/audio_capture.sock (override via AUDIO_CAPTURE_SOCK_PATH
// or the constructor). This is the *receive* path; the *send* path (two-way talk)
// is AudioClient::stream_pcm.
#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>

namespace hailo_ipc_sdk {

// One audio frame from the capture pipeline (mirrors audio_stream.py AudioFrame).
struct AudioFrame {
    std::uint8_t codec = 0;          // 0=pcm, 1=aac, 2=g711a, 3=g711u
    std::uint8_t flags = 0;          // bit0 = keyframe
    std::uint64_t pts_ns = 0;        // presentation timestamp, nanoseconds
    std::uint32_t sample_rate = 0;
    std::uint32_t channels = 0;
    std::uint32_t bits_per_sample = 0;
    std::uint64_t dts_ns = 0;        // decode timestamp; set only when the daemon
                                     // sends the video-style header tail (0 otherwise)
    std::string data;                // raw payload bytes

    bool is_keyframe() const noexcept { return (flags & 0x01) != 0; }
    std::string codec_name() const;  // "pcm" / "aac" / "g711a" / "g711u" / "unknown(N)"
    // Estimated frame duration in ms (PCM only; 0.0 for compressed/unknown).
    double duration_ms() const;
};

class AudioStreamClient {
public:
    // Empty socket_path => $AUDIO_CAPTURE_SOCK_PATH or the default path.
    explicit AudioStreamClient(std::string socket_path = "");
    ~AudioStreamClient();
    AudioStreamClient(const AudioStreamClient&) = delete;
    AudioStreamClient& operator=(const AudioStreamClient&) = delete;
    AudioStreamClient(AudioStreamClient&&) noexcept;
    AudioStreamClient& operator=(AudioStreamClient&&) noexcept;

    // Get one frame, or std::nullopt on timeout / peer-close.
    std::optional<AudioFrame> get_frame(int timeout_ms = 5000);

    // Pull the next frame with automatic reconnect (the subscribe() generator).
    // Returns std::nullopt only after close() (i.e. never, unless closed).
    std::optional<AudioFrame> next_frame(bool reconnect = true);

    // Start a daemon background thread invoking `callback` per frame until close.
    std::thread on_frame(std::function<void(const AudioFrame&)> callback);

    void close();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace hailo_ipc_sdk
