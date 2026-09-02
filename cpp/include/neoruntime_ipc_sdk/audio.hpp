// audio.hpp — audio capture/playback control client. 1:1 port of audio.py.
//
// Drives the audio RPCs of the camera-daemon CameraControl service (package
// `aipc.camera`): device enumeration, capture/playback start/stop, status,
// config (volume/mute), and client-streamed PCM for two-way talk (StreamAudioPcm).
// Transport: sync gRPC stubs over UDS (Config::get_camera_control_endpoint()).
// For *receiving* the captured audio stream (raw UDS), see audio_stream.hpp.
#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace neoruntime_ipc_sdk {

// One audio device entry (mirrors audio.py AudioDevice).
struct AudioDevice {
    std::string name;
    std::string description;
};

// Current audio pipeline state (mirrors audio.py AudioStatus).
struct AudioStatus {
    bool capturing = false;
    bool playing = false;
    std::string device;
    std::uint32_t sample_rate = 0;
    std::uint32_t channels = 0;
    std::string codec;
    float volume = 0.0f;
    bool mute = false;
};

class AudioClient {
public:
    // Empty endpoint => Config::get_camera_control_endpoint().
    explicit AudioClient(std::string endpoint = "");
    ~AudioClient();
    AudioClient(const AudioClient&) = delete;
    AudioClient& operator=(const AudioClient&) = delete;
    AudioClient(AudioClient&&) noexcept;
    AudioClient& operator=(AudioClient&&) noexcept;

    void connect();
    void close();
    bool connected() const noexcept;

    // -- Device enumeration --
    std::vector<AudioDevice> list_capture_devices();
    std::vector<AudioDevice> list_playback_devices();

    // -- Capture. Leave numeric/string params at 0/"" to use daemon defaults. --
    void start_capture(const std::string& device = "", std::uint32_t sample_rate = 0,
                       std::uint32_t channels = 0, const std::string& codec = "",
                       std::uint32_t bitrate = 0);
    void stop_capture();

    // -- Playback (two-way talk sink) --
    void start_playback(const std::string& device = "", std::uint32_t sample_rate = 0,
                        std::uint32_t channels = 0);
    void stop_playback();

    // -- Status & config --
    AudioStatus get_status();
    // volume / mute are std::nullopt => no change (mirrors Python's -1.0 / None
    // sentinels). Other params left at 0/"" keep current values.
    void set_config(const std::string& device = "", std::uint32_t sample_rate = 0,
                    std::uint32_t channels = 0, const std::string& codec = "",
                    std::uint32_t bitrate = 0,
                    std::optional<float> volume = std::nullopt,
                    std::optional<bool> mute = std::nullopt);

    // -- Two-way talk: push PCM to the device --
    // `next` is a pull callback returning the next PCM chunk, or std::nullopt
    // when the stream is exhausted (1:1 with audio.py's pcm_iter iterator).
    void stream_pcm(const std::function<std::optional<std::string>()>& next,
                    std::uint32_t sample_rate = 48000, std::uint32_t channels = 1,
                    const std::string& fmt = "S16LE");
    // Convenience: stream a raw PCM file in `chunk_size`-byte chunks.
    void stream_pcm_file(const std::string& path, int chunk_size = 4096,
                         std::uint32_t sample_rate = 48000, std::uint32_t channels = 1,
                         const std::string& fmt = "S16LE");

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace neoruntime_ipc_sdk
