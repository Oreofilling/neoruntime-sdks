// audio.cpp — AudioClient implementation. See audio.hpp.
//
// Port of audio.py over the audio RPCs of the `aipc.camera` CameraControl
// service. Shares the CameraControl::Stub with CameraClient. Status-bearing RPCs
// return a top-level bool success / string message. StreamAudioPcm is the only
// client-streaming RPC (a ClientWriter). pb::Empty is camera.proto's own Empty.
#include "neoruntime_ipc_sdk/audio.hpp"

#include <grpcpp/grpcpp.h>

#include <cstdint>
#include <fstream>
#include <functional>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "neoruntime_ipc_sdk/config.hpp"

#include "detail/grpc_channel.hpp"
#include "camera-daemon/camera.grpc.pb.h"
#include "camera-daemon/camera.pb.h"

namespace neoruntime_ipc_sdk {

namespace pb = aipc::camera;

struct AudioClient::Impl {
    std::string endpoint;
    std::shared_ptr<grpc::Channel> channel;
    std::unique_ptr<pb::CameraControl::Stub> stub;

    void ensure_connected() {
        if (!stub) {
            channel = detail::make_channel(endpoint);
            stub = pb::CameraControl::NewStub(channel);
        }
    }
};

AudioClient::AudioClient(std::string endpoint) : impl_(std::make_unique<Impl>()) {
    impl_->endpoint = endpoint.empty() ? Config::get_camera_control_endpoint()
                                       : std::move(endpoint);
}

AudioClient::~AudioClient() = default;
AudioClient::AudioClient(AudioClient&&) noexcept = default;
AudioClient& AudioClient::operator=(AudioClient&&) noexcept = default;

void AudioClient::connect() { impl_->ensure_connected(); }

void AudioClient::close() {
    impl_->stub.reset();
    impl_->channel.reset();
}

bool AudioClient::connected() const noexcept { return impl_->stub != nullptr; }

// Anonymous helper: fill the common AudioConfigRequest fields.
namespace {
void fill_config(pb::AudioConfigRequest& req, const std::string& device,
                 std::uint32_t sample_rate, std::uint32_t channels,
                 const std::string& codec, std::uint32_t bitrate) {
    req.set_device(device);
    req.set_sample_rate(sample_rate);
    req.set_channels(channels);
    req.set_codec(codec);
    req.set_bitrate(bitrate);
}
}  // namespace

// ---- Device enumeration -----------------------------------------------------
std::vector<AudioDevice> AudioClient::list_capture_devices() {
    impl_->ensure_connected();
    pb::ListAudioDevicesResponse resp;
    grpc::ClientContext ctx;
    detail::check_grpc(impl_->stub->ListAudioCaptureDevices(&ctx, pb::Empty{}, &resp),
                       "ListAudioCaptureDevices");
    std::vector<AudioDevice> out;
    out.reserve(resp.devices_size());
    for (const auto& d : resp.devices()) {
        out.push_back(AudioDevice{d.name(), d.description()});
    }
    return out;
}

std::vector<AudioDevice> AudioClient::list_playback_devices() {
    impl_->ensure_connected();
    pb::ListAudioDevicesResponse resp;
    grpc::ClientContext ctx;
    detail::check_grpc(impl_->stub->ListAudioPlaybackDevices(&ctx, pb::Empty{}, &resp),
                       "ListAudioPlaybackDevices");
    std::vector<AudioDevice> out;
    out.reserve(resp.devices_size());
    for (const auto& d : resp.devices()) {
        out.push_back(AudioDevice{d.name(), d.description()});
    }
    return out;
}

// ---- Capture ----------------------------------------------------------------
void AudioClient::start_capture(const std::string& device, std::uint32_t sample_rate,
                                std::uint32_t channels, const std::string& codec,
                                std::uint32_t bitrate) {
    impl_->ensure_connected();
    pb::AudioConfigRequest req;
    fill_config(req, device, sample_rate, channels, codec, bitrate);
    pb::Status resp;
    grpc::ClientContext ctx;
    detail::check_grpc(impl_->stub->StartAudioCapture(&ctx, req, &resp), "StartAudioCapture");
    detail::require_success(resp.success(), resp.message(), "StartAudioCapture");
}

void AudioClient::stop_capture() {
    impl_->ensure_connected();
    pb::Status resp;
    grpc::ClientContext ctx;
    detail::check_grpc(impl_->stub->StopAudioCapture(&ctx, pb::Empty{}, &resp), "StopAudioCapture");
    detail::require_success(resp.success(), resp.message(), "StopAudioCapture");
}

// ---- Playback ---------------------------------------------------------------
void AudioClient::start_playback(const std::string& device, std::uint32_t sample_rate,
                                 std::uint32_t channels) {
    impl_->ensure_connected();
    pb::AudioConfigRequest req;
    fill_config(req, device, sample_rate, channels, "", 0);
    pb::Status resp;
    grpc::ClientContext ctx;
    detail::check_grpc(impl_->stub->StartAudioPlayback(&ctx, req, &resp), "StartAudioPlayback");
    detail::require_success(resp.success(), resp.message(), "StartAudioPlayback");
}

void AudioClient::stop_playback() {
    impl_->ensure_connected();
    pb::Status resp;
    grpc::ClientContext ctx;
    detail::check_grpc(impl_->stub->StopAudioPlayback(&ctx, pb::Empty{}, &resp), "StopAudioPlayback");
    detail::require_success(resp.success(), resp.message(), "StopAudioPlayback");
}

// ---- Status & config --------------------------------------------------------
AudioStatus AudioClient::get_status() {
    impl_->ensure_connected();
    pb::AudioStatusResponse resp;
    grpc::ClientContext ctx;
    detail::check_grpc(impl_->stub->GetAudioStatus(&ctx, pb::Empty{}, &resp), "GetAudioStatus");
    return AudioStatus{
        resp.capturing(), resp.playing(),    resp.device(),
        resp.sample_rate(), resp.channels(), resp.codec(),
        resp.volume(),      resp.mute(),
    };
}

void AudioClient::set_config(const std::string& device, std::uint32_t sample_rate,
                             std::uint32_t channels, const std::string& codec,
                             std::uint32_t bitrate, std::optional<float> volume,
                             std::optional<bool> mute) {
    impl_->ensure_connected();
    pb::AudioConfigRequest req;
    fill_config(req, device, sample_rate, channels, codec, bitrate);
    if (volume) req.set_volume(*volume);
    if (mute) req.set_mute(*mute);
    pb::Status resp;
    grpc::ClientContext ctx;
    detail::check_grpc(impl_->stub->SetAudioConfig(&ctx, req, &resp), "SetAudioConfig");
    detail::require_success(resp.success(), resp.message(), "SetAudioConfig");
}

// ---- Two-way talk (client-streamed PCM) -------------------------------------
void AudioClient::stream_pcm(const std::function<std::optional<std::string>()>& next,
                             std::uint32_t sample_rate, std::uint32_t channels,
                             const std::string& fmt) {
    impl_->ensure_connected();
    pb::Status resp;
    grpc::ClientContext ctx;
    std::unique_ptr<grpc::ClientWriter<pb::AudioPcmChunk>> writer(
        impl_->stub->StreamAudioPcm(&ctx, &resp));

    pb::AudioPcmChunk chunk;
    chunk.set_sample_rate(sample_rate);
    chunk.set_channels(channels);
    chunk.set_format(fmt);
    while (true) {
        auto data = next();
        if (!data) break;  // iterator exhausted
        chunk.set_data(*data);
        if (!writer->Write(chunk)) {
            writer->Finish();
            throw std::runtime_error("AudioClient::stream_pcm(): write failed mid-stream");
        }
    }
    writer->WritesDone();
    detail::check_grpc(writer->Finish(), "StreamAudioPcm");
    detail::require_success(resp.success(), resp.message(), "StreamAudioPcm");
}

void AudioClient::stream_pcm_file(const std::string& path, int chunk_size,
                                  std::uint32_t sample_rate, std::uint32_t channels,
                                  const std::string& fmt) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        throw std::runtime_error("AudioClient::stream_pcm_file(): cannot open " + path);
    }
    if (chunk_size <= 0) chunk_size = 4096;
    std::string buf(static_cast<std::size_t>(chunk_size), '\0');
    stream_pcm(
        [&]() -> std::optional<std::string> {
            f.read(buf.data(), static_cast<std::streamsize>(buf.size()));
            auto n = f.gcount();
            if (n <= 0) return std::nullopt;
            return std::string(buf.data(), static_cast<std::size_t>(n));
        },
        sample_rate, channels, fmt);
}

}  // namespace neoruntime_ipc_sdk
