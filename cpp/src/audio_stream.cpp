// audio_stream.cpp — AudioStreamClient + AudioFrame. See audio_stream.hpp.
//
// Port of audio_stream.py. Raw UDS reader using the shared detail raw-socket
// helpers and the AudioEncHeader packed struct (30-byte audio frame header).
#include "hailo_ipc_sdk/audio_stream.hpp"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>

#include <unistd.h>

#include "detail/fd_protocol.hpp"
#include "detail/raw_socket.hpp"

namespace hailo_ipc_sdk {

namespace {

constexpr int kAudioHeaderSize = 30;
constexpr int kAudioMaxPayload = 10 * 1024 * 1024;  // 10 MiB sanity cap
constexpr const char* kDefaultAudioSock = "/run/aipc/encoded/audio_capture.sock";

}  // namespace

// ---- AudioFrame -------------------------------------------------------------
std::string AudioFrame::codec_name() const {
    switch (codec) {
        case 0: return "pcm";
        case 1: return "aac";
        case 2: return "g711a";
        case 3: return "g711u";
        default: return "unknown(" + std::to_string(codec) + ')';
    }
}

double AudioFrame::duration_ms() const {
    if (codec == 0 && sample_rate > 0 && channels > 0 && bits_per_sample > 0) {
        std::uint32_t bytes_per_sample = bits_per_sample / 8;
        if (bytes_per_sample > 0) {
            std::uint64_t total_samples = data.size() /
                                          (static_cast<std::uint64_t>(bytes_per_sample) * channels);
            return static_cast<double>(total_samples) / sample_rate * 1000.0;
        }
    }
    return 0.0;
}

// ---- AudioStreamClient ------------------------------------------------------
struct AudioStreamClient::Impl {
    std::string socket_path;
    int fd = -1;
    std::mutex mtx;
    std::atomic<bool> closed{false};
    std::thread cb_thread;

    explicit Impl(std::string path) : socket_path(std::move(path)) {}

    ~Impl() {
        closed.store(true);
        if (cb_thread.joinable()) cb_thread.join();
        if (fd >= 0) ::close(fd);
    }

    void connect_locked() {
        fd = detail::connect_unix(socket_path);
        detail::set_recv_timeout(fd, 5000);
    }

    // Read one frame from `fd`, or std::nullopt on EOF/timeout/error.
    std::optional<AudioFrame> recv_frame() {
        detail::AudioEncHeader hdr{};
        if (!detail::recv_exact(fd, &hdr, kAudioHeaderSize)) return std::nullopt;

        std::int64_t payload_size =
            static_cast<std::int64_t>(hdr.total_size) - kAudioHeaderSize;
        if (payload_size < 0 || payload_size > kAudioMaxPayload) return std::nullopt;

        AudioFrame f;
        f.codec = hdr.codec;
        f.flags = hdr.flags;
        f.pts_ns = hdr.pts_ns;
        const detail::AudioFormatFields fmt = detail::decode_audio_format(hdr, payload_size);
        f.sample_rate = fmt.sample_rate;
        f.channels = fmt.channels;
        f.bits_per_sample = fmt.bits_per_sample;
        f.dts_ns = fmt.dts_ns;
        if (payload_size > 0) {
            f.data.resize(static_cast<std::size_t>(payload_size));
            if (!detail::recv_exact(fd, f.data.data(), f.data.size())) return std::nullopt;
        }
        return f;
    }
};

AudioStreamClient::AudioStreamClient(std::string socket_path)
    : impl_(std::make_unique<Impl>(std::move(socket_path))) {
    if (impl_->socket_path.empty()) {
        const char* env = std::getenv("AUDIO_CAPTURE_SOCK_PATH");
        impl_->socket_path = (env && *env) ? std::string(env) : std::string(kDefaultAudioSock);
    }
}

AudioStreamClient::~AudioStreamClient() = default;
AudioStreamClient::AudioStreamClient(AudioStreamClient&&) noexcept = default;
AudioStreamClient& AudioStreamClient::operator=(AudioStreamClient&&) noexcept = default;

std::optional<AudioFrame> AudioStreamClient::get_frame(int timeout_ms) {
    int fd = -1;
    {
        std::lock_guard<std::mutex> lk(impl_->mtx);
        if (impl_->fd < 0) impl_->connect_locked();
        fd = impl_->fd;
        detail::set_recv_timeout(fd, timeout_ms);
    }
    return impl_->recv_frame();
}

std::optional<AudioFrame> AudioStreamClient::next_frame(bool reconnect) {
    while (!impl_->closed.load()) {
        {
            std::lock_guard<std::mutex> lk(impl_->mtx);
            if (impl_->fd < 0) impl_->connect_locked();
        }
        auto frame = impl_->recv_frame();
        if (frame) return frame;
        if (!reconnect || impl_->closed.load()) return std::nullopt;

        // Reconnect: close the stale fd, back off, retry.
        {
            std::lock_guard<std::mutex> lk(impl_->mtx);
            if (impl_->fd >= 0) {
                ::close(impl_->fd);
                impl_->fd = -1;
            }
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
        try {
            std::lock_guard<std::mutex> lk(impl_->mtx);
            impl_->connect_locked();
        } catch (...) {
            std::this_thread::sleep_for(std::chrono::milliseconds(2000));
        }
    }
    return std::nullopt;
}

std::thread AudioStreamClient::on_frame(std::function<void(const AudioFrame&)> callback) {
    impl_->closed.store(false);
    return std::thread([this, cb = std::move(callback)]() {
        while (!impl_->closed.load()) {
            auto frame = next_frame(true);
            if (!frame) {
                if (impl_->closed.load()) break;
                continue;
            }
            try {
                cb(*frame);
            } catch (...) {
                // Swallow callback errors (matches audio_stream.py's except-Exception).
            }
        }
    });
}

void AudioStreamClient::close() {
    impl_->closed.store(true);
    if (impl_->cb_thread.joinable()) impl_->cb_thread.join();
    std::lock_guard<std::mutex> lk(impl_->mtx);
    if (impl_->fd >= 0) {
        ::close(impl_->fd);
        impl_->fd = -1;
    }
}

}  // namespace hailo_ipc_sdk
