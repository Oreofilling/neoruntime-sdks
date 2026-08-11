// fd_media_client.cpp — FdMediaClient + FdMediaStream.
// Port of media.py FdMediaClient. Raw UDS + SCM_RIGHTS fd-passing: each frame
// arrives as an 80-byte FdPubFrameMsg carrying up to 3 DMA-BUF fds as ancillary
// data. Each fd is mmapped (by its fstat size) and `sizes[i]` bytes copied out,
// then the frame is RELEASEd and all fds closed (mirrors media.py _recv_frame).
#include "hailo_ipc_sdk/media.hpp"

#include <cerrno>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include "hailo_ipc_sdk/config.hpp"

#include "detail/fd_protocol.hpp"
#include "detail/raw_socket.hpp"

namespace hailo_ipc_sdk {

using detail::FdPubFrameMsg;
using detail::FdPubMsgHeader;
using detail::FdPubReleaseMsg;
using detail::FdPubResponseMsg;
using detail::FdPubSubscribeMsg;
using detail::connect_unix;
using detail::recv_exact;
using detail::send_all;
using detail::set_recv_timeout;

namespace {

constexpr int kFdFrameSize = sizeof(FdPubFrameMsg);       // 80
constexpr int kFdRespSize = sizeof(FdPubResponseMsg);     // 12
constexpr int kFdSubSize = sizeof(FdPubSubscribeMsg);     // 76
constexpr int kFdRelSize = sizeof(FdPubReleaseMsg);       // 16
constexpr int kFdMaxFds = detail::FD_PUB_MAX_FDS;         // 3
constexpr int kFdMaxAttempts = 32;
constexpr int kFdReconnectAttempts = 5;
constexpr int kFdRecvTimeoutMs = 5000;

std::string format_code_to_name(std::uint32_t code) {
    switch (static_cast<PixelFormat>(code)) {
        case PixelFormat::NV12: return "NV12";
        case PixelFormat::NV21: return "NV21";
        case PixelFormat::RGB: return "RGB";
        case PixelFormat::BGR: return "BGR";
        case PixelFormat::RGBA: return "RGBA";
        case PixelFormat::BGRA: return "BGRA";
        case PixelFormat::GRAY8: return "GRAY8";
        case PixelFormat::YUYV: return "YUYV";
        default: return "UNKNOWN(" + std::to_string(code) + ")";
    }
}

// Connect to the FdPublisher and SUBSCRIBE to `stream_id`. Returns the fd or
// throws std::runtime_error if the server rejects the subscription.
int connect_stream(const std::string& socket_path, const std::string& stream_id) {
    int fd = connect_unix(socket_path);

    FdPubSubscribeMsg sub{};
    sub.header.type = detail::FD_PUB_MSG_SUBSCRIBE;
    sub.header.size = static_cast<std::uint32_t>(kFdSubSize);
    sub.version = detail::FD_PUB_PROTOCOL_VERSION;
    auto name = stream_id.substr(0, detail::FD_PUB_MAX_STREAM_NAME - 1);
    std::memcpy(sub.stream_name, name.data(), name.size());  // rest stays '\0'

    if (!send_all(fd, &sub, kFdSubSize)) {
        ::close(fd);
        throw std::runtime_error("FdMediaClient: failed to send SUBSCRIBE for '" + stream_id + "'");
    }

    FdPubResponseMsg resp{};
    if (!recv_exact(fd, &resp, kFdRespSize)) {
        ::close(fd);
        throw std::runtime_error("FdMediaClient: no response for stream '" + stream_id + "'");
    }
    if (resp.header.type != detail::FD_PUB_MSG_OK) {
        ::close(fd);
        throw std::runtime_error("FdMediaClient: subscribe rejected for '" + stream_id +
                                 "' (code=" + std::to_string(resp.code) + ")");
    }
    return fd;
}

void release_frame(int fd, std::uint64_t frame_id) {
    FdPubReleaseMsg rel{};
    rel.header.type = detail::FD_PUB_MSG_RELEASE;
    rel.header.size = static_cast<std::uint32_t>(kFdRelSize);
    rel.frame_id = frame_id;
    send_all(fd, &rel, kFdRelSize);  // best-effort; ignore failure
}

void unsubscribe(int fd) {
    FdPubMsgHeader hdr{};
    hdr.type = detail::FD_PUB_MSG_UNSUBSCRIBE;
    hdr.size = static_cast<std::uint32_t>(sizeof(FdPubMsgHeader));
    send_all(fd, &hdr, sizeof(hdr));  // best-effort
}

// Receive data + SCM_RIGHTS fds in one recvmsg. Returns bytes received (0 on
// clean EOF, -1 on error/timeout). `fds_out` is appended the received fds.
ssize_t recvmsg_with_fds(int fd, void* buf, std::size_t len, std::vector<int>& fds_out) {
    iovec iov;
    iov.iov_base = buf;
    iov.iov_len = len;

    msghdr msg{};
    msg.msg_iov = &iov;
    msg.msg_iovlen = 1;

    std::vector<char> cbuf(CMSG_SPACE(kFdMaxFds * sizeof(int)));
    msg.msg_control = cbuf.data();
    msg.msg_controllen = cbuf.size();

    ssize_t n = ::recvmsg(fd, &msg, 0);
    if (n <= 0) return n;  // 0 = EOF, -1 = error/timeout

    for (cmsghdr* cmsg = CMSG_FIRSTHDR(&msg); cmsg != nullptr; cmsg = CMSG_NXTHDR(&msg, cmsg)) {
        if (cmsg->cmsg_level == SOL_SOCKET && cmsg->cmsg_type == SCM_RIGHTS) {
            int nf = static_cast<int>((cmsg->cmsg_len - sizeof(cmsghdr)) / sizeof(int));
            auto fd_data = reinterpret_cast<int*>(CMSG_DATA(cmsg));
            for (int i = 0; i < nf; ++i) fds_out.push_back(fd_data[i]);
        }
    }
    return n;
}

void close_all(std::vector<int>& fds) {
    for (int f : fds) {
        if (f >= 0) ::close(f);
    }
    fds.clear();
}

// Copy all plane data out of the received DMA-BUF fds into `raw`. Returns false
// if no fds or an mmap failed (caller still owns closing fds).
bool extract_planes(const std::vector<int>& fds, std::uint32_t num_planes,
                    const std::uint32_t sizes[3], std::vector<std::uint8_t>& raw) {
    if (fds.empty()) return false;

    std::uint32_t n = num_planes;
    if (n > fds.size()) n = static_cast<std::uint32_t>(fds.size());
    if (n > 3) n = 3;

    for (std::uint32_t i = 0; i < n; ++i) {
        struct stat st;
        if (::fstat(fds[i], &st) != 0 || st.st_size <= 0) return false;

        void* mapped = ::mmap(nullptr, static_cast<std::size_t>(st.st_size),
                              PROT_READ, MAP_SHARED, fds[i], 0);
        if (mapped == MAP_FAILED) return false;

        std::size_t copy_bytes = sizes[i];
        if (copy_bytes > static_cast<std::size_t>(st.st_size)) {
            copy_bytes = static_cast<std::size_t>(st.st_size);
        }
        auto base = static_cast<const std::uint8_t*>(mapped);
        raw.insert(raw.end(), base, base + copy_bytes);
        ::munmap(mapped, static_cast<std::size_t>(st.st_size));
    }
    return true;
}

// Reshape the concatenated plane bytes into a cv::Mat matching `fmt`.
cv::Mat decode_image(const std::vector<std::uint8_t>& raw, int w, int h, const std::string& fmt) {
    if (raw.empty() || w <= 0 || h <= 0) return {};
    const auto* p = raw.data();
    if (fmt == "NV12" || fmt == "NV21") {
        return cv::Mat(h * 3 / 2, w, CV_8UC1,
                       const_cast<std::uint8_t*>(p)).clone();
    }
    if (fmt == "RGB" || fmt == "BGR") {
        return cv::Mat(h, w, CV_8UC3, const_cast<std::uint8_t*>(p)).clone();
    }
    if (fmt == "RGBA" || fmt == "BGRA") {
        return cv::Mat(h, w, CV_8UC4, const_cast<std::uint8_t*>(p)).clone();
    }
    if (fmt == "GRAY8") {
        return cv::Mat(h, w, CV_8UC1, const_cast<std::uint8_t*>(p)).clone();
    }
    if (fmt == "YUYV") {
        return cv::Mat(h, w, CV_8UC2, const_cast<std::uint8_t*>(p)).clone();
    }
    return cv::Mat(h, w, CV_8UC3, const_cast<std::uint8_t*>(p)).clone();
}

// Read one Frame off a connected, subscribed socket. std::nullopt on timeout or
// after the server closes. On a hard connection failure, throws std::runtime_error
// (mirrors media.py raising ConnectionError on 3 consecutive EOFs).
std::optional<Frame> recv_frame(int fd) {
    int skipped = 0;
    int eof_count = 0;
    FdPubFrameMsg msg{};
    std::vector<int> fds;

    for (int attempt = 0; attempt < kFdMaxAttempts; ++attempt) {
        fds.clear();
        ssize_t n = recvmsg_with_fds(fd, &msg, kFdFrameSize, fds);

        if (n == 0) {  // EOF
            close_all(fds);
            if (++eof_count >= 3) {
                throw std::runtime_error("FdMediaClient: socket EOF (server closed)");
            }
            continue;
        }
        if (n < 0) {  // timeout / error
            close_all(fds);
            return std::nullopt;
        }
        if (static_cast<std::size_t>(n) < kFdFrameSize) {  // short read
            close_all(fds);
            ++skipped;
            continue;
        }
        if (msg.header.type != detail::FD_PUB_MSG_FRAME) {
            close_all(fds);
            ++skipped;
            continue;
        }
        // Got a full FRAME message.
        std::vector<std::uint8_t> raw;
        bool ok = extract_planes(fds, msg.num_planes, msg.sizes, raw);
        close_all(fds);  // always close received fds (mapped or not)

        if (!ok) {
            release_frame(fd, msg.frame_id);
            return std::nullopt;
        }

        std::string fmt = format_code_to_name(msg.format);
        cv::Mat image = decode_image(raw, static_cast<int>(msg.width),
                                     static_cast<int>(msg.height), fmt);
        release_frame(fd, msg.frame_id);

        Frame frame;
        frame.sequence = msg.sequence;
        frame.timestamp_ns = msg.timestamp_ns;
        frame.width = static_cast<int>(msg.width);
        frame.height = static_cast<int>(msg.height);
        frame.format = std::move(fmt);
        frame.image = std::move(image);
        return frame;
    }
    // Exhausted attempts without a frame.
    return std::nullopt;
}
}  // namespace

// ============================================================================
// FdMediaStream — owns its own connection, bounded auto-reconnect.
// ============================================================================
struct FdMediaStream::Impl {
    std::string socket_path;
    std::string stream_id;
    bool reconnect_enabled = true;
    int fd = -1;
    bool exhausted = false;

    ~Impl() { close_fd(); }

    void close_fd() {
        if (fd >= 0) {
            unsubscribe(fd);
            ::close(fd);
            fd = -1;
        }
    }

    bool ensure_connected() {
        if (fd >= 0) return true;
        if (!reconnect_enabled) {
            try { fd = connect_stream(socket_path, stream_id); } catch (...) { return false; }
            return fd >= 0;
        }
        for (int i = 0; i < kFdReconnectAttempts; ++i) {
            try {
                fd = connect_stream(socket_path, stream_id);
                if (fd >= 0) return true;
            } catch (...) {
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(200 * (i + 1)));
        }
        return false;
    }
};

FdMediaStream::FdMediaStream() : impl_(std::make_unique<Impl>()) {}
FdMediaStream::~FdMediaStream() = default;
FdMediaStream::FdMediaStream(FdMediaStream&&) noexcept = default;
FdMediaStream& FdMediaStream::operator=(FdMediaStream&&) noexcept = default;

std::optional<Frame> FdMediaStream::next() {
    if (!impl_ || impl_->exhausted) return std::nullopt;
    while (true) {
        if (!impl_->ensure_connected()) {
            impl_->exhausted = true;
            return std::nullopt;
        }
        set_recv_timeout(impl_->fd, kFdRecvTimeoutMs);
        try {
            auto frame = recv_frame(impl_->fd);
            if (frame) return frame;
            // timeout/EOF -> drop and reconnect (if enabled)
        } catch (...) {
            // connection error -> drop and reconnect
        }
        impl_->close_fd();
        if (!impl_->reconnect_enabled) {
            impl_->exhausted = true;
            return std::nullopt;
        }
    }
}

// ============================================================================
// FdMediaClient
// ============================================================================
struct FdMediaClient::Impl {
    std::string socket_path;
    std::unordered_map<std::string, int> streams;

    explicit Impl(std::string path) : socket_path(std::move(path)) {
        if (socket_path.empty()) {
            // media.py: os.getenv("CAMERA_SOCK_PATH", "/run/aipc/camera.sock").
            // This is the raw FdPublisher UDS, NOT the camera-control gRPC endpoint.
            const char* env = std::getenv("CAMERA_SOCK_PATH");
            socket_path = env && *env ? std::string(env) : "/run/aipc/camera.sock";
        }
    }
    ~Impl() { close_all_streams(); }

    void close_all_streams() {
        for (auto& [id, fd] : streams) {
            if (fd >= 0) {
                unsubscribe(fd);
                ::close(fd);
                fd = -1;
            }
        }
        streams.clear();
    }

    // Cached per-stream socket (connects lazily). Throws on subscribe failure.
    int get_sock(const std::string& stream_id) {
        auto it = streams.find(stream_id);
        if (it != streams.end() && it->second >= 0) return it->second;
        int fd = connect_stream(socket_path, stream_id);
        streams[stream_id] = fd;
        return fd;
    }

    // Drop a stale cached socket so the next call reconnects.
    void drop_sock(const std::string& stream_id) {
        auto it = streams.find(stream_id);
        if (it != streams.end()) {
            if (it->second >= 0) {
                ::close(it->second);
            }
            streams.erase(it);
        }
    }
};

FdMediaClient::FdMediaClient(std::string socket_path)
    : impl_(std::make_unique<Impl>(std::move(socket_path))) {}
FdMediaClient::~FdMediaClient() = default;
FdMediaClient::FdMediaClient(FdMediaClient&&) noexcept = default;
FdMediaClient& FdMediaClient::operator=(FdMediaClient&&) noexcept = default;

void FdMediaClient::close() { impl_->close_all_streams(); }

std::optional<Frame> FdMediaClient::get_frame(const std::string& stream_id, int timeout_ms) {
    int fd = -1;
    try {
        fd = impl_->get_sock(stream_id);
    } catch (...) {
        return std::nullopt;
    }
    set_recv_timeout(fd, timeout_ms > 0 ? timeout_ms : 5000);
    try {
        auto frame = recv_frame(fd);
        if (!frame) {
            // Timeout or EOF: drop the stale socket so the next call reconnects.
            impl_->drop_sock(stream_id);
        }
        return frame;
    } catch (...) {
        impl_->drop_sock(stream_id);
        return std::nullopt;
    }
}

FdMediaStream FdMediaClient::subscribe(const std::string& stream_id, bool reconnect) {
    FdMediaStream s;
    s.impl_->socket_path = impl_->socket_path;
    s.impl_->stream_id = stream_id;
    s.impl_->reconnect_enabled = reconnect;
    s.impl_->fd = -1;
    s.impl_->exhausted = false;
    return s;
}

std::thread FdMediaClient::on_frame(const std::string& stream_id,
                                    std::function<void(const Frame&)> cb,
                                    bool reconnect) {
    auto path = impl_->socket_path;
    return std::thread([path, stream_id, reconnect, cb = std::move(cb)]() {
        FdMediaStream stream;
        stream.impl_->socket_path = path;
        stream.impl_->stream_id = stream_id;
        stream.impl_->reconnect_enabled = reconnect;
        stream.impl_->fd = -1;
        stream.impl_->exhausted = false;
        while (auto frame = stream.next()) {
            try { cb(*frame); } catch (...) {}
        }
    });
}

EncodedStreamClient FdMediaClient::get_encoded_stream(const std::string& stream_id,
                                                      std::string socket_dir) {
    if (socket_dir.empty()) socket_dir = Config::get_encoded_socket_dir();
    std::string path = socket_dir;
    if (!path.empty() && path.back() != '/') path.push_back('/');
    path += stream_id + ".sock";
    return EncodedStreamClient(std::move(path));
}

std::vector<std::string> FdMediaClient::list_streams() const {
    return {"main", "sub"};
}

std::string FdMediaClient::get_rtsp_url(const std::string& stream_id,
                                        const std::string& host, int port) const {
    return "rtsp://" + host + ":" + std::to_string(port) + "/" + stream_id;
}

}  // namespace hailo_ipc_sdk
