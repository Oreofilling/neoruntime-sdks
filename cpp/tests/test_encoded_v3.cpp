// test_encoded_v3.cpp — EncodedStreamClient V3 packet-seq semantics against an
// in-process fake EncodedPublisher (port of python tests/test_encoded_v3.py).
//
// The fake is a UDS listener thread speaking the real wire framing
// (detail/fd_protocol.hpp): a 30-byte little-endian EncHeader, plus 8 more
// bytes of packet seq when flags bit1 (V3) is set, plus the payload. Pinned
// behaviors:
//   * pre-V3 frames decode with seq == nullopt; keyframe flag / codec_name /
//     timestamps / payload round-trip
//   * V3 seqs feed the client-shared SeqTracker: no gap for 1,2,3; one gap
//     event / two missing for 1,2,5; publisher restart (seq going backwards)
//     is NOT a loss
//   * subscribe() streams advance the owning client's seq_stats()
//   * ctor path resolution {socket_dir}/{stream_id}.sock; an unreachable
//     socket throws on connect/get_frame (Python ConnectionError parity)
//   * a bogus total_size behaves like timeout/EOF (nullopt, reconnect next)
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#include <errno.h>
#include <poll.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <unistd.h>

#include "neoruntime_ipc_sdk/encoded.hpp"

#include "detail/fd_protocol.hpp"

using namespace neoruntime_ipc_sdk;
using detail::EncHeader;

namespace {

bool write_all(int fd, const void* buf, std::size_t n) {
    auto* p = static_cast<const char*>(buf);
    std::size_t off = 0;
    while (off < n) {
        ssize_t r = ::send(fd, p + off, n - off, MSG_NOSIGNAL);
        if (r <= 0) {
            if (r < 0 && errno == EINTR) continue;
            return false;
        }
        off += static_cast<std::size_t>(r);
    }
    return true;
}

std::string unique_sock_path(const char* prefix) {
    static std::atomic<unsigned> counter{0};
    return "/tmp/" + std::string(prefix) + "_" + std::to_string(::getpid()) +
           "_" + std::to_string(counter.fetch_add(1)) + ".sock";
}

std::string unique_sock_dir() {
    std::string tmpl = "/tmp/encv3_dir_XXXXXX";
    std::vector<char> buf(tmpl.begin(), tmpl.end());
    buf.push_back('\0');
    char* dir = ::mkdtemp(buf.data());
    return dir != nullptr ? std::string(dir) : std::string();
}

// ---- the fake EncodedPublisher --------------------------------------------------

class FakeEncodedPublisher {
public:
    explicit FakeEncodedPublisher(std::string path) : path_(std::move(path)) {}
    ~FakeEncodedPublisher() { shutdown(); }
    FakeEncodedPublisher(const FakeEncodedPublisher&) = delete;
    FakeEncodedPublisher& operator=(const FakeEncodedPublisher&) = delete;

    bool start() {
        ::unlink(path_.c_str());
        listen_fd_ = ::socket(AF_UNIX, SOCK_STREAM, 0);
        if (listen_fd_ < 0) return false;
        sockaddr_un addr{};
        addr.sun_family = AF_UNIX;
        path_.copy(addr.sun_path, path_.size());
        addr.sun_path[path_.size()] = '\0';
        if (::bind(listen_fd_, reinterpret_cast<sockaddr*>(&addr),
                   sizeof(addr)) < 0 ||
            ::listen(listen_fd_, 4) < 0) {
            return false;
        }
        running_ = true;
        thread_ = std::thread([this] { accept_loop(); });
        return true;
    }

    // Enqueue one frame. seq == nullopt sends the plain 30-byte header
    // (pre-V3); otherwise flags|ENC_FLAG_SEQ_PRESENT and 8 LE seq bytes.
    void push(std::uint8_t codec, std::uint32_t flags, std::uint64_t pts_ns,
              std::uint32_t width, std::uint32_t height, std::uint64_t dts_ns,
              std::optional<std::uint64_t> seq, std::string payload,
              std::optional<std::uint32_t> total_size_override = std::nullopt) {
        const int header_size = sizeof(EncHeader) + (seq.has_value() ? 8 : 0);
        const std::uint32_t total =
            total_size_override.has_value()
                ? *total_size_override
                : static_cast<std::uint32_t>(header_size + payload.size());

        EncHeader hdr{};
        hdr.total_size = total;
        hdr.codec = codec;
        hdr.flags = static_cast<std::uint8_t>(
            seq.has_value() ? (flags | detail::ENC_FLAG_SEQ_PRESENT) : flags);
        hdr.pts_ns = pts_ns;
        hdr.width = width;
        hdr.height = height;
        hdr.dts_ns = dts_ns;

        std::string wire(reinterpret_cast<const char*>(&hdr), sizeof(hdr));
        if (seq.has_value()) {
            std::uint64_t v = *seq;  // little-endian wire order
            for (int i = 0; i < 8; ++i) {
                wire.push_back(static_cast<char>(v & 0xff));
                v >>= 8;
            }
        }
        wire += payload;

        std::lock_guard<std::mutex> lk(mu_);
        pending_.push_back(std::move(wire));
    }

    const std::string& path() const { return path_; }

    void shutdown() {
        if (!running_.exchange(false)) {
            if (thread_.joinable()) thread_.join();
            return;
        }
        if (listen_fd_ >= 0) {
            ::shutdown(listen_fd_, SHUT_RDWR);
            ::close(listen_fd_);
            listen_fd_ = -1;
        }
        if (conn_fd_ >= 0) {
            ::shutdown(conn_fd_, SHUT_RDWR);
        }
        if (thread_.joinable()) thread_.join();
        if (conn_fd_ >= 0) ::close(conn_fd_);
        conn_fd_ = -1;
        ::unlink(path_.c_str());
    }

private:
    void accept_loop() {
        while (running_) {
            pollfd p{listen_fd_, POLLIN, 0};
            if (::poll(&p, 1, 100) <= 0) continue;
            int conn = ::accept(listen_fd_, nullptr, nullptr);
            if (conn < 0) continue;
            conn_fd_ = conn;
            serve(conn);
            ::close(conn);
            conn_fd_ = -1;
        }
    }

    void serve(int conn) {
        while (running_) {
            // Liveness before send: a frame queued while the peer was gone
            // must not be written (and consumed) into a dead connection.
            if (!peer_alive(conn)) return;
            std::string frame;
            {
                std::lock_guard<std::mutex> lk(mu_);
                if (!pending_.empty()) {
                    frame = std::move(pending_.front());
                    pending_.erase(pending_.begin());
                }
            }
            if (!frame.empty() && !write_all(conn, frame.data(), frame.size())) {
                return;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
        }
    }

    // Idle-queue liveness probe: encoded clients never send, so a readable
    // connection means EOF (POLLHUP/recv 0) — the peer went away.
    // Idle-queue liveness probe: encoded clients never send, so a readable
    // connection means EOF — or ECONNRESET when the peer closed with unread
    // bytes still queued (e.g. it abandoned a malformed frame mid-read and
    // the kernel reset the socket instead of a clean FIN). Either way, dead.
    bool peer_alive(int conn) {
        pollfd p{conn, POLLIN, 0};
        if (::poll(&p, 1, 0) <= 0) return true;
        if (p.revents & (POLLERR | POLLHUP | POLLRDHUP)) return false;
        char probe = 0;
        ssize_t n = ::recv(conn, &probe, 1, MSG_PEEK | MSG_DONTWAIT);
        if (n == 0) return false;  // clean EOF
        return n < 0 && (errno == EAGAIN || errno == EWOULDBLOCK);
    }

    std::string path_;
    std::thread thread_;
    std::atomic<bool> running_{false};
    int listen_fd_ = -1;
    int conn_fd_ = -1;
    mutable std::mutex mu_;
    std::vector<std::string> pending_;
};

// ---- shared fixture -------------------------------------------------------------

class EncodedV3 : public ::testing::Test {
protected:
    void SetUp() override {
        pub_ = std::make_unique<FakeEncodedPublisher>(unique_sock_path("encv3_test"));
        ASSERT_TRUE(pub_->start());
    }
    void TearDown() override { pub_.reset(); }

    void push_seq(std::uint64_t seq, bool keyframe = true) {
        pub_->push(0, keyframe ? 0x01u : 0x00u, seq * 1'000'000ull, 640, 360,
                   seq * 900'000ull, seq, "nalu-" + std::to_string(seq));
    }

    std::unique_ptr<FakeEncodedPublisher> pub_;
};

}  // namespace

// ---- pre-V3 frames -----------------------------------------------------------------

TEST_F(EncodedV3, PreV3FrameHasNoSeq) {
    pub_->push(0, 0x01u, 111'000'000ull, 1920, 1080, 110'000'000ull,
               std::nullopt, std::string("\x00\x00\x00\x01\x65payload"));
    EncodedStreamClient client(pub_->path());

    auto f = client.get_frame(3000);
    ASSERT_TRUE(f.has_value());
    EXPECT_EQ(f->seq, std::nullopt);
    EXPECT_TRUE(f->is_keyframe());
    EXPECT_EQ(f->codec_name(), "h264");
    EXPECT_EQ(f->pts_ns, 111'000'000ull);
    EXPECT_EQ(f->dts_ns, 110'000'000ull);
    EXPECT_EQ(f->width, 1920);
    EXPECT_EQ(f->height, 1080);
    EXPECT_EQ(f->data, std::string("\x00\x00\x00\x01\x65payload"));

    const auto stats = client.seq_stats();
    EXPECT_EQ(stats.seq_packets, 0u);  // pre-V3 frames never feed the tracker
    EXPECT_FALSE(stats.last_seq.has_value());
}

TEST(EncodedFrameUnits, CodecNamesAndKeyframeFlag) {
    auto make = [](int codec, std::uint32_t flags) {
        EncodedFrame f;
        f.codec = codec;
        f.flags = flags;
        return f;
    };
    EXPECT_EQ(make(1, 0).codec_name(), "h265");
    EXPECT_EQ(make(0, 0).codec_name(), "h264");
    EXPECT_EQ(make(7, 0).codec_name(), "unknown(7)");
    EXPECT_TRUE(make(0, 0x01).is_keyframe());
    EXPECT_FALSE(make(0, 0x02).is_keyframe());
}

// ---- V3 seq observability ------------------------------------------------------------

TEST_F(EncodedV3, ContiguousSeqsCountNoGap) {
    EncodedStreamClient client(pub_->path());
    push_seq(1);
    push_seq(2);
    push_seq(3);
    ASSERT_TRUE(client.get_frame(3000).has_value());
    ASSERT_TRUE(client.get_frame(3000).has_value());
    ASSERT_TRUE(client.get_frame(3000).has_value());

    const auto s = client.seq_stats();
    EXPECT_EQ(s.seq_packets, 3u);
    EXPECT_EQ(s.seq_gap_events, 0u);
    EXPECT_EQ(s.seq_missing, 0u);
    ASSERT_TRUE(s.last_seq.has_value());
    EXPECT_EQ(*s.last_seq, 3u);
}

TEST_F(EncodedV3, HoleCountsOneGapEventAndMissingPackets) {
    EncodedStreamClient client(pub_->path());
    push_seq(1);
    push_seq(2);
    push_seq(5);  // 3 and 4 were dropped somewhere between publisher and client
    ASSERT_TRUE(client.get_frame(3000).has_value());
    ASSERT_TRUE(client.get_frame(3000).has_value());
    ASSERT_TRUE(client.get_frame(3000).has_value());

    const auto s = client.seq_stats();
    EXPECT_EQ(s.seq_packets, 3u);
    EXPECT_EQ(s.seq_gap_events, 1u);
    EXPECT_EQ(s.seq_missing, 2u);
    ASSERT_TRUE(s.last_seq.has_value());
    EXPECT_EQ(*s.last_seq, 5u);
}

TEST_F(EncodedV3, PublisherRestartIsNotALoss) {
    EncodedStreamClient client(pub_->path());
    push_seq(1);
    push_seq(2);
    push_seq(1);  // publisher restarted, seq rebased to 1
    push_seq(2);
    for (int i = 0; i < 4; ++i) {
        ASSERT_TRUE(client.get_frame(3000).has_value());
    }

    const auto s = client.seq_stats();
    EXPECT_EQ(s.seq_packets, 4u);
    EXPECT_EQ(s.seq_gap_events, 0u);
    EXPECT_EQ(s.seq_missing, 0u);
    ASSERT_TRUE(s.last_seq.has_value());
    EXPECT_EQ(*s.last_seq, 2u);
}

TEST_F(EncodedV3, SubscribeStreamAdvancesClientCounters) {
    EncodedStreamClient client(pub_->path());
    push_seq(1);
    push_seq(2);

    auto stream = client.subscribe();
    auto f1 = stream.next();
    ASSERT_TRUE(f1.has_value());
    EXPECT_EQ(f1->seq, std::optional<std::uint64_t>(1));
    auto f2 = stream.next();
    ASSERT_TRUE(f2.has_value());
    EXPECT_EQ(f2->seq, std::optional<std::uint64_t>(2));

    // The stream shares the client's tracker (Python client-instance attrs).
    const auto s = client.seq_stats();
    EXPECT_EQ(s.seq_packets, 2u);
    ASSERT_TRUE(s.last_seq.has_value());
    EXPECT_EQ(*s.last_seq, 2u);
}

TEST_F(EncodedV3, BogusTotalSizeBehavesLikeTimeout) {
    // total_size smaller than the header itself: malformed, dropped like a
    // timeout/EOF (nullopt, connection closed), and the client reconnects and
    // delivers the next (well-formed) frame.
    pub_->push(0, 0x01u, 1, 64, 64, 1, std::optional<std::uint64_t>(9), "x",
               /*total_size_override=*/10u);
    EncodedStreamClient client(pub_->path());
    EXPECT_FALSE(client.get_frame(2000).has_value());

    push_seq(1);
    auto good = client.get_frame(3000);
    ASSERT_TRUE(good.has_value());
    EXPECT_EQ(good->seq, std::optional<std::uint64_t>(1));
}

// ---- ctor path resolution -----------------------------------------------------------

TEST_F(EncodedV3, CtorDerivesPathFromDirAndStreamId) {
    const std::string dir = unique_sock_dir();
    ASSERT_FALSE(dir.empty());
    FakeEncodedPublisher sub_pub(dir + "/sub.sock");
    ASSERT_TRUE(sub_pub.start());

    // socket_path empty -> {socket_dir}/{stream_id}.sock (encoded.py rule).
    EncodedStreamClient client("", "sub", dir);
    EXPECT_NO_THROW(client.connect());
    EXPECT_TRUE(client.connected());

    sub_pub.push(1, 0x01u, 42, 320, 240, 40, std::optional<std::uint64_t>(1),
                 "hevc");
    auto f = client.get_frame(3000);
    ASSERT_TRUE(f.has_value());
    EXPECT_EQ(f->codec_name(), "h265");
    EXPECT_EQ(f->data, std::string("hevc"));

    client.close();
    EXPECT_FALSE(client.connected());
    sub_pub.shutdown();
    ::rmdir(dir.c_str());
}

TEST(EncodedCtor, UnreachableSocketThrowsOnConnect) {
    // The explicit socket_path is used verbatim (no fallback), and connect
    // failures propagate — Python get_frame lets ConnectionError through.
    EncodedStreamClient client("/run/aipc/encoded/definitely-not-here.sock");
    EXPECT_THROW(client.connect(), std::runtime_error);
    EXPECT_THROW(client.get_frame(100), std::runtime_error);
}
