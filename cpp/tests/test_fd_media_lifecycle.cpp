// test_fd_media_lifecycle.cpp — FdMediaClient lifecycle against an in-process
// fake FdPublisher (port of the daemon-timing half of python
// tests/test_media_ext.py keep_fd tests).
//
// The fake publisher is a UDS listener thread speaking the real wire protocol
// (detail/fd_protocol.hpp): SUBSCRIBE 76B → OK 12B handshake, 80B
// FdPubFrameMsg frames with one memfd "dma-buf" per frame passed as
// SCM_RIGHTS, and client→server RELEASE 16B / UNSUBSCRIBE 8B messages
// recorded under a mutex. Pinned behaviors:
//   * copy path: planes materialized + immediate RELEASE (matching frame_id)
//   * keep_fd: dma-buf retained, RELEASE deferred to frame.release() /
//     last-copy destruction, tracked by retained_frames()
//   * subscribe decimation: every Nth frame yielded, decimated keep-fd frames
//     released back immediately
//   * get_frame timeout keeps the cached socket; close() releases retained
//     frames first, then sends UNSUBSCRIBE
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <errno.h>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#include <fcntl.h>
#include <poll.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

#include "neoruntime_ipc_sdk/fd_client.hpp"
#include "neoruntime_ipc_sdk/types.hpp"

#include "detail/fd_protocol.hpp"

using namespace neoruntime_ipc_sdk;
using detail::FdPubFrameMsg;
using detail::FdPubMsgHeader;
using detail::FdPubReleaseMsg;
using detail::FdPubResponseMsg;
using detail::FdPubSubscribeMsg;

namespace {

// ---- helpers -----------------------------------------------------------------

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

// memfd standing in for a dma-buf: written once, passed with SCM_RIGHTS.
int make_memfd(const std::vector<std::uint8_t>& bytes) {
    int fd = ::memfd_create("fdpub-plane", 0);
    if (fd < 0) return -1;
    std::size_t off = 0;
    while (off < bytes.size()) {
        ssize_t n = ::write(fd, bytes.data() + off, bytes.size() - off);
        if (n <= 0) {
            ::close(fd);
            return -1;
        }
        off += static_cast<std::size_t>(n);
    }
    return fd;
}

std::string unique_sock_path(const char* prefix) {
    static std::atomic<unsigned> counter{0};
    return "/tmp/" + std::string(prefix) + "_" + std::to_string(::getpid()) +
           "_" + std::to_string(counter.fetch_add(1)) + ".sock";
}

// ---- the fake FdPublisher ------------------------------------------------------

struct PendingFrame {
    std::uint64_t frame_id = 0;
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    std::uint32_t format_code = 0;  // PixelFormat code
    std::uint32_t flags = 0;
    std::vector<std::uint8_t> plane;  // single plane
};

class FakeFdPublisher {
public:
    FakeFdPublisher() : path_(unique_sock_path("fdpub_test")) {}
    ~FakeFdPublisher() { shutdown(); }
    FakeFdPublisher(const FakeFdPublisher&) = delete;
    FakeFdPublisher& operator=(const FakeFdPublisher&) = delete;

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

    const std::string& path() const { return path_; }

    void push_frame(std::uint64_t frame_id, std::uint32_t w, std::uint32_t h,
                    std::uint32_t format_code, std::vector<std::uint8_t> plane,
                    std::uint32_t flags = 0) {
        PendingFrame f;
        f.frame_id = frame_id;
        f.width = w;
        f.height = h;
        f.format_code = format_code;
        f.flags = flags;
        f.plane = std::move(plane);
        std::lock_guard<std::mutex> lk(mu_);
        pending_.push_back(std::move(f));
    }

    struct Record {
        std::string kind;         // "subscribe" / "release" / "unsubscribe"
        std::uint64_t frame_id;   // release only
    };

    std::vector<Record> records() const {
        std::lock_guard<std::mutex> lk(mu_);
        return records_;
    }

    // Poll until a record of `kind` (optionally with frame_id) shows up.
    bool wait_for(const std::string& kind, int timeout_ms,
                  std::optional<std::uint64_t> frame_id = std::nullopt) const {
        const auto deadline = std::chrono::steady_clock::now() +
                              std::chrono::milliseconds(timeout_ms);
        while (std::chrono::steady_clock::now() < deadline) {
            for (const auto& r : records()) {
                if (r.kind == kind &&
                    (!frame_id.has_value() || r.frame_id == *frame_id)) {
                    return true;
                }
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
        }
        return false;
    }

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
    void record(Record r) {
        std::lock_guard<std::mutex> lk(mu_);
        records_.push_back(std::move(r));
    }

    std::optional<PendingFrame> pop_frame() {
        std::lock_guard<std::mutex> lk(mu_);
        if (pending_.empty()) return std::nullopt;
        PendingFrame f = std::move(pending_.front());
        pending_.erase(pending_.begin());
        return f;
    }

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

    // Handshake then pump: drain client control messages, emit queued frames.
    // Returns when the client goes away or shutdown() runs.
    void serve(int conn) {
        FdPubSubscribeMsg sub{};
        if (!read_exact(conn, &sub, sizeof(sub))) return;
        if (sub.header.type != detail::FD_PUB_MSG_SUBSCRIBE) return;
        record({"subscribe", 0});

        FdPubResponseMsg ok{};
        ok.header.type = detail::FD_PUB_MSG_OK;
        ok.header.size = sizeof(ok);
        ok.code = 0;
        if (!write_all(conn, &ok, sizeof(ok))) return;

        std::vector<std::uint8_t> inbox;
        while (running_) {
            if (!drain_client(conn, inbox)) return;
            auto f = pop_frame();
            if (f.has_value() && !send_frame(conn, *f)) return;
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
        }
    }

    bool read_exact(int fd, void* buf, std::size_t n) {
        auto* p = static_cast<char*>(buf);
        std::size_t got = 0;
        while (got < n) {
            pollfd pp{fd, POLLIN, 0};
            if (::poll(&pp, 1, 2000) <= 0) continue;
            ssize_t r = ::recv(fd, p + got, n - got, 0);
            if (r <= 0) {
                if (r < 0 && errno == EINTR) continue;
                return false;
            }
            got += static_cast<std::size_t>(r);
        }
        return true;
    }

    // Non-blocking drain of RELEASE / UNSUBSCRIBE from the client, with a
    // reassembly buffer (message framing by header.size). false = peer closed.
    bool drain_client(int conn, std::vector<std::uint8_t>& inbox) {
        std::uint8_t chunk[256];
        while (true) {
            ssize_t n = ::recv(conn, chunk, sizeof(chunk), MSG_DONTWAIT);
            if (n == 0) return false;
            if (n < 0) {
                if (errno == EINTR) continue;
                return errno == EAGAIN || errno == EWOULDBLOCK;
            }
            inbox.insert(inbox.end(), chunk, chunk + n);
            while (inbox.size() >= sizeof(FdPubMsgHeader)) {
                FdPubMsgHeader hdr{};
                std::memcpy(&hdr, inbox.data(), sizeof(hdr));
                if (hdr.size < sizeof(FdPubMsgHeader) ||
                    hdr.size > 4096) {  // corrupt stream; drop the connection
                    return false;
                }
                if (inbox.size() < hdr.size) break;
                if (hdr.type == detail::FD_PUB_MSG_RELEASE &&
                    hdr.size == sizeof(FdPubReleaseMsg)) {
                    FdPubReleaseMsg rel{};
                    std::memcpy(&rel, inbox.data(), sizeof(rel));
                    record({"release", rel.frame_id});
                } else if (hdr.type == detail::FD_PUB_MSG_UNSUBSCRIBE) {
                    record({"unsubscribe", 0});
                }
                inbox.erase(inbox.begin(), inbox.begin() + hdr.size);
            }
        }
    }

    bool send_frame(int conn, const PendingFrame& f) {
        FdPubFrameMsg msg{};
        msg.header.type = detail::FD_PUB_MSG_FRAME;
        msg.header.size = sizeof(msg);
        msg.frame_id = f.frame_id;
        msg.timestamp_ns = f.frame_id * 1'000'000ull;
        msg.sequence = f.frame_id;
        msg.width = f.width;
        msg.height = f.height;
        msg.format = f.format_code;
        msg.num_planes = 1;
        msg.strides[0] = f.plane.empty() ? 0u : f.width * 3u;  // RGB rows
        msg.sizes[0] = static_cast<std::uint32_t>(f.plane.size());
        msg.num_fds = 1;
        msg.flags = f.flags;

        int plane_fd = make_memfd(f.plane);
        if (plane_fd < 0) return false;

        iovec iov;
        iov.iov_base = &msg;
        iov.iov_len = sizeof(msg);
        std::vector<char> cbuf(CMSG_SPACE(sizeof(int)));
        msghdr m{};
        m.msg_iov = &iov;
        m.msg_iovlen = 1;
        m.msg_control = cbuf.data();
        m.msg_controllen = cbuf.size();
        cmsghdr* cmsg = CMSG_FIRSTHDR(&m);
        cmsg->cmsg_level = SOL_SOCKET;
        cmsg->cmsg_type = SCM_RIGHTS;
        cmsg->cmsg_len = CMSG_LEN(sizeof(int));
        *reinterpret_cast<int*>(CMSG_DATA(cmsg)) = plane_fd;

        bool ok = ::sendmsg(conn, &m, MSG_NOSIGNAL) ==
                  static_cast<ssize_t>(sizeof(msg));
        ::close(plane_fd);  // the kernel holds its own reference in flight
        return ok;
    }

    std::string path_;
    std::thread thread_;
    std::atomic<bool> running_{false};
    int listen_fd_ = -1;
    int conn_fd_ = -1;  // current connection (for shutdown nudges)
    mutable std::mutex mu_;
    std::vector<Record> records_;
    std::vector<PendingFrame> pending_;
};

// ---- shared fixture -------------------------------------------------------------

class FdMediaLifecycle : public ::testing::Test {
protected:
    void SetUp() override {
        ASSERT_TRUE(pub_.start());
        client_ = std::make_unique<FdMediaClient>(pub_.path());
    }
    void TearDown() override {
        client_.reset();  // close() runs first; fake torn down after
    }

    // RGB 4x2 (24 B) test plane with a recognizable ramp.
    static std::vector<std::uint8_t> rgb_plane() {
        std::vector<std::uint8_t> p(4 * 2 * 3);
        for (std::size_t i = 0; i < p.size(); ++i) {
            p[i] = static_cast<std::uint8_t>(7 + i);
        }
        return p;
    }

    FakeFdPublisher pub_;
    std::unique_ptr<FdMediaClient> client_;
};

}  // namespace

// ---- copy path --------------------------------------------------------------------

TEST_F(FdMediaLifecycle, CopyPathMaterializesAndReleasesImmediately) {
    pub_.push_frame(101, 4, 2, static_cast<std::uint32_t>(PixelFormat::RGB),
                    rgb_plane(), detail::FD_PUB_FRAME_FLAG_OVERLAY_BAKED);

    auto frame = client_->get_frame("main", 3000);
    ASSERT_TRUE(frame.has_value());
    EXPECT_EQ(frame->format, "RGB");
    EXPECT_EQ(frame->width, 4);
    EXPECT_EQ(frame->height, 2);
    EXPECT_EQ(frame->sequence, 101u);
    EXPECT_EQ(frame->flags, detail::FD_PUB_FRAME_FLAG_OVERLAY_BAKED);
    EXPECT_TRUE(frame->baked_overlay());
    EXPECT_FALSE(frame->baked_dpm());
    EXPECT_EQ(frame->handle, nullptr);          // copy path keeps no dma-bufs
    ASSERT_FALSE(frame->image.empty());
    EXPECT_EQ(frame->image.rows, 2);
    EXPECT_EQ(frame->image.cols, 4);
    EXPECT_EQ(frame->image.at<cv::Vec3b>(0, 0)[0], 7);      // plane[0]
    EXPECT_EQ(frame->image.at<cv::Vec3b>(1, 3)[2], 7 + 23); // plane.back()

    EXPECT_EQ(client_->retained_frames(), 0);
    EXPECT_TRUE(pub_.wait_for("release", 2000, 101))
        << "copy path must RELEASE the frame buffer back to the daemon";
}

TEST_F(FdMediaLifecycle, GetFrameTimeoutKeepsSocket) {
    auto nothing = client_->get_frame("main", 150);
    EXPECT_FALSE(nothing.has_value());

    // Same cached subscription delivers the next frame (no reconnect needed).
    pub_.push_frame(102, 4, 2, static_cast<std::uint32_t>(PixelFormat::RGB),
                    rgb_plane());
    auto frame = client_->get_frame("main", 3000);
    ASSERT_TRUE(frame.has_value());
    EXPECT_EQ(frame->sequence, 102u);
    EXPECT_TRUE(pub_.wait_for("release", 2000, 102));
}

// ---- keep_fd path -------------------------------------------------------------------

TEST_F(FdMediaLifecycle, KeepFdDefersReleaseUntilExplicitRelease) {
    pub_.push_frame(202, 4, 2, static_cast<std::uint32_t>(PixelFormat::RGB),
                    rgb_plane());

    auto frame = client_->get_frame("main", 3000, /*keep_fd=*/true);
    ASSERT_TRUE(frame.has_value());
    ASSERT_NE(frame->handle, nullptr);
    EXPECT_EQ(frame->handle->frame_id, 202u);
    EXPECT_TRUE(frame->image.empty());  // lazy: nothing materialized yet
    EXPECT_EQ(client_->retained_frames(), 1);

    // Lazy materialization still works on the retained dma-buf.
    cv::Mat arr = frame->to_array();
    ASSERT_FALSE(arr.empty());
    EXPECT_EQ(arr.rows, 2);
    EXPECT_EQ(arr.at<cv::Vec3b>(0, 0)[0], 7);

    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    for (const auto& r : pub_.records()) {
        EXPECT_NE(r.kind, "release")
            << "no RELEASE may be sent while the frame is retained";
    }
    frame->release();
    EXPECT_TRUE(pub_.wait_for("release", 2000, 202));
    EXPECT_EQ(client_->retained_frames(), 0);
}

TEST_F(FdMediaLifecycle, KeepFdLastCopyReleasesOnDestruction) {
    pub_.push_frame(203, 4, 2, static_cast<std::uint32_t>(PixelFormat::RGB),
                    rgb_plane());

    auto f1 = client_->get_frame("main", 3000, /*keep_fd=*/true);
    ASSERT_TRUE(f1.has_value());
    Frame f2 = *f1;  // shares the FrameHandle
    f1.reset();
    EXPECT_EQ(client_->retained_frames(), 1);  // f2 still holds a copy
    f2.release();                              // explicit release on the copy
    EXPECT_TRUE(pub_.wait_for("release", 2000, 203));

    // And the pure-destruction leg: the last copy going out of scope fires it.
    pub_.push_frame(204, 4, 2, static_cast<std::uint32_t>(PixelFormat::RGB),
                    rgb_plane());
    {
        auto f3 = client_->get_frame("main", 3000, /*keep_fd=*/true);
        ASSERT_TRUE(f3.has_value());
        EXPECT_EQ(f3->handle->frame_id, 204u);
    }  // last Frame copy destroyed -> FrameHandle dtor -> deferred RELEASE
    EXPECT_TRUE(pub_.wait_for("release", 2000, 204));
    EXPECT_EQ(client_->retained_frames(), 0);
}

// ---- subscribe decimation -------------------------------------------------------------

TEST_F(FdMediaLifecycle, SubscribeDecimationYieldsEveryNthAndReleasesSkipped) {
    auto stream = client_->subscribe("main", /*skip_frames=*/2, /*keep_fd=*/true);

    for (std::uint64_t id = 301; id <= 304; ++id) {
        pub_.push_frame(id, 4, 2, static_cast<std::uint32_t>(PixelFormat::RGB),
                        rgb_plane());
    }

    auto f1 = stream.next();
    ASSERT_TRUE(f1.has_value());
    EXPECT_EQ(f1->sequence, 301u);  // seen=1: (1-1)%2==0 -> yielded
    auto f2 = stream.next();
    ASSERT_TRUE(f2.has_value());
    EXPECT_EQ(f2->sequence, 303u);  // 302 decimated away, 303 yielded

    EXPECT_EQ(client_->retained_frames(), 2);  // yielded frames kept (keep_fd)
    EXPECT_TRUE(pub_.wait_for("release", 2000, 302))
        << "decimated keep-fd frame must be released immediately";
    for (const auto& r : pub_.records()) {
        if (r.kind == "release") {
            EXPECT_EQ(r.frame_id, 302u) << "unexpected RELEASE of a yielded frame";
        }
    }
}

// ---- close() ---------------------------------------------------------------------------

TEST_F(FdMediaLifecycle, CloseReleasesRetainedThenUnsubscribes) {
    pub_.push_frame(401, 4, 2, static_cast<std::uint32_t>(PixelFormat::RGB),
                    rgb_plane());
    auto frame = client_->get_frame("main", 3000, /*keep_fd=*/true);
    ASSERT_TRUE(frame.has_value());
    EXPECT_EQ(client_->retained_frames(), 1);

    client_->close();
    EXPECT_EQ(client_->retained_frames(), 0);
    EXPECT_TRUE(pub_.wait_for("release", 2000, 401));
    EXPECT_TRUE(pub_.wait_for("unsubscribe", 2000));

    // Order matters: the daemon recycles buffers before the subscription dies.
    const auto recs = pub_.records();
    std::size_t rel = recs.size(), unsub = recs.size();
    for (std::size_t i = 0; i < recs.size(); ++i) {
        if (recs[i].kind == "release" && recs[i].frame_id == 401) rel = i;
        if (recs[i].kind == "unsubscribe") unsub = i;
    }
    ASSERT_LT(rel, recs.size());
    ASSERT_LT(unsub, recs.size());
    EXPECT_LT(rel, unsub) << "retained frames must be RELEASEd before UNSUBSCRIBE";
}
