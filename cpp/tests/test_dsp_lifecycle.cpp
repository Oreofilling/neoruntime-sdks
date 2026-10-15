// test_dsp_lifecycle.cpp — DspClient buffer-plane lifecycle against an
// in-process fake camera.sock daemon (memfd planes over SCM_RIGHTS, the
// same trick as test_fd_media_lifecycle's fake FdPublisher). This is the
// "memfd 模拟池" half of Phase 5's test_dsp_fallback milestone.
//
// Covers what the daemon-independent tests cannot: the DSP_ALLOC /
// DSP_IMPORT / DSP_BUF_RELEASE UDS exchange, DspBufferPool write/read with
// stride padding (padding bytes preserved — checked through the daemon's
// own copy of the plane), pool release and client-close poisoning,
// SharedImportPool slot lifecycle, and the wire error surface (alloc
// refusal -3, fd-count mismatch, wrong response type, import timeout).
// The gRPC job plane is never touched — no rpcs are made, so no daemon is
// needed beyond the fake UDS listener.
#include <gtest/gtest.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "neoruntime_ipc_sdk/dsp.hpp"

#include "detail/dsp_wire.hpp"

namespace wire = neoruntime_ipc_sdk::detail::dsp_wire;
using neoruntime_ipc_sdk::DspClient;
using neoruntime_ipc_sdk::DspError;

namespace {

void put32(std::uint8_t* p, std::uint32_t v) {
    p[0] = static_cast<std::uint8_t>(v);
    p[1] = static_cast<std::uint8_t>(v >> 8);
    p[2] = static_cast<std::uint8_t>(v >> 16);
    p[3] = static_cast<std::uint8_t>(v >> 24);
}
void put64(std::uint8_t* p, std::uint64_t v) {
    for (int i = 0; i < 8; ++i) p[i] = static_cast<std::uint8_t>(v >> (8 * i));
}
std::uint32_t get32(const std::uint8_t* p) {
    return static_cast<std::uint32_t>(p[0]) | (static_cast<std::uint32_t>(p[1]) << 8) |
           (static_cast<std::uint32_t>(p[2]) << 16) | (static_cast<std::uint32_t>(p[3]) << 24);
}
std::uint64_t get64(const std::uint8_t* p) {
    std::uint64_t v = 0;
    for (int i = 7; i >= 0; --i) v = (v << 8) | p[i];
    return v;
}

// One fake camera.sock: answers DSP_ALLOC with memfd planes (sent as
// dma-bufs over SCM_RIGHTS), counts DSP_BUF_RELEASEs, answers or mutes
// DSP_IMPORT. Connections are served sequentially — the client serializes
// its exchanges under one connection anyway. No gtest macros run on this
// thread; failures surface as a dropped connection and the test's own
// expectation then fails.
struct FakeDspDaemon {
    std::string path;
    int listen_fd = -1;
    std::thread accept_thread;
    std::atomic<bool> stop{false};

    // knobs (set before the client call they shape)
    std::atomic<int> alloc_code{0};
    std::atomic<int> alloc_fds_to_send{-1};  // -1 = correct count
    std::atomic<std::uint32_t> stride_pad{0};
    std::atomic<bool> mute_import{false};
    std::atomic<std::uint32_t> wrong_alloc_type{0};  // 0 = correct type

    // observations
    std::mutex mu;
    std::vector<std::uint64_t> released_ids;
    std::vector<int> kept_plane_fds;  // daemon-side originals (padding checks)
    int import_count = 0;

    explicit FakeDspDaemon(std::string sock_path) : path(std::move(sock_path)) {
        listen_fd = ::socket(AF_UNIX, SOCK_STREAM, 0);
        sockaddr_un addr{};
        addr.sun_family = AF_UNIX;
        std::strncpy(addr.sun_path, path.c_str(), sizeof(addr.sun_path) - 1);
        ::unlink(path.c_str());
        if (::bind(listen_fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0)
            std::fprintf(stderr, "fake daemon bind failed: %s\n", std::strerror(errno));
        ::listen(listen_fd, 4);
        accept_thread = std::thread([this] {
            while (!stop) {
                int c = ::accept(listen_fd, nullptr, nullptr);
                if (c < 0) return;  // listener closed by join()
                serve(c);
                ::close(c);
            }
        });
    }

    ~FakeDspDaemon() { join(); }

    void join() {
        if (stop.exchange(true)) return;
        ::shutdown(listen_fd, SHUT_RDWR);
        ::close(listen_fd);
        if (accept_thread.joinable()) accept_thread.join();
        for (int fd : kept_plane_fds) ::close(fd);
        ::unlink(path.c_str());
    }

    static bool recv_bytes(int fd, std::uint8_t* buf, std::size_t len, std::vector<int>* fds) {
        std::size_t got = 0;
        while (got < len) {
            msghdr msg{};
            iovec iov{buf + got, len - got};
            msg.msg_iov = &iov;
            msg.msg_iovlen = 1;
            char cbuf[CMSG_SPACE(64 * sizeof(int))];
            msg.msg_control = cbuf;
            msg.msg_controllen = sizeof(cbuf);
            const ssize_t n = ::recvmsg(fd, &msg, 0);
            if (n <= 0) return false;
            if (fds) {
                for (cmsghdr* c = CMSG_FIRSTHDR(&msg); c; c = CMSG_NXTHDR(&msg, c)) {
                    if (c->cmsg_level == SOL_SOCKET && c->cmsg_type == SCM_RIGHTS) {
                        const int nf = static_cast<int>((c->cmsg_len - CMSG_LEN(0)) / sizeof(int));
                        const int* f = reinterpret_cast<const int*>(CMSG_DATA(c));
                        fds->insert(fds->end(), f, f + nf);
                    }
                }
            }
            got += static_cast<std::size_t>(n);
        }
        return true;
    }

    static bool send_with_fds(int fd, const std::uint8_t* buf, std::size_t len,
                              const std::vector<int>& fds) {
        msghdr msg{};
        iovec iov{const_cast<std::uint8_t*>(buf), len};
        msg.msg_iov = &iov;
        msg.msg_iovlen = 1;
        std::vector<char> cbuf(CMSG_SPACE(fds.size() * sizeof(int)));
        if (!fds.empty()) {
            msg.msg_control = cbuf.data();
            msg.msg_controllen = cbuf.size();
            cmsghdr* c = CMSG_FIRSTHDR(&msg);
            c->cmsg_level = SOL_SOCKET;
            c->cmsg_type = SCM_RIGHTS;
            c->cmsg_len = CMSG_LEN(fds.size() * sizeof(int));
            std::memcpy(CMSG_DATA(c), fds.data(), fds.size() * sizeof(int));
        }
        return ::sendmsg(fd, &msg, 0) >= 0;
    }

    void serve(int fd) {
        for (;;) {
            std::uint8_t hdr[8];
            if (!recv_bytes(fd, hdr, 8, nullptr)) return;
            const std::uint32_t type = get32(hdr);
            const std::uint32_t size = get32(hdr + 4);
            if (size < 8 || size > (1u << 20)) return;
            std::vector<std::uint8_t> body(size - 8);
            std::vector<int> fds;
            if (!recv_bytes(fd, body.data(), body.size(), &fds)) return;
            for (int f : fds) ::close(f);  // imported planes are ours to keep; drop

            if (type == wire::kMsgDspAlloc) {
                if (!send_alloc(fd, get32(body.data()), get32(body.data() + 4),
                                get32(body.data() + 8), get32(body.data() + 12)))
                    return;
            } else if (type == wire::kMsgDspBufRelease) {
                std::lock_guard<std::mutex> g(mu);
                released_ids.push_back(get64(body.data()));
            } else if (type == wire::kMsgDspImport) {
                if (mute_import) continue;  // answer nothing: timeout path
                std::vector<std::uint8_t> resp(wire::kImportRespSize, 0);
                put32(resp.data(), wire::kMsgDspImportResp);
                put32(resp.data() + 4, wire::kImportRespSize);
                put32(resp.data() + 8, 0);  // code ok
                {
                    std::lock_guard<std::mutex> g(mu);
                    put64(resp.data() + 16, 0x9000 + static_cast<std::uint64_t>(++import_count));
                }
                if (!send_with_fds(fd, resp.data(), resp.size(), {})) return;
            }
            // DSP_BUF_RELEASE gets no ack (fire-and-forget, like the daemon)
        }
    }

    bool send_alloc(int fd, std::uint32_t w, std::uint32_t h, std::uint32_t fmt,
                    std::uint32_t count) {
        const int planes = fmt == wire::kHalFormatNv12 ? 2 : 1;
        const int row_bytes = planes == 2
                                  ? static_cast<int>(w)
                                  : static_cast<int>(w * (fmt == wire::kHalFormatRgb24 ? 3 : 1));
        const int rows0 = static_cast<int>(h);
        std::vector<std::uint8_t> resp(wire::kAllocRespSize, 0);
        put32(resp.data(), wrong_alloc_type ? wrong_alloc_type.load() : wire::kMsgDspAllocResp);
        put32(resp.data() + 4, wire::kAllocRespSize);
        put32(resp.data() + 8, static_cast<std::uint32_t>(alloc_code));
        put32(resp.data() + 12, count);
        put32(resp.data() + 16, planes);
        std::uint32_t strides[3] = {0, 0, 0}, sizes[3] = {0, 0, 0};
        for (int p = 0; p < planes; ++p) {
            const int rows = p == 0 ? rows0 : rows0 / 2;  // nv12 UV half-height
            strides[p] = static_cast<std::uint32_t>(row_bytes) + (p == 0 ? stride_pad.load() : 0);
            sizes[p] = strides[p] * static_cast<std::uint32_t>(rows);
            put32(resp.data() + 20 + 4 * p, strides[p]);
            put32(resp.data() + 32 + 4 * p, sizes[p]);
        }
        int total = static_cast<int>(count) * planes;
        if (alloc_fds_to_send >= 0) total = alloc_fds_to_send.load();
        std::vector<int> fds;
        for (int i = 0; i < total; ++i) {
            const int p = i % planes;
            const int m = ::memfd_create("fake-dsp-plane", 0);
            if (m < 0) return false;
            if (::ftruncate(m, static_cast<off_t>(sizes[p])) != 0) {
                ::close(m);
                return false;
            }
            const std::vector<std::uint8_t> fill(sizes[p], 0xAA);  // padding sentinel
            if (::write(m, fill.data(), fill.size()) != static_cast<ssize_t>(fill.size())) {
                ::close(m);
                return false;
            }
            fds.push_back(m);
            std::lock_guard<std::mutex> g(mu);
            kept_plane_fds.push_back(m);
        }
        for (std::uint32_t i = 0; i < count && i < 64; ++i) put64(resp.data() + 48 + 8 * i, 0x100 + i);
        return send_with_fds(fd, resp.data(), resp.size(), fds);
    }
};

std::string temp_sock_path() {
    static std::atomic<int> seq{0};
    return "/tmp/dsp_fake_" + std::to_string(::getpid()) + "_" + std::to_string(seq++);
}

bool mat_eq(const cv::Mat& a, const cv::Mat& b) {
    if (a.size() != b.size() || a.type() != b.type()) return false;
    cv::Mat diff;
    cv::absdiff(a, b, diff);
    return cv::countNonZero(diff.reshape(1, static_cast<int>(diff.total()))) == 0;
}

}  // namespace

// ---- alloc + padded write/read round-trip --------------------------------------
TEST(DspLifecycle, AllocGray8PaddedRoundTrip) {
    FakeDspDaemon daemon(temp_sock_path());
    daemon.stride_pad = 4;  // stride 68 > row bytes 64
    DspClient client(daemon.path);
    auto pool = client.alloc_buffers(64, 32, "gray8", 2);
    ASSERT_EQ(pool->count(), 2);
    EXPECT_EQ(pool->strides[0], 68u);
    EXPECT_EQ(pool->plane_fds.size(), 2u);

    cv::Mat img(32, 64, CV_8UC1);
    cv::randu(img, cv::Scalar(0), cv::Scalar(255));
    ASSERT_NO_THROW(pool->write(1, img));
    EXPECT_TRUE(mat_eq(pool->read(1), img));
    EXPECT_EQ(pool->buffer_id(1), 0x101u);

    // padding columns survive the write untouched (stride 68, rows 64 B) —
    // checked through the daemon-side copy of the same plane
    void* map = ::mmap(nullptr, 68u * 32u, PROT_READ, MAP_SHARED, daemon.kept_plane_fds[1], 0);
    ASSERT_NE(map, MAP_FAILED);
    const auto* rows = static_cast<const std::uint8_t*>(map);
    for (int r = 0; r < 32; ++r)
        for (int c = 64; c < 68; ++c) EXPECT_EQ(rows[r * 68 + c], 0xAA) << "r=" << r;
    ::munmap(map, 68u * 32u);
}

TEST(DspLifecycle, AllocNv12RoundTrip) {
    FakeDspDaemon daemon(temp_sock_path());
    DspClient client(daemon.path);
    auto pool = client.alloc_buffers(64, 32, "nv12", 1);
    ASSERT_EQ(pool->plane_fds.size(), 2u);  // one dma-buf fd per plane
    cv::Mat nv12(48, 64, CV_8UC1);          // (h*3/2, w)
    cv::randu(nv12, cv::Scalar(0), cv::Scalar(255));
    ASSERT_NO_THROW(pool->write(0, nv12));
    EXPECT_TRUE(mat_eq(pool->read(0), nv12));
}

// ---- wire error surface --------------------------------------------------------
TEST(DspLifecycle, AllocRefused) {
    FakeDspDaemon daemon(temp_sock_path());
    daemon.alloc_code = -3;  // quota exceeded
    DspClient client(daemon.path);
    try {
        client.alloc_buffers(64, 32, "gray8", 1);
        FAIL() << "expected DspError";
    } catch (const DspError& e) {
        EXPECT_NE(std::string(e.what()).find("quota exceeded"), std::string::npos);
    }
}

TEST(DspLifecycle, AllocFdCountMismatch) {
    FakeDspDaemon daemon(temp_sock_path());
    daemon.alloc_fds_to_send = 1;  // claims count=2 x 1 plane, sends one fd
    DspClient client(daemon.path);
    try {
        client.alloc_buffers(64, 32, "gray8", 2);
        FAIL() << "expected DspError";
    } catch (const DspError& e) {
        EXPECT_NE(std::string(e.what()).find("alloc returned"), std::string::npos);
    }
}

TEST(DspLifecycle, AllocWrongResponseType) {
    FakeDspDaemon daemon(temp_sock_path());
    daemon.wrong_alloc_type = wire::kMsgOk;
    DspClient client(daemon.path);
    try {
        client.alloc_buffers(64, 32, "gray8", 1);
        FAIL() << "expected DspError";
    } catch (const DspError& e) {
        EXPECT_NE(std::string(e.what()).find("unexpected DSP alloc response type"),
                  std::string::npos);
    }
}

// ---- release + close poisoning -------------------------------------------------
TEST(DspLifecycle, ReleaseSendsWireMessages) {
    FakeDspDaemon daemon(temp_sock_path());
    DspClient client(daemon.path);
    auto pool = client.alloc_buffers(64, 32, "gray8", 2);
    const std::uint64_t id0 = pool->buffer_id(0);
    const std::uint64_t id1 = pool->buffer_id(1);
    pool->release();
    EXPECT_TRUE(pool->released());
    pool->release();  // idempotent
    client.release_buffer(0x7777);
    std::this_thread::sleep_for(std::chrono::milliseconds(200));  // fire-and-forget
    std::lock_guard<std::mutex> g(daemon.mu);
    ASSERT_EQ(daemon.released_ids.size(), 3u);
    EXPECT_EQ(daemon.released_ids[0], id0);
    EXPECT_EQ(daemon.released_ids[1], id1);
    EXPECT_EQ(daemon.released_ids[2], 0x7777u);
}

TEST(DspLifecycle, ClientClosePoisonsPools) {
    FakeDspDaemon daemon(temp_sock_path());
    DspClient client(daemon.path);
    auto pool = client.alloc_buffers(64, 32, "gray8", 1);
    client.close();
    EXPECT_TRUE(pool->released());
    EXPECT_FALSE(client.has_live_pools());
    const cv::Mat img(32, 64, CV_8UC1);
    try {
        pool->write(0, img);
        FAIL() << "expected DspError";
    } catch (const DspError& e) {
        EXPECT_EQ(std::string(e.what()), "write on released pool");
    }
}

// ---- shared import ring ----------------------------------------------------------
TEST(DspLifecycle, SharedImportLifecycle) {
    FakeDspDaemon daemon(temp_sock_path());
    DspClient client(daemon.path);
    auto ring = client.import_shared_buffers(32, 16, "argb", 2, 2.0);
    ASSERT_EQ(ring->count(), 2);
    EXPECT_EQ(ring->buffer_id(0), 0x9001u);
    EXPECT_EQ(ring->buffer_id(1), 0x9002u);
    const cv::Mat argb(16, 32, CV_8UC4, cv::Scalar(9, 8, 7, 6));
    ASSERT_NO_THROW(ring->write(1, argb));
    try {
        ring->write(0, cv::Mat(8, 8, CV_8UC1));  // wrong shape
        FAIL() << "expected DspError";
    } catch (const DspError& e) {
        EXPECT_NE(std::string(e.what()).find("write expects"), std::string::npos);
    }
    ring->release();
    EXPECT_TRUE(ring->released());
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    std::lock_guard<std::mutex> g(daemon.mu);
    ASSERT_EQ(daemon.released_ids.size(), 2u);
    EXPECT_EQ(daemon.released_ids[0], 0x9001u);
    EXPECT_EQ(daemon.released_ids[1], 0x9002u);
}

TEST(DspLifecycle, ImportTimeout) {
    FakeDspDaemon daemon(temp_sock_path());
    daemon.mute_import = true;
    DspClient client(daemon.path);
    try {
        client.import_shared_buffers(32, 16, "argb", 1, 0.4);
        FAIL() << "expected DspError";
    } catch (const DspError& e) {
        EXPECT_NE(std::string(e.what()).find("may predate DSP_IMPORT"), std::string::npos);
    }
}

// ---- pool write shape validation (no wire traffic beyond the alloc) ---------------
TEST(DspLifecycle, PoolWriteShapeMismatch) {
    FakeDspDaemon daemon(temp_sock_path());
    DspClient client(daemon.path);
    auto pool = client.alloc_buffers(64, 32, "gray8", 1);
    try {
        pool->write(0, cv::Mat(8, 8, CV_8UC3));
        FAIL() << "expected DspError";
    } catch (const DspError& e) {
        EXPECT_NE(std::string(e.what()).find("write expects"), std::string::npos);
    }
    try {
        pool->buffer_id(5);
        FAIL() << "expected DspError";
    } catch (const DspError& e) {
        EXPECT_NE(std::string(e.what()).find("outside the pool's"), std::string::npos);
    }
}
