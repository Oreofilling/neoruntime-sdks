// test_injection_slots.cpp — FramePublisher against in-process fakes: a
// trimmed copy of test_dsp_lifecycle's fake camera.sock daemon (memfd
// planes over SCM_RIGHTS, globally-unique buffer ids) plus an in-process
// gRPC CameraControl service (ServerBuilder on a temp unix socket, the
// plugin.cpp PluginServer trick). This is the "进程内 fake daemon 覆盖
// injection 协议时序" half of the plan's translation-layer strategy.
//
// Ported expectations from python/tests/test_injection.py: the ctor
// validation ladder, legacy depth-paced rotation ([0, 1, 0]), lease slot
// selection / recycle / exhaustion, publish_stream's request shape (pts
// padding, EOS tail with empty stream_id), the argb shared-import ring
// (HAL refuses ARGB32 alloc), publish_rgb's hardware staging chain (job
// src/dst ids, no CPU stage write on the hw leg, CPU write on fallback),
// staging pools allocated once, and session_id on every request.
#include <gtest/gtest.h>
#include <grpcpp/grpcpp.h>
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
#include <deque>
#include <initializer_list>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "camera-daemon/camera.grpc.pb.h"
#include "neoruntime_ipc_sdk/camera.hpp"
#include "neoruntime_ipc_sdk/dsp.hpp"
#include "neoruntime_ipc_sdk/injection.hpp"

#include "detail/dsp_wire.hpp"
#include "detail/injection_logic.hpp"

namespace pb = aipc::camera;
namespace wire = neoruntime_ipc_sdk::detail::dsp_wire;
namespace logic = neoruntime_ipc_sdk::detail::injection_logic;
using neoruntime_ipc_sdk::CameraClient;
using neoruntime_ipc_sdk::DspClient;
using neoruntime_ipc_sdk::FramePublisher;

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

const char* fmt_name(std::uint32_t f) {
    if (f == wire::kHalFormatNv12) return "nv12";
    if (f == wire::kHalFormatRgb24) return "rgb24";
    if (f == wire::kHalFormatArgb) return "argb";
    if (f == wire::kHalFormatGray8) return "gray8";
    return "?";
}

// Fake camera.sock: answers DSP_ALLOC with memfd planes (0xAA-filled, kept
// daemon-side so tests can mmap them), DSP_IMPORT with fresh ids, and
// records DSP_BUF_RELEASEs. Buffer ids are globally unique across allocs —
// the injection publisher resolves its pool ids once at construction, so
// per-response ids (as in test_dsp_lifecycle) would collide across pools.
struct FakeDspDaemon {
    std::string path;
    int listen_fd = -1;
    std::thread accept_thread;
    std::atomic<bool> stop{false};

    struct AllocRec {
        std::string fmt;
        std::uint32_t count = 0;
        int planes = 0;
        std::uint64_t first_id = 0;
        int first_fd = 0;  // index into kept_plane_fds
    };

    std::mutex mu;
    std::vector<AllocRec> allocs;
    std::vector<std::uint64_t> released_ids;
    std::vector<int> kept_plane_fds;  // daemon-side memfd originals (0xAA fill)
    int import_count = 0;
    std::uint64_t next_id = 0x100;

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
                std::vector<std::uint8_t> resp(wire::kImportRespSize, 0);
                put32(resp.data(), wire::kMsgDspImportResp);
                put32(resp.data() + 4, wire::kImportRespSize);
                put32(resp.data() + 8, 0);  // code ok
                {
                    std::lock_guard<std::mutex> g(mu);
                    put64(resp.data() + 16,
                          0x9000 + static_cast<std::uint64_t>(++import_count));
                }
                if (!send_with_fds(fd, resp.data(), resp.size(), {})) return;
            }
        }
    }

    bool send_alloc(int fd, std::uint32_t w, std::uint32_t h, std::uint32_t fmt,
                    std::uint32_t count) {
        const int planes = fmt == wire::kHalFormatNv12 ? 2 : 1;
        const int row_bytes =
            planes == 2 ? static_cast<int>(w)
                        : static_cast<int>(w * (fmt == wire::kHalFormatRgb24 ? 3 : 4));
        std::uint32_t strides[3] = {0, 0, 0}, sizes[3] = {0, 0, 0};
        for (int p = 0; p < planes; ++p) {
            const int rows = p == 0 ? static_cast<int>(h)
                                    : static_cast<int>(h) / 2;  // nv12 UV half-height
            strides[p] = static_cast<std::uint32_t>(row_bytes);
            sizes[p] = strides[p] * static_cast<std::uint32_t>(rows);
        }
        std::vector<std::uint8_t> resp(wire::kAllocRespSize, 0);
        put32(resp.data(), wire::kMsgDspAllocResp);
        put32(resp.data() + 4, wire::kAllocRespSize);
        put32(resp.data() + 8, 0);  // code ok
        put32(resp.data() + 12, count);
        put32(resp.data() + 16, planes);
        for (int p = 0; p < planes; ++p) {
            put32(resp.data() + 20 + 4 * p, strides[p]);
            put32(resp.data() + 32 + 4 * p, sizes[p]);
        }
        std::vector<int> fds;
        for (std::uint32_t i = 0; i < count; ++i) {
            for (int p = 0; p < planes; ++p) {
                const int m = ::memfd_create("fake-dsp-plane", 0);
                if (m < 0) return false;
                if (::ftruncate(m, static_cast<off_t>(sizes[p])) != 0) {
                    ::close(m);
                    return false;
                }
                const std::vector<std::uint8_t> fill(sizes[p], 0xAA);  // sentinel
                if (::write(m, fill.data(), fill.size()) !=
                    static_cast<ssize_t>(fill.size())) {
                    ::close(m);
                    return false;
                }
                fds.push_back(m);
            }
        }
        {
            std::lock_guard<std::mutex> g(mu);
            AllocRec rec;
            rec.fmt = fmt_name(fmt);
            rec.count = count;
            rec.planes = planes;
            rec.first_id = next_id;
            rec.first_fd = static_cast<int>(kept_plane_fds.size());
            for (std::uint32_t i = 0; i < count && i < 64; ++i)
                put64(resp.data() + 48 + 8 * i, next_id + i);
            next_id += count;
            kept_plane_fds.insert(kept_plane_fds.end(), fds.begin(), fds.end());
            allocs.push_back(rec);
        }
        return send_with_fds(fd, resp.data(), resp.size(), fds);
    }
};

// In-process CameraControl service: queueable GetStreamStatus streams,
// GetInjectionStatus snapshots and PushFrame results; records every
// PushFrame / PushFrameStream request and SubmitDspJob (the dsp_unimplemented
// toggle drives convert_hw's CPU-fallback leg).
struct FakeCameraControl : pb::CameraControl::Service {
    std::mutex mu;
    std::vector<pb::StreamStatusInfo> streams;
    std::deque<pb::InjectionStatusResponse> statuses;
    pb::InjectionStatusResponse status_default;
    std::deque<pb::PushFrameResponse> push_results;
    pb::PushFrameResponse push_result_default;
    std::vector<pb::PushFrameRequest> pushes;         // unary PushFrame
    std::vector<pb::PushFrameRequest> stream_pushes;  // PushFrameStream
    std::vector<pb::DspJobRequest> dsp_jobs;
    bool dsp_unimplemented = false;

    grpc::Status GetStreamStatus(grpc::ServerContext*, const pb::GetStreamStatusRequest*,
                                 pb::GetStreamStatusResponse* resp) override {
        std::lock_guard<std::mutex> g(mu);
        for (const auto& s : streams) resp->add_streams()->CopyFrom(s);
        return grpc::Status::OK;
    }
    grpc::Status PushFrame(grpc::ServerContext*, const pb::PushFrameRequest* req,
                           pb::PushFrameResponse* resp) override {
        std::lock_guard<std::mutex> g(mu);
        pushes.push_back(*req);
        if (!push_results.empty()) {
            *resp = std::move(push_results.front());
            push_results.pop_front();
        } else {
            *resp = push_result_default;
        }
        return grpc::Status::OK;
    }
    grpc::Status PushFrameStream(
        grpc::ServerContext*, grpc::ServerReader<pb::PushFrameRequest>* reader,
        pb::PushFrameResponse* resp) override {
        pb::PushFrameRequest r;
        while (reader->Read(&r)) {
            std::lock_guard<std::mutex> g(mu);
            stream_pushes.push_back(r);
        }
        std::lock_guard<std::mutex> g(mu);
        if (!push_results.empty()) {
            *resp = std::move(push_results.front());
            push_results.pop_front();
        } else {
            *resp = push_result_default;
        }
        return grpc::Status::OK;
    }
    grpc::Status GetInjectionStatus(grpc::ServerContext*, const pb::Empty*,
                                    pb::InjectionStatusResponse* resp) override {
        std::lock_guard<std::mutex> g(mu);
        if (!statuses.empty()) {
            *resp = std::move(statuses.front());
            statuses.pop_front();
        } else {
            *resp = status_default;
        }
        return grpc::Status::OK;
    }
    grpc::Status SubmitDspJob(grpc::ServerContext*, const pb::DspJobRequest* req,
                              pb::DspJobResponse* resp) override {
        std::lock_guard<std::mutex> g(mu);
        dsp_jobs.push_back(*req);
        if (dsp_unimplemented)
            return grpc::Status(grpc::StatusCode::UNIMPLEMENTED, "dsp service off");
        resp->set_success(true);
        resp->set_elapsed_ms(1);
        return grpc::Status::OK;
    }
};

std::string temp_sock_path(const std::string& tag) {
    static std::atomic<int> seq{0};
    return "/tmp/inj_" + tag + "_" + std::to_string(::getpid()) + "_" + std::to_string(seq++);
}

// One full fake platform: DSP buffer-plane daemon + camera-control gRPC
// service, sharing nothing (the two real daemons don't either).
struct FakeStack {
    FakeDspDaemon daemon;
    FakeCameraControl cam_svc;
    std::string endpoint;
    std::unique_ptr<grpc::Server> server;

    explicit FakeStack(const std::string& tag) : daemon(temp_sock_path(tag)) {
        cam_svc.status_default.set_success(true);
        cam_svc.push_result_default.set_success(true);
        const std::string sock = temp_sock_path(tag) + "_ctl";
        endpoint = "unix://" + sock;  // triple-slash, the SDK's endpoint form
        grpc::ServerBuilder builder;
        builder.RegisterService(&cam_svc);
        builder.AddListeningPort(endpoint, grpc::InsecureServerCredentials());
        server = builder.BuildAndStart();
        if (!server) std::fprintf(stderr, "fake camera-control server failed to start\n");
    }
    ~FakeStack() {
        if (server) server->Shutdown();
    }
};

void add_stream(FakeCameraControl& svc, const std::string& id, int w, int h,
                bool encoder = true) {
    pb::StreamStatusInfo s;
    s.set_stream_id(id);
    s.set_status("active");
    s.set_has_encoder(encoder);
    s.set_width(static_cast<std::uint32_t>(w));
    s.set_height(static_cast<std::uint32_t>(h));
    std::lock_guard<std::mutex> g(svc.mu);
    svc.streams.push_back(std::move(s));
}

void queue_status(FakeCameraControl& svc, bool reports,
                  std::initializer_list<std::uint64_t> busy) {
    pb::InjectionStatusResponse r;
    r.set_success(true);
    r.set_reports_in_flight_buffers(reports);
    for (std::uint64_t id : busy) r.add_in_flight_buffer_ids(id);
    std::lock_guard<std::mutex> g(svc.mu);
    svc.statuses.push_back(std::move(r));
}

void queue_push_result(FakeCameraControl& svc, std::initializer_list<std::uint64_t> busy) {
    pb::PushFrameResponse r;
    r.set_success(true);
    for (std::uint64_t id : busy) r.add_in_flight_buffer_ids(id);
    std::lock_guard<std::mutex> g(svc.mu);
    svc.push_results.push_back(std::move(r));
}

// First byte of a daemon-kept memfd plane (PROT_READ mmap of size `bytes`).
std::uint8_t first_byte(int fd, std::size_t bytes) {
    void* map = ::mmap(nullptr, bytes, PROT_READ, MAP_SHARED, fd, 0);
    if (map == MAP_FAILED) return 0;
    const std::uint8_t v = *static_cast<const std::uint8_t*>(map);
    ::munmap(map, bytes);
    return v;
}

}  // namespace

// ---- pure helpers (python's _to_pool_array / _acquire_slot math) ------------------
TEST(InjectionLogic, PickFreeSlot) {
    const std::vector<std::uint64_t> ids{10, 11, 12, 13};
    std::set<std::uint64_t> busy{12};
    EXPECT_EQ(logic::pick_free_slot(2, ids, busy), 3);  // slot 2 busy -> 3
    EXPECT_EQ(logic::pick_free_slot(3, ids, busy), 3);  // slot 3 itself free
    EXPECT_EQ(logic::pick_free_slot(1, ids, busy), 1);  // slot 1 free at cursor
    busy = {10, 11, 12};
    EXPECT_EQ(logic::pick_free_slot(1, ids, busy), 3);  // wraps past 3 busy
    busy = {10, 11, 12, 13};
    EXPECT_EQ(logic::pick_free_slot(0, ids, busy), -1);
}

TEST(InjectionLogic, PoolArrayValidation) {
    const cv::Mat good(48, 64, CV_8UC1);
    EXPECT_NO_THROW(logic::to_pool_array(good, "nv12", 64, 32));
    try {
        logic::to_pool_array(cv::Mat(48, 64, CV_32FC1), "nv12", 64, 32);
        FAIL() << "expected invalid_argument";
    } catch (const std::invalid_argument& e) {
        EXPECT_NE(std::string(e.what()).find("dtype must be uint8, got float32"),
                  std::string::npos);
    }
    try {
        logic::to_pool_array(cv::Mat(8, 8, CV_8UC1), "nv12", 64, 32);
        FAIL() << "expected invalid_argument";
    } catch (const std::invalid_argument& e) {
        EXPECT_NE(std::string(e.what()).find("shape must be (48, 64), got (8, 8)"),
                  std::string::npos);
    }
    try {
        logic::to_pool_array(cv::Mat(32, 64, CV_8UC3), "argb", 64, 32);
        FAIL() << "expected invalid_argument";
    } catch (const std::invalid_argument& e) {
        EXPECT_NE(std::string(e.what()).find("shape must be (32, 64, 4), got (32, 64, 3)"),
                  std::string::npos);
    }
    try {
        logic::to_pool_array("12345678", 8, "nv12", 64, 32);
        FAIL() << "expected invalid_argument";
    } catch (const std::invalid_argument& e) {
        EXPECT_NE(std::string(e.what()).find("tight nv12 bytes must be 3072 bytes "
                                             "(64x32 nv12), got 8"),
                  std::string::npos);
    }
    // Tight-byte overload wraps without copying (2x2 nv12 = 3 rows = 6 bytes).
    const cv::Mat wrapped = logic::to_pool_array("abcdef", 6, "nv12", 2, 2);
    EXPECT_EQ(wrapped.rows, 3);
    EXPECT_EQ(wrapped.cols, 2);
    EXPECT_EQ(wrapped.channels(), 1);
}

TEST(InjectionLogic, ExhaustionMessage) {
    std::set<std::uint64_t> busy{101, 102};
    const std::string msg = logic::exhaustion_message("sub", 0.05, busy);
    EXPECT_NE(msg.find("no free pool slot"), std::string::npos);
    EXPECT_NE(msg.find("the daemon still holds [101, 102] in flight"), std::string::npos);
    const std::string all = logic::exhaustion_message("sub", 5.0, {});
    EXPECT_NE(all.find("the daemon still holds all slots in flight"), std::string::npos);
}

// ---- ctor validation ladder --------------------------------------------------------
TEST(FramePublisherCtor, ValidationLadder) {
    FakeStack st("ctor");
    add_stream(st.cam_svc, "main", 1920, 1080);
    add_stream(st.cam_svc, "sub", 640, 480);
    add_stream(st.cam_svc, "raw", 640, 480, /*encoder=*/false);
    add_stream(st.cam_svc, "odd", 641, 480);
    CameraClient cam(st.endpoint);
    DspClient dsp(st.daemon.path, st.endpoint);

    const auto expect_invalid = [&](const char* needle, const std::string& stream, int depth,
                                    FramePublisher::Config cfg = FramePublisher::Config{}) {
        try {
            FramePublisher pub(cam, dsp, stream, depth, cfg);
            FAIL() << "expected invalid_argument: " << needle;
        } catch (const std::invalid_argument& e) {
            EXPECT_NE(std::string(e.what()).find(needle), std::string::npos) << e.what();
        }
    };
    const auto expect_runtime = [&](const char* needle, const std::string& stream, int depth,
                                    FramePublisher::Config cfg) {
        try {
            FramePublisher pub(cam, dsp, stream, depth, cfg);
            FAIL() << "expected runtime_error: " << needle;
        } catch (const std::runtime_error& e) {
            EXPECT_NE(std::string(e.what()).find(needle), std::string::npos) << e.what();
        }
    };

    FramePublisher::Config cfg;
    expect_invalid("pool_depth must be >= 1", "sub", 0);
    cfg.mode = "blend";
    expect_invalid("mode must be one of ('replace', 'overlay'), got 'blend'", "sub", 4, cfg);
    cfg = FramePublisher::Config{};
    expect_runtime("no live stream named 'third'; known: ['main', 'sub', 'raw', 'odd']", "third",
                   4, cfg);
    expect_runtime("stream 'raw' has no encoder", "raw", 4, cfg);
    expect_invalid("must be nonzero and even", "odd", 4);
    cfg.fmt = "rgb24";
    expect_invalid("REPLACE consumes NV12 only", "sub", 4, cfg);
    cfg.mode = "overlay";
    cfg.fmt = "yuv";
    expect_invalid(
        "overlay fmt must be 'nv12' (opaque paste) or 'argb' (alpha blend), got 'yuv'", "main", 4,
        cfg);
    cfg.fmt = "nv12";
    cfg.inset = std::make_pair(481, 270);
    expect_invalid("inset 481x270 must be nonzero and even", "main", 4, cfg);
    cfg.inset = std::make_pair(480, 270);
    cfg.dest = {1, 0};
    expect_invalid("dest (1, 0) must be even", "main", 4, cfg);
    cfg.dest = {1500, 900};
    expect_invalid("inset 480x270 at (1500, 900) exceeds stream 1920x1080", "main", 4, cfg);

    // Every failed ctor above validated before the first allocation.
    {
        std::lock_guard<std::mutex> g(st.daemon.mu);
        EXPECT_TRUE(st.daemon.allocs.empty());
    }
    // A fitting overlay constructs with the PIP default geometry (an even
    // quarter of 1920x1080 = 480x270).
    cfg = FramePublisher::Config{};
    cfg.mode = "overlay";
    FramePublisher pub(cam, dsp, "main", 4, cfg);
    EXPECT_EQ(pub.width(), 480);
    EXPECT_EQ(pub.height(), 270);
    EXPECT_EQ(pub.dest(), (std::pair<int, int>{0, 0}));
    {
        std::lock_guard<std::mutex> g(st.daemon.mu);
        ASSERT_EQ(st.daemon.allocs.size(), 1u);
        EXPECT_EQ(st.daemon.allocs[0].fmt, "nv12");
        EXPECT_EQ(st.daemon.allocs[0].count, 4u);
    }
}

// ---- legacy rotation + request shape ------------------------------------------------
TEST(FramePublisherPublish, LegacyRotationAndRequestShape) {
    FakeStack st("legacy");
    add_stream(st.cam_svc, "main", 1920, 1080);
    add_stream(st.cam_svc, "sub", 640, 480);
    CameraClient cam(st.endpoint);
    DspClient dsp(st.daemon.path, st.endpoint);

    FramePublisher::Config cfg;
    cfg.session_id = "sess-1";
    FramePublisher pub(cam, dsp, "sub", 2, cfg);
    EXPECT_FALSE(pub.lease_mode());  // default status: legacy daemon

    cv::Mat nv12(720, 640, CV_8UC1);  // (h*3/2, w)
    cv::randu(nv12, cv::Scalar(0), cv::Scalar(255));
    ASSERT_NO_THROW(pub.publish(nv12, 11));
    ASSERT_NO_THROW(pub.publish(nv12, 22));
    ASSERT_NO_THROW(pub.publish(nv12, 33));

    std::uint64_t id0, id1;
    {
        std::lock_guard<std::mutex> g(st.daemon.mu);
        ASSERT_EQ(st.daemon.allocs.size(), 1u);
        id0 = st.daemon.allocs[0].first_id;
        id1 = id0 + 1;
    }
    {
        std::lock_guard<std::mutex> g(st.cam_svc.mu);
        ASSERT_EQ(st.cam_svc.pushes.size(), 3u);
        const std::uint64_t want[3] = {id0, id1, id0};  // blind depth-2 rotation
        for (int i = 0; i < 3; ++i) {
            const pb::PushFrameRequest& r = st.cam_svc.pushes[i];
            EXPECT_EQ(r.buffer_id(), want[i]) << "push " << i;
            EXPECT_EQ(r.width(), 640u);
            EXPECT_EQ(r.height(), 480u);
            EXPECT_EQ(r.stride(), 640u);
            EXPECT_EQ(r.mode(), pb::INJECT_REPLACE);
            EXPECT_EQ(r.pts_ns(), static_cast<std::uint64_t>(11 * (i + 1)));
            EXPECT_EQ(r.dest_x(), 0u);
            EXPECT_EQ(r.dest_y(), 0u);
            EXPECT_EQ(r.stream_id(), "sub");
            EXPECT_EQ(r.session_id(), "sess-1");
            EXPECT_FALSE(r.end_of_stream());
        }
    }

    // EOS: buffer_id 0 + end_of_stream; python's dict leaves stream_id and
    // mode unset, so the request carries an empty stream_id.
    pub.publish_eos();
    {
        std::lock_guard<std::mutex> g(st.cam_svc.mu);
        ASSERT_EQ(st.cam_svc.pushes.size(), 4u);
        const pb::PushFrameRequest& r = st.cam_svc.pushes[3];
        EXPECT_EQ(r.buffer_id(), 0u);
        EXPECT_TRUE(r.end_of_stream());
        EXPECT_EQ(r.session_id(), "sess-1");
        EXPECT_EQ(r.stream_id(), "");
        EXPECT_EQ(r.width(), 640u);
        EXPECT_EQ(r.height(), 480u);
        EXPECT_EQ(r.stride(), 640u);
    }

    pub.close();
    std::this_thread::sleep_for(std::chrono::milliseconds(200));  // fire-and-forget
    std::lock_guard<std::mutex> g(st.daemon.mu);
    ASSERT_EQ(st.daemon.released_ids.size(), 2u);
    EXPECT_EQ(st.daemon.released_ids[0], id0);
    EXPECT_EQ(st.daemon.released_ids[1], id1);
}

TEST(FramePublisherPublish, LegacyRotationIgnoresLeaseSnapshots) {
    FakeStack st("legsnap");
    add_stream(st.cam_svc, "sub", 640, 480);
    CameraClient cam(st.endpoint);
    DspClient dsp(st.daemon.path, st.endpoint);
    FramePublisher pub(cam, dsp, "sub", 2);  // legacy: reports_in_flight_buffers=false

    // The daemon claims every slot busy in every response; legacy mode
    // must keep rotating blindly (it cannot see the lease set).
    queue_push_result(st.cam_svc, {0x100, 0x101});
    queue_push_result(st.cam_svc, {0x100, 0x101});
    queue_push_result(st.cam_svc, {0x100, 0x101});

    cv::Mat nv12(720, 640, CV_8UC1);
    ASSERT_NO_THROW(pub.publish(nv12));
    ASSERT_NO_THROW(pub.publish(nv12));
    ASSERT_NO_THROW(pub.publish(nv12));
    std::lock_guard<std::mutex> g(st.cam_svc.mu);
    ASSERT_EQ(st.cam_svc.pushes.size(), 3u);
    EXPECT_EQ(st.cam_svc.pushes[0].buffer_id(), 0x100u);
    EXPECT_EQ(st.cam_svc.pushes[1].buffer_id(), 0x101u);
    EXPECT_EQ(st.cam_svc.pushes[2].buffer_id(), 0x100u);
}

// ---- lease mode ----------------------------------------------------------------------
TEST(FramePublisherPublish, LeaseSlotSelectionAndRecycle) {
    FakeStack st("lease");
    add_stream(st.cam_svc, "sub", 640, 480);
    CameraClient cam(st.endpoint);
    DspClient dsp(st.daemon.path, st.endpoint);

    queue_status(st.cam_svc, /*reports=*/true, {});  // ctor capability probe
    FramePublisher pub(cam, dsp, "sub", 2);
    EXPECT_TRUE(pub.lease_mode());

    queue_push_result(st.cam_svc, {0x100});         // daemon holds slot 0
    queue_push_result(st.cam_svc, {0x100, 0x101});  // holds both
    queue_status(st.cam_svc, /*reports=*/true, {0x100});  // slot 1 baked+released
    queue_push_result(st.cam_svc, {0x101});

    cv::Mat nv12(720, 640, CV_8UC1);
    ASSERT_NO_THROW(pub.publish(nv12));
    ASSERT_NO_THROW(pub.publish(nv12));
    ASSERT_NO_THROW(pub.publish(nv12));  // skips busy 0x100, recycles 0x101

    std::lock_guard<std::mutex> g(st.cam_svc.mu);
    ASSERT_EQ(st.cam_svc.pushes.size(), 3u);
    EXPECT_EQ(st.cam_svc.pushes[0].buffer_id(), 0x100u);
    EXPECT_EQ(st.cam_svc.pushes[1].buffer_id(), 0x101u);
    EXPECT_EQ(st.cam_svc.pushes[2].buffer_id(), 0x101u);
}

TEST(FramePublisherPublish, LeaseExhaustionRaisesWithoutWriteOrPush) {
    FakeStack st("exhaust");
    add_stream(st.cam_svc, "sub", 640, 480);
    CameraClient cam(st.endpoint);
    DspClient dsp(st.daemon.path, st.endpoint);

    // The daemon reports every slot busy, forever.
    queue_status(st.cam_svc, /*reports=*/true, {});
    {
        std::lock_guard<std::mutex> g(st.cam_svc.mu);
        st.cam_svc.status_default.set_success(true);
        st.cam_svc.status_default.set_reports_in_flight_buffers(true);
        st.cam_svc.status_default.add_in_flight_buffer_ids(0x100);
        st.cam_svc.status_default.add_in_flight_buffer_ids(0x101);
        st.cam_svc.push_result_default.add_in_flight_buffer_ids(0x100);
        st.cam_svc.push_result_default.add_in_flight_buffer_ids(0x101);
    }
    FramePublisher::Config cfg;
    cfg.lease_timeout_s = 0.05;
    FramePublisher pub(cam, dsp, "sub", 2, cfg);
    EXPECT_TRUE(pub.lease_mode());

    cv::Mat nv12(720, 640, CV_8UC1);
    ASSERT_NO_THROW(pub.publish(nv12));  // local set starts empty -> slot 0

    const std::size_t pushes_before = [&] {
        std::lock_guard<std::mutex> g(st.cam_svc.mu);
        return st.cam_svc.pushes.size();
    }();
    try {
        pub.publish(nv12);  // every slot busy; lease_timeout_s elapses
        FAIL() << "expected runtime_error";
    } catch (const std::runtime_error& e) {
        EXPECT_NE(std::string(e.what()).find("no free pool slot"), std::string::npos);
        EXPECT_NE(std::string(e.what()).find("in flight"), std::string::npos);
    }
    std::lock_guard<std::mutex> g(st.cam_svc.mu);
    EXPECT_EQ(st.cam_svc.pushes.size(), pushes_before);  // no doomed push went out
}

// ---- zero-copy chain -------------------------------------------------------------------
TEST(FramePublisherPublish, AcquireSlotPushSlotAndClosed) {
    FakeStack st("zc");
    add_stream(st.cam_svc, "sub", 640, 480);
    CameraClient cam(st.endpoint);
    DspClient dsp(st.daemon.path, st.endpoint);
    FramePublisher::Config cfg;
    cfg.session_id = "s-zc";
    FramePublisher pub(cam, dsp, "sub", 4, cfg);

    EXPECT_EQ(pub.acquire_slot(), 0);  // legacy rotation
    EXPECT_EQ(pub.acquire_slot(), 1);
    EXPECT_EQ(pub.acquire_slot(), 2);
    ASSERT_NO_THROW(pub.push_slot(1, 7));
    {
        std::lock_guard<std::mutex> g(st.cam_svc.mu);
        ASSERT_EQ(st.cam_svc.pushes.size(), 1u);
        EXPECT_EQ(st.cam_svc.pushes[0].buffer_id(), 0x101u);  // slot 1's id
        EXPECT_EQ(st.cam_svc.pushes[0].pts_ns(), 7u);
        EXPECT_EQ(st.cam_svc.pushes[0].session_id(), "s-zc");
    }
    try {
        pub.push_slot(9);
        FAIL() << "expected invalid_argument";
    } catch (const std::invalid_argument& e) {
        EXPECT_NE(std::string(e.what()).find("slot 9 outside the pool's 4 slots"),
                  std::string::npos);
    }

    // Pool accessors: nv12 publisher owns a dma-buf pool, no shared ring.
    EXPECT_NE(pub.dsp_pool(), nullptr);
    EXPECT_EQ(pub.dsp_pool()->count(), 4);
    EXPECT_EQ(pub.shared_pool(), nullptr);

    // Python converts the payload before the closed check: a bad shape on
    // a closed publisher still reports the shape error.
    pub.close();
    try {
        pub.publish(cv::Mat(8, 8, CV_8UC1));
        FAIL() << "expected invalid_argument";
    } catch (const std::invalid_argument& e) {
        EXPECT_NE(std::string(e.what()).find("shape must be"), std::string::npos);
    }
    const cv::Mat good(720, 640, CV_8UC1);
    try {
        pub.publish(good);
        FAIL() << "expected runtime_error";
    } catch (const std::runtime_error& e) {
        EXPECT_EQ(std::string(e.what()), "FramePublisher is closed");
    }
    try {
        pub.push_slot(0);
        FAIL() << "expected runtime_error";
    } catch (const std::runtime_error& e) {
        EXPECT_EQ(std::string(e.what()), "FramePublisher is closed");
    }
    try {
        pub.publish_eos();
        FAIL() << "expected runtime_error";
    } catch (const std::runtime_error& e) {
        EXPECT_EQ(std::string(e.what()), "FramePublisher is closed");
    }
    try {
        pub.dsp_pool();
        FAIL() << "expected runtime_error";
    } catch (const std::runtime_error& e) {
        EXPECT_EQ(std::string(e.what()), "FramePublisher is closed");
    }
}

TEST(FramePublisherPublish, DestructorSendsBestEffortEosOnce) {
    FakeStack st("dtreos");
    add_stream(st.cam_svc, "sub", 640, 480);
    CameraClient cam(st.endpoint);
    DspClient dsp(st.daemon.path, st.endpoint);
    {
        FramePublisher pub(cam, dsp, "sub", 2);
        cv::Mat nv12(720, 640, CV_8UC1);
        ASSERT_NO_THROW(pub.publish(nv12));
        // no explicit close: dtor must publish EOS, then close
    }
    {
        std::lock_guard<std::mutex> g(st.cam_svc.mu);
        ASSERT_EQ(st.cam_svc.pushes.size(), 2u);  // frame + dtor EOS
        EXPECT_EQ(st.cam_svc.pushes[1].buffer_id(), 0u);
        EXPECT_TRUE(st.cam_svc.pushes[1].end_of_stream());
    }
    {
        FramePublisher pub(cam, dsp, "sub", 2);
        cv::Mat nv12(720, 640, CV_8UC1);
        ASSERT_NO_THROW(pub.publish(nv12));
        ASSERT_NO_THROW(pub.publish_eos());  // explicit: dtor must not repeat
    }
    // The second publisher's pool starts past the first's ids (globally
    // unique daemon ids), so read its slot-0 id instead of assuming 0x100.
    const std::uint64_t second_pool0 = [&] {
        std::lock_guard<std::mutex> g(st.daemon.mu);
        return st.daemon.allocs[1].first_id;
    }();
    {
        std::lock_guard<std::mutex> g(st.cam_svc.mu);
        ASSERT_EQ(st.cam_svc.pushes.size(), 4u);  // frame + EOS, frame + EOS
        EXPECT_TRUE(st.cam_svc.pushes[3].end_of_stream());
        EXPECT_EQ(st.cam_svc.pushes[2].buffer_id(), second_pool0);
    }
}

// ---- publish_stream -----------------------------------------------------------------------
TEST(FramePublisherPublish, PublishStreamRequestShape) {
    FakeStack st("pstream");
    add_stream(st.cam_svc, "sub", 640, 480);
    CameraClient cam(st.endpoint);
    DspClient dsp(st.daemon.path, st.endpoint);

    queue_status(st.cam_svc, /*reports=*/true, {});  // ctor probe -> lease mode
    FramePublisher::Config cfg;
    cfg.session_id = "stream-s";
    FramePublisher pub(cam, dsp, "sub", 4, cfg);
    EXPECT_TRUE(pub.lease_mode());

    std::vector<cv::Mat> frames(3, cv::Mat(720, 640, CV_8UC1));
    for (auto& f : frames) cv::randu(f, cv::Scalar(0), cv::Scalar(255));
    ASSERT_NO_THROW(pub.publish_stream(frames, /*pts_ns=*/{7, 8}, /*end_with_eos=*/true));

    std::lock_guard<std::mutex> g(st.cam_svc.mu);
    ASSERT_EQ(st.cam_svc.stream_pushes.size(), 4u);  // 3 frames + EOS
    const std::uint64_t want_ids[3] = {0x100, 0x101, 0x102};  // lease: distinct slots
    const std::uint64_t want_pts[3] = {7, 8, 0};              // missing pts pad 0
    for (int i = 0; i < 3; ++i) {
        const pb::PushFrameRequest& r = st.cam_svc.stream_pushes[i];
        EXPECT_EQ(r.buffer_id(), want_ids[i]) << "frame " << i;
        EXPECT_EQ(r.pts_ns(), want_pts[i]) << "frame " << i;
        EXPECT_EQ(r.width(), 640u);
        EXPECT_EQ(r.height(), 480u);
        EXPECT_EQ(r.stride(), 640u);
        EXPECT_EQ(r.mode(), pb::INJECT_REPLACE);
        EXPECT_EQ(r.stream_id(), "sub");
        EXPECT_EQ(r.session_id(), "stream-s");
        EXPECT_FALSE(r.end_of_stream());
    }
    const pb::PushFrameRequest& eos = st.cam_svc.stream_pushes[3];
    EXPECT_EQ(eos.buffer_id(), 0u);
    EXPECT_TRUE(eos.end_of_stream());
    EXPECT_EQ(eos.session_id(), "stream-s");
    EXPECT_EQ(eos.stream_id(), "");  // python's EOS dict leaves it unset
    EXPECT_EQ(st.cam_svc.pushes.size(), 0u);  // nothing rode the unary rpc
}

// ---- argb overlay ring -----------------------------------------------------------------------
TEST(FramePublisherPublish, ArgbOverlayRidesSharedImportRing) {
    FakeStack st("argb");
    add_stream(st.cam_svc, "main", 1920, 1080);
    CameraClient cam(st.endpoint);
    DspClient dsp(st.daemon.path, st.endpoint);

    FramePublisher::Config cfg;
    cfg.mode = "overlay";
    cfg.fmt = "argb";
    FramePublisher pub(cam, dsp, "main", 4, cfg);
    EXPECT_EQ(pub.width(), 480);  // PIP default: even quarter of 1920x1080
    EXPECT_EQ(pub.height(), 270);
    EXPECT_EQ(pub.mode(), "overlay");

    {
        std::lock_guard<std::mutex> g(st.daemon.mu);
        EXPECT_TRUE(st.daemon.allocs.empty());  // HAL refuses ARGB32 alloc
        EXPECT_EQ(st.daemon.import_count, 4);   // one DSP_IMPORT per ring slot
    }
    EXPECT_EQ(pub.dsp_pool(), nullptr);
    ASSERT_NE(pub.shared_pool(), nullptr);
    EXPECT_EQ(pub.shared_pool()->count(), 4);

    const cv::Mat badge(270, 480, CV_8UC4, cv::Scalar(9, 8, 7, 6));
    ASSERT_NO_THROW(pub.publish(badge));
    try {
        pub.publish(cv::Mat(8, 8, CV_8UC1));  // wrong shape
        FAIL() << "expected invalid_argument";
    } catch (const std::invalid_argument& e) {
        // numpy shape repr: single-channel has no third axis, so got is
        // "(8, 8)" — python prints the same.
        EXPECT_NE(std::string(e.what()).find("argb ndarray shape must be (270, 480, 4), got "
                                             "(8, 8)"),
                  std::string::npos);
    }
    const std::string tight(480 * 270 * 4, '\x10');
    ASSERT_NO_THROW(pub.publish(tight.data(), tight.size()));
    try {
        pub.publish("short", 5);
        FAIL() << "expected invalid_argument";
    } catch (const std::invalid_argument& e) {
        EXPECT_NE(std::string(e.what()).find(
                      "tight argb bytes must be 518400 bytes (480x270 argb), got 5"),
                  std::string::npos);
    }

    {
        std::lock_guard<std::mutex> g(st.cam_svc.mu);
        ASSERT_EQ(st.cam_svc.pushes.size(), 2u);
        for (int i = 0; i < 2; ++i) {
            const pb::PushFrameRequest& r = st.cam_svc.pushes[i];
            EXPECT_EQ(r.buffer_id(), 0x9000 + static_cast<std::uint64_t>(i + 1));
            EXPECT_EQ(r.width(), 480u);
            EXPECT_EQ(r.height(), 270u);
            EXPECT_EQ(r.stride(), 480u * 4u);  // ARGB32 byte stride
            EXPECT_EQ(r.mode(), pb::INJECT_OVERLAY);
            EXPECT_EQ(r.stream_id(), "main");
        }
    }
}

// ---- publish_rgb -------------------------------------------------------------------------------
TEST(FramePublisherPublishRgb, HardwareConvertChain) {
    FakeStack st("rgbhw");
    add_stream(st.cam_svc, "sub", 640, 480);
    CameraClient cam(st.endpoint);
    DspClient dsp(st.daemon.path, st.endpoint);
    FramePublisher pub(cam, dsp, "sub");  // REPLACE, nv12 pool depth 4

    const cv::Mat rgb(480, 640, CV_8UC3, cv::Scalar(60, 120, 30));
    ASSERT_NO_THROW(pub.publish_rgb(rgb, 5));
    ASSERT_NO_THROW(pub.publish_rgb(rgb, 6));  // staging pools allocated once

    // allocs: [0] nv12 main pool x4 (ids 0x100..0x103), [1] rgb24 staging
    // (0x104), [2] nv12 stage (0x105) — and no more after the second call.
    std::uint64_t rgb_id = 0, stage_id = 0;
    int stage_y_fd = -1;
    {
        std::lock_guard<std::mutex> g(st.daemon.mu);
        ASSERT_EQ(st.daemon.allocs.size(), 3u);
        ASSERT_EQ(st.daemon.allocs[1].fmt, "rgb24");
        ASSERT_EQ(st.daemon.allocs[2].fmt, "nv12");
        rgb_id = st.daemon.allocs[1].first_id;
        stage_id = st.daemon.allocs[2].first_id;
        stage_y_fd = st.daemon.kept_plane_fds[st.daemon.allocs[2].first_fd];
    }
    {
        std::lock_guard<std::mutex> g(st.cam_svc.mu);
        ASSERT_EQ(st.cam_svc.dsp_jobs.size(), 2u);
        for (int i = 0; i < 2; ++i) {
            const pb::DspJobRequest& j = st.cam_svc.dsp_jobs[i];
            EXPECT_EQ(j.op(), pb::DSP_OP_CONVERT_FORMAT) << "job " << i;
            EXPECT_EQ(j.src_buffer_id(), rgb_id) << "job " << i;  // rgb staging id
            ASSERT_EQ(j.dst_buffer_ids_size(), 1);
            EXPECT_EQ(j.dst_buffer_ids(0), stage_id) << "job " << i;  // nv12 stage id
        }
        ASSERT_EQ(st.cam_svc.pushes.size(), 2u);
        for (int i = 0; i < 2; ++i) {
            const pb::PushFrameRequest& r = st.cam_svc.pushes[i];
            EXPECT_EQ(r.buffer_id(), stage_id) << "push " << i;
            EXPECT_EQ(r.mode(), pb::INJECT_REPLACE);
            EXPECT_EQ(r.width(), 640u);
            EXPECT_EQ(r.height(), 480u);
            EXPECT_EQ(r.stride(), 640u);
            EXPECT_EQ(r.pts_ns(), static_cast<std::uint64_t>(5 + i));
        }
    }
    EXPECT_EQ(dsp.last_used_hw(), std::optional<bool>(true));
    // Hardware path: the DSP owns the stage, so the publisher never wrote
    // it client-side — the daemon's 0xAA sentinel fill is still intact.
    EXPECT_EQ(first_byte(stage_y_fd, 640u * 480u), 0xAA);
}

TEST(FramePublisherPublishRgb, CpuFallbackWritesStage) {
    FakeStack st("rgbcpu");
    add_stream(st.cam_svc, "sub", 640, 480);
    {
        std::lock_guard<std::mutex> g(st.cam_svc.mu);
        st.cam_svc.dsp_unimplemented = true;  // daemon predates the DSP surface
    }
    CameraClient cam(st.endpoint);
    DspClient dsp(st.daemon.path, st.endpoint);
    FramePublisher pub(cam, dsp, "sub");

    const cv::Mat rgb(480, 640, CV_8UC3, cv::Scalar(60, 120, 30));
    ASSERT_NO_THROW(pub.publish_rgb(rgb));
    EXPECT_EQ(dsp.last_used_hw(), std::optional<bool>(false));

    std::uint64_t stage_id = 0;
    int stage_y_fd = -1;
    {
        std::lock_guard<std::mutex> g(st.daemon.mu);
        ASSERT_EQ(st.daemon.allocs.size(), 3u);  // pools still staged client-side
        stage_id = st.daemon.allocs[2].first_id;
        stage_y_fd = st.daemon.kept_plane_fds[st.daemon.allocs[2].first_fd];
    }
    {
        std::lock_guard<std::mutex> g(st.cam_svc.mu);
        ASSERT_EQ(st.cam_svc.dsp_jobs.size(), 1u);  // submit attempted once
        ASSERT_EQ(st.cam_svc.pushes.size(), 1u);
        EXPECT_EQ(st.cam_svc.pushes[0].buffer_id(), stage_id);
    }
    // CPU fallback: the publisher wrote the converted NV12 into the stage
    // (BT.601 luma of (60,120,30) ~ 92, not the 0xAA sentinel).
    EXPECT_NE(first_byte(stage_y_fd, 640u * 480u), 0xAA);
}

TEST(FramePublisherPublishRgb, TargetsReplaceOnly) {
    FakeStack st("rgbovl");
    add_stream(st.cam_svc, "main", 1920, 1080);
    CameraClient cam(st.endpoint);
    DspClient dsp(st.daemon.path, st.endpoint);
    FramePublisher::Config cfg;
    cfg.mode = "overlay";
    FramePublisher pub(cam, dsp, "main", 2, cfg);
    try {
        pub.publish_rgb(cv::Mat(270, 480, CV_8UC3));
        FAIL() << "expected runtime_error";
    } catch (const std::runtime_error& e) {
        EXPECT_EQ(std::string(e.what()), "publish_rgb targets REPLACE publishers only");
    }
}
