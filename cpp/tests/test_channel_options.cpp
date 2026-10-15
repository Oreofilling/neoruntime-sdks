// test_channel_options.cpp — transport.hpp unit tests (Phase 1 transport base).
//
//   * kMaxGrpcMessageLength pins the Python _transport.py MAX_GRPC_MESSAGE_LENGTH
//     contract (daemon replies can exceed gRPC's 4 MiB default).
//   * GrpcClientBase lazy lifecycle: connect / connected / stub / channel /
//     close / re-connect — channels are created lazily and no RPC is issued, so
//     no daemon is needed.
//   * UdsStreamClientBase: bounded reconnect budget, single attempt when
//     reconnecting is disabled, fd reuse, close_fd hook ordering, and derived
//     destructor teardown.
#include <gtest/gtest.h>

#include <grpcpp/grpcpp.h>

#include <sys/socket.h>
#include <unistd.h>

#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "camera-daemon/camera.grpc.pb.h"

#include "detail/transport.hpp"

namespace {

namespace detail = neoruntime_ipc_sdk::detail;

// grpc::ChannelArguments has no getter for the receive-size cap, so pin the
// constant and prove the shared options object constructs cleanly.
TEST(ChannelOptions, MaxReceiveSizeMatchesPythonTransport) {
    EXPECT_EQ(detail::kMaxGrpcMessageLength, 64 * 1024 * 1024);
    grpc::ChannelArguments args = detail::default_channel_args();
    (void)args;
}

TEST(GrpcClientBase, LazyConnectCloseReconnect) {
    struct TestClient : detail::GrpcClientBase<aipc::camera::CameraControl> {
        explicit TestClient(std::string ep) : GrpcClientBase(std::move(ep)) {}
    };

    TestClient c("127.0.0.1:1");  // nothing listening; channel creation is lazy
    EXPECT_FALSE(c.connected());
    EXPECT_EQ(c.endpoint(), "127.0.0.1:1");

    c.connect();
    EXPECT_TRUE(c.connected());
    EXPECT_NE(&c.stub(), nullptr);
    EXPECT_NE(c.channel(), nullptr);

    c.close();
    EXPECT_FALSE(c.connected());

    c.connect();  // re-connect after close
    EXPECT_TRUE(c.connected());
}

// do_connect() that always fails; counts attempts.
struct CountingUds : detail::UdsStreamClientBase {
    int attempts = 0;

    CountingUds(std::string path, bool reconnect, int max_attempts, int backoff_ms)
        : UdsStreamClientBase(std::move(path), reconnect, max_attempts, backoff_ms) {}

    int do_connect() override {
        ++attempts;
        throw std::runtime_error("no server");
    }
};

TEST(UdsStreamClientBase, BoundedReconnectBudget) {
    CountingUds c("/nonexistent/test.sock", true, 3, 1);
    EXPECT_FALSE(c.ensure_connected());
    EXPECT_EQ(c.attempts, 3);  // exactly the budget
}

TEST(UdsStreamClientBase, NoReconnectMeansSingleAttempt) {
    CountingUds c("/nonexistent/test.sock", false, 5, 1);
    EXPECT_FALSE(c.ensure_connected());
    EXPECT_EQ(c.attempts, 1);  // first connect only, no retry budget
}

// do_connect() returning a real fd; on_disconnect() recorded into an external
// log so the destructor case can be inspected after delete.
//
// Follows the documented derived-class contract: a subclass whose teardown
// hook must run at destruction calls close_fd() in ITS destructor — the base
// destructor cannot virtually dispatch to the derived on_disconnect (same
// reason EncodedStream::Impl / FdMediaStream::Impl define ~Impl).
struct RecordingUds : detail::UdsStreamClientBase {
    std::vector<int>* disconnect_log;
    int next_fd = -1;

    RecordingUds(std::vector<int>* log)
        : UdsStreamClientBase("/nonexistent/test.sock", true, 1, 0),
          disconnect_log(log) {}

    ~RecordingUds() override { close_fd(); }

    int do_connect() override { return next_fd; }

    void on_disconnect(int fd) override { disconnect_log->push_back(fd); }
};

TEST(UdsStreamClientBase, ReusesConnectedFdWithoutNewAttempt) {
    int sv[2];
    ASSERT_EQ(::socketpair(AF_UNIX, SOCK_STREAM, 0, sv), 0);
    ::close(sv[1]);

    std::vector<int> log;
    RecordingUds c(&log);
    c.next_fd = sv[0];
    ASSERT_TRUE(c.ensure_connected());
    EXPECT_EQ(c.fd(), sv[0]);

    EXPECT_TRUE(c.ensure_connected());  // already connected: no re-handshake
    EXPECT_EQ(c.fd(), sv[0]);
    EXPECT_TRUE(log.empty());
}

TEST(UdsStreamClientBase, CloseFdRunsOnDisconnectThenClosesOnce) {
    int sv[2];
    ASSERT_EQ(::socketpair(AF_UNIX, SOCK_STREAM, 0, sv), 0);
    ::close(sv[1]);

    std::vector<int> log;
    RecordingUds c(&log);
    c.next_fd = sv[0];
    ASSERT_TRUE(c.ensure_connected());

    c.close_fd();
    ASSERT_EQ(log.size(), 1u);
    EXPECT_EQ(log[0], sv[0]);
    EXPECT_EQ(c.fd(), -1);

    c.close_fd();  // idempotent: hook must not run twice
    EXPECT_EQ(log.size(), 1u);
}

TEST(UdsStreamClientBase, DestructorTeardownReachesDerivedHook) {
    int sv[2];
    ASSERT_EQ(::socketpair(AF_UNIX, SOCK_STREAM, 0, sv), 0);
    ::close(sv[1]);

    std::vector<int> log;
    auto* c = new RecordingUds(&log);
    c->next_fd = sv[0];
    ASSERT_TRUE(c->ensure_connected());

    delete c;  // derived dtor's close_fd() dispatches to the derived hook
    ASSERT_EQ(log.size(), 1u);
    EXPECT_EQ(log[0], sv[0]);
}

}  // namespace
