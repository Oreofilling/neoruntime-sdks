// detail/transport.hpp — C++ counterpart of the Python SDK's _transport.py:
// the shared connection plumbing every daemon/stream client builds on.
//
//   * GrpcClientBase<StubT> — lazy channel + typed stub cache with the
//     64 MiB receive cap (kMaxGrpcMessageLength). Client Impl structs
//     privately inherit it, so the public headers stay proto-free.
//   * UdsStreamClientBase — raw AF_UNIX socket lifecycle with an optional
//     bounded reconnect budget, shared by the pull-style stream objects
//     (EncodedStream, FdMediaStream).
//
// Not public (lives under src/). Included only by the .cpp client impls.
// Like their Python counterparts, the clients built on these bases are not
// internally synchronized: callers serialize connect/RPC/close per client
// (background on_* threads capture their own channel/fd handles).
#pragma once

#include <chrono>
#include <memory>
#include <string>
#include <thread>
#include <utility>

#include <grpcpp/grpcpp.h>

#include "endpoint.hpp"
#include "raw_socket.hpp"

namespace neoruntime_ipc_sdk::detail {

// Python _transport.py MAX_GRPC_MESSAGE_LENGTH. Daemon responses can exceed
// gRPC's 4 MiB default (large batch results, verbose stats), which would fail
// the call client-side with RESOURCE_EXHAUSTED before the daemon is even
// consulted. Every channel this SDK creates applies this cap.
inline constexpr int kMaxGrpcMessageLength = 64 * 1024 * 1024;

// Channel options applied to every channel this SDK creates. Python adds the
// same option in GrpcClient.channel_options (and OverlayClient layers its
// poll strategy on top).
inline grpc::ChannelArguments default_channel_args() {
    grpc::ChannelArguments args;
    args.SetMaxReceiveMessageSize(kMaxGrpcMessageLength);
    return args;
}

// Python GrpcClient: lazy channel + stub cache per client. Template parameter
// is the generated service type (e.g. pb::EventBus); the stub is created via
// ServiceT::NewStub, mirroring the Python _stub_factory pattern.
template <typename ServiceT>
class GrpcClientBase {
public:
    using StubT = typename ServiceT::Stub;

    explicit GrpcClientBase(std::string endpoint)
        : endpoint_(std::move(endpoint)) {}

    // Lazily create the channel + stub on first use (Python _connect).
    void connect() {
        if (stub_) return;
        channel_ = grpc::CreateCustomChannel(grpc_endpoint(endpoint_),
                                             grpc::InsecureChannelCredentials(),
                                             default_channel_args());
        stub_ = ServiceT::NewStub(channel_);
    }

    void close() noexcept {
        stub_.reset();
        channel_.reset();
    }

    bool connected() const noexcept { return stub_ != nullptr; }

    // Ensure-connected stub access; every RPC call site uses this.
    StubT& stub() {
        connect();
        return *stub_;
    }

    // Shared channel handle for capturing into background threads / async
    // tasks that must stay alive even if the client is closed first
    // (Python: on_event threads and infer_async closures).
    std::shared_ptr<grpc::Channel> channel() {
        connect();
        return channel_;
    }

    const std::string& endpoint() const noexcept { return endpoint_; }

private:
    std::string endpoint_;
    std::shared_ptr<grpc::Channel> channel_;
    std::unique_ptr<StubT> stub_;
};

// Raw AF_UNIX lifecycle for the pull-style stream objects. Subclasses define
// the connect handshake (plain connect(2) vs. the FdPublisher SUBSCRIBE
// negotiation) via do_connect(); on_disconnect() runs best-effort teardown
// (e.g. send UNSUBSCRIBE) before the fd is closed.
//
// Reconnect semantics are bounded (Python generators reconnect forever; C++
// pull streams stop after the budget so a `while (next())` loop can end) —
// on_frame background threads loop forever instead, matching Python.
class UdsStreamClientBase {
public:
    UdsStreamClientBase(std::string socket_path, bool reconnect_enabled,
                        int max_attempts, int backoff_base_ms)
        : socket_path_(std::move(socket_path)),
          reconnect_enabled_(reconnect_enabled),
          max_attempts_(max_attempts > 0 ? max_attempts : 1),
          backoff_base_ms_(backoff_base_ms) {}

    virtual ~UdsStreamClientBase() { close_fd(); }

    UdsStreamClientBase(const UdsStreamClientBase&) = delete;
    UdsStreamClientBase& operator=(const UdsStreamClientBase&) = delete;

    // (Re)connect if needed. Returns false once the budget is spent (or the
    // single first connect failed when reconnecting is disabled). Backoff is
    // linear: backoff_base_ms * attempt.
    bool ensure_connected() {
        if (fd_ >= 0) return true;
        const int attempts = reconnect_enabled_ ? max_attempts_ : 1;
        for (int i = 0; i < attempts; ++i) {
            try {
                fd_ = do_connect();
                if (fd_ >= 0) return true;
            } catch (...) {
            }
            if (i + 1 < attempts) {
                std::this_thread::sleep_for(
                    std::chrono::milliseconds(backoff_base_ms_ * (i + 1)));
            }
        }
        return false;
    }

    // Best-effort teardown hook, then close. Safe to call repeatedly.
    void close_fd() noexcept {
        if (fd_ >= 0) {
            on_disconnect(fd_);
            ::close(fd_);
            fd_ = -1;
        }
    }

    bool reconnect_enabled() const noexcept { return reconnect_enabled_; }
    int fd() const noexcept { return fd_; }

protected:
    // Subclass connect handshake. Returns the connected fd; throws on failure.
    virtual int do_connect() = 0;
    // Teardown hook run before ::close(fd) (best-effort, must not throw out).
    // Runs when close_fd() is called on a live derived object. NOTE: the base
    // destructor cannot dispatch here (C++ resolves virtuals to the base during
    // ~UdsStreamClientBase), so subclasses whose hook must run at destruction
    // call close_fd() from their own destructor (see EncodedStream::Impl).
    virtual void on_disconnect(int fd) { (void)fd; }

    std::string socket_path_;
    bool reconnect_enabled_;
    int max_attempts_;
    int backoff_base_ms_;
    int fd_ = -1;
};

}  // namespace neoruntime_ipc_sdk::detail
