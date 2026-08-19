// detail/grpc_channel.hpp — shared gRPC plumbing for the sync-stub clients.
//
// Centralizes three concerns every client repeats:
//   1. make_channel()      — build a grpc::Channel from an endpoint string.
//   2. check_grpc() / require_success() — turn transport/app-level failures
//      into a thrown RpcError, mirroring Python's `raise RuntimeError(...)`.
//   3. now_ns()             — wall-clock nanoseconds, used by EventClient.
//
// Not public (lives under src/). Included only by the .cpp client impls.
#pragma once

#include <chrono>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>

#include <grpcpp/grpcpp.h>

#include "endpoint.hpp"
#include "hailo_ipc_sdk/config.hpp"

namespace hailo_ipc_sdk::detail {

inline std::shared_ptr<grpc::Channel> make_channel(std::string_view endpoint) {
    return grpc::CreateChannel(grpc_endpoint(endpoint),
                               grpc::InsecureChannelCredentials());
}

// Thrown when an RPC fails (transport error OR daemon replied success=false).
class RpcError : public std::runtime_error {
public:
    RpcError(std::string method, std::string detail)
        : std::runtime_error(format(std::move(method), std::move(detail))) {}

private:
    static std::string format(std::string method, std::string detail) {
        return detail.empty() ? (method + " failed")
                              : (method + " failed: " + detail);
    }
};

// Raise if the gRPC transport itself failed.
inline void check_grpc(const grpc::Status& st, std::string_view method) {
    if (!st.ok()) {
        throw RpcError(std::string(method), st.error_message());
    }
}

// Raise if the daemon returned success=false in its Status message.
inline void require_success(bool success, std::string_view message,
                            std::string_view method) {
    if (!success) {
        throw RpcError(std::string(method), std::string(message));
    }
}

// Wall-clock nanoseconds since epoch (Python: int(time.time() * 1e9)).
inline std::uint64_t now_ns() {
    return static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::system_clock::now().time_since_epoch())
            .count());
}

}  // namespace hailo_ipc_sdk::detail
