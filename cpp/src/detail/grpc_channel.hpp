// detail/grpc_channel.hpp — RPC error mapping for the sync-stub clients.
//
// Channel/stub lifecycle used to live here as make_channel(); it moved to
// transport.hpp (GrpcClientBase, with the 64 MiB receive cap — see
// kMaxGrpcMessageLength there). What remains are the concerns every RPC call
// site shares:
//   1. check_grpc() / require_success() — turn transport/app-level failures
//      into a thrown RpcError, mirroring Python's `raise RuntimeError(...)`.
//   2. now_ns() — wall-clock nanoseconds, used by EventClient.
//
// Not public (lives under src/). Included only by the .cpp client impls.
#pragma once

#include <chrono>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>

#include <grpcpp/grpcpp.h>

namespace neoruntime_ipc_sdk::detail {

// Thrown when an RPC fails (transport error OR daemon replied success=false).
class RpcError : public std::runtime_error {
public:
    // Most error messages follow "X failed: detail"; (method, detail) builds
    // that. The single-argument form is for messages with a different shape
    // (e.g. "StartOneShotAf rejected: lens busy") that Python raises verbatim.
    RpcError(std::string method, std::string detail)
        : std::runtime_error(format(std::move(method), std::move(detail))) {}
    explicit RpcError(std::string message) : std::runtime_error(std::move(message)) {}

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

}  // namespace neoruntime_ipc_sdk::detail
