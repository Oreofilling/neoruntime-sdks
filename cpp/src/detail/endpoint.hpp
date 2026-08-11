// detail/endpoint.hpp — internal helpers shared by the gRPC and raw-UDS
// clients. Not part of the public API (lives under src/, not include/).
//
// Two concerns:
//   1. grpc_endpoint()  — resolve a Config endpoint string for a gRPC channel.
//   2. unix_socket_path() — strip the `unix:` URI scheme so the raw-UDS
//      clients (FdMediaClient, EncodedStreamClient, AudioStreamClient) can
//      pass a plain filesystem path to connect(AF_UNIX).
//
// Endpoint strings throughout the SDK keep the `unix:///run/aipc/<svc>.sock`
// triple-slash form used by the Python SDK; the gRPC C-core resolver accepts
// it unchanged, so grpc_endpoint() is effectively identity today. It exists
// as the single place to adjust if a future platform build needs a different
// scheme (e.g. `unix:` single-slash, or an abstract @namespace socket).
#pragma once

#include <string>
#include <string_view>

namespace hailo_ipc_sdk::detail {

// Returns the endpoint string verbatim for use as a grpc::CreateChannel target.
// Kept as a function (not a passthrough) so call sites say what they mean and
// so scheme normalization is centralized.
inline std::string grpc_endpoint(std::string_view endpoint) {
    return std::string(endpoint);
}

// Strips a leading "unix://" or "unix:" scheme and returns the plain socket
// filesystem path for connect(2) with AF_UNIX. Returns the input unchanged
// (as a std::string) when no scheme prefix is present.
inline std::string unix_socket_path(std::string_view endpoint) {
    constexpr std::string_view kUnixDouble = "unix://";
    constexpr std::string_view kUnixSingle = "unix:";
    if (endpoint.rfind(kUnixDouble, 0) == 0) {
        return std::string(endpoint.substr(kUnixDouble.size()));
    }
    if (endpoint.rfind(kUnixSingle, 0) == 0) {
        return std::string(endpoint.substr(kUnixSingle.size()));
    }
    return std::string(endpoint);
}

}  // namespace hailo_ipc_sdk::detail
