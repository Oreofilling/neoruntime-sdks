// media_smoke.cpp — raw-UDS media-path smoke (NOT gRPC) for the NE503 C++ SDK.
//
// Exercises both non-gRPC transports against the live camera, independently and
// safely (timeouts return std::nullopt rather than throwing, so an idle camera
// is reported, not crashed):
//   * EncodedStreamClient — length-prefixed H.264/H.265 over UDS, one socket
//                           per stream (/run/aipc/encoded/{main,sub,third}.sock).
//   * FdMediaClient       — zero-copy DMA-BUF video via SCM_RIGHTS fd-passing
//                           against /run/aipc/camera.sock (stream "cam0_main").
//
// Run on-device (the fd-passing path cannot traverse ssh socket-forwarding).
//
//   media_smoke
//
// Exit code: 0 if at least one frame arrived on each path that the camera serves.

#include <cstdio>
#include <exception>
#include <string>
#include <vector>

#include "hailo_ipc_sdk/config.hpp"
#include "hailo_ipc_sdk/media.hpp"

int main() {
    using namespace hailo_ipc_sdk;

    std::printf("=== NE503 C++ SDK media-path smoke ===\n");
    std::printf("encoded dir : %s\n", Config::get_encoded_socket_dir().c_str());
    int failures = 0;

    // ---------------------------------------------------------------------
    // Encoded path: each stream has its own socket (the socket IS the stream).
    // ---------------------------------------------------------------------
    const char* enc_sockets[] = {
        "/run/aipc/encoded/main.sock",
        "/run/aipc/encoded/sub.sock",
        "/run/aipc/encoded/third.sock",
    };
    bool encoded_ok = false;
    for (const char* sock : enc_sockets) {
        try {
            EncodedStreamClient client(sock);
            std::printf("\n[Encoded] %s : reading 1 frame (5s)...\n", sock);
            auto frame = client.get_frame(5000);
            if (frame) {
                std::printf("[Encoded] OK  %s %dx%d %-3s payload=%zu bytes\n",
                            frame->codec_name().c_str(), frame->width, frame->height,
                            frame->is_keyframe() ? "KEY" : "P", frame->data.size());
                encoded_ok = true;
                break;  // one working encoded stream is enough
            }
            std::printf("[Encoded] %s : no frame (timeout/EOF — stream idle?)\n", sock);
        } catch (const std::exception& e) {
            std::fprintf(stderr, "[Encoded] %s FAILED: %s\n", sock, e.what());
        }
    }
    if (!encoded_ok) ++failures;

    // ---------------------------------------------------------------------
    // FD path: zero-copy DMA-BUF frames via SCM_RIGHTS on camera.sock.
    // ---------------------------------------------------------------------
    bool fd_ok = false;
    try {
        FdMediaClient client;
        const auto streams = client.list_streams();
        std::printf("\n[FdMedia] camera.sock connected; candidate stream IDs:");
        for (const auto& s : streams) std::printf(" %s", s.c_str());
        std::printf("\n");

        // Try the device's primary decoded stream first, then any listed IDs.
        std::vector<std::string> try_ids = {"cam0_main"};
        for (const auto& s : streams) try_ids.push_back(s);

        for (const auto& sid : try_ids) {
            try {
                std::printf("[FdMedia] get_frame(\"%s\", 5s)...\n", sid.c_str());
                auto frame = client.get_frame(sid, 5000);
                if (frame) {
                    std::printf("[FdMedia] OK  seq=%llu %dx%d %s\n",
                                static_cast<unsigned long long>(frame->sequence),
                                frame->width, frame->height, frame->format.c_str());
                    fd_ok = true;
                    break;
                }
                std::printf("[FdMedia] \"%s\" : no frame (timeout/EOF)\n", sid.c_str());
            } catch (const std::exception& e) {
                std::fprintf(stderr, "[FdMedia] \"%s\" FAILED: %s\n", sid.c_str(), e.what());
            }
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[FdMedia] FAILED: %s\n", e.what());
    }
    if (!fd_ok) ++failures;

    std::printf("\n=== media smoke done: %d failure(s) (encoded=%d fd=%d) ===\n",
                failures, encoded_ok ? 1 : 0, fd_ok ? 1 : 0);
    return failures ? 1 : 0;
}
