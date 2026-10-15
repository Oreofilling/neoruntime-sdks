// video_processor.cpp — 1:1 port of python/examples/video_processor.py.
//
// Subscribes to a raw (zero-copy DMA-BUF) video stream, counts frames, saves a
// frame every N frames when in debug mode, and prints FPS statistics.
#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <string>
#include <sys/stat.h>
#include <thread>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "neoruntime_ipc_sdk/config.hpp"
#include "neoruntime_ipc_sdk/media.hpp"

namespace {
std::atomic<bool> g_running{true};

void on_signal(int sig) {
    (void)sig;
    std::printf("\nShutting down...\n");
    g_running.store(false);
}

double now_s() {
    return std::chrono::duration<double>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}
}  // namespace

int main() {
    std::signal(SIGINT, on_signal);
    std::signal(SIGTERM, on_signal);

    const std::string app_id = neoruntime_ipc_sdk::Config::get_app_id();
    const bool debug = neoruntime_ipc_sdk::Config::is_debug();

    neoruntime_ipc_sdk::FdMediaClient media;

    const int save_interval = 100;
    const std::string save_dir = "/app/data/frames";
    if (debug) ::mkdir(save_dir.c_str(), 0755);  // best-effort; ignore EEXIST

    std::printf("[%s] Video Processor App initialized\n", app_id.c_str());
    std::printf("[%s] SHM Base: %s\n", app_id.c_str(),
                neoruntime_ipc_sdk::Config::get_shm_base_path().c_str());

    auto streams = media.list_streams();
    std::printf("[%s] Available streams: %zu\n", app_id.c_str(), streams.size());
    for (const auto& s : streams) std::printf("  - %s\n", s.c_str());

    while (streams.empty() && g_running.load()) {
        std::printf("[%s] No streams available, waiting...\n", app_id.c_str());
        std::this_thread::sleep_for(std::chrono::seconds(1));
        streams = media.list_streams();
    }
    const std::string stream_id = streams.empty() ? "cam0_main" : streams[0];
    std::printf("[%s] Subscribing to stream: %s\n", app_id.c_str(), stream_id.c_str());

    std::uint64_t frame_count = 0;
    double start_time = now_s();
    double last_fps_time = start_time;
    std::uint64_t last_fps_count = 0;

    try {
        auto stream = media.subscribe(stream_id);
        while (g_running.load()) {
            auto frame = stream.next();
            if (!frame) break;

            ++frame_count;
            if (debug && frame_count % save_interval == 0) {
                char filename[256];
                std::snprintf(filename, sizeof(filename), "%s/frame_%08llu.jpg",
                              save_dir.c_str(), static_cast<unsigned long long>(frame->sequence));
                try {
                    cv::Mat rgb = frame->to_rgb();
                    cv::Mat bgr;
                    cv::cvtColor(rgb, bgr, cv::COLOR_RGB2BGR);
                    if (cv::imwrite(filename, bgr)) {
                        std::printf("[%s] Saved: %s\n", app_id.c_str(), filename);
                    }
                } catch (const std::exception& e) {
                    std::printf("[%s] Failed to save frame: %s\n", app_id.c_str(), e.what());
                }
            }

            double t = now_s();
            double elapsed = t - last_fps_time;
            if (elapsed >= 5.0) {
                double fps = (frame_count - last_fps_count) / elapsed;
                double avg_fps = frame_count / (t - start_time);
                std::printf("[%s] FPS: %.1f (avg: %.1f) | Frame: %llu | Size: %dx%d\n",
                            app_id.c_str(), fps, avg_fps,
                            static_cast<unsigned long long>(frame->sequence),
                            frame->width, frame->height);
                last_fps_time = t;
                last_fps_count = frame_count;
            }
        }
    } catch (const std::exception& e) {
        std::printf("[%s] Error: %s\n", app_id.c_str(), e.what());
    }

    media.close();
    double total_time = now_s() - start_time;
    if (frame_count > 0) {
        double avg_fps = frame_count / total_time;
        std::printf("[%s] Total frames: %llu\n", app_id.c_str(),
                    static_cast<unsigned long long>(frame_count));
        std::printf("[%s] Total time: %.1fs\n", app_id.c_str(), total_time);
        std::printf("[%s] Average FPS: %.2f\n", app_id.c_str(), avg_fps);
    }
    std::printf("[%s] Cleanup complete\n", app_id.c_str());
    return EXIT_SUCCESS;
}
