// frame_injection.cpp — replace a camera stream with generated frames
// (plan: examples 增 3 个).
//
// A FramePublisher on the "sub" stream pushes a moving color pattern: raw
// NV12 publish() into the pool for the zero-DSP path, and publish_rgb()
// for the staging convert chain (hardware convert with CPU fallback,
// visible via DspClient::last_used_hw). Stops cleanly on Ctrl-C — the
// destructor sends EOS and the pure ISP path restores at the next IDR.
//
//   frame_injection                # on-device, Ctrl-C to stop
#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <memory>
#include <optional>
#include <string>
#include <thread>

#include <opencv2/core.hpp>

#include "neoruntime_ipc_sdk/camera.hpp"
#include "neoruntime_ipc_sdk/dsp.hpp"
#include "neoruntime_ipc_sdk/injection.hpp"

namespace {
std::atomic<bool> g_running{true};

void on_signal(int sig) {
    std::printf("\n[signal] received %d, shutting down...\n", sig);
    g_running.store(false);
}

// An (h*3/2, w) NV12 Mat filled with one luma value and neutral chroma.
cv::Mat flat_nv12(int width, int height, uchar luma) {
    cv::Mat nv12(height * 3 / 2, width, CV_8UC1);
    nv12.rowRange(0, height).setTo(luma);
    nv12.rowRange(height, height * 3 / 2).setTo(128);
    return nv12;
}

// An RGB test pattern whose color walks with the frame counter.
cv::Mat pattern_rgb(int width, int height, int step) {
    cv::Mat rgb(height, width, CV_8UC3);
    const uchar r = static_cast<uchar>((step * 37) % 256);
    const uchar g = static_cast<uchar>((step * 91) % 256);
    const uchar b = static_cast<uchar>((step * 151) % 256);
    cv::Vec3b corner_colors[4] = {
        cv::Vec3b(r, 0, 0), cv::Vec3b(0, g, 0), cv::Vec3b(0, 0, b), cv::Vec3b(r, g, b)};
    for (int y = 0; y < height; ++y)
        for (int x = 0; x < width; ++x) {
            const int quadrant = (y < height / 2 ? 0 : 2) + (x < width / 2 ? 0 : 1);
            rgb.at<cv::Vec3b>(y, x) = corner_colors[quadrant];
        }
    return rgb;
}

}  // namespace

int main() {
    using namespace neoruntime_ipc_sdk;
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    std::signal(SIGINT, on_signal);
    std::signal(SIGTERM, on_signal);

    CameraClient camera;
    DspClient dsp;
    try {
        camera.connect();
        dsp.connect();
    } catch (const std::exception& e) {
        std::fprintf(stderr, "connect failed: %s\n", e.what());
        return 1;
    }

    FramePublisher::Config cfg;  // REPLACE the sub stream, session-tagged
    cfg.session_id = "frame-injection-example";
    std::unique_ptr<FramePublisher> pub;
    try {
        pub = std::make_unique<FramePublisher>(camera, dsp, "sub", 4, cfg);
        std::printf("publishing %dx%d onto 'sub' (mode=%s lease=%d) — Ctrl-C to stop\n",
                    pub->width(), pub->height(), pub->mode().c_str(),
                    pub->lease_mode() ? 1 : 0);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FramePublisher construction failed: %s\n", e.what());
        camera.close();
        dsp.close();
        return 1;
    }

    // ~15 fps of alternating publish paths until stopped
    int step = 0;
    std::uint64_t pushed = 0;
    try {
        while (g_running.load()) {
            InjectionResult result = (step % 4 == 3)
                ? pub->publish_rgb(pattern_rgb(pub->width(), pub->height(), step))
                : pub->publish(flat_nv12(pub->width(), pub->height(),
                                         static_cast<uchar>(60 + (step * 13) % 160)));
            if (!result.success) {
                std::fprintf(stderr, "push rejected (error_code=%d): %s\n",
                             result.error_code, result.message.c_str());
                break;
            }
            ++pushed;
            std::this_thread::sleep_for(std::chrono::milliseconds(66));
            ++step;
        }
    } catch (const std::exception& e) {
        // e.g. "frame injection disabled by configuration" on gated daemons
        std::fprintf(stderr, "publish failed after %llu frame(s): %s\n",
                     static_cast<unsigned long long>(pushed), e.what());
    }

    const std::optional<bool> hw = dsp.last_used_hw();
    if (hw.has_value())
        std::printf("publish_rgb convert path: %s\n", *hw ? "hardware" : "CPU fallback");
    // destructor: best-effort EOS, then close — the ISP stream restores
    pub.reset();
    std::printf("pushed %llu frame(s); stream restored\n",
                static_cast<unsigned long long>(pushed));
    camera.close();
    dsp.close();
    return 0;
}
