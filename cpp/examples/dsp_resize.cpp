// dsp_resize.cpp — the DSP domain in one pass (plan: examples 增 3 个).
//
// Allocates daemon-side DSP buffers, writes an NV12 source, rescales it on
// the DSP, reads the result back, and encodes it to JPEG straight from a
// daemon-side buffer id — the zero-read-back chain. Prints which path each
// job took (last_used_hw) so CPU fallback is visible, never silent.
//
//   dsp_resize                    # on-device; writes /tmp/dsp_resize.jpg
#include <cstdio>
#include <exception>
#include <fstream>
#include <string>
#include <variant>

#include <opencv2/core.hpp>

#include "neoruntime_ipc_sdk/dsp.hpp"

namespace {

// A deterministic NV12 test frame: horizontal Y gradient, neutral chroma.
cv::Mat gradient_nv12(int width, int height) {
    cv::Mat nv12(height * 3 / 2, width, CV_8UC1);
    for (int y = 0; y < height; ++y)
        for (int x = 0; x < width; ++x)
            nv12.at<uchar>(y, x) = static_cast<uchar>((x * 255) / (width - 1));
    nv12.rowRange(height, height * 3 / 2).setTo(128);  // U/V planes
    return nv12;
}

}  // namespace

int main() {
    using namespace neoruntime_ipc_sdk;
    std::setvbuf(stdout, nullptr, _IONBF, 0);

    DspClient dsp;
    try {
        dsp.connect();
        std::printf("connected (DSP UDS + camera-control gRPC)\n");
    } catch (const std::exception& e) {
        std::fprintf(stderr, "connect failed: %s\n", e.what());
        return 1;
    }

    // ---- 1. pool round-trip: pixels land in a daemon dma-buf once ------
    try {
        auto pool = dsp.alloc_buffers(640, 360, "nv12", 2);
        pool->write(0, gradient_nv12(640, 360));
        const cv::Mat back = pool->read(0);
        std::printf("[1] alloc_buffers(640x360 nv12 x2): write+read back %dx%d\n",
                    back.cols, back.rows);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[1] alloc_buffers failed: %s\n", e.what());
        dsp.close();
        return 1;
    }

    // ---- 2. hardware resize: Mat in, Mat out ----------------------------
    try {
        auto out = dsp.resize_hw(gradient_nv12(640, 480), 320, 240, "nv12");
        if (const cv::Mat* m = std::get_if<cv::Mat>(&out)) {
            const bool hw = dsp.last_used_hw().value_or(false);
            std::printf("[2] resize_hw 640x480 -> %dx%d (%s path)\n", m->cols, m->rows,
                        hw ? "hardware" : "CPU fallback");
        } else {
            std::fprintf(stderr, "[2] resize_hw returned a non-sync result\n");
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[2] resize_hw failed: %s\n", e.what());
    }

    // ---- 3. zero-read-back chain: resize to a ref, encode by buffer id --
    try {
        DspJobParams ref_out;
        ref_out.out_ref = true;  // resize without reading pixels back
        auto out = dsp.resize_hw(gradient_nv12(640, 480), 320, 240, "nv12", ref_out);
        if (DspBufferRef* ref = std::get_if<DspBufferRef>(&out)) {
            // encode straight from the daemon-side buffer — the JPEG bytes
            // are the only thing that crosses back
            const std::vector<std::uint8_t> jpeg =
                dsp.encode_jpeg_hw(nullptr, 85, "nv12", 5.0, nullptr, false,
                                   ref->buffer_id());
            const std::string path = "/tmp/dsp_resize.jpg";
            std::ofstream(path, std::ios::binary)
                .write(reinterpret_cast<const char*>(jpeg.data()),
                       static_cast<std::streamsize>(jpeg.size()));
            std::printf("[3] out_ref resize + encode by buffer_id: %zu bytes -> %s\n",
                        jpeg.size(), path.c_str());
        } else {
            std::fprintf(stderr, "[3] out_ref resize did not return a DspBufferRef\n");
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[3] zero-read-back chain failed: %s\n", e.what());
    }

    dsp.close();
    std::printf("done\n");
    return 0;
}
