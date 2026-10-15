// phase5_smoke.cpp — Phase 5 on-device smoke (plan 冒烟③).
//
// Exercises the DSP + injection surface against the live daemons on 93.72:
//   * DspClient::alloc_buffers round-trip (dma-buf pool write/read with
//     stride-preserved planes, distinct registry ids per slot)
//   * resize_hw (flat-field luma through bilinear), convert_hw (rgb24->nv12
//     BT.601 luma check), blend_hw (RGBA overlay onto NV12 base, in/out
//     luma separation) — each reporting the hardware-vs-CPU path taken
//   * encode_jpeg_hw from a Mat AND the zero-read-back DspBufferRef chain
//     (resize_hw out_ref -> encode by buffer_id)
//   * FramePublisher end-to-end on "sub": REPLACE nv12 publish burst,
//     publish_rgb staging chain, publish_stream + EOS, injection_status
//     lease reporting, and the argb overlay import ring — or a SKIP when
//     the deployed daemon still gates injection by configuration
//     (phase3's error_code=-1 case).
//
// Exit 0 only if no section FAILed (SKIP is acceptable and explained).
#include <opencv2/core.hpp>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "neoruntime_ipc_sdk/camera.hpp"
#include "neoruntime_ipc_sdk/dsp.hpp"
#include "neoruntime_ipc_sdk/injection.hpp"

namespace {

int failures = 0;
int skips = 0;

void pass(const std::string& what) { std::printf("  [PASS] %s\n", what.c_str()); }
void fail(const char* what, const std::string& err) {
    std::fprintf(stderr, "  [FAIL] %s: %s\n", what, err.c_str());
    ++failures;
}
void skip(const char* what, const std::string& why) {
    std::printf("  [SKIP] %s: %s\n", what, why.c_str());
    ++skips;
}

// Neutral or flat NV12 plane: full H*3/2 rows, W cols.
cv::Mat flat_nv12(int luma_h, int w, int luma) {
    cv::Mat m(luma_h * 3 / 2, w, CV_8UC1, cv::Scalar(128));
    m.rowRange(0, luma_h).setTo(cv::Scalar(luma));
    return m;
}

bool luma_at(const cv::Mat& nv12, int y, int x, int expect, int tol, const char* tag) {
    const int v = nv12.at<uint8_t>(y, x);
    if (v < expect - tol || v > expect + tol) {
        fail(tag, "luma " + std::to_string(v) + " at (" + std::to_string(y) + "," +
                      std::to_string(x) + ") not in [" + std::to_string(expect - tol) + "," +
                      std::to_string(expect + tol) + "]");
        return false;
    }
    return true;
}

bool is_jpeg(const std::vector<std::uint8_t>& b) {
    return b.size() > 2 && b[0] == 0xFF && b[1] == 0xD8;
}

const char* hw_text(const neoruntime_ipc_sdk::DspClient& dsp) {
    const auto hw = dsp.last_used_hw();
    if (!hw) return "no-job";
    return *hw ? "hardware" : "cpu-fallback";
}

}  // namespace

int main() {
    using namespace neoruntime_ipc_sdk;
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    std::printf("=== Phase 5 smoke (DSP jobs + FramePublisher end-to-end) ===\n");

    DspClient dsp;
    CameraClient cam;

    // ---- 1. alloc round-trip -------------------------------------------------
    std::printf("[1] alloc_buffers pool round-trip\n");
    try {
        auto pool = dsp.alloc_buffers(320, 240, "nv12", 2);
        cv::Mat src = flat_nv12(240, 320, 100);
        for (int r = 0; r < 240; ++r) src.row(r).setTo(cv::Scalar((r * 255) / 240));
        pool->write(0, src);
        const cv::Mat back = pool->read(0);
        bool ok = back.rows == 360 && back.cols == 320;
        for (int r = 0; ok && r < 240; r += 40)
            ok = back.at<uint8_t>(r, 7) == src.at<uint8_t>(r, 7);
        ok = ok && pool->buffer_id(0) != pool->buffer_id(1);
        ok = ok && pool->strides[0] >= 320 && pool->strides[1] >= 160;
        if (ok)
            pass("write->read round-trip, per-slot ids, plane strides");
        else
            fail("pool round-trip", "geometry/readback mismatch");
    } catch (const std::exception& e) {
        fail("alloc_buffers", e.what());
    }

    // ---- 2. resize_hw ----------------------------------------------------------
    std::printf("[2] resize_hw 640x480 -> 320x240\n");
    try {
        DspJobParams p;
        auto out = dsp.resize_hw(flat_nv12(480, 640, 77), 320, 240, "nv12", p);
        const cv::Mat* m = std::get_if<cv::Mat>(&out);
        if (!m) {
            fail("resize_hw", "default params did not return a Mat");
        } else if (m->rows != 360 || m->cols != 320) {
            fail("resize_hw", "output " + std::to_string(m->cols) + "x" +
                                  std::to_string(m->rows / 3 * 2) + " != 320x240");
        } else if (luma_at(*m, 120, 160, 77, 3, "resize_hw")) {
            pass(std::string("flat-field luma preserved (") + hw_text(dsp) + ")");
        }
    } catch (const std::exception& e) {
        fail("resize_hw", e.what());
    }

    // ---- 3. convert_hw ----------------------------------------------------------
    std::printf("[3] convert_hw rgb24 -> nv12\n");
    try {
        cv::Mat rgb(480, 640, CV_8UC3, cv::Scalar(60, 120, 30));  // BT.601 luma ~92
        auto out = dsp.convert_hw(rgb, "nv12");
        const cv::Mat* m = std::get_if<cv::Mat>(&out);
        if (!m) {
            fail("convert_hw", "default params did not return a Mat");
        } else if (m->rows != 720 || m->cols != 640) {
            fail("convert_hw", "output geometry mismatch");
        } else if (luma_at(*m, 240, 320, 92, 6, "convert_hw")) {
            pass(std::string("rgb(60,120,30) -> luma ~92 (") + hw_text(dsp) + ")");
        }
    } catch (const std::exception& e) {
        fail("convert_hw", e.what());
    }

    // ---- 4. blend_hw -------------------------------------------------------------
    std::printf("[4] blend_hw RGBA overlay onto NV12 base\n");
    try {
        DspOverlay ov;
        ov.rgba = cv::Mat(64, 64, CV_8UC4, cv::Scalar(0, 255, 0, 255));  // opaque green
        ov.x = 16;
        ov.y = 16;
        auto out = dsp.blend_hw(flat_nv12(240, 320, 40), {ov});
        const cv::Mat* m = std::get_if<cv::Mat>(&out);
        if (!m) {
            fail("blend_hw", "default params did not return a Mat");
        } else {
            // Green luma ~150 inside, base 40 outside.
            const bool in_ok = luma_at(*m, 48, 48, 150, 12, "blend_hw(in)");
            const bool out_ok = luma_at(*m, 4, 4, 40, 4, "blend_hw(out)");
            if (in_ok && out_ok) pass(std::string("overlay composited (") + hw_text(dsp) + ")");
        }
    } catch (const std::exception& e) {
        fail("blend_hw", e.what());
    }

    // ---- 5. encode_jpeg_hw (Mat source) ---------------------------------------------
    std::printf("[5] encode_jpeg_hw from Mat + DspBufferRef chain\n");
    try {
        cv::Mat blended = flat_nv12(240, 320, 80);
        for (int r = 0; r < 240; ++r) blended.row(r).setTo(cv::Scalar((r * 255) / 240));
        const DspJobSource blended_src = blended;
        const auto jpg = dsp.encode_jpeg_hw(&blended_src, 85, "nv12");
        if (is_jpeg(jpg) && jpg.size() > 1500)
            pass("jpeg from Mat (" + std::to_string(jpg.size()) + " bytes, " + hw_text(dsp) +
                 ")");
        else
            fail("encode_jpeg_hw(Mat)", "bytes=" + std::to_string(jpg.size()));

        // Zero-read-back: resize lands in a daemon buffer, encode reads the id.
        DspJobParams rp;
        rp.out_ref = true;
        auto refout = dsp.resize_hw(flat_nv12(480, 640, 90), 320, 240, "nv12", rp);
        if (auto* ref = std::get_if<DspBufferRef>(&refout)) {
            const auto jpg2 = dsp.encode_jpeg_hw(nullptr, 85, "nv12", 5.0, nullptr, true,
                                                 ref->buffer_id());
            if (is_jpeg(jpg2) && jpg2.size() > 1500)
                pass("jpeg from DspBufferRef id " + std::to_string(ref->buffer_id()) + " (" +
                     std::to_string(jpg2.size()) + " bytes)");
            else
                fail("encode_jpeg_hw(ref)", "bytes=" + std::to_string(jpg2.size()));
        } else {
            fail("resize_hw(out_ref)", "no DspBufferRef alternative");
        }
    } catch (const std::exception& e) {
        fail("encode_jpeg_hw", e.what());
    }

    // ---- 6/7. FramePublisher end-to-end ----------------------------------------------
    std::printf("[6] FramePublisher REPLACE publish burst on 'sub'\n");
    bool injection_live = true;
    FramePublisher::Config cfg;
    cfg.session_id = "cpp-p5-smoke";
    std::unique_ptr<FramePublisher> pub;
    int injected = 0;
    try {
        pub = std::make_unique<FramePublisher>(cam, dsp, "sub", 4, cfg);
        std::printf("  geometry %dx%d, lease_mode=%d (daemon %s the in-flight set)\n",
                    pub->width(), pub->height(), pub->lease_mode() ? 1 : 0,
                    pub->lease_mode() ? "reports" : "does NOT report");
        for (int i = 0; i < 8; ++i) {
            const InjectionResult res =
                pub->publish(flat_nv12(pub->height(), pub->width(), 60 + i * 18));
            if (!res.success)
                throw std::runtime_error("publish " + std::to_string(i) + ": " + res.message);
            ++injected;
            std::this_thread::sleep_for(std::chrono::milliseconds(25));
        }
        pass("8 nv12 frames published");
    } catch (const std::runtime_error& e) {
        const std::string msg = e.what();
        if (msg.find("disabled") != std::string::npos ||
            msg.find("error_code=-1") != std::string::npos) {
            skip("FramePublisher", std::string("daemon gates injection (") + msg + ")");
            injection_live = false;
        } else {
            fail("FramePublisher publish", msg);
            injection_live = false;
        }
    } catch (const std::exception& e) {
        fail("FramePublisher ctor/publish", e.what());
        injection_live = false;
    }

    if (injection_live && pub) {
        std::printf("[7] publish_rgb + publish_stream + EOS + status\n");
        try {
            cv::Mat rgb(pub->height(), pub->width(), CV_8UC3, cv::Scalar(30, 200, 90));
            const InjectionResult r1 = pub->publish_rgb(rgb);
            if (r1.success)
                pass(std::string("publish_rgb staged+converted+pushed (dsp ") +
                     (dsp.last_used_hw().value_or(false) ? "hw" : "cpu") + ")");
            else
                fail("publish_rgb", r1.message);

            std::vector<cv::Mat> frames(3, flat_nv12(pub->height(), pub->width(), 110));
            const InjectionResult r2 = pub->publish_stream(frames, {1, 2, 3}, true);
            if (r2.success)
                pass("publish_stream 3 frames + EOS trailer");
            else
                fail("publish_stream", r2.message);

            // EOS drops the daemon session: poll up to 2s for active=false.
            InjectionStatus st = cam.injection_status();
            for (int i = 0; i < 20 && st.active; ++i) {
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
                st = cam.injection_status();
            }
            std::printf("  status: active=%d injected=%u dropped=%u queue=%u\n",
                        st.active ? 1 : 0, static_cast<unsigned>(st.frames_injected),
                        static_cast<unsigned>(st.frames_dropped),
                        static_cast<unsigned>(st.queue_depth));
            if (!st.active)
                pass("session closed by EOS, frames_injected=" +
                     std::to_string(st.frames_injected));
            else
                fail("injection_status", "session still active after EOS");
        } catch (const std::exception& e) {
            fail("publish_rgb/stream/EOS", e.what());
        }

        std::printf("[8] argb overlay import ring (PIP)\n");
        try {
            FramePublisher::Config ocfg;
            ocfg.mode = "overlay";
            ocfg.fmt = "argb";
            ocfg.session_id = "cpp-p5-pip";
            FramePublisher pip(cam, dsp, "main", 2, ocfg);
            std::printf("  pip inset %dx%d at (%d,%d)\n", pip.width(), pip.height(),
                        pip.dest().first, pip.dest().second);
            const cv::Mat badge(pip.height(), pip.width(), CV_8UC4,
                                cv::Scalar(0, 255, 0, 255));  // opaque green RGBA
            const InjectionResult r = pip.publish(badge);
            if (r.success) {
                pip.publish_eos();
                pass("argb PIP frame via shared import ring + EOS");
            } else {
                fail("argb PIP publish", r.message);
            }
        } catch (const std::exception& e) {
            fail("argb PIP", e.what());
        }
    }

    // Publisher destructor runs its best-effort EOS before pool release.
    pub.reset();

    std::printf("=== Phase 5 smoke: %s (failures=%d skips=%d) ===\n",
                failures == 0 ? "PASS" : "FAIL", failures, skips);
    return failures == 0 ? 0 : 1;
}
