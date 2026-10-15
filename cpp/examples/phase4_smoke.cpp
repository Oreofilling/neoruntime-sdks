// phase4_smoke.cpp — Phase 4 on-device smoke (plan 冒烟②).
//
// Exercises the inference 0.8 surface against the live ai-runtime on 93.72:
//   * typed registration WITHOUT owner_id (global-table gotcha) + full-parse
//     get_model_info (layout / owner / transient / batch_size)
//   * geometry-matched NV12 single-image infer on the detection HEF +
//     the InferInput variant paths (cv::Mat, Frame materialization,
//     BufferRefInput wire/error path)
//   * infer_async (std::future) parity
//   * tight-loop infer CPU quantification gate (sync stub must block on a
//     futex, not spin — CPU/wall ratio < 30%)
//   * subscribe on "main": results + telemetry (dropped / latency EMA /
//     skew EMA), or the exact consecutive-failure RuntimeError when the
//     deployed daemon fails every frame (-2814 family) — both are PASS,
//     the message text is the assertion
//   * cancel() from another thread wakes a blocked next()
//   * windowed get_stats (client clamp) + skew aggregates
//
// Exit 0 only if all sections pass. The zero-copy FrameHandle→buffer_id
// infer lands with DspClient (Phase 5 smoke ③): Phase 4 materializes
// fd-frames through the handle.
#include <sys/resource.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <string>
#include <thread>
#include <vector>

#include "neoruntime_ipc_sdk/frame.hpp"
#include "neoruntime_ipc_sdk/inference.hpp"

namespace {

int failures = 0;

void pass(const char* what) { std::printf("  [PASS] %s\n", what); }
void fail(const char* what, const char* err) {
    std::fprintf(stderr, "  [FAIL] %s: %s\n", what, err);
    ++failures;
}

double now_ms() {
    return std::chrono::duration<double, std::milli>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

// CPU seconds (user+sys) spent by this process.
double cpu_seconds() {
    rusage ru{};
    getrusage(RUSAGE_SELF, &ru);
    return ru.ru_utime.tv_sec + ru.ru_utime.tv_usec / 1e6 +
           ru.ru_stime.tv_sec + ru.ru_stime.tv_usec / 1e6;
}

constexpr const char* kModelPath = "/data/aipc-data/models/hailo_yolov8n_384_640.hef";
constexpr const char* kModelId = "cpp-smoke-y8n";
constexpr const char* kStream = "main";

// Neutral NV12 plane for the given luma geometry: full H*3/2 rows, W cols,
// luma 128, chroma (128,128).
cv::Mat gray_nv12(int luma_h, int w) {
    return cv::Mat(luma_h * 3 / 2, w, CV_8UC1, cv::Scalar(128));
}

}  // namespace

int main() {
    using namespace neoruntime_ipc_sdk;

    // Unbuffered stdout so PASS/FAIL keeps its true order in ssh pipes.
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    std::printf("=== Phase 4 smoke (inference 0.8: InferInput / subscribe health / stats) ===\n");

    InferenceClient inf;
    bool registered = false;

    // ---- registration + model info (full parse) -----------------------------
    try {
        inf.connect();
        // Pre-clean: the daemon stages registered HEFs under runtime/<id>/
        // and may re-materialize an untyped registration after a previous
        // run unregistered — drop any leftover so the typed register below
        // cannot collide ("different configuration").
        for (const auto& m : inf.list_models())
            if (m.model_id == kModelId) {
                inf.unregister_model(kModelId);
                std::printf("[model] pre-cleaned leftover '%s'\n", kModelId);
            }
        const std::string id = inf.register_model(kModelPath, kModelId,
                                                  /*owner_id=*/"",
                                                  /*model_type=*/"detection");
        registered = true;
        std::printf("[model] registered as %s (typed detection, no owner_id)\n", id.c_str());
        pass("register_model(typed, no owner_id)");

        const auto info = inf.get_model_info(kModelId);
        if (!info) throw std::runtime_error("get_model_info returned nullopt");
        std::printf(
            "[model] %s v%s owner='%s' transient=%d batch=%u inputs=%zu outputs=%zu\n",
            info->model_id.c_str(), info->version.c_str(), info->owner_id.c_str(),
            info->transient ? 1 : 0, info->batch_size, info->inputs.size(),
            info->outputs.size());
        for (std::size_t i = 0; i < info->inputs.size() && i < 2; ++i) {
            std::string shape = "[";
            for (int d : info->inputs[i].shape) shape += std::to_string(d) + ",";
            shape += "]";
            std::printf("[model]   in[%zu] %s %s layout='%s'\n", i, shape.c_str(),
                        info->inputs[i].name.c_str(), info->inputs[i].layout.c_str());
        }
        pass("get_model_info() full parse (layout/owner/transient/batch)");
    } catch (const std::exception& e) {
        fail("model registration/info", e.what());
    }

    // ---- geometry-matched NV12 infer (cv::Mat) + variant paths ---------------
    cv::Mat nv12;
    try {
        const auto info = inf.get_model_info(kModelId);
        if (!info || info->inputs.empty())
            throw std::runtime_error("no input spec to match geometry");
        const auto& s = info->inputs[0].shape;
        int h = 384, w = 640;  // fallback: the HEF's nominal geometry
        if (s.size() == 4) {
            // NV12 models report NHWC [1,H,W,3] on this stack (verified
            // live: [1,384,640,3] layout='NV12'); NCHW [1,C,H,W] would put
            // the small channel dim at index 1.
            if (s[3] <= 4 && s[1] > 4) {
                h = s[1];
                w = s[2];
            } else {
                h = s[2];
                w = s[3];
            }
        } else if (s.size() == 2) {
            h = s[0];
            w = s[1];
        }
        if (w < 64 || h < 64) throw std::runtime_error("implausible input geometry");
        // NV12 full plane: if the spec already carries H*3/2 rows, use it
        // as-is; otherwise build 3/2 rows from the luma height.
        if (s.size() == 2 && h % 3 == 0 && (h / 3) % 2 == 0 && (h * 2) / 3 != h)
            nv12 = gray_nv12(h * 2 / 3, w);  // spec rows were H*3/2
        else
            nv12 = gray_nv12(h, w);

        const InferenceResult r = inf.infer(nv12, kModelId, 10000);
        std::printf("[infer] mat %dx%d: %zu objects infer_us=%llu hw_us=%llu status='%s'\n",
                    nv12.rows, nv12.cols, r.objects.size(),
                    (unsigned long long)r.infer_time_us,
                    (unsigned long long)r.hw_infer_time_us, r.status_message.c_str());
        pass("infer(geometry-matched NV12)");
    } catch (const std::exception& e) {
        fail("infer(NV12 mat)", e.what());
    }

    if (!nv12.empty()) {
        // Frame materialization path (no handle: image-carrying Frame).
        try {
            Frame f;
            f.format = "NV12";
            f.width = nv12.cols;
            f.height = nv12.rows * 2 / 3;
            f.image = nv12;
            const InferenceResult r = inf.infer(f, kModelId, 10000);
            std::printf("[infer] Frame path: %zu objects infer_us=%llu\n", r.objects.size(),
                        (unsigned long long)r.infer_time_us);
            pass("infer(InferInput::Frame)");
        } catch (const std::exception& e) {
            fail("infer(Frame)", e.what());
        }

        // infer_async parity.
        try {
            auto fut = inf.infer_async(nv12, kModelId, 10000);
            const InferenceResult r = fut.get();
            std::printf("[infer] async: %zu objects infer_us=%llu\n", r.objects.size(),
                        (unsigned long long)r.infer_time_us);
            pass("infer_async(future)");
        } catch (const std::exception& e) {
            fail("infer_async", e.what());
        }

        // BufferRefInput: a bogus registry id must be rejected by the daemon —
        // proving the buffer-id-only tensor shape reached it.
        try {
            const BufferRefInput bogus{0x7fffffff0000ull, 64, 64};
            (void)inf.infer(bogus, kModelId, 10000);
            fail("infer(BufferRefInput bogus)", "daemon accepted an unknown buffer id");
        } catch (const std::runtime_error& e) {
            std::printf("[infer] buffer-ref rejected as expected: %s\n", e.what());
            pass("infer(BufferRefInput) wire path + daemon rejection");
        }
    }

    // ---- tight-loop CPU quantification gate -----------------------------------
    if (!nv12.empty()) {
        try {
            constexpr int kCalls = 60;
            const double cpu0 = cpu_seconds();
            const double t0 = now_ms();
            std::uint64_t us = 0;
            for (int i = 0; i < kCalls; ++i) {
                const InferenceResult r = inf.infer(nv12, kModelId, 10000);
                us += r.infer_time_us;
            }
            const double wall = (now_ms() - t0) / 1000.0;
            const double cpu = cpu_seconds() - cpu0;
            const double ratio = wall > 0 ? cpu / wall : 99.0;
            std::printf("[cpu] %d infers: wall=%.2fs cpu=%.2fs ratio=%.0f%% "
                        "(daemon avg %.1f ms/call)\n",
                        kCalls, wall, cpu, ratio * 100.0,
                        static_cast<double>(us) / kCalls / 1000.0);
            // Sync stubs must block on a futex while the NPU works; a spinning
            // client would burn a whole core (ratio -> 100%). 30% leaves room
            // for serialization + memcpy yet catches a CQ spin.
            if (ratio < 0.30)
                pass("tight-loop CPU gate (<30%)");
            else
                fail("tight-loop CPU gate",
                     (std::string("cpu/wall ratio ") + std::to_string(ratio * 100.0) + "%")
                         .c_str());
        } catch (const std::exception& e) {
            fail("tight-loop CPU gate", e.what());
        }
    }

    // ---- subscribe: results + telemetry, or the exact failure raise -----------
    try {
        InferenceStream stream = inf.subscribe(kStream, kModelId, 5, "", false,
                                               /*max_consecutive_failures=*/10,
                                               /*queue_size=*/100);
        int got = 0;
        bool raised = false;
        const double deadline = now_ms() + 30000.0;
        try {
            for (int i = 0; i < 5 && now_ms() < deadline; ++i) {
                auto item = stream.next();
                if (!item) break;  // server closed the stream
                ++got;
                if (got == 1)
                    std::printf("[sub] first frame_seq=%llu skew_us=%lld status='%s'\n",
                                (unsigned long long)item->first,
                                (long long)item->second.skew_us,
                                item->second.status_message.c_str());
            }
        } catch (const std::runtime_error& e) {
            // The consecutive-failure raise is an exact-format contract; the
            // deployed daemon failing every frame (-2814) lands here.
            raised = true;
            const std::string msg = e.what();
            std::printf("[sub] raised: %s\n", msg.c_str());
            if (msg.find("Stream inference failed 10 consecutive times") == 0 &&
                msg.find("(stream='main', model='") != std::string::npos)
                pass("subscribe health raise (exact message)");
            else
                fail("subscribe health raise", "message format drifted");
        }
        if (!raised) {
            if (got > 0) {
                std::printf("[sub] %d result(s) dropped=%llu last_lat=%.1fms "
                            "avg_lat=%.1fms last_skew=%lldus avg_skew=%.0fus\n",
                            got, (unsigned long long)stream.dropped(),
                            stream.last_latency_ms(), stream.avg_latency_ms(),
                            (long long)stream.last_skew_us(), stream.avg_skew_us());
                pass("subscribe results + telemetry");
            } else {
                fail("subscribe", "stream ended with zero results and no raise");
            }
        }
    } catch (const std::exception& e) {
        fail("subscribe", e.what());
    }

    // ---- cancel() wakes a blocked next() ---------------------------------------
    try {
        InferenceStream stream = inf.subscribe(kStream, kModelId, 5);
        InferenceStream* sp = &stream;
        std::thread canceller([sp] {
            std::this_thread::sleep_for(std::chrono::seconds(2));
            sp->cancel();
        });
        const double t0 = now_ms();
        bool ended_clean = false;
        try {
            // Drain until end/raise; cancel fires mid-wait.
            while (stream.next()) {
            }
            ended_clean = true;
        } catch (const std::runtime_error&) {
            ended_clean = true;  // daemon -2814 raise arrived first: also an end
        }
        const double wake_ms = now_ms() - t0;
        canceller.join();
        std::printf("[sub] cancel(): stream ended in %.0f ms (clean=%d)\n", wake_ms,
                    ended_clean ? 1 : 0);
        // Bound: cancel at 2 s + slack; anything ≤ 6 s proves wake-on-cancel
        // (an unwakeable next() would hang until the 30 s test timeout).
        if (ended_clean && wake_ms < 6000.0)
            pass("cancel() wakes blocked next()");
        else
            fail("cancel() wake",
                 (std::string("ended in ") + std::to_string(wake_ms) +
                  " ms, clean=" + (ended_clean ? "1" : "0"))
                     .c_str());
    } catch (const std::exception& e) {
        fail("cancel() path", e.what());
    }

    // ---- windowed get_stats + skew aggregates -----------------------------------
    try {
        const InferenceSystemStats st = inf.get_stats(10);  // clamped [1,5000]
        std::printf("[stats] util=%.2f temp=%.1fC cpu=%.2f models=%zu\n",
                    st.device_utilization, st.device_temperature, st.cpu_utilization,
                    st.model_stats.size());
        for (const auto& m : st.model_stats) {
            if (m.model_id == kModelId)
                std::printf("[stats] %s: %llu inferences, skew avg=%llums max=%llums "
                            "samples=%llu\n",
                            m.model_id.c_str(), (unsigned long long)m.total_inferences,
                            (unsigned long long)m.avg_skew_us / 1000,
                            (unsigned long long)m.max_skew_us / 1000,
                            (unsigned long long)m.skew_samples);
        }
        pass("get_stats(windowed, clamped)");
    } catch (const std::exception& e) {
        fail("get_stats windowed", e.what());
    }

    // ---- cleanup ------------------------------------------------------------------
    if (registered) {
        try {
            inf.unregister_model(kModelId);
            bool gone = true;
            for (const auto& m : inf.list_models())
                if (m.model_id == kModelId) gone = false;
            if (gone)
                pass("unregister_model + verified absent");
            else
                fail("unregister_model", "model still listed");
        } catch (const std::exception& e) {
            fail("unregister_model", e.what());
        }
    }
    inf.close();

    std::printf("\n=== phase4 smoke done: %d failure(s) ===\n", failures);
    return failures ? 1 : 0;
}
