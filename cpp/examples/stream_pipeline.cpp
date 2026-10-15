// stream_pipeline.cpp — one call, whole pipeline (plan: examples 增 3 个).
//
// StreamPipeline is the platform-scheduled path: the camera daemon feeds
// frames to the model (no pixels cross into this process), results are
// drawn on the hardware overlay, and the app reads lightweight
// InferenceResult structs. This example registers a detection model,
// draws a zone, filters through an on_result hook, and reports pipeline
// health from status() — Ctrl-C for a clean stop that clears the overlay.
//
//   stream_pipeline                # defaults: stream=third model env-overridable
#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#include "neoruntime_ipc_sdk/inference.hpp"
#include "neoruntime_ipc_sdk/overlay.hpp"
#include "neoruntime_ipc_sdk/stream_pipeline.hpp"

namespace {
std::atomic<bool> g_running{true};

void on_signal(int sig) {
    std::printf("\n[signal] received %d, shutting down...\n", sig);
    g_running.store(false);
}

constexpr const char* kDefaultModelPath =
    "/data/aipc-data/models/hailo_yolov8n_384_640.hef";
constexpr const char* kDefaultModelId = "example-y8n";

}  // namespace

int main() {
    using namespace neoruntime_ipc_sdk;
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    std::signal(SIGINT, on_signal);
    std::signal(SIGTERM, on_signal);

    const char* env_path = std::getenv("NE503_MODEL_PATH");
    const char* env_id = std::getenv("NE503_MODEL_ID");
    const std::string model_path = env_path ? env_path : kDefaultModelPath;
    const std::string model_id = env_id ? env_id : kDefaultModelId;

    InferenceClient inference;
    bool registered = false;
    try {
        inference.connect();
        // pre-clean: a stale untyped registration of the same id would be
        // re-materialized by the daemon — unregister first (no owner_id on
        // the typed register so it lands in the global table)
        for (const ModelInfo& m : inference.list_models())
            if (m.model_id == model_id) inference.unregister_model(model_id);
        inference.register_model(model_path, model_id, /*owner_id=*/"",
                                 /*model_type=*/"detection");
        registered = true;
        std::printf("model %s registered (detection)\n", model_id.c_str());
    } catch (const std::exception& e) {
        std::fprintf(stderr, "model registration failed: %s\n", e.what());
        return 1;
    }

    int exit_code = 0;
    try {
        // a zone to draw regardless of detections, plus per-result rules
        OverlayPolygon zone;
        zone.points = {OverlayPoint{0.10f, 0.10f}, OverlayPoint{0.90f, 0.10f},
                       OverlayPoint{0.90f, 0.90f}, OverlayPoint{0.10f, 0.90f}};
        zone.label = "watch zone";

        StreamPipeline pipe(
            "third", model_id,
            /*fps=*/5, "stream-pipeline-example",
            /*raw_output_only=*/false,
            /*max_consecutive_failures=*/10,
            /*queue_size=*/8, /*draw=*/true,
            /*overlay_config=*/std::nullopt,
            /*ttl_ms=*/2000, /*min_score=*/0.35,
            /*labels=*/std::nullopt,
            /*polygons=*/std::vector<OverlayPolygon>{zone},
            /*on_result=*/[](const InferenceResult& result)
                -> std::optional<InferenceResult> {
                // Hook runs on the worker thread per result: return a
                // replaced result, or nullopt to drop it (not drawn, not
                // queued). Deliberately a pass-through here — a hook that
                // drops most results starves next_result(), and a Ctrl-C
                // loop parked in it would never observe the flag. Filter
                // in the consuming loop instead (below), where every
                // result still reaches the app at frame cadence.
                return result;
            });

        pipe.start();
        std::printf("pipeline running on 'third' — Ctrl-C to stop\n");

        std::uint64_t kept = 0;
        while (g_running.load()) {
            auto item = pipe.next_result();  // blocks; nullopt = stopped
            if (!item.has_value()) break;
            const auto& [seq, result] = *item;
            if (result.objects.empty()) continue;  // app-side rule: 1+ objects
            ++kept;
            std::printf("[frame %llu] %zu object(s)",
                        static_cast<unsigned long long>(seq), result.objects.size());
            for (const DetectedObject& obj : result.objects)
                std::printf("  %s %.2f", obj.label.c_str(), obj.score);
            std::printf("\n");
        }

        const StreamPipelineStatus st = pipe.status();
        std::printf("status: seen=%llu annotated=%llu annotate_errors=%llu "
                    "queue_drops=%llu latency=%.1fms skew=%lldus\n",
                    static_cast<unsigned long long>(st.results_seen),
                    static_cast<unsigned long long>(st.results_annotated),
                    static_cast<unsigned long long>(st.annotate_errors),
                    static_cast<unsigned long long>(st.result_queue_drops),
                    st.last_latency_ms, static_cast<long long>(st.last_skew_us));
        if (!st.last_error.empty())
            std::printf("last_error: %s\n", st.last_error.c_str());

        pipe.stop();  // clears boxes + the zone; closes owned clients
        std::printf("stopped after %llu kept result(s)\n",
                    static_cast<unsigned long long>(kept));
    } catch (const std::exception& e) {
        std::fprintf(stderr, "pipeline failed: %s\n", e.what());
        exit_code = 1;
    }

    if (registered) {
        // brief grace: the daemon releases the model reference from the
        // just-stopped subscription asynchronously
        std::this_thread::sleep_for(std::chrono::seconds(1));
        try {
            inference.unregister_model(model_id);
        } catch (const std::exception& e) {
            std::fprintf(stderr, "unregister_model: %s\n", e.what());
        }
    }
    inference.close();
    return exit_code;
}
