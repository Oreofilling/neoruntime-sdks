// person_detection.cpp — 1:1 port of python/examples/person_detection.py.
//
// Subscribes to a video-stream inference feed, counts detected persons each
// frame, publishes a JSON event per positive frame, and (optionally) hooks
// device control. Run on-device inside an app container.
//
//   person_detection                        # defaults: stream=cam0_main model=person_v1
#include <atomic>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <string>

#include <nlohmann/json.hpp>

#include "hailo_ipc_sdk/config.hpp"
#include "hailo_ipc_sdk/events.hpp"
#include "hailo_ipc_sdk/inference.hpp"

namespace {
std::atomic<bool> g_running{true};

void on_signal(int sig) {
    std::printf("\n[signal] received %d, shutting down...\n", sig);
    g_running.store(false);
}
}  // namespace

int main() {
    // signal() is not async-signal-safe in general, but a single atomic store is
    // fine here (mirrors the Python handler's only action: set running=false).
    std::signal(SIGINT, on_signal);
    std::signal(SIGTERM, on_signal);

    const std::string app_id = hailo_ipc_sdk::Config::get_app_id();
    const bool debug = hailo_ipc_sdk::Config::is_debug();

    std::printf("[%s] Person Detection App initialized\n", app_id.c_str());
    std::printf("[%s] AI Runtime: %s\n", app_id.c_str(),
                hailo_ipc_sdk::Config::get_inference_endpoint().c_str());
    std::printf("[%s] Event Bus:  %s\n", app_id.c_str(),
                hailo_ipc_sdk::Config::get_event_bus_endpoint().c_str());

    hailo_ipc_sdk::InferenceClient inference;
    hailo_ipc_sdk::EventClient events;
    std::printf("[%s] Starting person detection...\n", app_id.c_str());

    try {
        auto stream = inference.subscribe("cam0_main", "person_v1", /*fps=*/10);

        while (g_running.load()) {
            auto item = stream.next();
            if (!item) break;  // server closed/broke the stream

            const auto frame_seq = item->first;
            const auto& result = item->second;

            auto persons = result.get_objects_by_label("person");
            if (persons.empty()) continue;

            std::printf("[Frame %llu] Detected %zu person(s)\n",
                        static_cast<unsigned long long>(frame_seq), persons.size());

            if (debug) {
                for (const auto& obj : persons) {
                    std::printf("  - confidence: %.2f\n", obj.score);
                    std::printf("    position: (%.2f, %.2f)\n", obj.bbox.x, obj.bbox.y);
                    std::printf("    size: %.2f x %.2f\n", obj.bbox.width, obj.bbox.height);
                }
            }

            // Publish a detection event (payload mirrors the Python dict exactly).
            nlohmann::json objects_json = nlohmann::json::array();
            for (const auto& obj : persons) {
                objects_json.push_back({
                    {"confidence", obj.score},
                    {"bbox",
                     {obj.bbox.x, obj.bbox.y, obj.bbox.width, obj.bbox.height}},
                });
            }
            events.publish("app/" + app_id + "/person_detected",
                           {
                               {"frame_sequence", frame_seq},
                               {"timestamp_ns", result.timestamp_ns},
                               {"person_count", persons.size()},
                               {"objects", objects_json},
                           });
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[%s] Error: %s\n", app_id.c_str(), e.what());
        if (debug) std::fprintf(stderr, "[%s] (set NE503_DEBUG for a traceback)\n", app_id.c_str());
        inference.close();
        events.close();
        return EXIT_FAILURE;
    }

    std::printf("[%s] Cleaning up...\n", app_id.c_str());
    inference.close();
    events.close();
    std::printf("[%s] Shutdown complete\n", app_id.c_str());
    return EXIT_SUCCESS;
}
