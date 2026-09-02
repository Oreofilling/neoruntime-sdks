// perimeter_guard.cpp — 1:1 port of python/examples/perimeter_guard.py.
//
// Subscribes to a person-detection inference stream, fires a (rate-limited)
// perimeter-crossing alert when a person crosses a virtual line, and switches
// the white light on/off around the event.
#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "neoruntime_ipc_sdk/config.hpp"
#include "neoruntime_ipc_sdk/device.hpp"
#include "neoruntime_ipc_sdk/events.hpp"
#include "neoruntime_ipc_sdk/inference.hpp"

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

    neoruntime_ipc_sdk::InferenceClient inference;
    neoruntime_ipc_sdk::EventClient events;
    neoruntime_ipc_sdk::DeviceClient device;

    const double alert_cooldown = 5.0;
    const double light_timeout = 10.0;
    const double detection_line_x = 0.3;  // right of this x ...
    const double detection_line_y = 0.7;  // ... and below this y => crossing
    double last_alert_time = -alert_cooldown;
    bool light_on = false;
    double light_on_time = 0.0;

    std::printf("[%s] Perimeter Guard App initialized\n", app_id.c_str());

    auto turn_on_light = [&]() {
        try {
            device.set_white_light(100);
            light_on = true;
            light_on_time = now_s();
            std::printf("[%s] Light turned ON\n", app_id.c_str());
        } catch (const std::exception& e) {
            std::printf("[%s] Failed to control light: %s\n", app_id.c_str(), e.what());
        }
    };
    auto turn_off_light = [&]() {
        try {
            device.set_white_light(0);
            light_on = false;
            std::printf("[%s] Light turned OFF\n", app_id.c_str());
        } catch (const std::exception& e) {
            std::printf("[%s] Failed to control light: %s\n", app_id.c_str(), e.what());
        }
    };
    auto is_crossing = [&](const neoruntime_ipc_sdk::DetectedObject& p) {
        double cx = p.bbox.x + p.bbox.width / 2.0;
        double cy = p.bbox.y + p.bbox.height / 2.0;
        return cx > detection_line_x && cy > detection_line_y;
    };

    std::printf("[%s] Starting perimeter guard...\n", app_id.c_str());
    try {
        auto stream = inference.subscribe("cam0_main", "person_v1", /*fps=*/15);
        while (g_running.load()) {
            auto item = stream.next();
            if (!item) break;
            const auto frame_seq = item->first;
            const auto& result = item->second;

            auto persons = result.get_objects_by_label("person");

            std::vector<neoruntime_ipc_sdk::DetectedObject> crossed;
            for (const auto& p : persons) {
                if (is_crossing(p)) crossed.push_back(p);
            }

            if (!crossed.empty()) {
                double t = now_s();
                if (t - last_alert_time >= alert_cooldown) {
                    std::printf("[ALERT] %zu person(s) crossed the boundary!\n", crossed.size());
                    nlohmann::json confidences = nlohmann::json::array();
                    for (const auto& p : crossed) confidences.push_back(p.score);
                    events.publish("app/" + app_id + "/perimeter_alert",
                                   {
                                       {"type", "boundary_crossing"},
                                       {"frame_sequence", frame_seq},
                                       {"person_count", crossed.size()},
                                       {"confidence", confidences},
                                   },
                                   /*persistent=*/true);
                    turn_on_light();
                    last_alert_time = t;
                }
            }

            if (!persons.empty() && !light_on) turn_on_light();

            if (light_on && now_s() - light_on_time >= light_timeout) turn_off_light();
        }
    } catch (const std::exception& e) {
        std::printf("[%s] Error: %s\n", app_id.c_str(), e.what());
    }

    if (light_on) turn_off_light();
    inference.close();
    events.close();
    device.close();
    std::printf("[%s] Cleanup complete\n", app_id.c_str());
    return EXIT_SUCCESS;
}
