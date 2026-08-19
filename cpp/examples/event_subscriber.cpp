// event_subscriber.cpp — 1:1 port of python/examples/event_subscriber.py.
//
// Subscribes to several event topics (detections / alerts / system), prints
// them, acks alerts, and toggles the white light when a person is detected.
#include <atomic>
#include <algorithm>
#include <cctype>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <string>
#include <thread>

#include <nlohmann/json.hpp>

#include "hailo_ipc_sdk/config.hpp"
#include "hailo_ipc_sdk/device.hpp"
#include "hailo_ipc_sdk/events.hpp"

namespace {
std::atomic<bool> g_running{true};

void on_signal(int sig) {
    (void)sig;
    std::printf("\nShutting down...\n");
    g_running.store(false);
}

bool contains_ci(const std::string& haystack, const std::string& needle) {
    std::string h = haystack, n = needle;
    std::transform(h.begin(), h.end(), h.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    std::transform(n.begin(), n.end(), n.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    return h.find(n) != std::string::npos;
}
}  // namespace

int main() {
    std::signal(SIGINT, on_signal);
    std::signal(SIGTERM, on_signal);

    const std::string app_id = hailo_ipc_sdk::Config::get_app_id();
    const bool debug = hailo_ipc_sdk::Config::is_debug();

    hailo_ipc_sdk::EventClient events;
    hailo_ipc_sdk::DeviceClient device;

    std::printf("[%s] Event Subscriber App initialized\n", app_id.c_str());
    std::printf("[%s] Starting event subscriptions...\n", app_id.c_str());

    auto on_detection = [&](const hailo_ipc_sdk::Event& ev) {
        std::printf("[%s] Detection event: %s\n", app_id.c_str(), ev.topic.c_str());
        if (debug) std::printf("  Payload: %s\n", ev.payload.dump().c_str());
        if (contains_ci(ev.payload.dump(), "person")) {
            std::printf("[%s] Person detected!\n", app_id.c_str());
            try {
                device.set_white_light(80);
            } catch (const std::exception& e) {
                std::printf("[%s] Failed to control device: %s\n", app_id.c_str(), e.what());
            }
        }
    };

    auto on_alert = [&](const hailo_ipc_sdk::Event& ev) {
        std::printf("[%s] Alert received: %s\n", app_id.c_str(), ev.topic.c_str());
        std::printf("  Source: %s\n", ev.source.c_str());
        std::printf("  Data: %s\n", ev.payload.dump().c_str());
        events.publish("app/" + app_id + "/alert_ack",
                       {
                           {"original_topic", ev.topic},
                           {"original_event_id", ev.event_id},
                           {"acknowledged", true},
                       });
    };

    auto on_system = [&](const hailo_ipc_sdk::Event& ev) {
        if (debug) std::printf("[%s] System event: %s\n", app_id.c_str(), ev.topic.c_str());
    };

    struct TopicCb {
        const char* topic;
        std::function<void(const hailo_ipc_sdk::Event&)> cb;
    };
    const TopicCb topics[] = {
        {"model/+/detections", on_detection},
        {"app/+/alert", on_alert},
        {"system/device/#", on_system},
    };

    for (const auto& t : topics) {
        try {
            events.on_event(t.topic, t.cb);
            std::printf("[%s] Subscribed to: %s\n", app_id.c_str(), t.topic);
        } catch (const std::exception& e) {
            std::printf("[%s] Failed to subscribe %s: %s\n", app_id.c_str(), t.topic, e.what());
        }
    }

    while (g_running.load()) {
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }

    events.close();
    device.close();
    std::printf("[%s] Cleanup complete\n", app_id.c_str());
    return EXIT_SUCCESS;
}
