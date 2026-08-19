// connectivity_smoke.cpp — read-only gRPC round-trip smoke for the NE503 C++ SDK.
//
// Runs on-device against the real /run/aipc/*.sock daemons. It issues ONLY
// read-only query RPCs (no device mutation, no model registration, no streams),
// so it is safe to run against live hardware at any time. Each daemon is tried
// independently in its own try/catch; a failure on one does not mask the others.
//
//   connectivity_smoke
//
// Exit code: 0 if every queried daemon answered, non-zero if any RPC failed.

#include <cstdio>
#include <cstdint>
#include <exception>
#include <string>

#include "hailo_ipc_sdk/config.hpp"
#include "hailo_ipc_sdk/inference.hpp"
#include "hailo_ipc_sdk/events.hpp"
#include "hailo_ipc_sdk/device.hpp"

int main() {
    using namespace hailo_ipc_sdk;

    std::printf("=== NE503 C++ SDK connectivity smoke ===\n");
    std::printf("app_id    : %s\n", Config::get_app_id().c_str());
    std::printf("ai-runtime: %s\n", Config::get_inference_endpoint().c_str());
    std::printf("event-bus : %s\n", Config::get_event_bus_endpoint().c_str());
    std::printf("device-ctl: %s\n", Config::get_device_control_endpoint().c_str());

    int failures = 0;

    // --- InferenceClient -> ai-runtime.sock : get_stats() ---
    try {
        InferenceClient inference;
        std::printf("\n[InferenceClient] connected=%d\n", inference.connected() ? 1 : 0);
        const auto stats = inference.get_stats();
        std::printf("[InferenceClient] get_stats() OK\n");
        std::printf("  loaded models    : %zu\n", stats.model_stats.size());
        std::printf("  device util      : %.1f%%\n", stats.device_utilization * 100.0f);
        std::printf("  device temp (C)  : %.1f\n", stats.device_temperature);
        std::printf("  mem used/total   : %llu / %llu KiB\n",
                    static_cast<unsigned long long>(stats.ram_used_kib),
                    static_cast<unsigned long long>(stats.ram_total_kib));
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[InferenceClient] FAILED: %s\n", e.what());
        ++failures;
    }

    // --- EventClient -> event-bus.sock : get_stats() ---
    try {
        EventClient events;
        std::printf("\n[EventClient] connected=%d\n", events.connected() ? 1 : 0);
        const auto stats = events.get_stats();
        std::printf("[EventClient] get_stats() OK\n");
        std::printf("  topics           : %u\n", stats.total_topics);
        std::printf("  subscribers      : %u\n", stats.total_subscribers);
        std::printf("  bus uptime (ms)  : %llu\n",
                    static_cast<unsigned long long>(stats.uptime_ms));
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[EventClient] FAILED: %s\n", e.what());
        ++failures;
    }

    // --- DeviceClient -> device-control.sock : get_device_status() ---
    try {
        DeviceClient device;
        std::printf("\n[DeviceClient] connected=%d\n", device.connected() ? 1 : 0);
        const auto status = device.get_device_status();
        std::printf("[DeviceClient] get_device_status() OK\n");
        std::printf("  SoC temp (C)     : %.1f\n", status.soc_temp_c);
        std::printf("  MCU temp (C)     : %.1f\n", status.mcu_temp_c);
        std::printf("  zoom/focus pos   : %d / %d\n", status.zoom_pos, status.focus_pos);
        std::printf("  MCU version      : %s\n", status.mcu_version.c_str());
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[DeviceClient] FAILED: %s\n", e.what());
        ++failures;
    }

    std::printf("\n=== smoke done: %d failure(s) ===\n", failures);
    return failures ? 1 : 0;
}
