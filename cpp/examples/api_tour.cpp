// api_tour.cpp — full-SDK API tour for the NE503 C++ SDK.
//
// Calls one or two core interfaces from every client module and prints a
// per-module PASS / SKIP / FAIL summary. Read-only by design: safe to run on
// a live device at any time.
//
//   * gRPC clients (Inference, Events, Device, Camera, App, Audio, Overlay) —
//     connect + query RPCs only. Write-side RPCs (lights, PTZ, encoder
//     reconfig, overlay config, install/start/stop) are NOT exercised. The
//     single exception: the event-bus round trip publishes one JSON ping on
//     the private topic "sdk/api_tour/echo" and reads it back — that IS the
//     core publish/subscribe interface and touches no other subscriber.
//   * Raw-UDS stream clients (FdMedia, Encoded, AudioStream) — one bounded
//     get_frame() each. Timeout / absent socket => SKIP (stream simply not
//     served on this device), not FAIL.
//
// Run on-device:
//
//   api_tour
//
// Exit code: number of FAILED modules (0 = every exercised module healthy).

#include <opencv2/core.hpp>
#include <nlohmann/json.hpp>

#include <atomic>
#include <chrono>
#include <cstdio>
#include <exception>
#include <future>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "hailo_ipc_sdk/app.hpp"
#include "hailo_ipc_sdk/audio.hpp"
#include "hailo_ipc_sdk/audio_stream.hpp"
#include "hailo_ipc_sdk/camera.hpp"
#include "hailo_ipc_sdk/config.hpp"
#include "hailo_ipc_sdk/device.hpp"
#include "hailo_ipc_sdk/events.hpp"
#include "hailo_ipc_sdk/inference.hpp"
#include "hailo_ipc_sdk/media.hpp"
#include "hailo_ipc_sdk/overlay.hpp"
#include "hailo_ipc_sdk/plugin.hpp"

namespace {

enum class Status { Pass, Skip, Fail };

const char* status_label(Status s) {
    switch (s) {
        case Status::Pass: return "PASS";
        case Status::Skip: return "SKIP";
        case Status::Fail: return "FAIL";
    }
    return "????";
}

struct Result {
    std::string name;
    Status status;
    std::string detail;
};

std::vector<Result> g_results;

void report(const std::string& name, Status status, const std::string& detail) {
    g_results.push_back({name, status, detail});
    std::printf("[%s] %-13s %s\n", status_label(status), name.c_str(), detail.c_str());
}

// printf-friendly float ("3.7").
std::string f1(float v) {
    char buf[32];
    std::snprintf(buf, sizeof buf, "%.1f", static_cast<double>(v));
    return buf;
}

}  // namespace

int main() {
    using namespace hailo_ipc_sdk;
    std::printf("=== NE503 C++ SDK API tour ===\n");

    // ---- Config (static; no RPC) ------------------------------------------
    {
        std::string detail = "app_id=\"" + Config::get_app_id() + "\"" +
                             " inference=" + Config::get_inference_endpoint() +
                             " events=" + Config::get_event_bus_endpoint() +
                             " device=" + Config::get_device_control_endpoint();
        report("config", Status::Pass, detail);
    }

    // ---- Inference: list_models + get_stats (+ infer when a model exists) --
    try {
        InferenceClient inference;
        const auto models = inference.list_models();
        std::string detail = std::to_string(models.size()) + " model(s)";

        if (!models.empty()) {
            const std::string& model_id = models.front().model_id;
            try {
                const cv::Mat probe(480, 640, CV_8UC3, cv::Scalar(0, 0, 0));
                const InferenceResult r = inference.infer(probe, model_id);
                detail += "; infer(\"" + model_id + "\") -> " +
                          std::to_string(r.objects.size()) + " objects, status=\"" +
                          r.status_message + "\", " + std::to_string(r.infer_time_us / 1000) + "ms";
            } catch (const std::exception& e) {
                detail += "; infer(\"" + model_id + "\") rejected: " + e.what();
            }
        } else {
            detail += "; infer skipped (no model loaded)";
        }

        const InferenceSystemStats st = inference.get_stats();
        detail += "; NPU util " + f1(st.device_utilization * 100.0f) + "%" +
                  ", temp " + f1(st.device_temperature) + "C" +
                  ", " + std::to_string(st.model_stats.size()) + " model stat(s)";
        report("inference", Status::Pass, detail);
    } catch (const std::exception& e) {
        report("inference", Status::Fail, e.what());
    }

    // ---- Events: stats + topics + publish<->subscribe echo -----------------
    try {
        EventClient events;
        const auto topics = events.list_topics();
        const SystemStats st = events.get_stats();
        std::string detail = std::to_string(topics.size()) + " topic(s), " +
                             std::to_string(st.total_subscribers) + " subscriber(s), uptime " +
                             std::to_string(st.uptime_ms / 1000) + "s";

        // Round trip on a private topic: on_event -> publish -> future.
        const std::string topic = "sdk/api_tour/echo";
        auto echoed = std::make_shared<std::promise<std::string>>();
        std::future<std::string> echoed_fut = echoed->get_future();
        const auto done = std::make_shared<std::atomic<bool>>(false);
        events.on_event(topic, [echoed, done](const Event& ev) {
            if (!done->exchange(true)) echoed->set_value(ev.payload.dump());
        });
        std::this_thread::sleep_for(std::chrono::milliseconds(300));  // let the subscription register
        const std::string event_id = events.publish(topic, nlohmann::json{{"ping", true}});
        if (echoed_fut.wait_for(std::chrono::seconds(3)) == std::future_status::ready) {
            detail += "; echo OK (event_id=" + event_id + ", payload " + echoed_fut.get() + ")";
            report("events", Status::Pass, detail);
        } else {
            report("events", Status::Fail,
                   detail + "; published event_id=" + event_id + " but no echo within 3s");
        }
    } catch (const std::exception& e) {
        report("events", Status::Fail, e.what());
    }

    // ---- Device: status + lens --------------------------------------------
    try {
        DeviceClient device;
        const DeviceStatus st = device.get_device_status();
        const LensStatus lens = device.get_lens_status();
        std::string detail = "SoC " + f1(st.soc_temp_c) + "C, MCU " + f1(st.mcu_temp_c) + "C" +
                             ", zoom " + std::to_string(st.zoom_pos) +
                             "/" + std::to_string(lens.zoom_limit.max_pos) +
                             ", focus " + std::to_string(st.focus_pos) +
                             "/" + std::to_string(lens.focus_limit.max_pos) +
                             ", mcu \"" + st.mcu_version + "\"";
        report("device", Status::Pass, detail);
    } catch (const std::exception& e) {
        report("device", Status::Fail, e.what());
    }

    // ---- Camera: capabilities + streams + sensor ---------------------------
    try {
        CameraClient camera;
        const Capabilities caps = camera.get_capabilities();
        const auto streams = camera.get_stream_status();
        const SensorInfo sensor = camera.get_sensor_info();
        std::string detail = "video=" + std::string(caps.has_video ? "y" : "n") +
                             ", codec=" + std::string(caps.has_codec ? "y" : "n") +
                             ", audio=" + std::string(caps.has_audio ? "y" : "n") +
                             ", " + std::to_string(streams.size()) + " stream(s)";
        if (!streams.empty()) {
            const auto& s = streams.front();
            detail += " [first: " + s.stream_id + " " + s.codec + " " +
                      std::to_string(s.width) + "x" + std::to_string(s.height) +
                      "@" + std::to_string(s.fps) + "]";
        }
        if (sensor.available) detail += ", sensor " + sensor.sensor_model;
        report("camera", Status::Pass, detail);
    } catch (const std::exception& e) {
        report("camera", Status::Fail, e.what());
    }

    // ---- App: list_apps -----------------------------------------------------
    try {
        AppClient apps;
        const auto list = apps.list_apps();
        std::string detail = std::to_string(list.size()) + " app(s)";
        for (const auto& a : list)
            detail += " [" + a.id + ":" + a.state + "]";
        report("app", Status::Pass, detail);
    } catch (const std::exception& e) {
        report("app", Status::Fail, e.what());
    }

    // ---- Audio control: status + capture devices ----------------------------
    try {
        AudioClient audio;
        const AudioStatus st = audio.get_status();
        const auto devices = audio.list_capture_devices();
        std::string detail = "capturing=" + std::string(st.capturing ? "y" : "n") +
                             ", playing=" + std::string(st.playing ? "y" : "n") +
                             ", " + std::to_string(devices.size()) + " capture device(s)";
        if (!devices.empty()) detail += " [first: " + devices.front().name + "]";
        report("audio", Status::Pass, detail);
    } catch (const std::exception& e) {
        report("audio", Status::Fail, e.what());
    }

    // ---- Overlay: connect only (every RPC is write-side) --------------------
    try {
        OverlayClient overlay;
        overlay.connect();
        report("overlay", Status::Pass,
               "connected; write-side RPCs not exercised (read-only tour)");
    } catch (const std::exception& e) {
        report("overlay", Status::Fail, e.what());
    }

    // ---- FdMedia: zero-copy frame + encoded stream ---------------------------
    try {
        FdMediaClient media;
        const auto streams = media.list_streams();
        std::string detail = "streams:";
        for (const auto& s : streams) detail += " " + s;

        bool got_frame = false;
        std::vector<std::string> try_ids = {"cam0_main"};
        for (const auto& s : streams) try_ids.push_back(s);
        for (const auto& sid : try_ids) {
            if (auto frame = media.get_frame(sid, 2000)) {
                detail += "; get_frame(\"" + sid + "\") -> seq " +
                          std::to_string(frame->sequence) + " " +
                          std::to_string(frame->width) + "x" +
                          std::to_string(frame->height) + " " + frame->format;
                got_frame = true;
                break;
            }
        }
        if (!got_frame) detail += "; no frame within 2s (stream idle?)";

        // Encoded path via the SDK convenience wrapper.
        try {
            EncodedStreamClient enc = media.get_encoded_stream("main");
            if (auto frame = enc.get_frame(2000)) {
                detail += "; encoded main " + frame->codec_name() + " " +
                          std::to_string(frame->width) + "x" +
                          std::to_string(frame->height) + " " +
                          std::to_string(frame->data.size()) + "B";
            } else {
                detail += "; encoded main: no frame within 2s";
            }
        } catch (const std::exception& e) {
            detail += std::string("; encoded main: ") + e.what();
        }

        report("media(fd)", got_frame ? Status::Pass : Status::Skip, detail);
    } catch (const std::exception& e) {
        report("media(fd)", Status::Skip, e.what());
    }

    // ---- AudioStream: one captured frame -------------------------------------
    try {
        AudioStreamClient audio_stream;
        if (auto frame = audio_stream.get_frame(2000)) {
            report("audio_stream", Status::Pass,
                   frame->codec_name() + " " + std::to_string(frame->sample_rate) + "Hz " +
                   std::to_string(frame->channels) + "ch " +
                   std::to_string(frame->bits_per_sample) + "bit, " +
                   std::to_string(frame->data.size()) + "B");
        } else {
            report("audio_stream", Status::Skip, "no frame within 2s (capture idle?)");
        }
    } catch (const std::exception& e) {
        report("audio_stream", Status::Skip, e.what());
    }

    // ---- Plugin discovery -----------------------------------------------------
    try {
        PluginDiscovery discovery;
        const auto caps = discovery.list_capabilities();
        report("plugin", Status::Pass,
               std::to_string(caps.size()) + " capabilit(ies)" +
               (caps.empty() ? " (no plugins deployed)" : ""));
    } catch (const std::exception& e) {
        report("plugin", Status::Skip, e.what());
    }

    // ---- Summary ---------------------------------------------------------------
    int passes = 0, skips = 0, fails = 0;
    for (const auto& r : g_results) {
        if (r.status == Status::Pass) ++passes;
        else if (r.status == Status::Skip) ++skips;
        else ++fails;
    }
    std::printf("\n=== API tour: %d PASS / %d SKIP / %d FAIL ===\n", passes, skips, fails);
    return fails;
}
