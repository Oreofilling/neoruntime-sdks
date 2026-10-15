// phase6_smoke.cpp — Phase 6 on-device smoke (plan 冒烟④): the P2 tool
// layer's whole-pipeline facade end-to-end on the live platform.
//
// StreamPipeline("third", model, fps=5, ttl, session, zones) against the
// real ai-runtime / event-bus / camera daemons, observed from the outside
// by the smoke's own event-bus subscription on "inference/third" — the
// same vantage a second process (web viewer, app) would have.
//
//   [1] connectivity + endpoints
//   [2] typed model registration (pre-clean recipe from phase 4)
//   [3] observer thread recording overlay events
//   [4] live run: start → overlay config + zone push; results — or the
//       daemon's documented camera-fed DMA rejection surfaced through
//       status().last_error, the exact parity contract with Python on
//       this daemon build; stop → box + zone clears
//   [5] observer assertions: operator writes untagged, result writes
//       tagged with the pipeline session
//
// Exit 0 only if no section FAILed (SKIP is acceptable and explained).
#include <chrono>
#include <cstdio>
#include <future>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "neoruntime_ipc_sdk/config.hpp"
#include "neoruntime_ipc_sdk/events.hpp"
#include "neoruntime_ipc_sdk/inference.hpp"
#include "neoruntime_ipc_sdk/overlay.hpp"
#include "neoruntime_ipc_sdk/stream_pipeline.hpp"

namespace {

constexpr const char* kModelPath = "/data/aipc-data/models/hailo_yolov8n_384_640.hef";
constexpr const char* kModelId = "cpp-p6-y8n";
constexpr const char* kStream = "third";
constexpr const char* kSession = "cpp-p6";

int failures = 0;
int skips = 0;

void pass(const std::string& what) { std::printf("  [PASS] %s\n", what.c_str()); }
void fail(const char* what, const std::string& err) {
    // stdout, not stderr: both smoke logs get diffed, and cross-stream
    // interleaving scrambles the section order
    std::printf("  [FAIL] %s: %s\n", what, err.c_str());
    ++failures;
}
void skip(const char* what, const std::string& why) {
    std::printf("  [SKIP] %s: %s\n", what, why.c_str());
    ++skips;
}

// One overlay event as seen from the outside.
struct Seen {
    std::string session;      // metadata session_id ("" = operator write)
    bool has_polygons = false;
    bool polygons_empty = false;
    bool has_detections = false;
    int num_detections = -1;
    std::string ttl;          // metadata result_ttl_ms ("" = none)
};

std::mutex seen_mu;
std::vector<Seen> seen;

void record(const neoruntime_ipc_sdk::Event& ev) {
    Seen s;
    if (ev.metadata.count("session_id")) s.session = ev.metadata.at("session_id");
    if (ev.metadata.count("result_ttl_ms")) s.ttl = ev.metadata.at("result_ttl_ms");
    s.has_polygons = ev.payload.contains("polygons");
    if (s.has_polygons) s.polygons_empty = ev.payload["polygons"].empty();
    s.has_detections = ev.payload.contains("num_detections");
    if (s.has_detections) s.num_detections = ev.payload["num_detections"];
    std::lock_guard<std::mutex> lock(seen_mu);
    seen.push_back(s);
}

std::vector<Seen> seen_snapshot() {
    std::lock_guard<std::mutex> lock(seen_mu);
    return seen;
}

bool wait_seen(int count, double timeout_s = 8.0) {
    const auto deadline = std::chrono::steady_clock::now() +
                          std::chrono::duration<double>(timeout_s);
    while (std::chrono::steady_clock::now() < deadline) {
        if (seen_snapshot().size() >= static_cast<std::size_t>(count)) return true;
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    return seen_snapshot().size() >= static_cast<std::size_t>(count);
}

}  // namespace

int main() {
    using namespace neoruntime_ipc_sdk;
    std::setvbuf(stdout, nullptr, _IONBF, 0);

    std::printf("[1] connectivity\n");
    std::printf("  inference=%s\n  events=%s\n  camera=%s\n",
                Config::get_inference_endpoint().c_str(),
                Config::get_event_bus_endpoint().c_str(),
                Config::get_camera_control_endpoint().c_str());
    InferenceClient inf;
    std::size_t models_before = 0;
    try {
        inf.connect();
        models_before = inf.list_models().size();
        pass("connect + list_models (" + std::to_string(models_before) + " models)");
    } catch (const std::exception& e) {
        fail("connect", e.what());
        std::printf("=== Phase 6 smoke: FAIL (failures=1) ===\n");
        return 1;
    }

    std::printf("[2] typed model registration\n");
    bool registered = false;
    try {
        for (const auto& m : inf.list_models())
            if (m.model_id == kModelId) inf.unregister_model(kModelId);
        inf.register_model(kModelPath, kModelId, /*owner_id=*/"",
                           /*model_type=*/"detection");
        registered = true;
        pass("register_model(typed detection, no owner_id)");
    } catch (const std::exception& e) {
        fail("register_model", e.what());
    }

    std::printf("[3] overlay event observer on '%s'\n", kStream);
    EventClient bus;
    // Pull subscription + registry handshake, NOT on_event: on_event's
    // detached thread races start()'s synchronous zone push (observed on
    // 93.72: the push is published before the daemon registered the
    // subscriber and is missed). subscribe() initiates the stream here;
    // polling get_topic_info() until subscriber_count grows proves the
    // daemon's registry holds us — events published after that point are
    // delivered. The drainer is detached and owns the stream via
    // shared_ptr, mirroring Python's daemon-thread observer.
    std::shared_ptr<EventStream> observer;
    try {
        bus.connect();
        const std::string topic = std::string("inference/") + kStream;
        std::uint32_t baseline = 0;
        for (const TopicInfo& t : bus.list_topics())
            if (t.topic == topic) baseline = t.subscriber_count;
        observer = std::make_shared<EventStream>(bus.subscribe(topic));
        std::thread([observer] {
            while (std::optional<Event> ev = observer->next()) record(*ev);
        }).detach();
        bool live = false;
        for (int i = 0; i < 100 && !live; ++i) {
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
            const std::optional<TopicInfo> info = bus.get_topic_info(topic);
            if (info.has_value() && info->subscriber_count > baseline) live = true;
        }
        if (live)
            pass("subscribed + registry handshake (outside vantage)");
        else
            fail("observer handshake", "subscriber never appeared in topic registry");
    } catch (const std::exception& e) {
        fail("event observer", e.what());
    }

    std::printf("[4] StreamPipeline live run\n");
    std::string dma_reject;
    try {
        OverlayPolygon zone;
        zone.points = {OverlayPoint{0.05f, 0.05f}, OverlayPoint{0.95f, 0.05f},
                       OverlayPoint{0.95f, 0.95f}, OverlayPoint{0.05f, 0.95f}};
        zone.label = "smoke-zone";
        StreamPipeline pipe(kStream, kModelId,
                            /*fps=*/5, kSession,
                            /*raw_output_only=*/false,
                            /*max_consecutive_failures=*/10,
                            /*queue_size=*/8, /*draw=*/true,
                            /*overlay_config=*/std::nullopt,
                            /*ttl_ms=*/2000,
                            /*min_score=*/0.3,
                            /*labels=*/std::nullopt,
                            /*polygons=*/std::vector<OverlayPolygon>{zone});
        pipe.start();
        const bool zone_pushed = wait_seen(1);
        if (zone_pushed && seen_snapshot()[0].has_polygons)
            pass("start(): static zone pushed on the bus (untagged)");
        else
            fail("zone push", "observer saw no polygon event after start()");

        // next_result ends on the daemon's stream state: results when the
        // camera-fed path works, nullopt after max_consecutive_failures
        // when it rejects DMA inputs. Box it so neither case can hang.
        auto fut = std::async(std::launch::async, [&pipe] { return pipe.next_result(); });
        bool got_result = false;
        if (fut.wait_for(std::chrono::seconds(30)) == std::future_status::ready) {
            auto item = fut.get();
            got_result = item.has_value();
            if (got_result) {
                std::printf("  first result: seq=%llu objects=%zu\n",
                            static_cast<unsigned long long>(item->first),
                            item->second.objects.size());
            }
        } else {
            pipe.stop();
            fut.get();  // stop marker ends it
            fail("next_result", "no terminal state within 30s");
        }

        const StreamPipelineStatus st = pipe.status();
        std::printf("  status: running=%d seen=%llu annotated=%llu drops=%llu "
                    "skew_last=%lldus latency_last=%.1fms\n",
                    st.running ? 1 : 0, static_cast<unsigned long long>(st.results_seen),
                    static_cast<unsigned long long>(st.results_annotated),
                    static_cast<unsigned long long>(st.result_queue_drops),
                    static_cast<long long>(st.last_skew_us), st.last_latency_ms);
        if (!st.last_error.empty())
            std::printf("  last_error: %s\n", st.last_error.c_str());

        if (got_result && st.results_seen > 0) {
            pass("results flowed end-to-end (subscribe -> overlay -> app queue)");
        } else if (st.last_error.find("2811") != std::string::npos ||
                   st.last_error.find("DMA input rejected") != std::string::npos) {
            dma_reject = st.last_error;
            skip("camera-fed results",
                 "daemon rejects DMA inputs on this build (documented -2811); "
                 "SDK surfaced it verbatim through status().last_error after "
                 "max_consecutive_failures, the Python-parity contract");
        } else {
            fail("results", st.last_error.empty() ? "no results and no error" : st.last_error);
        }

        const std::size_t before_stop = seen_snapshot().size();
        pipe.stop();
        const bool cleared = wait_seen(static_cast<int>(before_stop) + 2);
        if (cleared)
            pass("stop(): box + zone clears published");
        else
            fail("teardown clears", "observer did not see both clear events");
    } catch (const std::exception& e) {
        fail("StreamPipeline run", e.what());
    }

    std::printf("[5] observer verdict\n");
    try {
        const std::vector<Seen> evs = seen_snapshot();
        int zone_push = 0, clear_boxes = 0, clear_zones = 0, tagged = 0;
        for (const Seen& s : evs) {
            if (s.has_polygons && !s.polygons_empty && s.session.empty()) ++zone_push;
            if (s.has_detections && s.num_detections == 0 && !s.has_polygons &&
                s.session.empty())
                ++clear_boxes;
            if (s.has_polygons && s.polygons_empty && !s.has_detections && s.session.empty())
                ++clear_zones;
            if (!s.session.empty()) {
                ++tagged;
                if (s.session != kSession)
                    fail("session tag", "unexpected session '" + s.session + "'");
                if (s.ttl != "2000")
                    fail("result ttl", "metadata ttl='" + s.ttl + "' expected 2000");
            }
        }
        if (zone_push == 1)
            pass("exactly one untagged zone push");
        else
            fail("zone push count", std::to_string(zone_push));
        if (clear_boxes == 1 && clear_zones == 1)
            pass("exactly one untagged box clear + one zone clear at teardown");
        else
            fail("clear counts", "boxes=" + std::to_string(clear_boxes) +
                                     " zones=" + std::to_string(clear_zones));
        if (tagged == 0 && dma_reject.empty())
            fail("result tagging", "results flowed but none carried the session tag");
        else if (tagged > 0)
            pass(std::to_string(tagged) + " tagged result write(s), ttl forwarded");
        else
            skip("result tagging", "no results on this daemon build (see above)");
        std::printf("  observer total events: %zu\n", evs.size());
    } catch (const std::exception& e) {
        fail("observer verdict", e.what());
    }

    if (registered) {
        // retry: the daemon releases the model reference from the just-
        // stopped subscription asynchronously (observed: immediate call
        // after stop() fails with "Failed to unregister")
        bool unregistered = false;
        std::string last_err;
        for (int attempt = 0; attempt < 6 && !unregistered; ++attempt) {
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
            try {
                inf.unregister_model(kModelId);
                unregistered = true;
            } catch (const std::exception& e) {
                last_err = e.what();
            }
        }
        if (unregistered)
            pass("unregister_model");
        else
            fail("unregister_model", last_err);
    }

    std::printf("=== Phase 6 smoke: %s (failures=%d skips=%d) ===\n",
                failures == 0 ? "PASS" : "FAIL", failures, skips);
    return failures == 0 ? 0 : 1;
}
