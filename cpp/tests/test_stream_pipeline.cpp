// test_stream_pipeline.cpp — StreamPipeline parity (port of python
// tests/test_p211_stream_pipeline.py). python injects duck-typed fake
// clients; C++ takes concrete InferenceClient*/OverlayClient*, so the fakes
// live one level down on the wire: an in-process aipc.inference gRPC
// service (scripted server-streaming StreamInfer), an in-process
// aipc.event bus recording Publish payloads, and an in-process
// aipc.camera CameraControl recording UpdateAiOverlay configs. The
// endpoint env vars route any owned clients to the same fakes.
//
// What the wire can't carry (python asserted it on the fake objects):
//  * subscribe()'s client-side queue_size knob (never leaves the process)
//  * "closed" on owned clients at stop() — channel teardown is not a bus
//    message; the owned-client test asserts the clients were created and
//    used against the env endpoints instead.
//
// Stopping discipline: stop() cancels the subscription through iterator_,
// which only exists once the worker completed subscribe(). Every test that
// stops a live stream first proves the worker got that far (a pushed
// result observed via results_seen / the bus), so the join can never wait
// on an uncancellable worker.
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <grpcpp/grpcpp.h>

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <cstdint>
#include <deque>
#include <functional>
#include <future>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <unistd.h>
#include <utility>
#include <vector>

#include "ai-runtime/inference.grpc.pb.h"
#include "camera-daemon/camera.grpc.pb.h"
#include "event-bus/event.grpc.pb.h"

#include "neoruntime_ipc_sdk/inference.hpp"
#include "neoruntime_ipc_sdk/overlay.hpp"
#include "neoruntime_ipc_sdk/stream_pipeline.hpp"

using namespace neoruntime_ipc_sdk;

namespace pb_inf = aipc::inference;
namespace pb_ev = aipc::event;
namespace pb_cam = aipc::camera;

namespace {

template <typename Pred>
bool wait_until(Pred pred, double timeout_s = 5.0) {
    const auto deadline = std::chrono::steady_clock::now() +
                          std::chrono::duration<double>(timeout_s);
    while (std::chrono::steady_clock::now() < deadline) {
        if (pred()) return true;
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    return pred();
}

std::int64_t wall_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::system_clock::now().time_since_epoch())
        .count();
}

// ---- fakes --------------------------------------------------------------------------

struct SubRecord {
    std::string stream, model, session;
    std::uint32_t fps = 0;
    bool raw = false;
};

// Scripted StreamInfer: results are queued from the test thread and written
// to the subscriber in order; End closes the stream cleanly, Break fails it
// ("stream broken"), and a client-side cancel is observed. While no item is
// scripted the handler idles in 20ms slices so cancel is seen promptly.
struct FakeInference final : public pb_inf::InferenceService::Service {
    struct Item {
        enum class Kind { kResult, kEnd, kBreak } kind = Kind::kResult;
        std::uint64_t seq = 0;
        std::vector<std::pair<std::string, float>> dets;  // (label, score)
        std::int64_t skew_us = 0;
        std::chrono::milliseconds stamp_age{3};  // timestamp_ns = now - age
    };

    std::mutex mu;
    std::condition_variable cv;
    std::vector<SubRecord> subs;
    std::deque<Item> pending;
    bool saw_cancel = false;

    void push(std::uint64_t seq, std::vector<std::pair<std::string, float>> dets,
              std::int64_t skew_us = 0) {
        {
            std::lock_guard<std::mutex> lock(mu);
            Item item;
            item.kind = Item::Kind::kResult;
            item.seq = seq;
            item.dets = std::move(dets);
            item.skew_us = skew_us;
            pending.push_back(std::move(item));
        }
        cv.notify_one();
    }
    void break_stream() {
        {
            std::lock_guard<std::mutex> lock(mu);
            pending.push_back(Item{Item::Kind::kBreak, 0, {}, 0, {}});
        }
        cv.notify_one();
    }

    grpc::Status StreamInfer(grpc::ServerContext* ctx,
                             const pb_inf::StreamInferRequest* req,
                             grpc::ServerWriter<pb_inf::StreamInferResponse>* writer) override {
        {
            std::lock_guard<std::mutex> lock(mu);
            subs.push_back(SubRecord{req->stream_id(), req->model_id(), req->session_id(),
                                     req->fps_limit(), req->raw_output_only()});
        }
        for (;;) {
            Item item;
            {
                std::unique_lock<std::mutex> lock(mu);
                cv.wait_for(lock, std::chrono::milliseconds(20),
                            [this] { return !pending.empty(); });
                if (ctx->IsCancelled()) {
                    saw_cancel = true;
                    return {grpc::StatusCode::CANCELLED, "cancelled"};
                }
                if (pending.empty()) continue;
                item = std::move(pending.front());
                pending.pop_front();
            }
            if (item.kind == Item::Kind::kBreak)
                return {grpc::StatusCode::INTERNAL, "stream broken"};
            pb_inf::StreamInferResponse resp;
            resp.set_frame_sequence(item.seq);
            resp.set_timestamp_ns(static_cast<std::uint64_t>(
                wall_ns() - std::chrono::duration_cast<std::chrono::nanoseconds>(item.stamp_age)
                                .count()));
            resp.set_skew_us(item.skew_us);
            resp.mutable_status()->set_success(true);
            for (const auto& [label, score] : item.dets) {
                pb_inf::Detection* det = resp.mutable_post_result()->add_detections();
                det->set_label(label);
                det->set_confidence(score);
                pb_inf::BBox* bb = det->mutable_bbox();
                bb->set_x(0.1);
                bb->set_y(0.2);
                bb->set_w(0.3);
                bb->set_h(0.4);
            }
            if (!writer->Write(resp))
                return {grpc::StatusCode::CANCELLED, "subscriber went away"};
        }
    }
};

// One recorded overlay event (the wire form of python's ov.calls entry).
struct BusRecord {
    std::string topic;
    nlohmann::json payload;
    std::map<std::string, std::string> metadata;
};

struct FakeBus final : public pb_ev::EventBus::Service {
    std::mutex mu;
    std::vector<BusRecord> events;
    int fail_next = 0;
    // when valid, Publish parks here (the observability test paces the
    // worker through its annotate). One shared future: promise::get_future
    // may only be called once, so it is captured when the test arms it.
    std::shared_future<void> gate;

    grpc::Status Publish(grpc::ServerContext* ctx, const pb_ev::PublishRequest* req,
                         pb_ev::PublishResponse* resp) override {
        if (gate.valid()) {
            while (gate.wait_for(std::chrono::milliseconds(10)) !=
                   std::future_status::ready) {
                if (ctx->IsCancelled()) return {grpc::StatusCode::CANCELLED, "cancelled"};
            }
        }
        {
            std::lock_guard<std::mutex> lock(mu);
            if (fail_next > 0) {
                --fail_next;
                return {grpc::StatusCode::INTERNAL, "overlay down"};
            }
            BusRecord rec;
            rec.topic = req->event().topic();
            try {
                rec.payload = nlohmann::json::parse(req->event().payload());
            } catch (...) {
                rec.payload = nlohmann::json();
            }
            for (const auto& [k, v] : req->event().metadata()) rec.metadata[k] = v;
            events.push_back(std::move(rec));
        }
        resp->mutable_status()->set_success(true);
        resp->set_event_id("evt");
        return grpc::Status::OK;
    }
};

struct FakeCamera final : public pb_cam::CameraControl::Service {
    std::mutex mu;
    std::vector<pb_cam::AiOverlayConfig> applied;

    grpc::Status UpdateAiOverlay(grpc::ServerContext*, const pb_cam::AiOverlayConfig* req,
                                 pb_cam::Status* resp) override {
        {
            std::lock_guard<std::mutex> lock(mu);
            applied.push_back(*req);
        }
        resp->set_success(true);
        resp->set_message("");
        return grpc::Status::OK;
    }
};

// Three services, one server, three sockets; the endpoint env vars route
// SDK-owned clients to the same fakes the injected ones use.
struct Rig {
    FakeInference inf;
    FakeBus bus;
    FakeCamera cam;
    std::unique_ptr<grpc::Server> server;
    std::string inf_path, bus_path, cam_path;

    void start(const char* tag) {
        const std::string base = "/tmp/ne503-test-sp-" + std::string(tag);
        inf_path = base + "-inf.sock";
        bus_path = base + "-bus.sock";
        cam_path = base + "-cam.sock";
        ::unlink(inf_path.c_str());
        ::unlink(bus_path.c_str());
        ::unlink(cam_path.c_str());
        grpc::ServerBuilder builder;
        builder.AddListeningPort("unix://" + inf_path, grpc::InsecureServerCredentials());
        builder.AddListeningPort("unix://" + bus_path, grpc::InsecureServerCredentials());
        builder.AddListeningPort("unix://" + cam_path, grpc::InsecureServerCredentials());
        builder.RegisterService(&inf);
        builder.RegisterService(&bus);
        builder.RegisterService(&cam);
        server = builder.BuildAndStart();
        ::setenv("AI_RUNTIME_ENDPOINT", ("unix://" + inf_path).c_str(), 1);
        ::setenv("EVENT_BUS_ENDPOINT", ("unix://" + bus_path).c_str(), 1);
        ::setenv("CAMERA_CONTROL_ENDPOINT", ("unix://" + cam_path).c_str(), 1);
    }
    ~Rig() {
        if (server) server->Shutdown();
        ::unlink(inf_path.c_str());
        ::unlink(bus_path.c_str());
        ::unlink(cam_path.c_str());
    }
    InferenceClient inference() { return InferenceClient("unix://" + inf_path); }
    OverlayClient overlay() { return OverlayClient("unix://" + cam_path); }
};

// ---- value helpers ----------------------------------------------------------------

DetectedObject obj(const std::string& label = "person", float score = 0.9f) {
    DetectedObject out;
    out.label = label;
    out.score = score;
    out.bbox = BoundingBox{0.1f, 0.2f, 0.3f, 0.4f};
    out.class_id = 1;
    return out;
}

std::vector<OverlayPolygon> zones() {
    OverlayPolygon poly;
    poly.points = {OverlayPoint{0.0f, 0.0f}, OverlayPoint{1.0f, 0.0f},
                   OverlayPoint{1.0f, 1.0f}};
    poly.label = "yard";
    return {poly};
}

// python's kwargs map onto C++ ctor positionals; this is the ladder's
// read-friendly form. make_pipe returns the prvalue directly (C++17
// elision — StreamPipeline is neither copyable nor movable); make_pipe_ptr
// constructs in place for unique_ptr ownership.
struct PipeSpec {
    std::string stream = "third";
    std::string model = "m";
    std::uint32_t fps = 10;
    std::string session_id;
    std::optional<std::uint32_t> queue_size = std::nullopt;
    bool draw = true;
    std::optional<OverlayConfig> overlay_config = std::nullopt;
    std::optional<std::uint32_t> ttl_ms = std::nullopt;
    double min_score = 0.0;
    std::optional<std::vector<std::string>> labels = std::nullopt;
    std::optional<std::vector<OverlayPolygon>> polygons = std::nullopt;
    StreamPipeline::OnResult on_result = nullptr;
    std::size_t result_queue_size = 100;
};

StreamPipeline make_pipe(const PipeSpec& s, InferenceClient* inference,
                         OverlayClient* overlay) {
    return StreamPipeline(s.stream, s.model, s.fps, s.session_id,
                          /*raw_output_only=*/false,
                          /*max_consecutive_failures=*/std::nullopt, s.queue_size, s.draw,
                          s.overlay_config, s.ttl_ms, s.min_score, s.labels, s.polygons,
                          s.on_result, s.result_queue_size, inference, overlay);
}

std::unique_ptr<StreamPipeline> make_pipe_ptr(const PipeSpec& s, InferenceClient* inference,
                                              OverlayClient* overlay) {
    return std::make_unique<StreamPipeline>(
        s.stream, s.model, s.fps, s.session_id,
        /*raw_output_only=*/false,
        /*max_consecutive_failures=*/std::nullopt, s.queue_size, s.draw, s.overlay_config,
        s.ttl_ms, s.min_score, s.labels, s.polygons, s.on_result, s.result_queue_size,
        inference, overlay);
}

std::vector<BusRecord> bus_snapshot(Rig& rig) {
    std::lock_guard<std::mutex> lock(rig.bus.mu);
    return rig.bus.events;
}

// payload predicates (python matched on the fake's call args). The clear
// payloads: boxes = annotate(detections=[]) → {num_detections: 0,
// detections: []}; zones = annotate(polygons=[]) → {polygons: []}.
bool is_clear_boxes(const BusRecord& rec) {
    return rec.topic == "inference/third" && rec.payload.contains("num_detections") &&
           rec.payload["num_detections"] == 0 && !rec.payload.contains("polygons");
}
bool is_clear_zones(const BusRecord& rec) {
    return rec.topic == "inference/third" && rec.payload.contains("polygons") &&
           rec.payload["polygons"].empty() && !rec.payload.contains("num_detections");
}

}  // namespace

// ---- TestStartAndLifecycle ----------------------------------------------------------

TEST(StartAndLifecycle, StartForwardsSubscribeParamsAppliesConfigPushesZones) {
    Rig rig;
    rig.start("fwd");
    InferenceClient inf_client = rig.inference();
    OverlayClient ov_client = rig.overlay();
    OverlayConfig cfg;
    cfg.enabled = true;
    cfg.show_label = false;
    PipeSpec spec;
    spec.model = "yolov5n";
    spec.fps = 7;
    spec.session_id = "s1";
    spec.queue_size = 11;
    spec.overlay_config = cfg;
    spec.polygons = zones();
    StreamPipeline p = make_pipe(spec, &inf_client, &ov_client);
    p.start();
    try {
        // overlay config + static zones go out synchronously in start()
        ASSERT_EQ(rig.cam.applied.size(), 1u);
        EXPECT_TRUE(rig.cam.applied[0].enabled());
        EXPECT_FALSE(rig.cam.applied[0].show_label());
        const std::vector<BusRecord> after_start = bus_snapshot(rig);
        ASSERT_EQ(after_start.size(), 1u);
        EXPECT_TRUE(after_start[0].payload.contains("polygons"));
        EXPECT_EQ(after_start[0].payload["polygons"].size(), 1u);

        // subscribe runs on the worker: wait for it, then check the wire
        ASSERT_TRUE(wait_until([&] { return rig.inf.subs.size() == 1; }));
        EXPECT_EQ(rig.inf.subs[0].stream, "third");
        EXPECT_EQ(rig.inf.subs[0].model, "yolov5n");
        EXPECT_EQ(rig.inf.subs[0].fps, 7u);
        EXPECT_EQ(rig.inf.subs[0].session, "s1");
        EXPECT_FALSE(rig.inf.subs[0].raw);
        // (subscribe()'s queue_size knob is client-side — not on the wire)

        rig.inf.push(1, {{"person", 0.9f}});
        std::optional<std::pair<std::uint64_t, InferenceResult>> item = p.next_result();
        ASSERT_TRUE(item.has_value());
        EXPECT_EQ(item->first, 1u);
        EXPECT_EQ(item->second.frame_sequence, 1u);
        ASSERT_TRUE(wait_until([&] { return p.status().results_seen == 1; }));
        EXPECT_TRUE(wait_until([&] { return p.status().results_annotated == 1; }));
    } catch (...) {
        p.stop();
        throw;
    }
    p.stop();
}

TEST(StartAndLifecycle, TtlForwardedToAnnotateResult) {
    Rig rig;
    rig.start("ttl");
    InferenceClient inf_client = rig.inference();
    OverlayClient ov_client = rig.overlay();
    PipeSpec spec;
    spec.ttl_ms = 800;
    StreamPipeline p = make_pipe(spec, &inf_client, &ov_client);
    p.start();
    rig.inf.push(2, {{"person", 0.9f}});
    EXPECT_TRUE(wait_until([&] { return !bus_snapshot(rig).empty(); }));
    const std::vector<BusRecord> events = bus_snapshot(rig);  // pre-stop: just the result
    ASSERT_EQ(events.size(), 1u);
    EXPECT_EQ(events[0].topic, "inference/third");
    EXPECT_EQ(events[0].metadata.at("result_ttl_ms"), "800");
    EXPECT_EQ(events[0].payload["detections"].size(), 1u);
    p.stop();
}

TEST(StartAndLifecycle, DrawFalseNeverTouchesOverlay) {
    Rig rig;
    rig.start("draw-off");
    InferenceClient inf_client = rig.inference();
    OverlayClient ov_client = rig.overlay();
    PipeSpec spec;
    spec.draw = false;
    spec.polygons = zones();  // draw=false suppresses static zones too
    StreamPipeline p = make_pipe(spec, &inf_client, &ov_client);
    p.start();
    rig.inf.push(1, {{"person", 0.9f}});
    std::optional<std::pair<std::uint64_t, InferenceResult>> item = p.next_result();
    ASSERT_TRUE(item.has_value());
    EXPECT_EQ(item->first, 1u);
    ASSERT_TRUE(wait_until([&] { return p.status().results_seen == 1; }));
    const StreamPipelineStatus st = p.status();
    EXPECT_EQ(st.results_annotated, 0u);
    EXPECT_TRUE(rig.cam.applied.empty());
    EXPECT_TRUE(bus_snapshot(rig).empty());
    p.stop();
    EXPECT_TRUE(bus_snapshot(rig).empty());  // nothing was drawn: nothing to clear
}

TEST(StartAndLifecycle, StartTwiceAndRestartAfterStopBothRaise) {
    Rig rig;
    rig.start("twice");
    InferenceClient inf_client = rig.inference();
    OverlayClient ov_client = rig.overlay();
    StreamPipeline p = make_pipe(PipeSpec{}, &inf_client, &ov_client);
    p.start();
    EXPECT_THROW(p.start(), std::runtime_error);
    // prove the worker finished subscribe() before stopping (see file note)
    rig.inf.push(1, {});
    ASSERT_TRUE(wait_until([&] { return p.status().results_seen == 1; }));
    p.stop();
    EXPECT_THROW(p.start(), std::runtime_error);  // single-shot: build a new one
}

TEST(StartAndLifecycle, StopBeforeStartIsANoop) {
    StreamPipeline p = make_pipe(PipeSpec{}, nullptr, nullptr);
    p.stop();  // must not raise
}

TEST(StartAndLifecycle, StopClosesClientsItCreated) {
    // python monkeypatches the client factories; here the endpoint env vars
    // route the internally-created clients to the fakes. Channel close at
    // stop() is not bus-observable, so the port asserts the owned clients
    // were created and used against those endpoints, and that teardown
    // still went through the owned overlay (the box clear hit the bus).
    Rig rig;
    rig.start("owned");
    StreamPipeline p = make_pipe(PipeSpec{}, nullptr, nullptr);
    p.start();
    ASSERT_TRUE(wait_until([&] { return rig.inf.subs.size() == 1; }));
    rig.inf.push(1, {});
    ASSERT_TRUE(wait_until([&] { return p.status().results_seen == 1; }));
    p.stop();
    EXPECT_FALSE(p.status().running);
    EXPECT_TRUE(wait_until([&] {
        for (const BusRecord& rec : bus_snapshot(rig))
            if (is_clear_boxes(rec)) return true;
        return false;
    }));
}

TEST(StartAndLifecycle, StopLeavesInjectedClientsOpenAndIsIdempotent) {
    Rig rig;
    rig.start("injected");
    InferenceClient inf_client = rig.inference();
    OverlayClient ov_client = rig.overlay();
    StreamPipeline p = make_pipe(PipeSpec{}, &inf_client, &ov_client);
    p.start();
    rig.inf.push(1, {{"person", 0.9f}});
    ASSERT_TRUE(wait_until([&] { return p.status().results_seen == 1; }));
    p.stop();
    p.stop();  // second stop: no error, no double-clear
    // injected client is still usable after the pipeline stopped — with a
    // NON-empty list so the publish is distinguishable from a box clear
    ov_client.annotate("third", std::vector<DetectedObject>{obj()});
    EXPECT_TRUE(wait_until([&] { return bus_snapshot(rig).size() == 3; }));
    const std::vector<BusRecord> events = bus_snapshot(rig);
    int clears = 0, draws = 0;
    for (const BusRecord& rec : events) {
        if (is_clear_boxes(rec)) ++clears;
        if (rec.payload.contains("num_detections") && rec.payload["num_detections"] == 1)
            ++draws;
        EXPECT_FALSE(is_clear_zones(rec));  // no polygons configured: none cleared
    }
    EXPECT_EQ(clears, 1);  // exactly one clear, from the first stop()
    EXPECT_EQ(draws, 2);   // the pipeline's result + our own annotate
}

TEST(StartAndLifecycle, StopClearsStaticPolygonsToo) {
    Rig rig;
    rig.start("zones-clear");
    InferenceClient inf_client = rig.inference();
    OverlayClient ov_client = rig.overlay();
    PipeSpec spec;
    spec.polygons = zones();
    StreamPipeline p = make_pipe(spec, &inf_client, &ov_client);
    p.start();
    rig.inf.push(1, {{"person", 0.9f}});
    ASSERT_TRUE(wait_until([&] { return p.status().results_seen == 1; }));
    p.stop();
    const std::vector<BusRecord> events = bus_snapshot(rig);
    bool boxes = false, zs = false;
    for (const BusRecord& rec : events) {
        if (is_clear_boxes(rec)) boxes = true;
        if (is_clear_zones(rec)) zs = true;
    }
    EXPECT_TRUE(boxes);
    EXPECT_TRUE(zs);
}

TEST(StartAndLifecycle, ContextManagerStopsOnExit) {
    // python's `with StreamPipeline(...) as p`; C++: the destructor is the
    // context exit, so observe its teardown on the rig after the reset.
    // The result carries a detection so its publish can't be mistaken for
    // the destructor's box clear.
    Rig rig;
    rig.start("raii");
    InferenceClient inf_client = rig.inference();
    OverlayClient ov_client = rig.overlay();
    auto p = make_pipe_ptr(PipeSpec{}, &inf_client, &ov_client);
    p->start();
    rig.inf.push(1, {{"person", 0.9f}});
    std::optional<std::pair<std::uint64_t, InferenceResult>> item = p->next_result();
    ASSERT_TRUE(item.has_value());
    EXPECT_EQ(item->first, 1u);
    EXPECT_TRUE(wait_until([&] { return p->status().results_annotated == 1; }));
    EXPECT_TRUE(p->status().running);
    p.reset();  // ~StreamPipeline stops
    EXPECT_TRUE(wait_until([&] {
        for (const BusRecord& rec : bus_snapshot(rig))
            if (is_clear_boxes(rec)) return true;
        return false;
    })) << "destructor did not clear boxes";
}

TEST(StartAndLifecycle, ResultsBeforeStartRaises) {
    StreamPipeline p = make_pipe(PipeSpec{}, nullptr, nullptr);
    EXPECT_THROW(p.next_result(), std::runtime_error);
}

TEST(StartAndLifecycle, InvalidTtlRejectedAtConstruction) {
    // A bad ttl_ms would otherwise surface only as annotate_errors inside
    // the worker thread: boxes silently never drawn. Fail at construction.
    // python also probes non-int ttl values ("soon", 50.5, True) — the C++
    // optional<uint32_t> parameter makes those ill-typed instead.
    PipeSpec spec;
    spec.ttl_ms = 0;
    EXPECT_THROW(make_pipe(spec, nullptr, nullptr), std::invalid_argument);
    StreamPipeline good = make_pipe(PipeSpec{}, nullptr, nullptr);  // no ttl: fine
    (void)good;
    PipeSpec ok;
    ok.ttl_ms = 100;
    StreamPipeline positive = make_pipe(ok, nullptr, nullptr);  // positive int passes
    (void)positive;
    PipeSpec bad_queue;
    bad_queue.result_queue_size = 0;
    EXPECT_THROW(make_pipe(bad_queue, nullptr, nullptr), std::invalid_argument);
    PipeSpec bad_score;
    bad_score.min_score = 1.5;
    EXPECT_THROW(make_pipe(bad_score, nullptr, nullptr), std::invalid_argument);
}

TEST(StartAndLifecycle, StopFromOnResultClearsBoxesWithoutSelfJoin) {
    // stop() from the hook runs on the worker thread: the self-join must be
    // skipped and the clear annotate still happens.
    Rig rig;
    rig.start("self-stop");
    InferenceClient inf_client = rig.inference();
    OverlayClient ov_client = rig.overlay();
    StreamPipeline p = make_pipe(PipeSpec{}, &inf_client, &ov_client);
    StreamPipeline* pp = &p;
    p.on_result = [pp](const InferenceResult&) -> std::optional<InferenceResult> {
        pp->stop();
        return std::nullopt;  // dropped: nothing drawn after the clear
    };
    p.start();
    rig.inf.push(1, {{"person", 0.9f}});
    EXPECT_TRUE(wait_until([&] { return !p.status().running; })) << "worker did not stop";
    EXPECT_EQ(p.status().last_error, "");
    bool cleared = false;
    for (const BusRecord& rec : bus_snapshot(rig))
        if (is_clear_boxes(rec)) cleared = true;
    EXPECT_TRUE(cleared);
    EXPECT_FALSE(p.next_result().has_value());  // dropped; stop marker seen
}

// ---- TestProcessing -----------------------------------------------------------------

TEST(Processing, MinScoreAndLabelsFilterOnlyTheDrawing) {
    Rig rig;
    rig.start("filter");
    InferenceClient inf_client = rig.inference();
    OverlayClient ov_client = rig.overlay();
    PipeSpec spec;
    spec.min_score = 0.5;
    spec.labels = std::vector<std::string>{"person"};
    StreamPipeline p = make_pipe(spec, &inf_client, &ov_client);
    p.start();
    try {
        rig.inf.push(1, {{"person", 0.9f}, {"car", 0.8f}, {"person", 0.3f}});
        rig.inf.push(2, {{"car", 0.9f}});  // fully filtered
        std::optional<std::pair<std::uint64_t, InferenceResult>> r1 = p.next_result();
        std::optional<std::pair<std::uint64_t, InferenceResult>> r2 = p.next_result();
        ASSERT_TRUE(r1.has_value());
        ASSERT_TRUE(r2.has_value());
        EXPECT_EQ(r1->second.objects.size(), 3u);  // queue sees the UNFILTERED result
        EXPECT_EQ(r2->second.objects.size(), 1u);
        EXPECT_TRUE(wait_until([&] {
            int n = 0;
            for (const BusRecord& rec : bus_snapshot(rig))
                if (rec.payload.contains("num_detections")) ++n;
            return n == 2;
        }));
        std::vector<BusRecord> draw_calls;
        for (const BusRecord& rec : bus_snapshot(rig))
            if (rec.payload.contains("num_detections")) draw_calls.push_back(rec);
        ASSERT_EQ(draw_calls.size(), 2u);
        EXPECT_EQ(draw_calls[0].payload["num_detections"], 1);
        EXPECT_EQ(draw_calls[0].payload["detections"][0]["label"], "person");
        EXPECT_NEAR(draw_calls[0].payload["detections"][0]["confidence"].get<double>(), 0.9,
                    1e-6);
        // fully-filtered still publishes an empty list → clears stale boxes
        EXPECT_EQ(draw_calls[1].payload["num_detections"], 0);
    } catch (...) {
        p.stop();
        throw;
    }
    p.stop();
}

TEST(Processing, OnResultReplacesOrDrops) {
    Rig rig;
    rig.start("hook");
    InferenceClient inf_client = rig.inference();
    OverlayClient ov_client = rig.overlay();
    InferenceResult replacement;
    replacement.frame_sequence = 99;
    replacement.objects.push_back(obj());
    std::vector<std::uint64_t> seen;
    std::atomic<int> hook_calls{0};
    PipeSpec spec;
    spec.on_result = [&](const InferenceResult& r) -> std::optional<InferenceResult> {
        seen.push_back(r.frame_sequence);
        hook_calls.fetch_add(1, std::memory_order_release);  // publish `seen`
        if (r.frame_sequence == 1) return std::optional<InferenceResult>(replacement);
        return std::nullopt;
    };
    StreamPipeline p = make_pipe(spec, &inf_client, &ov_client);
    p.start();
    try {
        rig.inf.push(1, {});
        rig.inf.push(2, {});
        std::optional<std::pair<std::uint64_t, InferenceResult>> got = p.next_result();
        ASSERT_TRUE(got.has_value());
        EXPECT_EQ(got->second.frame_sequence, 99u);  // replacement flows onward
        EXPECT_EQ(got->second.objects.size(), 1u);
        ASSERT_TRUE(wait_until([&] {
            return hook_calls.load(std::memory_order_acquire) == 2;
        }));
        EXPECT_EQ(seen, (std::vector<std::uint64_t>{1, 2}));
        EXPECT_TRUE(wait_until([&] { return p.status().results_seen == 2; }));
        EXPECT_TRUE(wait_until([&] { return p.status().results_annotated == 1; }));
    } catch (...) {
        p.stop();
        throw;
    }
    p.stop();
}

TEST(Processing, AnnotateErrorCountedAndNonFatal) {
    Rig rig;
    rig.start("annotate-fail");
    InferenceClient inf_client = rig.inference();
    OverlayClient ov_client = rig.overlay();
    StreamPipeline p = make_pipe(PipeSpec{}, &inf_client, &ov_client);
    p.start();
    try {
        rig.bus.fail_next = 1;
        rig.inf.push(1, {{"person", 0.9f}});
        rig.inf.push(2, {{"person", 0.9f}});
        std::optional<std::pair<std::uint64_t, InferenceResult>> a = p.next_result();
        std::optional<std::pair<std::uint64_t, InferenceResult>> b = p.next_result();
        ASSERT_TRUE(a.has_value());
        ASSERT_TRUE(b.has_value());
        EXPECT_EQ(a->first, 1u);  // both results still reach the app
        EXPECT_EQ(b->first, 2u);
        EXPECT_TRUE(wait_until([&] { return p.status().annotate_errors == 1; }));
        const StreamPipelineStatus st = p.status();
        EXPECT_EQ(st.results_seen, 2u);
        EXPECT_EQ(st.results_annotated, 1u);
        EXPECT_NE(st.last_error.find("overlay down"), std::string::npos);
    } catch (...) {
        p.stop();
        throw;
    }
    p.stop();
}

TEST(Processing, StreamErrorRecordedAndResultsEnd) {
    Rig rig;
    rig.start("stream-break");
    InferenceClient inf_client = rig.inference();
    OverlayClient ov_client = rig.overlay();
    StreamPipeline p = make_pipe(PipeSpec{}, &inf_client, &ov_client);
    p.start();
    rig.inf.push(1, {});
    rig.inf.break_stream();
    std::optional<std::pair<std::uint64_t, InferenceResult>> item = p.next_result();
    ASSERT_TRUE(item.has_value());
    EXPECT_EQ(item->first, 1u);
    // the worker's failure ends the app-side queue too (nullopt, not a hang)
    EXPECT_FALSE(p.next_result().has_value());
    EXPECT_TRUE(wait_until([&] { return !p.status().running; }));
    EXPECT_NE(p.status().last_error.find("stream broken"), std::string::npos);
}

TEST(Processing, ResultQueueDropOldest) {
    Rig rig;
    rig.start("drop-oldest");
    InferenceClient inf_client = rig.inference();
    OverlayClient ov_client = rig.overlay();
    PipeSpec spec;
    spec.draw = false;
    spec.result_queue_size = 2;
    StreamPipeline p = make_pipe(spec, &inf_client, &ov_client);
    p.start();
    try {
        for (std::uint64_t seq = 1; seq <= 4; ++seq) rig.inf.push(seq, {});
        EXPECT_TRUE(wait_until([&] { return p.status().result_queue_drops == 2; }));
        std::optional<std::pair<std::uint64_t, InferenceResult>> a = p.next_result();
        std::optional<std::pair<std::uint64_t, InferenceResult>> b = p.next_result();
        ASSERT_TRUE(a.has_value());
        ASSERT_TRUE(b.has_value());
        EXPECT_EQ(a->first, 3u);  // newest kept
        EXPECT_EQ(b->first, 4u);
    } catch (...) {
        p.stop();
        throw;
    }
    p.stop();
    EXPECT_FALSE(p.next_result().has_value());
}

TEST(Processing, StatusCarriesSubscribeObservability) {
    Rig rig;
    rig.start("observability");
    auto gate = std::make_shared<std::promise<void>>();
    rig.bus.gate = gate->get_future().share();  // Publish parks until released
    InferenceClient inf_client = rig.inference();
    OverlayClient ov_client = rig.overlay();
    PipeSpec spec;
    spec.queue_size = 1;  // pump queue overflows while the worker parks in Publish
    StreamPipeline p = make_pipe(spec, &inf_client, &ov_client);
    p.start();
    try {
        // 5 frames: the worker parks annotating frame 1, so the pump's
        // capacity-1 queue sheds the rest (>= 3 drops in every interleaving)
        for (std::uint64_t seq = 1; seq <= 5; ++seq)
            rig.inf.push(seq, {}, /*skew_us=*/1234);
        EXPECT_TRUE(wait_until([&] { return p.status().subscribe_dropped >= 2; }));
        const StreamPipelineStatus st = p.status();
        EXPECT_TRUE(st.running);
        EXPECT_EQ(st.last_skew_us, 1234);   // scripted on every response
        EXPECT_DOUBLE_EQ(st.avg_skew_us, 1234.0);
        EXPECT_GT(st.last_latency_ms, 0.0);  // stamp = now - 3ms
    } catch (...) {
        gate->set_value();
        p.stop();
        throw;
    }
    gate->set_value();
    p.stop();
}

// ---- TestSessionTagForwarding ----------------------------------------------------------
// P2-13: result annotations carry the pipeline's session_id; operator
// writes (static zones at start, box/zone clears at stop) stay untagged.

TEST(SessionTagForwarding, ResultAnnotationsCarrySessionId) {
    Rig rig;
    rig.start("tag-result");
    InferenceClient inf_client = rig.inference();
    OverlayClient ov_client = rig.overlay();
    PipeSpec spec;
    spec.session_id = "s1";
    StreamPipeline p = make_pipe(spec, &inf_client, &ov_client);
    p.start();
    try {
        rig.inf.push(1, {{"person", 0.9f}});
        EXPECT_TRUE(wait_until([&] { return !bus_snapshot(rig).empty(); }));
        const std::vector<BusRecord> events = bus_snapshot(rig);  // pre-stop
        ASSERT_EQ(events.size(), 1u);
        ASSERT_EQ(events[0].metadata.count("session_id"), 1u);
        EXPECT_EQ(events[0].metadata.at("session_id"), "s1");
    } catch (...) {
        p.stop();
        throw;
    }
    p.stop();
}

TEST(SessionTagForwarding, FilterBranchAnnotateCarriesSessionId) {
    Rig rig;
    rig.start("tag-filter");
    InferenceClient inf_client = rig.inference();
    OverlayClient ov_client = rig.overlay();
    PipeSpec spec;
    spec.session_id = "s1";
    spec.min_score = 0.5;
    StreamPipeline p = make_pipe(spec, &inf_client, &ov_client);
    p.start();
    try {
        rig.inf.push(1, {{"person", 0.9f}});
        EXPECT_TRUE(wait_until([&] { return !bus_snapshot(rig).empty(); }));
        const std::vector<BusRecord> events = bus_snapshot(rig);  // pre-stop
        ASSERT_EQ(events.size(), 1u);
        ASSERT_EQ(events[0].metadata.count("session_id"), 1u);
        EXPECT_EQ(events[0].metadata.at("session_id"), "s1");
        EXPECT_EQ(events[0].payload["num_detections"], 1);
    } catch (...) {
        p.stop();
        throw;
    }
    p.stop();
}

TEST(SessionTagForwarding, StaticZonesAndTeardownClearsStayUntagged) {
    Rig rig;
    rig.start("tag-zones");
    InferenceClient inf_client = rig.inference();
    OverlayClient ov_client = rig.overlay();
    PipeSpec spec;
    spec.session_id = "s1";
    spec.polygons = zones();
    StreamPipeline p = make_pipe(spec, &inf_client, &ov_client);
    p.start();
    rig.inf.push(1, {{"person", 0.9f}});
    EXPECT_TRUE(wait_until([&] {
        for (const BusRecord& rec : bus_snapshot(rig))
            if (rec.payload.contains("num_detections") && rec.payload["num_detections"] == 1)
                return true;
        return false;
    }));
    p.stop();
    // the zone push at start() and both clears at stop() are operator
    // writes: untagged, never the session's tag
    int tagged = 0;
    for (const BusRecord& rec : bus_snapshot(rig))
        if (rec.metadata.count("session_id")) ++tagged;
    EXPECT_EQ(tagged, 1);  // exactly one tagged write: the result
}

TEST(SessionTagForwarding, PipelineWithoutSessionTagsNothing) {
    Rig rig;
    rig.start("tag-none");
    InferenceClient inf_client = rig.inference();
    OverlayClient ov_client = rig.overlay();
    StreamPipeline p = make_pipe(PipeSpec{}, &inf_client, &ov_client);
    p.start();
    try {
        rig.inf.push(1, {{"person", 0.9f}});
        EXPECT_TRUE(wait_until([&] { return !bus_snapshot(rig).empty(); }));
        for (const BusRecord& rec : bus_snapshot(rig))
            EXPECT_EQ(rec.metadata.count("session_id"), 0u);
    } catch (...) {
        p.stop();
        throw;
    }
    p.stop();
}
