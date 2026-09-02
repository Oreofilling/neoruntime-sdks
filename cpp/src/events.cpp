// events.cpp — EventClient implementation. See events.hpp.
#include "neoruntime_ipc_sdk/events.hpp"

#include <grpcpp/grpcpp.h>

#include <atomic>
#include <memory>
#include <string>
#include <thread>
#include <utility>

#include "detail/grpc_channel.hpp"
#include "event-bus/event.grpc.pb.h"
#include "event-bus/event.pb.h"

namespace neoruntime_ipc_sdk {

namespace pb = aipc::event;

// File-local: proto Event → public Event. Same decode logic as EventStream::next.
namespace {
Event event_from_proto(const pb::Event& msg) {
    Event ev;
    ev.topic = msg.topic();
    ev.source = msg.source();
    ev.event_id = msg.event_id();
    ev.timestamp_ns = msg.timestamp_ns();
    if (!msg.payload().empty()) {
        try {
            ev.payload = nlohmann::json::parse(msg.payload());
        } catch (...) {
            ev.payload = nlohmann::json{{"raw", std::string(msg.payload())}};
        }
    }
    for (const auto& [k, v] : msg.metadata()) {
        ev.metadata[k] = v;
    }
    return ev;
}
}  // namespace

// ---------------------------------------------------------------------------
// Event
// ---------------------------------------------------------------------------
Event::Event()
    : source(Config::get_app_id()), timestamp_ns(detail::now_ns()) {}

Event::Event(std::string topic_in,
             nlohmann::json payload_in,
             std::string source_in,
             std::string event_id_in,
             std::uint64_t timestamp_ns_in,
             std::map<std::string, std::string> metadata_in)
    : topic(std::move(topic_in)),
      payload(std::move(payload_in)),
      source(source_in.empty() ? Config::get_app_id() : std::move(source_in)),
      event_id(std::move(event_id_in)),
      timestamp_ns(timestamp_ns_in == 0 ? detail::now_ns() : timestamp_ns_in),
      metadata(std::move(metadata_in)) {}

std::string Event::to_json() const { return payload.dump(); }

// ---------------------------------------------------------------------------
// EventStream
// ---------------------------------------------------------------------------
struct EventStream::Impl {
    grpc::ClientContext ctx;
    std::unique_ptr<grpc::ClientReader<pb::Event>> reader;
};

EventStream::EventStream() : impl_(std::make_unique<Impl>()) {}
EventStream::~EventStream() = default;
EventStream::EventStream(EventStream&&) noexcept = default;
EventStream& EventStream::operator=(EventStream&&) noexcept = default;

std::optional<Event> EventStream::next() {
    if (!impl_->reader) return std::nullopt;
    pb::Event msg;
    if (!impl_->reader->Read(&msg)) return std::nullopt;
    return event_from_proto(msg);
}

// ---------------------------------------------------------------------------
// EventClient
// ---------------------------------------------------------------------------
struct EventClient::Impl {
    std::string endpoint;
    std::string app_id;
    std::shared_ptr<grpc::Channel> channel;
    std::unique_ptr<pb::EventBus::Stub> stub;
    // Shared with detached on_event threads so they can stop after close().
    std::shared_ptr<std::atomic<bool>> running =
        std::make_shared<std::atomic<bool>>(true);

    void ensure_connected() {
        if (!stub) {
            channel = detail::make_channel(endpoint);
            stub = pb::EventBus::NewStub(channel);
        }
    }
};

EventClient::EventClient(std::string endpoint)
    : impl_(std::make_unique<Impl>()) {
    impl_->endpoint =
        endpoint.empty() ? Config::get_event_bus_endpoint() : std::move(endpoint);
    impl_->app_id = Config::get_app_id();
}

EventClient::~EventClient() { close(); }
EventClient::EventClient(EventClient&&) noexcept = default;
EventClient& EventClient::operator=(EventClient&&) noexcept = default;

void EventClient::connect() { impl_->ensure_connected(); }

void EventClient::close() {
    // Signal detached on_event threads to stop after their next read returns.
    if (impl_->running) impl_->running->store(false);
    impl_->stub.reset();
    impl_->channel.reset();
}

bool EventClient::connected() const noexcept { return impl_->stub != nullptr; }

std::string EventClient::publish(const std::string& topic,
                                 const nlohmann::json& payload,
                                 bool persistent,
                                 std::optional<std::uint32_t> ttl_ms,
                                 const std::map<std::string, std::string>& metadata) {
    impl_->ensure_connected();
    pb::PublishRequest req;
    auto* ev = req.mutable_event();
    ev->set_topic(topic);
    ev->set_source(impl_->app_id);
    ev->set_timestamp_ns(detail::now_ns());
    ev->set_payload(payload.dump());
    ev->set_payload_type("json");
    for (const auto& [k, v] : metadata) (*ev->mutable_metadata())[k] = v;
    req.set_persistent(persistent);
    req.set_ttl_ms(ttl_ms ? *ttl_ms : 0);

    grpc::ClientContext ctx;
    pb::PublishResponse resp;
    detail::check_grpc(impl_->stub->Publish(&ctx, req, &resp), "Publish");
    detail::require_success(resp.status().success(), resp.status().message(),
                            "Publish");
    return resp.event_id();
}

void EventClient::publish_batch(
    const std::vector<std::pair<std::string, nlohmann::json>>& events,
    bool persistent) {
    impl_->ensure_connected();
    grpc::ClientContext ctx;
    pb::Status resp;
    std::unique_ptr<grpc::ClientWriter<pb::PublishRequest>> writer =
        impl_->stub->PublishBatch(&ctx, &resp);

    for (const auto& [topic, payload] : events) {
        pb::PublishRequest req;
        auto* ev = req.mutable_event();
        ev->set_topic(topic);
        ev->set_source(impl_->app_id);
        ev->set_timestamp_ns(detail::now_ns());
        ev->set_payload(payload.dump());
        ev->set_payload_type("json");
        req.set_persistent(persistent);
        if (!writer->Write(req)) break;  // stream broken; stop writing
    }
    writer->WritesDone();
    detail::check_grpc(writer->Finish(), "PublishBatch");
    detail::require_success(resp.success(), resp.message(), "PublishBatch");
}

EventStream EventClient::subscribe(const std::string& topic,
                                   const std::map<std::string, std::string>& filters,
                                   std::uint32_t queue_size,
                                   bool drop_old) {
    impl_->ensure_connected();
    pb::SubscribeRequest req;
    req.set_topic(topic);
    req.set_subscriber_id(impl_->app_id);
    req.set_queue_size(queue_size);
    req.set_drop_old(drop_old);
    for (const auto& [k, v] : filters) (*req.mutable_filters())[k] = v;

    EventStream stream;
    stream.impl_->reader = impl_->stub->Subscribe(&stream.impl_->ctx, req);
    return stream;
}

void EventClient::on_event(const std::string& topic,
                           std::function<void(const Event&)> callback,
                           const std::map<std::string, std::string>& filters) {
    impl_->ensure_connected();
    // Capture shared handles so the thread is safe even if the client is
    // destroyed before the stream ends.
    auto channel = impl_->channel;
    auto running = impl_->running;
    auto app_id = impl_->app_id;
    std::thread([channel, running, app_id, topic,
                 filters = filters, callback = std::move(callback)]() {
        auto stub = pb::EventBus::NewStub(channel);
        pb::SubscribeRequest req;
        req.set_topic(topic);
        req.set_subscriber_id(app_id);
        req.set_queue_size(100);
        req.set_drop_old(true);
        for (const auto& [k, v] : filters) (*req.mutable_filters())[k] = v;
        grpc::ClientContext ctx;
        std::unique_ptr<grpc::ClientReader<pb::Event>> reader =
            stub->Subscribe(&ctx, req);
        pb::Event msg;
        while (running->load() && reader->Read(&msg)) {
            try {
                callback(event_from_proto(msg));
            } catch (...) {
                // Swallow callback errors (mirrors events.py).
            }
        }
    }).detach();
}

void EventClient::unsubscribe(const std::string& topic) {
    impl_->ensure_connected();
    pb::SubscribeRequest req;
    req.set_topic(topic);
    req.set_subscriber_id(impl_->app_id);
    grpc::ClientContext ctx;
    pb::Status resp;
    detail::check_grpc(impl_->stub->Unsubscribe(&ctx, req, &resp), "Unsubscribe");
    detail::require_success(resp.success(), resp.message(), "Unsubscribe");
}

std::vector<TopicInfo> EventClient::list_topics() {
    impl_->ensure_connected();
    grpc::ClientContext ctx;
    pb::Empty req;
    pb::TopicListResponse resp;
    detail::check_grpc(impl_->stub->ListTopics(&ctx, req, &resp), "ListTopics");
    std::vector<TopicInfo> out;
    out.reserve(resp.topics_size());
    for (const auto& t : resp.topics()) {
        out.push_back(TopicInfo{t.topic(), t.subscriber_count(),
                                t.total_messages(), t.last_message_ts()});
    }
    return out;
}

std::optional<TopicInfo> EventClient::get_topic_info(const std::string& topic) {
    impl_->ensure_connected();
    pb::TopicInfo req;
    req.set_topic(topic);
    grpc::ClientContext ctx;
    pb::TopicInfo resp;
    detail::check_grpc(impl_->stub->GetTopicInfo(&ctx, req, &resp), "GetTopicInfo");
    if (resp.topic().empty()) return std::nullopt;
    return TopicInfo{resp.topic(), resp.subscriber_count(),
                     resp.total_messages(), resp.last_message_ts()};
}

SystemStats EventClient::get_stats() {
    impl_->ensure_connected();
    grpc::ClientContext ctx;
    pb::Empty req;
    pb::SystemStats resp;
    detail::check_grpc(impl_->stub->GetStats(&ctx, req, &resp), "GetStats");
    SystemStats out;
    out.total_subscribers = resp.total_subscribers();
    out.total_topics = resp.total_topics();
    out.uptime_ms = resp.uptime_ms();
    out.topic_stats.reserve(resp.topic_stats_size());
    for (const auto& s : resp.topic_stats()) {
        out.topic_stats.push_back(EventStats{
            s.topic(), s.published_count(), s.delivered_count(),
            s.dropped_count(), s.avg_latency_us()});
    }
    return out;
}

EventStats EventClient::get_topic_stats(const std::string& topic) {
    impl_->ensure_connected();
    pb::TopicInfo req;
    req.set_topic(topic);
    grpc::ClientContext ctx;
    pb::EventStats resp;
    detail::check_grpc(impl_->stub->GetTopicStats(&ctx, req, &resp), "GetTopicStats");
    return EventStats{resp.topic(), resp.published_count(),
                      resp.delivered_count(), resp.dropped_count(),
                      resp.avg_latency_us()};
}

}  // namespace neoruntime_ipc_sdk
