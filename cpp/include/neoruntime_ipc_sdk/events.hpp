// events.hpp — EventBus gRPC client. 1:1 port of events.py.
//
// Transport: sync gRPC stubs over UDS. JSON payloads use nlohmann::json (the
// C++ analogue of Python dict + json.dumps/loads). Streaming RPCs (Subscribe,
// PublishBatch) use grpc::ClientReader/ClientWriter.
#pragma once

#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

namespace neoruntime_ipc_sdk {

// An event: a JSON payload routed under a topic. Outbound events auto-fill
// `source` (from APP_ID) and `timestamp_ns` (now) when left empty, matching
// Python's Event.__post_init__.
struct Event {
    std::string topic;
    nlohmann::json payload = nullptr;
    std::string source;
    std::string event_id;
    std::uint64_t timestamp_ns = 0;
    std::map<std::string, std::string> metadata;

    Event();  // fills source + timestamp_ns defaults

    Event(std::string topic,
          nlohmann::json payload,
          std::string source = "",
          std::string event_id = "",
          std::uint64_t timestamp_ns = 0,
          std::map<std::string, std::string> metadata = {});

    // Serialized payload (what goes on the wire). Inspect/debug helper.
    std::string to_json() const;
};

struct TopicInfo {
    std::string topic;
    std::uint32_t subscriber_count = 0;
    std::uint64_t total_messages = 0;
    std::uint64_t last_message_ts = 0;
};

struct EventStats {
    std::string topic;
    std::uint64_t published_count = 0;
    std::uint64_t delivered_count = 0;
    std::uint64_t dropped_count = 0;
    float avg_latency_us = 0.0f;
};

struct SystemStats {
    std::vector<EventStats> topic_stats;
    std::uint32_t total_subscribers = 0;
    std::uint32_t total_topics = 0;
    std::uint64_t uptime_ms = 0;
};

// Pull-based equivalent of Python's `for ev in events.subscribe(topic):`.
// Must not outlive the EventClient that produced it (borrows the channel).
class EventStream {
public:
    ~EventStream();
    EventStream(EventStream&&) noexcept;
    EventStream& operator=(EventStream&&) noexcept;
    EventStream(const EventStream&) = delete;
    EventStream& operator=(const EventStream&) = delete;

    std::optional<Event> next();

private:
    EventStream();
    struct Impl;
    std::unique_ptr<Impl> impl_;
    friend class EventClient;
};

class EventClient {
public:
    explicit EventClient(std::string endpoint = "");
    ~EventClient();
    EventClient(const EventClient&) = delete;
    EventClient& operator=(const EventClient&) = delete;
    EventClient(EventClient&&) noexcept;
    EventClient& operator=(EventClient&&) noexcept;

    void connect();
    void close();
    bool connected() const noexcept;

    // Publish one event; returns the assigned event_id.
    std::string publish(const std::string& topic,
                        const nlohmann::json& payload,
                        bool persistent = false,
                        std::optional<std::uint32_t> ttl_ms = std::nullopt,
                        const std::map<std::string, std::string>& metadata = {});

    // Stream-publish a batch. Each item is {topic, payload}.
    void publish_batch(
        const std::vector<std::pair<std::string, nlohmann::json>>& events,
        bool persistent = false);

    // Open a subscription stream (wildcards supported, e.g. "model/*/detections").
    EventStream subscribe(const std::string& topic,
                          const std::map<std::string, std::string>& filters = {},
                          std::uint32_t queue_size = 100,
                          bool drop_old = true);

    // Spawn a detached background thread invoking `callback` per event until
    // close() or the client is destroyed. (Mirrors Python's on_event daemon
    // thread.) The callback may be invoked after the client is destroyed if
    // events are still in flight — keep it self-contained.
    void on_event(const std::string& topic,
                  std::function<void(const Event&)> callback,
                  const std::map<std::string, std::string>& filters = {});

    void unsubscribe(const std::string& topic);

    std::vector<TopicInfo> list_topics();
    std::optional<TopicInfo> get_topic_info(const std::string& topic);
    SystemStats get_stats();
    EventStats get_topic_stats(const std::string& topic);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace neoruntime_ipc_sdk
