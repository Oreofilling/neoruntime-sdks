// injection.cpp — FramePublisher implementation (port of python
// injection.py, 0.8.1). Pure geometry/validation/slot helpers live in
// detail/injection_logic.hpp; this file owns the client orchestration:
// pool lifecycle, the write-lease state machine, the PushFrame request
// shape, and publish_rgb's hardware staging chain.
#include <chrono>
#include <cstdio>
#include <exception>
#include <set>
#include <stdexcept>
#include <thread>
#include <utility>
#include <variant>
#include <vector>

#include "neoruntime_ipc_sdk/injection.hpp"

#include "detail/injection_logic.hpp"

namespace neoruntime_ipc_sdk {

namespace {

namespace logic = detail::injection_logic;

// The variant pool's common surface (DspBufferPool and SharedImportPool
// share the operations FramePublisher needs but no base class).
std::uint64_t pool_buffer_id(
    const std::variant<std::unique_ptr<DspBufferPool>, std::unique_ptr<SharedImportPool>>& pool,
    int index) {
    if (const auto* p = std::get_if<std::unique_ptr<DspBufferPool>>(&pool))
        return (*p)->buffer_id(index);
    return std::get<std::unique_ptr<SharedImportPool>>(pool)->buffer_id(index);
}

void pool_write(
    const std::variant<std::unique_ptr<DspBufferPool>, std::unique_ptr<SharedImportPool>>& pool,
    int index, const cv::Mat& arr) {
    if (const auto* p = std::get_if<std::unique_ptr<DspBufferPool>>(&pool))
        (*p)->write(index, arr);
    else
        std::get<std::unique_ptr<SharedImportPool>>(pool)->write(index, arr);
}

// Luma stride (nv12 dma-buf pool) or byte stride (argb shared ring).
std::uint32_t pool_stride0(
    const std::variant<std::unique_ptr<DspBufferPool>, std::unique_ptr<SharedImportPool>>& pool) {
    if (const auto* p = std::get_if<std::unique_ptr<DspBufferPool>>(&pool))
        return (*p)->strides[0];
    return std::get<std::unique_ptr<SharedImportPool>>(pool)->strides()[0];
}

void set_from(const std::vector<std::uint64_t>& ids, std::set<std::uint64_t>& out) {
    out.clear();
    out.insert(ids.begin(), ids.end());
}

}  // namespace

// Default-config convenience overload (see header note): building Config{}
// here keeps the NSDMIs out of an in-class default argument (CWG 1360).
FramePublisher::FramePublisher(CameraClient& camera, DspClient& dsp, std::string stream_id,
                               int pool_depth)
    : FramePublisher(camera, dsp, std::move(stream_id), pool_depth, Config{}) {}

FramePublisher::FramePublisher(CameraClient& camera, DspClient& dsp, std::string stream_id,
                               int pool_depth, Config config)
    : camera_(camera),
      dsp_(dsp),
      stream_id_(std::move(stream_id)),
      session_id_(std::move(config.session_id)),
      mode_(std::move(config.mode)),
      fmt_(std::move(config.fmt)),
      lease_timeout_s_(config.lease_timeout_s) {
    logic::validate_pool_depth(pool_depth);
    logic::validate_mode(mode_);

    const std::vector<StreamStatus> streams = camera_.get_stream_status();
    const StreamStatus* st = logic::find_stream(streams, stream_id_);
    if (!st)
        throw std::runtime_error("no live stream named '" + stream_id_ +
                                 "'; known: " + logic::known_streams_text(streams));
    const logic::Geometry g =
        logic::resolve_geometry(*st, mode_, fmt_, config.inset, config.dest);
    width_ = g.pool_w;
    height_ = g.pool_h;
    dest_x_ = g.dest_x;
    dest_y_ = g.dest_y;
    pool_depth_ = pool_depth;

    // Allocating last: every validation above runs before the pools exist,
    // so a failed constructor leaks nothing by construction.
    if (fmt_ == "argb") {
        // The deployed HAL refuses ARGB32 pool allocation (wire OOM):
        // alpha overlays ride the shared memfd-import ring instead — the
        // daemon maps the same pages the bake reads.
        pool_ = dsp_.import_shared_buffers(width_, height_, fmt_, pool_depth);
    } else {
        pool_ = dsp_.alloc_buffers(width_, height_, fmt_, pool_depth);
    }
    pool_ids_.reserve(static_cast<std::size_t>(pool_depth));
    for (int i = 0; i < pool_depth; ++i) pool_ids_.push_back(pool_buffer_id(pool_, i));

    // Write-lease capability probe (Fix-1): best-effort — any failure
    // (daemon restarting, ancient build) degrades to the legacy
    // depth-paced rotation with one warning.
    try {
        lease_ = camera_.injection_status().reports_in_flight_buffers;
    } catch (...) {
        lease_ = false;
    }
    if (!lease_) {
        std::fprintf(stderr,
                     "neoruntime_ipc_sdk: FramePublisher(%s): daemon does not report the "
                     "write-lease set; falling back to blind slot rotation — publish at "
                     "stream rate with pool_depth >= 4 or a queued frame may bake with "
                     "newer pixels\n",
                     stream_id_.c_str());
    }
}

FramePublisher::~FramePublisher() {
    // Context-manager-exit parity: a publisher that published frames and
    // never sent an EOS would hold the session (and its queue) until the
    // connection drops — send one best-effort EOS before the pools go.
    if (!closed_ && published_any_ && !eos_sent_) {
        try {
            publish_eos();
        } catch (const std::exception& e) {
            std::fprintf(stderr,
                         "neoruntime_ipc_sdk: FramePublisher best-effort EOS failed; the "
                         "injection session stays open until the connection drops (%s)\n",
                         e.what());
        }
    }
    close();
}

// ---- publish -----------------------------------------------------------------

InjectionResult FramePublisher::publish(const cv::Mat& frame, std::uint64_t pts_ns) {
    // Python converts before the closed check — mirrored: a bad payload
    // reports its shape error even on a closed publisher.
    return publish_arr(logic::to_pool_array(frame, fmt_, width_, height_), pts_ns);
}

InjectionResult FramePublisher::publish(const void* data, std::size_t len,
                                        std::uint64_t pts_ns) {
    return publish_arr(logic::to_pool_array(data, len, fmt_, width_, height_), pts_ns);
}

InjectionResult FramePublisher::publish_arr(const cv::Mat& arr, std::uint64_t pts_ns) {
    if (closed_) throw std::runtime_error("FramePublisher is closed");
    const int slot = lease_ ? acquire_slot_ids(pool_ids_) : legacy_slot();
    pool_write(pool_, slot, arr);
    published_any_ = true;
    return push_slot_request(slot, pts_ns);
}

// The shared tail of publish (slot written client-side) and push_slot
// (slot written on the DSP): the buffer crosses as its DSP-registry id,
// never as bytes, and the response's lease snapshot replaces the local
// in-flight set.
InjectionResult FramePublisher::push_slot_request(int slot, std::uint64_t pts_ns) {
    PushFrameSpec spec;
    spec.buffer_id = pool_buffer_id(pool_, slot);
    spec.width = static_cast<std::uint32_t>(width_);
    spec.height = static_cast<std::uint32_t>(height_);
    spec.stride = pool_stride0(pool_);
    spec.mode = mode_;
    spec.pts_ns = pts_ns;
    spec.dest_x = static_cast<std::uint32_t>(dest_x_);
    spec.dest_y = static_cast<std::uint32_t>(dest_y_);
    spec.stream_id = stream_id_;
    spec.session_id = session_id_;
    InjectionResult res = camera_.push_frame(spec);
    absorb_lease(res);
    return res;
}

InjectionResult FramePublisher::publish_stream(const std::vector<cv::Mat>& frames,
                                               const std::vector<std::uint64_t>& pts_ns,
                                               bool end_with_eos,
                                               std::optional<double> timeout_s) {
    if (closed_) throw std::runtime_error("FramePublisher is closed");
    std::vector<PushFrameSpec> specs;
    specs.reserve(frames.size() + (end_with_eos ? 1 : 0));
    for (std::size_t i = 0; i < frames.size(); ++i) {
        const cv::Mat arr = logic::to_pool_array(frames[i], fmt_, width_, height_);
        int slot;
        if (lease_) {
            slot = acquire_slot_ids(pool_ids_);
            // No per-frame response exists on this transport, so the lease
            // is marked locally at request-build time (the request is on
            // its way to the daemon's queue) and a later status poll
            // clears it once the daemon reports the id released.
            in_flight_.insert(pool_ids_[slot]);
        } else {
            slot = legacy_slot();
        }
        pool_write(pool_, slot, arr);
        PushFrameSpec spec;
        spec.buffer_id = pool_buffer_id(pool_, slot);
        spec.width = static_cast<std::uint32_t>(width_);
        spec.height = static_cast<std::uint32_t>(height_);
        spec.stride = pool_stride0(pool_);
        spec.mode = mode_;
        spec.pts_ns = i < pts_ns.size() ? pts_ns[i] : 0;  // python repeat(0)
        spec.dest_x = static_cast<std::uint32_t>(dest_x_);
        spec.dest_y = static_cast<std::uint32_t>(dest_y_);
        spec.stream_id = stream_id_;
        spec.session_id = session_id_;
        specs.push_back(std::move(spec));
    }
    if (!frames.empty()) published_any_ = true;
    if (end_with_eos) {
        specs.push_back(eos_spec());
        eos_sent_ = true;
    }
    return camera_.push_frame_stream(specs, timeout_s);
}

// python's publish_rgb: rgb Mat -> hardware CONVERT -> one-deep NV12 stage.
InjectionResult FramePublisher::publish_rgb(const cv::Mat& rgb, std::uint64_t pts_ns) {
    return publish_rgb_arr(rgb, pts_ns);
}

InjectionResult FramePublisher::publish_rgb(const void* data, std::size_t len,
                                            std::uint64_t pts_ns) {
    const std::size_t expected =
        static_cast<std::size_t>(height_) * width_ * 3;  // tight rgb24 bytes
    if (len != expected)
        throw std::invalid_argument("tight rgb bytes must be " + std::to_string(expected) +
                                    " bytes (" + std::to_string(width_) + "x" +
                                    std::to_string(height_) + " rgb), got " +
                                    std::to_string(len));
    return publish_rgb_arr(cv::Mat(height_, width_, CV_8UC3, const_cast<void*>(data)),
                           pts_ns);
}

InjectionResult FramePublisher::publish_rgb_arr(const cv::Mat& rgb, std::uint64_t pts_ns) {
    if (mode_ != "replace")
        throw std::runtime_error("publish_rgb targets REPLACE publishers only");
    if (closed_) throw std::runtime_error("FramePublisher is closed");
    if (!rgb_pool_) {
        rgb_pool_ = dsp_.alloc_buffers(width_, height_, "rgb24", 1);
        nv12_stage_ = dsp_.alloc_buffers(width_, height_, "nv12", 1);
    } else if (lease_) {
        // The staging slot is reused every call; wait until the daemon has
        // released the staging NV12 (the pushed id; the rgb staging is
        // only ever read by the convert itself).
        acquire_slot_ids({nv12_stage_->buffer_id(0)});
    }
    // Default params: sync submit with read-back. convert_hw stages the
    // rgb Mat into rgb_pool_ itself and lands the result in nv12_stage_.
    auto out = dsp_.convert_hw(rgb, "nv12", /*fmt=*/"", DspJobParams{}, rgb_pool_.get(),
                               nv12_stage_.get());
    const cv::Mat* pixels = std::get_if<cv::Mat>(&out);
    if (!pixels) throw std::runtime_error("convert_hw returned an unexpected form");
    if (!dsp_.last_used_hw().value_or(false)) {
        // CPU fallback computed pixels without touching the pool.
        nv12_stage_->write(0, *pixels);
    }
    published_any_ = true;
    PushFrameSpec spec;
    spec.buffer_id = nv12_stage_->buffer_id(0);
    spec.width = static_cast<std::uint32_t>(width_);
    spec.height = static_cast<std::uint32_t>(height_);
    spec.stride = nv12_stage_->strides[0];
    spec.mode = "replace";
    spec.pts_ns = pts_ns;
    spec.stream_id = stream_id_;
    spec.session_id = session_id_;
    InjectionResult res = camera_.push_frame(spec);
    absorb_lease(res);
    return res;
}

// python's EOS request dict: buffer_id 0, geometry, stride, end_of_stream,
// session_id — stream_id and mode stay unset (empty / proto default).
PushFrameSpec FramePublisher::eos_spec() const {
    PushFrameSpec spec;
    spec.buffer_id = 0;
    spec.width = static_cast<std::uint32_t>(width_);
    spec.height = static_cast<std::uint32_t>(height_);
    spec.stride = pool_stride0(pool_);
    spec.end_of_stream = true;
    spec.session_id = session_id_;
    return spec;
}

void FramePublisher::publish_eos() {
    if (closed_) throw std::runtime_error("FramePublisher is closed");
    camera_.push_frame(eos_spec());
    eos_sent_ = true;
}

// ---- zero-copy chain ------------------------------------------------------------

DspBufferPool* FramePublisher::dsp_pool() {
    if (closed_) throw std::runtime_error("FramePublisher is closed");
    const auto* p = std::get_if<std::unique_ptr<DspBufferPool>>(&pool_);
    return p ? p->get() : nullptr;
}

SharedImportPool* FramePublisher::shared_pool() {
    if (closed_) throw std::runtime_error("FramePublisher is closed");
    const auto* p = std::get_if<std::unique_ptr<SharedImportPool>>(&pool_);
    return p ? p->get() : nullptr;
}

int FramePublisher::acquire_slot() {
    if (closed_) throw std::runtime_error("FramePublisher is closed");
    if (lease_) return acquire_slot_ids(pool_ids_);
    return legacy_slot();
}

InjectionResult FramePublisher::push_slot(int slot, std::uint64_t pts_ns) {
    if (closed_) throw std::runtime_error("FramePublisher is closed");
    if (slot < 0 || slot >= pool_depth_)
        throw std::invalid_argument("slot " + std::to_string(slot) +
                                    " outside the pool's " + std::to_string(pool_depth_) +
                                    " slots");
    published_any_ = true;
    return push_slot_request(slot, pts_ns);
}

// ---- write-lease slots (Fix-1) ----------------------------------------------------

// Return a pool slot index whose buffer the daemon has released; poll the
// bake frontier at kLeasePollMs until one appears or lease_timeout_s
// elapses, then raise rather than silently rewriting pixels the daemon may
// still bake. `ids` is the round-robin pool's ids, or publish_rgb's
// one-deep NV12 staging id.
int FramePublisher::acquire_slot_ids(const std::vector<std::uint64_t>& ids) {
    const int depth = static_cast<int>(ids.size());
    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::duration<double>(lease_timeout_s_);
    for (;;) {
        const int slot = logic::pick_free_slot(slot_cursor_, ids, in_flight_);
        if (slot >= 0) {
            slot_cursor_ = (slot + 1) % depth;
            return slot;
        }
        if (std::chrono::steady_clock::now() >= deadline)
            throw std::runtime_error(
                logic::exhaustion_message(stream_id_, lease_timeout_s_, in_flight_));
        refresh_in_flight();
        std::this_thread::sleep_for(std::chrono::milliseconds(logic::kLeasePollMs));
    }
}

int FramePublisher::legacy_slot() {
    const int slot = slot_cursor_;
    slot_cursor_ = (slot + 1) % pool_depth_;
    return slot;
}

// Re-read the daemon's write-lease snapshot (GetInjectionStatus). Used
// only where no fresher source exists: pacing publish_stream's writes
// (its single response arrives after the last frame) and the initial wait
// inside acquire_slot_ids. Unary publish() updates from each PushFrame
// response instead (absorb_lease).
void FramePublisher::refresh_in_flight() {
    const InjectionStatus st = camera_.injection_status();
    if (st.reports_in_flight_buffers) set_from(st.in_flight_buffer_ids, in_flight_);
}

void FramePublisher::absorb_lease(const InjectionResult& res) {
    if (lease_) set_from(res.in_flight_buffer_ids, in_flight_);
}

// ---- lifecycle -----------------------------------------------------------------

void FramePublisher::close() noexcept {
    if (closed_) return;
    closed_ = true;
    // reset() runs each pool's destructor, which releases it (and is a
    // no-op on a pool the DspClient already poisoned at its close).
    std::visit([](auto& p) { p.reset(); }, pool_);
    rgb_pool_.reset();
    nv12_stage_.reset();
}

}  // namespace neoruntime_ipc_sdk
