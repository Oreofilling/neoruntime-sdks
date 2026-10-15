// accel.cpp — port of accel.py (0.8.1): AccelRouter (route/run/health
// semantics and reason strings byte-identical), the process-resident DSP
// dispatch machinery behind shared_dsp_call (python's _dsp_shared /
// _dsp_active / _dsp_retired), the router legs, and the pre-registered
// default router.
#include "neoruntime_ipc_sdk/accel.hpp"

#include <algorithm>
#include <any>
#include <chrono>
#include <cstddef>
#include <memory>
#include <mutex>
#include <set>
#include <stdexcept>
#include <string>
#include <typeinfo>
#include <utility>
#include <vector>

#include "neoruntime_ipc_sdk/color.hpp"
#include "neoruntime_ipc_sdk/draw.hpp"
#include "neoruntime_ipc_sdk/dsp.hpp"
#include "neoruntime_ipc_sdk/frame.hpp"
#include "neoruntime_ipc_sdk/postprocess.hpp"
#include "detail/dsp_dispatch.hpp"
#include "detail/dsp_format.hpp"
#include "detail/jpeg_encode.hpp"

namespace neoruntime_ipc_sdk {

namespace {

double now_unix() {
    using namespace std::chrono;
    return duration_cast<duration<double>>(system_clock::now().time_since_epoch()).count();
}

// ---- default dispatch ---------------------------------------------------------

// The production DspDispatch: a process-resident DspClient (both
// transports lazy). The client's own CPU fallback is always disabled on
// router legs — the router runs its own software leg so the degradation
// is recorded exactly once.
class DefaultDspDispatch final : public detail::DspDispatch {
public:
    bool has_live_pools() override { return client_.has_live_pools(); }
    void close() override { client_.close(); }

    std::any call(const std::string& method, std::vector<std::any> args) override {
        if (method == "resize_hw") {
            const RouterDspOpts opts = std::any_cast<RouterDspOpts>(args[3]);
            auto result = client_.resize_hw(source(args[0]), std::any_cast<int>(args[1]),
                                            std::any_cast<int>(args[2]), opts.fmt,
                                            params_from(opts));
            return job_result(std::move(result));
        }
        if (method == "convert_hw") {
            const RouterDspOpts opts = std::any_cast<RouterDspOpts>(args[2]);
            auto result = client_.convert_hw(source(args[0]),
                                             std::any_cast<std::string>(args[1]), opts.fmt,
                                             params_from(opts));
            return job_result(std::move(result));
        }
        if (method == "encode_jpeg_hw") {
            DspJobSource src = source(args[0]);
            auto bytes = client_.encode_jpeg_hw(&src, std::any_cast<int>(args[1]),
                                                std::any_cast<std::string>(args[2]),
                                                /*timeout_s=*/5.0, /*src_pool=*/nullptr,
                                                /*cpu_fallback=*/false);
            return std::any(std::move(bytes));
        }
        if (method == "blend_hw") {
            DspBlendParams params;
            params.cpu_fallback = false;
            DspJobSource base = source(args[0]);
            auto result = client_.blend_hw(
                base, std::any_cast<const std::vector<DspOverlay>&>(args[1]),
                std::any_cast<std::string>(args[2]), params);
            if (auto* mat = std::get_if<cv::Mat>(&result)) return std::any(std::move(*mat));
            throw std::runtime_error(
                "blend_hw returned no Mat (dst_pool path is not a router leg)");
        }
        if (method == "ensure_sock") {
            client_.connect();  // python _ensure_sock: fail fast when the daemon is down
            return {};
        }
        throw std::invalid_argument("unknown DSP dispatch method: " + method);
    }

private:
    static DspJobParams params_from(const RouterDspOpts& opts) {
        DspJobParams params;
        params.scaling = opts.scaling;
        params.cpu_fallback = opts.cpu_fallback;
        params.out_ref = opts.out_ref;
        return params;
    }

    static DspJobSource source(const std::any& a) {
        if (const auto* mat = std::any_cast<cv::Mat>(&a)) return *mat;
        if (const auto* frame = std::any_cast<Frame>(&a)) return *frame;
        if (const auto* handle = std::any_cast<std::shared_ptr<FrameHandle>>(&a)) return *handle;
        throw std::invalid_argument("dsp source must be cv::Mat, Frame or FrameHandle");
    }

    // wait=true jobs land as a Mat, or — with out_ref — a DspBufferRef.
    // Both riders below are move-only, so the ref crosses the type-erased
    // boundary as a shared_ptr (RAII intact: the last owner releases the
    // daemon buffer). PendingDspJob (wait=false) is not a router-leg
    // shape and never crosses.
    static std::any job_result(std::variant<cv::Mat, PendingDspJob, DspBufferRef> result) {
        if (auto* mat = std::get_if<cv::Mat>(&result)) return std::any(std::move(*mat));
        if (auto* ref = std::get_if<DspBufferRef>(&result))
            return std::any(std::make_shared<DspBufferRef>(std::move(*ref)));
        throw std::runtime_error(
            "DSP job returned without waiting (PendingDspJob) — router legs always wait");
    }

    DspClient client_;
};

// python accel.py module globals _dsp_shared/_dsp_active/_dsp_retired.
// Leak-on-exit (never destructed): the resident DspClient must not see
// static destruction order — the process exiting closes its sockets and
// the daemon reclaims the buffers.
struct ResidentState {
    std::mutex mutex;
    detail::DspDispatch* shared = nullptr;      // python _dsp_shared
    std::unique_ptr<DefaultDspDispatch> owned;  // default dispatch storage
    long active = 0;                            // python _dsp_active
    std::vector<detail::DspDispatch*> retired;  // python _dsp_retired
};

ResidentState& resident_state() {
    static ResidentState* const state = new ResidentState();
    return *state;
}

// python shared_dsp_call's finally: decrement the in-flight counter and
// drain whatever the failure path retired.
void shared_call_leave() {
    ResidentState& s = resident_state();
    {
        std::lock_guard<std::mutex> lock(s.mutex);
        --s.active;
    }
    detail::drain_retired();
}

// python _dsp_call: one DspClient method with its built-in CPU fallback
// disabled, errors wrapped as HardwareUnavailable so the router records
// the degradation and runs its own software leg exactly once.
std::any dsp_call(const std::string& method, std::vector<std::any> args) {
    try {
        return shared_dsp_call(method, std::move(args));
    } catch (const std::exception& exc) {
        throw HardwareUnavailable("DSP " + method + " failed: " + exc.what());
    } catch (...) {
        throw HardwareUnavailable("DSP " + method + " failed: unknown error");
    }
}

// ---- router legs ---------------------------------------------------------------

using LegFn = AccelRouter::Fn;

// python _frame_like: keep-fd sources import zero-copy instead of copying
// pixels — legs pass them straight through to the dispatch.
bool is_frame_like(const std::any& a) {
    return std::any_cast<Frame>(&a) != nullptr ||
           std::any_cast<std::shared_ptr<FrameHandle>>(&a) != nullptr;
}

LegFn nms_sw_leg() {
    // args: {boxes, scores[, iou_threshold][, class_ids* or empty any]}
    return [](std::vector<std::any> args) -> std::any {
        const auto& boxes = std::any_cast<const std::vector<std::array<float, 4>>&>(args[0]);
        const auto& scores = std::any_cast<const std::vector<float>&>(args[1]);
        const double iou = args.size() > 2 ? std::any_cast<double>(args[2]) : 0.5;
        const std::vector<int>* class_ids = nullptr;
        if (args.size() > 3 && args[3].has_value())
            class_ids = std::any_cast<const std::vector<int>*>(args[3]);
        return std::any(nms(boxes, scores, iou, class_ids));
    };
}

LegFn nv12_resize_sw_leg() {
    // args: {nv12, src_size, dst_size}
    return [](std::vector<std::any> args) -> std::any {
        return std::any(nv12_resize(std::any_cast<const cv::Mat&>(args[0]),
                                    std::any_cast<std::pair<int, int>>(args[1]),
                                    std::any_cast<std::pair<int, int>>(args[2])));
    };
}

LegFn rgb_to_nv12_sw_leg() {
    return [](std::vector<std::any> args) -> std::any {
        return std::any(rgb_to_nv12_impl(std::any_cast<const cv::Mat&>(args[0])));
    };
}

LegFn nv12_to_rgb_sw_leg() {
    return [](std::vector<std::any> args) -> std::any {
        return std::any(nv12_to_rgb_impl(std::any_cast<const cv::Mat&>(args[0]),
                                         std::any_cast<int>(args[1]),
                                         std::any_cast<int>(args[2])));
    };
}

LegFn encode_jpeg_sw_leg() {
    // args: {Mat|Frame[, quality]} — python _encode_jpeg_sw
    return [](std::vector<std::any> args) -> std::any {
        const int quality = args.size() > 1 ? std::any_cast<int>(args[1]) : 85;
        if (auto* frame = std::any_cast<Frame>(&args[0]))
            return std::any(detail::encode_jpeg_rgb(frame->to_rgb(), quality));
        return std::any(detail::encode_jpeg_rgb(std::any_cast<const cv::Mat&>(args[0]), quality));
    };
}

LegFn resize_nv12_hw_leg() {
    // python _resize_nv12_hw(nv12, src_size, dst_size): keep-fd frames
    // ride the zero-copy import; arrays arrive as Mats already.
    return [](std::vector<std::any> args) -> std::any {
        const auto dst = std::any_cast<std::pair<int, int>>(args[2]);
        RouterDspOpts opts;  // fmt="nv12", cpu_fallback=False
        opts.fmt = "nv12";
        opts.cpu_fallback = false;
        return dsp_call("resize_hw", {std::move(args[0]), std::any(dst.first),
                                      std::any(dst.second), std::any(opts)});
    };
}

LegFn rgb_to_nv12_hw_leg() {
    // python _rgb_to_nv12_hw: validate the impl's contract (3D HWC, even
    // dims) before touching the daemon — a caller error must raise the
    // software leg's ValueError, not ride a doomed submit into the
    // degradation counters.
    return [](std::vector<std::any> args) -> std::any {
        if (!is_frame_like(args[0])) {
            const cv::Mat& rgb = std::any_cast<const cv::Mat&>(args[0]);
            if (rgb.dims != 2 || rgb.channels() != 3)
                throw std::invalid_argument("rgb source must be 3D (h, w, 3), got shape " +
                                            shape_text(rgb));
            check_nv12_even(rgb.rows, rgb.cols, rgb);
        }
        RouterDspOpts opts;  // fmt="rgb24", cpu_fallback=False
        opts.fmt = "rgb24";
        opts.cpu_fallback = false;
        return dsp_call("convert_hw",
                        {std::move(args[0]), std::any(std::string("nv12")), std::any(opts)});
    };
}

LegFn nv12_to_rgb_hw_leg() {
    // python _nv12_to_rgb_hw(nv12[, width, height]).
    return [](std::vector<std::any> args) -> std::any {
        const bool has_wh = args.size() >= 3;
        const int width = has_wh ? std::any_cast<int>(args[1]) : 0;
        const int height = has_wh ? std::any_cast<int>(args[2]) : 0;
        RouterDspOpts opts;  // fmt="nv12", cpu_fallback=False
        opts.fmt = "nv12";
        opts.cpu_fallback = false;
        if (const auto* frame = std::any_cast<Frame>(&args[0])) {
            // geometry comes with the handle — arrays infer it from the shape
            if (has_wh && (width != frame->width || height != frame->height))
                throw std::invalid_argument(
                    "frame is " + std::to_string(frame->width) + "x" +
                    std::to_string(frame->height) + ", got width/height " +
                    std::to_string(width) + "/" + std::to_string(height));
            return dsp_call("convert_hw",
                            {std::any(*frame), std::any(std::string("rgb24")), std::any(opts)});
        }
        const cv::Mat& nv12 = std::any_cast<const cv::Mat&>(args[0]);
        if (nv12.dims != 2 || nv12.channels() != 1)
            throw std::invalid_argument("nv12 source must be 2D, got shape " + shape_text(nv12));
        const int src_w = nv12.cols;
        const int src_h = nv12.rows * 2 / 3;
        if (has_wh && (width != src_w || height != src_h)) {
            // a well-formed buffer with a mismatched declaration raises the
            // leg's message, a malformed buffer the software leg's
            // (python checks rows == src_h * 3 // 2 first)
            if (nv12.rows == src_h * 3 / 2)
                throw std::invalid_argument("nv12 is " + std::to_string(src_w) + "x" +
                                            std::to_string(src_h) + ", got width/height " +
                                            std::to_string(width) + "/" + std::to_string(height));
            check_nv12_dims(width, height, nv12);
        }
        return dsp_call("convert_hw",
                        {std::any(nv12), std::any(std::string("rgb24")), std::any(opts)});
    };
}

LegFn encode_jpeg_hw_leg() {
    // python _encode_jpeg_hw(rgb[, quality]): the handle carries
    // geometry+format (fmt "" lets the client infer); arrays are rgb24.
    return [](std::vector<std::any> args) -> std::any {
        const int quality = args.size() > 1 ? std::any_cast<int>(args[1]) : 85;
        std::string fmt = "rgb24";
        if (is_frame_like(args[0])) fmt = "";  // python fmt=None
        return dsp_call("encode_jpeg_hw", {std::move(args[0]), std::any(quality), std::any(fmt)});
    };
}

// python _draw_detections_hw: render the annotation once as a minimal
// RGBA overlay and composite it on the DSP in a single blend job. NV12
// arrays only — keep-fd frames raise (the zero-copy frame blend chain
// has wedged the DSP device-wide in the field), RGB arrays stay on the
// software raster.
LegFn draw_detections_hw_leg() {
    return [](std::vector<std::any> args) -> std::any {
        if (is_frame_like(args[0]))
            throw HardwareUnavailable(
                "draw_detections hardware leg takes NV12 arrays, not keep-fd frames: the "
                "zero-copy frame blend chain has wedged the DSP device-wide in the field "
                "(state-dependent; DspClient.blend_hw refuses it). Call frame.to_array() "
                "and route the array.");
        const cv::Mat& nv12 = std::any_cast<const cv::Mat&>(args[0]);
        if (nv12.dims != 2 || nv12.channels() != 1)
            throw HardwareUnavailable(
                "draw_detections hardware leg is nv12-only (DSP blends onto NV12), got shape " +
                shape_text(nv12));
        const int width = nv12.cols;
        const int height = nv12.rows * 2 / 3;
        const Detections& dets = std::any_cast<const Detections&>(args[1]);
        const std::optional<Rgb> color =
            args.size() > 2 ? std::any_cast<std::optional<Rgb>>(args[2]) : std::nullopt;
        const Annotations ann = extract_annotations(dets, color);
        if (ann.objects.empty()) return std::any(nv12.clone());  // nothing to draw
        const std::vector<std::optional<Rgb>> colors(ann.colors.begin(), ann.colors.end());
        const RgbaCanvas canvas =
            render_overlay_rgba(width, height, ann.boxes, ann.labels, ann.scores, colors);
        return dsp_call(
            "blend_hw", {std::any(nv12),
                         std::any(std::vector<DspOverlay>{{canvas.rgba, canvas.x, canvas.y}}),
                         std::any(std::string("nv12"))});
    };
}

// python _draw_detections_sw: the draw.cpp raster on RGB arrays; on NV12
// the cv mirror of the hardware path (render_overlay_rgba +
// straight-alpha composite), so a degradation never changes the output
// format.
LegFn draw_detections_sw_leg() {
    return [](std::vector<std::any> args) -> std::any {
        const cv::Mat& image = std::any_cast<const cv::Mat&>(args[0]);
        const Detections& dets = std::any_cast<const Detections&>(args[1]);
        const std::optional<Rgb> color =
            args.size() > 2 ? std::any_cast<std::optional<Rgb>>(args[2]) : std::nullopt;
        if (image.dims != 2 || image.channels() != 1)
            return std::any(draw_detections_impl(image, dets, color));
        const int width = image.cols;
        const int height = image.rows * 2 / 3;
        const Annotations ann = extract_annotations(dets, color);
        if (ann.objects.empty()) return std::any(image.clone());
        const std::vector<std::optional<Rgb>> colors(ann.colors.begin(), ann.colors.end());
        const RgbaCanvas canvas =
            render_overlay_rgba(width, height, ann.boxes, ann.labels, ann.scores, colors);
        return std::any(detail::dsp_format::cpu_blend(
            image, "nv12", {{canvas.rgba, canvas.x, canvas.y}}));
    };
}

bool probe_cv2() {
    return true;  // OpenCV is a hard dependency of the C++ SDK
}

bool probe_dsp() {
    try {
        // a real reachability check, not just "the client constructs":
        // opening the transports fails fast when the daemon is down
        detail::lazy_dsp_dispatch()->call("ensure_sock", {});
        return true;
    } catch (...) {
        return false;
    }
}

}  // namespace

// ---- resident machinery (detail) -------------------------------------------------

namespace detail {

DspDispatch* lazy_dsp_dispatch() {
    ResidentState& s = resident_state();
    std::lock_guard<std::mutex> lock(s.mutex);
    if (s.shared == nullptr) {
        s.owned = std::make_unique<DefaultDspDispatch>();
        s.shared = s.owned.get();
    }
    return s.shared;
}

void dsp_drop_shared(DspDispatch* dispatch) {
    ResidentState& s = resident_state();
    {
        std::lock_guard<std::mutex> lock(s.mutex);
        if (s.shared != dispatch) return;  // not ours to retire
        s.shared = nullptr;
        s.retired.push_back(dispatch);
    }
    drain_retired();
}

void drain_retired() {
    ResidentState& s = resident_state();
    std::vector<DspDispatch*> done;
    {
        std::lock_guard<std::mutex> lock(s.mutex);
        if (s.active != 0) return;  // a shared call is in flight somewhere
        for (auto it = s.retired.begin(); it != s.retired.end();) {
            // no has_live_pools answer counts as idle (transport-only fakes)
            if (!(*it)->has_live_pools()) {
                done.push_back(*it);
                it = s.retired.erase(it);
            } else {
                ++it;
            }
        }
    }
    for (DspDispatch* dispatch : done) {
        try {
            dispatch->close();  // outside the lock — must not block callers
        } catch (...) {  // cleanup must not mask result/error
        }
    }
}

DspDispatch* current_resident() {
    ResidentState& s = resident_state();
    std::lock_guard<std::mutex> lock(s.mutex);
    return s.shared;
}

void install_dsp_dispatch(DspDispatch* dispatch) {
    ResidentState& s = resident_state();
    std::unique_ptr<DefaultDspDispatch> old;
    {
        std::lock_guard<std::mutex> lock(s.mutex);
        if (s.shared == s.owned.get()) old = std::move(s.owned);  // close below, unlocked
        s.shared = dispatch;
        s.active = 0;
        s.retired.clear();
    }
    if (old) old->close();
}

}  // namespace detail

// ---- shared_dsp_call -------------------------------------------------------------

std::any shared_dsp_call(const std::string& method, std::vector<std::any> args) {
    detail::DspDispatch* client = detail::lazy_dsp_dispatch();
    {
        ResidentState& s = resident_state();
        std::lock_guard<std::mutex> lock(s.mutex);
        ++s.active;
    }
    try {
        std::any result = client->call(method, std::move(args));
        shared_call_leave();
        return result;
    } catch (...) {
        // retire the resident (deferred close via the drain below); the
        // original exception propagates so the caller keeps its own
        // fallback policy. Never retired on success.
        detail::dsp_drop_shared(client);
        shared_call_leave();
        throw;
    }
}

// ---- RoutePolicy -----------------------------------------------------------------

std::string route_policy_name(RoutePolicy policy) {
    switch (policy) {
        case RoutePolicy::PreferHardware: return "prefer_hardware";
        case RoutePolicy::SoftwareOnly: return "software_only";
        case RoutePolicy::HardwareOnly: return "hardware_only";
    }
    return {};
}

RoutePolicy route_policy_from_string(const std::string& name) {
    if (name == "prefer_hardware") return RoutePolicy::PreferHardware;
    if (name == "software_only") return RoutePolicy::SoftwareOnly;
    if (name == "hardware_only") return RoutePolicy::HardwareOnly;
    throw std::invalid_argument("'" + name + "' is not a valid RoutePolicy");
}

// ---- AccelRouter -----------------------------------------------------------------

AccelRouter::AccelRouter(RoutePolicy policy) : policy_(policy) {}

void AccelRouter::set_policy(RoutePolicy policy) {
    std::lock_guard<std::mutex> lock(mutex_);
    policy_ = policy;
}

RoutePolicy AccelRouter::policy() const noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    return policy_;
}

void AccelRouter::register_op(const std::string& op, Leg software, Leg hardware,
                              std::string note) {
    std::lock_guard<std::mutex> lock(mutex_);
    routes_[op] = OpEntry{std::move(software), std::move(hardware), std::move(note)};
    counters_[op] = Counters{};
}

void AccelRouter::use_hardware(const std::string& op, Leg hardware) {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto it = routes_.find(op);
    if (it == routes_.end())
        throw std::out_of_range("operation '" + op + "' is not registered");
    it->second.hardware = std::move(hardware);
}

void AccelRouter::add_probe(const std::string& name, std::function<bool()> probe) {
    std::lock_guard<std::mutex> lock(mutex_);
    probes_[name] = std::move(probe);
}

AccelRouter::OpEntry AccelRouter::snapshot(const std::string& op) const {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto it = routes_.find(op);
    if (it == routes_.end()) {
        // python KeyError text: "operation 'x' is not registered (known: ['a', 'b'])"
        std::string known = "[";
        bool first = true;
        for (const auto& name : routes_) {  // std::map: sorted by key
            if (!first) known += ", ";
            first = false;
            known += "'" + name.first + "'";
        }
        known += "]";
        throw std::out_of_range("operation '" + op + "' is not registered (known: " + known +
                                ")");
    }
    return it->second;  // copy — use_hardware may swap legs concurrently
}

RouteDecision AccelRouter::decide(const std::string& op, const OpEntry& entry) const {
    const RoutePolicy p = policy();
    if (p == RoutePolicy::SoftwareOnly) {
        if (!entry.software)
            return RouteDecision{op, "unavailable", "none",
                                 "software-only policy and no software provider"};
        return RouteDecision{op, "software", entry.software.name, "policy is software-only"};
    }
    if (entry.hardware)
        return RouteDecision{op, "hardware", entry.hardware.name, "hardware leg registered"};
    if (!entry.software) return RouteDecision{op, "unavailable", "none", "no provider registered"};
    return RouteDecision{op, "software", entry.software.name,
                         entry.note.empty() ? "no hardware provider registered" : entry.note};
}

RouteDecision AccelRouter::route(const std::string& op) const {
    const OpEntry entry = snapshot(op);
    return decide(op, entry);
}

std::any AccelRouter::run(const std::string& op, std::vector<std::any> args) {
    const RouteDecision decision = route(op);
    if (decision.backend == "unavailable")
        throw HardwareUnavailable(op + ": " + decision.reason);

    const OpEntry entry = snapshot(op);
    if (decision.backend == "software") {
        ++counters_for(op).software_calls;
        return entry.software.fn(std::move(args));
    }

    // the fallback leg may need the same args after the hardware attempt —
    // Mats/Frames are shallow handles, so the copy is cheap
    std::vector<std::any> args_copy = args;
    std::any result;
    try {
        result = entry.hardware.fn(std::move(args));
    } catch (const HardwareUnavailable& exc) {
        const std::string reason = exc.what();
        if (policy() == RoutePolicy::HardwareOnly) throw;  // the original error
        record_degradation(op, reason);
        if (!entry.software)
            throw HardwareUnavailable(op + ": hardware failed (" + reason +
                                      ") and no software fallback exists");
        ++counters_for(op).software_calls;
        return entry.software.fn(std::move(args_copy));
    }
    ++counters_for(op).hardware_calls;
    return result;
}

void AccelRouter::note_degradation(const std::string& op, const std::string& reason) {
    counters_for(op);  // setdefault: an unregistered op still counts
    record_degradation(op, reason);
}

void AccelRouter::record_degradation(const std::string& op, const std::string& reason) {
    const DegradationRecord record{op, reason, now_unix()};
    {
        std::lock_guard<std::mutex> lock(mutex_);
        degradations_.push_back(record);
        if (degradations_.size() > kMaxDegradations)
            degradations_.erase(degradations_.begin());
        ++counters_[op].fallbacks;  // already under the lock (counters_for would relock)
    }
    if (on_degradation) {
        try {
            on_degradation(record);
        } catch (...) {  // a health-sink failure must not break the app
        }
    }
}

AccelRouter::Counters& AccelRouter::counters_for(const std::string& op) {
    std::lock_guard<std::mutex> lock(mutex_);
    return counters_[op];
}

std::map<std::string, bool> AccelRouter::probe() const {
    std::map<std::string, std::function<bool()>> probes;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        probes = probes_;
    }
    std::map<std::string, bool> out;
    for (const auto& [name, probe] : probes) {
        try {
            out[name] = probe();
        } catch (...) {
            out[name] = false;
        }
    }
    return out;
}

RouterHealth AccelRouter::health() const {
    std::map<std::string, OpEntry> routes;
    std::map<std::string, Counters> counters;
    std::vector<DegradationRecord> degradations;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        routes = routes_;
        counters = counters_;
        degradations = degradations_;
    }
    RouterHealth out;
    out.policy = route_policy_name(policy());
    std::set<std::string> ops;  // sorted union, python sorted(set(...) | set(...))
    for (const auto& route : routes) ops.insert(route.first);
    for (const auto& counter : counters) ops.insert(counter.first);
    for (const std::string& op : ops) {
        OpHealth row;
        const auto cit = counters.find(op);
        if (cit != counters.end()) {
            row.hardware_calls = cit->second.hardware_calls;
            row.software_calls = cit->second.software_calls;
            row.fallbacks = cit->second.fallbacks;
        }
        const auto rit = routes.find(op);
        if (rit != routes.end()) {
            row.backend = decide(op, rit->second).backend;
        } else {
            // counter-only row: an external fast path (Frame::resize)
            // reported a fallback for an op this router never registered
            row.backend = "unavailable";
        }
        out.ops[op] = row;
    }
    out.recent_degradations = std::move(degradations);
    return out;
}

// ---- default router --------------------------------------------------------------

AccelRouter& get_default_router() {
    // heap-resident singleton: AccelRouter is non-movable (mutex), and the
    // deliberate leak-on-exit mirrors python's never-dying module-global
    // router (and dodges static destruction order vs the resident dispatch)
    static AccelRouter* router_ptr = new AccelRouter();
    static const bool configured = [&r = *router_ptr] {
        r.register_op(
            "resize_nv12", {"nv12_resize", nv12_resize_sw_leg()},
            {"_resize_nv12_hw", resize_nv12_hw_leg()},
            "DSP resize via DspClient.resize_hw (keep-fd frames import zero-copy; arrays "
            "copy in)");
        r.register_op(
            "rgb_to_nv12", {"_rgb_to_nv12_impl", rgb_to_nv12_sw_leg()},
            {"_rgb_to_nv12_hw", rgb_to_nv12_hw_leg()},
            "DSP convert via DspClient.convert_hw (keep-fd frames import zero-copy; arrays "
            "copy in)");
        r.register_op(
            "nv12_to_rgb", {"_nv12_to_rgb_impl", nv12_to_rgb_sw_leg()},
            {"_nv12_to_rgb_hw", nv12_to_rgb_hw_leg()},
            "DSP convert via DspClient.convert_hw (keep-fd frames import zero-copy; arrays "
            "copy in)");
        r.register_op(
            "encode_jpeg", {"_encode_jpeg_sw", encode_jpeg_sw_leg()},
            {"_encode_jpeg_hw", encode_jpeg_hw_leg()},
            "camera-daemon EncodeImage via DspClient.encode_jpeg_hw (libjpeg on the DSP core "
            "— no dedicated JPEG block on hailo15)");
        r.register_op(
            "nms", {"nms", nms_sw_leg()}, {},
            "HEF-integrated hardware NMS already suppressed pre-app; runtime "
            "detection_threshold via InferenceClient.update_postprocess_config (family "
            "functions only)");
        r.register_op(
            "draw_detections", {"_draw_detections_sw", draw_detections_sw_leg()},
            {"_draw_detections_hw", draw_detections_hw_leg()},
            "DSP blend via DspClient.blend_hw on a minimal RGBA canvas (nv12 — keep-fd "
            "frames ride the zero-copy import+copy+blend chain; RGB arrays stay on the "
            "software raster)");
        r.add_probe("cv2", probe_cv2);
        r.add_probe("dsp", probe_dsp);
        return true;
    }();
    (void)configured;
    return *router_ptr;
}

void set_route_policy(RoutePolicy policy) { get_default_router().set_policy(policy); }

}  // namespace neoruntime_ipc_sdk
