// injection.hpp's pure half: geometry resolution, payload validation and
// slot selection — everything in python injection.py that needs no client,
// no pool and no proto (port of the __init__/_to_pool_array/_acquire_slot
// logic, 0.8.1). Kept pb-free and header-inline so test_injection_slots can
// assert the validation ladder directly, the same way python's suite pokes
// FramePublisher without a daemon.
//
// Error-model parity: python ValueError -> std::invalid_argument, python
// RuntimeError -> std::runtime_error; message texts are reproduced
// character-for-character where the type system allows (python tuple/list
// reprs included, e.g. "known: ['main', 'sub']").
#pragma once

#include <opencv2/core.hpp>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "neoruntime_ipc_sdk/camera_types.hpp"

namespace neoruntime_ipc_sdk::detail::injection_logic {

// _acquire_slot's poll cadence (time.sleep(0.01)).
inline constexpr int kLeasePollMs = 10;

// python _MODES repr: f"mode must be one of {_MODES}, got {mode!r}".
inline constexpr const char* kModesText = "('replace', 'overlay')";

inline void validate_pool_depth(int pool_depth) {
    if (pool_depth < 1) throw std::invalid_argument("pool_depth must be >= 1");
}

inline void validate_mode(const std::string& mode) {
    if (mode != "replace" && mode != "overlay")
        throw std::invalid_argument(std::string("mode must be one of ") + kModesText +
                                    ", got '" + mode + "'");
}

// "['main', 'sub']" — python's list-of-str repr in the no-live-stream error.
inline std::string known_streams_text(const std::vector<StreamStatus>& streams) {
    std::string out = "[";
    for (std::size_t i = 0; i < streams.size(); ++i) {
        if (i) out += ", ";
        out += "'" + streams[i].stream_id + "'";
    }
    return out + "]";
}

inline const StreamStatus* find_stream(const std::vector<StreamStatus>& streams,
                                       const std::string& stream_id) {
    for (const auto& s : streams)
        if (s.stream_id == stream_id) return &s;
    return nullptr;
}

// Resolved pool geometry: what the ctor allocates and stamps on every push.
struct Geometry {
    int pool_w = 0;
    int pool_h = 0;
    int dest_x = 0;
    int dest_y = 0;
};

// The FramePublisher ctor's validation ladder, in python's exact order:
// encoder presence -> stream geometry -> per-mode fmt/geometry. REPLACE
// pins the pool to the encode resolution (NV12 only, dest forced to 0,0);
// OVERLAY pins it to the inset (defaulting to an even quarter of the
// stream) placed at an even dest that stays inside the frame.
inline Geometry resolve_geometry(const StreamStatus& st, const std::string& mode,
                                 const std::string& fmt,
                                 const std::optional<std::pair<int, int>>& inset,
                                 std::pair<int, int> dest) {
    if (!st.has_encoder)
        throw std::runtime_error("stream '" + st.stream_id + "' has no encoder");
    if (st.width <= 0 || st.height <= 0 || (st.width & 1) || (st.height & 1))
        throw std::invalid_argument("stream '" + st.stream_id + "' geometry " +
                                    std::to_string(st.width) + "x" + std::to_string(st.height) +
                                    " must be nonzero and even");

    Geometry g;
    if (mode == "replace") {
        if (fmt != "nv12")
            throw std::invalid_argument(
                "REPLACE consumes NV12 only (fmt=\"nv12\"); "
                "RGB sources go through publish_rgb");
        g.pool_w = static_cast<int>(st.width);
        g.pool_h = static_cast<int>(st.height);
        g.dest_x = g.dest_y = 0;
        return g;
    }
    // overlay
    if (fmt != "nv12" && fmt != "argb")
        throw std::invalid_argument(
            "overlay fmt must be 'nv12' (opaque paste) or 'argb' "
            "(alpha blend), got '" + fmt + "'");
    if (inset) {
        g.pool_w = inset->first;
        g.pool_h = inset->second;
    } else {
        // PIP default: one quarter of the stream, even-aligned.
        g.pool_w = (static_cast<int>(st.width) / 4) & ~1;
        g.pool_h = (static_cast<int>(st.height) / 4) & ~1;
    }
    if (g.pool_w <= 0 || g.pool_h <= 0 || (g.pool_w & 1) || (g.pool_h & 1))
        throw std::invalid_argument("inset " + std::to_string(g.pool_w) + "x" +
                                    std::to_string(g.pool_h) +
                                    " must be nonzero and even "
                                    "(the bake walks 2x2 chroma blocks)");
    g.dest_x = dest.first;
    g.dest_y = dest.second;
    if ((g.dest_x & 1) || (g.dest_y & 1))
        throw std::invalid_argument("dest (" + std::to_string(g.dest_x) + ", " +
                                    std::to_string(g.dest_y) +
                                    ") must be even — NV12 chroma is 2x2 subsampled");
    if (g.dest_x + g.pool_w > static_cast<int>(st.width) ||
        g.dest_y + g.pool_h > static_cast<int>(st.height))
        throw std::invalid_argument("inset " + std::to_string(g.pool_w) + "x" +
                                    std::to_string(g.pool_h) + " at (" +
                                    std::to_string(g.dest_x) + ", " + std::to_string(g.dest_y) +
                                    ") exceeds stream " + std::to_string(st.width) + "x" +
                                    std::to_string(st.height));
    return g;
}

// ---- payload validation (python _to_pool_array) --------------------------------

// numpy dtype names for the "dtype must be uint8" message.
inline const char* depth_name(int depth) {
    switch (depth) {
        case CV_8U: return "uint8";
        case CV_8S: return "int8";
        case CV_16U: return "uint16";
        case CV_16S: return "int16";
        case CV_32S: return "int32";
        case CV_32F: return "float32";
        case CV_64F: return "float64";
        default: return "unknown";
    }
}

// "(48, 64)" / "(32, 64, 4)" — python tuple repr of the expected ndarray
// shape (channels appear as the third axis exactly when the format has them).
inline std::string shape_text(int rows, int cols, int channels) {
    std::string out = "(" + std::to_string(rows) + ", " + std::to_string(cols);
    if (channels > 1) out += ", " + std::to_string(channels);
    return out + ")";
}

// Validate + pass through a caller Mat (no copy — pools write row-wise).
// nv12 expects (h*3/2, w) 8UC1; argb expects (h, w, 4) 8UC4.
inline cv::Mat to_pool_array(const cv::Mat& frame, const std::string& fmt, int w, int h) {
    const int rows = fmt == "nv12" ? h * 3 / 2 : h;
    const int cols = w;
    const int channels = fmt == "nv12" ? 1 : 4;
    if (frame.depth() != CV_8U)
        throw std::invalid_argument(fmt + " ndarray dtype must be uint8, got " +
                                    depth_name(frame.depth()));
    if (frame.rows != rows || frame.cols != cols || frame.channels() != channels)
        throw std::invalid_argument(fmt + " ndarray shape must be " +
                                    shape_text(rows, cols, channels) + ", got " +
                                    shape_text(frame.rows, frame.cols, frame.channels()));
    return frame;
}

// Tight-bytes overload: wraps the buffer in a non-owning Mat (the pool write
// copies synchronously, exactly like python's np.frombuffer(bytes) view).
inline cv::Mat to_pool_array(const void* data, std::size_t len, const std::string& fmt, int w,
                             int h) {
    const int rows = fmt == "nv12" ? h * 3 / 2 : h;
    const int channels = fmt == "nv12" ? 1 : 4;
    const std::size_t expected = static_cast<std::size_t>(rows) * w * channels;
    if (len != expected)
        throw std::invalid_argument("tight " + fmt + " bytes must be " +
                                    std::to_string(expected) + " bytes (" + std::to_string(w) +
                                    "x" + std::to_string(h) + " " + fmt + "), got " +
                                    std::to_string(len));
    return cv::Mat(rows, w, CV_MAKETYPE(CV_8U, channels), const_cast<void*>(data));
}

// ---- slot selection (python _acquire_slot) -------------------------------------

// One rotation pass: the first slot at or after `cursor` whose registry id
// is absent from the in-flight set, or -1 when every slot is busy. The
// caller advances its cursor to (slot + 1) % depth on success.
inline int pick_free_slot(int cursor, const std::vector<std::uint64_t>& ids,
                          const std::set<std::uint64_t>& in_flight) {
    const int depth = static_cast<int>(ids.size());
    for (int off = 0; off < depth; ++off) {
        const int slot = (cursor + off) % depth;
        if (in_flight.find(ids[slot]) == in_flight.end()) return slot;
    }
    return -1;
}

// python str(float) for the exhaustion message ("%g" keeps the substring
// the tests assert — "no free pool slot" — intact).
inline std::string format_seconds(double s) {
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%g", s);
    return buf;
}

inline std::string exhaustion_message(const std::string& stream_id, double lease_timeout_s,
                                      const std::set<std::uint64_t>& in_flight) {
    std::string held = "all slots";  // python: sorted(...) or 'all slots'
    if (!in_flight.empty()) {
        held = "[";
        bool first = true;
        for (std::uint64_t id : in_flight) {  // std::set iterates sorted
            if (!first) held += ", ";
            first = false;
            held += std::to_string(id);
        }
        held += "]";
    }
    return "FramePublisher('" + stream_id + "'): no free pool slot after " +
           format_seconds(lease_timeout_s) + "s — the daemon still holds " + held +
           " in flight; is the encoder feed consuming frames?";
}

}  // namespace neoruntime_ipc_sdk::detail::injection_logic
