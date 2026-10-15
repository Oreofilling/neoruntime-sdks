// injection.hpp — app frame injection front end (PushFrame P0-P2). Port of
// python injection.py (0.8.1); the pure validation/slot logic lives in
// src/detail/injection_logic.hpp.
//
// FramePublisher composes the two halves of the output write path the
// daemon exposes: a daemon-side DSP buffer pool (geometry-pinned dma-bufs,
// NV12 or ARGB32) and the CameraControl PushFrame RPCs (metadata only —
// the buffer crosses as a DSP-registry id, never as a raw fd number).
//
// Slots and the write lease (Fix-1): frames round-robin across the pool;
// on daemons that report the in-flight buffer set, a slot is rewritten
// only after the daemon acknowledges it is no longer reading it — every
// PushFrame response carries the ids still queued or mid-bake, and with
// all slots busy publish() waits for the bake frontier (up to
// Config::lease_timeout_s) and throws instead of silently tearing pixels.
// On older daemons the publisher falls back to the legacy blind rotation
// (one stderr warning); give such deployments pool_depth >= 4 and publish
// at stream rate. The daemon-side queue is cap-3 drop-oldest, so
// publishing never backpressures the encoder; overflow shows up in
// CameraClient::injection_status() as frames_dropped.
//
// C++ shape (deliberate deviations from python, all mechanical):
//   * publish()/publish_rgb() take a cv::Mat or tight bytes
//     (const void*, len) instead of ndarray-or-bytes duck typing.
//   * publish_stream() is vector-based: every frame is written to its slot
//     before the single PushFrameStream RPC starts (python's generator
//     interleaves the writes with gRPC pulls). Lease acquire still paces
//     the writes; a shorter pts_ns vector pads with 0 like python's
//     repeat(0) fallback.
//   * The context manager's best-effort EOS (__exit__) lives in the
//     destructor: if frames were published and no EOS went out, ~Frame
//     Publisher tries one publish_eos() (failure = stderr warning) before
//     close(). Bare close() never sends an EOS, same as python.
//   * python's `pool` property becomes dsp_pool()/shared_pool() split by
//     format (the two pool types have no common base).
//
// Usage:
//   FramePublisher pub(cam, dsp, "sub", 4, {.session_id = "app-1"});
//   pub.publish(nv12_mat);          // (h*3/2, w) 8UC1, or tight NV12 bytes
//   pub.publish_eos();              // flush; ISP path restores at next IDR
//   pub.close();
//
//   FramePublisher pip(cam, dsp, "main", 4,
//                      {.mode = "overlay", .fmt = "argb",
//                       .dest = pip_dest(1920, 1080, 480, 270, "bottom-right", 32)});
//   pip.publish(argb_mat);          // (h, w, 4) 8UC4, wire order [A, R, G, B]
#pragma once

#include <opencv2/core.hpp>

#include <cstdint>
#include <memory>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "camera.hpp"
#include "camera_types.hpp"
#include "dsp.hpp"

namespace neoruntime_ipc_sdk {

class FramePublisher {
public:
    // Python keyword tail of FramePublisher(...). inset: nullopt = the PIP
    // default (an even quarter of the stream); dest: OVERLAY paste origin.
    struct Config {
        std::string mode = "replace";  // "replace" | "overlay"
        std::string fmt = "nv12";      // "nv12" (replace/overlay) | "argb" (overlay)
        std::optional<std::pair<int, int>> inset;
        std::pair<int, int> dest = {0, 0};
        std::string session_id;      // lifecycle tag on every request (P2-13)
        double lease_timeout_s = 5.0;
    };

    // Resolves the target stream's geometry from GetStreamStatus and
    // rejects wrong-shaped input client-side, so the daemon never sees a
    // doomed request. Throws std::invalid_argument for bad arguments
    // (python ValueError), std::runtime_error for stream-state problems
    // (python RuntimeError); a failed construction allocates nothing.
    // Two overloads rather than one defaulted `Config config = ...`: a
    // default argument value-initializing the nested aggregate needs its
    // NSDMIs complete before FramePublisher ends (CWG 1360), so the 4-arg
    // form delegates through a Config{} built in the .cpp instead.
    FramePublisher(CameraClient& camera, DspClient& dsp, std::string stream_id = "sub",
                   int pool_depth = 4);
    FramePublisher(CameraClient& camera, DspClient& dsp, std::string stream_id,
                   int pool_depth, Config config);
    ~FramePublisher();  // best-effort EOS (see class comment), then close()

    FramePublisher(const FramePublisher&) = delete;
    FramePublisher& operator=(const FramePublisher&) = delete;

    // -- geometry ---------------------------------------------------------
    int width() const noexcept { return width_; }    // stream w (REPLACE) / inset w
    int height() const noexcept { return height_; }  // stream h (REPLACE) / inset h
    const std::string& stream_id() const noexcept { return stream_id_; }
    const std::string& session_id() const noexcept { return session_id_; }
    const std::string& mode() const noexcept { return mode_; }
    std::pair<int, int> dest() const noexcept { return {dest_x_, dest_y_}; }
    // True when the daemon reports the write-lease set (slot-aware
    // publishing); false on legacy daemons (blind rotation).
    bool lease_mode() const noexcept { return lease_; }

    // -- publish ------------------------------------------------------------
    // Write one frame to the next pool slot and push its id. NV12 pools
    // take an (height*3/2, width) 8UC1 Mat (or tight NV12 bytes); ARGB32
    // pools take an (height, width, 4) 8UC4 Mat (wire order [A, R, G, B])
    // or tight bytes of width*height*4. pts_ns paces the bake (device
    // CLOCK_MONOTONIC): 0 = due immediately, a future value holds the
    // frame until the encoder frontier passes it. Returns the daemon's
    // InjectionResult; throws on RPC failure (the slot was still written —
    // the daemon-side drop-oldest queue absorbs it, see frames_dropped).
    InjectionResult publish(const cv::Mat& frame, std::uint64_t pts_ns = 0);
    InjectionResult publish(const void* data, std::size_t len, std::uint64_t pts_ns = 0);

    // Push a run of frames over one client-streaming RPC. The daemon
    // applies per-request validation; the first rejection ends the stream
    // and surfaces as RpcError with the accepted-frame count in the text.
    // end_with_eos=true appends a final EOS request (closing the session)
    // before the RPC returns; otherwise the stream ends in a clean
    // half-close that leaves the session open (close it later via
    // publish_eos()). A shorter pts_ns vector pads with 0.
    InjectionResult publish_stream(const std::vector<cv::Mat>& frames,
                                   const std::vector<std::uint64_t>& pts_ns = {},
                                   bool end_with_eos = false,
                                   std::optional<double> timeout_s = std::nullopt);

    // Convert an RGB frame to NV12 and REPLACE the stream with it: one
    // DMA-side rgb24 staging buffer, a hardware convert (DspClient::
    // convert_hw) into a one-deep NV12 staging pool, then a push of that
    // pool's buffer id. `rgb` is an (height, width, 3) 8UC3 Mat in RGB
    // order (swap BGR beforehand) or tight bytes. If the DSP refuses or is
    // unavailable the convert falls back to CPU and the result is copied
    // into the staging pool, so the push is always well-defined
    // (DspClient::last_used_hw records the path taken). REPLACE publishers
    // only. The staging slot is reused every call: lease mode blocks until
    // the daemon has released the staging NV12, legacy mode relies on the
    // caller pacing at stream rate.
    InjectionResult publish_rgb(const cv::Mat& rgb, std::uint64_t pts_ns = 0);
    InjectionResult publish_rgb(const void* data, std::size_t len, std::uint64_t pts_ns = 0);

    // Flush the injection session (buffer_id 0 + end_of_stream). The
    // daemon drops the session and its queue; the pure ISP path restores
    // at the next IDR.
    void publish_eos();

    // -- zero-copy chain (P2-5) ---------------------------------------------
    // The publisher's daemon-side pool — the slots PushFrame accepts.
    // dsp_pool() serves nv12 publishers, shared_pool() argb ones (the
    // deployed HAL refuses ARGB32 pool allocation, so alpha overlays ride
    // a shared memfd-import ring); the other accessor returns nullptr.
    // A DSP job can write this pool directly (blend_hw(..., dst_pool =
    // pub.dsp_pool(), dst_slot = slot)) and the frame is publishable
    // without a single client-side pixel. The pool dies with the
    // publisher — close() releases it — so hold the publisher open for as
    // long as any writer targets it. Throws when closed.
    DspBufferPool* dsp_pool();
    SharedImportPool* shared_pool();

    // Reserve one pool slot for a device-side write: on lease daemons the
    // returned slot's buffer id is absent from the queued+mid-bake set;
    // legacy daemons get the blind rotation (the caller owns the pacing).
    // The reservation is client-side bookkeeping only — push the slot
    // promptly via push_slot(), or the lease set may hand it out again.
    int acquire_slot();

    // Push one already-written pool slot by id (no client write): the
    // slot's pixels were produced on the DSP straight into the pool. The
    // request is identical to publish()'s; only the write is skipped, and
    // the lease snapshot updates the same way.
    InjectionResult push_slot(int slot, std::uint64_t pts_ns = 0);

    // Release the buffer pools (idempotent; sends no EOS itself).
    void close() noexcept;

private:
    InjectionResult publish_arr(const cv::Mat& arr, std::uint64_t pts_ns);
    InjectionResult publish_rgb_arr(const cv::Mat& rgb, std::uint64_t pts_ns);
    InjectionResult push_slot_request(int slot, std::uint64_t pts_ns);
    PushFrameSpec eos_spec() const;
    int acquire_slot_ids(const std::vector<std::uint64_t>& ids);
    int legacy_slot();
    void refresh_in_flight();
    void absorb_lease(const InjectionResult& res);

    CameraClient& camera_;
    DspClient& dsp_;
    const std::string stream_id_;
    const std::string session_id_;
    const std::string mode_;
    const std::string fmt_;
    int width_ = 0;
    int height_ = 0;
    int dest_x_ = 0;
    int dest_y_ = 0;
    int pool_depth_ = 0;
    double lease_timeout_s_ = 5.0;
    // nv12 publishers: alloc'd dma-buf pool; argb: shared memfd-import ring.
    std::variant<std::unique_ptr<DspBufferPool>, std::unique_ptr<SharedImportPool>> pool_;
    std::vector<std::uint64_t> pool_ids_;  // pool_->buffer_id(i), resolved once
    int slot_cursor_ = 0;                  // round-robin cursor (python _slot)
    bool lease_ = false;
    std::set<std::uint64_t> in_flight_;
    // Lazily created by publish_rgb (RGB->NV12 staging, REPLACE only).
    std::unique_ptr<DspBufferPool> rgb_pool_;
    std::unique_ptr<DspBufferPool> nv12_stage_;
    bool closed_ = false;
    bool published_any_ = false;
    bool eos_sent_ = false;
};

}  // namespace neoruntime_ipc_sdk
